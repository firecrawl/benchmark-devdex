"""control.py — the rules of the eval, in ONE place, shared by every arm.

`runner_sdk.py` (Claude Opus, via the Claude Agent SDK) enforces the rules through hooks, but
calls the same underlying code for every arm rather than each arm reimplementing them:

  gate()     which tools may be called at all (deny-by-default) + the search budget
  prepare()  k / github-scope injection, only into keys the tool's schema declares
  record()   harvest the response into engine_pool, log it, and trim the agent's preview

Plus the three local tools (read_target, read_page, submit) that every arm shares.
"""
import base64, json, subprocess, threading, time, urllib.parse

# Imported BOTH ways: as devdex.harness.control from an installed package, and as a bare
# `import control` by runner_sdk, which puts this directory on sys.path. The package path is tried
# first so the module is importable from a wheel -- a bare `import provenance` alone made
# `import devdex.harness.control` fail outright once the project was pip-installed.
try:
    from devdex.harness import provenance as P
except ImportError:                      # this directory is on sys.path (script-mode entry points)
    import provenance as P


class Cfg:
    """Every knob that materially moves a score. Stamped onto each record by the harness."""
    __slots__ = ("k", "restrict", "max_search", "max_read", "agent_snippet", "page_full",
                 "manifest")

    def __init__(self, k=10, restrict=True, max_search=None, max_read=None,
                 agent_snippet=4000, page_full=None, manifest=None):
        self.k, self.restrict = k, restrict
        self.max_search, self.max_read = max_search, max_read
        self.agent_snippet, self.page_full = agent_snippet, page_full
        self.manifest = manifest or {}


class State:
    """Everything one question produced. `engine_pool` is the scoring set: refs the SEARCH
    BACKEND surfaced. Anything the agent reached another way lands in `agent_pool` and can
    never be credited — that is what stops the model's memory from scoring."""

    def __init__(self):
        self.pool = []; self.search_log = []; self.url_content = {}
        self.blocked = []          # read_target refusals (the agent probing from memory)
        self.denied = []           # tools refused as out-of-allowlist
        self.free_calls = []       # declared non-retrieval calls (e.g. Context7 id lookup)
        self.budget_denied = 0     # search calls refused because MAX_SEARCH was hit
        self.raw = []; self.transcript = []
        # Vendor fetch calls (product tier). SEPARATE from search_log on purpose: coverage,
        # reliability and search latency must stay properties of the SEARCH surface, or the
        # headline stops meaning what it says.
        self.fetch_log = []
        self.engine_pool = set(); self.agent_pool = set()
        self.submission = None; self.answer = None; self.sources = []
        self.answer_rejected = False   # submit_answer refused once for missing sources
        self.uncited = False           # committed an answer with no citation anyway
        self.searches = self.reads = self.ws = 0
        self.cur = "search_direct"
        self.clock = P.Clock()
        self._pending = {}
        self._lock = threading.Lock()
        # Defaults so the arms that never build a Controller (websearch, no-tool) can still use
        # the shared reader. A Controller overwrites both from its Cfg.
        self._max_search, self._max_read = None, None      # None = unlimited; see count()

    def count(self, kind="search"):
        """Count the call, and say whether it is allowed.

        A limit of None means UNLIMITED. The search budget was removed: MAX_TURNS
        and TIMEOUT already bound a run, so effort is MEASURED rather than capped and
        `searches/q` / `tool_calls/q` become reported metrics. Known cost, measured before the
        change: the one arm that ran uncapped (websearch) used 7.5 searches/question against the
        capped arms' 2.0-4.0, and unlimited retries let a weaker index brute-force toward the
        gold, narrowing the gaps between engines. Set MAX_SEARCH / MAX_READ to restore a cap."""
        with self._lock:
            if kind == "read":
                if self._max_read is not None and self.reads >= self._max_read:
                    return False
                self.reads += 1; self.cur = "reader"
            else:
                if self._max_search is not None and self.searches >= self._max_search:
                    return False
                self.cur = "search_direct" if self.searches == 0 else "search_followup"
                self.searches += 1
            return True

    ENGINE_SRC = ("search_direct", "search_followup")

    def add(self, ref, src):
        assert src in self.ENGINE_SRC + ("reader",), f"unknown pool source {src!r}"
        if not ref:
            return
        with self._lock:
            self.pool.append((ref, src))
            (self.engine_pool if src in self.ENGINE_SRC else self.agent_pool).add(str(ref).lower())

    def engine_reachable(self, ref):
        """The engine surfaced this exact ref, or any artifact in the same repo. Container-level
        reads are allowed because the engine already pointed at that repo; the read still lands
        in agent_pool, so it can never be credited as an engine hit."""
        r = str(ref).lower()
        with self._lock:
            if r in self.engine_pool:
                return True
            base = r.split("#")[0]
            return any(p.split("#")[0] == base for p in self.engine_pool)


class Controller:
    """The three rules, applied identically for every driver."""

    def __init__(self, state, cfg, search_tools, scope_hint=None,
                 fixed_params=None, free_tools=(), fetch_tools=()):
        self.s, self.cfg = state, cfg
        self.search_tools = set(search_tools)
        # The vendor's OWN fetch tool (product tier). Spends the READ budget, never the search
        # budget, and is harvested to agent_pool via record_fetch — so the agent can SEE more
        # (that is the product working) but nothing it fetched can be credited as a retrieval
        # hit. That one rule is what keeps a fetch-enabled tier from becoming "whose scraper
        # reads its way to the gold".
        self.fetch_tools = set(fetch_tools)
        # Allowed, but not a retrieval call. `gate` returns before the search counter, so a free
        # tool never spends a budget and nothing it returns is harvested. Context7's
        # `resolve-library-id` is the case this exists for: it maps a library name to an id and
        # returns no documents, so counting it as a search would misreport that arm's search rate.
        # Declared per arm in arms.py and recorded in `free_calls`, never silently free.
        self.free_tools = set(free_tools)
        self.scope_hint = scope_hint
        self.fixed_params = fixed_params or {}
        state._max_search, state._max_read = cfg.max_search, cfg.max_read
        self.schemas = {t: (cfg.manifest.get(t) or {}).get("input_schema")
                        for t in self.search_tools | self.fetch_tools}

    LOCAL_PREFIXES = ("mcp__sub__", "mcp__rd__", "mcp__pg__")
    LOCAL_NAMES = ("submit_citations", "submit_answer", "read_target", "read_page")

    def is_local(self, name):
        return name.startswith(self.LOCAL_PREFIXES) or name in self.LOCAL_NAMES

    def gate(self, name):
        """None = allowed. A string = deny, with the reason the model should be told.

        DENY BY DEFAULT. Every vendor MCP ships extras (crawl, scrape, agent, deep-research);
        letting one arm use them and not another silently changes the task. Observed in the
        smoke run: the agent reached for firecrawl_search and firecrawl_research_search_papers
        on a repo task, and for web_search_exa when pinned to get_code_context_exa."""
        if self.is_local(name):
            return None
        if name in self.free_tools:
            with self.s._lock:
                self.s.free_calls.append(name)
            return None
        if name in self.fetch_tools:
            if not self.s.count("read"):
                with self.s._lock:
                    self.s.budget_denied += 1
                return "BUDGET EXHAUSTED (reads). Submit your answer now."
            return None
        if name not in self.search_tools:
            with self.s._lock:
                self.s.denied.append(name)
            return f"{name} is not the search tool for this arm. Do not retry it."
        if not self.s.count("search"):
            with self.s._lock:
                self.s.budget_denied += 1
            return "BUDGET EXHAUSTED (searches). Submit your answer now."
        return None

    def prepare(self, name, tool_input, call_id="c"):
        """Normalise the request and start the clock. Returns the input to actually send."""
        new, note = P.normalize_input(tool_input or {}, self.schemas.get(name),
                                      self.cfg.k, self.cfg.restrict, self.scope_hint,
                                      fixed=self.fixed_params)
        self.s.clock.start(call_id)
        # KEYED BY call_id, not a single slot. The model can emit two search calls in ONE
        # assistant message; the SDK then fires PreToolUse for both before either PostToolUse,
        # so a single `_pending` was overwritten by the second call and the first one's log line
        # got the WRONG query and the wrong `normalized` note. Scoring never moved (both sources
        # are engine sources), but search_log is the artifact a reviewer reads to check what was
        # actually sent. The Clock was already keyed this way; now the pending record is too.
        with self.s._lock:
            self.s._pending[call_id] = {"tool": name, "q": P.query_of(new), "norm": note,
                                        # the request itself: a fetch needs the requested url
                                        # (Exa returns none), and read options are worth logging
                                        # because vendors expose different, non-comparable ones.
                                        "args": new, "src": self.s.cur}
        return new

    def prepare_fetch(self, name, tool_input, call_id="c"):
        """The same bookkeeping as prepare(), with NO search normalisation. Both drivers call it.

        A fetch NEEDS the pending record: record_fetch reads the requested url out of it (Exa's
        web_fetch_exa returns none) and logs the read options, and the Clock has to be started or
        the call is logged with latency=None. But it must NOT be normalised like a search — k and
        the github scope are SEARCH controls, and `objective` is a query key, so running a fetch
        through prepare() appended "site:github.com" to the objective of a page read. Scoping a
        fetch is us configuring the vendor's reader; there is nothing to scope in "read this url".
        """
        args = dict(tool_input or {})
        self.s.clock.start(call_id)
        with self.s._lock:
            self.s._pending[call_id] = {"tool": name, "q": None, "norm": None,
                                        "args": args, "src": "reader"}
        return args

    def record(self, name, response, call_id="c"):
        """Harvest -> engine_pool + log, and return the TRIMMED result list the model should see.

        The trim is not cosmetic. Vendors return wildly different volumes (measured on
        one query, k=10: Exa 93,102 chars total / 7,793 median per result, Parallel 28,715 /
        1,531); without a shared ceiling the comparison is partly
        decided by how much text landed in context. Full text is kept for read_page and for
        docs grounding. NOTE a cap only trims — it cannot pad — so per-result volume is still
        recorded (`chars`) and must be read alongside any score."""
        dt = self.s.clock.stop(call_id)
        # An MCP error is an AVAILABILITY failure, not a retrieval result. Log it as an error
        # (it lands in `reliability`) instead of letting it harvest to zero results, which the
        # analyzer would book as an index-coverage miss. See provenance.error_envelope.
        # A THROTTLE REPORTED AS PROSE IS STILL A THROTTLE. Context7 answers a rate-limited call
        # with HTTP 200 and "Rate limit exceeded. Please try again in 547 seconds." as the tool
        # RESULT -- 279 of 387 calls in one measured run. error_envelope() only catches isError,
        # so that text harvested as an ordinary result and the cell read as poor retrieval
        # instead of as us being throttled. provenance.vendor_error() was written for exactly
        # this and was never wired to anything.
        errtxt = P.error_envelope(response) or P.vendor_error(response)
        if errtxt:
            pend = self._take(call_id)
            self.s.search_log.append({"q": pend.get("q"), "latency": dt, "n": 0, "results": [],
                                      "chars": 0, "tool": name, "error": errtxt})
            return []
        results, meta = P.harvest(response)
        pend = self._take(call_id)
        src = pend.get("src", "search_direct")
        rich, trimmed = [], []
        for r in results:
            if r.get("ref"):
                self.s.add(r["ref"], src)
            if r.get("url"):
                self.s.url_content[r["url"]] = r.get("text") or self.s.url_content.get(r["url"], "")
            body = r.get("text") or ""
            cut = bool(self.cfg.agent_snippet) and len(body) > self.cfg.agent_snippet
            # `chars` is the FULL length and the full text is persisted in `raw` below, so the
            # preview cap never hides how much a vendor actually returned. `trunc` makes the cut
            # EXPLICIT rather than derivable: a cap on what the model
            # sees is fine, a cap you have to reverse-engineer from two columns is not.
            rich.append({"rank": r["rank"], "url": r.get("url"), "ref": r.get("ref"),
                         "type": r.get("type"), "chars": len(body), **({"trunc": True} if cut else {})})
            trimmed.append({"rank": r["rank"], "ref": r.get("ref"), "url": r.get("url"),
                            "type": r.get("type"),
                            "content": body[:self.cfg.agent_snippet] if self.cfg.agent_snippet else body})
        self.s.raw.append({"tool": name, "q": pend.get("q"), "latency": dt, "n": len(results),
                           "meta": meta,
                           "results": [{"rank": r["rank"], "url": r.get("url"), "ref": r.get("ref"),
                                        "type": r.get("type"), "text": r.get("text") or ""}
                                       for r in results]})
        entry = {"q": pend.get("q"), "latency": dt, "n": len(results), "results": rich,
                 "chars": sum(x["chars"] for x in rich), "tool": name,
                 # How many results the preview cap actually bit on, and at what ceiling. 0 means
                 # the cap was non-binding for this call — which is the usual case and is itself
                 # a finding (see README: the cap trims, it cannot pad).
                 "preview_truncated": sum(1 for x in rich if x.get("trunc")),
                 "preview_cap": self.cfg.agent_snippet,
                 "meta": {**meta, "normalized": pend.get("norm")}}
        # A call that FAILED and a call that legitimately returned nothing both land here as
        # n=0, and they mean opposite things: one is our infrastructure, the other is the
        # engine's coverage. Observed once in a smoke run (a 1,586-char non-JSON body from a
        # query that replays fine), so the difference gets recorded, not inferred later.
        # A response we could not split into results is NOT a coverage miss and must not be
        # scored as one. harvest() now returns a single call-level record with unparsed=True
        # instead of regexing every url out of the blob (which credited engine_pool with repos
        # the engine only linked to). Name it so triage can see it; the agent still gets the text.
        #
        # `unparsed` IS THE WHOLE TEST. An empty n=0 with a non-empty payload also arrives from a
        # payload we parsed FINE that simply held no results — provenance.harvest returns
        # ([], unparsed=False, raw_chars>0) for exactly that, e.g. `gh search repos` answering
        # `{"results": []}` on a bug description. That is an HONEST COVERAGE MISS, the thing this
        # eval measures; flagging it as an error booked it against `reliability` and removed it
        # from the coverage denominator, so an engine returning nothing read as our plumbing.
        if meta.get("unparsed"):
            entry["error"] = "unparsed_response"
            entry["raw_head"] = " ".join(P._blocks(response))[:300]
        self.s.search_log.append(entry)
        return trimmed

    def _take(self, call_id):
        """Pop this call's pending record.

        The fallback exists for a driver that cannot supply a stable id (the neutral loop passes
        a constant), and it is deliberately only safe when there is NOTHING TO CONFUSE IT WITH.
        With one pending record, that record is unambiguously this call's. With several, the
        entries belong to concurrent or abandoned calls -- a raised tool call never reaches
        record()/log_error(), so its entry stays behind -- and popping an arbitrary one logs a
        real search under some other call's query and args. Guessing is worse than an empty row:
        an empty row is visibly missing, a mis-attributed one is silently wrong."""
        with self.s._lock:
            if call_id in self.s._pending:
                return self.s._pending.pop(call_id)
            if len(self.s._pending) == 1:
                return self.s._pending.popitem()[1]
            return {}

    def record_fetch(self, name, response, call_id="c"):
        """PRODUCT TIER. Harvest a vendor fetch call -> `agent_pool` (never engine_pool) + text.

        This is the one place where allowing a vendor's own fetch tool stays safe. The refs it
        surfaces are added with src="reader", exactly like read_target's, so the two-pool rule
        does the rest: the agent can now SEE more (that is the product working), but nothing it
        fetched can be CREDITED as a retrieval hit. Without this the tier would degrade into
        "whose fetcher reads its way to the gold", which is what rule 2 forbids in Tier A/B and
        what a namespace-wide allow-list has no defence against.

        Logged to `fetch_log`, never `search_log`: coverage, reliability and search latency must
        stay properties of the SEARCH surface or the headline stops meaning what it says."""
        dt = self.s.clock.stop(call_id)
        # POP, do not peek. A consumed record left in `_pending` is exactly what `_take`'s
        # popitem() fallback later hands to a real search, which then logs that search under this
        # fetch's url and args. Popped on the error path too, or a failed fetch leaks one.
        with self.s._lock:
            pend = self.s._pending.pop(call_id, None) or {}
        errtxt = P.error_envelope(response)
        if errtxt:
            self.s.fetch_log.append({"tool": name, "latency": dt, "n": 0, "chars": 0,
                                     "error": errtxt})
            return None                          # None => hand the vendor's own error to the model
        # ONE DOCUMENT — not the search parser. See provenance.harvest_document.
        _url, _text = P.harvest_document(response)
        if not _url:
            # Exa's web_fetch_exa returns bare markdown with no url in it at all. We always know
            # which url was requested — it is in the call's own arguments — so use that rather
            # than dropping the document out of url_content, which is where a cited answer's
            # `cited_content` (the persisted evidence for that answer) is read from.
            _a = pend.get("args") or {}
            _req = _a.get("url") or _a.get("urls") or _a.get("uri")
            _url = (_req[0] if isinstance(_req, list) and _req else _req) or ""
        results = [{"rank": 0, "url": _url, "ref": P.ref_from_url(_url),
                    "text": _text, "type": "document"}]
        meta = {"shape": "document", "structured": bool(_url), "blocks": 1,
                "raw_chars": len(_text)}
        for r in results:
            if r.get("ref"):
                self.s.add(r["ref"], "reader")   # <- agent_pool. Cannot score. The whole point.
            if r.get("url"):
                self.s.url_content.setdefault(r["url"], r.get("text") or "")
        # EVERY ref, not the first 10 — this is a persisted record, and a persisted record that
        # silently drops entries is the bug this design exists to remove. The list is short
        # (one fetch = one or a few documents) and it is the only trace of what the vendor's own
        # fetcher reached, which is exactly what a reviewer needs to audit the agent_pool rule.
        # WHICH READ OPTIONS THE AGENT CHOSE. Vendors expose different, non-comparable read
        # controls — firecrawl_scrape has `formats`/`onlyMainContent`, web_fetch_exa has
        # `maxCharacters`, Parallel's web_fetch has none — so there is no shared knob to
        # normalise and we do not invent one. What we CAN do is stop a thin read being
        # misread as a thin vendor: in one measured case the agent called firecrawl_scrape
        # with formats=["summary"] and got 437 chars out of a 6,800-char payload. Without
        # this field that looks like Firecrawl returning nothing; with it, it is the agent
        # picking a lossy format. Attribution, not equalisation.
        _opts = {k: v for k, v in (pend.get("args") or {}).items()
                 if k not in ("url", "urls")}
        self.s.fetch_log.append({"tool": name, "opts": _opts, "latency": dt, "n": len(results),
                                 "chars": sum(len(r.get("text") or "") for r in results),
                                 "refs": [r.get("ref") for r in results if r.get("ref")],
                                 "meta": meta})
        # No trim: the product tier has no preview ceiling (cap=False), and a fetch tool exists
        # precisely to return a full document. Volume is recorded instead of bounded.
        return None

    def log_error(self, name, err, call_id="c"):
        dt = self.s.clock.stop(call_id)
        pend = self._take(call_id)
        self.s.search_log.append({"q": pend.get("q"), "latency": dt, "n": 0, "results": [],
                                  "chars": 0, "tool": name, "error": str(err)[:200]})

    def log_fetch_error(self, name, err, call_id="c"):
        """A fetch that RAISED, as opposed to one that returned an error envelope.

        record_fetch never runs for a raised call, so without this the read simply vanished:
        the clock kept running, `_pending` leaked an entry that `_take`'s popitem() fallback
        could later hand to an unrelated search, and reader reliability read as 100% on a driver
        where every fetch was failing. Logged to `fetch_log`, never `search_log` -- a broken
        reader must not move search coverage or search latency."""
        dt = self.s.clock.stop(call_id)
        with self.s._lock:
            self.s._pending.pop(call_id, None)
        self.s.fetch_log.append({"tool": name, "latency": dt, "n": 0, "chars": 0,
                                 "error": str(err)[:200]})



# ---------------------------------------------------------------------------------
# THE CONTROL ARM: GitHub's own search, via the `gh` CLI.
# ---------------------------------------------------------------------------------
# This is the baseline that matters commercially. `gh` is free, already installed, and already
# authenticated on any developer's machine — so it is the honest "why add anything?" line. If a
# paid dev index does not clearly beat it, the index has no reason to exist for this task.
#
# WHY IT IS A SEARCH TOOL AND NOT A READER. `gh search` is a retrieval engine, so whatever it
# returns is harvested into engine_pool and CAN score, exactly like a vendor's MCP search. Our
# reader also shells out to gh — but to `gh api`, a different command, and its results go to
# agent_pool. Same binary, two roles, cleanly separated: the control cannot read its way to gold.
#
# NO EXTRA SCOPE PARAMETER. GitHub search takes `repo:owner/name`, `language:`, `stars:` etc.
# INSIDE the query string. That is the vendor-sanctioned mechanism, the same status as Parallel's
# `site:` operator, so the model may use it and we inject nothing.
#
# MEASURED before wiring it: GitHub search is LEXICAL, over name/description/topics.
# gh search repos "MWM mobile world model"                       -> AIGeeksGroup/MWM   (hit)
# gh search repos "image-goal navigation world model diffusion"  -> 0 results
# The repo track deliberately withholds the method name (verified 0/300 queries leak it), which
# is exactly where lexical search fails. So a low gh number is "keyword search cannot do this
# task", NOT "gh is broken" — state it that way next to the number.
# The same control, wired to WORK where a naive
# wiring returns nothing:
# * `search_type=hybrid` — GitHub's GA semantic/hybrid issue search. Plain lexical search is a
# different product; run as separate arms, the two score differently.
# * a 256-char guard — GitHub REJECTS longer queries. Our GT maxes at 237, but the AGENT
# concatenates, so without this a too-long query fails as an opaque error.
# * throttling — the search API allows ~30 req/min. At concurrency 3 x 4 searches we would be
# rate-limited and score the limiter, not the index.
GH_MAX_QUERY = 256
GH_MIN_INTERVAL = 2.5              # seconds between GitHub search calls, process-wide
_GH_GATE = threading.Lock()
_GH_LAST = [0.0]


def _gh_throttle():
    with _GH_GATE:
        wait = GH_MIN_INTERVAL - (time.time() - _GH_LAST[0])
        if wait > 0:
            time.sleep(wait)
        _GH_LAST[0] = time.time()


def gh_search(state, args, limit=10, search_type=None, default_kind="repo"):
    """One GitHub search -> an MCP-shaped JSON body, so it flows through the SAME
    provenance.harvest / Controller.record path as every vendor response.

    type=repo  -> `gh search repos`      (name/description/topics — how you find a repository)
    type=issue -> GET /search/issues     (issue+PR bodies; the endpoint the GitHub arm uses)
    type=pr    -> same, filtered to PRs
    """
    q = str(args.get("query", "")).strip()
    # Default to the TRACK's artifact, not always "repo". On the fix track the agent left
    # `type` unset and got `gh search repos` on a bug description — four empty searches per item,
    # its whole budget spent on the wrong endpoint. That measured our default, not GitHub.
    kind = str(args.get("type") or default_kind).lower()
    st = search_type or args.get("search_type")
    if kind not in ("repo", "issue", "pr"):
        kind = "repo"
    if not q:
        return json.dumps({"results": []}), True
    if len(q) > GH_MAX_QUERY:
        return (f"SEARCH ERROR: query is {len(q)} chars; GitHub rejects anything over "
                f"{GH_MAX_QUERY}. Shorten it and retry."), True
    _gh_throttle()
    try:
        if kind == "repo":
            cmd = ["gh", "search", "repos", q, "--limit", str(limit),
                   "--json", "fullName,description,url,stargazersCount"]
        else:
            qq = q + (" is:pr" if kind == "pr" else "")
            path = f"search/issues?q={urllib.parse.quote(qq)}&per_page={limit}"
            if st:
                path += f"&search_type={st}"
            cmd = ["gh", "api", path]
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    except Exception as e:
        return f"GH SEARCH ERROR: {e}", True
    if out.returncode != 0:
        return f"GH SEARCH ERROR: {(out.stderr or '').strip()[:300]}", True
    try:
        raw = json.loads(out.stdout or "[]")
    except Exception:
        return f"GH SEARCH ERROR: unparseable output: {(out.stdout or '')[:200]}", True
    results = []
    if kind == "repo":
        for r in raw:
            results.append({"full_name": r.get("fullName"), "url": r.get("url"),
                            "type": "repository",
                            "text": f"{r.get('fullName')} ({r.get('stargazersCount', 0)} stars)\n"
                                    f"{r.get('description') or ''}"})
    else:
        for it in (raw.get("items") or []):
            url = it.get("html_url") or ""
            results.append({"url": url, "type": "pr" if "/pull/" in url else "issue",
                            "text": f"{it.get('title') or ''}\n{it.get('body') or ''}"})
    return json.dumps({"results": results, "search_type": st or "lexical"}), False


# ---------------------------------------------------------------------------------
# Local tools — identical for every arm and every driver. The reader in particular MUST NOT
# vary: if one vendor's own fetch tool were allowed, that arm could read its way to the gold
# and we would be scoring the fetcher instead of the index.
# ---------------------------------------------------------------------------------
def read_target(state, args):
    """Read ONE GitHub target to verify it. Gated on 'the engine surfaced it'."""
    repo = str(args.get("repo", "")).strip("/ ")
    num = args.get("number")
    try:
        num = int(num) if num not in (None, "", 0) else None
    except Exception:
        num = None
    want = f"{repo}#{num}" if num else repo
    if not state.engine_reachable(want):
        with state._lock:
            state.blocked.append(want)
        return (f"NOT IN RESULTS: {want} was not returned by search. You may only verify "
                "candidates your search surfaced."), True
    if not state.count("read"):
        return "BUDGET EXHAUSTED (reads). Submit now.", True
    try:
        if num:
            d = json.loads(subprocess.run(["gh", "api", f"repos/{repo}/issues/{num}"],
                                          capture_output=True, text=True, timeout=40).stdout or "{}")
            cm = subprocess.run(["gh", "api", f"repos/{repo}/issues/{num}/comments?per_page=100"],
                                capture_output=True, text=True, timeout=40).stdout
            comments = [{"user": (c.get("user") or {}).get("login"), "body": c.get("body") or ""}
                        for c in (json.loads(cm) if cm else [])]
            state.add(f"{repo}#{num}", "reader")
            out = {"title": d.get("title"), "state": d.get("state"),
                   "body": d.get("body") or "", "comments": comments}
        else:
            d = json.loads(subprocess.run(["gh", "api", f"repos/{repo}"],
                                          capture_output=True, text=True, timeout=40).stdout or "{}")
            rd = subprocess.run(["gh", "api", f"repos/{repo}/readme"],
                                capture_output=True, text=True, timeout=40).stdout
            readme = base64.b64decode(json.loads(rd)["content"]).decode(errors="replace") if rd else ""
            state.add(repo, "reader")
            out = {"repo": repo, "description": d.get("description"),
                   "stars": d.get("stargazers_count"), "readme": readme}
    except Exception as e:
        return f"READ ERROR: {e}", True
    return json.dumps(out), False


def read_page(state, args, page_full=None):
    if not state.count("read"):
        return "BUDGET EXHAUSTED (reads). Submit now.", True
    url = str(args.get("url", "")).strip()
    full = state.url_content.get(url, "")
    if not full:
        return "No stored content for that url — pass a url exactly as returned by search.", True
    return (full[:page_full] if page_full else full), False


def submit_citations(state, args, k=10):
    state.submission = [str(c) for c in (args.get("citations") or [])][:k]
    return f"Submitted {len(state.submission)}. Done.", False


def submit_answer(state, args, k=10):
    """Docs submission. An ANSWER WITH NO SOURCES IS REJECTED ONCE.

    WHY. Measured on the full docs run: the agent submitted an empty `sources` array
    on 221/294 items (fc) and 170/294 (exa) — 75% and 58%. `groundedness` is computed over the
    CITED sources, so an empty list scores 0 by definition. The result: `correctness` 0.517
    vs `groundedness` 0.085, a 6x collapse that had nothing to do with retrieval — a direct probe
    found the answer content present in 7 of 8 "failures". The metric was measuring
    instruction-following, and it did so UNEVENLY across arms, which is worse than measuring it
    badly: most of the fc/exa groundedness gap was the citation-rate gap.

    `required: ["answer","sources"]` does not help — an empty array satisfies the schema.

    So the rule is enforced here, at the boundary both drivers share, exactly like the tool gate:
    a committed answer must carry at least one source. ABSTENTION IS STILL ALLOWED — "I don't
    know" with no sources is a legitimate response the prompt explicitly asks for, so it passes.
    The rejection happens once; if the agent submits again with no sources, we accept and record
    `uncited=True` so the item is visible as a compliance failure rather than a retrieval one."""
    ans = str(args.get("answer") or "")
    src = [str(u) for u in (args.get("sources") or [])][:k]
    abstain = (not ans.strip()) or "i don't know" in ans.lower() or "i dont know" in ans.lower()
    if src or abstain or state.answer_rejected:
        state.answer = ans
        state.sources = src
        state.uncited = bool(not src and not abstain)
        return "Submitted. Done.", False
    state.answer_rejected = True
    return ("REJECTED: you gave an answer but no sources. Every answer must cite at least one "
            "url from your search results. Call submit_answer again with `sources` populated "
            "(or answer \"I don't know\" with empty sources if nothing relevant was found)."), True


# Tool descriptions in ONE place. runner_sdk.py's @tool decorators pull the description string
# from here rather than inlining it, so the wording a submit/read tool presents to the model can't
# drift between where it's declared and where it's documented.
LOCAL_TOOL_SPECS = {
    "gh_search": {
        "description": "Search GitHub itself. type='repo' finds repositories, 'issue' finds "
                       "issues and PRs, 'pr' finds pull requests only. Defaults to the artifact "
                       "this task asks for. GitHub search qualifiers "
                       "such as repo:owner/name, language:python or stars:>100 go INSIDE the query.",
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "GitHub search query"},
            "type": {"type": "string", "enum": ["repo", "issue", "pr"],
                     "description": "what to search for"}},
            "required": ["query"]}},
    "submit_citations": {
        "description": "Submit FINAL ranked answer (up to 10 ids, most likely first). Call once.",
        "parameters": {"type": "object", "properties": {
            "citations": {"type": "array", "items": {"type": "string"},
                          "description": "Ranked ids: 'owner/repo' or 'owner/repo#number'."}},
            "required": ["citations"]}},
    "submit_answer": {
        "description": "Submit your FINAL answer (1-2 sentences) and the source URLs you used. Call once.",
        "parameters": {"type": "object", "properties": {
            "answer": {"type": "string"},
            "sources": {"type": "array", "items": {"type": "string"}}},
            "required": ["answer", "sources"]}},
    "read_target": {
        "description": "Read ONE GitHub target to verify it matches. fix: repo+number; repo: repo only.",
        "parameters": {"type": "object", "properties": {
            "repo": {"type": "string", "description": "owner/repo"},
            "number": {"type": "integer", "description": "issue or PR number (omit for a repo)"}},
            "required": ["repo"]}},
    "read_page": {
        "description": "Read the FULL content of ONE search result (pass its exact url).",
        "parameters": {"type": "object", "properties": {"url": {"type": "string"}},
                       "required": ["url"]}},
}

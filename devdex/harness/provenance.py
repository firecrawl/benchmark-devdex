"""provenance.py — the hard part of an MCP-level eval: knowing what the ENGINE returned.

In an adapter harness the search tool is OURS, so `engine_pool` (the set of refs the backend
actually surfaced) fell out for free, and precision could honestly exclude anything the
model guessed from memory. With a vendor MCP the results arrive as an opaque tool
result, so that set has to be RECONSTRUCTED from the tool response. If you skip this,
the eval silently degrades into "did Opus remember the repo", which is the single most
common way a retrieval benchmark lies.

Two hooks do all of it:

  PreToolUse   deny-by-default gate + budget + parameter normalisation (updatedInput)
  PostToolUse  harvest refs/urls/text -> engine_pool, and cap the preview the agent sees
               (updatedToolOutput) so no vendor wins on payload volume

Both are pure functions of the tool name/input/response, so every decision they make is
recorded and re-derivable from the saved run.
"""
import json, re, time

GH_ARTIFACT = re.compile(r"github\.com/([\w.-]+/[\w.-]+)/(?:issues|pull)/(\d+)")
GH_REPO = re.compile(r"github\.com/([\w.-]+/[\w.-]+?)(?:/|$|#|\?|\s|\))")
URL_RE = re.compile(r"https?://[^\s\"'<>)\]]+")

# Structured keys a vendor may use for a bare "owner/repo" (no URL). Harvesting bare refs
# ONLY from these keys is deliberate: a regex over prose would also credit repos the
# engine merely mentioned in a sentence, which inflates coverage. A known undercount was
# traced to exactly this in an earlier eval — the fix is structured keys, not looser regex.
REPO_KEYS = ("repo", "repository", "full_name", "repo_full_name", "nameWithOwner")
# MEASURED against the live servers. `excerpts` (PLURAL) is Parallel's content key
# and was missing here, so EVERY Parallel result harvested with text="" — the agent saw a bare
# URL list, `chars` logged 0, and on the docs track `cited_content` was empty, which forces
# groundedness=0 and therefore grounded_correctness=0 for the whole arm BY CONSTRUCTION.
# A missing key in this tuple is not a parsing detail, it is a silent zero for a competitor.
# `passages` is Firecrawl's /v2/developer/search key (docs.firecrawl.dev/features/developer);
# it is a list of {"text": ...} objects, which _pick resolves via _texts_of.
TEXT_KEYS = ("content", "text", "markdown", "snippet", "summary", "excerpt", "excerpts",
             "passage", "passages", "highlights", "body", "description")
TITLE_KEYS = ("title", "name", "heading")
URL_KEYS = ("url", "link", "html_url", "permalink", "source", "uri")


def ref_from_url(url):
    m = GH_ARTIFACT.search(url or "")
    if m:
        return f"{m.group(1)}#{m.group(2)}"
    m = GH_REPO.search(url or "")
    if m and m.group(1).count("/") == 1:
        return m.group(1).removesuffix(".git")
    return None


def _blocks(resp):
    """Flatten an MCP tool_response into its text blocks. Vendors differ: some return
    {'content':[{'type':'text','text':...}]}, some a bare list, some a dict already.

    `structuredContent` is the SAME payload again in machine form (MCP ships both when a tool
    declares an output schema — Parallel does). Walking both counted every byte twice: measured
    58,899 raw_chars and 39 blocks on a 29k single-block response, which lands straight in the
    per-arm volume comparison. Prefer `content` and ignore the duplicate."""
    if isinstance(resp, dict) and resp.get("content") is not None and "structuredContent" in resp:
        resp = {k: v for k, v in resp.items() if k != "structuredContent"}
    out = []
    def walk(o):
        if isinstance(o, str):
            out.append(o)
        elif isinstance(o, dict):
            if o.get("type") == "text" and isinstance(o.get("text"), str):
                out.append(o["text"])
            else:
                for v in o.values():
                    walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(resp)
    return out


def _texts_of(v):
    """A content value in any shape a vendor actually ships: a string, a list of strings
    (Parallel `excerpts`), or a list of {"text": ...} objects (Firecrawl `passages`)."""
    if isinstance(v, str):
        return v if v.strip() else ""
    if isinstance(v, list):
        parts = []
        for x in v:
            if isinstance(x, str) and x.strip():
                parts.append(x)
            elif isinstance(x, dict):
                t = x.get("text") or x.get("content") or x.get("passage")
                if isinstance(t, str) and t.strip():
                    parts.append(t)
        return "\n".join(parts)
    return ""


def _pick(d, keys):
    for k in keys:
        t = _texts_of(d.get(k))
        if t:
            return t
    return ""


def _from_json(obj, acc):
    """Depth-first over parsed JSON, emitting one record per dict that looks like a result.

    Does NOT recurse into a dict it already emitted. A result object routinely nests other
    url-bearing objects (an author profile, a repository stub, a source record); recursing made
    each of those a RESULT OF ITS OWN, which both credited the engine with refs it never ranked
    and shifted every later result's rank by one (measured: an `author.url` landing at rank 2,
    demoting the real second hit). Rank is what the scorer's engine-side rank metrics are computed
    over (scorer/suite.py `_engine_rank` / `_docs_rank`, both capped at 10), so this is not
    cosmetic — a phantom result at rank 2 pushes a real gold past the cap."""
    if isinstance(obj, list):
        for o in obj:
            _from_json(o, acc)
        return
    if not isinstance(obj, dict):
        return
    url = _pick(obj, URL_KEYS)
    bare = _pick(obj, REPO_KEYS)
    if url or bare:
        ref = ref_from_url(url) or (bare if bare.count("/") == 1 else None)
        # The title is content the agent should see — the markdown path already folds it in,
        # so the json path must too or the two shapes are not comparable.
        title, body = _pick(obj, TITLE_KEYS), _pick(obj, TEXT_KEYS)
        acc.append({"url": url, "ref": ref,
                    "text": f"{title}\n{body}".strip() if title else body,
                    "type": obj.get("type") or obj.get("artifact") or ""})
        # Emitted — so do not walk this dict's OBJECT children (author, repository, source:
        # metadata of this result, not results). But DO walk its LIST children. Without that,
        # an envelope that carries a url of its own AND holds the result array — e.g.
        # {"url": "<the request url>", "results": [...]} — returned as ONE result (itself) and
        # every real hit was silently dropped. A phantom result is bad; losing the entire
        # payload and logging it as an honest zero-coverage miss is worse.
        for v in obj.values():
            if isinstance(v, list):
                _from_json(v, acc)
        return
    for v in obj.values():
        if isinstance(v, (dict, list)):
            _from_json(v, acc)


# A result URL sits alone on its line, optionally behind a label. Both forms are in the
# wild today: Firecrawl emits `https://…` bare under a `[owner/repo] (type)` header, Exa
# emits `URL: https://…` under a `Title: …` header. One anchor pattern covers both.
ANCHOR = re.compile(r"^\s*(?:(?:URL|Url|url|Link|Source)\s*:\s*)?(https?://\S+)\s*$")
HDR_REF = re.compile(r"\[([\w.-]+/[\w.-]+)(?:#(\d+))?\]")
# THE DEVELOPER INDEX'S STABLE TYPED ID. Firecrawl heads every hit with
# `## [<type>:<rest>] (<type>) <title>`, where <rest> is `owner/repo[#n]` for issue /
# pull_request / readme but a CONTENT HASH for doc:
# [issue:pandas-dev/pandas#42763]      [readme:owner/repo]
# [doc:736005df3f088c460f60f859dc2f4c47ab7809a0]
# HDR_REF above needs a slash, so every doc: header failed it and we recorded ref=None —
# measured 2/5 refs extracted on a live 5-result response, and 87% vs Exa's 100% across the
# fix run. Two separate things were lost: the repo-backed ref for issue/pr/readme (recoverable
# from the url, which is why the damage was only partial) and the DOC ID, which has no url
# equivalent. That id is the only stable, mirror-independent handle on a document — the docs
# track's canonical target should be `doc:<hash>`, not a github blob path that the same content
# also lives at on the rendered site.
HDR_TYPED = re.compile(r"\[(doc|issue|pull_request|pr|readme)\s*:\s*([^\]]+)\]")
HDR_TITLE = re.compile(r"^\s*(?:Title|Name)\s*:\s*(.+)$")


def _from_markdown(blob):
    """Split a markdown/prose result page into per-result blocks.

    Vendors that return a formatted document instead of JSON still delimit results the same
    way: an optional header line, the result URL alone on its own line, then the body.
    Anchoring there is what recovers per-result attribution — without it every inline image
    and every link inside a README counts as its own "result" (measured: 31–45 phantom
    results on a payload that actually held 10) and all the text is attributed to result #1.
    Both numbers feed the fairness comparison, so getting this wrong silently decides it."""
    lines = blob.splitlines()
    anchors = [(i, m.group(1)) for i, l in enumerate(lines) if (m := ANCHOR.match(l))]
    if len(anchors) < 2:              # one anchor is a link in prose, not a result list
        return []
    out = []
    for j, (i, url) in enumerate(anchors):
        header = lines[i - 1].strip() if i > 0 else ""
        # A header may legitimately CONTAIN a markdown link — Firecrawl's is
        # `## [pull_request:owner/repo#N] (pull_request) closes [#42763](https://…)`. Testing the
        # raw line for a URL threw that whole header away and with it the result's `type`
        # (measured: type='' on the rank-1 hit of a live developer-search response). Strip link
        # targets first, then test: a line that is still URL-bearing really is body text.
        if URL_RE.search(re.sub(r"\]\([^)]*\)", "]", header)):
            header = ""
        end = anchors[j + 1][0] - 1 if j + 1 < len(anchors) else len(lines)
        body = "\n".join(lines[i + 1:max(i + 1, end)]).strip().strip("-").strip()
        # Typed id first (Firecrawl), then the bare `[owner/repo#n]` form (other vendors).
        doc_id, typed = None, HDR_TYPED.search(header)
        ref = None
        if typed:
            doc_id = f"{typed.group(1)}:{typed.group(2).strip()}"
            rest = typed.group(2).strip()
            m3 = re.match(r"([\w.-]+/[\w.-]+?)(?:#(\d+))?$", rest)
            if m3:
                ref = f"{m3.group(1)}#{m3.group(2)}" if m3.group(2) else m3.group(1)
        if not ref:
            hm = HDR_REF.search(header)
            ref = ref_from_url(url) or (f"{hm.group(1)}#{hm.group(2)}" if hm and hm.group(2)
                                        else (hm.group(1) if hm else None))
        tm = HDR_TITLE.match(header)
        if tm:                                    # a title is content the agent should see
            body = f"{tm.group(1)}\n{body}".strip()
            typ = ""
        else:
            m2 = re.search(r"\(([^)]+)\)", header)
            typ = (m2.group(1).split(",")[0] if m2 else header.split("]")[-1]).strip() if header else ""
        out.append({"url": url, "ref": ref, "text": body, "type": typ, "id": doc_id})
    return out


DOC_TEXT_KEYS = ("markdown", "content", "text", "extract", "html", "raw", "summary")
DOC_URL_KEYS = ("url", "sourceURL", "source_url", "link", "finalUrl")


def harvest_document(resp):
    """A FETCH response -> (url, text). ONE document, not a ranked list.

    A search response is a list of results, each carrying its own url and snippet, so
    `_from_json` looks for objects holding BOTH. A fetch response is shaped the other way
    round — Firecrawl's scrape returns

        {"markdown": "<16,604 chars>", "metadata": {"url": ..., "description": "<302 chars>"}}

    content at the ROOT, url in a CHILD. Run through harvest(), the root is skipped (no url),
    the parser descends into `metadata`, emits THAT as the result and takes `description` as its
    text. MEASURED: a 22,973-char page recorded as 437 chars of page metadata.

    The AGENT was unaffected — the SDK hook passes fetch responses through untouched. What was
    corrupted is `url_content`, the store every docs record's `cited_content` is built from: the
    persisted evidence for a cited answer became a 437-char meta description instead of the page
    the agent actually read, which is exactly the kind of hole an audit of that answer falls
    into."""
    blocks = _blocks(resp)
    blob = "\n".join(blocks)
    for b in blocks:
        t = b.strip()
        if not t.startswith(("{", "[")):
            continue
        try:
            o = json.loads(t)
        except Exception:
            continue
        if isinstance(o, list):
            o = next((x for x in o if isinstance(x, dict)), {})
        if not isinstance(o, dict):
            continue
        # THREE SHAPES, all live all different:
        # firecrawl_scrape  {"markdown": ..., "metadata": {"url": ...}}   root text, child url
        # web_fetch (par)   {"results": [{"url":..., "excerpts": [...]}]}  a LIST of documents
        # web_fetch_exa     raw markdown, no JSON and no url anywhere
        # so: unwrap a document list, then look for text/url at root or under metadata.
        for key in ("results", "data", "documents", "pages"):
            v = o.get(key)
            if isinstance(v, list) and v and isinstance(v[0], dict):
                o = v[0]
                break
        meta = o.get("metadata") if isinstance(o.get("metadata"), dict) else {}
        text = _pick(o, DOC_TEXT_KEYS) or _pick(meta, DOC_TEXT_KEYS)
        url = _pick(o, DOC_URL_KEYS) or _pick(meta, DOC_URL_KEYS)
        if text or url:
            return url, (text or blob)
    return "", blob        # not JSON — the vendor returned markdown/prose directly


def error_envelope(resp):
    """Return the error text if this tool_response is an MCP ERROR, else None.

    WHY THIS EXISTS: without it a rate-limited / auth-failed / 5xx call harvests to zero
    results and is logged exactly like "the engine returned nothing" — which this eval scores
    as an INDEX COVERAGE MISS. Coverage-vs-ranking is the headline diagnostic, so silently
    booking infrastructure failures as coverage gaps would put a vendor's throttling into our
    product conclusion. An earlier harness hit the same thing on api.firecrawl.dev's per-minute
    limit. Errors are recorded as errors and surface in `reliability`, never as a miss."""
    if isinstance(resp, dict):
        if resp.get("isError") or resp.get("is_error"):
            return " ".join(_blocks(resp))[:300] or "isError"
        for v in resp.values():                       # SDK may nest the envelope one level
            if isinstance(v, dict) and (v.get("isError") or v.get("is_error")):
                return " ".join(_blocks(v))[:300] or "isError"
    txt = " ".join(_blocks(resp)).strip()
    low = txt[:200].lower()
    # LENGTH GUARD. This benchmark's `fix` track is made of error reports, so a perfectly good
    # result list whose first hit is an issue titled "Error: cannot import name X" matched the
    # prefix test and was thrown away whole: zero results, nothing into engine_pool, and the call
    # booked against `reliability`. A vendor error envelope is short; a result page is not.
    if txt and len(txt) < 2000 and any(low.startswith(p) for p in
                                       ("error:", "mcp error", "tool error", "request failed",
                                        "rate limit")):
        return txt[:300]
    # Vendors also prefix with the TOOL NAME: "web_search_advanced_exa error (402): ...".
    if re.match(r"^[\w.-]{3,60}\s+error\s*[(:]", txt[:80], re.I):
        return txt[:300]
    return None


# A vendor that reports throttling as PROSE inside a 200 response is indistinguishable from a
# search that legitimately found nothing. Context7 returned "Rate limit exceeded. Please try
# again in 547 seconds." as the tool RESULT for 279 of 387 calls in one run; harvest() parsed it
# as a single ref, so the cell read as poor retrieval (dead_runs 0.570) instead of as us being
# throttled. Anything matching this is an ERROR to surface, never a result.
VENDOR_ERROR = re.compile(
    # `limit(?:ed| exceeded)?` — the optional suffix — matched bare "rate limit" with no error
    # context at all, so a legitimate result whose URL or title merely mentions rate limiting
    # (e.g. a docs page at docs/v3/concepts/rate-limits.mdx) was misread as a vendor throttle and
    # discarded as a dead call instead of scored. \b anchors plus a REQUIRED "ed"/"exceeded"
    # suffix keep the true positive (Context7's literal "Rate limit exceeded...") while dropping
    # the plain noun phrase. Leading \b also stops "corporate limit" matching on the "rate" tail.
    r"\brate[ _-]?limit(?:ed|\s+exceeded)\b|too many requests|quota (?:exceeded|exhausted)|"
    r"try again in \d+|exceeded your .{0,20}limit|upgrade your plan|higher limits",
    re.I)


def vendor_error(payload):
    """Return the offending snippet when a 200-status payload is really a vendor refusal."""
    try:
        t = payload if isinstance(payload, str) else json.dumps(payload)[:4000]
    except Exception:
        t = str(payload)[:4000]
    m = VENDOR_ERROR.search(t)
    return t[max(0, m.start() - 40):m.end() + 60].strip() if m else None


def harvest(resp):
    """tool_response -> [{url, ref, text, type}] in the vendor's own rank order.

    `shape` in the returned meta records HOW the vendor answered — json | markdown | prose.
    That is not bookkeeping: it decides whether per-result char accounting is even possible,
    and payload shape is part of the product being evaluated."""
    texts = _blocks(resp)
    acc, shape = [], "prose"
    for t in texts:
        s = t.strip()
        if s.startswith(("{", "[")):
            try:
                _from_json(json.loads(s), acc)
                shape = "json"
                continue
            except Exception:
                pass
    unparsed = False
    if not acc:
        # A payload we PARSED that legitimately contains zero results is a COVERAGE MISS, not a
        # parse failure. Only fall through to the text paths when the JSON attempt never happened.
        # Conflating the two double-counts: the call lands in `reliability` as an error AND is
        # hidden from the coverage denominator, so an engine that genuinely returns nothing looks
        # like our infrastructure broke. Example: `gh search repos` on a bug description returns
        # `{"results": []}` — an honest empty, flagged as unparsed.
        if shape == "json":
            return [], {"shape": "json", "structured": True, "blocks": len(texts),
                        "raw_chars": sum(len(t) for t in texts), "unparsed": False}
        blob = "\n".join(texts)
        acc = _from_markdown(blob)
        if acc:
            shape = "markdown"
        else:
            # LAST RESORT, and deliberately a SINGLE record. The old fallback emitted one
            # "result" per url anywhere in the blob, which credited engine_pool with every repo
            # merely LINKED FROM inside another result's body — measured: a 1-result payload
            # whose README linked torvalds/linux and psf/requests harvested as 4 results, two of
            # them never ranked by the engine. That inflates pool_hit (the coverage headline) and
            # widens the read_target gate, and it does it unevenly across vendors: it only fires
            # for the vendor whose format the parser failed on. This module's own rule for bare
            # refs — "never from prose, a prose regex also credits what was merely mentioned" —
            # applies just as much to urls. So: attribute the blob to the CALL, claim no ranking,
            # and let `unparsed` mark the call for triage instead of scoring it.
            if blob.strip():
                first = next(URL_RE.finditer(blob), None)
                u = first.group(0).rstrip(".,);") if first else ""
                acc = [{"url": u, "ref": None, "text": blob, "type": ""}]
                unparsed = True
    out, seen = [], set()
    for r in acc:
        key = (r.get("url") or "") + "|" + (r.get("ref") or "")
        if key == "|" or key in seen:
            continue
        seen.add(key)
        r["rank"] = len(out)
        out.append(r)
    meta = {"shape": shape, "structured": shape == "json", "blocks": len(texts),
            "raw_chars": sum(len(t) for t in texts), "unparsed": unparsed}
    # The HTTP /v2/developer/search response carries a per-index-type `coverage` map
    # (doc/issue/pull_request/readme -> ok|degraded|unavailable|skipped) and vendors may ship
    # `warnings`. analyze.reliability() already reads meta.degraded — it was simply never being
    # populated, so a degraded index scored as a quality miss. Capture it where it exists, and
    # note that the hosted MCP surface does NOT expose coverage today.
    cov = _find_key(resp, "coverage")
    if isinstance(cov, dict):
        meta["coverage"] = cov
        meta["degraded"] = any(v not in ("ok", "skipped") for v in cov.values() if isinstance(v, str))
    warn = _find_key(resp, "warnings")
    if warn:
        meta["warnings"] = str(warn)      # persisted diagnostic -> keep it whole
    return out, meta


_FIND_SCAN = 200


def _find_key(o, key, depth=0):
    """First value for `key` anywhere in the response (vendors nest it differently)."""
    if depth > 6:
        return None
    if isinstance(o, dict):
        if key in o:
            return o[key]
        for v in o.values():
            r = _find_key(v, key, depth + 1)
            if r is not None:
                return r
    elif isinstance(o, list):
        # Search bound, not truncation: this walks the response looking for a `coverage`/`warnings`
        # key, it does not build the result set. Wide enough for any real payload shape.
        for v in o[:_FIND_SCAN]:
            r = _find_key(v, key, depth + 1)
            if r is not None:
                return r
    return None


# ---------------------------------------------------------------------------------
# Parameter normalisation. ONLY keys the tool's own schema declares are injected, so we
# never invent a capability a vendor does not have — we record its absence instead.
# ---------------------------------------------------------------------------------
K_KEYS = ("k", "numResults", "num_results", "max_results", "maxResults", "limit", "topK", "count")
DOMAIN_KEYS = ("includeDomains", "include_domains", "domains", "site", "allowed_domains")
# Query lives under a different key per vendor, and is not always a string: Parallel's
# `web_search` takes `objective` (a string) PLUS `search_queries` (a list). Scope hints have
# to reach every one of them or one arm silently searches the open web while another is
# github-scoped — the exact asymmetry this module exists to prevent.
QUERY_KEYS = ("query", "q", "search_queries", "objective", "search", "prompt")


def _append_hint(v, hint):
    if isinstance(v, str) and v.strip() and hint not in v:
        return f"{v} {hint}"
    if isinstance(v, list):
        return [f"{x} {hint}" if isinstance(x, str) and hint not in x else x for x in v]
    return v


def normalize_input(inp, schema, k, scope_github, scope_hint, query_keys=QUERY_KEYS, fixed=None):
    """Return (updatedInput, note). note records WHAT WAS AND WAS NOT expressible."""
    props = set((schema or {}).get("properties", {}).keys())
    new = dict(inp or {})
    note = {"k": None, "scope": None, "fixed": None}
    # Arm-pinned params (e.g. Exa category="github"), injected ONLY where declared — same
    # discipline as k and scope: never hand an engine a capability its schema does not have.
    if fixed:
        applied = {k2: v for k2, v in fixed.items() if k2 in props}
        new.update(applied)
        note["fixed"] = applied or None
    for key in K_KEYS:
        if key in props:
            new[key] = k
            note["k"] = key
            break
    if scope_github:
        for key in DOMAIN_KEYS:
            if key in props:
                spec = ((schema or {}).get("properties", {}).get(key) or {})
                new[key] = ["github.com"] if spec.get("type") == "array" else "github.com"
                note["scope"] = key
                break
        if note["scope"] is None and scope_hint:
            touched = [qk for qk in query_keys if qk in new and new[qk]]
            for qk in touched:
                new[qk] = _append_hint(new[qk], scope_hint)
            if touched:
                note["scope"] = f"query:{scope_hint}"
    return new, note


def query_of(inp, keys=QUERY_KEYS):
    """The human-readable query for the log. Joins list-valued query fields so a Parallel
    call that fanned out into three sub-queries is not logged as if it were one."""
    for k in keys:
        v = (inp or {}).get(k)
        if isinstance(v, str) and v.strip():
            return v
        if isinstance(v, list) and v:
            return " | ".join(str(x) for x in v)
    # Fallback only: no recognised query key. Persisted, so say when it was cut rather than
    # leaving a silently-clipped blob in the log (see the record-completeness rule above). Never fires for the three
    # live arms — fc/exa use `query`, Parallel uses `objective` + `search_queries`.
    blob = json.dumps(inp)
    return blob if len(blob) <= 2000 else blob[:2000] + f"…[+{len(blob)-2000} chars]"


class Clock:
    """Pre/Post pairing by tool_use_id -> per-call wall time.

    CAVEAT, stated wherever this number is printed: this is the MCP ROUND TRIP (transport
    + server process + vendor search), not the vendor's search latency. A stdio server
    spawned via npx and a remote HTTPS server are not on the same footing, so latency is
    a diagnostic in this harness and NOT an index-tier SLA metric."""
    def __init__(self):
        self.t = {}
    def start(self, tuid):
        self.t[tuid] = time.time()
    def stop(self, tuid):
        t0 = self.t.pop(tuid, None)
        return round(time.time() - t0, 3) if t0 else None

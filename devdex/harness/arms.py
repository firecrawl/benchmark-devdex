"""arms.py — the arm registry. THIS IS THE ONLY FILE YOU EDIT TO SWAP AN ENGINE.

A python-adapter harness swaps a FUNCTION (adapter -> raw HTTP). This harness swaps an
MCP SERVER: the agent calls the vendor's own MCP tool, exactly as a developer's agent
would. That changes what is being measured — tool descriptions, parameter defaults,
result formatting and the shape of the returned payload are now part of the product
under test, not something we wrote around.

    index tier     : "how good is the index?"          (we control the request)
    devdex-mcp     : "how good is the product?"        (the vendor controls the request)

Both are worth having. Neither replaces the other, and the numbers are NOT comparable
across the two harnesses — see README, "Why the numbers will not match".

--------------------------------------------------------------------------------------
`fc-mcp` calls the real product, `firecrawl_developer_search`, on the HOSTED Firecrawl MCP.
The stdio `npx firecrawl-mcp` server does not expose that tool, so this arm MUST run against the
hosted MCP — pointing it at stdio silently falls back to the wrong corpus (the research index).
--------------------------------------------------------------------------------------

Fairness at the MCP layer is HARDER than at the HTTP layer, and pretending otherwise is
the main way this kind of eval goes wrong. Three things are enforced here:

  1. ONE search tool per arm.        Every vendor MCP ships extra tools (crawl, scrape,
     agent, deep-research). Exposing them to one arm and not another silently changes
     the task. `search_tools` is an allowlist and everything else is denied twice —
     `allowed_tools` in the SDK options AND a deny-by-default PreToolUse hook, because
     `allowed_tools` alone has leaked before.
  2. ONE vendor fetch tool per arm, and nothing it reads can ever be CREDITED. Every vendor
     ships exactly one reader (firecrawl_scrape / web_fetch_exa / web_fetch), so handing each
     arm its own is symmetric in kind and makes reading part of the product under test; an arm
     with no fetch tool reads with OUR `read_target` (the gh API), gated on "the engine
     surfaced it". Either way control.record_fetch harvests to agent_pool and never to
     engine_pool, so no arm can read its way to the gold — which is the only reason allowing a
     vendor fetcher is safe. See `fetch_tools` on Arm.
  3. Parameters normalised where the tool DECLARES them. k / result-count and github
     scoping are injected by a PreToolUse hook (`updatedInput`), but ONLY for keys that
     appear in the tool's own input schema, read from `tools_manifest.json` (written by
     preflight.py). A vendor whose tool cannot express k or a domain filter simply does
     not get one, and that asymmetry is RECORDED per call rather than hidden.
"""
import os

# Target results/call, injected into whichever key the tool declares (k / numResults / …).
#
# 10, NOT Firecrawl's documented default of 20. The A/B was run before choosing (repo, n=15,
# a smaller driver, identical items):
#
# fc-mcp    precision -0.067   coverage (pool_hit) +0.000   chars/q  22,520 ->  50,148
# exa-mcp   precision +0.000   coverage (pool_hit) -0.066   chars/q 132,782 -> 328,984
#
# Coverage is the metric depth is supposed to move, and for Firecrawl it moved by EXACTLY ZERO:
# doubling the result budget surfaced no additional golds. Its misses are not depth misses — the
# gold is not sitting at rank 11-20, it is not retrieved at all. Meanwhile per-question payload
# roughly doubled for both arms, and at k=20 Exa returned 844,359 chars in a single response,
# blowing the MCP output ceiling and failing the arm outright until the cap was raised.
#
# So k=20 bought no accuracy and cost 2x the context plus a live failure. 10 it is.
#
# Depth parity is what actually matters here and it holds at any value: K is injected into every
# arm whose schema declares a count parameter, so fc and exa both run at 10. Parallel declares
# none and is unaffected — recorded per call in meta.normalized, never faked.
#
#
# Env-driven, so re-checking the product default is one flag: `K=20 python3 runner_*.py ...`.
K = int(os.environ.get("K", 10))
TRACKS = ("repo", "fix", "docs")


class Arm:
    """One experimental arm. `servers` is passed straight to ClaudeAgentOptions.mcp_servers."""
    __slots__ = ("name", "kind", "servers", "search_tools", "fetch_tools", "scope_hint",
                 "fixed_params", "note", "tracks", "free_tools", "role", "no_scope", "no_depth")

    def __init__(self, name, kind, servers=None, search_tools=None, fetch_tools=None,
                 scope_hint=None, fixed_params=None, note="", tracks=None, free_tools=(),
                 role="engine", no_scope=False, no_depth=False):
        self.name = name
        self.kind = kind                        # "mcp" | "local" | "native" | "none"
        self.servers = servers or {}            # {ns: McpServerConfig}
        # track -> ORDERED CANDIDATE LIST. Exactly one is live per run: the first candidate the
        # vendor actually exposes today (see resolve()). More than one entry is a declared
        # fallback for a rename, NOT a second tool the agent gets.
        self.search_tools = search_tools or {}
        # PRODUCT TIER: the vendor's OWN fetch tool, one per arm. Every vendor ships exactly one
        # (firecrawl_scrape / web_fetch_exa / web_fetch), so giving each arm its own is symmetric
        # part of the product under test. It cannot corrupt scoring: record_fetch harvests to
        # agent_pool, so a fetched ref can never be credited as a retrieval hit.
        self.fetch_tools = fetch_tools or {}
        self.scope_hint = scope_hint            # query-level github scoping when no param exists
        # Opt OUT of the repo/fix github restriction even though the tool declares a domain param.
        # The restriction exists to stop an arm wasting its k slots on non-github noise when the
        # gold IS a github artifact. But injecting it is US configuring the product. fc-web is
        # run as the DOCUMENTED call — a plain query, no filter — because that is the surface the
        # docs describe; adding includeDomains would measure our tuning, not the product, and
        # would hand it a filter fc-mcp's tool cannot express.
        self.no_scope = no_scope
        # The vendor exposes no k/limit, so its result list is a different size from every other
        # arm's. Rate metrics computed over that list (pool_hit, found@k, engR1, MRR) are not
        # comparable and are suppressed rather than printed alongside depth-10 numbers.
        self.no_depth = no_depth
        # Extra params pinned on every call, injected ONLY if the tool's schema declares them
        # (e.g. Exa's category="github"). Same discipline as k/scope: never invent a capability.
        self.fixed_params = fixed_params or {}
        self.note = note
        # Tracks this arm can legitimately run. Context7 indexes library DOCS — no repos, no
        # issues, no PRs — so running it on repo/fix would produce a ~0 that reflects its CORPUS,
        # not its quality. An empty cell is honest; a zero is a lie.
        self.tracks = set(tracks or TRACKS)
        # Allowed, but NOT a search: nothing harvested, and the search counter is never reached.
        # Context7 requires `resolve-library-id` before it will answer, and that call retrieves no
        # documents — it maps a library name to an id. Counting it as a search would misreport the
        # arm's search rate for what is a lookup step.
        self.free_tools = set(free_tools)
        # "engine"    — a general developer-search index that runs ALL THREE tracks. These are the
        # head-to-head comparison and the only arms that share a ranking.
        # "control"   — a baseline, not a competitor: what you already have (gh), what the model
        # remembers (no-tool), what a generic agent has (websearch).
        # "vertical"  — a specialist whose corpus covers ONE track. Context7 indexes library docs
        # and nothing else, so ranking it against a general index would compare a
        # scalpel to a toolbox. Reported on its own line, never in the ranking.
        # scorer/report_metrics.py prints engines as THE table and everything else as reference
        # lines below.
        self.role = role

    def runs(self, track):
        return track in self.tracks

    def fetch_for(self, track):
        return list(self.fetch_tools.get(track) or self.fetch_tools.get("*") or [])

    def tools_for(self, track):
        """The CANDIDATE list. Callers that build a live toolset must use resolve()."""
        return list(self.search_tools.get(track) or self.search_tools.get("*") or [])

    def resolve(self, track, manifest=None):
        """Candidates -> (live_tool, fallback_note). Exactly ONE tool goes live.

        WHY THIS EXISTS. A pinned tool name is brittle: rename it vendor-side and every call is
        denied, which reads exactly like a product scoring 0.000 (CONTEXT §11). An earlier harness
        dodges that by allow-listing the whole `mcp__exa` namespace — immune to renames, but it
        also hands the arm every tool on the server, which is the one thing Tier A may not do.

        So: declare the fallback instead of wildcarding. The first candidate the vendor exposes
        TODAY goes live, the arm keeps exactly one tool, and a fallback is never silent — it is
        stamped on every record as `tool_fallback` and printed by preflight, because substituting
        a surface can change the CORPUS or the SCOPE (Exa's plain web search cannot express
        includeDomains). A substituted arm is readable, not quotable.

        Arms with no equivalent surface declare no fallback on purpose: Firecrawl Developer has
        no second tool over the same corpus, so a rename there MUST fail preflight rather than
        quietly fall back to the research index — the wrong corpus for this arm."""
        cands = self.tools_for(track)
        if not cands:
            return None, None
        if not manifest:                        # no preflight: primary, and main() already warns
            return cands[0], None
        live = [t for t in cands if t in manifest]
        if not live:
            return cands[0], None               # preflight/guards report this as a hard failure
        if live[0] != cands[0]:
            return live[0], {"from": cands[0], "to": live[0],
                             "why": "primary tool absent from the live manifest"}
        return live[0], None


# ---------------------------------------------------------------------------------
# Firecrawl — THE REAL PRODUCT.
# ---------------------------------------------------------------------------------
# `firecrawl_developer_search` (Firecrawl Developer, docs.firecrawl.dev/features/developer)
# is the index built for coding agents: GitHub issues, merged PRs, repository READMEs and
# curated docs sites. That is exactly this benchmark's artifact space, so repo/fix/docs all
# map onto one tool.
#
# IT IS ONLY ON THE HOSTED MCP. Verified against one key: hosted mcp.firecrawl.dev exposes
# 27 tools INCLUDING firecrawl_developer_search; `npx firecrawl-mcp` (stdio) exposes 26 and
# does NOT. Being wired to stdio is the entire reason this arm was stuck on the research
# placeholder. Do not "simplify" this back to npx without re-running preflight.
#
# The key is in the URL PATH (that is Firecrawl's hosted-MCP auth scheme), so it is read from
# the environment and never committed. The account also needs the developer-search beta flag —
# an un-flagged key returns 403 "Developer search is in beta and is not enabled for this team".
FC_MCP_URL = os.environ.get(
    "FIRECRAWL_MCP_URL",
    f"https://mcp.firecrawl.dev/{os.environ.get('FIRECRAWL_API_KEY', '')}/v2/mcp")
FC_SERVER = {"fc": {"type": "http", "url": FC_MCP_URL, "alwaysLoad": True}}

# The MCP surface declares ONLY (query, k). The HTTP API additionally exposes types, repos,
# min_stars, language, license, fork/archived — none of which are reachable here. So at the
# MCP layer Firecrawl cannot filter to issues-only on the fix track, exactly as Parallel
# cannot filter by domain. Recorded per call in meta.normalized, not silently equalised.
FC_DEV = "mcp__fc__firecrawl_developer_search"
# Firecrawl's general web search, the surface fc-web uses. It is scored AS DELIVERED: the response
# carries data.web (and data.developer when a categories filter is set), and the agent sees the
# merged list, not a filtered slice — filtering would score a result set the agent never received.
#
# It differs from firecrawl_developer_search in ways that are the point of running it:
# content  a `description` per result, not matched passages
# depth    limit applies PER GROUP — limit=10 returned 20 results (10 web + 10 developer)
# filters  includeDomains / sources / tbs, none of which the dedicated tool exposes
# auth     keyless-capable
# The response returns JSON carrying both groups.
FC_SEARCH = "mcp__fc__firecrawl_search"
FC_FETCH = "mcp__fc__firecrawl_scrape"       # the vendor's own reader (product tier)
# Checked against the hosted MCP server (firecrawl-fastmcp 3.23.0): it
# exposes 27 tools including firecrawl_developer_search, and its schema declares ONLY (query, k).
# The HTTP API documents types / repos / sources / passages / language / topic / license /
# min_stars / max_stars / archived / fork — none reachable at the MCP layer.
# (Retired from Tier A: firecrawl_research_search_github, the paper-built research corpus this arm
# used before. Not wired anywhere; deleted rather than left as a dead constant that
# reads like an option.)

# ---------------------------------------------------------------------------------
# Exa — remote MCP. `?tools=` pins the tool set so the arm cannot drift when Exa changes
# its defaults (their docs already disagree with their release notes about what is on by
# default; pinning turns that into a versioned, reproducible choice).
#
# SCOPING — the fairness decision that actually moves this eval. Exa's other two tools
# (`get_code_context_exa`, `web_search_exa`) accept ONLY (query, numResults): no domain filter,
# no category. Firecrawl's github tool is github-only BY CONSTRUCTION, so pairing it against an
# unscopeable Exa tool hands Firecrawl a scope advantage the competitor cannot express.
# Measured on one repo query: the code tool returned 1 GitHub repo in 10 slots (the rest
# arxiv/project pages); `web_search_advanced_exa` + includeDomains=["github.com"] returned
# 10/10 with the gold at rank 1 — not a ranking difference, 90% of the result budget spent
# off-target. So Exa runs on the ADVANCED tool: the only surface that can be scoped, and the
# the same mechanism used over HTTP (exa /search + includeDomains).
# ---------------------------------------------------------------------------------
# PIN THE WHOLE SURFACE ON THE SERVER, GATE IT DOWN TO ONE IN THE HARNESS. These are two
# different jobs and conflating them is what made the old single-name pin brittle:
# ?tools=          decides what the SERVER exposes  -> pin all four, so the surface is a
# versioned, reproducible contract and a rename shows up in preflight as a
# manifest diff instead of an unreachable tool.
# allowed_tools +  decides what the AGENT may call  -> still exactly one for Tier A/B.
# the gate
# rename-immune but also hands that arm three tools while Firecrawl's gets two — the asymmetry
# Tier A exists to eliminate. Pinning wide and gating narrow buys the immunity without it.
# Pinned to the primary, its DECLARED FALLBACK (Arm.resolve) and the vendor's own fetch tool.
# web_fetch_exa is on the surface because the PRODUCT TIER gives every arm its own reader
# (Arm.fetch_tools) — record_fetch keeps that safe by harvesting it to agent_pool. Exa's CODE
# tool is the one deliberately left off: the gate would deny it anyway, and an unusable tool the
# model can see is just wasted turns (observed — the agent reached for web_search_exa when pinned
# elsewhere). this pin returns exactly the tools named in it.
MINT_SERVER = {"mint": {"type": "http", "url": "https://index.mintlify.com/mcp",
                       "alwaysLoad": True}}          # public, no API key
MINT_CTX = "mcp__mint__context"

EXA_TOOLS = "web_search_advanced_exa,web_search_exa,web_fetch_exa"
# EXA_ANON=1 — send NO api key. Measured: mcp.exa.ai serves anonymous callers on every
# tool (web_search_exa, get_code_context_exa, web_search_advanced_exa), honours `includeDomains`
# and `numResults`, and did not throttle at 12 calls / concurrency 3. The 402 on an exhausted key
# is the ACCOUNT, not the tool: every tool 402s with such a key and every tool works without one.
#
# IT IS NOT A DROP-IN. It is a different service tier from the paid Firecrawl beta key the fc arm
# runs on, its rate limit and result depth are undocumented, and it inverts this harness's own
# rule for Parallel ("always send the token — an arm that gets throttled is measuring the wrong
# thing"). So it is OPT-IN, never automatic, and it is stamped into `arm_note` on every record so
# a number can never be quoted without its tier. Top the account up before publishing anything.
EXA_ANON = os.environ.get("EXA_ANON") == "1"
EXA_HEADERS = {} if EXA_ANON else {"x-api-key": os.environ.get("EXA_API_KEY", "")}
EXA_NOTE = ("scopeable surface: includeDomains=[github.com] on repo/fix, open on docs"
            + (" | ANONYMOUS TIER (EXA_ANON=1, no api key) — different service tier from the "
               "keyed arms; state this next to any exa number" if EXA_ANON else ""))
EXA_SERVER = {"exa": {"type": "http", "url": f"https://mcp.exa.ai/mcp?tools={EXA_TOOLS}",
                      "headers": EXA_HEADERS,
                      # Load tool schemas EAGERLY. Without this the SDK can defer them behind
                      # ToolSearch — which this harness blocks — and the arm makes zero searches
                      # and scores 0.000 for a plumbing reason.
                      "alwaysLoad": True}}
EXA_FETCH = "mcp__exa__web_fetch_exa"        # the vendor's own reader (product tier)
EXA_ADV = "mcp__exa__web_search_advanced_exa"   # scopeable: includeDomains + category
EXA_WEB = "mcp__exa__web_search_exa"            # plain web search — NOT scopeable
# NO CATEGORY SCOPING — `includeDomains` is the mechanism, and `category="github"` is not real.
#
# An earlier version offered EXA_SCOPE=category on the theory that `category="github"` was the
# analogue of domain scoping. It is not. Exa's category enum is a FIXED set —
# company | people | research paper | news | personal site | financial report — and Exa's own
# published `exa-search` skill says outright: "do not invent categories like `github`,
# `documentation`, `qa`, or `pdf`". Such an arm would have sent a value Exa does not accept and
# produced a meaningless result under an official-looking flag.
#
# `includeDomains` is the supported route, Exa's own schema gives `github.com` as its example,
# and it is also simply the best Exa surface for this task. Measured over 8 repo queries:
#
# web_search_advanced_exa + includeDomains   160/160 on-target   gold 7/8   median rank 1
# web_search_advanced_exa, unscoped           28/160             gold 7/8   median rank 2
# get_code_context_exa (Exa's code product)   25/160             gold 5/8   median rank 6
#
# Note what that says: scoping barely changes whether Exa FINDS the gold (7/8 either way) — it
# changes how much of the result budget is usable. And Exa's purpose-built code surface is the
# WEAKEST of the three here, so running the advanced tool is the generous choice, not a slight.
EXA_FIXED = {}

# ---------------------------------------------------------------------------------
# Parallel — remote Search MCP. Free/unauthenticated for light use; the Bearer token
# raises the rate limit, so we always send it (an arm that gets throttled is measuring
# the wrong thing).
# ---------------------------------------------------------------------------------
# ENDPOINT: /mcp-oauth, NOT /mcp. This is the whole reason the parallel arm looked broken all
# day — 33% errors at cc=8, ~3.6 min/item at cc=2, cells that never reached a checkpoint — while
# a single cold probe always answered in ~1s because one call fits inside the free allowance.
#
# Parallel's docs say /mcp "accepts optional Bearer token" for higher limits. It does not.
# Measured on the same key, same minute:
# /mcp        + x-api-key   -> 429 "You've hit the free-tier rate limit"
# /mcp        + Bearer      -> 429  (same)
# /mcp        + both        -> 429  (same)
# /mcp-oauth  + Bearer      -> 200, 3 back-to-back searches at 1.2 / 0.8 / 0.8s
# The key itself was never the problem: api.parallel.ai/v1beta/search returned 200 with the same
# key at the same moment. Authentication simply is not honoured on /mcp.
#
# SUPERSEDED NOTE: an earlier x-api-key curl returned 200 and looked like the fix. It was a
# free-tier allowance that happened to be available, not successful auth. Parallel's MCP accepts a Bearer header
# without complaint and then serves the request as ANONYMOUS FREE TIER — their own 429 body
# lists both forms as valid, but only x-api-key actually authenticates. Measured on
# the same key, same session, same query:
# Authorization: Bearer  ->  HTTP 429 "You've hit the free-tier rate limit"
# x-api-key              ->  HTTP 200, 0.8s, 61,611 bytes
# This is the whole reason the parallel arm looked broken all day: 33% errors at cc=8, ~3.6
# min/item at cc=2, and a "stalled" cell that never reached its first checkpoint — while a
# single cold probe always answered in ~1s, because one call fits inside the free allowance.
# Not a rate-limit problem, not concurrency, not a stale session: a silently-ignored header.
PAR_SERVER = {"par": {"type": "http", "url": "https://search.parallel.ai/mcp-oauth",
                      "headers": {"Authorization": f"Bearer {os.environ.get('PARALLEL_API_KEY', '')}"},
                      "alwaysLoad": True}}      # eager schemas — see EXA_SERVER
PAR_FETCH = "mcp__par__web_fetch"            # the vendor's own reader (product tier)
PAR_WEB = "mcp__par__web_search"                # the SEARCH surface; web_fetch is the reader above

# Parallel exposes NO domain filter at the MCP layer — its `web_search` takes only
# (objective, search_queries, session_id, model_name). Its own schema does say search_queries
# "may include search operators", so `site:github.com` is the vendor-sanctioned mechanism
# rather than a hack. It is a WEAK lever though: measured 2-3 github refs in 10 slots with or
# without it, against Exa's 10/10 under includeDomains. Parallel therefore competes with less
# scope control than the others; that is a property of its MCP surface, it is recorded per
# call (`meta.normalized.scope`), and it must be stated next to any Parallel number.
GH_HINT = "site:github.com"


# ---------------------------------------------------------------------------------
# THE CONTROL: GitHub's own search via the `gh` CLI. Free, already installed, already
# authenticated — the honest "why add anything?" baseline. Implemented in control.gh_search as an
# in-process tool (NOT Bash: no arm gets a shell), harvested into engine_pool like any vendor
# search. Our reader uses `gh api`, a different command whose results go to agent_pool, so the
# control cannot read its way to the gold. Docs track excluded: `gh` does not index docs sites.
# ---------------------------------------------------------------------------------
GH_TOOL = "mcp__gh__gh_search"

# ---------------------------------------------------------------------------------
# Context7 — a documentation index built for coding agents. DOCS IS ITS ONLY IN-DOMAIN TRACK: it
# indexes library documentation keyed by library, with no GitHub repos, issues or PRs. It is run on
# all three tracks so every arm has a reproducible cell, but a repo/fix number measures corpus
# coverage (mostly dead, out-of-domain) rather than retrieval, and those cells are excluded from
# ranking by the dead-run bar. Our docs GT is passages from exactly its corpus
# (huggingface/transformers, pydantic, next.js), which makes it the natural docs competitor.
#
# It is a TWO-STEP product: `resolve-library-id` then `query-docs`. That does not break the
# one-search-tool rule, because only `query-docs` retrieves documents — `resolve-library-id`
# is a name->id lookup, declared here as a FREE tool (allowed, no search budget, not harvested).
# 2 tools, no API key required.
# ---------------------------------------------------------------------------------
# The key is a QUOTA key, not a capability key. Measured keyed vs anonymous, same
# questions: library inference from a bare question 0/4 either way, and query-docs returned an
# identical n=2 / 805 chars. So it buys rate limit, nothing else — and we send it for exactly the
# reason we always send Parallel's Bearer token: an arm that gets throttled at 294 docs items x 3
# passes is measuring the limiter, not the index.
# C7_ANON=1 — send NO api key, exactly like EXA_ANON and for the same reason. A keyed call can
# return a monthly-quota error on every query-docs call while the SAME call with no Authorization
# header returns real documentation: the cap is attached to the ACCOUNT, not the tool, so removing
# the key removes the limit. A monthly cap does not clear by waiting, and a second exhausted key
# changes nothing.
#
# SAME CAVEAT AS EXA: this is a different service tier from a paid key, so it is OPT-IN and is
# stamped into the arm note, and any published Context7 number must say which tier produced it.
C7_ANON = os.environ.get("C7_ANON") == "1"
C7_KEY = "" if C7_ANON else os.environ.get("CONTEXT7_API_KEY", "")
C7_SERVER = {"c7": {"type": "http", "url": "https://mcp.context7.com/mcp",
                    **({"headers": {"Authorization": f"Bearer {C7_KEY}"}} if C7_KEY else {}),
                    "alwaysLoad": True}}
C7_RESOLVE = "mcp__c7__resolve-library-id"
C7_QUERY = "mcp__c7__query-docs"

ARMS = {
    # ---- The three engines. ONE search tool each, k and github scope normalised where the
    # vendor's schema declares them, uniform preview cap, shared reader. Only the index varies.
    "fc-mcp": Arm("fc-mcp", "mcp", FC_SERVER,
                  {"repo": [FC_DEV], "fix": [FC_DEV], "docs": [FC_DEV]},
                  fetch_tools={"*": [FC_FETCH]},
                  note="Firecrawl Developer index (real product; hosted MCP only)"),
    # Firecrawl's general web search (firecrawl_search), run as the documented call: no categories
    # filter, no domain scoping. A different surface from the Developer Index — it returns a merged
    # web+developer list, scored as delivered — kept as a plain-web-search baseline.
    "fc-web": Arm("fc-web", "mcp", FC_SERVER,
                  {"repo": [FC_SEARCH], "fix": [FC_SEARCH], "docs": [FC_SEARCH]},
                  fetch_tools={"*": [FC_FETCH]},
                  no_scope=True,
                  note="firecrawl_search — Firecrawl's general web search, no categories filter"),
    # Mintlify Index. Publisher-maintained docs + web, one tool, NO api key. Its `context` tool
    # returns "### <title> / Source: <url> / <content>" blocks -- the same shape Context7 emits,
    # which provenance.harvest() already parses, so no new parser.
    #
    # DEPTH IS NOT CONTROLLABLE. There is no k/limit parameter; measured 1-7 sources per call
    # (median 2) and tokenBudget does not change it. Every other engine returns 10. precision,
    # correctness and groundedness stay comparable because they ask "did the agent commit to the
    # right thing"; pool_hit / found@10 / engR1 do NOT, because they are rates over a result list
    # a third the size. no_depth=True makes the scorer report those as unmeasurable.
    #
    # Public endpoint is capped at 10 req/s and 1,000 req/day per IP. A full pass over all three
    # tracks is ~1,400 requests, so it does not fit in one day -- run one track per day and use
    # --resume-dir to continue into the next.
    "mintlify": Arm("mintlify", "mcp", MINT_SERVER,
                    {"repo": [MINT_CTX], "fix": [MINT_CTX], "docs": [MINT_CTX]},
                    fixed_params={"tokenBudget": 6000}, no_scope=True, no_depth=True,
                    note="Mintlify Index — publisher-sourced docs+web, keyless, vendor-fixed depth"),
    # EXA_FALLBACK: two candidates, ONE live (Arm.resolve). web_search_exa is the declared
    # stand-in if Exa retires the advanced tool — same index, but it cannot express
    # includeDomains, so a run that falls back is stamped `tool_fallback` and reported as
    # SUBSTITUTED. fc and parallel declare no fallback: neither vendor has a second surface over
    # the same corpus, so a rename there must fail preflight instead of silently swapping corpora.
    "exa-mcp": Arm("exa-mcp", "mcp", EXA_SERVER,
                   {"repo": [EXA_ADV, EXA_WEB], "fix": [EXA_ADV, EXA_WEB],
                    "docs": [EXA_ADV, EXA_WEB]},
                   fetch_tools={"*": [EXA_FETCH]}, fixed_params=EXA_FIXED, note=EXA_NOTE),
    "parallel-mcp": Arm("parallel-mcp", "mcp", PAR_SERVER,
                        {"repo": [PAR_WEB], "fix": [PAR_WEB], "docs": [PAR_WEB]},
                        fetch_tools={"*": [PAR_FETCH]}, scope_hint=GH_HINT),

    # ---- The control and the docs competitor. Both are declared exceptions to "every arm runs
    # every track", because neither vendor's corpus spans all three — see `tracks` on Arm.
    # ONE GitHub arm, not two: a lexical variant and a "hybrid" variant are the SAME ARM here:
    # * repo track — `search_type` is only appended to the /search/issues path, never to
    # `gh search repos`, so the two run a byte-identical command by construction.
    # * fix track  — 4/4 ground-truth queries returned byte-identical result sets with and
    # without `search_type=hybrid`.
    # Running both would have printed two "independent" control lines that are one measurement,
    # and doubled the control spend for nothing. `search_type=hybrid` is kept on the surviving
    # arm (it is inert rather than harmful) but the NAME must not
    # imply we measured GitHub's semantic mode as distinct — on this GT, we could not.
    "gh-hybrid": Arm("gh-hybrid", "local", search_tools={"repo": [GH_TOOL], "fix": [GH_TOOL]},
                     tracks={"repo", "fix"}, role="control",
                     fixed_params={"search_type": "hybrid"},
                     note="CONTROL: GitHub's own search — free and preinstalled, the 'why add a "
                          "dev index at all?' line. `search_type=hybrid` is sent but measured "
                          "INERT (identical results with and without, 4/4 fix queries; not "
                          "applied at all on repo). Read a low number carefully: GitHub's issue "
                          "search returned ZERO results for natural-language bug descriptions, "
                          "so this is a keyword engine being handed prose — a real property of "
                          "the free option, not a ranking failure."),
    # RUNS ALL THREE TRACKS, but its corpus is library DOCS ONLY. repo/fix are out-of-domain:
    # query-docs is keyed by a library that a repo/bug query does not name, so those tracks return
    # mostly nothing and post a high dead-run fraction. They are run anyway so the table has a real,
    # REPRODUCIBLE cell for every arm, then excluded from ranking by the >10% dead-run bar (and so
    # from the combined, which needs all three tracks live) -- exactly how gh-hybrid's missing docs
    # cell is handled. Read repo/fix as coverage, not retrieval quality; docs is its real result.
    "context7": Arm("context7", "mcp", C7_SERVER, {"*": [C7_QUERY]},
                    free_tools={C7_RESOLVE}, role="vertical",
                    no_depth=True,
                    note="Context7 indexes library DOCS; docs is its only in-domain track. It runs "
                         "repo/fix too so every arm has a reproducible cell, but there it is "
                         "out-of-domain and mostly dead -- excluded from ranking by the dead-run "
                         "bar, not scored as retrieval quality. "
                         "Two-step product: resolve-library-id (free lookup, no search budget) "
                         "then query-docs, which is keyed by LIBRARY and infers nothing from a "
                         "bare question (measured 0/5). That BLOCKED it on docs_v3.1.0, where "
                         "only 14% of queries named a library; docs_v5.0.0 names one in 100%, so "
                         "the arm finally runs as designed. Read its v5 number against that "
                         "change, not against its v3 number."
                         + (" | ANONYMOUS TIER (C7_ANON=1, no api key): every available key "
                            "returned 'Monthly quota reached' — an ACCOUNT cap that does not "
                            "clear by waiting — while the same call unauthenticated returns real "
                            "docs. Different service tier from a paid key; state this next to "
                            "any context7 number." if C7_ANON else "")),

    # ---- Reference lines. websearch is a rough baseline (its pool is harvested from the model
    # stream, a laxer standard); no-tool is the memory floor and MUST score ~0.
    "websearch": Arm("websearch", "native", role="control"),
    "no-tool": Arm("no-tool", "none", role="control"),
}

def live_tools(arm, track, manifest=None):
    """The tools that actually go live for this (arm, track) -> (search, fallback).

    Exactly ONE search tool, always. Everything that builds an allowlist goes through here, so
    "how many tools does this arm have" has one answer and the two drivers cannot disagree."""
    if arm.kind not in ("mcp", "local"):
        return [], None
    if not arm.runs(track):
        return [], None
    tool, fb = arm.resolve(track, manifest)
    return ([tool] if tool else []), fb


# ---------------------------------------------------------------------------------
# EXTERNAL SUBMISSION. Registers a third-party engine as the arm `external`, so `--arm external`
# drives the SAME agent loop, gate, depth cap, provenance and scorer as every published arm.
#
# REGISTERED HERE, not in the benchmark wrapper, because run_eval validates --arm against this
# registry in the PARENT process and then spawns the runner as a CHILD. Registering in the wrapper
# only mutates the parent, so the child would reject its own arm name.
#
# ---- PREFERRED ROUTE: YOUR MCP SERVER ----
# Every published arm is the vendor's OWN MCP surface mounted as shipped -- its tool description,
# its parameter defaults, its result formatting. Those are product decisions and they move the
# score: result counts per search range from ~2 (Mintlify) to ~17 (fc-web), and
# readers return 914 to 30,650 chars. Hand us a REST adapter instead and WE author the tool the
# agent sees, which measures our wrapper rather than your index. So point us at the server you
# ship to your own users.
#
# DEVDEX_EXT_MCP_URL=https://mcp.you.com/mcp \
# DEVDEX_EXT_SEARCH_TOOL=your_search \
# DEVDEX_EXT_FETCH_TOOL=your_fetch      # optional; omit if you ship no reader
# DEVDEX_EXT_AUTH="Bearer $YOUR_KEY"    # optional
#
# `limit`/`includeDomains` are injected ONLY where your own schema declares them -- the same
# discipline every other arm gets, so we never hand you a capability you do not ship, nor withhold
# one you do.
if os.environ.get("DEVDEX_EXT_MCP_URL"):
    _ext_url = os.environ["DEVDEX_EXT_MCP_URL"]
    _ext_search = os.environ.get("DEVDEX_EXT_SEARCH_TOOL", "search")
    _ext_fetch = os.environ.get("DEVDEX_EXT_FETCH_TOOL")
    _ext_srv = {"ext": {"type": "http", "url": _ext_url, "alwaysLoad": True}}
    if os.environ.get("DEVDEX_EXT_AUTH"):
        _ext_srv["ext"]["headers"] = {"Authorization": os.environ["DEVDEX_EXT_AUTH"]}
    _ext_stool = f"mcp__ext__{_ext_search}"
    ARMS["external"] = Arm(
        "external", "mcp", _ext_srv,
        {"repo": [_ext_stool], "fix": [_ext_stool], "docs": [_ext_stool]},
        fetch_tools=({"*": [f"mcp__ext__{_ext_fetch}"]} if _ext_fetch else None),
        note=(f"EXTERNAL SUBMISSION over MCP — url={_ext_url} tool={_ext_search!r}. Mounted as "
              f"shipped, exactly like every published arm. Fetch: "
              + (f"{_ext_fetch!r}." if _ext_fetch else
                 "NOT provided. Arms without a reader score lower on repo/fix, where reading "
                 "rescues 0.12-0.23 recall for arms that have one; that is a product difference, "
                 "recorded rather than corrected.")))


# Sanity, enforced at import so a bad edit cannot reach a run: exactly one tool goes live per
# track. More than one and "only the backend varies" quietly becomes "and also how many tools it
# has" — the single property this harness exists to hold.
for _a in ARMS.values():
    if _a.kind != "mcp":
        continue
    for _t in TRACKS:
        if not _a.runs(_t):                 # declared: this arm's corpus does not cover the track
            continue
        assert _a.tools_for(_t), f"{_a.name}/{_t}: needs at least one search tool"
        assert len(live_tools(_a, _t)[0]) == 1, (
            f"{_a.name}/{_t}: exactly one search tool must go live; candidates are a declared "
            "rename fallback, not extra tools")

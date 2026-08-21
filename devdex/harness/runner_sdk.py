"""runner_sdk — the agent-side eval; the search backend is an MCP SERVER.

One Claude Opus agent, one developer question, three tracks (repo | fix | docs). The only
thing that changes between arms is which MCP server is mounted and which single tool on it
the agent is allowed to call. Ground truth, scoring, and the record schema are IDENTICAL to
the same schema, so `devdex/scorer/report_metrics.py` scores these runs unchanged.

  fc-mcp        Firecrawl hosted MCP — `firecrawl_developer_search`, the real Developer index
                (GitHub issues + merged PRs + READMEs + docs sites). NOT the stdio npx server,
                which does not expose that tool.
  exa-mcp       Exa remote MCP (`web_search_advanced_exa` — the only Exa surface that can be
                github-scoped, so it is the scope-matched competitor; see arms.py)
  parallel-mcp  Parallel Search MCP (`web_search`)
  websearch     Claude native web search (reference line)
  no-tool       memory floor — MUST be ~0 or the run is invalid

Usage: python3 runner_sdk.py <repo|fix|docs> <arm> <n|all> <concurrency> [tag]
Run preflight.py FIRST — it pins each vendor's tool names/schemas into tools_manifest.json.
"""
import json, os, re, signal, subprocess, sys, tempfile, time, threading

# Auto-memory is loaded into the agent's system prompt REGARDLESS of setting_sources, so it
# must be switched off before the first SDK call (subprocesses inherit it). Without this, any
# note saved under ~/.claude/projects/<this dir>/memory/ — including notes about THIS eval's
# results — would be read by the agent under test.
os.environ["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] = "1"

# The CLI caps an MCP tool result (default ~25k tokens) and REPLACES the payload with
# "Error: result (N characters) exceeds maximum allowed tokens". MEASURED: that killed 2 of 4
# Exa searches (85k and 76k chars) — the results never reached the agent AND never entered
# engine_pool, so they scored as index-coverage misses. It penalises verbose vendors only
# (Exa ~85k/call vs Firecrawl ~12k), i.e. it manufactures exactly the coverage gap this eval
# reports. Raise the cap so every payload reaches our PostToolUse hook, and let the UNIFORM
# AGENT_SNIPPET trim be the only thing that bounds what the model sees.
#
# RAISED 200k -> 800k. At k=20 Exa returned 844,359 chars (~211k tokens) in ONE
# response on the fix track, blew the 200k ceiling, and the whole payload was replaced by the
# error string — the arm failed with `no_successful_search`. Raising k re-created the exact bug
# this line was added to fix. The ceiling only has to be large enough for a payload to REACH our
# PostToolUse hook; what the model sees is still bounded by AGENT_SNIPPET (4,000 chars/result),
# so a bigger cap costs nothing and a too-small one silently deletes a vendor's results.
os.environ.setdefault("MAX_MCP_OUTPUT_TOKENS", "800000")

import anyio
from concurrent.futures import ThreadPoolExecutor
from claude_agent_sdk import (AssistantMessage, ClaudeAgentOptions, HookMatcher,
                              ResultMessage, TextBlock, create_sdk_mcp_server, query, tool)
try:
    from claude_agent_sdk import ToolUseBlock
except ImportError:
    ToolUseBlock = None

import arms as ARMS_MOD
import control as C
import provenance as P

HERE = os.path.dirname(os.path.abspath(__file__))
# Ground truth lives in devdex/gt. BENCH_DIR overrides, for running this harness against an
# external ground-truth checkout.
_here = os.path.dirname(os.path.abspath(__file__))
# BENCH_DIR overrides, for running against an external ground-truth checkout.
BENCH = os.environ.get("BENCH_DIR") or _here
DATASETS = os.path.abspath(os.path.join(_here, "..", "gt"))
sys.path.insert(0, BENCH)

# OUT_DIR is set by devdex/scorer/suite.py so each pass writes into its own
# <output-dir>/<timestamp>-<label>/run<N>/ directory instead of the working directory.
# Unset (bare invocation) keeps the old flat-file behaviour.
OUT_DIR = os.environ.get("OUT_DIR", "")
def _out(name):
    if not OUT_DIR:
        return name
    os.makedirs(os.path.join(OUT_DIR, "tasks"), exist_ok=True)
    return os.path.join(OUT_DIR, "tasks", name)

TRACK = sys.argv[1]
ARM_NAME = sys.argv[2]
N = sys.argv[3] if len(sys.argv) > 3 else "all"
CONCURRENCY = int(sys.argv[4]) if len(sys.argv) > 4 else 4
TAG = sys.argv[5] if len(sys.argv) > 5 else "mcp"
ARM = ARMS_MOD.ARMS[ARM_NAME]
INTENT = {"repo": "find_repo", "fix": "find_fix", "docs": "find_docs"}[TRACK]

# Same GT file, same items, same order as the source corpus — that is the whole point of
# referencing one corpus rather than copying it: one corpus, two consumption paths.
# run_eval.py always sets GT_FILE explicitly; this fallback only fires when the harness is
# invoked directly, so it must point at what this checkout actually ships.
GT_FILE = os.environ.get("GT_FILE", os.path.join(DATASETS, "repo_public.jsonl"))
MODEL = os.environ.get("MODEL", "claude-opus-4-8")
K = ARMS_MOD.K
# How many ranked ids the agent may submit. NOT the same thing as K (search depth), even though
# both were wired to K originally — so raising depth to 20 silently let the agent submit 20
# citations, against a tool description that says "up to 10", an @10 metric suite, and a docs
# citation_precision that averages over however many sources were cited. Retrieval depth and
# answer length are separate experimental parameters and must not move together.
CITE_K = int(os.environ.get("CITE_K", 10))

# Arms may opt out (Arm.no_scope) when the vendor's documented call takes no domain filter.
RESTRICT = TRACK in ("repo", "fix") and not ARM.no_scope

# SKILLS, loaded the way they actually work in production (as in production): the vendor's
# SKILL.md is installed into an isolated jail at <jail>/.claude/skills/<name>/SKILL.md, the
# agent's cwd is set to that jail, and the agent DISCOVERS and invokes it through the Skill
# tool. Appending the text to the system prompt would have been easier and wrong — it forces
# the guidance on every turn instead of letting the agent decide to reach for it, which is the
# behaviour the product actually ships.
#
# The jail holds ONLY the skill: no real project CLAUDE.md, no settings, nothing from this
# checkout can leak into the run.
#
# Default OFF. A skills-on matrix scores "who wrote documentation" alongside "whose index is
# better", and the two cannot be separated after the fact. Run the A/B and report the delta per
# arm, never a skills-on number alone.
#
# PROVENANCE IS PART OF THE RESULT. Firecrawl and Exa publish MCP skills, so those two are the
# vendor's own bytes. Parallel publishes only a CLI skill — it forks context to a
# `parallel:parallel-subagent` and shells out to `parallel-cli`, none of which survives to MCP.
# Its entry is a PORT: vendor frontmatter and vendor guidance retargeted onto web_search /
# web_fetch, with the CLI-only options dropped because the MCP schema does not expose them
# (--after-date, --include-domains, --exclude-domains, --mode, --location). Everything the port
# keeps is the vendor's text, including the mandatory Sources section.
#
# So a skills-on comparison is two arms on vendor prose against one on prose we retargeted.
# That is stamped per record as skill_provenance and must be quoted with the number.
SKILL_BY_ARM = {
    "fc-mcp":       ("firecrawl-developer-index", "fc-mcp.md",      "verbatim"),
    "exa-mcp":      ("exa-search",                "exa-mcp.md",     "verbatim"),
    "parallel-mcp": ("parallel-web-search",       "parallel-mcp.md", "ported"),
}
SKILL_ON = os.environ.get("SKILL", "off").lower() in ("1", "on", "true", "yes")
SKILL_NAME, SKILL_SHA, SKILL_PROV = None, None, None



# The gh control searches repositories on the repo track and issues/PRs on the fix
# track — the artifact the track actually asks for. The agent may still override.
GH_DEFAULT_KIND = "repo" if TRACK == "repo" else "issue"            # gold is a github artifact on repo/fix
MAX_TURNS = int(os.environ.get("MAX_TURNS", 40))
# Unlimited by default; MAX_TURNS + TIMEOUT are the runaway guards. Effort is a
# REPORTED metric (searches/q, tool_calls/q), not a cap. MAX_SEARCH=4 restores it.
MAX_SEARCH = int(os.environ.get("MAX_SEARCH", 0)) or None
MAX_READ = int(os.environ.get("MAX_READ", 0)) or None
TIMEOUT = int(os.environ.get("TIMEOUT", 600))
RUNS = int(os.environ.get("RUNS", 1))          # independent passes; 3 for a reportable run
# 20,000, not 4,000. At 4,000 this was a FAIRNESS device and it only ever fired on one arm:
# measured across the full repo+fix runs it trimmed 68-86% of Exa's results and 0.0% of
# Firecrawl's, because fc's own MCP client already caps every passage at 1,200 chars. Capping
# Exa on the vendor's behalf is the same class of intervention we removed with the search budget.
# At 20,000 it is a SAFETY guard instead: Exa's median result is 5,929-7,950 chars and passes
# through untouched, while the pathological tail is still stopped — one Exa result measured
# 1,000,029 chars, and an 844,359-char payload previously blew MAX_MCP_OUTPUT_TOKENS and cost
# the whole call (the arm scored no_successful_search). this harness explicitly preserves
# this class of cap ("agent-loop context guards, full results available via save_messages");
# what it forbids is truncating the PERSISTED record, which we never do — full text is in _raw
# and every cut is flagged. Run AGENT_SNIPPET=0 for the uncapped sensitivity.
AGENT_SNIPPET = int(os.environ.get("AGENT_SNIPPET", 20000)) or None
# Saved transcript. 0 = UNLIMITED, and that is the default: the transcript is a persisted record.
# It used to be a hard-coded 20,000-char cut with no flag, so a 200k-char payload was silently 90%
# missing from the very
# artifact you would open to investigate that call. Set a positive value only if disk forces it;
# when it bites, the entry carries `truncated: <full length>`.
TRANSCRIPT_CHARS = int(os.environ.get("TRANSCRIPT_CHARS", 0)) or None
PAGE_FULL = int(os.environ.get("PAGE_FULL", 0)) or None
# Left unset, this silently gives the arm with the longest pages an
# advantage on groundedness. Default it to a real number here and stamp it on every record.
GROUND_CHARS = int(os.environ.get("GROUND_CHARS", 6000)) or None

MANIFEST = json.load(open(os.path.join(HERE, "tools_manifest.json"))) \
    if os.path.exists(os.path.join(HERE, "tools_manifest.json")) else {}


def norm(s):
    return re.sub(r"\s+", "", str(s).strip().lower())


def gt_sha():
    import hashlib
    return hashlib.sha256(open(GT_FILE, "rb").read()).hexdigest()[:16]


GT_SHA = gt_sha()


def load_items():
    rows = (json.loads(l) for l in open(GT_FILE) if l.strip())
    items = [r for r in rows if r["intent"] == INTENT]
    return items if N == "all" else items[: int(N)]


import prompts
SYSTEM = prompts.SYSTEM[TRACK]


# State and the three rules (gate/prepare/record) live in control.py — see its module docstring.
State = C.State


def _cfg():
    return C.Cfg(k=K, restrict=RESTRICT, max_search=MAX_SEARCH, max_read=MAX_READ,
                 agent_snippet=AGENT_SNIPPET, page_full=PAGE_FULL, manifest=MANIFEST)


# The live toolset for this run, resolved ONCE against the manifest. Everything downstream —
# the allowlist, the gate, the record, the printed header — reads these, so there is exactly one
# answer to "which tools does this arm have" and the two drivers cannot disagree.
SEARCH_TOOLS, TOOL_FALLBACK = ARMS_MOD.live_tools(ARM, TRACK, MANIFEST)
FETCH_TOOLS = ARM.fetch_for(TRACK)


def _build_jail():
    """A throwaway cwd holding ONLY this arm's skill — never the real project directory.

    MEASURED, not theoretical: running with cwd=<this dir> put the monorepo's CLAUDE.md
    ("Firecrawl is a web scraper API...") into the agent's context on EVERY arm, including
    Tier A arms that set no setting_sources at all. An eval comparing Firecrawl against Exa
    and Parallel must not begin by telling the agent it is working inside Firecrawl.

    With SKILL=off the jail is EMPTY: no skills, no settings, no project files. With SKILL=on
    it holds exactly one thing — this arm's vendor SKILL.md at the path the agent looks in —
    so the skill is DISCOVERED and invoked through the Skill tool rather than forced into the
    system prompt on every turn."""
    global SKILL_NAME, SKILL_SHA, SKILL_PROV
    jail = tempfile.mkdtemp(prefix=f"devdex-{ARM_NAME}-")
    ent = SKILL_BY_ARM.get(ARM_NAME)
    if not SKILL_ON:
        return jail
    if not ent:
        print(f"SKILL=on but {ARM_NAME} has no registered skill — running bare", flush=True)
        return jail
    name, fname, prov = ent
    src = os.path.join(_here, "skills", fname)
    if not os.path.exists(src):
        print(f"SKILL=on but {src} is missing — running bare", flush=True)
        return jail
    import hashlib, shutil
    dest = os.path.join(jail, ".claude", "skills", name)
    os.makedirs(dest, exist_ok=True)
    shutil.copyfile(src, os.path.join(dest, "SKILL.md"))
    SKILL_NAME, SKILL_PROV = name, prov
    SKILL_SHA = hashlib.sha256(open(src, "rb").read()).hexdigest()[:16]
    print(f"skill installed: {name} ({fname}, sha {SKILL_SHA}, provenance={prov})", flush=True)
    if prov != "verbatim":
        print(f"  NOTE: {ARM_NAME}'s skill is a PORT of the vendor's CLI skill, not vendor MCP "
              f"text. Quote skill_provenance alongside any skills-on number.", flush=True)
    return jail


JAIL = _build_jail()


# =====================================================================================
# HOOKS — the whole MCP-layer control system lives here.
# =====================================================================================
def make_hooks(ctl):
    """Two hooks, both thin: every decision is control.Controller's, so the SDK driver and the
    neutral driver enforce the exact same rules."""

    async def pre_tool(inp, tool_use_id, ctx):
        name = inp.get("tool_name", "")
        deny = ctl.gate(name)
        if deny:
            return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                    "permissionDecision": "deny", "permissionDecisionReason": deny}}
        if ctl.is_local(name) or name in ctl.free_tools:
            # FREE tools skip prepare() as well as the budget. They retrieve nothing, so there is
            # nothing to normalise — and running prepare() on them started a Clock entry that
            # post_tool never stops (it returns early for non-search tools) and left a `_pending`
            # record that is never popped. A stale pending entry is exactly what `_take`'s
            # popitem() fallback can mis-attribute to a later real search.
            return {}
        if name in ctl.fetch_tools:
            # A FETCH still needs its pending record (record_fetch reads the requested url and the
            # read options out of it) but must NOT be normalised like a search — see
            # Controller.prepare_fetch. The input is handed on unchanged, so no updatedInput.
            ctl.prepare_fetch(name, inp.get("tool_input") or {}, tool_use_id)
            return {}
        sent = ctl.prepare(name, inp.get("tool_input") or {}, tool_use_id)
        return {"hookSpecificOutput": {"hookEventName": "PreToolUse",
                "permissionDecision": "allow", "updatedInput": sent}}

    async def post_tool(inp, tool_use_id, ctx):
        name = inp.get("tool_name", "")
        if name in ctl.fetch_tools:
            # agent_pool, never engine_pool — the two-pool rule is what makes a vendor fetch safe.
            ctl.record_fetch(name, inp.get("tool_response"), tool_use_id)
            return {}
        if name not in ctl.search_tools:
            return {}
        trimmed = ctl.record(name, inp.get("tool_response"), tool_use_id)
        # ALWAYS rewrite, even when the harvest is empty. Returning {} let the vendor's RAW,
        # untrimmed payload through to the model — and it did so precisely in the error/unparsed
        # case, i.e. the largest payloads, after MAX_MCP_OUTPUT_TOKENS was raised to 200k. That
        # bypassed the shared AGENT_SNIPPET ceiling control.py exists to enforce.
        return {"hookSpecificOutput": {"hookEventName": "PostToolUse",
                "updatedMCPToolOutput": [{"type": "text", "text": json.dumps(trimmed, indent=1)}]}}

    return ([HookMatcher(matcher=None, hooks=[pre_tool])],
            [HookMatcher(matcher=None, hooks=[post_tool])])


# =====================================================================================
def make_options(state):
    """Wrap the SHARED local tools (control.py) as SDK tools. The bodies must not be
    re-implemented here — the reader in particular has to be byte-identical across drivers, or
    the arm with the friendlier reader wins on reading rather than on retrieval."""
    def _wrap(text_err):
        text, is_err = text_err
        return {"content": [{"type": "text", "text": text}], **({"is_error": True} if is_err else {})}

    @tool("submit_citations", C.LOCAL_TOOL_SPECS["submit_citations"]["description"], {"citations": list})
    async def submit_cit(args):
        return _wrap(C.submit_citations(state, args, CITE_K))

    @tool("submit_answer", C.LOCAL_TOOL_SPECS["submit_answer"]["description"],
          {"answer": str, "sources": list})
    async def submit_ans(args):
        return _wrap(C.submit_answer(state, args, CITE_K))

    @tool("read_target", C.LOCAL_TOOL_SPECS["read_target"]["description"], {"repo": str, "number": int})
    async def read_target(args):
        return _wrap(C.read_target(state, args))

    @tool("read_page", C.LOCAL_TOOL_SPECS["read_page"]["description"], {"url": str})
    async def read_page(args):
        return _wrap(C.read_page(state, args, PAGE_FULL))

    # The CONTROL arm's search tool. In-process, not Bash — no arm gets a shell. Its results go
    # through the same gate/prepare/record path as a vendor MCP response, so `gh` is harvested
    # into engine_pool and scored by identical code.
    @tool("gh_search", C.LOCAL_TOOL_SPECS["gh_search"]["description"], {"query": str, "type": str})
    async def gh_search(args):
        return _wrap(C.gh_search(state, args, K, ARM.fixed_params.get('search_type'), GH_DEFAULT_KIND))

    BASE_BLOCK = ["Bash", "Read", "Write", "Edit", "NotebookEdit", "Glob", "Grep",
                  "Task", "TodoWrite", "KillShell", "BashOutput", "ToolSearch"]
    submit_tool = submit_ans if TRACK == "docs" else submit_cit
    submit_name = "mcp__sub__submit_answer" if TRACK == "docs" else "mcp__sub__submit_citations"
    srv = {"sub": create_sdk_mcp_server("sub", tools=[submit_tool])}
    allowed = [submit_name]
    # AN ARM WITH ITS OWN FETCH TOOL USES ONLY THAT. Our `read_target` is not part of any
    # product — it is the GitHub API, and it returns a structured issue body plus every comment
    # (measured: 4,253 chars + 3 comments) that no vendor's reader matches. While both were
    # mounted the agent simply ignored the vendor's (3 fetch calls across 40 items) and the
    # "product tier" measured nothing. Reading is half the loop, so in a product comparison each
    # engine reads with its own tool or the comparison is only half a product.
    #
    # WHAT THIS COSTS, stated because it is real: reading quality now lands in the score, and the
    # readers are not close — same GitHub issue, firecrawl_scrape 23,430 chars, parallel 8,994,
    # exa 3,121, a 7.5x spread with PAGE_FULL unset. That is the product difference in a product
    # tier, and a confound in an index tier.
    #
    # WHAT IT DOES NOT COST: scoring integrity. record_fetch harvests to agent_pool, so nothing
    # fetched can ever be credited as a retrieval hit — the same rule that let a vendor fetch be
    # allowed at all. What IS lost is `blocked_reads`: read_target refuses targets the engine
    # never surfaced, and that refusal is our audit of the agent probing from memory. A vendor
    # fetch takes any url, so for these arms that signal reads 0 — absent, not clean.
    if ARM.fetch_for(TRACK):
        pass                                   # vendor fetch is this arm's reader; see above
    elif TRACK in ("repo", "fix"):
        srv["rd"] = create_sdk_mcp_server("rd", tools=[read_target])
        allowed.append("mcp__rd__read_target")
    else:
        srv["pg"] = create_sdk_mcp_server("pg", tools=[read_page])
        allowed.append("mcp__pg__read_page")

    # `Skill` is blocked for EVERY arm — no arm ships one, and without the block the agent
    # burns turns trying to invoke one (observed in the smoke run), so arms would differ in
    # turns spent rather than in retrieval.
    common = dict(model=MODEL, system_prompt=SYSTEM, max_turns=MAX_TURNS,
                  permission_mode="bypassPermissions", strict_mcp_config=True,
                  max_buffer_size=64 * 1024 * 1024, cwd=JAIL,
                  env={"CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
                       "MAX_MCP_OUTPUT_TOKENS": os.environ["MAX_MCP_OUTPUT_TOKENS"]},
                  disallowed_tools=BASE_BLOCK + ["WebSearch", "WebFetch"]
                                    + ([] if SKILL_NAME else ["Skill"]))

    if ARM.kind == "none":
        c = dict(common)
        c["system_prompt"] = SYSTEM + " You have NO search tools; answer from your own knowledge only."
        return ClaudeAgentOptions(mcp_servers={"sub": srv["sub"]}, allowed_tools=[submit_name], **c)

    if ARM.kind == "local":
        # gh-cli: our in-process search tool, mounted under the `gh` namespace so it is named
        # exactly like an MCP tool and the Controller treats it identically.
        srv["gh"] = create_sdk_mcp_server("gh", tools=[gh_search])
        ctl = C.Controller(state, _cfg(), SEARCH_TOOLS, ARM.scope_hint,
                           fixed_params=ARM.fixed_params, free_tools=ARM.free_tools)
        pre, post = make_hooks(ctl)
        return ClaudeAgentOptions(mcp_servers=srv, allowed_tools=[*SEARCH_TOOLS, *allowed],
                                  hooks={"PreToolUse": pre, "PostToolUse": post}, **common)

    if ARM.kind == "native":
        # THE SAME BUDGET AS EVERY OTHER ARM. The native tools bypass the Controller (there is no
        # MCP server to gate), so until this arm ran UNCAPPED while every MCP arm was
        # held to MAX_SEARCH. The readiness smoke measured the damage: websearch made up to 18
        # searches/question (mean 7.5) against everyone else's 4 — 4.5x the budget, which makes
        # any number it produces uncomparable. Earlier harnesses do not have this problem only
        # because they budget nobody; we introduced it by budgeting everybody except this arm.
        #
        # WebSearch spends the SEARCH budget, WebFetch spends the READ budget — the same two
        # counters, the same limits, so the arm is bounded exactly like the others.
        async def native_pre(inp, tool_use_id, ctx):
            nm = inp.get("tool_name", "")
            kind = {"WebSearch": "search", "WebFetch": "read"}.get(nm)
            if kind and not state.count(kind):
                with state._lock:
                    state.budget_denied += 1
                return {"hookSpecificOutput": {
                    "hookEventName": "PreToolUse", "permissionDecision": "deny",
                    "permissionDecisionReason": f"BUDGET EXHAUSTED ({kind}s). Submit your answer now."}}
            return {}
        c = dict(common)
        # Native arms never get a skill, so the Skill tool stays blocked for them.
        c["disallowed_tools"] = BASE_BLOCK + ["Skill"]
        return ClaudeAgentOptions(mcp_servers=srv,
                                  allowed_tools=["WebSearch", "WebFetch", *allowed],
                                  hooks={"PreToolUse": [HookMatcher(matcher=None,
                                                                    hooks=[native_pre])]}, **c)

    # ---- MCP arm ----
    search_tools = SEARCH_TOOLS
    opts = dict(common)
    ctl = C.Controller(state, _cfg(), search_tools, ARM.scope_hint,
                       fixed_params=ARM.fixed_params, free_tools=ARM.free_tools,
                       fetch_tools=FETCH_TOOLS)
    pre, post = make_hooks(ctl)
    # Note what is NOT in the allowlist: `Task`. A subagent's tool calls are not guaranteed to
    # reach PostToolUse, and a search that misses the hook never enters engine_pool — it would
    # read as a coverage miss. An earlier harness gave Exa subagents because it had no grounding
    # to lose; we cannot.
    return ClaudeAgentOptions(mcp_servers={**ARM.servers, **srv},
                              allowed_tools=[*search_tools, *FETCH_TOOLS,
                                             *ARM.free_tools, *allowed,
                                             # Skill only where one is installed; an arm with
                                             # no vendor skill must not even see the tool.
                                             *(["Skill"] if SKILL_NAME else [])],
                              hooks={"PreToolUse": pre, "PostToolUse": post}, **opts)


# ---------------------------------------------------------------------------------------------
# WALL-CLOCK REAPER — the hard backstop behind `anyio.move_on_after(TIMEOUT)` in run_one().
#
# move_on_after is COOPERATIVE: it can only fire at an await checkpoint. When the Agent SDK's
# bundled `claude` subprocess wedges on a read, control never returns to the event loop, the
# cancel scope never gets to fire, `scope.cancelled_caught` stays False, and the item runs
# forever while holding its concurrency slot.
#
# MEASURED, not theoretical 58 items blew a 600s TIMEOUT, 31 ran over an hour,
# worst 7.1h. Because wedged slots never free, effective concurrency decayed to ~0 and the cells
# fell from 220 items/h to 2.2 items/h. The wedged items eventually died with the SDK's malformed
# `Claude Code returned an error result: success` envelope — that error and the 7h latencies are
# one bug, not two.
#
# Why a global reaper and not per-item PID tracking: under concurrency we cannot attribute a
# freshly spawned child to the item that spawned it without a race. We do not need to. Any
# bundled-claude descendant older than TIMEOUT + GRACE is wedged BY DEFINITION, because no
# legitimate item may exceed TIMEOUT. SIGKILL closes the pipe the SDK is blocked on, the async
# iterator raises, run_one's `except Exception` catches it, and the slot frees.
#
# SAFETY: two independent guards. (1) only descendants of THIS process are considered, walked
# from os.getpid(); (2) the command must contain `claude_agent_sdk/_bundled/` — the SDK's own
# vendored binary. The user's Claude.app and any other `claude` on the box cannot match either
# test, let alone both.
REAP_GRACE = int(os.environ.get("REAP_GRACE", 120))
REAP_EVERY = int(os.environ.get("REAP_EVERY", 30))
SDK_MARK = "claude_agent_sdk/_bundled/"
REAPED = []


def _etime_to_s(t):
    """macOS ps has no `etimes`; parse `etime` == [[DD-]HH:]MM:SS."""
    try:
        days, _, rest = t.strip().rpartition("-")
        parts = [int(x) for x in rest.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        h, m, sec = parts[-3:]
        return (int(days) if days else 0) * 86400 + h * 3600 + m * 60 + sec
    except Exception:
        return 0


def _reap_once():
    """SIGKILL every wedged bundled-claude descendant of this process. Returns how many."""
    try:
        out = subprocess.run(["ps", "-axo", "pid=,ppid=,etime=,command="],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return 0
    rows, kids = {}, {}
    for ln in out.splitlines():
        f = ln.split(None, 3)
        if len(f) < 4:
            continue
        try:
            pid, ppid = int(f[0]), int(f[1])
        except ValueError:
            continue
        rows[pid] = (_etime_to_s(f[2]), f[3])
        kids.setdefault(ppid, []).append(pid)
    # walk only OUR subtree
    mine, stack = [], [os.getpid()]
    seen = set()
    while stack:
        for pid in kids.get(stack.pop(), []):
            if pid in seen:
                continue
            seen.add(pid)
            mine.append(pid)
            stack.append(pid)
    n = 0
    for pid in mine:
        age, cmd = rows.get(pid, (0, ""))
        if SDK_MARK not in cmd or age <= TIMEOUT + REAP_GRACE:
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except Exception:
            continue
        REAPED.append({"pid": pid, "age_s": age})
        n += 1
        print(f"  REAPED wedged sdk subprocess pid={pid} age={age}s "
              f"(> TIMEOUT {TIMEOUT}s + grace {REAP_GRACE}s) — freeing its slot", flush=True)
    return n


def _start_reaper():
    def loop():
        while True:
            time.sleep(REAP_EVERY)
            try:
                _reap_once()
            except Exception:
                pass
    threading.Thread(target=loop, daemon=True, name="reaper").start()


async def run_one(x):
    state = State()
    opts = make_options(state)
    final, err, cost = "", None, 0.0
    # tool_use_id -> tool name, so a ToolResultBlock (which carries no name of its own) can be
    # attributed to the call that produced it. The native harvest below needs that to tell a
    # SEARCH result from a page the agent fetched.
    called = {}
    with anyio.move_on_after(TIMEOUT) as scope:
        try:
            async for msg in query(prompt=f"Task:\n{x['query']}", options=opts):
                for b in (msg.content if isinstance(getattr(msg, "content", None), list) else []):
                    bt = type(b).__name__
                    if bt == "TextBlock":
                        state.transcript.append({"t": "text", "v": b.text})
                    elif bt == "ToolUseBlock":
                        state.transcript.append({"t": "call", "name": b.name, "input": b.input})
                        called[getattr(b, "id", None)] = (b.name, b.input)
                    elif bt == "ToolResultBlock":
                        _v = str(getattr(b, "content", ""))
                        state.transcript.append(
                            {"t": "result", "v": _v[:TRANSCRIPT_CHARS] if TRANSCRIPT_CHARS else _v,
                             **({"truncated": len(_v)} if TRANSCRIPT_CHARS and len(_v) > TRANSCRIPT_CHARS else {})})
                if isinstance(msg, AssistantMessage):
                    for b in msg.content:
                        if isinstance(b, TextBlock):
                            final = b.text
                        elif ToolUseBlock and isinstance(b, ToolUseBlock) and b.name in ("WebSearch", "WebFetch"):
                            # Every native call the model EMITTED, including ones native_pre then
                            # denied on budget. Reported as `ext_tool_calls` only — native_pre has
                            # already counted the allowed ones as searches/reads, so adding this
                            # into `tool_calls` counted the same call twice and published a native
                            # effort number about 2x everyone else's.
                            state.ws += 1
                elif isinstance(msg, ResultMessage):
                    cost = msg.total_cost_usd or 0.0
                if ARM.kind == "native":
                    # NATIVE CALLS ARE OBSERVABLE, JUST NOT THROUGH MCP. WebSearch runs
                    # server-side, so no PostToolUse fires and record() never sees it — but the
                    # call and its verbatim result are both in this stream. Reconstructing a
                    # search_log entry here is what stops the arm being structurally unmeasurable:
                    # without it n, ranks and pool_hit are all empty, the depth cap cannot apply,
                    # and every engine-side column reads 0.000 for an arm that answered fine.
                    #
                    # This block ran only on repo/fix, so on DOCS the native arm harvested nothing
                    # at all: empty pool, and therefore pool_hit / groundedness / precision
                    # structurally 0.000 however well it answered. Docs matches on URLs, so URLs
                    # are harvested too, not just github refs.
                    #
                    # Harvest ONLY from the verbatim tool result, never from the model's own
                    # prose — crediting prose would let a memorised guess count as retrieved.
                    # And only a WebSearch result is ENGINE output. WebFetch is this arm's reader,
                    # so every ref on a page it pulled back is reader-origin and belongs in
                    # agent_pool (State's two-pool rule: what the agent read can never be
                    # credited). A result whose call we never saw announced is treated as a read:
                    # agent_pool cannot score, so that is the safe direction to be wrong in.
                    for b in (msg.content if isinstance(getattr(msg, "content", None), list) else []):
                        if type(b).__name__ != "ToolResultBlock":
                            continue
                        nm, args = called.get(getattr(b, "tool_use_id", None), (None, {}))
                        is_search = (nm == "WebSearch")
                        src = "search_direct" if is_search else "reader"
                        s_ = str(getattr(b, "content", "") or "")
                        for m in P.GH_ARTIFACT.finditer(s_):
                            state.add(f"{m.group(1)}#{m.group(2)}", src)
                        for m in P.GH_REPO.finditer(s_):
                            if m.group(1).count("/") == 1:
                                state.add(m.group(1), src)
                        items, seen_u = [], set()
                        for m in P.URL_RE.finditer(s_):
                            u = m.group(0).rstrip(".,);")
                            if u in seen_u:
                                continue
                            seen_u.add(u)
                            state.add(u, src)
                            items.append({"rank": len(items), "url": u,
                                          "ref": P.ref_from_url(u), "type": ""})
                            if len(items) >= K:      # same depth the other arms are held to
                                break
                        if nm not in ("WebSearch", "WebFetch"):
                            continue
                        if is_search:
                            state.search_log.append({
                                "q": (args or {}).get("query"),
                                # server-side: the SDK reports no per-call timing, and inventing
                                # one would put a fabricated number in the latency column
                                "latency": None, "n": len(items), "results": items,
                                "chars": len(s_), "tool": "WebSearch",
                                # RANKS ARE APPEARANCE ORDER, not the engine's declared ranking —
                                # the stream carries no rank field. Good enough to apply the depth
                                # cap consistently, not good enough to quote as a ranking metric.
                                "meta": {"native": True, "rank_source": "appearance"}})
                        else:
                            state.fetch_log.append({
                                "tool": "WebFetch", "latency": None, "n": len(items),
                                "chars": len(s_), "url": (args or {}).get("url", ""),
                                "meta": {"native": True}})
        except Exception as e:
            err = str(e)[:200]
    if scope.cancelled_caught:
        err = f"timeout>{TIMEOUT}s"
    # RECONCILE. gate() counts a search the moment it is allowed; record() logs it when the
    # result comes back. The SDK does NOT fire PostToolUse when a tool call ERRORS, so a failed
    # call leaves those two out of sync and the failure becomes invisible: `reliability` sees
    # nothing and the arm just looks like an engine that returned nothing. MEASURED on Exa
    # returning 402 "exceeded your credits limit" — 3 counted searches, 0 log entries, and a
    # 0.000 that would have read as a product result. Any unrecorded call is logged as an error.
    #
    # MCP ARMS ONLY. A native arm's searches never pass through record() at all — WebSearch
    # results are harvested from the message stream, not from a tool response — so its
    # search_log is legitimately empty. It only has a `searches` count because the budget hook
    # added for that arm calls state.count(). Reconciling it therefore invented one failure per
    # search: measured on this smoke, 28 phantom "no PostToolUse — tool call failed" entries for
    # an arm that made 32 successful native searches and answered every item. That drove
    # `reliability` to 0.000 and left `search_p90` empty — an arm in perfect health reading as
    # total infrastructure failure. The guard below is what makes the reconcile mean what its
    # comment says: it repairs MCP calls that errored, not calls that were never MCP calls.
    for _ in range(max(0, state.searches - len(state.search_log)) if ARM.kind in ("mcp", "local") else 0):
        detail = next((t["v"][:300] for t in reversed(state.transcript)
                       if t.get("t") == "result" and P.error_envelope(t.get("v"))), None)
        state.search_log.append({"q": None, "latency": None, "n": 0, "results": [], "chars": 0,
                                 "tool": (SEARCH_TOOLS or [None])[0],
                                 "error": detail or "no PostToolUse — tool call failed"})
    # Integrity guard: an empty SDK stream is indistinguishable from
    # an abstain in the record, so it must be named rather than scored as a miss.
    if not err and not cost and state.searches == 0 and state.reads == 0 and state.ws == 0:
        err = "empty_stream(no cost, no tool calls) — infra failure, NOT an abstain"
    # MCP-specific integrity: the arm mounted a server but never produced a single search
    # call. Usually a renamed vendor tool (allowlist misses -> every call denied). Scoring
    # that as 0.000 would publish a plumbing bug as a product result.
    # Not "no log entries" — "no SUCCESSFUL call". Since the reconcile above logs failed calls,
    # an arm whose every search errored now has a full search_log and would otherwise sail
    # through as a legitimate 0.000.
    ok_calls = [s for s in state.search_log if not s.get("error")]
    if ARM.kind == "mcp" and not ok_calls and not err:
        why = (state.search_log[0].get("error") if state.search_log
               else f"denied={state.denied[:3]}")
        err = f"no_successful_search ({str(why)[:120]}) — check credits / preflight"

    base = {"qid": x["id"], "arm": ARM_NAME, "track": TRACK, "repo": x["container"]["ref"],
            "gt_file": os.path.basename(GT_FILE), "gt_sha": GT_SHA, "model": MODEL,
            "harness": "mcp", "arm_kind": ARM.kind, "arm_tools": SEARCH_TOOLS,
            "arm_note": ARM.note,
            "arm_role": ARM.role,
            # Non-null only when the primary vendor tool vanished and the declared
            # stand-in ran in its place. Readable, not quotable — see arms.Arm.resolve.
            "tool_fallback": TOOL_FALLBACK,
            "arm_fetch_tools": FETCH_TOOLS, "fetch_log": state.fetch_log,
            # PROVENANCE IS TRACK-INDEPENDENT. `pool_prov` used to be written only in the
            # repo/fix branch, so docs records carried none -- and a server-side arm, which
            # writes no search_log either, then had NOTHING to score against: its docs
            # pool_hit, groundedness and precision were structurally 0.000 however well it
            # answered. The harness collects this pool on every track; persist it on every
            # track.
            "pool": [q[0] for q in state.pool], "pool_prov": state.pool,
            "search_log": state.search_log, "searches": state.searches, "reads": state.reads,
            # EACH CALL COUNTED ONCE. `ws` is the native arm's own tally of emitted WebSearch /
            # WebFetch blocks and those SAME calls are already in searches/reads (native_pre
            # counts them); it is 0 for every other arm, since WebSearch/WebFetch are disallowed
            # there. So summing all three inflated exactly one arm's effort metric.
            "ext_tool_calls": state.ws, "tool_calls": state.searches + state.reads,
            "cost": cost, "error": err, "latency": None,
            # FULL lists. These are the audit trail for the two gates (the reader gate and the
            # deny-by-default tool gate); a truncated audit trail understates how often a gate
            # bound, and `denied_tools` had no count beside it, so the cut was invisible.
            "blocked_reads": len(state.blocked), "blocked_targets": state.blocked,
            "denied_tools": state.denied, "n_denied": len(state.denied),
            "free_calls": state.free_calls,
            "budget_denied": state.budget_denied,
            "_raw": state.raw, "_transcript": state.transcript}
    if TRACK == "docs":
        ans = state.answer if state.answer is not None else final
        cited = [{"url": u, "content": state.url_content.get(u, "")} for u in (state.sources or [])]
        # `cited_content` is the CITED pages only, and is the persisted evidence behind the
        # answer. Docs is scored on the cited URLs, so the whole read corpus (every page the
        # agent fetched, cited or not) is deliberately not written into the record.
        base.update({"expected": x["expected_answer"], "answer": ans or "", "sources": state.sources,
                     "source_kind": ("github" if "github.com" in (x.get("acceptable_sources") or [{}])[0].get("url", "")
                                     else "non-github"),
                     "cited_content": [c["content"] for c in cited],
                     "committed": bool((ans or "").strip()) and "i don't know" not in (ans or "").lower(),
                     # A committed answer that cites NOTHING is a compliance failure, not an
                     # ungrounded retrieval. Kept distinct so `groundedness` stops absorbing it.
                     "uncited": bool(getattr(state, "uncited", False)),
                     "answer_rejected": bool(getattr(state, "answer_rejected", False))})
    else:
        art = x["artifact"]
        gold_all = [a["ref"] for a in art]
        # PROSE IS NOT A SUBMISSION. This used to fall back to regexing owner/repo out of the
        # model's last message when it never called submit_citations. The analyzer then counted
        # the item as a precision HIT *and* as an abstain — outcome_sum came out at 2.000, its own
        # stated invariant being 1.000 — and whole_product.hit() credited an answer the agent
        # never committed to. The task is "commit one ranked answer"; not committing is a miss.
        cited = state.submission if state.submission is not None else []
        base.update({"tier": x.get("tier", "?"), "language": x.get("language"),
                     "gold": [gold_all[0]], "gold_all": gold_all, "gold_id": art[0].get("repo_id"),
                     # Every fix item ships a canonical PR *and* an `acceptable` issue (role in the
                     # GT, confidence 0.8) — and SYSTEM_FIX asks the agent for "the issue AND/OR
                     # the pull request". Scoring only the PR contradicted the prompt we gave and
                     # penalised the artifact an index is likelier to surface. analyze.golds()
                     # already reads gold_accept; nothing was ever writing it.
                     "gold_accept": [a["ref"] for a in art
                                     if a.get("role") in ("canonical", "acceptable")],
                     "citations": cited or [], "committed": state.submission is not None})
    return base


# ---------------------------------------------------------------------------------
# ITEM-LEVEL CHECKPOINTING. The harness used to write its JSON only at end-of-pass, so
# interrupting a 418-item Opus cell (~$190, several hours) lost every item. That happened:
# ~$120 of completed work was thrown away because six cells were killed mid-flight and the
# records only ever existed in memory. Now every CHUNK completions are flushed to the same
# output path, and a restart loads what is already there and skips those qids.
# ---------------------------------------------------------------------------------
CHUNK = int(os.environ.get("CHUNK", 25))


# An item that never produced a measurement is RETRIED on resume rather than inherited. It is
# the difference between "this engine did not find it" and "we never asked": a rate-limited
# vendor books a whole day of items as dead runs, and dead runs score 0 and stay in the
# denominator — so inheriting them means the cap, not the index, sets that arm's number, and no
# amount of re-running ever repairs it. This is what made a per-day-capped arm (Mintlify caps at
# 1,000 requests/IP; a three-track pass is ~1,400) unrunnable as a normal sweep cell.
# A genuine miss or abstention has no error and is always kept. RESUME_KEEP_DEAD=1 restores the
# old inherit-everything behaviour.
# MUST STAY A SUPERSET OF THE SCORER'S DEAD_ERRORS. The two lists answer the same question --
# "is this row a measurement or an absence?" -- and when they disagree the gap is silent and
# permanent: the scorer counted "returned an error result" against the 10% dead bar while resume
# did not recognise it, so those rows were inherited as good and never re-run. On one defective
# cell 13 items sat frozen in that state across five retry windows, penalising the arm for runs it
# was never allowed to retry. Anything the scorer calls dead has to be retryable here.
_DEAD_ON_RESUME = ("empty_stream", "timeout", "no_successful_search", "no_tool_calls",
                   "429", "rate limit", "too many requests", "unparsed_response",
                   "returned an error result")


def _is_dead(row):
    err = str(row.get("error") or "").lower()
    return bool(err) and any(d in err for d in _DEAD_ON_RESUME)


def _resume(pass_tag, track, arm, items):
    """Load a partial cell if one exists and return (done_rows, remaining_items)."""
    path = _out(f"{pass_tag}_{track}_{arm}.json")
    if not os.path.exists(path):
        return [], items
    try:
        done = json.load(open(path))
    except Exception:
        return [], items
    if os.environ.get("RESUME_KEEP_DEAD") == "1":
        keep, redo = done, set()
    else:
        keep = [r for r in done if not _is_dead(r)]
        redo = {r.get("qid") for r in done if _is_dead(r)}
    seen = {r.get("qid") for r in keep}
    todo = [x for x in items if x["id"] not in seen]
    if done:
        extra = f", retrying {len(redo)} dead" if redo else ""
        print(f"resume: {len(keep)} already done, {len(todo)} remaining{extra}", flush=True)
    return keep, todo


_CFG_SNAPSHOT = {}       # filled by _one_pass before any worker starts


def _flush(rows, pass_tag, track, arm):
    """Atomic-ish write so a kill mid-flush cannot corrupt the file."""
    # Stamp config on every row AS WE FLUSH. It used to be applied only at end-of-pass, so an
    # interrupted cell left rows with no `config` — not self-describing, and enough to KeyError
    # a consumer. 8 cells were in that state when this was found.
    for r in rows:
        if "config" not in r and _CFG_SNAPSHOT:
            r["config"] = _CFG_SNAPSHOT
    path = _out(f"{pass_tag}_{track}_{arm}.json")
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(rows, f, indent=1)
    os.replace(tmp, path)


def _one_pass(items, pass_tag, pass_no):
    """One INDEPENDENT pass over every item. Fresh State per question, fresh agent per
    question — nothing is carried between passes except the frozen GT."""

    cfg = {"max_search": MAX_SEARCH, "max_read": MAX_READ, "max_turns": MAX_TURNS,
           "agent_snippet": AGENT_SNIPPET, "page_full": PAGE_FULL, "ground_chars": GROUND_CHARS,
           "k": K, "cite_k": CITE_K, "timeout": TIMEOUT, "reap_grace": REAP_GRACE,
           "reaped": len(REAPED), "restrict_github": RESTRICT, "model": MODEL,
           "harness": "mcp", "arm_tools": SEARCH_TOOLS,

           "manifest": bool(MANIFEST), "no_depth": ARM.no_depth, "no_scope": ARM.no_scope, "runs": RUNS, "pass": pass_no,
           "skill": SKILL_NAME, "skill_sha": SKILL_SHA, "skill_provenance": SKILL_PROV}
    _CFG_SNAPSHOT.update(cfg)

    rows, items = _resume(pass_tag, TRACK, ARM_NAME, items)
    if not items:
        print("nothing to do — cell already complete", flush=True)
        return rows
    sem = anyio.Semaphore(CONCURRENCY)
    # Start the wall-clock backstop BEFORE any worker runs. See _reap_once(): move_on_after
    # cannot cancel out of a wedged SDK subprocess, so without this a single hung item holds its
    # semaphore slot forever and the cell's effective concurrency decays to zero.
    _start_reaper()

    async def worker(x):
        async with sem:
            t0 = time.time()
            n_reaped_before = len(REAPED)
            r = await run_one(x)
            r["latency"] = round(time.time() - t0, 2)
            # A reap during this item means the SDK subprocess was killed under us: the error we
            # captured is the CONSEQUENCE of the kill, not a product signal. Flag it so the
            # analyzer's DEAD_ERRORS path and any hand audit can tell the two apart.
            if len(REAPED) > n_reaped_before and r["latency"] > TIMEOUT:
                r["reaped"] = True
            r["pass"] = pass_no
            rows.append(r)
            if len(rows) % CHUNK == 0:
                _flush(rows, pass_tag, TRACK, ARM_NAME)
            tag = ("committed" if TRACK == "docs" else
                   ("HIT" if any(norm(g) in {norm(c) for c in r["citations"]} for g in r["gold"]) else "miss"))
            print(f"p{pass_no} {len(rows):>3}/{len(items)} [{tag:>9}] ${r['cost']:.2f} "
                  f"calls={r['tool_calls']} "
                  f"{'ERR:' + r['error'][:40] if r['error'] else x['id'][:44]}", flush=True)

    async def go():
        async with anyio.create_task_group() as tg:
            for x in items:
                tg.start_soon(worker, x)

    anyio.run(go)

    cfg = {"max_search": MAX_SEARCH, "max_read": MAX_READ, "max_turns": MAX_TURNS,
           "agent_snippet": AGENT_SNIPPET, "page_full": PAGE_FULL, "ground_chars": GROUND_CHARS,
           "k": K, "cite_k": CITE_K, "timeout": TIMEOUT, "reap_grace": REAP_GRACE,
           "reaped": len(REAPED), "restrict_github": RESTRICT, "model": MODEL,
           "harness": "mcp", "arm_tools": SEARCH_TOOLS,

           "manifest": bool(MANIFEST), "no_depth": ARM.no_depth, "no_scope": ARM.no_scope, "runs": RUNS, "pass": pass_no,
           "skill": SKILL_NAME, "skill_sha": SKILL_SHA, "skill_provenance": SKILL_PROV}
    for r in rows:
        r["config"] = cfg
    import gzip
    rawp = _out(f"{pass_tag}_{TRACK}_{ARM_NAME}.raw.jsonl.gz")
    with gzip.open(rawp, "wt", encoding="utf-8") as g:
        for r in rows:
            g.write(json.dumps({"qid": r["qid"], "arm": ARM_NAME, "track": TRACK, "config": cfg,
                                "searches": r.pop("_raw", []),
                                "transcript": r.pop("_transcript", [])}) + "\n")
    json.dump(rows, open(_out(f"{pass_tag}_{TRACK}_{ARM_NAME}.json"), "w"), indent=1)
    print(f"p{pass_no} saved {len(rows)} -> {pass_tag}_{TRACK}_{ARM_NAME}.json + raw {rawp} "
          f"({os.path.getsize(rawp)/1e6:.1f} MB)", flush=True)
    return rows


def main():
    assert os.environ.get("ANTHROPIC_API_KEY") or ARM.kind == "none"
    if not ARM.runs(TRACK):
        # An arm whose corpus does not cover this track produces an EMPTY CELL, not a zero.
        # Context7 has no repos/issues/PRs; a 0.000 there would read as bad retrieval.
        print(f"{ARM_NAME} does not run the {TRACK} track (corpus does not cover it) — "
              f"skipping, by design. See `tracks` in arms.py.", flush=True)
        return
    if ARM.kind == "mcp" and not MANIFEST:
        print("WARNING: no tools_manifest.json — run `python3 preflight.py` first. "
              "Without it k/scope normalisation is skipped and arms are NOT comparable.", flush=True)
    items = load_items()
    print(f"mcp-harness — track={TRACK} arm={ARM_NAME} kind={ARM.kind} "
          f"tool={[t.split('__')[-1] for t in SEARCH_TOOLS]} n={len(items)} runs={RUNS} "
          f"gt={os.path.basename(GT_FILE)}", flush=True)
    # RUNS independent passes. Vendors are non-deterministic (observed: Parallel returning a
    # different result set for two identical calls), and so is the agent. One pass cannot
    # distinguish a real gap between engines from run-to-run noise, so a single-pass number is
    # not quotable. Passes are written to SEPARATE tags so scorer/report_metrics.py scores each
    # one unchanged; run_eval.py then reports mean/sd/range across the passes.
    for i in range(1, RUNS + 1):
        pass_tag = TAG if RUNS == 1 else f"{TAG}p{i}"
        _one_pass(items, pass_tag, i)
    # A bare invocation writes flat files; the scorer reads the runs/<timestamp>-<label>/run<N>/
    # layout that devdex/run_eval.py creates, and it is also what reports cross-pass variance.
    print("\nscore with:  devdex-report --pass p1"
          "\n(for a reportable number run the cell through devdex/run_eval.py --n-runs 3, which "
          "writes that layout and aggregates the passes)")


if __name__ == "__main__":
    main()

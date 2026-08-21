"""report_metrics.py — the published table, regenerated from persisted run records.

One definition per metric, applied identically to every arm. This module owns every gold-matching
and per-item scoring rule; benchmark/run_benchmark.py imports score_records() from here rather than
keeping a second copy, so a submitter's numbers cannot drift from ours. The root README's headline
recall@10 / MRR@10 table -- with bootstrap 95% CIs and the equal-weight combined -- is produced by
run_benchmark.py ON TOP of this scorer; that is aggregation, not a second rule set. If a published
number disagrees with the definition here, the definition here is authoritative.

LOCKED SETTINGS. Three choices move every number, so they are constants rather than flags:

  CAP = 10        Depth cap. Arms return different result counts (fc-web returns ~19, everyone
                  else <=10). Scoring them as delivered rewards an arm for returning more, not for
                  ranking better, so every arm is truncated to 10.
  LENIENT         On the two-wide fix gold (issue + PR), citing EITHER counts. Strict scoring
                  (both required) is a different, harder question and is not what we report.
  FULL DENOM      A dead run scores 0 and stays in the denominator. The question is "if I send
                  this tool N tasks, how many come back resolved", not "of the ones that
                  worked".

FIX TRACK = SHORT QUERIES. Only one fix dataset exists (fix_short_v1.0.0). The diagnosed-phrasing
set it was derived from is not shipped -- those queries were maintainer post-hoc summaries, 47%
stating the bug's cause, so they measured lexical lookup rather than retrieval. Cells written by
`--track fix` carry the token `fix`; older sweeps labelled them `fixshort`. Both name the same
track and both are read; see run_files().

WHAT EACH METRIC MEANS
  precision      committed AND the agent's FIRST citation is gold AND THAT artifact is one the
                 engine returned. The end-to-end "did this resolve the task" number. The join is
                 on the cited artifact, not on "any gold for this item": on the two-wide fix gold
                 the looser reading credited a memorised citation of the issue because the engine
                 surfaced the PR. On docs the several canonical sources are alternate locations of
                 ONE answer, so any-of-them is the correct join there.
  recall         fraction of the item's golds that appear anywhere in the agent's top 10.
  correctness    a gold appears anywhere in the agent's list. Ignores where it came from, so
                 it includes answers recalled from model memory -- read it against the no-tool
                 floor, never on its own.
  groundedness   a cited gold was also returned by the engine this arm is allowed to call.
                 correctness minus groundedness is memory plus reads plus harvest gap.
  pool_hit       the engine returned a gold at all, whatever the agent then did with it. The
                 retrieval ceiling: precision can never exceed it.
  eng_recall     ENGINE-side recall@10: fraction of the item's golds inside the engine's first
                 10 slots. `recall` scores the AGENT's list; this scores the retrieval. On the
                 two-wide fix gold the two diverge -- pool_hit says "it found something",
                 eng_recall says "how much of the answer set it found".
  aMRR           1/rank of the first gold in the AGENT's submitted list.
  from_memory    correct, but the gold entered NO pool: the model recalled it. Read this, not
                 correctness-minus-groundedness, as parametric recall.
  from_read      correct, and the gold arrived only via a page the agent FETCHED. The engine
                 never listed it; the reader rescued the item.
  below_cap      correct, and the engine DID return the gold -- past rank 10, so the depth cap
                 excluded it. A ranking failure, not memory.
                 groundedness + from_memory + from_read + below_cap == correctness, asserted.

Every metric is deterministic: goldness is reference matching on repo/fix and canonical-URL
matching on docs. No LLM judge is involved in producing any published number.

INVARIANTS, asserted per cell. A violation means a definition is wrong, not that an arm did
badly, so it is surfaced instead of printed:
    precision + wrong + abstain == 1
    precision <= groundedness <= correctness
    recall <= correctness

Usage:
    python3 devdex/scorer/report_metrics.py                    # pass 1, devdex/runs/
    python3 devdex/scorer/report_metrics.py --pass p2
    python3 devdex/scorer/report_metrics.py --runs devdex/runs/<timestamp>-p1
"""
import argparse
import glob
import json
import os
import re
import statistics as st
import sys
from pathlib import Path

DEVDEX_ROOT = Path(__file__).resolve().parents[1]      # devdex/
ROOT = DEVDEX_ROOT.parent                              # repo root
sys.path.insert(0, str(DEVDEX_ROOT.parent))

CAP = 10

# A run that never reached the engine is an INFRASTRUCTURE failure, not a retrieval result. It
# still scores 0 and stays in the denominator (that is the "full denominator" rule), but a cell
# with many of them is not a measurement of the engine at all and must not be ranked.
#
# The dead-run check lived only in suite.py, so the published table could not tell a clean cell
# from a broken one. A cell that was 133/400 dead (33%) printed beside the others with no marker,
# showing a plausible-looking recall of 0.398.
DEAD_ERRORS = ("empty_stream", "timeout", "no_successful_search", "no_tool_calls",
               "returned an error result")
DEAD_LIMIT = 0.10

# track key -> (run-directory token, scoring family)
TRACKS = {"repo": ("repo", "repo"), "fix-short": ("fixshort", "fix"), "docs": ("docs", "docs")}
ARMS = ["fc-mcp", "fc-web", "exa-mcp", "parallel-mcp", "mintlify", "context7",
        "gh-hybrid", "websearch", "no-tool"]
CONTROLS = {"gh-hybrid", "websearch", "no-tool"}

# CANONICAL NAMES. The short keys above are identifiers, not names: they are baked into every
# run directory ever produced (`runs/<ts>-p1-opus-repo-fc-mcp/`), into --arm and --dataset, and
# into the labels of runs currently in flight. Renaming them would orphan the whole corpus, so
# the product name lives here as a presentation layer and the identifier stays stable.
#
# Use these in anything a reader sees -- reports, slides, PR bodies. Use the key in code, paths
# and CLI flags. Do not invent a third spelling.
ARM_NAMES = {
    "fc-mcp": "Firecrawl Developer Index",       # firecrawl_developer_search
    "fc-web": "Firecrawl Search (no category)",   # firecrawl_search, categories omitted
    "exa-mcp": "Exa",
    "parallel-mcp": "Parallel",
    "mintlify": "Mintlify",
    "context7": "Context7",
    "gh-hybrid": "GitHub CLI",                    # control: lexical keyword search
    "websearch": "Native web search",             # control: model's built-in tool
    "no-tool": "No tools",                        # control: parametric memory floor
}
TRACK_NAMES = {
    "repo": "Repository discovery",
    # The fix track has two query forms over identical items and golds. "Symptom" is what the
    # developer can see before diagnosing; "diagnosed" is the maintainer's post-hoc summary.
    # We report symptom. Never write "fix short" in prose -- it names a file, not a benchmark.
    "fix-short": "Defect resolution (symptom queries)",
    "fix": "Defect resolution (diagnosed queries)",
    "docs": "Documentation QA",
}


def arm_name(key):
    return ARM_NAMES.get(key, key)


def track_name(key):
    return TRACK_NAMES.get(key, key)

_CITE_ID = re.compile(r"['\"](?:id|ref|url)['\"]\s*:\s*['\"]([^'\"]+)")


# ------------------------------------------------------------------ record helpers
def norm(s):
    return str(s).strip().lower().rstrip("/")


def cite_ref(c):
    """Agents sometimes submit {'id': .., 'reason': ..} instead of a bare ref."""
    m = _CITE_ID.search(str(c))
    return norm(m.group(1)) if m else norm(c)


def golds(record):
    return {norm(g) for g in (record.get("gold_accept") or record.get("gold") or [])}


def submitted(record, family):
    """The agent's final ordered list."""
    if family == "docs":
        return [norm(u) for u in (record.get("sources") or [])]
    return [cite_ref(c) for c in (record.get("citations") or [])]


def engine_slots(record, cap=CAP):
    """Refs the ENGINE returned, first `cap` ranks of every search.

    FALLBACK for native arms. Claude's WebSearch runs server-side and never reaches the MCP
    PostToolUse hook, so `websearch` writes no search_log; its refs are harvested from the model
    stream into pool_prov. Reading only search_log scored that arm 0.000 groundedness, which is
    a harness artifact rather than a result.

    The fallback is NOT like-for-like: stream-harvested refs have no rank, so `cap` cannot be
    applied and engine-side rank metrics are undefined. Read native arms as a reference line.
    """
    out = set()
    for search in (record.get("search_log") or []):
        for x in (search.get("results") or []):
            if (x.get("rank") or 0) >= cap:
                continue
            for v in (x.get("ref"), x.get("url")):
                if v:
                    out.add(norm(v))
    if not out and not (record.get("search_log") or []):
        out = {norm(p[0]) for p in (record.get("pool_prov") or [])
               if len(p) > 1 and p[1] != "reader"}
    return out


# ------------------------------------------------------------------ scoring
def score_records(rows, family, is_gold, pool_test=None, cap=CAP):
    """Aggregate metrics over one cell's records.

    `is_gold(item, record) -> bool` is injected because the three tracks decide goldness
    differently: repo/fix compare against the record's own gold_accept, docs resolves a URL
    against a canonical-source map. Keeping it a parameter is what lets the tests exercise the
    arithmetic on synthetic records with no dataset present.

    `pool_test(record, engine_slots) -> bool` answers "did the engine return a gold at all".
    On repo/fix that is a set intersection. On docs it MUST be the suite's own ranker rather
    than is_gold() over the slot set: the ranker walks search_log with its rank and canonical
    matching intact, and re-deriving it from the flattened slots disagrees.
    """
    if pool_test is None:
        def pool_test(record, engine):
            return bool(golds(record) & engine)
    n = len(rows)
    if not n:
        return None
    prec = corr = gnd = pool = abstain = wrong = 0
    eng_rec_sum = 0.0
    from_mem = from_read = below_cap = 0
    rec_sum = amrr_sum = 0.0

    for r in rows:
        listed = submitted(r, family)
        engine = engine_slots(r, cap)
        committed = bool(r.get("committed", True))
        gold = golds(r)

        hits = [i + 1 for i, x in enumerate(listed[:10]) if is_gold(x, r)]
        # docs items carry one canonical answer that may live on several pages, so the
        # denominator is 1; repo/fix count each registered gold.
        n_gold = 1 if family == "docs" else max(len(gold), 1)
        # DISTINCT golds, not hit positions. An agent that cites the same PR twice -- once as
        # owner/repo#N and once as its URL -- produced two hits for one gold and scored recall
        # 1.0 on a two-wide item it half-answered.
        if family == "docs":
            found = 1 if hits else 0
        else:
            found = len({x for x in listed[:10] if is_gold(x, r) and x in gold})
        rec_sum += min(found, n_gold) / n_gold
        if hits:
            amrr_sum += 1 / hits[0]

        pooled = pool_test(r, engine)
        # ENGINE-SIDE recall@10, distinct from `recall` (which scores the AGENT's list) and from
        # `pool_hit` (binary: ANY gold). On the two-wide fix gold these separate sharply --
        # pool_hit 0.655 against eng_recall 0.439 for fc-mcp means that when the engine hits, it
        # usually surfaces ONE of the issue/PR pair, not both. That is what the lenient gold rule
        # is quietly buying, and it is invisible in a binary column.
        if family == "docs":
            eng_rec_sum += 1.0 if pooled else 0.0        # one canonical answer per item
        else:
            eng_rec_sum += len(gold & engine) / max(len(gold), 1)
        if family == "docs":
            # Deterministic, same shape as repo/fix: goldness is canonical-URL matching
            # (is_gold via the suite's ranker probe), engine membership is pool_test over the
            # actual search_log. No judge fields are read; a run needs no LLM to score.
            correct = bool(hits)
            grounded = bool(hits) and pooled
            ok = bool(committed and pooled and hits and hits[0] == 1)
        else:
            correct = bool(gold & set(listed))
            grounded = bool(gold & set(listed) & engine)
            # THE CITED GOLD, NOT ANY GOLD. `pooled` is item-level: on the two-wide fix gold it
            # is true when the engine returned EITHER the issue or the PR. Gating precision on
            # that let a memorised rank-1 citation of gold A pass as grounded because the engine
            # happened to surface gold B -- the exact memory/retrieval conflation the gap
            # decomposition exists to prevent, reappearing inside the metric that CLAIMS
            # grounding. Measured on this run it inflated fix precision by up to 0.083 and moved
            # Exa from 1st to 3rd. Require the artifact the agent actually led with.
            #
            # docs is deliberately NOT changed: its several canonical_sources are alternate
            # LOCATIONS of one answer, so engine-returned page B and agent-cited page A mean the
            # answer was retrieved. The fix pair is two DISTINCT artifacts; that is the difference.
            ok = bool(committed and listed and listed[0] in gold and listed[0] in engine)

        # WHY THE GAP IS DECOMPOSED. `correctness - groundedness` was read as "the model
        # remembered it", but three different things land in it and they mean opposite things:
        # from_memory  the gold is in NO pool          -> parametric recall, the model knew it
        # from_read    the gold is reader-origin only   -> a FETCH found what search never listed
        # below_cap    the engine DID return it, rank>10 -> a ranking failure, not memory
        # Measured on this run the composition flips per arm: fc-web/fix is 82% below_cap,
        # websearch/fix is 96% from_read, parallel/repo is 74% from_memory — and the same arm
        # flips between tracks. One number cannot carry that, and reading it as memory
        # mis-attributes a ranking failure as a hallucination.
        if correct and not grounded:
            hit = next((x for x in listed[:10] if is_gold(x, r)), None)
            uncapped = set()
            for _s in (r.get("search_log") or []):
                for _x in (_s.get("results") or []):
                    for _v in (_x.get("ref"), _x.get("url")):
                        if _v:
                            uncapped.add(norm(_v))
            prov = {norm(p[0]): (p[1] if len(p) > 1 else "")
                    for p in (r.get("pool_prov") or [])}
            if hit is not None and hit in uncapped:
                below_cap += 1                      # engine had it, ranked past the cap
            elif hit is not None and prov.get(hit) == "reader":
                from_read += 1                      # a page the agent fetched carried it
            else:
                from_mem += 1                       # nothing retrieved it at all

        corr += correct
        gnd += grounded
        pool += pooled
        prec += ok
        if not committed:
            abstain += 1
        elif not ok:
            wrong += 1

    dead = sum(1 for r in rows
               if any(e in str(r.get("error") or "") for e in DEAD_ERRORS))
    call_lat = sorted(x["latency"] for r in rows for x in (r.get("search_log") or [])
                      if x.get("latency"))
    e2e = sorted(r.get("latency") or 0 for r in rows)
    return dict(
        n=n, resolved=prec,
        dead=dead, dead_frac=dead / n, invalid=(dead / n) > DEAD_LIMIT,
        # Native arms (Claude's server-side WebSearch) produce no search_log at all. Several
        # engine-side quantities are undefined for them; this flag is what lets the invariant
        # checker skip the ones that cannot hold rather than reporting a phantom failure.
        has_search_log=any(r.get("search_log") for r in rows),
        precision=prec / n, recall=rec_sum / n, correctness=corr / n,
        groundedness=gnd / n, pool_hit=pool / n, amrr=amrr_sum / n,
        eng_recall=eng_rec_sum / n,
        # the correctness-groundedness gap, split by what actually produced the answer
        from_memory=from_mem / n, from_read=from_read / n, below_cap=below_cap / n,
        wrong=wrong / n, abstain=abstain / n,
        # EVERY CALL ONCE. The three counters do not overlap: `searches` and `reads` are the
        # budget counters (the native arm's WebSearch/WebFetch spend them too), and a vendor
        # fetch tool lands only in `fetch_log`. `ext_tool_calls` is NOT added -- it is the native
        # arm's own tally of the same blocks already in `searches`/`reads`, so including it
        # published roughly 2x the real effort for that arm alone, and `reads` was missing
        # entirely, so read_target calls never appeared in any arm's tool count.
        tools=st.mean([(r.get("searches") or 0) + (r.get("reads") or 0)
                       + len(r.get("fetch_log") or []) for r in rows]),
        searches=st.mean([(r.get("searches") or 0) for r in rows]),
        call_s=(call_lat[len(call_lat) // 2] if call_lat else None),
        e2e_s=e2e[len(e2e) // 2],
        cost=st.mean([r.get("cost") or 0 for r in rows]),
    )


def check_cell(label, m):
    """Return the invariant violations for one cell. Empty list means the definitions held."""
    bad = []
    if abs(m["precision"] + m["wrong"] + m["abstain"] - 1) > 1e-9:
        bad.append(f"{label}: precision + wrong + abstain != 1")
    if m["precision"] > m["groundedness"] + 1e-9:
        bad.append(f"{label}: precision > groundedness")
    if m["groundedness"] > m["correctness"] + 1e-9:
        bad.append(f"{label}: groundedness > correctness")
    # precision <= pool_hit holds only where pool_hit is measurable: a native arm has no
    # search_log, so its engine-side quantities are undefined rather than zero.
    if m.get("has_search_log", True) and m["precision"] > m["pool_hit"] + 1e-9:
        bad.append(f"{label}: precision > pool_hit")
    # THE DECOMPOSITION MUST BE COMPLETE. Every correct-but-ungrounded item is exactly one of
    # memory / read / below-cap; if these three do not close the gap, a fourth case exists and
    # the columns are quietly under-reporting it.
    if all(k in m for k in ("from_memory", "from_read", "below_cap")):
        lhs = m["groundedness"] + m["from_memory"] + m["from_read"] + m["below_cap"]
        if abs(lhs - m["correctness"]) > 1e-9:
            bad.append(f"{label}: groundedness + memory + read + below_cap != correctness "
                       f"({lhs:.4f} vs {m['correctness']:.4f})")
    return bad


# ------------------------------------------------------------------ run discovery
# The fix track answers to two directory tokens. `--track fix` writes `-fix-`, while sweeps
# predating this cleanup wrote `-fixshort-`. There is only one fix dataset, so both name the same
# cell and both must be found -- reading only one spelling is why a documented repro command
# could produce runs the reporter then reported as "no cells".
_TOKEN_ALIASES = {"fixshort": ("fixshort", "fix")}


def run_files(runs_dir, pass_label, token, arm, ext):
    """Task files for one cell: every directory token that names this track, and EVERY pass.

    `--n-runs 3` writes run1/, run2/ and run3/ under one cell directory. This globbed `run1`
    alone, so two thirds of a paid sweep was silently discarded and a three-pass cell reported
    the same numbers as a one-pass cell -- while the docs told people to run --n-runs 3."""
    hits = []
    for tok in _TOKEN_ALIASES.get(token, (token,)):
        hits += glob.glob(f"{runs_dir}/*{pass_label}-*-{tok}-{arm}/run*/tasks/*{ext}")
    return sorted(set(hits))


def _mean_cells(cells):
    """Average per-pass metric dicts into one cell. Each pass is an independent repeat of the
    SAME items, so metrics are averaged (n stays the item count); None means a quantity was
    undefined for that pass and is dropped rather than read as zero."""
    if len(cells) == 1:
        out = dict(cells[0])
        out["passes"] = 1
        return out
    out = {}
    for k in cells[0]:
        vals = [c.get(k) for c in cells]
        if all(isinstance(v, bool) for v in vals):
            out[k] = any(vals)
        elif any(v is None for v in vals):
            live = [v for v in vals if v is not None]
            out[k] = (st.mean(live) if live else None)
        else:
            out[k] = st.mean(vals)
    out["n"] = cells[0]["n"]
    out["resolved"] = round(st.mean([c["resolved"] for c in cells]))
    out["passes"] = len(cells)
    return out


# ------------------------------------------------------------------ report
# Docs runs may carry any docs_v3.x GT: the versions share one item set and differ only in how
# wide canonical_sources is. Scoring them all against the widest is deliberate. What is NOT safe
# is scoring a run whose items are absent from this file -- those qids resolve to no metadata and
# score 0.000 as though the arm failed. check_docs_coverage() turns that into a hard error.
#
# OVERRIDABLE because an expanded docs set is a
# DIFFERENT item set, not a wider view of the same one -- scoring v4 runs against v3.1.0 would trip
# check_docs_coverage on every qid. Set DOCS_SCORING_GT to the file the run was executed with:
# DOCS_SCORING_GT=<your docs dataset> python3 devdex/scorer/report_metrics.py
# PUBLIC RELEASE DEFAULT. Only the public sample ships here, so it is also the widest docs
# set; internally this defaults to the full docs file. Override with the env var to rescore a
# run against the dataset it was actually executed with.
DOCS_SCORING_GT = os.environ.get("DOCS_SCORING_GT", "docs_public.jsonl")


def check_docs_coverage(records, docs_meta, cell):
    """Every docs qid must exist in the scoring GT, else its score is a silent zero."""
    missing = sorted({r.get("qid") for r in records} - set(docs_meta)) if docs_meta else []
    if missing:
        raise SystemExit(
            f"{cell}: {len(missing)} of {len(records)} items are absent from {DOCS_SCORING_GT} "
            f"(run used gt_file={records[0].get('gt_file')!r}). Those qids would score 0.000 as "
            f"though the arm missed them. First: {missing[:3]}")


def build(runs_dir, pass_label):
    import devdex.scorer.suite as suite
    suite._DOCS_META = None
    suite.TRACKS["docs"] = DOCS_SCORING_GT
    docs_meta = suite._docs_meta()

    def gold_test(family):
        if family != "docs":
            return lambda item, record: item in golds(record)

        def _docs(item, record):
            probe = {"search_log": [{"results": [{"rank": 0, "url": item}]}]}
            return bool(suite._docs_rank(probe, docs_meta.get(record.get("qid"))))
        return _docs

    def pool_test(family):
        if family != "docs":
            return lambda record, engine: bool(golds(record) & engine)

        _is_gold = gold_test("docs")

        def _docs_pool(record, engine):
            if record.get("search_log"):
                # the ranker walks the real result list with its rank and canonical matching
                return bool(suite._docs_rank(record, docs_meta.get(record.get("qid"))))
            # A SERVER-SIDE ARM HAS NO search_log. Reading only the ranker scored native web
            # search 0.000 pool_hit -- and therefore 0.000 groundedness and precision -- on docs
            # even when it cited the canonical page, which is a gap in what the harness can
            # observe, not a retrieval failure. engine_slots already falls back to the
            # stream-harvested refs for exactly this; honour the same fallback here.
            return any(_is_gold(u, record) for u in engine)

        return _docs_pool

    out, failures = {}, []
    for track, (token, family) in TRACKS.items():
        out[track] = {}
        for arm in ARMS:
            files = run_files(runs_dir, pass_label, token, arm, ".json")
            if not files:
                continue
            per_pass = []
            for f in files:
                records = json.load(open(f))
                if family == "docs" and records:
                    check_docs_coverage(records, docs_meta, f"{track}/{arm}")
                one = score_records(records, family, gold_test(family), pool_test(family))
                if one:
                    per_pass.append(one)
            if not per_pass:
                continue
            m = _mean_cells(per_pass)
            # correctness == 1[recall > 0] on every track now that docs correctness is
            # citation matching too, so correctness >= recall must always hold.
            if m["recall"] > m["correctness"] + 1e-9:
                failures.append(f"{track}/{arm}: recall > correctness")
            failures += check_cell(f"{track}/{arm}", m)
            out[track][arm] = m
    return out, failures


def _tag(arm):
    return " [control]" if arm in CONTROLS else ""


W = 30      # name column; widest is "Firecrawl Search (developer)" + " [control]"


# (metric key, heading, width). ONE table drives both the heading and the figure under it --
# they were written as two independent format strings and drifted: a stale 'nDCG' heading with
# no column, and 'poolhit' at width 9 over a width-7 figure, put every number right of
# groundedness under the wrong heading.
SCORE_COLS = (("precision", "prec", 7), ("recall", "recall", 8), ("correctness", "corr", 7),
              ("groundedness", "grnd", 7), ("pool_hit", "poolhit", 9), ("amrr", "aMRR", 7))


def _row(name, m, tail):
    return (f"  {name:<{W}}"
            + "".join(f"{m[k]:>{w}.3f}" for k, _h, w in SCORE_COLS) + tail)


def _print(all_cells, failures):
    cols = tuple(k for k, _h, _w in SCORE_COLS)
    head = (f"  {'system':<{W}}" + "".join(f"{h:>{w}}" for _k, h, w in SCORE_COLS)
            + f"{'tools':>7}{'call_s':>8}{'e2e_s':>8}{'$/q':>7}")
    rule = "=" * len(head)
    for track, cells in all_cells.items():
        if not cells:
            continue
        n = next(iter(cells.values()))["n"]
        print(rule)
        print(f"{track_name(track).upper()}   n={n}   [{track}]")
        print(rule)
        print(head)
        print("  " + "-" * (len(head) - 2))
        for arm, m in sorted(cells.items(), key=lambda kv: -kv[1]["precision"]):
            cs = f"{m['call_s']:.1f}" if m["call_s"] else "  -  "
            print(_row(arm_name(arm) + _tag(arm), m,
                       f"{m['tools']:>7.1f}{cs:>8}{m['e2e_s']:>8.1f}{m['cost']:>7.3f}"))
        print()

    shared = [a for a in ARMS if all(a in c for c in all_cells.values() if c)]
    if shared:
        cells_present = [c for c in all_cells.values() if c]
        total = sum(next(iter(c.values()))["n"] for c in cells_present)
        n_tracks = len(cells_present)
        print(rule)
        # EQUAL-WEIGHT mean of the per-track means -- the SAME combined the published table uses
        # (README + benchmark/run_benchmark.py). Item-weighting here made this row silently
        # disagree with the headline; one definition, applied in both places.
        print(f"COMBINED   equal-weight mean of {n_tracks} track means ({total} items)")
        print(rule)
        chead = (f"  {'system':<{W}}" + "".join(f"{h:>{w}}" for _k, h, w in SCORE_COLS)
                 + f"{'resolved':>11}{'$/q':>7}")
        print(chead)
        print("  " + "-" * (len(chead) - 2))
        rows = []
        for arm in shared:
            cs = [c[arm] for c in all_cells.values() if c]
            N = sum(c["n"] for c in cs)
            T = len(cs)
            m = {k: sum(c[k] for c in cs) / T for k in cols}
            # cost is a resource total, not a per-track rate, so it stays item-weighted ($/q).
            m["cost"] = sum(c["cost"] * c["n"] for c in cs) / N
            rows.append((m["precision"], arm, m, sum(c["resolved"] for c in cs), N))
        for _, arm, m, res, N in sorted(rows, reverse=True):
            print(_row(arm_name(arm) + _tag(arm), m, f"{res:>7}/{N:<3}{m['cost']:>7.3f}"))

    print("\n  [control] = not a search product; excluded from engine rankings")
    print(f"  invariant checks: {len(failures)} failures"
          + ("" if not failures else "\n    " + "\n    ".join(failures)))
    return 1 if failures else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pass", dest="pass_label", default="p1",
                    help="run-directory pass prefix (p1, p2, ...)")
    ap.add_argument("--runs", default=str(DEVDEX_ROOT / "runs"),
                    help="directory holding <timestamp>-<label>/ run dirs")
    args = ap.parse_args()

    cells, failures = build(args.runs, args.pass_label)
    if not any(cells.values()):
        sys.exit(f"no {args.pass_label} cells under {args.runs}")
    sys.exit(_print(cells, failures))


if __name__ == "__main__":
    main()

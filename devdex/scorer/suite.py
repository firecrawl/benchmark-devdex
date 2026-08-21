"""suite — orchestration for the developer-retrieval eval.

Owns three things the harnesses deliberately do not:

  1. INDEPENDENT PASSES. A single pass cannot separate a real gap between engines from
     run-to-run noise. Vendors are non-deterministic and so is the agent. Every headline
     ships as mean with sample standard deviation over `--n-runs` passes.

  2. RUN LAYOUT. Each invocation writes to <output-dir>/<timestamp>-<label>/, one run<N>/
     directory per pass, plus a single results.json. Nothing overwrites anything, and a run
     is self-describing after the fact.

  3. AGGREGATION. Per-pass means, then cross-pass variance. The denominator is
     scored + errored — a failed item is a miss, not an exclusion. Items with no parseable
     gold are counted separately and excluded.

The harnesses remain runnable on their own; OUT_DIR is the only coupling.
"""
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

DEVDEX_ROOT = Path(__file__).resolve().parents[1]      # devdex/
REPO_ROOT = DEVDEX_ROOT.parent
HARNESS = DEVDEX_ROOT / "harness"
DATASETS = DEVDEX_ROOT / "gt"

# THE PUBLIC RELEASE SHIPS THE PUBLIC SAMPLE. These are the ~50% stratified subsets of each
# track; the full sets are held back so the memorisation gate keeps working (a fully published
# benchmark becomes training data). Gold rules, schema and scoring are identical -- only the item
# count differs, so a number produced here is comparable to another run of this same sample, not
# to the full-set table in the paper/blog.
TRACKS = {"repo": "repo_public.jsonl",
          "fix": "fix_public.jsonl",
          "docs": "docs_public.jsonl"}

# The two docs sets ask DIFFERENT questions and neither supersedes the other:
# v3.0.0  174 items, every query names its library, memory-gated at 5 samples.
# The shape real traffic has. Default.
# v2.1.0  294 items, 96% of queries name nothing, 32% answerable from memory alone.
# Larger, and the harder memory-recall slice. Reach for it with --dataset.
# Pass --dataset to override a track's default; the filename is stamped into results.json so a
# run always says which set produced it.
DATASET_ALIASES = {}          # every shipped set is a track default; no aliases needed

DRIVERS = {"opus": ("runner_sdk.py", None)}

DEAD_ERRORS = ("empty_stream", "timeout", "no_successful_search", "no_tool_calls")


# ---------------------------------------------------------------- scoring helpers
# ONE DEFINITION, imported rather than restated. This module prints the per-pass headline and
# feeds compare_to_baseline, while report_metrics.py prints the published table -- when the two
# kept private copies of "gold", "engine pool" and "precision" they silently disagreed on the
# same run: this file scored the pool UNCAPPED while the published table capped at 10, which on
# its own moved fc-web groundedness 0.450 -> 0.802. Import the definitions so a change to
# scoring cannot land in one scorer and not the other.
from devdex.scorer.report_metrics import (      # noqa: E402
    CAP, cite_ref as _cite_ref, engine_slots as _engine_slots, golds as _golds, norm as _norm,
)


def _cites(row):
    return [_cite_ref(c) for c in (row.get("citations") or [])]


def _engine_pool(row):
    """Refs the engine returned, capped at CAP -- identical to the published scorer."""
    return _engine_slots(row, CAP)


def _engine_rank(row, cap=CAP):
    """Position at which the SEARCH BACKEND returned a gold, within the depth cap. None if it
    never did. Capped for the same reason the pool is: an arm returning ~19 results must not be
    credited for a hit in slots the other arms were never allowed to return."""
    g, best = _golds(row), None
    for s in row.get("search_log") or []:
        for x in s.get("results") or []:
            if cap and (x.get("rank") or 0) >= cap:
                continue
            if _norm(x.get("ref")) in g or _norm(x.get("url")) in g:
                rk = (x.get("rank") or 0) + 1
                best = rk if best is None else min(best, rk)
    return best


# ------------------------------------------------------- docs retrieval scoring
# The docs golds are chunk anchors (`doc::path.md#42`), so for a long time the track had no
# rank metric at all. It does have a resolvable file URL, and from v2.1.0 each item also
# carries the project's own rendered form of the same page. A returned result matches the gold
# in EITHER canonical form:
#
# repo file    github.com/<owner>/<repo>/blob/<any ref>/<path>   (ref/anchor ignored)
# rendered     <declared docs site>/.../<page stem>
#
# Both are the project publishing its own documentation. Mirrors are excluded structurally:
# the site comes from the repo's GitHub metadata, so translations (cn.vite.dev) and pinned
# versions (v6.vite.dev) are different hosts. The page stem pins the specific document, so
# landing anywhere on a canonical host is not sufficient.
_DOCS_META = None
_GH_BLOB = re.compile(r"https?://(?:www\.)?github\.com/([^/]+)/([^/]+)/blob/[^/]+/(.+?)(?:#.*)?$")


def _strip_host(u):
    u = str(u or "").strip().lower().rstrip("/")
    for pre in ("https://", "http://"):
        if u.startswith(pre):
            u = u[len(pre):]
    return u[4:] if u.startswith("www.") else u


_ORDER_PREFIX = re.compile(r"^\d+[_-]")


def _published_stems(stems):
    """Registered stems come from the SOURCE filename; docs sites serve the PUBLISHED slug.

    Gradio's tree orders guides on disk (`07_streaming/01_streaming-ai-generated-audio.md`) and
    publishes them without the ordering prefix (`gradio.app/guides/streaming-ai-generated-audio`).
    Comparing the raw stem meant the site branch could never fire for those items: of 21 distinct
    stems agents cited on gradio.app, ZERO matched, so an arm citing the rendered docs scored a
    miss while an arm citing the GitHub blob scored a hit. That is not a coverage difference, it
    is a filename convention -- and it hit hardest the arms that answer from rendered docs.

    Both forms are kept, so this only ever ADDS a valid gold. 14 of 174 items carry such a prefix
    (gradio 12, svelte 2)."""
    out = set()
    for st in stems:
        out.add(st)
        stripped = _ORDER_PREFIX.sub("", st)
        if stripped and stripped != st:
            out.add(stripped)
    return out


def _docs_meta():
    """qid -> {repo_files, site, page_stems, memory_answerable} from the docs dataset."""
    global _DOCS_META
    if _DOCS_META is None:
        _DOCS_META = {}
        p = DATASETS / TRACKS["docs"]
        if p.exists():
            for line in open(p):
                r = json.loads(line)
                cs = r.get("canonical_sources") or {}
                # CASE-FOLD BOTH SIDES. _docs_rank lowercases the candidate URL (via _strip_host
                # and its own .lower() on the blob path) before looking it up here, so a gold
                # carrying any uppercase character could never be matched -- the lookup key was
                # lowercase and the registered key was not. It silently zeroed BOTH branches for
                # those items: 14 of 367 on docs v4 (PrefectHQ/prefect, sgl-project Meituan
                # paths) and 3 of 174 on the shipped v3.1.0, plus stems like README and sumArray.
                # A correct retrieval scored as a miss purely because of filename casing.
                _DOCS_META[r.get("id")] = {
                    "repo_files": {str(f).lower() for f in (cs.get("repo_files") or [])},
                    "site": _strip_host(cs.get("site") or ""),
                    "page_stems": {s.lower() for s in
                                   _published_stems(cs.get("page_stems") or [])},
                    "memory_answerable": bool(r.get("memory_answerable")),
                }
    return _DOCS_META


def _docs_rank(row, meta, cap=CAP):
    """Best rank at which the engine returned the gold document. None if it never did.

    CAPPED, like every other track. This walked the whole result list at any depth while
    repo/fix went through engine_slots(cap=10), so docs alone credited a gold the engine
    returned in a slot no other arm was allowed to return -- measured live, fc-web (which
    returns ~19 results because Firecrawl applies `limit` per group) was credited for a gold at
    engine rank 18. And this is not only the pool_hit column: on docs `pooled` also gates
    `groundedness` and `precision`, so an uncapped pool silently makes the SAME agent behaviour
    score higher on docs than on repo or fix.

    The is_gold probe passes a synthetic single result at rank 0, so it is unaffected."""
    if not meta:
        return None
    files, site, stems = meta["repo_files"], meta["site"], meta["page_stems"]
    best = None
    for s in row.get("search_log") or []:
        for x in s.get("results") or []:
            if cap and (x.get("rank") or 0) >= cap:
                continue
            url = x.get("url")
            hit = False
            m = _GH_BLOB.match(str(url or ""))
            if m:
                hit = f"{m.group(1).lower()}/{m.group(2).lower()}:{m.group(3).lower()}" in files
            if not hit and site and stems:
                u = _strip_host(url)
                if u.startswith(site):
                    tail = u[len(site):].strip("/").split("?")[0].split("#")[0]
                    last = tail.split("/")[-1] if tail else ""
                    # Docs sites serve the same page under several extensions — vite.dev
                    # publishes both /guide/x and /guide/x.md, and Firecrawl returns the .md
                    # form. Comparing the raw segment rejected the canonical page as a miss.
                    for ext in (".md", ".mdx", ".html", ".htm"):
                        if last.endswith(ext):
                            last = last[: -len(ext)]
                            break
                    hit = bool(last) and last in stems
            if hit:
                rk = (x.get("rank") or 0) + 1
                best = rk if best is None else min(best, rk)
    return best


def _no_depth(rows):
    """True when the arm's tool exposes no k/limit, so its result list is a different size from
    every other arm's. Read off the record rather than importing arms, so a rescore of an old
    run does not depend on the current registry."""
    for r in rows:
        if (r.get("config") or {}).get("no_depth"):
            return True
    return False


def score_pass(rows, track):
    """Metrics for one pass. Denominator is every row: abstains and failed runs are misses."""
    n = len(rows)
    if not n:
        return {}
    dead = sum(1 for r in rows if any(e in (r.get("error") or "") for e in DEAD_ERRORS))

    if track == "docs":
        meta = _docs_meta()
        clean = [r for r in rows if not meta.get(r.get("qid"), {}).get("memory_answerable")]
        has_log = any(r.get("search_log") for r in rows)
        # DETERMINISTIC docs scoring. A cited source counts when it resolves to one of the
        # item's canonical pages (repo source file, or the project's own rendered page); the
        # engine gets credit when it returned that page itself. No judge, no answer text.
        def _cited_rank(r):
            m = meta.get(r.get("qid"))
            for i, u in enumerate(r.get("sources") or [], 1):
                probe = {"search_log": [{"results": [{"rank": 0, "url": u}]}]}
                if _docs_rank(probe, m):
                    return i
            return None
        found10 = rank1 = 0
        mrr = 0.0
        prec = corr = gnd = 0
        for r in rows:
            best = _docs_rank(r, meta.get(r.get("qid")))
            if best:
                mrr += 1 / best
                found10 += best <= 10
                rank1 += best == 1
            cr = _cited_rank(r)
            corr += cr is not None
            gnd += bool(cr is not None and best)
            prec += bool(r.get("committed", True) and cr == 1 and best)
        gated_prec = ([r for r in clean] and
                      sum(1 for r in clean
                          if r.get("committed", True) and _cited_rank(r) == 1
                          and _docs_rank(r, meta.get(r.get("qid")))) / len(clean))
        return {
            "n": n,
            "precision": round(prec / n, 4),
            # Same headline over the items the memory floor could NOT answer, so the figure is
            # not part parametric recall. Report both; quote the gated one.
            "precision_gated": round(gated_prec, 4) if clean else None,
            "gated_n": len(clean),
            "correctness": round(corr / n, 4),
            "groundedness": round(gnd / n, 4),
            # Retrieval alone: did the engine return the gold document in EITHER canonical form?
            "doc_found@10": None if (not has_log or _no_depth(rows)) else round(found10 / n, 4),
            "doc_rank1": None if (not has_log or _no_depth(rows)) else round(rank1 / n, 4),
            "doc_MRR": None if (not has_log or _no_depth(rows)) else round(mrr / n, 4),
            "no_depth": _no_depth(rows),
            "dead_runs": round(dead / n, 4),
            "cost_usd": round(sum(r.get("cost") or 0 for r in rows), 4),
        }

    precision = 0
    for r in rows:
        grounded = _golds(r) & set(_cites(r)) & _engine_pool(r)
        c = _cites(r)
        if r.get("committed") and grounded and c and c[0] in grounded:
            precision += 1
    ranks = [x for x in (_engine_rank(r) for r in rows) if x]
    # pool_hit / engR1 / engMRR are rates over the returned list. If the vendor fixes that
    # list's size and we cannot request k=10, those rates are not comparable to arms that
    # return 10 -- suppress rather than print a depth-2 number in a depth-10 column.
    nd = _no_depth(rows)
    return {
        "n": n,
        "precision": round(precision / n, 4),
        "correctness": round(sum(1 for r in rows if _golds(r) & set(_cites(r))) / n, 4),
        "groundedness": round(sum(1 for r in rows
                                  if _golds(r) & set(_cites(r)) & _engine_pool(r)) / n, 4),
        "pool_hit": None if nd else round(sum(1 for r in rows if _golds(r) & _engine_pool(r)) / n, 4),
        "engR1": None if nd or not ranks else round(sum(1 for x in ranks if x == 1) / len(ranks), 4),
        "engMRR": None if nd or not ranks else round(sum(1 / x for x in ranks) / len(ranks), 4),
        "no_depth": nd,
        "abstain": round(sum(1 for r in rows if not r.get("committed")) / n, 4),
        "dead_runs": round(dead / n, 4),
        "cost_usd": round(sum(r.get("cost") or 0 for r in rows), 4),
    }


def stat(xs):
    """Mean and SAMPLE standard deviation (n-1). One pass has no dispersion, so std is null
    rather than 0 — a zero would read as 'measured and stable'."""
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / (n - 1) if n > 1 else None
    return {"mean": round(mean, 4),
            "std": round(math.sqrt(var), 4) if var is not None else None,
            "min": round(min(xs), 4), "max": round(max(xs), 4),
            "passes": n, "values": [round(x, 4) for x in xs]}


def aggregate(per_pass):
    """Cross-pass variance for every metric present in the passes."""
    keys = {k for p in per_pass for k in p if k not in ("n", "graded")}
    return {"per_pass": per_pass,
            **{k: stat([p.get(k) for p in per_pass]) for k in sorted(keys)}}


# ---------------------------------------------------------------- run
def run_cell(track, arm, driver, n_runs, concurrency, out_root, label=None, limit=None,
             dataset=None, resume_dir=None, auto_resume=False, model_override=None):
    if track not in TRACKS:
        raise SystemExit(f"unknown track: {track}")
    if driver not in DRIVERS:
        raise SystemExit(f"unknown driver: {driver}")

    script, model = DRIVERS[driver]
    # THE DRIVER'S MODEL IS A DEFAULT, NOT A PIN. The runner reads argv[6] BEFORE $MODEL, so
    # appending the DRIVERS value unconditionally made the env var dead on this path: a run
    # launched with an unsupported MODEL value silently executed the default model and wrote records
    # stamped with that default. Two "different" model runs came out identical to three decimals,
    # which is what an unoverridable default looks like from the outside.
    model = model_override or os.environ.get("MODEL") or model
    # DEFAULT LABEL CARRIES THE PASS TOKEN. report_metrics.py's discovery glob requires the
    # sweep's pass label (default "p1") literally inside the run directory name -- see the CI
    # workflow, which was patched to pass `--label ci-opus-...` after landing without one made
    # every matrix run unscoreable (devdex/tests/test_portability.py). Baking "p1-" into the
    # untouched default closes the same gap for the plain `run_eval.py` invocation the README
    # documents, so `report_metrics.py --pass p1` finds it with no extra flag.
    label = label or f"p1-{driver}-{track}-{arm}"
    # RESUME. Pointing at an existing run directory reuses its run<N>/tasks/ files, and the
    # harness's own _resume() then skips every qid already finished. This is what makes a
    # rate-limited vendor runnable across days: stop when the daily cap is hit, continue
    # tomorrow into the same directory and the same results.json.
    # AUTO-RESUME. Without this, a killed wrapper restarts every unfinished cell from zero:
    # each invocation minted a fresh timestamp, so the harness's own _resume() never saw the
    # partial cell file. With it, a rerun of the same label lands in the SAME directory and
    # _resume() skips every qid already written (flushed every CHUNK=25 items).
    if auto_resume and not resume_dir:
        prior = sorted(Path(out_root).glob(f"*-{label}"))
        if prior:
            resume_dir = str(prior[-1])          # newest wins

    if resume_dir:
        # MUST be absolute. The harness subprocess runs with cwd=devdex/harness, so a relative
        # OUT_DIR resolved against THAT directory and silently built a shadow tree at
        # devdex/harness/runs/... — _resume then read the shadow copy while the real cell sat
        # untouched, reporting "6 already done" against a 3-row file.
        run_dir = Path(resume_dir).resolve()
        if not run_dir.exists():
            raise SystemExit(f"--resume-dir {run_dir} does not exist")
        stamp = run_dir.name.split("-")[0]
        done = sum(1 for _ in run_dir.glob("run*/tasks/*.json"))
        print(f"resuming into {run_dir} ({done} cell file(s) already present)", flush=True)
    else:
        stamp = time.strftime("%Y%m%dT%H%M%S")
        run_dir = (Path(out_root) / f"{stamp}-{label}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)

    ds = DATASET_ALIASES.get(dataset, dataset) if dataset else TRACKS[track]
    gt = DATASETS / ds
    if not gt.exists():
        raise SystemExit(f"dataset not found: {gt}")
    env = {**os.environ, "GT_FILE": str(gt), "CHUNK": os.environ.get("CHUNK", "25")}

    per_pass, pass_dirs = [], []
    for i in range(1, n_runs + 1):
        pdir = run_dir / f"run{i}"
        pdir.mkdir(exist_ok=True)
        env["OUT_DIR"] = str(pdir)
        argv = [sys.executable, str(HARNESS / script), track, arm,
                str(limit) if limit else "all", str(concurrency), f"p{i}"]
        if model:
            argv.append(model)
        print(f"\n=== pass {i}/{n_runs}  {label}  -> {pdir}", flush=True)
        rc = subprocess.call(argv, cwd=str(HARNESS), env=env)
        if rc != 0:
            print(f"    pass {i} exited {rc}", flush=True)

        cell = pdir / "tasks" / f"p{i}_{track}_{arm}.json"
        if cell.exists():
            rows = json.load(open(cell))
            m = score_pass(rows, track)
            per_pass.append(m)
            pass_dirs.append(str(pdir.relative_to(run_dir)))
            print(f"    pass {i}: {m}", flush=True)
        else:
            print(f"    pass {i}: no cell written at {cell}", flush=True)

    results = {
        "label": label, "track": track, "arm": arm, "driver": driver,
        "model": model or "claude-opus-4-8",
        "dataset": ds,
        "n_runs": n_runs, "completed_passes": len(per_pass),
        "concurrency": concurrency,
        "started": stamp,
        "pass_dirs": pass_dirs,
        "aggregate": aggregate(per_pass) if per_pass else None,
    }
    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {run_dir / 'results.json'}", flush=True)
    return results


def compare_to_baseline(results, baseline_path):
    """Delta against a frozen baseline, for the SAME cell.

    A baseline manifest keys its cells by driver/track/arm, so we look up the matching one
    rather than assuming the file describes a single cell. Returns None (rather than a
    misleading zero) when the baseline has no comparable cell.
    """
    p = Path(baseline_path)
    if not p.exists():
        return None
    base = json.load(open(p))
    key = f"{results['driver']}/{results['track']}/{results['arm']}"
    cell = (base.get("cells") or {}).get(key)
    if cell is None:
        return {"error": f"baseline has no cell {key}",
                "available": sorted(base.get("cells") or {})[:8]}

    head = "precision"
    cur = (results.get("aggregate") or {}).get(head)
    ref = cell.get(head)
    if not cur or ref is None:
        return None

    delta = cur["mean"] - ref
    out = {"cell": key, "metric": head, "baseline": ref, "current": cur["mean"],
           "delta": round(delta, 4),
           "baseline_passes": cell.get("n_runs", 1), "current_passes": cur["passes"]}
    # A delta is only a significance claim if BOTH sides have dispersion. The shipped baseline
    # is single-pass, so it never does — say that instead of implying the delta is meaningful.
    if cur.get("std") and cell.get("n_runs", 1) > 1:
        out["delta_in_std"] = round(delta / cur["std"], 2) if cur["std"] else None
    else:
        out["note"] = ("baseline is single-pass — this is a raw delta, not a significance "
                       "claim. Re-run the baseline with --n-runs 3 to compare distributions.")
    return out



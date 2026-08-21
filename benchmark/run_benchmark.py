"""run_benchmark.py — run DevDex against your MCP server.

ONE MODE, ONE SCORER. Your engine is mounted as a real arm over MCP and driven by the SAME agent
harness that produced the published table: same Controller, same deny-by-default tool gate, same
depth cap, same scorer. Scores come from devdex.scorer.report_metrics — the module that publishes
our numbers — so there is no second implementation that can drift from it.

WHAT WAS REMOVED AND WHY. An earlier version had a cheap `retrieval` mode that called your
`search()` once per query and scored the raw top 10 with its own local copies of the gold-matching
rules. Two problems, both fatal for a public benchmark:

  * A SECOND SCORER. Those local rules missed fixes the real scorer has (case-folded docs matching,
    published-slug stems, dead-run detection). Two scorers means a submitter's number is not
    comparable to ours and nobody can say which is authoritative.
  * A NOT-COMPARABLE NUMBER. An agent reformulates queries and can rescue a miss by reading a page.
    A raw top-10 is a different quantity, and publishing it beside the table invited exactly the
    comparison it does not support.

So this costs real money to run (~$0.28/item), and that is inherent: the published metric is
agent-side, and measuring it means driving the agent.

Usage:
    export ANTHROPIC_API_KEY=...        # you drive the same model the table used

    python3 benchmark/run_benchmark.py --name yourco --mcp-url https://mcp.yourco.com/mcp \\
        --search-tool your_search --track repo --limit 10   # smoke
    python3 benchmark/run_benchmark.py --name yourco --mcp-url https://mcp.yourco.com/mcp \\
        --search-tool your_search --track all                # full
    python3 benchmark/run_benchmark.py --name yourco --report-only                # re-print scores
"""
import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np

BOOT = 2000     # bootstrap resamples, matching the published table's methodology
ALPHA = 0.05    # 95% CI

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

# The public sample lives ONCE, at devdex/gt/ — the same files devdex/run_eval.py reads for the
# scoring step; the two are supposed to be identical but nothing enforced that, so a dataset
# update landing in one and not the other would score a submitter against stale metadata while
# the actual run used the current file. DATA is read from suite.TRACKS so it can't drift again.
import devdex.scorer.suite as suite                                     # noqa: E402

DATA = dict(suite.TRACKS)
TRACKS = ("repo", "fix", "docs")


def _scorer_for(track):
    """(is_gold, pool_test) exactly as report_metrics expects — imported, never reimplemented."""
    from devdex.scorer.report_metrics import golds
    if track != "docs":
        return (lambda item, rec: item in golds(rec)), (lambda rec, eng: bool(golds(rec) & eng))

    # docs matches on canonical URL, so it needs the dataset's own metadata.
    suite._DOCS_META = None
    meta = suite._docs_meta()

    def is_gold(item, rec):
        probe = {"search_log": [{"results": [{"rank": 0, "url": item}]}]}
        return bool(suite._docs_rank(probe, meta.get(rec.get("qid"))))

    def pool_test(rec, eng):
        if rec.get("search_log"):
            return bool(suite._docs_rank(rec, meta.get(rec.get("qid"))))
        return any(is_gold(u, rec) for u in eng)

    return is_gold, pool_test


def _item_values(recs, family, is_gold, pool_test):
    """Per-item recall and aMRR, via the real scorer on a one-row list each time -- not a second
    copy of the hit/rank arithmetic. Bootstrapping needs the item-level values score_records()
    only returns pre-averaged; this is the only way to get them without reimplementing the rule
    for what counts as a hit, which is exactly the drift `_scorer_for` above already warns about."""
    from devdex.scorer.report_metrics import score_records
    recall, amrr = [], []
    for r in recs:
        m = score_records([r], family, is_gold, pool_test)
        recall.append(m["recall"])
        amrr.append(m["amrr"])
    return np.array(recall), np.array(amrr)


def _ci(values, rng):
    """95% percentile bootstrap CI on the mean, resampling WITH replacement -- "how much would
    this estimate move on a rerun", the same question the published table's CI answers."""
    n = len(values)
    means = [rng.choice(values, size=n, replace=True).mean() for _ in range(BOOT)]
    lo, hi = np.percentile(means, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)])
    return round(float(lo), 3), round(float(hi), 3)


def report(engine, out=None):
    """Read the run records back and score them with the published scorer."""
    from devdex.scorer.report_metrics import score_records
    rng = np.random.default_rng(0)
    rows, results = [], {}
    for t in TRACKS:
        files = sorted(glob.glob(str(ROOT / f"devdex/runs/*ext-{engine}-{t}/run*/tasks/*.json")))
        if not files:
            continue
        recs = []
        for f in files:
            recs += json.load(open(f))
        is_gold, pool_test = _scorer_for(t)
        m = score_records(recs, t, is_gold, pool_test)
        if not m:
            continue
        # THE 10% BAR, same rule as the published table: a cell that mostly failed is not a
        # score, so it is excluded here rather than reported as if it were one.
        if m["dead_frac"] > 0.10:
            print(f"  {t:8}UNMEASURED — {m['dead_frac']:.0%} dead runs (>10% bar)")
            continue
        recall_vals, amrr_vals = _item_values(recs, t, is_gold, pool_test)
        r_ci, m_ci = _ci(recall_vals, rng), _ci(amrr_vals, rng)
        results[t] = {"recall@10": round(m["recall"], 3), "recall@10_ci": list(r_ci),
                      "MRR@10": round(m["amrr"], 3), "MRR@10_ci": list(m_ci)}
        rows.append((t, m, recall_vals, amrr_vals, r_ci, m_ci))

    if not rows:
        print(f"no runs found for engine {engine!r} — run without --report-only first")
        return 1

    print(f"\n  {'track':8}{'recall@10':>12}{'95% CI':>16}{'MRR@10':>9}{'95% CI':>16}")
    for t, m, _, _, r_ci, m_ci in rows:
        print(f"  {t:8}{m['recall']:>12.3f}{str(list(r_ci)):>16}"
              f"{m['amrr']:>9.3f}{str(list(m_ci)):>16}")

    # Combined is defined ONLY on all three tracks, and the CI is stratified within each --
    # resample every track's items separately (keeping the mix fixed), then average the three
    # per-resample track means. A single-track product has no combined score.
    if len(rows) == 3:
        cr = sum(m["recall"] for _, m, _, _, _, _ in rows) / 3
        cm = sum(m["amrr"] for _, m, _, _, _, _ in rows) / 3
        recall_boot = np.mean([[rng.choice(rv, size=len(rv), replace=True).mean()
                                for _, _, rv, _, _, _ in rows] for _ in range(BOOT)], axis=1)
        amrr_boot = np.mean([[rng.choice(av, size=len(av), replace=True).mean()
                              for _, _, _, av, _, _ in rows] for _ in range(BOOT)], axis=1)
        r_lo, r_hi = (round(float(x), 3) for x in
                      np.percentile(recall_boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]))
        m_lo, m_hi = (round(float(x), 3) for x in
                      np.percentile(amrr_boot, [100 * ALPHA / 2, 100 * (1 - ALPHA / 2)]))
        print(f"  {'combined':8}{cr:>12.3f}{str([r_lo, r_hi]):>16}"
              f"{cm:>9.3f}{str([m_lo, m_hi]):>16}")
        results["combined"] = {"recall@10": round(cr, 3), "recall@10_ci": [r_lo, r_hi],
                               "MRR@10": round(cm, 3), "MRR@10_ci": [m_lo, m_hi]}
    else:
        print(f"  {'combined':8}{'--':>12}{'':>16}{'--':>9}{'':>16}   (needs all three tracks)")

    payload = {"engine": engine, "mode": "agent", "depth_cap": 10,
               "dataset": "public sample (stratified ~50% of each track)",
               "scorer": "devdex.scorer.report_metrics.score_records",
               "tracks": results}
    if out:
        Path(out).write_text(json.dumps(payload, indent=2))
        print(f"\nwrote {out}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", help="label for your engine, e.g. 'yourco'. Defaults to 'external'.")
    ap.add_argument("--mcp-url", help="your MCP server URL")
    ap.add_argument("--search-tool", default="search", help="search tool name on your MCP server")
    ap.add_argument("--fetch-tool", default=None, help="your reader's tool name, if you ship one")
    ap.add_argument("--auth", default=None, help="Authorization header, e.g. 'Bearer sk-...'")
    ap.add_argument("--track", default="all", choices=("repo", "docs", "fix", "all"))
    ap.add_argument("--n-runs", type=int, default=1)
    ap.add_argument("--limit", type=int, help="first N items only, for a smoke run")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--report-only", action="store_true",
                    help="skip running; just re-score records already on disk")
    a = ap.parse_args()

    if not a.report_only and not a.mcp_url:
        sys.exit("give --mcp-url: your MCP server, mounted as shipped, exactly like every arm in "
                 "the published table.")
    label = a.name or "external"

    if a.mcp_url:
        # arms.py reads these env vars directly, so the CLI stays the single entry point.
        os.environ["DEVDEX_EXT_MCP_URL"] = a.mcp_url
        os.environ["DEVDEX_EXT_SEARCH_TOOL"] = a.search_tool
        if a.fetch_tool:
            os.environ["DEVDEX_EXT_FETCH_TOOL"] = a.fetch_tool
        if a.auth:
            os.environ["DEVDEX_EXT_AUTH"] = a.auth

    if not a.report_only:
        import register_arm
        rc = register_arm.run_agent_mode(
            engine=label, track=a.track, limit=a.limit,
            n_runs=a.n_runs, workers=a.workers, datasets=DATA)
        if rc:
            print(f"\none or more tracks exited non-zero (rc={rc}); scoring what landed", flush=True)

    return report(label, a.out)


if __name__ == "__main__":
    sys.exit(main())

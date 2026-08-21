#!/usr/bin/env python3
"""Entry point for the developer-retrieval eval.

    python3 devdex/run_eval.py --track repo --arm fc-mcp --driver opus --n-runs 3

Runs N independent passes of one (track, arm, driver) cell, writes each pass to its own
directory, and aggregates with cross-pass standard deviation. Credentials come from .env.

Preflight runs first unless --skip-preflight: it pins each vendor's tool names and schemas to
devdex/harness/tools_manifest.json. If a vendor renames a tool, this fails loudly instead of the
arm silently scoring 0.000.
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

DEVDEX_ROOT = Path(__file__).resolve().parent          # devdex/
sys.path.insert(0, str(DEVDEX_ROOT.parent))

from devdex.scorer.suite import (  # noqa: E402
    DRIVERS, TRACKS, compare_to_baseline, run_cell,
)

# Read the arm list from the registry, not a copy. A hardcoded list silently rejected
# newly-added arms (mintlify, context7) with "invalid choice" long after they were wired.
sys.path.insert(0, str(DEVDEX_ROOT / "harness"))
import arms as _arms  # noqa: E402
ARMS = sorted(_arms.ARMS)


def load_env():
    f = DEVDEX_ROOT.parent / ".env"
    if not f.exists():
        return
    for line in f.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--track", required=True, choices=sorted(TRACKS))
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--driver", required=True, choices=sorted(DRIVERS))
    ap.add_argument("--n-runs", type=int, default=1,
                    help="independent passes. 3 for a reportable number; 1 gives no variance.")
    ap.add_argument("--max-workers", type=int, default=4, help="concurrent items per pass")
    ap.add_argument("--output-dir", default=str(DEVDEX_ROOT / "runs"))
    ap.add_argument("--label", default=None, help="run directory suffix")
    ap.add_argument("--limit", type=int, default=None, help="first N items only, for smoke runs")
    ap.add_argument("--dataset", default=None,
                    help="override the track's dataset with a filename under devdex/gt/ "
                         "(each track already defaults to its own shipped set)")
    ap.add_argument("--baseline", default=None,
                    help="path to a baseline results.json to diff the headline against")
    ap.add_argument("--resume-dir", default=None,
                    help="continue into an existing runs/<timestamp>-<label> directory instead "
                         "of creating a new one; finished qids are skipped. Use for vendors with "
                         "a daily request cap (Mintlify: 1,000/day).")
    ap.add_argument("--auto-resume", action="store_true",
                    help="reuse an existing runs/ dir with the same --label instead of minting a "
                         "new timestamp, so a killed run continues from its last flush. Makes a "
                         "wrapper script crash-safe without tracking directory names.")
    ap.add_argument("--model", default=None,
                    help="override the driver's default model. The runner "
                         "reads argv before $MODEL, so this is the only reliable way to change it.")
    ap.add_argument("--skip-preflight", action="store_true")
    args = ap.parse_args()

    load_env()

    if args.n_runs == 1:
        print("note: --n-runs 1 produces no variance estimate. A single-pass number cannot "
              "separate a real gap from run-to-run noise.\n", flush=True)

    if not args.skip_preflight and args.arm not in ("no-tool", "websearch"):
        rc = subprocess.call([sys.executable, "preflight.py", args.arm],
                             cwd=str(DEVDEX_ROOT / "harness"))
        if rc != 0:
            sys.exit("preflight failed — fix arm tool names or credentials before running")

    results = run_cell(track=args.track, arm=args.arm, driver=args.driver,
                       n_runs=args.n_runs, concurrency=args.max_workers,
                       out_root=args.output_dir, label=args.label, limit=args.limit,
                       dataset=args.dataset, resume_dir=args.resume_dir, auto_resume=args.auto_resume, model_override=args.model)

    agg = results.get("aggregate") or {}
    head = "precision"
    if agg.get(head):
        s = agg[head]
        std = f" ± {s['std']}" if s.get("std") is not None else " (single pass, no std)"
        print(f"\n{head}: {s['mean']}{std}   range [{s['min']}, {s['max']}]   "
              f"passes={s['passes']}")

    if args.baseline:
        cmp = compare_to_baseline(results, args.baseline)
        if cmp:
            print("\nvs baseline:", json.dumps(cmp, indent=2))


if __name__ == "__main__":
    main()

"""register_arm.py — mount a third-party engine as a real DevDex arm.

WHY THIS EXISTS. The published numbers are AGENT-side: Claude Opus chooses its own queries, may
read a page it found, and commits to a ranked answer. A bare "call search once, score the top 10"
harness measures a different quantity and cannot be compared to the table. So a submitter has to
run the same agent loop we do — which means their engine has to become an Arm, not a side script.

YOU NEED AN MCP SERVER (Exa, Parallel, Mintlify, Context7 and Firecrawl all do). Point
DEVDEX_EXT_MCP_URL at it and devdex/harness/arms.py mounts it as an `mcp` arm exactly like every
published one — same Controller, same deny-by-default tool gate, same depth cap, same scorer.

WHAT THE HARNESS ENFORCES ON YOUR ARM, unchanged from every published arm:

  * exactly one search tool live per run; everything else is denied by a PreToolUse hook
  * results truncated to depth 10 before scoring
  * refs your search returned may score; refs reached by READING a page are tagged agent_pool and
    can never be credited as a retrieval hit
  * a failed call is a miss that stays in the denominator; >10% dead runs => the cell is reported
    unmeasured rather than scored

Usage (agent mode, reproduces the published methodology):

    export ANTHROPIC_API_KEY=...        # you drive the same model we did
    export DEVDEX_EXT_MCP_URL=https://mcp.yourco.com/mcp
    python3 benchmark/run_benchmark.py --name yourco --mcp-url $DEVDEX_EXT_MCP_URL --track repo
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

CAP = 10


def run_agent_mode(engine, track="all", limit=None, n_runs=1, workers=3, datasets=None):
    """Run the PUBLISHED methodology: the real agent loop with the submitted engine as an arm.

    Shells out to devdex/run_eval.py rather than re-implementing anything. One code path, one
    scorer, no lenient branch for custom arms. DEVDEX_EXT_MCP_URL must already be set (see
    run_benchmark.py's --mcp-url); arms.py mounts it as kind="mcp" -- identical treatment to Exa,
    Parallel, Mintlify and Context7, whose numbers in the published table came from their own MCP
    surfaces mounted the same way.
    """
    import os
    import subprocess
    import sys

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("agent mode needs ANTHROPIC_API_KEY — you drive the same model the table used.\n"
                 "This is the cost barrier, not a formality: ~$0.28/item, so ~$170 for one arm\n"
                 "across the 594-item public sample.")

    tracks = ["repo", "docs", "fix"] if track == "all" else [track]
    env = dict(os.environ)
    print(f"  mounting MCP server {env['DEVDEX_EXT_MCP_URL']} "
          f"(tool={env.get('DEVDEX_EXT_SEARCH_TOOL', 'search')!r}) as arm 'external'", flush=True)
    rc = 0
    for t in tracks:
        cmd = [sys.executable, "devdex/run_eval.py", "--track", t, "--arm", "external",
               "--driver", "opus", "--n-runs", str(n_runs), "--max-workers", str(workers),
               "--skip-preflight", "--auto-resume",
               "--label", f"ext-{engine}-{t}"]
        if datasets and t in datasets:
            cmd += ["--dataset", datasets[t]]
        if limit:
            cmd += ["--limit", str(limit)]
        print(f"\n=== {t}: {' '.join(cmd[1:])}", flush=True)
        rc |= subprocess.call(cmd, env=env, cwd=str(ROOT))
    return rc

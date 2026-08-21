"""Load `_resume` from a runner without importing the runner.

The runners parse sys.argv and open MCP clients at import time -- they are built to be run, not
imported -- so the test extracts just the resume logic and the module-level constants it needs.
"""
import ast
import os
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[1] / "harness" / "runner_sdk.py"
WANT = {"_resume", "_is_dead", "_DEAD_ON_RESUME"}


def make_ns(out_dir, monkeypatch):
    """The extracted namespace: `_resume`, `_is_dead` and `_DEAD_ON_RESUME`."""
    monkeypatch.setenv("OUT_DIR", str(out_dir))
    tree = ast.parse(RUNNER.read_text())
    keep = [n for n in tree.body
            if (isinstance(n, (ast.FunctionDef,)) and n.name in WANT)
            or (isinstance(n, ast.Assign)
                and any(getattr(t, "id", None) in WANT for t in n.targets))]
    ns = {"os": os, "json": __import__("json"),
          "_out": lambda name: os.path.join(os.environ["OUT_DIR"], "tasks", name)}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(RUNNER), "exec"), ns)
    return ns


def make_resume(out_dir, monkeypatch):
    return make_ns(out_dir, monkeypatch)["_resume"]

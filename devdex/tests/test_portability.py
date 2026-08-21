"""Guards for the six defects found in review of the devdex port.

Each test here failed before its fix. They are cheap, need no credentials and no network, and
they cover the class of bug the rest of the suite cannot: not "is the arithmetic right" but
"does this code still run, and does it read the files it claims to read". Every one of these
was invisible to the existing tests because nothing imported the broken module or executed the
broken CI step.
"""
import ast
import glob
import json
import re
import sys
from pathlib import Path

import pytest

DEVDEX = Path(__file__).resolve().parents[1]           # devdex/
ROOT = DEVDEX.parent                                   # repo root
WORKFLOW = ROOT / ".github" / "workflows" / "devdex-eval.yml"

sys.path.insert(0, str(ROOT))


def devdex_sources():
    """Every python file in the devdex package -- harness, scorer, gt builders, tests."""
    return (sorted(glob.glob(str(DEVDEX / "harness/*.py")))
            + sorted(glob.glob(str(DEVDEX / "scorer/*.py")))
            + sorted(glob.glob(str(DEVDEX / "gt/builders/*.py")))
            + sorted(glob.glob(str(DEVDEX / "tests/*.py")))
            + [str(DEVDEX / "run_eval.py")])


# ---- 1. the declared floor must actually parse -------------------------------------------
# A devdex file once carried a backslash inside an f-string expression, legal only from 3.12
# (PEP 701), while pyproject declares >=3.11 -- so the file could not be imported at all, and
# `checks` stayed green purely because nothing imported it.

def test_every_devdex_file_parses_on_the_declared_python_floor():
    floor = re.search(r'requires-python\s*=\s*"[^0-9]*(\d+)\.(\d+)',
                      (ROOT / "pyproject.toml").read_text())
    assert floor, "pyproject.toml must declare requires-python"
    target = (int(floor.group(1)), int(floor.group(2)))

    broken = []
    for f in devdex_sources():
        try:
            ast.parse(Path(f).read_text(), feature_version=target)
        except SyntaxError as e:
            broken.append(f"{Path(f).relative_to(ROOT)}:{e.lineno}: {e.msg}")
    assert not broken, f"does not parse on python{target[0]}.{target[1]}: " + "; ".join(broken)


def test_workflow_python_is_not_older_than_the_declared_floor():
    """A workflow pinned below requires-python makes the floor a fiction."""
    pinned = set(re.findall(r"python-version:\s*'([^']+)'", WORKFLOW.read_text()))
    floor = re.search(r'requires-python\s*=\s*"[^0-9]*(\d+)\.(\d+)',
                      (ROOT / "pyproject.toml").read_text())
    fl = (int(floor.group(1)), int(floor.group(2)))
    for p in pinned:
        parts = tuple(int(x) for x in p.split(".")[:2])
        assert parts >= fl, f"workflow pins {p}, below requires-python {fl[0]}.{fl[1]}"


# ---- 3/4. the CI job must be able to install, and its report must be able to fail ---------

def test_workflow_installs_with_something_that_exists():
    t = WORKFLOW.read_text()
    assert "requirements.txt" not in t, "no requirements.txt in this repo; use uv sync"
    assert "uv sync" in t


def test_report_step_can_match_its_own_run_directory(tmp_path):
    """The matrix job passed no --label, so the run dir had no pass segment and the reporter's
    glob could never match it. Then the label was fixed for `repo` only and the fix track stayed
    invisible, because the previous version of this test substituted `repo` and nothing else.
    Drive the REAL run_files() over EVERY track the workflow offers."""
    from devdex.scorer.report_metrics import TRACKS, run_files

    t = WORKFLOW.read_text()
    m = re.search(r'--label "([^"]+)"', t)
    assert m, "matrix job must pass an explicit --label"
    label_tpl = m.group(1)
    p = re.search(r"--pass (\S+)", t)
    assert p, "report step must pass --pass"
    pass_label = p.group(1)
    assert pass_label != '"$(ls', "--pass must not be derived by parsing ls output"

    tracks = re.search(r"track:.*?description:\s*'([^']+)'", t, re.S)
    assert tracks, "matrix job must declare which tracks it offers"
    offered = [x.strip() for x in tracks.group(1).split("|") if x.strip()]
    assert offered, "could not parse the track options"

    token_of = {key: tok for key, (tok, _family) in TRACKS.items()}
    for track in offered:
        label = (label_tpl.replace("${{ inputs.track }}", track)
                          .replace("${{ inputs.arm }}", "fc-mcp"))
        d = tmp_path / f"20260810T120000-{label}" / "run1" / "tasks"
        d.mkdir(parents=True)
        (d / "cell.json").write_text("[]")

    for track in offered:
        # the reporter addresses this cell by its TRACKS token, which need not equal the CLI word
        tok = token_of.get(track) or token_of.get(f"{track}-short")
        assert tok, f"workflow offers --track {track} but no TRACKS row claims it"
        found = run_files(str(tmp_path), pass_label, tok, "fc-mcp", ".json")
        assert found, (f"a CI run of --track {track} writes a directory the report step "
                       f"cannot find (token {tok!r})")


def test_report_step_does_not_swallow_its_exit_code():
    """report_metrics exits non-zero on an invariant violation. `|| true` discarded that."""
    line = next(l for l in WORKFLOW.read_text().splitlines() if "report_metrics.py" in l)
    assert "|| true" not in line, "the invariant check is the reason this step exists"


def test_bare_run_eval_is_discoverable_by_the_documented_report_command(tmp_path, monkeypatch):
    """README's 'Run it' section is two lines: `run_eval.py` with no --label, then
    `report_metrics.py --pass p1`. The CI workflow hit exactly this class of bug (a run
    directory with no pass token the reporter's glob could match) and was patched with an
    explicit --label -- see test_report_step_can_match_its_own_run_directory above. That fix
    never reached run_cell's own default, so the plain two-line quickstart silently produced a
    run `--pass p1` could never find. Drive the real run_cell(), subprocess mocked out, and
    confirm the reporter's real run_files() sees it."""
    sys.path.insert(0, str(DEVDEX))
    import devdex.scorer.suite as suite
    from devdex.scorer.report_metrics import TRACKS, run_files

    def fake_call(argv, cwd, env):
        pdir = Path(env["OUT_DIR"])
        (pdir / "tasks").mkdir(parents=True, exist_ok=True)
        (pdir / "tasks" / "p1_repo_no-tool.json").write_text("[]")
        return 0

    monkeypatch.setattr(suite.subprocess, "call", fake_call)
    suite.run_cell(track="repo", arm="no-tool", driver="opus", n_runs=1,
                    concurrency=1, out_root=str(tmp_path))

    tok, _family = TRACKS["repo"]
    found = run_files(str(tmp_path), "p1", tok, "no-tool", ".json")
    assert found, ("`run_eval.py --track repo --arm no-tool --driver opus` with no --label "
                   "wrote a run directory `report_metrics.py --pass p1` cannot find")


# ---- 5. an alias must resolve to the file its name promises ------------------------------

def test_dataset_aliases_resolve_and_match_the_version_in_their_name():
    from devdex.scorer.suite import DATASET_ALIASES, DATASETS, TRACKS
    for alias, fname in DATASET_ALIASES.items():
        assert (DATASETS / fname).exists(), f"alias {alias} -> missing file {fname}"
        v = re.search(r"v([\d.]+)$", alias)
        if v:                      # docs-v3.1 must not resolve to docs_v3.0.1
            want = v.group(1).replace(".", r"\.")
            assert re.search(rf"_v{want}(\.|_)", fname), \
                f"alias {alias} resolves to {fname}, a different version"
    for track, fname in TRACKS.items():
        assert (DATASETS / fname).exists(), f"track {track} -> missing default {fname}"


# ---- 6. scoring must not silently zero items it has no ground truth for -------------------

def test_docs_coverage_guard_rejects_items_absent_from_the_scoring_gt():
    from devdex.scorer.report_metrics import check_docs_coverage
    meta = {"a": object(), "b": object()}
    check_docs_coverage([{"qid": "a"}, {"qid": "b"}], meta, "docs/fc-mcp")   # full coverage: ok
    with pytest.raises(SystemExit) as e:
        check_docs_coverage([{"qid": "a"}, {"qid": "zzz", "gt_file": "docs_v2.1.0.jsonl"}],  # noqa: E501
                            meta, "docs/fc-mcp")
    assert "absent from" in str(e.value) and "zzz" in str(e.value)


def test_docs_scoring_gt_is_a_superset_of_every_docs_dataset_it_may_score():
    """Rescoring a v3.x run against the widest v3 is intended -- they share one item set and
    differ only in how many pages count as gold. This asserts that stays true."""
    from devdex.scorer.report_metrics import DOCS_SCORING_GT
    from devdex.scorer.suite import DATASETS
    ids = lambda p: {json.loads(l)["id"] for l in open(DATASETS / p)}  # noqa: E731
    scoring = ids(DOCS_SCORING_GT)
    # Glob EVERY docs dataset present, not a hardcoded version family: the public
    # release ships docs_public.jsonl and no v3 files at all, and a test that
    # silently matched nothing would pass while asserting nothing.
    present = sorted(glob.glob(str(DATASETS / "docs_*.jsonl")))
    assert present, "no docs dataset found to check"
    for f in present:
        assert ids(Path(f).name) == scoring, (
            f"{Path(f).name} has a different item set than {DOCS_SCORING_GT}; scoring one "
            f"against the other would zero the unmatched items")

def test_every_harness_module_imports_as_part_of_the_package():
    """Installed, `import devdex.harness.<mod>` must work for every module.

    control.py and preflight.py used bare `import provenance` / `import arms`, which resolve only
    when devdex/harness is on sys.path -- true for the script entry points, false for a wheel. So
    `import devdex.harness.control` raised ModuleNotFoundError in any installed copy while every
    documented command still worked, which is why nothing caught it.

    Run in a subprocess with a clean sys.path: importing here would be satisfied by the path
    inserts other tests in this suite already performed.
    """
    import subprocess
    mods = sorted(f"devdex.harness.{p.stem}" for p in (ROOT / "devdex" / "harness").glob("*.py")
                  if p.stem not in ("__init__", "runner_sdk"))   # runner_sdk runs setup on import
    assert mods, "no harness modules found"
    failed = []
    for m in mods:
        r = subprocess.run([sys.executable, "-c", f"import {m}"], cwd=ROOT,
                           capture_output=True, text=True)
        if r.returncode:
            failed.append(f"{m}: {r.stderr.strip().splitlines()[-1]}")
    assert not failed, "not importable as package modules:\n  " + "\n  ".join(failed)

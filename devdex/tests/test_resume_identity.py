"""Resume cannot relabel measurements from a different or unknown experiment."""
import ast
import inspect
import json
import sys
from types import SimpleNamespace

import pytest

from devdex.tests._resume_shim import RUNNER, make_resume


def saved(tmp_path, rows):
    path = tmp_path / "tasks/p1_repo_external.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(rows))
    return path


def resume(tmp_path, monkeypatch):
    callback = make_resume(tmp_path, monkeypatch)
    # Also run against the original four-argument function, so baseline failures show
    # incompatible rows being accepted rather than just a changed function signature.
    identity = ({"gt_sha": "current-gt", "model": "current-model"}
                if "gt_sha" in inspect.signature(callback).parameters else {})
    return callback("p1", "repo", "external", [{"id": "a"}], **identity)


@pytest.mark.parametrize("old", [
    {"gt_sha": "old-gt", "model": "current-model"},
    {"gt_sha": "current-gt", "model": "old-model"},
    {"model": "current-model"},
    {"gt_sha": "current-gt"},
])
def test_incompatible_or_unstamped_measurement_is_rejected_without_overwrite(old, tmp_path,
                                                                          monkeypatch, capsys):
    path = saved(tmp_path, [{"qid": "a", **old}])
    original = path.read_bytes()
    with pytest.raises(SystemExit) as error:
        resume(tmp_path, monkeypatch)
    assert error.value.code == 78
    assert "cannot resume" in capsys.readouterr().err
    assert path.read_bytes() == original


def test_identical_dataset_and_model_keep_finished_items(tmp_path, monkeypatch):
    saved(tmp_path, [{"qid": "a", "gt_sha": "current-gt", "model": "current-model"}])
    keep, todo = resume(tmp_path, monkeypatch)
    assert [r["qid"] for r in keep] == ["a"]
    assert todo == []


def test_dead_rows_are_retried_under_current_identity(tmp_path, monkeypatch):
    saved(tmp_path, [{"qid": "a", "error": "timeout", "gt_sha": "old-gt"}])
    keep, todo = resume(tmp_path, monkeypatch)
    assert keep == []
    assert todo == [{"id": "a"}]


def test_keep_dead_does_not_bypass_identity_check(tmp_path, monkeypatch):
    monkeypatch.setenv("RESUME_KEEP_DEAD", "1")
    saved(tmp_path, [{"qid": "a", "error": "timeout", "gt_sha": "old-gt"}])
    with pytest.raises(SystemExit) as error:
        resume(tmp_path, monkeypatch)
    assert error.value.code == 78


def test_fresh_run_needs_no_saved_identity(tmp_path, monkeypatch):
    assert resume(tmp_path, monkeypatch) == ([], [{"id": "a"}])


def test_real_runner_supplies_current_dataset_and_model_to_resume():
    tree = ast.parse(RUNNER.read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_resume"]
    assert len(calls) == 1
    arguments = {keyword.arg: ast.unparse(keyword.value) for keyword in calls[0].keywords}
    assert arguments.get("gt_sha") == "GT_SHA"
    assert arguments.get("model") == "MODEL"


def test_suite_does_not_rescore_or_relabel_rejected_saved_cell(tmp_path, monkeypatch):
    from devdex.scorer import suite

    run = tmp_path / "saved-run"
    cell = run / "run1/tasks/p1_repo_external.json"
    cell.parent.mkdir(parents=True)
    cell.write_text(json.dumps([{"qid": "a", "gt_sha": "old", "model": "old"}]))
    result = run / "results.json"
    result.write_text('{"model": "old", "dataset": "old.jsonl"}')
    before = (cell.read_bytes(), result.read_bytes())
    monkeypatch.setattr(suite.subprocess, "call", lambda *args, **kwargs: 78)
    with pytest.raises(SystemExit) as error:
        suite.run_cell(track="repo", arm="external", driver="opus", n_runs=1, concurrency=1,
                       out_root=str(tmp_path), resume_dir=str(run), model_override="new")
    assert error.value.code == 78
    assert (cell.read_bytes(), result.read_bytes()) == before


def test_register_arm_stops_all_tracks_and_preserves_resume_error(monkeypatch):
    import subprocess
    from benchmark import register_arm

    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-test-key")
    monkeypatch.setenv("DEVDEX_EXT_MCP_URL", "https://example.test/mcp")
    calls = []
    def call(*args, **kwargs):
        calls.append(args)
        return 78
    monkeypatch.setattr(subprocess, "call", call)
    assert register_arm.run_agent_mode("example", track="all") == 78
    assert len(calls) == 1


def test_public_cli_never_reports_rejected_saved_records(monkeypatch):
    from benchmark import run_benchmark as public

    monkeypatch.setitem(sys.modules, "register_arm", SimpleNamespace(run_agent_mode=lambda **kw: 78))
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--name", "example", "--mcp-url",
                                     "https://example.test/mcp"])
    called = []
    monkeypatch.setattr(public, "report", lambda *args: called.append(args))
    monkeypatch.setenv("DEVDEX_EXT_MCP_URL", "https://example.test/mcp")
    assert public.main() == 78
    assert called == []

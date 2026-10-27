"""Fresh runs cannot inherit old cells; automatic resume identifies one exact label."""
import json
from pathlib import Path

import pytest

from devdex.scorer import suite


@pytest.fixture
def runner(tmp_path, monkeypatch):
    outputs = []
    monkeypatch.setattr(suite.time, "strftime", lambda _: "20261026T120000")
    def fake_call(argv, cwd, env):
        directory = Path(env["OUT_DIR"])
        outputs.append(directory.parent)
        tasks = directory / "tasks"
        tasks.mkdir(exist_ok=True)
        (tasks / "p1_repo_no-tool.json").write_text(json.dumps([
            {"qid": f"q{len(outputs)}", "gold_accept": ["gold"], "citations": [],
             "committed": False}]))
        return 0
    monkeypatch.setattr(suite.subprocess, "call", fake_call)
    def run(label="example", **options):
        return suite.run_cell("repo", "no-tool", "opus", n_runs=1, concurrency=1,
                              out_root=str(tmp_path), label=label, **options)
    return run, outputs


def test_fresh_same_second_runs_reserve_distinct_directories(tmp_path, runner):
    run, outputs = runner
    run()
    first = outputs[0] / "run1/tasks/p1_repo_no-tool.json"
    original = first.read_bytes()
    run()
    assert len(set(outputs)) == 2
    assert first.read_bytes() == original
    assert len(list(tmp_path.glob("*/results.json"))) == 2


def test_auto_resume_still_selects_newest_same_label_after_collision(runner):
    run, outputs = runner
    run()
    run()
    latest = outputs[-1]
    run(auto_resume=True)
    assert outputs[-1] == latest
    assert outputs[0] != latest


def test_auto_resume_does_not_match_another_labels_suffix(tmp_path, runner):
    unrelated = tmp_path / "20261025T120000-team-example"
    unrelated.mkdir()
    run, outputs = runner
    run(auto_resume=True)
    assert outputs[-1] != unrelated


def test_glob_characters_in_a_label_are_literal(tmp_path, runner):
    unrelated = tmp_path / "20261025T120000-other"
    unrelated.mkdir()
    run, outputs = runner
    run(label="*", auto_resume=True)
    assert outputs[-1] != unrelated
    literal = outputs[-1]
    run(label="*", auto_resume=True)
    assert outputs[-1] == literal


@pytest.mark.parametrize("label", ["../outside", "../../../outside", "one/two", "one\\two", ".", "..", "bad\0name"])
def test_labels_are_single_directory_components_and_rejected_before_calls(label, runner):
    run, outputs = runner
    with pytest.raises(SystemExit, match="label"):
        run(label=label)
    assert outputs == []

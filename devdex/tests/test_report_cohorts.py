"""Exercise the public reporter on saved synthetic runs, with no provider calls."""
import hashlib
import json

from benchmark import run_benchmark as report


def save(root, stamp, engine="example", hit=True, run=1):
    directory = root / "devdex" / "runs" / f"{stamp}-ext-{engine}-repo"
    tasks = directory / f"run{run}" / "tasks"
    tasks.mkdir(parents=True, exist_ok=True)
    row = {"qid": "synthetic-1", "gold_accept": ["owner/project"],
           "citations": ["owner/project"] if hit else [], "search_log": []}
    (tasks / "records.json").write_text(json.dumps([row]))
    return directory


def test_latest_run_does_not_pool_old_smoke_results(tmp_path, monkeypatch):
    save(tmp_path, "20261001T120000", hit=True)
    save(tmp_path, "20261002T120000", hit=False)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    output = tmp_path / "report.json"
    assert report.report("example", output) == 0
    result = json.loads(output.read_text())
    assert result["tracks"]["repo"]["recall@10"] == 0.0
    source = tmp_path / "devdex/runs/20261002T120000-ext-example-repo/run1/tasks/records.json"
    assert result["inputs"]["repo"] == {
        "run": "20261002T120000-ext-example-repo", "records": 1,
        "files": {"run1/tasks/records.json": hashlib.sha256(source.read_bytes()).hexdigest()},
    }


def test_engine_label_is_literal_not_a_glob(tmp_path, monkeypatch):
    save(tmp_path, "20261001T120000", engine="other", hit=True)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    assert report.report("*") == 1


def test_latest_empty_run_does_not_fall_back_to_stale_success(tmp_path, monkeypatch):
    save(tmp_path, "20261001T120000", hit=True)
    latest = tmp_path / "devdex" / "runs" / "20261002T120000-ext-example-repo"
    latest.mkdir()
    monkeypatch.setattr(report, "ROOT", tmp_path)
    assert report.report("example") == 1


def test_repetitions_within_selected_run_are_preserved(tmp_path, monkeypatch):
    save(tmp_path, "20261001T120000", hit=True, run=1)
    save(tmp_path, "20261001T120000", hit=False, run=2)
    monkeypatch.setattr(report, "ROOT", tmp_path)
    output = tmp_path / "report.json"
    assert report.report("example", output) == 0
    assert json.loads(output.read_text())["tracks"]["repo"]["recall@10"] == 0.5

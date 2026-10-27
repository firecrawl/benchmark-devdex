"""Reports must not call different query sets repeated measurements of the same items."""
import json

import pytest

from devdex.scorer import report_metrics


def row(qid, hit=True):
    return {"qid": qid, "gold_accept": ["gold"], "citations": ["gold"] if hit else [],
            "committed": hit, "search_log": [{"results": [{"rank": 0, "ref": "gold"}]}]}


def save(root, records, number, public):
    label = "ext-example-repo" if public else "p1-opus-repo-fc-mcp"
    prefix = root / "devdex/runs" if public else root
    tasks = prefix / f"20261026T120000-{label}" / f"run{number}/tasks"
    tasks.mkdir(parents=True)
    (tasks / f"p{number}_repo_arm.json").write_text(json.dumps(records))


def report(root, public, monkeypatch):
    if public:
        from benchmark import run_benchmark
        monkeypatch.setattr(run_benchmark, "ROOT", root)
        return run_benchmark.report("example", root / "results.json")
    return report_metrics.build(str(root), "p1")


@pytest.mark.parametrize("public", [False, True])
@pytest.mark.parametrize("second", [[row("a")], [row("c"), row("d")], []])
def test_different_or_incomplete_repeat_cannot_publish_a_pooled_score(tmp_path, monkeypatch,
                                                                   public, second):
    save(tmp_path, [row("a"), row("b", False)], 1, public)
    save(tmp_path, second, 2, public)
    with pytest.raises(SystemExit, match="cohort"):
        report(tmp_path, public, monkeypatch)
    assert not (tmp_path / "results.json").exists()


@pytest.mark.parametrize("public", [False, True])
def test_duplicate_query_within_one_pass_is_rejected(tmp_path, monkeypatch, public):
    save(tmp_path, [row("a"), row("a")], 1, public)
    with pytest.raises(SystemExit, match="duplicate"):
        report(tmp_path, public, monkeypatch)


@pytest.mark.parametrize("public", [False, True])
def test_reordered_equal_cohort_and_repeated_queries_across_passes_are_valid(tmp_path, monkeypatch,
                                                                          public):
    save(tmp_path, [row("a"), row("b", False)], 1, public)
    save(tmp_path, [row("b", False), row("a")], 2, public)
    result = report(tmp_path, public, monkeypatch)
    if public:
        assert result == 0
        assert json.loads((tmp_path / "results.json").read_text())["tracks"]["repo"]["recall@10"] == 0.5
    else:
        cells, failures = result
        assert failures == []
        assert cells["repo"]["fc-mcp"]["n"] == 2
        assert cells["repo"]["fc-mcp"]["passes"] == 2
        assert cells["repo"]["fc-mcp"]["recall"] == 0.5


def test_one_smoke_subset_pass_remains_reportable(tmp_path, monkeypatch):
    save(tmp_path, [row("smoke-only")], 1, True)
    assert report(tmp_path, True, monkeypatch) == 0


@pytest.mark.parametrize("public", [False, True])
def test_missing_query_identity_is_rejected(tmp_path, monkeypatch, public):
    save(tmp_path, [row(None)], 1, public)
    with pytest.raises(SystemExit, match="qid"):
        report(tmp_path, public, monkeypatch)

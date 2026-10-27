"""Unavailable docs ground truth is an evaluation error, never a retrieval miss."""
import json

import pytest

from devdex.scorer import report_metrics, suite


def record(qid="unknown"):
    return {"qid": qid, "sources": ["https://docs.example.test/guide"], "committed": True,
            "search_log": [{"results": []}], "gt_file": "original.jsonl"}


def test_empty_metadata_cannot_disable_coverage_validation():
    with pytest.raises(SystemExit, match="absent from"):
        report_metrics.check_docs_coverage([record()], {}, "docs/example")


def test_no_records_needs_no_ground_truth():
    report_metrics.check_docs_coverage([], {}, "docs/example")


@pytest.mark.parametrize("meta", [{}, {"other": {}}])
def test_direct_pass_scoring_rejects_missing_qids(meta, monkeypatch):
    monkeypatch.setattr(suite, "_DOCS_META", meta)
    with pytest.raises(SystemExit, match="absent from"):
        suite.score_pass([record()], "docs")


def test_public_report_rejects_missing_qids(tmp_path, monkeypatch):
    from benchmark import run_benchmark as public

    tasks = tmp_path / "devdex/runs/20261026T120000-ext-example-docs/run1/tasks"
    tasks.mkdir(parents=True)
    (tasks / "p1_docs_external.json").write_text(json.dumps([record()]))
    monkeypatch.setattr(public, "ROOT", tmp_path)
    monkeypatch.setattr(suite, "DATASETS", tmp_path / "missing-gt")
    monkeypatch.setattr(suite, "_DOCS_META", None)
    with pytest.raises(SystemExit, match="absent from"):
        public.report("example", tmp_path / "results.json")
    assert not (tmp_path / "results.json").exists()


def test_existing_report_rejects_missing_gt_file(tmp_path, monkeypatch):
    tasks = tmp_path / "20261026T120000-p1-opus-docs-fc-mcp/run1/tasks"
    tasks.mkdir(parents=True)
    (tasks / "p1_docs_fc-mcp.json").write_text(json.dumps([record()]))
    monkeypatch.setattr(suite, "DATASETS", tmp_path / "missing-gt")
    monkeypatch.setattr(suite, "_DOCS_META", None)
    monkeypatch.setattr(suite, "TRACKS", dict(suite.TRACKS))
    with pytest.raises(SystemExit, match="absent from"):
        report_metrics.build(str(tmp_path), "p1")


def test_available_gold_with_no_retrieval_hit_still_scores_zero(monkeypatch):
    monkeypatch.setattr(suite, "_DOCS_META", {"known": {
        "repo_files": set(), "site": "docs.example.test", "page_stems": {"guide"},
        "memory_answerable": False}})
    metrics = suite.score_pass([record("known")], "docs")
    assert metrics["n"] == 1
    assert metrics["precision"] == 0
    assert metrics["doc_found@10"] == 0

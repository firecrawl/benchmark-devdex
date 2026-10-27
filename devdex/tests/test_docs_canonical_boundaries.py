"""Canonical docs URLs must stay on the declared host and within its base path."""
import json

import pytest

from devdex.scorer import report_metrics, suite


def metadata(site="docs.example.com"):
    return {"repo_files": set(), "site": site, "page_stems": {"guide"}}


def record(url):
    return {"qid": "docs-1", "sources": [url], "committed": True,
            "search_log": [{"results": [{"rank": 0, "url": url}]}]}


@pytest.mark.parametrize("url", [
    "https://docs.example.com.evil.test/guide",
    "https://docs.example.com-extra.test/guide",
    "https://docs.example.com@evil.test/guide",
    "https://docs.example.com:8443/guide",
    "https://docs.example.com?redirect=/guide",
    "https://docs.example.com#redirect/guide",
])
def test_lookalike_authorities_and_nonpath_suffixes_are_not_canonical(url):
    assert suite._docs_rank(record(url), metadata()) is None


@pytest.mark.parametrize("url", [
    "https://example.com/docs-other/guide",
    "https://example.com/docs2/guide",
    "https://example.com/docs?redirect=/guide",
    "https://example.com/docs#redirect/guide",
])
def test_base_path_must_end_at_a_path_segment(url):
    assert suite._docs_rank(record(url), metadata("example.com/docs")) is None


@pytest.mark.parametrize("url", [
    "https://docs.example.com/guide",
    "http://docs.example.com/guide",
    "https://www.docs.example.com/guide",
    "https://DOCS.EXAMPLE.COM/nested/guide.md?query=yes#section",
])
def test_existing_canonical_url_forms_still_match(url):
    assert suite._docs_rank(record(url), metadata()) == 1


def test_declared_base_path_and_page_stem_are_preserved():
    assert suite._docs_rank(record("https://example.com/docs/nested/guide.html"),
                            metadata("example.com/docs")) == 1
    assert suite._docs_rank(record("https://docs.example.com/unrelated"), metadata()) is None


def test_published_report_does_not_credit_a_lookalike_site(tmp_path, monkeypatch):
    # Exercise the real canonical-source loader, gold probe, pool match and scorer together.
    gt = tmp_path / "gt"
    gt.mkdir()
    (gt / report_metrics.DOCS_SCORING_GT).write_text(json.dumps({
        "id": "docs-1", "canonical_sources": {
            "site": "https://docs.example.com", "page_stems": ["guide"]}}) + "\n")
    monkeypatch.setattr(suite, "DATASETS", gt)
    monkeypatch.setattr(suite, "_DOCS_META", None)
    monkeypatch.setattr(suite, "TRACKS", dict(suite.TRACKS))
    tasks = tmp_path / "20260101T000000-p1-opus-docs-fc-mcp" / "run1" / "tasks"
    tasks.mkdir(parents=True)
    (tasks / "p1_docs_fc-mcp.json").write_text(json.dumps([
        record("https://docs.example.com/guide"),
        record("https://docs.example.com.evil.test/guide")]))

    cells, failures = report_metrics.build(str(tmp_path), "p1")
    metrics = cells["docs"]["fc-mcp"]
    assert not failures
    for metric in ("precision", "recall", "amrr", "groundedness", "pool_hit"):
        assert metrics[metric] == 0.5, metric

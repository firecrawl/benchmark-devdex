"""Schema and integrity tests for the shipped ground truth.

These read the committed .jsonl files and nothing else -- no credentials, no network. They guard
the properties the scorer assumes: unique ids, non-empty golds, the two-wide fix pair, and the
short-query contract (compressed, cause clause gone, gold number never leaked into the query).
"""
import json
import re
from pathlib import Path

import pytest

DATASETS = Path(__file__).resolve().parents[1] / "gt"

# The filenames come from suite.TRACKS -- the same mapping run_eval.py and the scorer read -- so
# these tests always target the dataset that is actually shipped. Hardcoding version-stamped names
# here made every test in this file skip silently once the shipped files were renamed, which left
# the committed ground truth untested by the CI job whose stated purpose is to catch a corrupted
# ground-truth file before any number is published.
from devdex.scorer import suite                                            # noqa: E402

REPO = suite.TRACKS["repo"]
FIX_SHORT = suite.TRACKS["fix"]
DOCS = suite.TRACKS["docs"]


def load(name):
    path = DATASETS / name
    # A MISSING SHIPPED DATASET IS A FAILURE, NOT A SKIP. `pytest.skip` here is what let 21 tests
    # disappear from the run without turning CI red.
    assert path.exists(), f"{name} is missing from {DATASETS} (suite.TRACKS points at it)"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("name", [REPO, FIX_SHORT, DOCS])
class TestCommonShape:
    def test_ids_are_unique(self, name):
        rows = load(name)
        ids = [r["id"] for r in rows]
        assert len(set(ids)) == len(ids)

    def test_every_item_has_a_non_empty_query(self, name):
        assert all(str(r.get("query") or "").strip() for r in load(name))

    def test_no_item_is_missing_its_gold(self, name):
        for r in load(name):
            gold = r.get("artifact") or (r.get("canonical_sources") or {}).get("repo_files")
            assert gold, f"{r['id']} has no gold"


class TestRepoTrack:
    def test_query_never_names_the_repository_being_sought(self):
        # The repo track is a capability description. If the query contains the repo's own name
        # the item degenerates into a lexical lookup and stops measuring conceptual retrieval.
        leaked = []
        for r in load(REPO):
            for a in (r.get("artifact") or []):
                tail = str(a.get("ref", "")).split("/")[-1].lower()
                if len(tail) > 3 and tail in re.sub(r"[-_ ]", "", r["query"].lower()):
                    leaked.append(r["id"])
        assert not leaked, f"{len(leaked)} repo queries leak the gold repo name: {leaked[:5]}"

    def test_exactly_one_gold_per_item(self):
        assert all(len(r.get("artifact") or []) == 1 for r in load(REPO))


class TestFixTrack:
    def test_gold_is_the_issue_and_pull_request_pair(self):
        for r in load(FIX_SHORT):
            kinds = sorted(a.get("type") for a in (r.get("artifact") or []))
            assert kinds == ["issue", "pull_request"], f"{r['id']}: {kinds}"

    def test_every_item_keeps_the_diagnosed_query_it_was_derived_from(self):
        """The diagnosed-phrasing set is not shipped, so `query_long` on each row is the only
        record of what the compression started from. Without it the A/B is unauditable."""
        for r in load(FIX_SHORT):
            assert r.get("query_long"), f"{r['id']}: no query_long"
            assert len(r["query"].split()) < len(r["query_long"].split()), \
                f"{r['id']}: compressed query is not shorter"

    def test_short_queries_are_actually_short(self):
        rows = load(FIX_SHORT)
        assert all(len(r["query"].split()) <= 9 for r in rows)
        median = sorted(len(r["query"].split()) for r in rows)[len(rows) // 2]
        assert median <= 7

    def test_short_queries_dropped_the_cause_clause(self):
        # A query that explains WHY the bug happens hands the engine the answer's own
        # explanation -- knowledge the searcher does not have yet.
        cause = re.compile(r"\b(causing|because|due to|instead of|unlike|rather than|"
                           r"resulting in|leading to|whereas)\b", re.I)
        offenders = [r["id"] for r in load(FIX_SHORT) if cause.search(r["query"])]
        assert not offenders, offenders[:5]

    def test_short_queries_never_leak_the_gold_number(self):
        for r in load(FIX_SHORT):
            for a in r["artifact"]:
                num = str(a["ref"]).split("#")[-1]
                if num.isdigit():
                    assert num not in r["query"], f"{r['id']} leaks #{num}"

    def test_original_query_is_retained_for_a_paired_diff(self):
        assert all(r.get("query_long") for r in load(FIX_SHORT))


class TestDocsTrack:
    def test_no_verbatim_source_documentation_is_redistributed(self):
        """The shipped docs set must not carry the source passage itself.

        The scorer matches on canonical_sources (repo file, site, page stem) and never reads a
        passage, so shipping one only redistributes third-party documentation -- 30k words across
        37 projects with differing licences, in an MIT repo. provenance.source carries the exact
        blob URL, so the passage stays retrievable without being republished here.
        """
        offenders = [r["id"] for r in load(DOCS) if str(r.get("gold_passage") or "").strip()]
        assert not offenders, (f"{len(offenders)} docs items carry gold_passage; "
                               f"the scorer does not read it: {offenders[:5]}")

    def test_every_item_has_an_expected_answer(self):
        assert all(str(r.get("expected_answer") or "").strip() for r in load(DOCS))

    def test_canonical_sources_carry_at_least_one_repo_file(self):
        for r in load(DOCS):
            assert (r.get("canonical_sources") or {}).get("repo_files"), r["id"]

    def test_sibling_pages_never_displace_the_original_gold(self):
        # v3.1 adds pages carrying the same answer sentence. The chunked original must stay
        # first, or a rescore against an older run would silently change which page is gold.
        for r in load(DOCS):
            cs = r.get("canonical_sources") or {}
            added = set(cs.get("siblings_added") or [])
            if added:
                first = cs["repo_files"][0].split(":", 1)[-1]
                assert first not in added, r["id"]

    def test_no_gold_file_is_a_non_english_mirror(self):
        # A translated copy of the same page is a mirror, not an independent source.
        langs = {"de", "es", "fr", "tr", "zh", "ja", "ko", "pt", "ru", "it", "zh-cn", "zh-tw"}
        for r in load(DOCS):
            for f in (r.get("canonical_sources") or {}).get("repo_files") or []:
                segs = {s.lower() for s in f.split(":", 1)[-1].split("/")}
                assert not (segs & langs), f"{r['id']}: {f}"

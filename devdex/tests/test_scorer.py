"""Tests for the devdex scorer: metric arithmetic, invariants, and the depth cap.

Every case is a synthetic record, so these run with no datasets, no credentials and no network.
The point is to pin the definitions that move the published table -- if someone changes what
precision means, a test here fails before a number does.
"""
import json

import pytest

from devdex.scorer.report_metrics import (
    ARM_NAMES, ARMS, CONTROLS, TRACK_NAMES, TRACKS, arm_name, check_cell, cite_ref, engine_slots,
    golds, norm, run_files, score_records, submitted, track_name,
)


def rec(citations, engine, gold, committed=True, **extra):
    """One repo/fix record. `engine` is a flat list of refs the search returned, ranked in order."""
    return {
        "citations": list(citations),
        "gold_accept": list(gold),
        "committed": committed,
        "search_log": [{"results": [{"rank": i, "ref": r} for i, r in enumerate(engine)]}],
        **extra,
    }


def gold_test(item, record):
    return item in golds(record)


def score(rows):
    return score_records(rows, "fix", gold_test)


class TestNormalisation:
    def test_norm_strips_case_space_and_trailing_slash(self):
        assert norm("  HTTPS://Example.com/Path/  ") == "https://example.com/path"

    def test_cite_ref_extracts_ref_from_a_dict_shaped_citation(self):
        # Agents sometimes submit {"id": .., "reason": ..} instead of a bare ref.
        assert cite_ref('{"id": "owner/repo#12", "reason": "fixes it"}') == "owner/repo#12"

    def test_cite_ref_passes_through_a_bare_ref(self):
        assert cite_ref("owner/repo#12") == "owner/repo#12"


class TestEngineSlots:
    def test_collects_ref_and_url_from_every_search(self):
        r = {"search_log": [{"results": [{"rank": 0, "ref": "A", "url": "http://a"}]},
                            {"results": [{"rank": 0, "ref": "B"}]}]}
        assert engine_slots(r) == {"a", "http://a", "b"}

    def test_applies_the_depth_cap(self):
        results = [{"rank": i, "ref": f"r{i}"} for i in range(19)]
        slots = engine_slots({"search_log": [{"results": results}]}, cap=10)
        assert "r9" in slots and "r10" not in slots
        assert len(slots) == 10

    def test_falls_back_to_pool_prov_when_there_is_no_search_log(self):
        # Claude's server-side WebSearch never reaches the MCP hook, so its refs arrive only as
        # stream-harvested provenance. Reading search_log alone scored that arm 0.000.
        r = {"pool_prov": [["http://a", "search_direct"], ["http://b", "reader"]]}
        assert engine_slots(r) == {"http://a"}

    def test_does_not_fall_back_when_a_search_log_exists_but_is_empty_at_depth(self):
        # An arm that searched and returned only results past the cap has genuinely surfaced
        # nothing; borrowing pool_prov there would credit reads as retrieval.
        r = {"search_log": [{"results": [{"rank": 12, "ref": "deep"}]}],
             "pool_prov": [["http://a", "search_direct"]]}
        assert engine_slots(r, cap=10) == set()


class TestPrecision:
    def test_hit_requires_gold_first_and_returned_by_the_engine(self):
        m = score([rec(["g"], ["g"], ["g"])])
        assert m["precision"] == 1.0

    def test_gold_cited_but_never_returned_by_the_engine_is_not_precise(self):
        # Recalled from model memory: correct, but not grounded, so it does not resolve.
        m = score([rec(["g"], ["other"], ["g"])])
        assert m["correctness"] == 1.0
        assert m["groundedness"] == 0.0
        assert m["precision"] == 0.0

    def test_gold_present_but_not_first_is_not_precise(self):
        m = score([rec(["wrong", "g"], ["g"], ["g"])])
        assert m["correctness"] == 1.0
        assert m["precision"] == 0.0

    def test_abstention_is_never_precise(self):
        m = score([rec(["g"], ["g"], ["g"], committed=False)])
        assert m["precision"] == 0.0
        assert m["abstain"] == 1.0

    def test_lenient_scoring_accepts_either_gold_of_a_two_wide_pair(self):
        # The fix track registers the issue AND the PR; citing either resolves the task.
        m = score([rec(["pr"], ["pr", "issue"], ["issue", "pr"])])
        assert m["precision"] == 1.0

    def test_dead_run_stays_in_the_denominator(self):
        m = score([rec(["g"], ["g"], ["g"]), rec([], [], ["g"], error="timeout")])
        assert m["n"] == 2
        assert m["precision"] == 0.5


class TestRecallAndMrr:
    def test_recall_is_the_fraction_of_golds_found(self):
        m = score([rec(["a"], ["a"], ["a", "b"])])
        assert m["recall"] == pytest.approx(0.5)

    def test_recall_is_one_when_every_gold_is_listed(self):
        m = score([rec(["a", "b"], ["a", "b"], ["a", "b"])])
        assert m["recall"] == 1.0

    def test_amrr_uses_the_first_gold_position_in_the_agent_list(self):
        m = score([rec(["x", "y", "g"], ["g"], ["g"])])
        assert m["amrr"] == pytest.approx(1 / 3)

    def test_amrr_is_zero_with_no_hit(self):
        assert score([rec(["x"], ["g"], ["g"])])["amrr"] == 0.0

    def test_only_the_first_ten_citations_count(self):
        cites = [f"x{i}" for i in range(10)] + ["g"]
        assert score([rec(cites, ["g"], ["g"])])["recall"] == 0.0

    def test_citing_one_gold_twice_does_not_count_as_two(self):
        # Agents cite the same artifact in two spellings (owner/repo#N and its URL). Counting
        # hit POSITIONS scored that 1.0 on a two-wide item the agent only half-answered.
        m = score([rec(["a", "a"], ["a"], ["a", "b"])])
        assert m["recall"] == pytest.approx(0.5)


class TestInvariants:
    def test_partition_sums_to_one(self):
        m = score([rec(["g"], ["g"], ["g"]),
                   rec(["wrong"], ["g"], ["g"]),
                   rec(["g"], ["g"], ["g"], committed=False)])
        assert m["precision"] + m["wrong"] + m["abstain"] == pytest.approx(1.0)
        assert check_cell("t", m) == []

    def test_ordering_holds_on_a_mixed_cell(self):
        m = score([rec(["g"], ["g"], ["g"]), rec(["g"], ["nope"], ["g"]), rec([], [], ["g"])])
        assert m["precision"] <= m["groundedness"] <= m["correctness"]
        assert check_cell("t", m) == []

    def test_checker_reports_a_broken_partition(self):
        assert check_cell("t", dict(precision=0.9, wrong=0.9, abstain=0.0, groundedness=1.0,
                                    correctness=1.0, pool_hit=1.0))

    def test_checker_reports_precision_above_groundedness(self):
        assert check_cell("t", dict(precision=0.5, wrong=0.5, abstain=0.0, groundedness=0.2,
                                    correctness=1.0, pool_hit=1.0))

    def test_checker_skips_pool_hit_for_arms_with_no_search_log(self):
        # A native arm's pool_hit is structurally 0 because it emits no ranked result list;
        # asserting precision <= pool_hit there reports a phantom failure.
        m = dict(precision=0.03, wrong=0.97, abstain=0.0, groundedness=0.03,
                 correctness=0.5, pool_hit=0.0, has_search_log=False)
        assert check_cell("t", m) == []
        assert check_cell("t", {**m, "has_search_log": True})


class TestToolAccounting:
    def test_tools_counts_searches_reads_and_vendor_fetches(self):
        m = score([rec(["g"], ["g"], ["g"], searches=2, reads=1,
                       fetch_log=[{"url": "a"}, {"url": "b"}])])
        assert m["tools"] == 5
        assert m["searches"] == 2

    def test_native_extras_are_not_added_on_top_of_the_budget_counters(self):
        # `ext_tool_calls` is the native arm's tally of the SAME blocks already counted in
        # searches/reads. Adding it published roughly double that arm's real effort, and only
        # that arm's -- so the reported searches/q gap between it and the others was an artifact.
        m = score([rec(["g"], ["g"], ["g"], searches=3, reads=2, ext_tool_calls=5)])
        assert m["searches"] == 3
        assert m["tools"] == 5

    def test_empty_cell_scores_none(self):
        assert score_records([], "fix", gold_test) is None


class TestRunDiscovery:
    def _cell(self, tmp_path, label):
        d = tmp_path / label / "run1" / "tasks"
        d.mkdir(parents=True)
        (d / "p1.json").write_text("[]")

    def test_both_fix_directory_spellings_are_found(self, tmp_path):
        # `--track fix` writes '-fix-'; sweeps predating the cleanup wrote '-fixshort-'. There is
        # only one fix dataset, so reading one spelling reported real runs as "no cells".
        self._cell(tmp_path, "20260101T000000-p1-opus-fix-exa-mcp")
        self._cell(tmp_path, "20260101T000000-p1-opus-fixshort-exa-mcp")
        assert len(run_files(str(tmp_path), "p1", "fixshort", "exa-mcp", ".json")) == 2

    def test_a_track_does_not_swallow_another_arms_cells(self, tmp_path):
        self._cell(tmp_path, "20260101T000000-p1-opus-fix-exa-mcp")
        self._cell(tmp_path, "20260101T000000-p1-opus-fix-fc-mcp")
        assert len(run_files(str(tmp_path), "p1", "fixshort", "exa-mcp", ".json")) == 1

    def test_repo_and_docs_tokens_match_only_themselves(self, tmp_path):
        self._cell(tmp_path, "20260101T000000-p1-opus-repo-exa-mcp")
        self._cell(tmp_path, "20260101T000000-p1-opus-docs-exa-mcp")
        assert len(run_files(str(tmp_path), "p1", "repo", "exa-mcp", ".json")) == 1
        assert len(run_files(str(tmp_path), "p1", "docs", "exa-mcp", ".json")) == 1


class TestNaming:
    """Keys are identifiers baked into run paths; names are what a reader sees. A key without a
    name silently prints the raw identifier into a report, which is how a third spelling gets
    into circulation."""

    def test_every_arm_key_has_a_name(self):
        assert not [a for a in ARMS if a not in ARM_NAMES]

    def test_every_track_key_has_a_name(self):
        assert not [t for t in TRACKS if t not in TRACK_NAMES]

    def test_the_unreported_long_fix_variant_is_still_named(self):
        # It is not in TRACKS (we report the symptom form) but it appears in A/B tables.
        assert "fix" in TRACK_NAMES

    def test_names_are_unique(self):
        assert len(set(ARM_NAMES.values())) == len(ARM_NAMES)

    def test_controls_are_all_known_arms(self):
        assert CONTROLS <= set(ARMS)

    def test_unknown_key_falls_back_to_itself_rather_than_raising(self):
        assert arm_name("brand-new-arm") == "brand-new-arm"
        assert track_name("brand-new-track") == "brand-new-track"

    def test_the_two_firecrawl_arms_are_distinguishable(self):
        # They are different products, not variants of one; a report that blurs them is wrong.
        assert ARM_NAMES["fc-mcp"] != ARM_NAMES["fc-web"]
        assert "Developer Index" in ARM_NAMES["fc-mcp"]

    def test_track_names_distinguish_the_two_fix_query_forms(self):
        assert TRACK_NAMES["fix-short"] != TRACK_NAMES["fix"]


class TestDocsFamily:
    def test_docs_scores_on_canonical_url_matching_with_no_judge_fields(self):
        # Docs goldness is canonical-URL matching, exactly like repo/fix reference matching.
        # Judge verdicts persisted on old records must be ignored entirely.
        rows = [{"sources": ["http://x"], "correctness": 0, "grounded_correctness": 0,
                 "committed": True,
                 "search_log": [{"results": [{"rank": 0, "url": "http://x"}]}]}]
        m = score_records(rows, "docs", lambda item, record: item == "http://x",
                          pool_test=lambda record, engine: True)
        assert m["correctness"] == 1.0        # cited a canonical page
        assert m["recall"] == 1.0
        assert m["precision"] == 1.0          # and cited it first
        assert m["groundedness"] == 1.0

    def test_docs_precision_needs_the_canonical_page_cited_first(self):
        rows = [{"sources": ["http://junk", "http://x"], "committed": True,
                 "search_log": [{"results": [{"rank": 0, "url": "http://x"}]}]}]
        m = score_records(rows, "docs", lambda item, record: item == "http://x",
                          pool_test=lambda record, engine: True)
        assert m["correctness"] == 1.0
        assert m["precision"] == 0.0

    def test_docs_submitted_reads_sources_not_citations(self):
        r = {"sources": ["http://A/"], "citations": ["ignored"]}
        assert submitted(r, "docs") == ["http://a"]


class TestOneDefinition:
    """The per-pass scorer (suite.py) and the published scorer (report_metrics.py) must not
    drift. They once kept private copies of these, and the suite scored the engine pool
    UNCAPPED while the published table capped at 10 -- the same run produced two different
    precision numbers."""

    def test_suite_imports_the_published_definitions_rather_than_restating_them(self):
        from devdex.scorer import report_metrics as rm
        from devdex.scorer import suite
        assert suite._golds is rm.golds
        assert suite._norm is rm.norm
        assert suite.CAP == rm.CAP

    def test_suite_engine_pool_honours_the_depth_cap(self):
        row = {"search_log": [{"results": [{"rank": 0, "ref": "near"},
                                           {"rank": 12, "ref": "deep"}]}]}
        from devdex.scorer import suite
        assert suite._engine_pool(row) == {"near"}

    def test_suite_engine_rank_ignores_hits_past_the_cap(self):
        from devdex.scorer import suite
        row = {"gold_accept": ["g"], "search_log": [{"results": [{"rank": 12, "ref": "g"}]}]}
        assert suite._engine_rank(row) is None


class TestTableRendering:
    def test_every_heading_sits_over_its_own_figure(self, capsys):
        """The heading row and the figure row were two independent format strings and drifted --
        a stale 'nDCG' heading with no column, and 'poolhit' at width 9 over a width-7 figure,
        silently printed every number right of groundedness under the wrong heading."""
        from devdex.scorer.report_metrics import _print

        cell = dict(n=111, precision=0.712, recall=0.775, correctness=0.775, groundedness=0.757,
                    pool_hit=0.775, amrr=0.737, tools=3.5, call_s=1.2, e2e_s=44.1, cost=0.417,
                    resolved=79, wrong=0.2, abstain=0.088)
        _print({"repo": {"fc-mcp": cell}}, [])
        lines = capsys.readouterr().out.splitlines()

        for head in [l for l in lines if l.lstrip().startswith("system")]:
            figures = next(l for l in lines[lines.index(head):] if "Firecrawl" in l)
            assert len(head) == len(figures), f"heading/figure width mismatch:\n{head}\n{figures}"
            # every heading must end where its figure ends, column by column
            for label in ("prec", "grnd", "poolhit", "aMRR"):
                assert label in head
                end = head.index(label) + len(label)
                assert figures[end - 1] != " ", f"{label!r} heading is not over a figure"


class TestEveryPassIsScored:
    """`--n-runs 3` writes run1/, run2/, run3/ under one cell directory. Globbing run1 alone
    silently discarded two thirds of a paid sweep and made a three-pass cell report the same
    numbers as a one-pass cell -- while the docs told people to run --n-runs 3."""

    def _pass(self, tmp_path, run_i, precision_gold):
        d = tmp_path / "20260101T000000-p1-opus-repo-fc-mcp" / f"run{run_i}" / "tasks"
        d.mkdir(parents=True)
        (d / f"p{run_i}_repo_fc-mcp.json").write_text(json.dumps([
            {"qid": "q1", "gold_accept": ["g"], "committed": True,
             "citations": ["g"] if precision_gold else ["wrong"],
             "search_log": [{"results": [{"rank": 0, "ref": "g"}]}]}]))

    def test_all_three_passes_are_read(self, tmp_path):
        for i in (1, 2, 3):
            self._pass(tmp_path, i, precision_gold=True)
        assert len(run_files(str(tmp_path), "p1", "repo", "fc-mcp", ".json")) == 3

    def test_cell_averages_across_passes_and_keeps_n_as_the_item_count(self, tmp_path):
        from devdex.scorer.report_metrics import build

        self._pass(tmp_path, 1, precision_gold=True)
        self._pass(tmp_path, 2, precision_gold=False)
        cells, _ = build(str(tmp_path), "p1")
        cell = cells["repo"]["fc-mcp"]
        assert cell["passes"] == 2
        assert cell["n"] == 1, "n is the item count, not items x passes"
        assert cell["precision"] == pytest.approx(0.5), "one pass hit, one missed"


class TestServerSideArmIsMeasurableOnDocs:
    def test_docs_pool_falls_back_when_there_is_no_search_log(self):
        # Claude's WebSearch runs server-side and writes no search_log; its refs arrive only as
        # stream provenance. Reading the ranker alone scored it 0.000 pool_hit -- and so 0.000
        # groundedness and precision -- on docs even when it cited the canonical page.
        rec = {"qid": "d1", "sources": ["http://x"], "committed": True,
               "pool_prov": [["http://x", "search_direct"]]}
        m = score_records([rec], "docs", lambda item, record: item == "http://x",
                          pool_test=lambda record, engine: (
                              bool(record.get("search_log")) or
                              any(u == "http://x" for u in engine)))
        assert m["pool_hit"] == 1.0
        assert m["groundedness"] == 1.0
        assert m["precision"] == 1.0


class TestDocsHonoursTheDepthCap:
    """The docs ranker walked the whole result list at any depth while repo/fix went through
    engine_slots(cap=10). Docs alone then credited a gold returned in a slot no other arm was
    allowed to return -- measured live, an arm returning ~19 results was credited at rank 18.
    And `pooled` gates docs `groundedness` and `precision`, not just the pool_hit column, so the
    SAME agent behaviour scored higher on docs than on repo or fix."""

    def _meta(self):
        return {"repo_files": {"o/r:docs/page.md"}, "site": "", "page_stems": set()}

    def _row(self, rank):
        return {"search_log": [{"results": [
            {"rank": rank, "url": "https://github.com/o/r/blob/main/docs/page.md"}]}]}

    def test_a_gold_inside_the_cap_still_counts(self):
        from devdex.scorer import suite
        assert suite._docs_rank(self._row(0), self._meta()) == 1

    def test_a_gold_past_the_cap_does_not_count(self):
        from devdex.scorer import suite
        assert suite._docs_rank(self._row(17), self._meta()) is None, \
            "rank 18 is a slot the other arms were never allowed to return"

    def test_the_cap_matches_the_published_constant(self):
        from devdex.scorer import report_metrics as rm
        from devdex.scorer import suite
        assert suite._docs_rank(self._row(rm.CAP - 1), self._meta()) == rm.CAP
        assert suite._docs_rank(self._row(rm.CAP), self._meta()) is None

    def test_the_is_gold_probe_is_unaffected(self):
        # gold_test builds a synthetic single result at rank 0; capping must not break it
        from devdex.scorer import suite
        probe = {"search_log": [{"results": [
            {"rank": 0, "url": "https://github.com/o/r/blob/main/docs/page.md"}]}]}
        assert suite._docs_rank(probe, self._meta())


class TestGapDecomposition:
    """`correctness - groundedness` was read as "the model remembered it", but three different
    things land in it and they mean opposite things: parametric recall, a fetch rescuing what
    search never listed, and the engine returning the gold past the depth cap. Measured on the
    full run the composition flips per arm (fc-web/fix is 82% below_cap, websearch/fix is 96%
    from_read, parallel/repo is 74% from_memory), so one number cannot carry it."""

    def _score(self, rows):
        return score_records(rows, "fix", lambda i, r: i in golds(r))

    def test_memory_is_a_gold_no_pool_ever_saw(self):
        r = rec(["g"], ["other"], ["g"])          # engine returned something else
        r["pool_prov"] = [["other", "search_direct"]]
        m = self._score([r])
        assert (m["correctness"], m["groundedness"]) == (1.0, 0.0)
        assert m["from_memory"] == 1.0 and m["from_read"] == 0.0 and m["below_cap"] == 0.0

    def test_a_fetch_rescue_is_not_memory(self):
        r = rec(["g"], ["other"], ["g"])
        r["pool_prov"] = [["other", "search_direct"], ["g", "reader"]]
        m = self._score([r])
        assert m["from_read"] == 1.0, "a page the agent fetched carried it — not recall"
        assert m["from_memory"] == 0.0

    def test_a_gold_past_the_cap_is_a_ranking_failure_not_memory(self):
        r = {"citations": ["g"], "gold_accept": ["g"], "committed": True,
             "search_log": [{"results": [{"rank": 17, "ref": "g"}]}]}
        m = self._score([r])
        assert m["below_cap"] == 1.0, "the engine DID return it, just past rank 10"
        assert m["from_memory"] == 0.0 and m["from_read"] == 0.0

    def test_the_decomposition_closes_the_gap(self):
        rows = [rec(["g"], ["g"], ["g"])]                                  # grounded
        r2 = rec(["g"], ["x"], ["g"]); r2["pool_prov"] = [["g", "reader"]]  # read
        r3 = rec(["g"], ["x"], ["g"])                                       # memory
        r4 = {"citations": ["g"], "gold_accept": ["g"], "committed": True,
              "search_log": [{"results": [{"rank": 12, "ref": "g"}]}]}      # below cap
        m = self._score(rows + [r2, r3, r4])
        total = m["groundedness"] + m["from_memory"] + m["from_read"] + m["below_cap"]
        assert total == pytest.approx(m["correctness"])
        assert check_cell("t", m) == [], "completeness invariant must hold"

    def test_checker_catches_an_incomplete_decomposition(self):
        bad = dict(precision=0.0, wrong=1.0, abstain=0.0, groundedness=0.0, correctness=1.0,
                   pool_hit=1.0, from_memory=0.0, from_read=0.0, below_cap=0.0)
        assert any("!= correctness" in v for v in check_cell("t", bad))


class TestPrecisionJoinsOnTheCitedArtifact:
    """`pooled` is item-level: on the two-wide fix gold it is true when the engine returned
    EITHER the issue or the PR. Gating precision on that credited a memorised rank-1 citation of
    gold A because the engine happened to surface gold B — a memory-driven hit passing as a
    grounded one, inside the metric that claims grounding. Measured: up to +0.083 on fix, and it
    moved Exa from 1st to 3rd."""

    def test_citing_gold_a_from_memory_while_engine_returned_gold_b_is_not_precise(self):
        r = rec(["issue"], ["pr"], ["issue", "pr"])      # engine returned the PR only
        m = score_records([r], "fix", lambda i, rec_: i in golds(rec_))
        assert m["pool_hit"] == 1.0, "the engine did return a gold — pool_hit is still true"
        assert m["correctness"] == 1.0, "the agent did name a gold"
        assert m["precision"] == 0.0, "but not the one the engine surfaced"

    def test_citing_the_gold_the_engine_returned_is_precise(self):
        r = rec(["pr"], ["pr"], ["issue", "pr"])
        m = score_records([r], "fix", lambda i, rec_: i in golds(rec_))
        assert m["precision"] == 1.0

    def test_either_member_of_the_pair_still_counts_when_engine_returned_it(self):
        # lenient gold is preserved: the ISSUE is just as valid as the PR
        r = rec(["issue"], ["issue"], ["issue", "pr"])
        m = score_records([r], "fix", lambda i, rec_: i in golds(rec_))
        assert m["precision"] == 1.0

    def test_precision_never_exceeds_groundedness_or_pool_hit(self):
        rows = [rec(["issue"], ["pr"], ["issue", "pr"]),
                rec(["pr"], ["pr"], ["issue", "pr"]),
                rec(["x"], ["pr"], ["issue", "pr"])]
        m = score_records(rows, "fix", lambda i, rec_: i in golds(rec_))
        assert m["precision"] <= m["groundedness"] <= m["correctness"]
        assert m["precision"] <= m["pool_hit"]
        assert check_cell("t", m) == []


class TestEngineRecallAtTen:
    """`recall` scores the AGENT's submitted list; `pool_hit` is binary "any gold". Neither says
    how MUCH of a multi-gold answer set the engine surfaced. On the two-wide fix gold that is the
    difference between "found the issue" and "found the issue and its fix"."""

    def test_one_of_two_golds_is_half_engine_recall(self):
        r = rec(["pr"], ["pr"], ["issue", "pr"])         # engine returned only the PR
        m = score_records([r], "fix", lambda i, rec_: i in golds(rec_))
        assert m["pool_hit"] == 1.0, "binary: it found something"
        assert m["eng_recall"] == pytest.approx(0.5), "but only half the answer set"

    def test_both_golds_is_full_engine_recall(self):
        r = rec(["pr"], ["pr", "issue"], ["issue", "pr"])
        m = score_records([r], "fix", lambda i, rec_: i in golds(rec_))
        assert m["eng_recall"] == 1.0

    def test_engine_recall_honours_the_depth_cap(self):
        r = {"citations": ["g"], "gold_accept": ["g"], "committed": True,
             "search_log": [{"results": [{"rank": 17, "ref": "g"}]}]}
        m = score_records([r], "fix", lambda i, rec_: i in golds(rec_))
        assert m["eng_recall"] == 0.0, "a gold past rank 10 is outside recall@10"

    def test_on_a_single_gold_track_it_equals_pool_hit(self):
        rows = [rec(["g"], ["g"], ["g"]), rec(["x"], ["y"], ["g"])]
        m = score_records(rows, "repo", lambda i, rec_: i in golds(rec_))
        assert m["eng_recall"] == pytest.approx(m["pool_hit"])

"""Harness-side recording rules: which log a call lands in, and what a failed call leaves behind.

These drive the real Controller against a fake state -- no MCP server, no credentials, no
network. They pin the properties the scorer assumes: a read never reaches search_log, a failed
call never leaves a pending record for a later call to inherit, and search and fetch calls are
prepared differently.
"""
import sys
from pathlib import Path

HARNESS = Path(__file__).resolve().parents[1] / "harness"
sys.path.insert(0, str(HARNESS))

import pytest  # noqa: E402

import control as C  # noqa: E402

from devdex.tests._resume_shim import make_ns, make_resume


def _cfg():
    return C.Cfg(k=10, restrict=False, max_search=None, max_read=None,
                 agent_snippet=None, page_full=None)


class TestFetchErrorsAreVisible:
    def test_a_raised_fetch_is_logged_and_leaves_no_pending_record(self):
        st = C.State()
        ctl = C.Controller(st, _cfg(), search_tools={"srch"}, scope_hint="site:github.com")
        ctl.fetch_tools = {"fetchy"}
        ctl.prepare_fetch("fetchy", {"url": "http://x"}, "call-1")
        assert st._pending, "prepare_fetch must register the call"

        ctl.log_fetch_error("fetchy", RuntimeError("connection reset"), "call-1")

        assert st._pending == {}, "a failed fetch must not leave a record behind"
        assert not st.search_log, "a reader failure must never touch search coverage"
        assert len(st.fetch_log) == 1
        assert "connection reset" in st.fetch_log[0]["error"]


class TestPendingCannotBeMisattributed:
    def test_a_missing_id_does_not_inherit_someone_elses_pending_record(self):
        # A raised call leaves its record behind. Popping an arbitrary one logs a real search
        # under another call's query -- silently wrong, where an empty row is visibly missing.
        st = C.State()
        ctl = C.Controller(st, _cfg(), search_tools={"srch"})
        ctl.prepare("srch", {"query": "abandoned one"}, "dead-1")
        ctl.prepare("srch", {"query": "abandoned two"}, "dead-2")

        assert ctl._take("never-registered") == {}

    def test_a_single_pending_record_is_still_matched_for_idless_drivers(self):
        st = C.State()
        ctl = C.Controller(st, _cfg(), search_tools={"srch"})
        ctl.prepare("srch", {"query": "the only call"}, "c")
        assert ctl._take("unknown-id").get("q") == "the only call"


class TestResumeRetriesDeadItems:
    """A rate-limited vendor books a whole day of items as dead runs, and a dead run scores 0 and
    stays in the denominator. Inheriting those on resume lets the daily cap — not the index — set
    that arm's score, permanently. This is what kept a per-day-capped arm out of normal sweeps."""

    def _cell(self, tmp_path, rows):
        import json as _json
        d = tmp_path / "tasks"
        d.mkdir(parents=True, exist_ok=True)
        (d / "p1_repo_mintlify.json").write_text(_json.dumps(rows))

    def test_rate_limited_items_are_retried_not_inherited(self, tmp_path, monkeypatch):
        resume = make_resume(tmp_path, monkeypatch)
        rows = [{"qid": "a", "error": "429 Too Many Requests"},
                {"qid": "b"},                                    # genuine result
                {"qid": "c", "error": "timeout"}]
        self._cell(tmp_path, rows)
        keep, todo = resume("p1", "repo", "mintlify",
                            [{"id": "a"}, {"id": "b"}, {"id": "c"}])
        assert [r["qid"] for r in keep] == ["b"], "only real results survive"
        assert sorted(x["id"] for x in todo) == ["a", "c"], "dead items are re-queued"

    def test_a_genuine_miss_is_never_retried(self, tmp_path, monkeypatch):
        resume = make_resume(tmp_path, monkeypatch)
        # no error field: the engine answered and was simply wrong, or the agent abstained
        self._cell(tmp_path, [{"qid": "a", "citations": [], "committed": False}])
        keep, todo = resume("p1", "repo", "mintlify", [{"id": "a"}])
        assert len(keep) == 1 and todo == []

    def test_keep_dead_env_restores_the_old_behaviour(self, tmp_path, monkeypatch):
        monkeypatch.setenv("RESUME_KEEP_DEAD", "1")
        resume = make_resume(tmp_path, monkeypatch)
        self._cell(tmp_path, [{"qid": "a", "error": "429"}])
        keep, todo = resume("p1", "repo", "mintlify", [{"id": "a"}])
        assert len(keep) == 1 and todo == []


class TestProvenanceIsPersistedOnEveryTrack:
    """`pool_prov` was written only in the repo/fix branch of the record, so docs rows carried
    none. A server-side arm writes no search_log either, so on docs it had NOTHING to score
    against: pool_hit, groundedness and precision were structurally 0.000 however well it
    answered. The pool is collected on every track; it must be persisted on every track."""

    def _base_dict_source(self, runner):
        src = (Path(__file__).resolve().parents[1] / "harness" / runner).read_text()
        start = src.index("    base = {")
        return src[start:src.index("\n    if TRACK ==", start)]

    @pytest.mark.parametrize("runner", ["runner_sdk.py"])
    def test_pool_prov_is_in_the_common_record_not_a_track_branch(self, runner):
        common = self._base_dict_source(runner)
        assert '"pool_prov"' in common, (
            f"{runner}: pool_prov must be in the track-independent record, or docs rows lose "
            "the only provenance a server-side arm has")

    @pytest.mark.parametrize("runner", ["runner_sdk.py"])
    def test_pool_prov_is_not_also_written_per_track(self, runner):
        src = (Path(__file__).resolve().parents[1] / "harness" / runner).read_text()
        assert src.count('"pool_prov"') == 1, f"{runner}: duplicate pool_prov write"


class TestVendorThrottleIsAnErrorNotAResult:
    """A vendor that reports throttling as PROSE inside a 200 response is indistinguishable from
    a search that legitimately found nothing. One measured run had 279 of 387 calls answered with
    "Rate limit exceeded. Please try again in 547 seconds."; parsed as a result, the cell read as
    poor retrieval instead of as us being throttled."""

    def _resp(self, text):
        return {"content": [{"type": "text", "text": text}]}

    def test_a_prose_rate_limit_is_recorded_as_an_error(self):
        st = C.State()
        ctl = C.Controller(st, _cfg(), search_tools={"srch"})
        ctl.prepare("srch", {"query": "anything"}, "c1")
        out = ctl.record("srch", self._resp(
            "Rate limit exceeded. Please try again in 547 seconds."), "c1")

        assert out == [], "a throttle must not hand the agent results"
        assert len(st.search_log) == 1
        assert st.search_log[0]["error"], "the call must be booked as an error, not n=0 coverage"
        assert st.search_log[0]["n"] == 0
        assert not st.engine_pool, "nothing may enter the engine pool from a refusal"

    def test_an_ordinary_empty_result_is_not_mistaken_for_a_throttle(self):
        st = C.State()
        ctl = C.Controller(st, _cfg(), search_tools={"srch"})
        ctl.prepare("srch", {"query": "anything"}, "c1")
        ctl.record("srch", self._resp('{"results": []}'), "c1")
        assert not st.search_log[0].get("error"), "an honest empty result is a coverage miss"


class TestDeadRunAgreement:
    """The harness decides what to RETRY; the scorer decides what to PENALISE. If the scorer's
    list is not a subset of the harness's, rows exist that count against the 10% dead bar but are
    never re-run -- permanently frozen, and no number of retry windows repairs them. Seen in
    practice: 13 items sat dead-but-inherited across five windows."""

    def test_every_error_the_scorer_calls_dead_is_retryable_by_resume(self, tmp_path,
                                                                      monkeypatch):
        from devdex.scorer.report_metrics import DEAD_ERRORS
        ns = make_ns(tmp_path, monkeypatch)

        missing = [e for e in DEAD_ERRORS if e not in ns["_DEAD_ON_RESUME"]]
        assert not missing, (
            f"scorer penalises {missing} but resume would inherit those rows as good")
        for e in DEAD_ERRORS:
            assert ns["_is_dead"]({"error": f"Claude Code {e}: success"}), f"{e} must be retried"

    def test_a_clean_row_is_never_treated_as_dead(self, tmp_path, monkeypatch):
        ns = make_ns(tmp_path, monkeypatch)
        assert not ns["_is_dead"]({"error": None})
        assert not ns["_is_dead"]({})

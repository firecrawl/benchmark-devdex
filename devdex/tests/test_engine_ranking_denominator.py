"""Engine rank rates use all attempted items, including misses and failed runs."""
import pytest

from devdex.scorer.suite import score_pass


def row(rank=None, **extra):
    results = [{"rank": rank, "ref": "gold"}] if rank is not None else []
    return {"gold_accept": ["gold"], "citations": [], "committed": False,
            "search_log": [{"results": results}], **extra}


@pytest.mark.parametrize("track", ["repo", "fix"])
def test_misses_and_timeouts_remain_in_engine_rank_denominators(track):
    # Reciprocal ranks are [1, 1/2, 0, 0]; rank-1 indicators are [1, 0, 0, 0].
    metrics = score_pass([row(0), row(1), row(), row(error="timeout")], track)
    assert metrics["engR1"] == 0.25
    assert metrics["engMRR"] == 0.375
    assert metrics["pool_hit"] == 0.5


@pytest.mark.parametrize("track", ["repo", "fix"])
def test_adding_a_miss_lowers_engine_rank_rates(track):
    hit = score_pass([row(0)], track)
    mixed = score_pass([row(0), row()], track)
    assert mixed["engR1"] == hit["engR1"] / 2
    assert mixed["engMRR"] == hit["engMRR"] / 2


@pytest.mark.parametrize("track", ["repo", "fix"])
def test_no_hits_within_the_depth_cap_is_zero_not_missing(track):
    metrics = score_pass([row(), row(10), row(error="timeout")], track)
    assert metrics["engR1"] == 0.0
    assert metrics["engMRR"] == 0.0


def test_noncomparable_depth_remains_undefined():
    metrics = score_pass([row(0, config={"no_depth": True}), row()], "repo")
    assert metrics["engR1"] is None
    assert metrics["engMRR"] is None


def test_empty_pass_remains_empty():
    assert score_pass([], "repo") == {}


def test_native_arms_without_observed_ranks_remain_undefined():
    metrics = score_pass([{"gold_accept": ["gold"], "pool_prov": [["gold", "search_direct"]]}],
                         "repo")
    assert metrics["engR1"] is None
    assert metrics["engMRR"] is None

"""Per-arm payload parsing: every vendor's response SHAPE must yield ranked, referenced results.

Each engine answers in its own format — Firecrawl and Exa and Parallel in three different JSON
layouts, Mintlify and Context7 as markdown blocks, the gh control as a JSON array. They all pass
through one parser, and a shape it silently fails on does not raise: it yields zero results, or
results with no `ref`, and that arm then scores a structural 0.000 that reads as bad retrieval.
That is the most expensive possible failure here, because it looks like a finding.

No credentials, no network. The fixtures are shapes, not live payloads — `preflight.py` and the
pinned tools_manifest.json are what catch a vendor changing its schema.
"""
import json
import sys
from pathlib import Path

import pytest

HARNESS = Path(__file__).resolve().parents[1] / "harness"
sys.path.insert(0, str(HARNESS))

import provenance as P  # noqa: E402

GOLD_URL = "https://github.com/huggingface/transformers/issues/47116"
GOLD_REF = "huggingface/transformers#47116"
HITS = [(GOLD_URL, GOLD_REF),
        ("https://github.com/unrelated/alpha", "unrelated/alpha"),
        ("https://github.com/unrelated/beta", "unrelated/beta")]


def _mcp(text):
    return {"content": [{"type": "text", "text": text}]}


def firecrawl():
    return _mcp(json.dumps({"data": [{"url": u, "title": r, "markdown": f"{r} body"}
                                     for u, r in HITS]}))


def exa():
    return _mcp(json.dumps({"results": [{"url": u, "title": r, "text": f"{r} body"}
                                        for u, r in HITS]}))


def parallel():
    return _mcp(json.dumps({"results": [{"url": u, "title": r, "excerpts": [f"{r} body"]}
                                        for u, r in HITS]}))


def markdown_blocks():
    """Mintlify's `context` and Context7 both emit header / URL-on-its-own-line / body."""
    return _mcp("\n\n".join(f"### {r}\nSource: {u}\n{r} body" for u, r in HITS))


def gh_array():
    return _mcp(json.dumps([{"url": u, "fullName": r, "description": f"{r} body"}
                            for u, r in HITS]))


SHAPES = {"firecrawl (fc-mcp, fc-web)": firecrawl,
          "exa": exa,
          "parallel": parallel,
          "markdown (mintlify, context7)": markdown_blocks,
          "gh array (gh-hybrid)": gh_array}


@pytest.mark.parametrize("name", sorted(SHAPES))
class TestEveryVendorShapeParses:
    def test_every_result_is_recovered(self, name):
        results, _meta = P.harvest(SHAPES[name]())
        assert len(results) == len(HITS), (
            f"{name}: parsed {len(results)} of {len(HITS)} results — an arm whose shape the "
            "parser drops scores a structural zero that reads as bad retrieval")

    def test_the_gold_ref_is_derived_so_it_can_match_a_citation(self, name):
        # repo/fix goldness matches on `owner/repo#n`. A result carrying only a url never
        # intersects the agent's citation, so groundedness and precision are structurally 0.
        results, _meta = P.harvest(SHAPES[name]())
        assert GOLD_REF in {r.get("ref") for r in results}, (
            f"{name}: no result derived the ref {GOLD_REF!r} from its url")

    def test_rank_is_the_vendors_own_order(self, name):
        results, _meta = P.harvest(SHAPES[name]())
        assert [r["rank"] for r in results] == list(range(len(HITS)))
        assert results[0].get("ref") == GOLD_REF, "rank 0 must stay the vendor's first result"


class TestParserGuardsThatMustNotRegress:
    def test_a_lone_link_in_prose_is_not_a_result_list(self):
        # One anchor is a link inside prose. Treating it as a result list is what credited
        # engine_pool with every repo merely LINKED FROM inside another result's body.
        results, meta = P.harvest(_mcp(f"See {GOLD_URL} for details."))
        assert meta.get("unparsed") is True
        assert len(results) == 1 and results[0].get("ref") is None, \
            "an unparsed blob is attributed to the CALL and claims no ranking"

    def test_an_empty_result_set_is_a_coverage_miss_not_a_parse_failure(self):
        results, meta = P.harvest(_mcp(json.dumps({"results": []})))
        assert results == []
        assert not meta.get("unparsed"), "an honest empty list is coverage, not a broken payload"


class TestNativeArmIsObservable:
    """WebSearch runs server-side: no PostToolUse fires, so `record()` never sees it. The call and
    its verbatim result are both in the message stream, and the harness reconstructs a search_log
    entry from them. Without that the arm is structurally unmeasurable -- no n, no ranks, the
    depth cap cannot apply, and every engine-side column reads 0.000 for an arm that answered."""

    SEARCH = ("1. https://github.com/huggingface/transformers/issues/47116 report\n"
              "2. https://github.com/unrelated/alpha other\n"
              "3. https://docs.example.com/guide/page.md docs\n")

    def _harvest(self, state, tool, args, text, cap=10):
        """The native branch of runner_sdk's stream loop, in the same order."""
        is_search = tool == "WebSearch"
        src = "search_direct" if is_search else "reader"
        for m in P.GH_ARTIFACT.finditer(text):
            state.add(f"{m.group(1)}#{m.group(2)}", src)
        for m in P.GH_REPO.finditer(text):
            if m.group(1).count("/") == 1:
                state.add(m.group(1), src)
        items, seen = [], set()
        for m in P.URL_RE.finditer(text):
            u = m.group(0).rstrip(".,);")
            if u in seen:
                continue
            seen.add(u)
            state.add(u, src)
            items.append({"rank": len(items), "url": u, "ref": P.ref_from_url(u), "type": ""})
            if len(items) >= cap:
                break
        return items

    def test_a_native_search_yields_ranked_results_with_refs(self):
        import control as C
        st = C.State()
        items = self._harvest(st, "WebSearch", {"query": "q"}, self.SEARCH)
        assert [i["rank"] for i in items] == [0, 1, 2]
        assert items[0]["ref"] == GOLD_REF, "a github url must still derive its ref"
        assert st.engine_pool, "a native search must populate the engine pool"

    def test_docs_urls_are_harvested_too_not_only_github_refs(self):
        # This block once ran only on repo/fix, so on docs the native arm harvested NOTHING:
        # empty pool, and pool_hit / groundedness / precision structurally 0.000.
        import control as C
        st = C.State()
        self._harvest(st, "WebSearch", {"query": "q"}, self.SEARCH)
        assert "https://docs.example.com/guide/page.md" in st.engine_pool, \
            "docs scores on URLs; harvesting only github refs leaves the docs pool empty"

    def test_a_native_fetch_is_reader_origin_and_can_never_score(self):
        import control as C
        st = C.State()
        self._harvest(st, "WebFetch", {"url": "https://x"},
                      "body linking https://github.com/torvalds/linux")
        assert "torvalds/linux" not in st.engine_pool, "a read must never be credited as retrieval"
        assert "torvalds/linux" in st.agent_pool

"""A URL mentioning GitHub is not necessarily a result returned from GitHub."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from devdex.harness import control, provenance
from devdex.scorer.report_metrics import golds, score_records


@pytest.mark.parametrize("url", [
    "https://notgithub.com/o/r/issues/12",
    "https://evil.test/github.com/o/r/issues/12",
    "https://evil.test/?redirect=https://github.com/o/r/issues/12",
    "https://evil.test/#https://github.com/o/r/issues/12",
    "https://github.com.evil.test/o/r/issues/12",
    "https://github.com@evil.test/o/r/issues/12",
    "https://github.com:8443/o/r/issues/12",
    "https://github.com:invalid/o/r/issues/12",
])
def test_non_github_authorities_do_not_derive_github_refs(url):
    assert provenance.ref_from_url(url) is None


@pytest.mark.parametrize(("url", "expected"), [
    ("https://github.com/o/r/issues/12#comment", "o/r#12"),
    ("https://www.github.com/o/r/pull/12/files", "o/r#12"),
    ("https://github.com/o/r.git", "o/r"),
    ("https://github.com/o/r?redirect=https://github.com/other/fake/issues/99", "o/r"),
    ("https://github.com/o/r/issues/12wrong", "o/r"),
])
def test_only_the_actual_github_path_determines_the_reference(url, expected):
    assert provenance.ref_from_url(url) == expected


def test_real_controller_cannot_turn_lookalike_url_into_grounded_gold():
    state = control.State()
    cfg = control.Cfg(k=10, restrict=False, max_search=None, max_read=None,
                      agent_snippet=None, page_full=None)
    controller = control.Controller(state, cfg, search_tools={"search"})
    controller.prepare("search", {"query": "find the fix"}, "call1")
    controller.record("search", {"content": [{"type": "text", "text": json.dumps({
        "results": [{"url": "https://notgithub.com/o/r/issues/12", "title": "A mirror"}]})}]},
                      "call1")
    row = {"qid": "synthetic", "gold_accept": ["o/r#12"], "citations": ["o/r#12"],
           "committed": True, "search_log": state.search_log}
    metrics = score_records([row], "fix", lambda item, rec: item in golds(rec))
    assert metrics["correctness"] == 1
    assert metrics["groundedness"] == metrics["precision"] == metrics["pool_hit"] == 0


def native_result(text):
    # Execute the actual stream-result loop without importing its paid SDK/setup code.
    source = Path(provenance.__file__).with_name("runner_sdk.py")
    tree = ast.parse(source.read_text())
    loops = [node for node in ast.walk(tree) if isinstance(node, ast.For)
             and "msg.content" in ast.unparse(node.iter)
             and "state.search_log.append" in ast.unparse(node)]
    assert len(loops) == 1
    block = type("ToolResultBlock", (), {})()
    block.tool_use_id = "native1"
    block.content = text
    state = control.State()
    namespace = {"P": provenance, "state": state, "K": 10,
                 "msg": SimpleNamespace(content=[block]),
                 "called": {"native1": ("WebSearch", {"query": "test"})}}
    exec(compile(ast.Module(body=loops, type_ignores=[]), str(source), "exec"), namespace)
    return state


def test_native_stream_pool_rejects_embedded_github_links_on_other_hosts():
    state = native_result("https://evil.test/?redirect=https://github.com/o/r/issues/12")
    assert "o/r#12" not in state.engine_pool
    assert "o/r" not in state.engine_pool
    assert state.search_log[0]["results"][0]["ref"] is None


def test_native_stream_preserves_real_artifact_and_repository_refs():
    state = native_result("https://github.com/o/r/issues/12")
    assert {"o/r#12", "o/r"} <= state.engine_pool
    assert state.search_log[0]["results"][0]["ref"] == "o/r#12"

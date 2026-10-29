"""Equivalent MCP representations must preserve ranked search and fetched evidence."""
import json
import pytest
from devdex.harness import provenance as P
from devdex.harness.control import Cfg, Controller, State

PAYLOAD = {'results': [{'url': 'https://github.com/example/project', 'text': 'result body'}]}

@pytest.mark.parametrize('response', [PAYLOAD, PAYLOAD['results'],
    {'content': [], 'structuredContent': PAYLOAD}, {'structuredContent': PAYLOAD}])
def test_structured_search_reaches_real_controller(response):
    state = State()
    ctl = Controller(state, Cfg(), ['search'])
    ctl.prepare('search', {'query': 'fixture'}, 's')
    preview = ctl.record('search', response, 's')
    assert len(preview) == 1
    assert preview[0]['ref'] == 'example/project'
    assert preview[0]['content'] == 'result body'
    assert not state.search_log[0].get('error')
    assert state.engine_pool == {'example/project'}


def test_dual_representation_is_counted_once():
    body = json.dumps(PAYLOAD)
    out, meta = P.harvest({'content': [{'type': 'text', 'text': body}], 'structuredContent': PAYLOAD})
    assert len(out) == 1
    assert meta['raw_chars'] == len(body)


def test_structured_fetch_retains_document_text_and_url():
    doc = {'markdown': 'Complete fetched document.', 'metadata': {'url': 'https://docs.example.test/page'}}
    assert P.harvest_document({'content': [], 'structuredContent': doc}) == (
        'https://docs.example.test/page', 'Complete fetched document.')


def test_structured_empty_is_an_honest_miss():
    out, meta = P.harvest({'content': [], 'structuredContent': {'results': []}})
    assert out == []
    assert meta['shape'] == 'json'
    assert not meta['unparsed']

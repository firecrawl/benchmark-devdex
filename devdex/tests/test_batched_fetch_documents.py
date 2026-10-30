"""All fetched pages belong in the reader evidence store, never the engine pool."""
import json
import pytest
from devdex.harness.control import Cfg, Controller, State
from devdex.harness import provenance as P

URLS = ['https://github.com/example/first', 'https://github.com/example/second']
TEXTS = ['Full first document.', 'Complete second document with its cited passage.']


def fetch(payload, requested=URLS):
    state = State()
    ctl = Controller(state, Cfg(), ['search'], fetch_tools=['fetch'])
    ctl.prepare_fetch('fetch', {'urls': requested}, 'f')
    response = {'content': [{'type': 'text', 'text': json.dumps(payload)}]}
    assert ctl.record_fetch('fetch', response, 'f') is None
    assert not state.engine_pool
    assert state.search_log == []
    assert state._pending == {}
    return state


@pytest.mark.parametrize('shape', ['results', 'data', 'documents', 'pages', 'root'])
def test_all_batch_documents_are_available_for_later_citations(shape):
    docs = [{'url': url, 'markdown': text} for url, text in zip(URLS, TEXTS)]
    state = fetch(docs if shape == 'root' else {shape: docs})
    assert state.url_content == dict(zip(URLS, TEXTS))
    assert state.agent_pool == {'example/first', 'example/second'}
    log = state.fetch_log[0]
    assert log['n'] == 2
    assert log['chars'] == sum(map(len, TEXTS))
    assert log['refs'] == ['example/first', 'example/second']


def test_single_document_keeps_root_body_and_metadata_url():
    state = fetch({'markdown': TEXTS[0], 'metadata': {'url': URLS[0], 'description': 'short metadata'}}, [URLS[0]])
    assert state.url_content == {URLS[0]: TEXTS[0]}
    assert state.fetch_log[0]['n'] == 1


def test_batch_excerpts_remain_separate_document_content():
    state = fetch({'results': [{'url': url, 'excerpts': [text]} for url, text in zip(URLS, TEXTS)]})
    assert state.url_content == dict(zip(URLS, TEXTS))


def test_legacy_single_document_helper_remains_a_tuple():
    response = {'content': [{'type': 'text', 'text': json.dumps({'results': [
        {'url': URLS[0], 'markdown': TEXTS[0]}, {'url': URLS[1], 'markdown': TEXTS[1]}]})}]}
    assert P.harvest_document(response) == (URLS[0], TEXTS[0])


def test_bare_markdown_still_uses_the_single_requested_url():
    state = State()
    ctl = Controller(state, Cfg(), ['search'], fetch_tools=['fetch'])
    ctl.prepare_fetch('fetch', {'url': URLS[0]}, 'f')
    ctl.record_fetch('fetch', {'content': [{'type': 'text', 'text': TEXTS[0]}]}, 'f')
    assert state.url_content == {URLS[0]: TEXTS[0]}


@pytest.mark.parametrize('body', ['true', '123', '"Quoted JSON document."',
                                  '{"name":"fixture","version":"1"}', '["first","second"]'])
def test_plain_document_bodies_that_happen_to_be_json_are_not_discarded(body):
    state = State()
    ctl = Controller(state, Cfg(), ['search'], fetch_tools=['fetch'])
    ctl.prepare_fetch('fetch', {'url': URLS[0]}, 'f')
    ctl.record_fetch('fetch', {'content': [{'type': 'text', 'text': body}]}, 'f')
    assert state.url_content == {URLS[0]: body}
    assert state.fetch_log[0]['n'] == 1
    assert state.fetch_log[0]['chars'] == len(body)


def test_unattributed_batch_prose_is_not_attached_to_an_arbitrary_first_url():
    state = State()
    ctl = Controller(state, Cfg(), ['search'], fetch_tools=['fetch'])
    ctl.prepare_fetch('fetch', {'urls': URLS}, 'f')
    ctl.record_fetch('fetch', {'content': [{'type': 'text', 'text': 'Unattributed batch response.'}]}, 'f')
    assert state.url_content == {}
    assert not state.agent_pool


def test_empty_batch_has_no_phantom_document():
    state = fetch({'results': []})
    assert state.url_content == {}
    assert state.fetch_log[0]['n'] == 0


def test_mcp_error_stays_a_fetch_error_without_evidence():
    state = State()
    ctl = Controller(state, Cfg(), ['search'], fetch_tools=['fetch'])
    ctl.prepare_fetch('fetch', {'urls': URLS}, 'f')
    ctl.record_fetch('fetch', {'isError': True, 'content': [{'type': 'text', 'text': 'unavailable'}]}, 'f')
    assert state.fetch_log[0]['error']
    assert state.fetch_log[0]['n'] == 0
    assert state.url_content == {}
    assert not state.agent_pool
    assert not state.engine_pool

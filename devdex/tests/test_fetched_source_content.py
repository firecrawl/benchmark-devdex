"""Fetched citation evidence must be the page read, rather than its search preview."""
import json
from devdex.harness.control import Cfg, Controller, State

URL = 'https://docs.example.test/guide'


def test_fetch_replaces_stored_search_preview_without_crediting_reader():
    state = State()
    ctl = Controller(state, Cfg(), ['search'], fetch_tools=['fetch'])
    ctl.prepare('search', {'query': 'guide'}, 's1')
    ctl.record('search', {'content': [{'type': 'text', 'text': json.dumps({
        'results': [{'url': URL, 'text': 'short search preview'}]})}]}, 's1')
    assert state.url_content[URL] == 'short search preview'
    engine_before = set(state.engine_pool)
    ctl.prepare_fetch('fetch', {'url': URL}, 'r1')
    ctl.record_fetch('fetch', {'content': [{'type': 'text', 'text': json.dumps({
        'url': URL, 'markdown': 'Full page including the section the answer cites.'})}]}, 'r1')
    assert state.url_content[URL] == 'Full page including the section the answer cites.'
    assert state.engine_pool == engine_before
    assert len(state.search_log) == 1
    assert len(state.fetch_log) == 1


def test_failed_fetch_preserves_previously_stored_content():
    state = State()
    state.url_content[URL] = 'existing evidence'
    ctl = Controller(state, Cfg(), ['search'], fetch_tools=['fetch'])
    ctl.prepare_fetch('fetch', {'url': URL}, 'r1')
    ctl.record_fetch('fetch', {'isError': True, 'content': [{'type': 'text', 'text': 'unavailable'}]}, 'r1')
    assert state.url_content[URL] == 'existing evidence'
    assert state.fetch_log[0]['error']

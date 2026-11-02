"""Optional result headers must not steal the preceding result's last body line."""
import pytest
from devdex.harness.control import Cfg, Controller, State
from devdex.harness import provenance as P

U1 = 'https://github.com/example/first'
U2 = 'https://github.com/example/second'


def record(text):
    state = State()
    ctl = Controller(state, Cfg(), ['search'])
    ctl.prepare('search', {'query': 'synthetic result boundary'}, 's')
    preview = ctl.record('search', {'content': [{'type': 'text', 'text': text}]}, 's')
    assert [r['ref'] for r in preview] == ['example/first', 'example/second']
    assert len(state.search_log) == 1
    return preview, state


@pytest.mark.parametrize('first', ['First complete body.', 'First paragraph.\nFinal body line.'])
def test_bare_url_results_keep_all_of_each_body(first):
    preview, state = record(f'{U1}\n{first}\n{U2}\nSecond complete body.')
    assert preview[0]['content'] == first
    assert preview[1]['content'] == 'Second complete body.'
    assert state.url_content == {U1: first, U2: 'Second complete body.'}


def test_a_title_is_a_header_only_for_its_own_result():
    preview, _ = record(f'Title: First title\n{U1}\nFirst body.\n{U2}\nSecond body.')
    assert preview[0]['content'] == 'First title\nFirst body.'
    assert preview[1]['content'] == 'Second body.'


def test_declared_title_headers_still_belong_to_their_result():
    preview, _ = record(f'Title: First title\n{U1}\nFirst body.\nTitle: Second title\n{U2}\nSecond body.')
    assert preview[0]['content'] == 'First title\nFirst body.'
    assert preview[1]['content'] == 'Second title\nSecond body.'


def test_typed_markdown_headers_keep_their_document_identity():
    parsed = P._from_markdown(
        f'## [readme:example/first] (readme)\n{U1}\nFirst body.\n'
        f'## [readme:example/second] (readme)\n{U2}\nSecond body.')
    assert [r['id'] for r in parsed] == ['readme:example/first', 'readme:example/second']
    assert [r['text'] for r in parsed] == ['First body.', 'Second body.']


def test_inline_prose_links_do_not_become_ranked_results():
    assert P._from_markdown(f'Read [one]({U1}) and [two]({U2}) for more information.') == []

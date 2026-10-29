"""A search about an error must not itself be classified as a failed search."""
import json
import pytest
from devdex.harness.control import State, Cfg, Controller


def record(response):
    state = State(); ctl = Controller(state, Cfg(), ['search'])
    ctl.prepare('search', {'query':'retry HTTP429 error'}, 's')
    return ctl.record('search', response, 's'), state


def mcp(payload):
    return {'content':[{'type':'text','text':json.dumps(payload)}]}

@pytest.mark.parametrize('words', ['Rate limit exceeded', 'Too many requests',
    'Quota exhausted', 'Upgrade your plan', 'Higher limits'])
def test_ranked_issue_explaining_an_error_is_still_a_successful_search(words):
    out, state = record(mcp({'results':[{'url':'https://github.com/example/project/issues/42',
        'title':'Fix '+words+' in client', 'text':'The patch implements bounded request retries.'}]}))
    assert len(out) == 1
    assert out[0]['ref'] == 'example/project#42'
    assert not state.search_log[0].get('error')
    assert state.engine_pool == {'example/project#42'}

@pytest.mark.parametrize('response', [
    {'content':[{'type':'text','text':'Rate limit exceeded. Please try again in547seconds.'}]},
    {'isError':True,'content':[{'type':'text','text':'Too many requests; see https://help.example.test/limits'}]},
    mcp({'message':'Quota exceeded','url':'https://help.example.test/limits'}),
    mcp({'data':{'message':'Quota exceeded','url':'https://help.example.test/limits'}}),
    mcp({'results':[{'url':'https://github.com/example/project','text':'partial data'}],
         'error':{'message':'Quota exhausted','url':'https://help.example.test/limits'}}),
])
def test_real_call_refusals_remain_errors_even_with_incidental_urls(response):
    out, state = record(response)
    assert out == []
    assert state.search_log[0]['error']
    assert not state.engine_pool


def test_error_metadata_belongs_to_the_ranked_result_not_the_call():
    out, state = record(mcp({'results':[{
        'url':'https://github.com/example/project/issues/42',
        'title':'Retry client failures', 'text':'A valid returned issue.',
        'metadata':{'error':'Quota exhausted'}
    }]}))
    assert out[0]['ref'] == 'example/project#42'
    assert not state.search_log[0].get('error')

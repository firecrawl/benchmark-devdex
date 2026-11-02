"""Exact GitHub query transport through a real subprocess peer and Controller."""
import json
import os
import sys
from urllib.parse import parse_qs,urlsplit
import pytest
from devdex.harness import control as C

@pytest.fixture
def gh_peer(tmp_path,monkeypatch):
    receipt=tmp_path/'request.json'
    exe=tmp_path/'gh'
    exe.write_text(f'''#!{sys.executable}
import json,pathlib,sys
pathlib.Path({str(receipt)!r}).write_text(json.dumps(sys.argv[1:]))
if sys.argv[1]!='api':
 print('fixture requires exact API query transport',file=sys.stderr);sys.exit(2)
if sys.argv[2].startswith('search/repositories?'):
 item={{'full_name':'example/project','html_url':'https://github.com/example/project',
        'stargazers_count':17,'description':'Synthetic repository'}}
else:
 item={{'html_url':'https://github.com/example/project/issues/42','title':'Fixture issue','body':'Fixture body'}}
print(json.dumps({{'items':[item]}}))
''')
    exe.chmod(0o755)
    monkeypatch.setenv('PATH',str(tmp_path)+os.pathsep+os.environ['PATH'])
    monkeypatch.setattr(C,'_gh_throttle',lambda:None)
    return receipt


def retrieve(query,kind):
    state=C.State();ctl=C.Controller(state,C.Cfg(restrict=False),['gh_search'])
    assert ctl.gate('gh_search') is None
    sent=ctl.prepare('gh_search',{'query':query,'type':kind},'q')
    text,failed=C.gh_search(state,sent,limit=3)
    preview=ctl.record('gh_search',{'content':[{'type':'text','text':text}],'isError':failed},'q')
    return failed,preview,state

@pytest.mark.parametrize('query',[
    '-language:Python topic:json', 'json parser retries',
    'repo:example/project "json parser" -archived:true','json café & decoder'])
def test_repository_query_reaches_api_unchanged(query,gh_peer):
    failed,preview,state=retrieve(query,'repo')
    assert not failed
    assert preview[0]['ref']=='example/project'
    assert preview[0]['content']=='example/project (17 stars)\nSynthetic repository'
    assert state.engine_pool=={'example/project'}
    assert len(state.search_log)==1
    args=json.loads(gh_peer.read_text())
    assert args[0]=='api'
    assert urlsplit(args[1]).path=='search/repositories'
    assert parse_qs(urlsplit(args[1]).query)=={'q':[query],'per_page':['3']}

@pytest.mark.parametrize('kind',['issue','pr'])
def test_issue_and_pull_request_transport_remains_unchanged(kind,gh_peer):
    query='repo:example/project -label:duplicate retry'
    failed,preview,state=retrieve(query,kind)
    assert not failed
    assert preview[0]['ref']=='example/project#42'
    args=json.loads(gh_peer.read_text())
    assert args[0]=='api'
    assert urlsplit(args[1]).path=='search/issues'
    expected=query+(' is:pr' if kind=='pr' else '')
    assert parse_qs(urlsplit(args[1]).query)=={'q':[expected],'per_page':['3']}

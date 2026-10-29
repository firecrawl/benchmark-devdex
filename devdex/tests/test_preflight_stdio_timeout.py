"""An actual silent stdio peer cannot hold preflight forever on readline()."""
import os
import sys
import time
import pytest
from devdex.harness import preflight as P

@pytest.mark.parametrize('prefix', ['', 'incomplete-json-with-no-newline'])
def test_silent_or_partial_line_peer_times_out_and_is_reaped(tmp_path, monkeypatch, prefix):
    pidfile = tmp_path/'pid'
    peer = tmp_path/'peer.py'
    peer.write_text(f'''import os,pathlib,sys,time
pathlib.Path({str(pidfile)!r}).write_text(str(os.getpid()))
sys.stdout.write({prefix!r});sys.stdout.flush()
time.sleep(2.5)
''')
    monkeypatch.setattr(P, 'STDIO_TIMEOUT', .5, raising=False)
    started = time.monotonic()
    with pytest.raises(TimeoutError, match='preflight'):
        P.list_stdio(sys.executable, ['-u',str(peer)], {})
    assert time.monotonic()-started < 2
    with pytest.raises(ProcessLookupError): os.kill(int(pidfile.read_text()), 0)


def test_responsive_peer_still_returns_its_schema(tmp_path, monkeypatch):
    peer = tmp_path/'peer.py'
    peer.write_text('''import json,sys
for line in sys.stdin:
 m=json.loads(line)
 if m.get('method')=='initialize': r={'serverInfo':{'name':'fixture'}}
 elif m.get('method')=='tools/list': r={'tools':[{'name':'search'}]}
 else: continue
 print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':r}),flush=True)
''')
    monkeypatch.setattr(P, 'STDIO_TIMEOUT', 2, raising=False)
    tools, info = P.list_stdio(sys.executable, ['-u',str(peer)], {})
    assert tools == [{'name':'search'}]
    assert info['name'] == 'fixture'

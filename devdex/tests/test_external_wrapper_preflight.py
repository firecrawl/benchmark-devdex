"""The public external wrapper must discover the submitted MCP contract before a driver.

Run the production wrapper, run_eval and preflight in real subprocesses against an owned
HTTP MCP peer. Only the paid run_cell boundary is replaced; no SDK/model session is started.
"""
import json
import shutil
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from benchmark import register_arm

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('tool_name', ['search', 'renamed_search'])
def test_public_wrapper_discovers_and_checks_external_tool_before_driver(tmp_path, monkeypatch, tool_name):
    project = tmp_path / 'project'
    shutil.copytree(ROOT / 'devdex', project / 'devdex', ignore=shutil.ignore_patterns(
        'tests', 'runs', '__pycache__', '*.pyc'))
    entry = project / 'devdex' / 'run_eval.py'
    entry.rename(entry.with_name('run_eval_production.py'))
    receipt = project / 'driver-boundary.json'
    entry.write_text('''import json,runpy,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from devdex.scorer import suite
def boundary(**kwargs):
 from devdex.harness.control import Cfg,Controller,State
 root=Path(__file__).resolve().parent
 file=root/'harness/tools_manifest.json'
 manifest=json.loads(file.read_text()) if file.exists() else {}
 ctl=Controller(State(),Cfg(k=10,restrict=True,manifest=manifest),['mcp__ext__search'])
 args=ctl.prepare('mcp__ext__search',{'query':'synthetic query'},'q')
 (root.parent/'driver-boundary.json').write_text(json.dumps({'manifest':manifest,'args':args}))
 return {'aggregate':{}}
suite.run_cell=boundary
runpy.run_path(str(Path(__file__).with_name('run_eval_production.py')),run_name='__main__')
''')
    calls = []

    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def do_POST(self):
            msg = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(msg['method'])
            if msg['method'] == 'notifications/initialized':
                self.send_response(202)
                self.end_headers()
                return
            result = {'serverInfo': {'name': 'owned-fixture'}} if msg['method'] == 'initialize' else {
                'tools': [{'name': tool_name, 'inputSchema': {'type': 'object', 'properties': {
                    'query': {'type': 'string'}, 'limit': {'type': 'integer'},
                    'includeDomains': {'type': 'array', 'items': {'type': 'string'}}}}}]}
            body = json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    peer = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    thread = threading.Thread(target=peer.serve_forever, daemon=True)
    thread.start()
    monkeypatch.setattr(register_arm, 'ROOT', project)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'synthetic-local-fixture')
    monkeypatch.setenv('DEVDEX_EXT_MCP_URL', f'http://127.0.0.1:{peer.server_port}/mcp')
    monkeypatch.setenv('DEVDEX_EXT_SEARCH_TOOL', 'search')
    monkeypatch.delenv('DEVDEX_EXT_AUTH', raising=False)
    try:
        rc = register_arm.run_agent_mode('fixture', track='repo', limit=1)
    finally:
        peer.shutdown()
        peer.server_close()
        thread.join()
    assert 'tools/list' in calls, 'the wrapper must not bypass real MCP discovery'
    if tool_name == 'search':
        assert rc == 0
        seen = json.loads(receipt.read_text())
        assert 'mcp__ext__search' in seen['manifest']
        assert seen['args']['limit'] == 10
        assert seen['args']['includeDomains'] == ['github.com']
    else:
        assert rc != 0
        assert not receipt.exists(), 'a renamed tool must fail before the paid driver boundary'

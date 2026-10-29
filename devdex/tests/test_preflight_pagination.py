"""Real localhost and stdio MCP peers expose a desired tool only on page two."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from devdex.harness import preflight as P


def test_http_lists_all_tool_pages_and_preserves_session():
    requests = []
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            msg = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            requests.append((msg, self.headers.get('Mcp-Session-Id')))
            if msg['method'] == 'initialize':
                result = {'serverInfo': {'name': 'fixture'}}
            elif msg['method'] == 'notifications/initialized':
                self.send_response(202); self.end_headers(); return
            elif msg.get('params', {}).get('cursor') == 'page-two':
                result = {'tools': [{'name': 'actual_search', 'inputSchema': {'type': 'object'}}]}
            else:
                result = {'tools': [{'name': 'other_tool'}], 'nextCursor': 'page-two'}
            body = json.dumps({'jsonrpc': '2.0', 'id': msg['id'], 'result': result}).encode()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Mcp-Session-Id', 'fixture-session')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers(); self.wfile.write(body)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Peer)
    worker = threading.Thread(target=server.serve_forever, daemon=True); worker.start()
    try:
        tools, info = P.list_http(f'http://127.0.0.1:{server.server_port}/mcp', {})
    finally:
        server.shutdown(); server.server_close(); worker.join()
    assert [t['name'] for t in tools] == ['other_tool', 'actual_search']
    assert info['name'] == 'fixture'
    pages = [(m, sid) for m, sid in requests if m['method'] == 'tools/list']
    assert len(pages) == 2
    assert pages[1][0]['params'] == {'cursor': 'page-two'}
    assert all(sid == 'fixture-session' for _, sid in pages)


def test_stdio_lists_all_tool_pages(tmp_path):
    peer = tmp_path / 'peer.py'
    peer.write_text('''import json,sys
for line in sys.stdin:
 m=json.loads(line)
 if m['method']=='notifications/initialized': continue
 if m['method']=='initialize': result={'serverInfo':{'name':'fixture'}}
 elif m.get('params',{}).get('cursor')=='next': result={'tools':[{'name':'actual_search'}]}
 else: result={'tools':[{'name':'other_tool'}],'nextCursor':'next'}
 print(json.dumps({'jsonrpc':'2.0','id':m['id'],'result':result}),flush=True)
''')
    tools, info = P.list_stdio(sys.executable, ['-u', str(peer)], {})
    assert [t['name'] for t in tools] == ['other_tool', 'actual_search']
    assert info['name'] == 'fixture'

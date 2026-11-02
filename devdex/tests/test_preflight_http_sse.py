"""Exercise public HTTP preflight against a real local SSE JSON-RPC peer."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import pytest
from devdex.harness import preflight as P

@pytest.mark.parametrize('framing', ['notification-after-reply', 'multiline-crlf', 'json'])
def test_http_preflight_reads_the_requested_rpc_reply(framing):
    calls = []
    class Peer(BaseHTTPRequestHandler):
        def log_message(self, *args): pass
        def do_POST(self):
            msg = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            calls.append(msg['method'])
            if msg['method'] == 'notifications/initialized':
                self.send_response(202); self.end_headers(); return
            result = ({'serverInfo': {'name':'fixture'}} if msg['method']=='initialize'
                      else {'tools':[{'name':'actual_search','inputSchema':{'type':'object'}}]})
            reply = {'jsonrpc':'2.0','id':msg['id'],'result':result}
            if framing == 'json':
                body=json.dumps(reply); ctype='application/json'
            elif framing == 'multiline-crlf':
                body=': keepalive\r\nevent: message\r\n' + ''.join(
                    'data: '+line+'\r\n' for line in json.dumps(reply,indent=2).splitlines())+'\r\n'
                ctype='text/event-stream'
            else:
                notification={'jsonrpc':'2.0','method':'notifications/progress','params':{'progress':1}}
                body='data: '+json.dumps(reply)+'\n\ndata: '+json.dumps(notification)+'\n\n'
                ctype='text/event-stream'
            raw=body.encode(); self.send_response(200)
            self.send_header('Content-Type',ctype);self.send_header('Content-Length',str(len(raw)))
            self.end_headers();self.wfile.write(raw)
    server=ThreadingHTTPServer(('127.0.0.1',0),Peer)
    worker=threading.Thread(target=server.serve_forever,daemon=True);worker.start()
    try: tools,info=P.list_http(f'http://127.0.0.1:{server.server_port}/mcp',{})
    finally: server.shutdown();server.server_close();worker.join()
    assert tools == [{'name':'actual_search','inputSchema':{'type':'object'}}]
    assert info == {'name':'fixture'}
    assert calls == ['initialize','notifications/initialized','tools/list']

"""Check SSE timing against a real local HTTP stream, including grouped events."""
import importlib.util
import json
from pathlib import Path
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

spec=importlib.util.spec_from_file_location('stream',Path(__file__).parents[1]/'benchmarks/kv_tier/stream.py')
stream=importlib.util.module_from_spec(spec);spec.loader.exec_module(stream)


class StreamTest(unittest.TestCase):
    def run_stream(self, grouped=False, output=32):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                self.send_response(200);self.end_headers()
                self.wfile.write(b'data: {"choices":[{"text":"","logprobs":null}]}\n\n')
                groups=[output] if grouped else [1]*output
                for n in groups:
                    event={'choices':[{'text':'','logprobs':{'tokens':['']*n}}]}
                    self.wfile.write(('data: '+json.dumps(event)+'\n\n').encode())
                    self.wfile.flush()
                event={'choices':[],'usage':{'prompt_tokens':len(body['prompt']),'completion_tokens':output}}
                self.wfile.write(('data: '+json.dumps(event)+'\n\ndata: [DONE]\n\n').encode())
            def log_message(self,*args):pass
        server=HTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.handle_request);thread.start()
        try: result=stream.request('http://127.0.0.1:'+str(server.server_port),[201]*1024)
        finally: thread.join(timeout=3);server.server_close()
        return result

    def test_empty_decoded_tokens_still_count(self):
        r=self.run_stream();self.assertTrue(r['success']);self.assertTrue(r['token_timing_exact'])
        self.assertEqual(len(r['token_intervals_ms']),31)
        self.assertGreaterEqual(r['ttft_ms'],0)

    def test_grouped_sse_is_not_exact_token_timing(self):
        r=self.run_stream(grouped=True);self.assertTrue(r['success']);self.assertFalse(r['token_timing_exact'])

    def test_short_generation_is_recorded_as_failure(self):
        r=self.run_stream(output=31);self.assertFalse(r['success']);self.assertFalse(r['token_timing_exact'])


if __name__=='__main__':unittest.main()

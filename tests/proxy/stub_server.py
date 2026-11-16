"""Stand-in for the reference inference server: one thread per request, no queue limit, no deadline.
A request costs CPU_MS outside a lock (decode / pre- and post-processing) and GPU_MS inside it."""
import json, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CPU_MS, GPU_MS = float(sys.argv[2]), float(sys.argv[3])
gpu = threading.Lock()
done = {"n": 0}

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body):
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self._send(200, b'{"status":"ok"}')
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0)); self.rfile.read(n)
        time.sleep(CPU_MS / 1000)
        with gpu:
            time.sleep(GPU_MS / 1000)
        done["n"] += 1
        try:
            self._send(200, json.dumps({"request_id": self.headers.get("X-Request-Id", "")}).encode())
        except OSError:
            pass

class S(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024

S(("127.0.0.1", int(sys.argv[1])), H).serve_forever()

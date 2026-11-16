"""Derive the staging variant of exam/template.yaml.

Usage: python3 make_template.py EXAM_TEMPLATE OUTPUT

The staging account has no GPU quota and cannot pull the exam's reference image, so the variant
swaps the GPU host for a small CPU host and gives the inference container a stand-in server.
Everything else (two-container task, proxy configuration, load balancer, security groups, health
checks, alarms) is the submitted template, untouched. Every edit must match exactly once.
"""
import sys

STUB = '''import json, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
gpu = threading.Lock()
class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _send(self, code, body):
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def do_GET(self):
        self._send(200, b'{"status":"ok"}')
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        time.sleep(0.03)
        with gpu:
            time.sleep(0.09)
        try:
            self._send(200, json.dumps({"request_id": self.headers.get("X-Request-Id", "")}).encode())
        except OSError:
            pass
class S(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 1024
print("stand-in inference server listening on 8080", flush=True)
S(("0.0.0.0", 8080), H).serve_forever()'''

INDENT = " " * 14
COMMAND = ("          Image: !Ref ReferenceImageUri\n"
           "          Command:\n            - python\n            - -u\n            - -c\n            - |\n"
           + "".join(INDENT + line + "\n" for line in STUB.splitlines()))

EDITS = [
    ("GPU host -> CPU host", "        InstanceType: g6.xlarge\n", "        InstanceType: t3.large\n"),
    ("no GPU on the host", '          ResourceRequirements: [{Type: GPU, Value: "1"}]\n', ""),
    ("no GPU support in the ECS agent", "            ECS_ENABLE_GPU_SUPPORT=true\n", ""),
    ("task CPU sized for t3.large", '      Cpu: "3072"\n', '      Cpu: "1536"\n'),
    ("task memory sized for t3.large", '      Memory: "12288"\n', '      Memory: "6144"\n'),
    ("stand-in server instead of the reference image's own", "          Image: !Ref ReferenceImageUri\n", COMMAND),
]

src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
for label, old, new in EDITS:
    if text.count(old) != 1:
        sys.exit(f"edit '{label}': expected exactly one match, found {text.count(old)}")
    text = text.replace(old, new)
    print(f"  staging edit: {label}")
open(dst, "w").write(text)

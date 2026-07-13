"""Dev fixture: a fake OpenAI-compatible lane for testing the rig without a GPU.

    python mock_lane.py [port] [tok_per_s]     # default 8872, ~40 tok/s

Then write results\\current-lane.json to point at it (or serve a real lane) and
run `qwen36.cmd bench` / race it from the dashboard. Streams word chunks at a
fixed cadence + a final usage chunk, mimicking vLLM/llama.cpp closely enough to
exercise racer.py, bench.py and dash\\serve.py end-to-end.
"""
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8872
TPS = float(sys.argv[2]) if len(sys.argv) > 2 else 40.0
WORDS = ("speculative decoding lets a small draft head propose several tokens "
         "which the target model verifies in one forward pass so the answer "
         "streams in bursts while staying exactly the model's own distribution ").split()


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        if self.path.endswith("/models"):
            body = json.dumps({"data": [{"id": "qwen36"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        req = json.loads(self.rfile.read(n) or b"{}")
        tokens = min(int(req.get("max_tokens", 64)), 96)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def chunk(delta, extra=None):
            obj = {"choices": [{"delta": delta, "index": 0}]}
            if extra:
                obj.update(extra)
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        time.sleep(0.35)                      # fake prefill -> measurable TTFT
        chunk({"role": "assistant"})
        for i in range(tokens):
            chunk({"content": WORDS[i % len(WORDS)] + " "})
            time.sleep(1.0 / TPS)
        self.wfile.write(("data: " + json.dumps(
            {"choices": [], "usage": {"completion_tokens": tokens, "prompt_tokens": 30}}
        ) + "\n\n").encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


if __name__ == "__main__":
    print(f"mock lane on :{PORT} at ~{TPS} tok/s")
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()

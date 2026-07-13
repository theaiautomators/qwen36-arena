"""Qwen3.6 A/B race dashboard — http://localhost:8870

DSpark-dashboard successor, engine-agnostic: races whatever lane qwen36.cmd is
currently serving (read from results\\current-lane.json, verified by a live
probe), records the best repeat's token timeline to results\\races\\<lane>.json,
and replays recordings side-by-side at true speed. Races survive browser
refreshes and lane swaps — the recording, not the tab, is the source of truth.

Stdlib only; run via  qwen36.cmd dash  (or  python dash\\serve.py).
"""
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from racer import lane_key, probe, stream_race  # noqa: E402

RESULTS = ROOT / "results"
RACES = RESULTS / "races"
PORT = 8870
LANE_PORTS = {"vllm": "http://localhost:8000/v1", "llamacpp": "http://localhost:8872/v1"}

clients = []          # list[queue.Queue] — one per open /events stream
clients_lock = threading.Lock()
race_lock = threading.Lock()


def emit(obj):
    data = json.dumps(obj)
    with clients_lock:
        for q in clients:
            q.put(data)


def load_races():
    races = {}
    if RACES.exists():
        for f in sorted(RACES.glob("*.json")):
            try:
                races[f.stem] = json.loads(f.read_text(encoding="utf-8-sig"))
            except Exception:
                pass
    return races


def active_lane():
    """The lane the launcher last started, verified alive. None if nothing answers."""
    f = RESULTS / "current-lane.json"
    if not f.exists():
        return None
    try:
        lane = json.loads(f.read_text(encoding="utf-8-sig"))
    except Exception:
        return None
    if probe(lane["base_url"], timeout=1.0) is None:
        return None
    return lane


def run_race(prompt, max_tokens, repeats):
    try:
        lane = active_lane()
        if lane is None:
            emit({"type": "error", "message": "no lane is answering — serve one first "
                  "(qwen36.cmd nvfp4 27b / gguf 27b / w4a16), wait for it to finish "
                  "loading, then race."})
            return
        key = lane_key(lane)
        emit({"type": "run_start", "regimes": [key], "prompt": prompt,
              "max_new_tokens": max_tokens, "repeats": repeats, "lane": lane})
        best = None          # (wall_tps, timeline, stats)
        ttft_cold = None
        for i in range(repeats):
            emit({"type": "repeat_start", "regime": key, "index": i, "total": repeats})
            timeline = []
            last_push = 0.0

            def snap(elapsed, text, tokens, tps):
                nonlocal last_push
                timeline.append({"elapsed": round(elapsed, 4), "text": text,
                                 "tokens": tokens, "tps": round(tps, 1)})
                now = time.perf_counter()
                if now - last_push > 0.045:   # ~22 pushes/s keeps browser work light
                    last_push = now
                    emit({"type": "tokens", "regime": key, "elapsed": elapsed,
                          "text": text, "tokens": tokens, "tps": tps})

            try:
                stats = stream_race(lane["base_url"], prompt, max_tokens=max_tokens,
                                    on_snapshot=snap)
            except Exception as e:
                emit({"type": "error", "message": f"{key} run {i+1}: {e}"})
                return
            if i == 0:
                ttft_cold = stats["ttft"]
            # final snapshot so the pane lands on the true totals
            timeline.append({"elapsed": stats["elapsed"], "text": stats["text"],
                             "tokens": stats["tokens"], "tps": stats["decode_tps"]})
            emit({"type": "tokens", "regime": key, "elapsed": stats["elapsed"],
                  "text": stats["text"], "tokens": stats["tokens"],
                  "tps": stats["decode_tps"]})
            emit({"type": "repeat_done", "regime": key, "index": i, "total": repeats,
                  "tps": stats["wall_tps"], "decode_tps": stats["decode_tps"],
                  "ttft": stats["ttft"]})
            if best is None or stats["wall_tps"] > best[0]:
                best = (stats["wall_tps"], timeline, stats)
        wall, timeline, stats = best
        summary = {"tps": wall, "decode_tps": stats["decode_tps"], "ttft": ttft_cold,
                   "tokens": stats["tokens"], "repeats": repeats,
                   "tokens_estimated": stats["tokens_estimated"]}
        emit({"type": "regime_done", "regime": key, **summary})
        RACES.mkdir(parents=True, exist_ok=True)
        rec = {"key": key, "lane": lane, "prompt": prompt, "max_tokens": max_tokens,
               "timeline": timeline,
               "summary": summary, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
        (RACES / f"{key}.json").write_text(json.dumps(rec))
        emit({"type": "run_done", "summary": {key: summary}})
    finally:
        race_lock.release()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            body = (Path(__file__).parent / "index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/state":
            lane = None
            f = RESULTS / "current-lane.json"
            if f.exists():
                try:
                    lane = json.loads(f.read_text(encoding="utf-8-sig"))
                except Exception:
                    pass
            up = {name: probe(url, timeout=0.6) for name, url in LANE_PORTS.items()}
            bench = None
            bf = RESULTS / "bench-results.json"
            if bf.exists():
                try:
                    bench = json.loads(bf.read_text(encoding="utf-8-sig"))["rows"]
                except Exception:
                    pass
            races_idx = {k: {"prompt": r["prompt"], "summary": r["summary"],
                             "max_tokens": r["max_tokens"], "saved_at": r["saved_at"],
                             "lane": r["lane"]}
                         for k, r in load_races().items()}
            self._json({"lane": lane, "up": up, "races": races_idx, "bench": bench,
                        "racing": race_lock.locked()})
        elif self.path == "/events":
            q = queue.Queue()
            with clients_lock:
                clients.append(q)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            try:
                restore = {"type": "restore", "races": load_races()}
                self.wfile.write(f"data: {json.dumps(restore)}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        data = q.get(timeout=15)
                        self.wfile.write(f"data: {data}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": ping\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionError, OSError):
                pass
            finally:
                with clients_lock:
                    if q in clients:
                        clients.remove(q)
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/start":
            n = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                body = {}
            prompt = (body.get("prompt") or "").strip()
            if not prompt:
                self._json({"started": False, "error": "empty prompt"})
                return
            if not race_lock.acquire(blocking=False):
                self._json({"started": False, "error": "a race is already running"})
                return
            t = threading.Thread(
                target=run_race,
                args=(prompt, int(body.get("max_new_tokens") or 256),
                      max(1, min(5, int(body.get("repeats") or 2)))),
                daemon=True)
            t.start()
            self._json({"started": True})
        elif self.path == "/clear":
            if RACES.exists():
                for f in RACES.glob("*.json"):
                    f.unlink()
            self._json({"cleared": True})
        else:
            self._json({"error": "not found"}, 404)


class DashServer(ThreadingHTTPServer):
    # On Windows SO_REUSEADDR lets a SECOND instance silently bind an already-used port,
    # then connections split randomly between the two (races appear/disappear per refresh).
    # Disable reuse on Windows so a double-launch fails loudly; keep it on POSIX (avoids
    # TIME_WAIT churn on quick restarts).
    allow_reuse_address = (sys.platform != "win32")


if __name__ == "__main__":
    RESULTS.mkdir(exist_ok=True)
    RACES.mkdir(parents=True, exist_ok=True)
    try:
        server = DashServer(("0.0.0.0", PORT), Handler)
    except OSError as e:
        sys.exit(f"could not bind port {PORT} ({e}). Is a dashboard already running? "
                 f"Stop it, or free the port, and retry.")
    print(f"QWEN36 A/B DASHBOARD READY  ->  http://localhost:{PORT}")
    print("serve a lane in another window (qwen36.cmd / ./qwen36.sh nvfp4 27b), then Start race.")
    server.serve_forever()

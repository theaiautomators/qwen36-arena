"""Tight A-B-A confirm of the two PEAK MTP configs (unlocked-clock best effort).

Measures nvfp4-mtp4 -> gguf-mtp4 -> nvfp4-mtp4 back-to-back (each ~2 min apart, vs
~15 min in the big sweep) so the thermal-drift confound is bounded even without a
clock lock. warmup 2 (past vLLM's cold-run poison) + n=8 median. Writes to
results\\confirm-results.json so the sweep data stays pristine. Prints the verdict.
"""
import json
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

Q = Path(__file__).resolve().parent; RESULTS = Q / "results"
VLLM_IMG = "vllm/vllm-openai:nightly"; LCPP_IMG = "ghcr.io/ggml-org/llama.cpp:server-cuda"
OUT = "confirm-locked-results.json"   # clock-locked run (GPU 2400 / mem 13801 pinned)


def write_lane(lane, engine, model, mtp, port, quant):
    (RESULTS / "current-lane.json").write_text(json.dumps({
        "lane": lane, "engine": engine, "model": model, "size": "27b", "mtp": mtp,
        "port": port, "base_url": f"http://localhost:{port}/v1", "served_name": "qwen36",
        "quant": quant, "started": "confirm"}))


def serve(lane, depth):
    subprocess.run(["docker", "rm", "-f", "qwen36-vllm", "qwen36-gguf"], capture_output=True)
    time.sleep(2)
    if lane == "nvfp4":
        model, port = "unsloth/Qwen3.6-27B-NVFP4", 8000
        cmd = ["docker", "run", "-d", "--rm", "--name", "qwen36-vllm", "--gpus", "all",
               "--ipc=host", "-p", "8000:8000", "-v", "qwen36-hf:/hf", "-e", "HF_HOME=/hf",
               "-e", "VLLM_WSL2_ENABLE_PIN_MEMORY=1", VLLM_IMG, "--model", model,
               "--served-model-name", "qwen36", "--kv-cache-dtype", "fp8",
               "--max-num-batched-tokens", "8192", "--max-num-seqs", "64",
               "--enable-auto-tool-choice", "--tool-call-parser", "qwen3_coder",
               "--default-chat-template-kwargs", '{"enable_thinking": false}',
               "--max-model-len", "65536", "--gpu-memory-utilization", "0.92",
               "--speculative-config", json.dumps({"method": "mtp", "num_speculative_tokens": depth})]
        write_lane("nvfp4", "vllm", model, depth, 8000, "W4A4")
    else:
        port = 8872
        inner = (f"FIRST=$(find /hf/gguf/27b-mtp -name '*.gguf' | sort | head -1); "
                 f"exec /app/llama-server -m $FIRST --alias qwen36 --host 0.0.0.0 --port 8080 "
                 f"-ngl 99 -c 65536 --jinja --spec-type draft-mtp --spec-draft-n-max {depth}")
        cmd = ["docker", "run", "-d", "--rm", "--name", "qwen36-gguf", "--gpus", "all",
               "-p", "8872:8080", "-v", "qwen36-hf:/hf", "--entrypoint", "/bin/bash",
               LCPP_IMG, "-c", inner]
        write_lane("gguf", "llamacpp", "unsloth/Qwen3.6-27B-MTP-GGUF", depth, 8872, "UD-Q4_K_XL")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"### DOCKER-RUN-FAIL {lane} mtp{depth}: {r.stderr[:200]}", flush=True); return None
    return port


def wait_ready(port, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2) as x:
                if b'"id"' in x.read():
                    return True
        except Exception:
            pass
        time.sleep(6)
    return False


for phase, lane, depth in [("A1", "nvfp4", 4), ("B", "gguf", 4), ("A2", "nvfp4", 4)]:
    print(f"### SERVE {phase} {lane} mtp{depth}", flush=True)
    port = serve(lane, depth)
    if not port or not wait_ready(port):
        print(f"### FAIL {phase} {lane} mtp{depth}", flush=True); continue
    print(f"### BENCH {phase} {lane} mtp{depth}", flush=True)
    subprocess.run([sys.executable, str(Q / "bench.py"), "--warmup", "2", "--repeats", "8",
                    "--stat", "median", "--save-text", "--out", OUT, "--presets", "code,math"])
    print(f"### DONE {phase} {lane} mtp{depth}", flush=True)

subprocess.run(["docker", "rm", "-f", "qwen36-vllm", "qwen36-gguf"], capture_output=True)

# verdict (A-B-A: split nvfp4 rows by ts into A1 early vs A2 late; B is gguf)
rows = json.loads((RESULTS / OUT).read_text(encoding="utf-8-sig"))["rows"]
def cell(key, preset):
    v = sorted(r["decode_tps"] for r in rows if r["key"] == key and r["preset"] == preset
               and not r.get("degenerate"))
    return v
print("\n### CONFIRM VERDICT (unlocked clocks, back-to-back A-B-A) ###", flush=True)
for preset in ("code", "math"):
    nv = sorted([r for r in rows if r["key"] == "nvfp4-27b-mtp4" and r["preset"] == preset
                 and not r.get("degenerate")], key=lambda r: r["ts"])
    gg = cell("gguf-27b-mtp4", preset)
    if not nv or not gg:
        continue
    half = len(nv) // 2
    a1 = statistics.median([r["decode_tps"] for r in nv[:half]])
    a2 = statistics.median([r["decode_tps"] for r in nv[half:]])
    nvm = statistics.median([r["decode_tps"] for r in nv])
    ggm = statistics.median(gg)
    drift = abs(a1 - a2) / a1 * 100
    print(f"  {preset}: nvfp4-d4 A1={a1:.1f} A2={a2:.1f} (drift {drift:.1f}%) median={nvm:.1f}"
          f"  |  gguf-d4={ggm:.1f}  ->  nvfp4/gguf = {nvm/ggm:.2f}x", flush=True)
print("### CONFIRM COMPLETE", flush=True)

"""Hardened re-run: per-engine MTP depth sweep + A-B-A drift bookend.

Settles F4's superlative — does GGUF+MTP still beat NVFP4+MTP at EACH engine's best
depth (not just depth 2)? Serves each (engine, depth) detached, health-polls, logs the
VRAM footprint, then runs the hardened bench (warmup 1 + n=5 + median + save-text +
degeneration gate). The two nvfp4-mtp2 bookends (first + last) test session thermal drift.

Docker flags mirror qwen36.cmd exactly (qwen3_coder parser, max-num-seqs 64, fp8 KV, etc).
Run detached via the Bash tool in the background; grep stdout for ### markers.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

Q = Path(__file__).resolve().parent
RESULTS = Q / "results"
VLLM_IMG = "vllm/vllm-openai:nightly"
LCPP_IMG = "ghcr.io/ggml-org/llama.cpp:server-cuda"


def write_lane(lane, engine, model, mtp, port, quant):
    (RESULTS / "current-lane.json").write_text(json.dumps({
        "lane": lane, "engine": engine, "model": model, "size": "27b", "mtp": mtp,
        "port": port, "base_url": f"http://localhost:{port}/v1", "served_name": "qwen36",
        "quant": quant, "started": "rerun"}))


def serve(lane, depth):
    subprocess.run(["docker", "rm", "-f", "qwen36-vllm", "qwen36-gguf"],
                   capture_output=True)
    time.sleep(2)
    if lane == "nvfp4":
        model, port = "unsloth/Qwen3.6-27B-NVFP4", 8000
        cmd = ["docker", "run", "-d", "--rm", "--name", "qwen36-vllm", "--gpus", "all",
               "--ipc=host", "-p", "8000:8000", "-v", "qwen36-hf:/hf", "-e", "HF_HOME=/hf",
               "-e", "VLLM_WSL2_ENABLE_PIN_MEMORY=1", VLLM_IMG,
               "--model", model, "--served-model-name", "qwen36",
               "--kv-cache-dtype", "fp8", "--max-num-batched-tokens", "8192",
               "--max-num-seqs", "64", "--enable-auto-tool-choice",
               "--tool-call-parser", "qwen3_coder",
               "--default-chat-template-kwargs", '{"enable_thinking": false}',
               "--max-model-len", "65536", "--gpu-memory-utilization", "0.92",
               "--speculative-config",
               json.dumps({"method": "mtp", "num_speculative_tokens": depth})]
        write_lane("nvfp4", "vllm", model, depth, 8000, "W4A4")
    else:
        port = 8872
        inner = (f"FIRST=$(find /hf/gguf/27b-mtp -name '*.gguf' | sort | head -1); "
                 f"echo serving $FIRST; exec /app/llama-server -m $FIRST --alias qwen36 "
                 f"--host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --jinja "
                 f"--spec-type draft-mtp --spec-draft-n-max {depth}")
        cmd = ["docker", "run", "-d", "--rm", "--name", "qwen36-gguf", "--gpus", "all",
               "-p", "8872:8080", "-v", "qwen36-hf:/hf", "--entrypoint", "/bin/bash",
               LCPP_IMG, "-c", inner]
        write_lane("gguf", "llamacpp", "unsloth/Qwen3.6-27B-MTP-GGUF", depth, 8872,
                   "UD-Q4_K_XL")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"### DOCKER-RUN-FAIL {lane} mtp{depth}: {r.stderr[:200]}", flush=True)
        return None
    return port


def wait_ready(port, timeout=1200):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models",
                                        timeout=2) as x:
                if b'"id"' in x.read():
                    return True
        except Exception:
            pass
        time.sleep(6)
    return False


def footprint(phase, lane, depth):
    r = subprocess.run(["nvidia-smi",
                        "--query-gpu=memory.used,clocks.sm,clocks.mem,temperature.gpu",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    line = r.stdout.strip().splitlines()[0] if r.stdout.strip() else "?,?,?,?"
    with open(RESULTS / "footprints.csv", "a") as f:
        f.write(f"{phase},{lane},{depth},{line}\n")


# depth sweep both engines + A-B-A bookend (nvfp4-mtp2 first & last)
CONFIGS = [
    ("nvfp4", 2, "bookendA"),
    ("nvfp4", 1, "sweep"), ("nvfp4", 3, "sweep"), ("nvfp4", 4, "sweep"),
    ("gguf", 2, "sweep"),
    ("gguf", 1, "sweep"), ("gguf", 3, "sweep"), ("gguf", 4, "sweep"),
    ("nvfp4", 2, "bookendB"),
]

for lane, depth, phase in CONFIGS:
    print(f"### SERVE {lane} mtp{depth} ({phase})", flush=True)
    port = serve(lane, depth)
    if not port or not wait_ready(port):
        print(f"### FAIL {lane} mtp{depth} ({phase}) — did not come up", flush=True)
        continue
    footprint(phase, lane, depth)
    print(f"### BENCH {lane} mtp{depth} ({phase})", flush=True)
    subprocess.run([sys.executable, str(Q / "bench.py"), "--warmup", "1",
                    "--repeats", "5", "--stat", "median", "--save-text",
                    "--presets", "code,math"])
    print(f"### DONE {lane} mtp{depth} ({phase})", flush=True)

subprocess.run(["docker", "rm", "-f", "qwen36-vllm", "qwen36-gguf"], capture_output=True)
print("### RERUN COMPLETE", flush=True)

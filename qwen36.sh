#!/usr/bin/env bash
# Qwen3.6 NVFP4-vs-GGUF A/B rig (Linux/macOS twin of qwen36.cmd).
# Three lanes, one served name ("qwen36"), one client-side measuring stick:
#   ./qwen36.sh nvfp4 [27b|35b] [mtp N]   Unsloth NVFP4 W4A4   -> vLLM      http://localhost:8000/v1
#   ./qwen36.sh w4a16 [27b]      [mtp N]   NVIDIA NVFP4 W4A16   -> vLLM      http://localhost:8000/v1  (the 2.5x claim's own baseline)
#   ./qwen36.sh gguf  [27b|35b] [mtp N]   Unsloth UD-Q4_K_XL   -> llama.cpp http://localhost:8872/v1
#   ./qwen36.sh dash                       race dashboard       -> http://localhost:8870
#   ./qwen36.sh bench [args]               scripted battery vs the running lane (-> results/bench-results.json)
#   ./qwen36.sh download                   pre-fetch every lane's weights into the qwen36-hf volume
#   ./qwen36.sh status | stop
# One model on the GPU at a time (27B pairs don't co-fit in 32 GB). mtp N = multi-token-prediction
# speculative decoding, the solo-speed lever (sweep 1..6). Ready when vLLM prints "Application
# startup complete" / llama.cpp prints "server is listening".
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_IMG="${VLLM_IMG:-vllm/vllm-openai:nightly}"
LCPP_IMG="${LCPP_IMG:-ghcr.io/ggml-org/llama.cpp:server-cuda}"
PY="${PYTHON:-python3}"

# VLLM_WSL2_ENABLE_PIN_MEMORY is only needed on Docker Desktop/WSL2 (the nightly's GPU worker
# dies with "UVA is not available" without it); it is harmless on native Linux.
VCOMMON=(--rm --name qwen36-vllm --gpus all --ipc=host -p 8000:8000 -v qwen36-hf:/hf -e HF_HOME=/hf -e VLLM_WSL2_ENABLE_PIN_MEMORY=1)
# Blackwell/Qwen3.6 gotchas baked in (see docs/METHODOLOGY.md):
#  - fp8 KV on this hybrid-attention arch needs --max-num-batched-tokens 8192
#  - Qwen3.6 emits Qwen-Coder XML tool calls -> --tool-call-parser qwen3_coder (hermes silently fails)
#  - --max-num-seqs 64: the hybrid Mamba cache fits ~121 decode seqs at this VRAM; vLLM's default
#    256 overflows it and crashes at CUDA-graph capture (bites non-MTP lanes; MTP auto-lowers it)
VFLAGS=(--served-model-name qwen36 --kv-cache-dtype fp8 --max-num-batched-tokens 8192 --max-num-seqs 64
        --enable-auto-tool-choice --tool-call-parser qwen3_coder
        --default-chat-template-kwargs '{"enable_thinking": false}')

write_lane(){ # lane engine model size mtp port quant
  mkdir -p "$ROOT/results"
  cat > "$ROOT/results/current-lane.json" <<EOF
{"lane": "$1", "engine": "$2", "model": "$3", "size": "$4", "mtp": $5, "port": $6, "base_url": "http://localhost:$6/v1", "served_name": "qwen36", "quant": "$7", "started": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"}
EOF
}

LANE="${1:-}"; SIZE="${2:-27b}"; MTP=0
[ "${2:-}" = "mtp" ] && { SIZE=27b; MTP="${3:-0}"; }
[ "${3:-}" = "mtp" ] && MTP="${4:-0}"

serve_vllm(){ # model quant size
  local model="$1" quant="$2" size="$3" ctx gmu spec=() moe=() arch=()
  if [ "$size" = "35b" ]; then ctx=98304; gmu=0.94; else ctx=65536; gmu=0.92; fi
  [ "$MTP" != "0" ] && spec=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$MTP}")
  # 35B-A3B MoE on consumer SM120: vLLM's picker only knows data-center Blackwell and silently
  # falls back to slow Marlin (vllm#33416) -> force FlashInfer. The dense 27B is immune.
  if [ "$size" = "35b" ]; then moe=(--moe-backend flashinfer); arch=(-e FLASHINFER_CUDA_ARCH_LIST=12.0); fi
  write_lane "$LANE" vllm "$model" "$size" "$MTP" 8000 "$quant"
  echo; echo "Lane $LANE $size mtp=$MTP  --  $model"
  echo "  vLLM on http://localhost:8000/v1 (served name: qwen36) - ready on \"Application startup complete\"."
  echo "  First start compiles kernels for a few minutes. Ctrl+C stops it."; echo
  docker rm -f qwen36-vllm >/dev/null 2>&1 || true
  exec docker run "${VCOMMON[@]}" "${arch[@]}" "$VLLM_IMG" --model "$model" "${VFLAGS[@]}" \
    --max-model-len "$ctx" --gpu-memory-utilization "$gmu" "${moe[@]}" "${spec[@]}"
}

case "$LANE" in
  nvfp4)
    if [ "$SIZE" = "35b" ]; then model="unsloth/Qwen3.6-35B-A3B-NVFP4-Fast"; else model="unsloth/Qwen3.6-27B-NVFP4"; fi
    serve_vllm "$model" W4A4 "$SIZE" ;;
  w4a16)
    serve_vllm "nvidia/Qwen3.6-27B-NVFP4" W4A16 27b ;;
  gguf)
    if [ "$MTP" != "0" ]; then gdir="$SIZE-mtp"; else gdir="$SIZE-std"; fi
    if [ "$SIZE" = "35b" ]; then gbase="unsloth/Qwen3.6-35B-A3B"; else gbase="unsloth/Qwen3.6-27B"; fi
    if [ "$MTP" != "0" ]; then grepo="$gbase-MTP-GGUF"; else grepo="$gbase-GGUF"; fi
    gspec=""; [ "$MTP" != "0" ] && gspec="--spec-type draft-mtp --spec-draft-n-max $MTP"
    write_lane gguf llamacpp "$grepo" "$SIZE" "$MTP" 8872 UD-Q4_K_XL
    echo; echo "Lane gguf $SIZE mtp=$MTP  --  /hf/gguf/$gdir (UD-Q4_K_XL)"
    echo "  llama.cpp on http://localhost:8872/v1 (served name: qwen36) - ready on \"server is listening\"."
    echo "  Ctrl+C stops it."; echo
    docker rm -f qwen36-gguf >/dev/null 2>&1 || true
    exec docker run --rm --name qwen36-gguf --gpus all -p 8872:8080 -v qwen36-hf:/hf \
      --entrypoint /bin/bash "$LCPP_IMG" -c \
      "FIRST=\$(find /hf/gguf/$gdir -name '*.gguf' | sort | head -1); echo serving \$FIRST; exec /app/llama-server -m \$FIRST --alias qwen36 --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --jinja $gspec" ;;
  dash)
    echo "Dashboard: http://localhost:8870  (Ctrl+C stops it)"
    exec "$PY" "$ROOT/dash/serve.py" ;;
  bench)
    shift; exec "$PY" "$ROOT/bench.py" "$@" ;;
  download)
    exec docker run --rm --name qwen36-dl -v qwen36-hf:/hf -v "$ROOT:/workspace" -e HF_HOME=/hf \
      python:3.11-slim bash -c "pip install -q --no-cache-dir 'huggingface_hub>=0.26' && python /workspace/download_models.py" ;;
  status)
    docker ps --filter name=qwen36 --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"; echo
    curl -s -m 2 http://localhost:8000/v1/models >/dev/null && echo "vLLM lane :8000 - up" || echo "vLLM lane :8000 - down"
    curl -s -m 2 http://localhost:8872/v1/models >/dev/null && echo "llama.cpp lane :8872 - up" || echo "llama.cpp lane :8872 - down"
    [ -f "$ROOT/results/current-lane.json" ] && cat "$ROOT/results/current-lane.json" ;;
  stop)
    docker rm -f qwen36-vllm qwen36-gguf 2>/dev/null || true ;;
  *)
    echo "usage: ./qwen36.sh [nvfp4 | w4a16 | gguf] [27b | 35b] [mtp N]"
    echo "       ./qwen36.sh [dash | bench | download | status | stop]" ;;
esac

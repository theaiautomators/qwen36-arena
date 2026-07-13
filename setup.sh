#!/usr/bin/env bash
# qwen36-arena setup (Linux/macOS): create the shared model volume + mark scripts executable.
# No upstream repo to clone (unlike dspark-arena) — the models come straight from Hugging Face.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker volume create qwen36-hf >/dev/null
chmod +x "$ROOT"/qwen36.sh "$ROOT"/*.sh 2>/dev/null || true
cat <<'EOF'

Done — created the qwen36-hf model volume. Next steps:
  ./qwen36.sh download          # pre-fetch the Qwen3.6 lanes into the volume (27B set first; ~130 GB all-in, one-time)
  ./qwen36.sh nvfp4 27b mtp 4   # serve the tuned NVFP4 lane   -> http://localhost:8000/v1
  ./qwen36.sh dash              # race dashboard               -> http://localhost:8870
  ./qwen36.sh bench             # scripted battery vs the running lane

Reproduce the video's headline finding (the MTP depth sweep):
  python3 rerun_depth_sweep.py  &&  python3 analyze_rerun.py
EOF

"""Pre-fetch every Qwen3.6 A/B lane into the qwen36-hf Docker volume (HF_HOME layout),
priority-ordered so the hero 27B lanes are testable before the 35B set lands.

vLLM lanes cache under /hf/hub (normal HF cache, served by repo id).
GGUF lanes land in fixed dirs /hf/gguf/<lane>/ so qwen36.cmd can glob the first part.
"""
import json
import os
from huggingface_hub import snapshot_download

HF = os.environ.get("HF_HOME", "/hf")

# (repo, gguf_lane_dir_or_None, allow_patterns_or_None)
JOBS = [
    # -- hero 27B lanes first --
    ("unsloth/Qwen3.6-27B-NVFP4",          None,       None),                  # vLLM W4A4 (the 2.5x artifact)  ~16 GB
    ("unsloth/Qwen3.6-27B-GGUF",           "27b-std",  ["*UD-Q4_K_XL*"]),      # llama.cpp, MTP off             ~16 GB
    ("unsloth/Qwen3.6-27B-MTP-GGUF",       "27b-mtp",  ["*UD-Q4_K_XL*"]),      # llama.cpp, MTP lever           ~16 GB
    ("nvidia/Qwen3.6-27B-NVFP4",           None,       None),                  # vLLM W4A16 claim-baseline      ~22 GB
    # -- 35B-A3B MoE second wave --
    ("unsloth/Qwen3.6-35B-A3B-NVFP4-Fast", None,       None),                  # vLLM W4A4 (1.79x variant)      ~20 GB
    ("unsloth/Qwen3.6-35B-A3B-GGUF",       "35b-std",  ["*UD-Q4_K_XL*"]),      # ~19 GB
    ("unsloth/Qwen3.6-35B-A3B-MTP-GGUF",   "35b-mtp",  ["*UD-Q4_K_XL*"]),      # ~19 GB
]

os.makedirs(os.path.join(HF, "gguf"), exist_ok=True)
manifest = {}
for repo, lane, patterns in JOBS:
    print(f"=== downloading {repo}", flush=True)
    if lane:
        dest = os.path.join(HF, "gguf", lane)
        path = snapshot_download(repo, allow_patterns=patterns + ["*.json", "README.md"],
                                 local_dir=dest)
        ggufs = sorted(
            os.path.join(dp, f)
            for dp, _, fs in os.walk(dest) for f in fs if f.endswith(".gguf")
        )
        if not ggufs:
            raise SystemExit(f"!! no .gguf matched {patterns} in {repo}")
        manifest[lane] = {"repo": repo, "first_part": ggufs[0], "files": ggufs}
        print(f"    -> {ggufs[0]} (+{len(ggufs)-1} more parts)", flush=True)
    else:
        path = snapshot_download(repo)
        print(f"    -> {path}", flush=True)
    # checkpoint the manifest after every job so a mid-run stop still leaves valid state
    with open(os.path.join(HF, "gguf", "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

print("ALL_DONE", flush=True)

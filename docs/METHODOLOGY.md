# Methodology

How Qwen36 Arena measures, why the numbers are trustworthy, and the story of a headline
finding that was **wrong at first** — caught by an adversarial review and reversed by a
hardened re-run. If you only read one section, read *"The depth-2 trap"*.

## The measuring stick

Every lane — vLLM (NVFP4 / W4A16) and llama.cpp (GGUF) — is treated as an identical black box:
an OpenAI-compatible `/v1/chat/completions` stream, timed **client-side** by `racer.py`. No
engine's self-reported throughput is used, only its authoritative **token count**
(`usage.completion_tokens` for vLLM, `timings.predicted_n` for llama.cpp). That single shared
clock is what makes cross-engine numbers comparable.

Headline metric = **decode tok/s** = `(tokens − 1) / (t_last_token − t_first_token)` — it
excludes prefill/TTFT, so it's cache-immune and reflects what streaming actually feels like.
Fixed conditions for every lane: **greedy (temp 0) · thinking off · 256 new tokens · 3 fixed
prompts (code / math / chat)**.

Hardened settings (added after the red-team, below):
- **`--warmup` discard** — the first request(s) after a fresh serve ramp as vLLM's CUDA graphs
  and FlashInfer autotune settle; discard them so they don't drag the median.
- **`--stat median` over n=5** — best-of-N flatters the luckiest boost-clock run; median is the
  honest estimator for a robustness pass, reported with the min–max spread.
- **Degeneration gate** — every generation is checked for repetition loops / tiny vocabulary /
  early stop. A fast-but-garbage run can post a huge tok/s that means nothing; it's flagged and
  excluded from the aggregate.

## The three lanes

| lane | engine | weights | quant | what it answers |
|---|---|---|---|---|
| `nvfp4` | vLLM nightly | `unsloth/Qwen3.6-27B-NVFP4` | NVFP4 **W4A4** | the "2.5×" artifact under test |
| `w4a16` | vLLM nightly | `nvidia/Qwen3.6-27B-NVFP4` | NVFP4 **W4A16** | the baseline Unsloth's 2.5× is *measured against* |
| `gguf` | llama.cpp | `unsloth/Qwen3.6-27B[-MTP]-GGUF` | **UD-Q4_K_XL** | the lane most people already run |

NVFP4 only exists for vLLM-class engines; GGUF is llama.cpp-native — a same-framework
NVFP4-vs-GGUF isn't a thing. So the rig offers both honest shapes: the **same-engine claim
check** (`nvfp4` vs `w4a16`, both vLLM, both NVFP4-family) and the **same-model cross-engine
race** (`nvfp4` vLLM vs `gguf` llama.cpp), everything else pinned.

## The depth-2 trap (the red-team story)

The first pass compared both engines at **MTP draft depth 2** and concluded *"GGUF+MTP is the
fastest solo lane"* (GGUF ~112 vs NVFP4 ~90). An adversarial review flagged the load-bearing
weakness: **only one depth was tested.** So we swept MTP depth **1–6 on both engines** (n=5
median, warmup-discarded, with an A-B-A thermal-drift bookend):

| MTP depth | 1 | 2 | 3 | **4** | 5 | 6 |
|---|---|---|---|---|---|---|
| NVFP4 · vLLM (code tok/s) | 70 | 98 | 107 | **140** | 138 | 136 |
| GGUF · llama.cpp | 93 | 110 | 109 | **114** | 109 | 85 |

The engines respond to depth completely differently: **NVFP4 rewards deep speculation** (2×,
70→140) and plateaus; **GGUF saturates by depth 2** (~110–114) then **degrades** (over-speculation
costs more than it saves — 85 at depth 6). At depth 2 GGUF is ahead (110 vs 98); at each engine's
**best** depth (4), **NVFP4 wins ~1.2–1.3×**. The original superlative was a depth-2 artifact. The
lesson that goes on camera: **draft depth is the biggest under-documented solo-speed lever — check
it before you conclude anything about which lane is faster.**

Reproduce: `rerun_depth_sweep.py` (the sweep + drift bookend) → `analyze_rerun.py` (per-depth
medians, best depth per engine, the head-to-head verdict + the A-B-A drift number).

## Why the ranking is trustworthy (artifact-killers)

- **Token parity.** Every run of every lane emitted exactly 256 tokens (~3.8 chars/token both
  engines) — the two engines count the same quantity.
- **Independent second clock.** llama.cpp's own server-side `eval` timer matches `racer.py`'s
  client-side number to **≤0.4%** on the GGUF lanes. vLLM's chunk cadence reconciles the same way.
- **Output validity.** Zero degenerate runs — the gate confirmed every measured tok/s is real,
  coherent, full-length text (saved with `--save-text` for audit).
- **Confirmed three ways.** The sweep, a back-to-back `confirm_peak.py` re-measure (peaks measured
  ~2 min apart to bound drift), and a **clock-locked** run all put NVFP4-tuned ahead by ~1.2–1.3×.

## Warm-up vs boost: what the ±5–8% actually is

The A-B-A bookend (`nvfp4-mtp` measured first and last) showed 5–11% variation — but with the GPU
clocks **locked**, it *didn't shrink*, and the runs ramp **monotonically** cold→warm. So the
variance is **vLLM engine warm-up** (CUDA graphs / FlashInfer autotune / caches reaching steady
state over the first several requests), **not** GPU boost jitter. The fix is more warm-up
requests, not clock-locking. The math-task numbers and the ranking are rock-solid; only the exact
NVFP4 code *peak* wobbles (~130–140). Clock-locking (via `nvidia-smi --lock-gpu-clocks` +
`--lock-memory-clocks`, admin) pins the on-screen number but changes no conclusion.

## The "2.5×" reconciliation (say this on camera)

Unsloth's headline **2.5×** is real, but it's **batched throughput** — 128 concurrent requests, on
one data-centre **B200**, vs NVIDIA's **W4A16** (their doc table shows ~2.85× there). It is *not*
what a solo user feels. Measured on one 5090, batch 1:

- The **same** comparison (NVFP4 vs W4A16), no MTP → **0.94×** (a wash). Single-user decode is
  memory-bandwidth-bound; the W4A4 compute win only pays off batched — and the "W4A4" checkpoint
  is actually ~7% heavier in VRAM (it carries FP8 W8A8 layers: `20.0 / 21.34 GiB ≈ 0.937`, almost
  exactly the measured ratio).
- The **~2×** solo speedup you *can* get comes from **MTP draft depth** (68 → 140 tok/s), not the
  quant.

Frame it warmly: the 2.5× isn't fake, it's a *different axis*. That axis distinction is the most
useful thing the video teaches.

## Gotchas baked into the launcher (each would fail silently on camera)

1. **`--tool-call-parser qwen3_coder`, not `hermes`.** Qwen3.6 emits Qwen-Coder **XML** tool calls
   (`<function=name><parameter=k>value</parameter>`), not Hermes JSON. With `hermes`, vLLM fails to
   parse them → `finish_reason: stop` + raw text, so an agent never sees the tool call and its loop
   stalls silently. (`qwen3_xml` is the same parser.)
2. **`--max-num-seqs 64`.** Qwen3.6's hybrid Mamba cache fits ~121 decode sequences at this VRAM;
   vLLM's default 256 overflows it → a hard crash at CUDA-graph capture (`exceeds available Mamba
   cache blocks`). Bites the non-MTP lanes; MTP auto-lowers it and dodges it. 64 is ample single-user.
3. **`vllm bench serve` needs `--tokenizer <real-repo>`** when the server uses a served-name alias
   (`--model qwen36` alone makes the bench try to load the tokenizer "qwen36" from HF and error).
4. **NVFP4 kernel check.** Both NVFP4 lanes must select `FlashInferCutlassNvFp4LinearKernel`
   (`arch=sm120`), *not* the slow Marlin fallback — verify in the serve log. The dense 27B auto-picks
   the fast path; the 35B-A3B MoE needs `--moe-backend flashinfer` + `FLASHINFER_CUDA_ARCH_LIST=12.0`
   on consumer SM120 (baked in). (The `w4a16` baseline runs Marlin weight-only — normal for that quant.)

## Environment

RTX 5090 (32 GB, SM120) · Docker Desktop / WSL2 · `vllm/vllm-openai:nightly` `0.23.1rc1.dev1060`
(2026-07-13) · `ghcr.io/ggml-org/llama.cpp:server-cuda`. WSL2/Docker-Desktop needs
`VLLM_WSL2_ENABLE_PIN_MEMORY=1` or vLLM's GPU worker dies with "UVA is not available" (baked in;
harmless on native Linux). Do **not** pair the llama.cpp lane with CUDA 13.2 (gibberish outputs —
Unsloth docs); the pinned image is fine.

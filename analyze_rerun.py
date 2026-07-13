"""Analyze the MTP depth sweep and print the head-to-head verdict.

For each engine: the best MTP depth by median code decode_tps, then the two engines
compared at their OWN best depth (the fair test). Also: the A-B-A thermal-drift check
and any degenerate rows.

Reads results/bench-results.json (produced by rerun_depth_sweep.py). On a fresh clone
that file won't exist yet, so it falls back to the shipped reference data
(results/sample-depth-sweep.json) and says so.
"""
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
live, sample = R / "bench-results.json", R / "sample-depth-sweep.json"
if live.exists():
    src = live
elif sample.exists():
    src = sample
    print(f"(no results/bench-results.json yet - showing the SHIPPED REFERENCE data from\n"
          f" {sample.name}. Run  python rerun_depth_sweep.py  to generate your own.)\n")
else:
    sys.exit("no results/bench-results.json and no results/sample-depth-sweep.json - "
             "run  python rerun_depth_sweep.py  first (after ./qwen36.sh download).")
rows = json.loads(src.read_text(encoding="utf-8-sig"))["rows"]

# group median decode_tps by (key, preset); track degenerates + raw for drift
by = defaultdict(list)
deg = []
for r in rows:
    if r.get("degenerate"):
        deg.append((r["key"], r["preset"], r["run"], r["degenerate"]))
        continue
    by[(r["key"], r["preset"])].append(r["decode_tps"])

def med(k, p):
    v = by.get((k, p))
    return statistics.median(v) if v else None

def depth_of(key):
    import re
    m = re.search(r"mtp(\d+)", key)
    return int(m.group(1)) if m else 0

print("=== per-depth median decode tok/s (code / math) ===")
lanes = sorted({k for k, _ in by})
for k in lanes:
    c, m = med(k, "code"), med(k, "math")
    n = len(by.get((k, "code"), []))
    cs = f"{c:.1f}" if c else "  - "
    ms = f"{m:.1f}" if m else "  - "
    print(f"  {k:<20} code {cs:>6}  math {ms:>6}   (n={n})")

# best depth per engine on CODE
best = {}
for eng in ("nvfp4", "gguf"):
    cand = [(k, med(k, "code")) for k in lanes if k.startswith(eng) and med(k, "code")]
    if cand:
        best[eng] = max(cand, key=lambda x: x[1])

print("\n=== best depth per engine (by code median) ===")
for eng, (k, v) in best.items():
    print(f"  {eng:<6} best = {k} @ {v:.1f} tok/s (depth {depth_of(k)})")

if "nvfp4" in best and "gguf" in best:
    gk, gv = best["gguf"]; nk, nv = best["nvfp4"]
    ratio = nv / gv
    print(f"\n=== HEAD-TO-HEAD (each engine at its OWN best depth) ===")
    print(f"  nvfp4 best {nv:.1f} (depth {depth_of(nk)})  vs  gguf best {gv:.1f} (depth {depth_of(gk)})"
          f"  ->  nvfp4/gguf = {ratio:.2f}x")
    if ratio > 1.05:
        print(f"  VERDICT: NVFP4+MTP is the fastest solo lane once depth-tuned ({ratio:.2f}x over GGUF).")
    elif ratio < 0.95:
        print(f"  VERDICT: GGUF+MTP is fastest even at each engine's best depth ({1/ratio:.2f}x over NVFP4).")
    else:
        print(f"  VERDICT: a tie at best depth ({ratio:.2f}x) - neither lane is clearly faster.")

# A-B-A drift: nvfp4-27b-mtp2 code, split by timestamp (bookendA early vs bookendB late).
# The sweep measures nvfp4-mtp2 first AND last; if they agree, session thermal/warm-up
# drift didn't bias the result. 5-11% is expected vLLM warm-up (see METHODOLOGY), not alarming.
mtp2 = sorted([r for r in rows if r["key"] == "nvfp4-27b-mtp2" and r["preset"] == "code"
               and not r.get("degenerate")], key=lambda r: r["ts"])
if len(mtp2) >= 6:
    half = len(mtp2) // 2
    a = statistics.median([r["decode_tps"] for r in mtp2[:half]])
    b = statistics.median([r["decode_tps"] for r in mtp2[half:]])
    drift = abs(a - b) / a * 100
    note = "steady" if drift < 6 else "within the expected vLLM warm-up band (see METHODOLOGY)"
    print(f"\n=== A-B-A drift check (nvfp4-mtp2 code, first vs last bookend) ===")
    print(f"  bookendA {a:.1f}  vs  bookendB {b:.1f}  ->  drift {drift:.1f}%  ({note})")

if deg:
    print(f"\n=== degenerate rows (excluded): {len(deg)} ===")
    for d in deg:
        print(f"  {d}")

fp = R / "footprints.csv"
if fp.exists():
    print("\n=== footprints ===")
    print(fp.read_text().strip())

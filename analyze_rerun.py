"""Analyze the hardened depth-sweep re-run and adjudicate F4's superlative.

For each engine: best MTP depth by median code decode_tps. Then compares the two
engines at their OWN best depth (the fair superlative test). Also: A-B-A drift check
(nvfp4-mtp2 code, first bookend vs last), degenerate-row flags, footprints.
"""
import json
import statistics
from collections import defaultdict
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
rows = json.loads((R / "bench-results.json").read_text(encoding="utf-8-sig"))["rows"]

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
    ratio = gv / nv
    print(f"\n=== F4 SUPERLATIVE TEST (each engine at its OWN best depth) ===")
    print(f"  gguf best {gv:.1f} (d{depth_of(gk)})  vs  nvfp4 best {nv:.1f} (d{depth_of(nk)})"
          f"  ->  gguf/nvfp4 = {ratio:.2f}x")
    if ratio > 1.05:
        print(f"  VERDICT: F4 SUPERLATIVE HOLDS — GGUF+MTP is fastest even at each engine's best depth.")
    elif ratio < 0.95:
        print(f"  VERDICT: F4 SUPERLATIVE FLIPS — NVFP4+MTP wins once depth-tuned. REWRITE the claim.")
    else:
        print(f"  VERDICT: TOO CLOSE ({ratio:.2f}x) — call it a tie at best depth, drop the superlative.")

# A-B-A drift: nvfp4-27b-mtp2 code, split by timestamp (bookendA early vs bookendB late)
mtp2 = sorted([r for r in rows if r["key"] == "nvfp4-27b-mtp2" and r["preset"] == "code"
               and not r.get("degenerate")], key=lambda r: r["ts"])
if len(mtp2) >= 6:
    half = len(mtp2) // 2
    a = statistics.median([r["decode_tps"] for r in mtp2[:half]])
    b = statistics.median([r["decode_tps"] for r in mtp2[half:]])
    print(f"\n=== A-B-A drift check (nvfp4-mtp2 code, first vs last bookend) ===")
    print(f"  bookendA {a:.1f}  vs  bookendB {b:.1f}  ->  drift {abs(a-b)/a*100:.1f}%"
          f"  ({'OK, <3% = no session drift' if abs(a-b)/a < 0.03 else 'WARN >3% drift'})")

print(f"\n=== degenerate rows: {len(deg)} ===")
for d in deg:
    print(f"  {d}")

fp = R / "footprints.csv"
if fp.exists():
    print("\n=== footprints ===")
    print(fp.read_text().strip())

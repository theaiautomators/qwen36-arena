"""Scripted battery vs the RUNNING lane — the numbers that go in the report.

Method (mirrors the DSpark A/B): single stream, greedy (temp 0), thinking off,
256 new tokens, the three fixed presets, best-of-2 per preset, wall-clock
measured client-side. Headline = decode tok/s (cache-immune); TTFT is quoted
from run 1 only (run once per fresh serve for a clean cold TTFT).

  qwen36.cmd bench                       # all three presets, best-of-2
  qwen36.cmd bench --presets code        # one preset
  qwen36.cmd bench --max-tokens 1024     # longer runs
  qwen36.cmd bench --vs gguf-27b         # speedup column vs a specific lane

Rows append to results\bench-results.json (lane-tagged, never overwritten), so
serve lane -> bench -> stop -> serve next lane -> bench builds the full table.
"""
import argparse
import json
import sys
import time
from pathlib import Path

# Windows consoles/pipes default to cp1252, which can't encode the status glyphs and
# raises UnicodeEncodeError mid-run (worst when stdout is piped, e.g. from the sweep
# driver). Force UTF-8 so output never crashes the benchmark. (No-op if already UTF-8.)
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).parent))
from racer import PRESETS, lane_key, probe, stream_race

ROOT = Path(__file__).parent
RESULTS = ROOT / "results"
OUT = RESULTS / "bench-results.json"


def degeneration(text):
    """Cheap guard against fast-but-garbage output inflating tok/s. Returns a short
    reason string if the text looks degenerate (heavy repetition / tiny vocabulary),
    else ''. A lane that loops 'the the the' can post a huge tok/s that means nothing."""
    words = text.split()
    if len(words) < 20:
        return "very short (<20 words) - did it stop early?"
    uniq = len(set(words)) / len(words)
    if uniq < 0.15:
        return f"low vocab (unique/total={uniq:.2f}) - likely a repetition loop"
    # longest run of one immediately-repeated line
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    run = mx = 1
    for a, b in zip(lines, lines[1:]):
        run = run + 1 if a == b else 1
        mx = max(mx, run)
    if mx >= 6:
        return f"{mx} identical consecutive lines — repetition loop"
    return ""


def load_rows():
    if OUT.exists():
        return json.loads(OUT.read_text(encoding="utf-8-sig"))["rows"]
    return []


def current_lane():
    f = RESULTS / "current-lane.json"
    if not f.exists():
        sys.exit("no results/current-lane.json - serve a lane first "
                 "(qwen36.cmd nvfp4 27b  /  ./qwen36.sh nvfp4 27b)")
    lane = json.loads(f.read_text(encoding="utf-8-sig"))
    models = probe(lane["base_url"])
    if models is None:
        sys.exit(f"lane '{lane['lane']}' not answering on {lane['base_url']} — "
                 "still loading, or serve it first")
    return lane


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--presets", default="code,math,chat")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--temp", type=float, default=0.0,
                    help="sampling temperature (default 0 = greedy, reproducible + lets MTP "
                         "prove losslessness). Sweep the MTP lane at 0.6/1.0 to show the felt "
                         "speedup shrink at realistic sampling — MTP acceptance is temp-sensitive.")
    ap.add_argument("--vs", default=None, help="baseline lane key for the speedup column")
    ap.add_argument("--warmup", type=int, default=0,
                    help="discard this many generations per preset before timing (kills the "
                         "cold-cache first-run outlier). Hardened re-run: use 1.")
    ap.add_argument("--stat", choices=["best", "median", "mean"], default="best",
                    help="headline estimator across --repeats. best-of biases to the luckiest "
                         "boost-clock run; median is the honest choice for a robustness run.")
    ap.add_argument("--save-text", action="store_true",
                    help="write each generation to results\\outputs\\ for a coherence audit.")
    ap.add_argument("--out", default=None,
                    help="write rows to a different file under results\\ (keeps the main "
                         "bench-results.json pristine, e.g. --out confirm-results.json).")
    args = ap.parse_args()
    global OUT
    if args.out:
        OUT = RESULTS / args.out

    lane = current_lane()
    key = lane_key(lane)
    tlabel = "greedy (temp 0)" if args.temp == 0 else f"temp {args.temp}"
    print(f"lane: {key}  ({lane['engine']} | {lane['model']} | {lane['quant']})")
    print(f"method: {tlabel}, thinking off, {args.max_tokens} new tokens, "
          f"best-of-{args.repeats}, single stream, client-side clock\n")

    import statistics
    if args.warmup:
        print(f"(warmup: {args.warmup} discarded generation(s) per preset)")
    rows = load_rows()
    new_rows = []
    for name in [p.strip() for p in args.presets.split(",") if p.strip()]:
        prompt = PRESETS.get(name)
        if not prompt:
            print(f"  !! unknown preset '{name}' — skipped"); continue
        for w in range(args.warmup):
            print(f"  {name:<5} warmup {w+1}/{args.warmup} ... ", end="", flush=True)
            try:
                ws = stream_race(lane["base_url"], prompt, max_tokens=args.max_tokens,
                                 temperature=args.temp)
                print(f"{ws['decode_tps']:7.1f} tok/s (discarded)")
            except Exception as e:
                print(f"FAILED: {e}")
        got = []          # all timed run dicts for this preset
        for i in range(args.repeats):
            print(f"  {name:<5} run {i+1}/{args.repeats} ... ", end="", flush=True)
            try:
                s = stream_race(lane["base_url"], prompt, max_tokens=args.max_tokens,
                                temperature=args.temp)
            except Exception as e:
                print(f"FAILED: {e}")
                continue
            deg = degeneration(s["text"])
            print(f"{s['decode_tps']:7.1f} tok/s decode | {s['wall_tps']:6.1f} wall | "
                  f"ttft {s['ttft']:.2f}s | {s['tokens']} tok"
                  + ("  (count estimated)" if s["tokens_estimated"] else "")
                  + (f"  !! DEGENERATE: {deg}" if deg else ""))
            if args.save_text:
                od = RESULTS / "outputs"; od.mkdir(exist_ok=True)
                (od / f"{key}_{name}_t{args.temp}_run{i+1}.txt").write_text(
                    s["text"], encoding="utf-8")
            row = {"ts": time.strftime("%Y-%m-%d %H:%M:%S"), "key": key, "preset": name,
                   "run": i + 1, "max_tokens": args.max_tokens, "temp": args.temp,
                   "degenerate": deg or None, **lane,
                   **{k: v for k, v in s.items() if k != "text"}}
            new_rows.append(row)
            got.append(s)
        if got:
            dts = [g["decode_tps"] for g in got]
            pick = {"best": max, "median": statistics.median, "mean": statistics.mean}[args.stat]
            head = pick(dts)
            spread = f"[{min(dts):.1f}-{max(dts):.1f}]" if len(dts) > 1 else ""
            rng = (max(dts) - min(dts)) / statistics.median(dts) * 100 if len(dts) > 1 else 0
            print(f"  {name:<5} {args.stat}: {head:.1f} tok/s decode  spread {spread} "
                  f"(+/-{rng:.0f}% range)\n")

    rows.extend(new_rows)
    RESULTS.mkdir(exist_ok=True)
    OUT.write_text(json.dumps({"rows": rows}, indent=1))
    print(f"appended {len(new_rows)} rows -> {OUT}")

    # cross-lane table — aggregate ALL rows per (lane,preset) with the chosen estimator
    # (so a --stat median run prints a median table, not a lucky best-of), scoped to this
    # run's temp + token budget. Degenerate rows excluded from the aggregate.
    pick = {"best": max, "median": statistics.median, "mean": statistics.mean}[args.stat]
    groups = {}
    for r in rows:
        if r["max_tokens"] != args.max_tokens or r.get("temp", 0.0) != args.temp:
            continue
        if r.get("degenerate"):
            continue
        groups.setdefault((r["key"], r["preset"]), []).append(r["decode_tps"])
    agg = {k: pick(v) for k, v in groups.items()}
    lanes = sorted({k for k, _ in agg})
    if len(lanes) > 1:
        base = args.vs or ("w4a16-27b" if any(l.startswith("w4a16") for l in lanes)
                           else next((l for l in lanes if l.startswith("gguf") and "mtp" not in l), lanes[0]))
        print(f"\n== all lanes @ {args.max_tokens} tok ({args.stat} decode tok/s, x vs {base}) ==")
        presets = sorted({p for _, p in agg})
        print(f"  {'lane':<18}" + "".join(f"{p:>16}" for p in presets))
        for l in lanes:
            cells = []
            for p in presets:
                r, b = agg.get((l, p)), agg.get((base, p))
                if r is None:
                    cells.append(f"{'-':>16}")
                else:
                    x = f" ({r/b:.2f}x)" if b and l != base else ""
                    cells.append(f"{r:>8.1f}{x:<8}")
            print(f"  {l:<18}" + "".join(cells))


if __name__ == "__main__":
    main()

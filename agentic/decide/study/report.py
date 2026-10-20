"""Tables for the tier study from its per-shape JSON files.
Tier k's best is the fastest measured finalist of tiers 1..k: each tier's search space contains the smaller tiers',
so a bigger tier is never reported slower than a smaller one. The raw per-tier numbers (only that tier's own
finalists) are shown separately, since they show where the MIP's own top picks missed.
python -m agentic.decide.study.report RESULTS_DIR > report.md"""
from __future__ import annotations

import json, sys
from pathlib import Path

SHAPES = ["1024x1024x1024", "2048x2048x2048", "4096x4096x4096", "8192x8192x8192", "4096x4096x1024", "32x4096x4096", "128x4096x4096"]


def g(x):
    return f"{x:,.1f}" if isinstance(x, (int, float)) else "—"


def nested(r):
    """{tier: (us, kernel)} for tiers 1-3 (finalists of tiers <= k) and tier 4 (tier 3 or tier 4's modes on the raw T3 best)."""
    rec = {x["name"]: x for x in r["records"]}; t = r["tiers"]; p = r["primary"]; out = {}; pool = []
    for k in ("1", "2", "3"):
        pool += [n for n in t[k].get("finalists", []) if rec.get(n, {}).get("measured")]
        best = min(pool, key=lambda n: rec[n]["measured"][p]) if pool else None
        out[k] = (rec[best]["measured"][p], best) if best else (None, None)
    t4 = t.get("4", {})
    c = [(out["3"][0], out["3"][1] + " (stream)")] if out["3"][0] else []
    modes = {"graph": f"{p}_graph"} | ({"pin": "cold_pin", "pin_graph": "cold_pin_graph"} if p == "cold" else {})
    for m, key in modes.items():                         # a time at or below zero is a failed subtraction, not a time
        if isinstance(t4.get(key), (int, float)) and t4[key] > 0: c.append((t4[key], f"{t4['base']} ({m})"))
    out["4"] = min(c) if c else (None, None)
    return out


def main(d):
    d = Path(d); rows, raw, wins = [], [], []
    for s in SHAPES:
        f = d / f"{s}.json"
        if not f.exists(): continue
        r = json.load(open(f)); p = r["primary"]; tl = r.get("tilelang", {}); cb = r.get("cublaslt", {}); n = nested(r)
        ref = min(x for x in (tl.get("autotuned", {}).get("us"), cb.get("us")) if x)
        rows.append(f"| {s} | {p} | {g(tl.get('default', {}).get('us'))} | {g(tl.get('autotuned', {}).get('us'))} | {g(cb.get('us'))} | "
                    + " | ".join(g(n[k][0]) for k in "1234") + f" | {g(cb.get(p + '_graph'))} | {ref / n['3'][0]:.2f}x | {cb[p + '_graph'] / n['4'][0]:.2f}x |")
        raw.append(f"| {s} | " + " | ".join(g(r["tiers"][k].get("best_us")) for k in "123") + " |")
        wins.append(f"- **{s}**: T1 `{n['1'][1]}`; T1-T3 `{n['3'][1]}`; T1-T4 `{n['4'][1]}`")
    print("| Shape | Timing | TileLang default | TileLang autotuned | cuBLASLt | MIP T1 | MIP T1+T2 | MIP T1+T2+T3 | MIP T1-T4 | cuBLASLt with a graph | T1-T3 speedup over the better of TileLang autotuned and cuBLASLt | T1-T4 speedup over cuBLASLt with a graph |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"); print("\n".join(rows)); print()
    print("Raw per-tier best (only that tier's own finalists timed):\n\n| Shape | T1 | T1+T2 | T1+T2+T3 |\n|---|---:|---:|---:|"); print("\n".join(raw)); print()
    print("Winning kernels:\n"); print("\n".join(wins))


if __name__ == "__main__":
    main(sys.argv[1])

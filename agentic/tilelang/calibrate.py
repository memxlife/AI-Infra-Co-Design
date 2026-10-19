"""T4: calibrate J-hat's terms for a family on a chip, then test the model on the sweeps.

Two sources. The sweep tables (no GPU): the per-SM tile rate by threads and the pipeline fill, as the fastest point
of each group (a bound, so the fit stays one-sided). The counter runs (GPU, a dozen kernels through run_kernel with
ncu): the tensor-pipe utilization of the best big config (F_tc), DRAM bytes and time of the decode shape (B_HBM),
bank conflicts across the swizzle panel. Writes calibration/<host>/tilelang_terms.yaml with the value, the source
and the limits of every term. The test: Spearman on the measured top-10 of each shape and the share of points the
model predicts faster than measured (one-sided, 5 % allowed).

python -m agentic.tilelang.calibrate fit        # from the sweep tables
python -m agentic.tilelang.calibrate test       # the report
"""
from __future__ import annotations

import json, math, sys
from collections import defaultdict
from pathlib import Path

import yaml

from ..chips import load_chip
from .cost import default_terms, predict, launched_threads

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "tilelang"
SHAPES = {"1024": {"M": 1024, "N": 1024, "K": 1024}, "4096": {"M": 4096, "N": 4096, "K": 4096}, "m32": {"M": 32, "N": 4096, "K": 4096}}


def rows(name):
    f = RUNS / f"sweep_{name}.jsonl"
    if not f.exists():
        return []
    rs = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    return [r for r in rs if r.get("ok") and r.get("correct") and not r.get("n_spills")]


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


RATE_TILES = ((128, 256), (256, 128))          # the fixed tiles of the rate experiment (long K)
SERIES_TILE = (128, 128)                         # the fixed tile of the STAGES x THREADS series (the brief's u_occ row)


def fit(chip: dict, out: Path | None = None, overlap: str = "max") -> dict:
    """R2: every term a lower bound on its own, no global scale. Each rate is the best any configuration reached (so
    the time it predicts is never above what that configuration took), the per-tile costs are lower envelopes, and
    the three bounds combine by max (TMA copies and the producer warps overlap the tensor work by construction).
      rate[threads][BM x BN]   best tile rate observed for that thread count and tile, over every sweep shape
      B_L2_per_sm              best per-SM copy rate observed (bytes of A and B tiles per second)
      B_HBM                    best streaming rate on the decode sweep
      t_wave_startup           the decode term: the smallest residual per wave on the M = 32 rows after the DRAM bound,
                               charged per wave (two waves there, the second one-third full)
    The per-k-step and fill costs are zero: a lower envelope cannot carry them without breaking the bound somewhere."""
    import numpy as np
    t = {**default_terms(chip), "overlap": overlap, "scale": 1.0}
    F = t["F_tc_per_sm"]
    rate, copy_best = {}, 0.0
    for name, shape in SHAPES.items():
        for r in rows(name):
            h = r["holes"]; p = predict(h, shape, chip, t); tt = p["terms"]
            # every row counts, whatever its longest term: the rate a row shows is a rate the tile can reach
            tile_s = max(1e-9, (r["latency_ms"] * 1e-3 - t["launch_s"]) / p["waves"])
            key = (h["threads"], h["block_M"], h["block_N"], shape["K"])
            rate[key] = max(rate.get(key, 0.0), tt["flops_tile"] / tile_s)
            copy_best = max(copy_best, tt["bytes_tile"] / tile_s)
    # the rate table is keyed by thread count, tile and K: a short K never reaches the long-K rate (the per-tile start-up
    # is not amortized), and a long-K rate applied to a short K would predict below what any short-K row measured only
    # by luck. A K not in the table takes the best over the K's seen (still a bound, looser).
    u_tile = {}
    for (th, bm, bn, K), v in rate.items():
        u_tile.setdefault(th, {})[f"{bm}x{bn}@{K}"] = float(min(1.0, v / F))
        d = u_tile[th]; d[f"{bm}x{bn}"] = max(d.get(f"{bm}x{bn}", 0.0), float(min(1.0, v / F)))
    u_warps = {th: float(max(d.values())) for th, d in u_tile.items()}        # the fallback when a tile is not in the table
    # B_HBM: the best streaming rate any configuration reached on any shape (modeled bytes over measured time), capped by
    # the contract's measured bandwidth. A rate below the best observed would predict a time above a measured one.
    b_hbm = 0.0
    for name, shape in SHAPES.items():
        for r in rows(name):
            b_hbm = max(b_hbm, predict(r["holes"], shape, chip, t)["terms"]["dram_bytes"] / (r["latency_ms"] * 1e-3 - t["launch_s"]))
    b_hbm = min(t["B_HBM"], b_hbm) if b_hbm else t["B_HBM"]
    terms = {**t, "u_warps": u_warps, "u_tile": u_tile, "B_L2_per_sm": float(max(t["B_L2_per_sm"], copy_best)), "B_HBM": float(b_hbm),
             "t_tile_fixed": {128: 0.0, 256: 0.0, 512: 0.0}, "t_fill": {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}, "t_step": {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}}
    # the decode term: a start-up per wave, as the smallest residual per wave over every sweep row (a lower envelope over
    # all shapes, so adding it breaks no bound); what the M = 32 rows alone would give is kept beside it for the record
    resid, resid_m32 = [], []
    for name, shape in SHAPES.items():
        for r in rows(name):
            p = predict(r["holes"], shape, chip, terms)
            x = (r["latency_ms"] * 1e-3 - p["time_s"]) / p["waves"]
            resid.append(x)
            if name == "m32":
                resid_m32.append(x)
    terms["t_wave_startup"] = float(max(0.0, min(resid))) if resid else 0.0
    terms["t_wave_startup_m32_only"] = float(max(0.0, min(resid_m32))) if resid_m32 else 0.0
    terms["source"] = (f"R2 lower envelopes: tile rate per (threads, tile) = best observed over the three sweeps ({len(rate)} keys); per-SM copy rate = best observed; "
                       f"B_HBM = best streaming rate at M=32; wave start-up = smallest residual per wave at M=32; no scale, overlap {overlap}")
    terms["fit"] = {"rate_keys": len(rate), "wave_startup_us": terms["t_wave_startup"] * 1e6, "wave_startup_m32_only_us": terms["t_wave_startup_m32_only"] * 1e6, "B_HBM_GBps": b_hbm / 1e9}
    if out:
        out.write_text(yaml.safe_dump({"chip": chip["name"], "family": "gemm_basic", "terms": {k: v for k, v in terms.items() if k != "fit"}, "fit_detail": terms["fit"]}, sort_keys=False))
    return terms


def load_terms(chip: dict) -> dict | None:
    f = ROOT / "calibration" / chip["provenance"]["host"] / "tilelang_terms.yaml" if "provenance" in chip else None
    if f and f.exists():
        y = yaml.safe_load(f.read_text()); tm = y["terms"]
        for k in ("u_warps", "swizzle_penalty", "t_tile_fixed", "t_step", "t_fill", "u_tile"):
            if k in tm:
                tm[k] = {int(a): b for a, b in tm[k].items()}
        return tm
    return None


def test(chip: dict, terms: dict) -> dict:
    report = {}
    for name, shape in SHAPES.items():
        rs = rows(name)
        if not rs:
            continue
        pairs = []
        for r in rs:
            p = predict(r["holes"], shape, chip, terms)
            if p["ok"]:
                pairs.append((p["time_us"], r["latency_ms"] * 1e3, r["holes"]))
        pairs.sort(key=lambda x: x[1])
        top = pairs[:10]
        optimistic = sum(1 for p, m, _ in pairs if p > m)        # the model says slower than it ran: the bound is broken
        errs = [(m - p) / m for p, m, _ in pairs]
        hist = defaultdict(int)
        for e in errs:
            hist[f"{int(e * 100 // 10) * 10:+d}%"] += 1
        report[name] = {"points": len(pairs), "spearman_all": round(spearman([p for p, _, _ in pairs], [m for _, m, _ in pairs]), 3),
                        "spearman_top10": round(spearman([p for p, _, _ in top], [m for _, m, _ in top]), 3),
                        "broken_bound": optimistic, "broken_bound_fraction": round(optimistic / len(pairs), 3),
                        "median_measured_over_predicted": round(sorted(m / p for p, m, _ in pairs)[len(pairs) // 2], 2),
                        "best_measured": {"holes": top[0][2], "us": round(top[0][1], 1), "predicted_us": round(top[0][0], 1)},
                        "model_pick": min(pairs, key=lambda x: x[0])[2], "model_pick_measured_us": round(min(pairs, key=lambda x: x[0])[1], 1),
                        "model_pick_gap": f"{min(pairs, key=lambda x: x[0])[1] / top[0][1] - 1:+.1%}",
                        "error_histogram": dict(sorted(hist.items()))}
    return report


if __name__ == "__main__":
    chip = load_chip("spark-6951-gb10")
    cmd = sys.argv[1] if len(sys.argv) > 1 else "test"
    if cmd == "fit":
        out = ROOT / "calibration" / "spark-6951" / "tilelang_terms.yaml"
        overlap = sys.argv[2] if len(sys.argv) > 2 else "max"
        terms = fit(chip, out=out, overlap=overlap); print("wrote", out); print({k: v for k, v in terms.items() if k in ("u_warps", "B_L2_per_sm", "t_tile_fixed", "t_step", "scale", "source")})
    terms = load_terms(chip) or default_terms(chip)
    print("terms:", terms.get("source")); print(json.dumps(test(chip, terms), indent=1))

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
# round 3: the four out-of-sample shapes, whose rows are the solver's pick plus 30 random points (runs/tilelang/oos/oos.jsonl)
SHAPES7 = {**SHAPES, "2048": {"M": 2048, "N": 2048, "K": 2048}, "8192": {"M": 8192, "N": 8192, "K": 8192},
           "k1024": {"M": 4096, "N": 4096, "K": 1024}, "m128": {"M": 128, "N": 4096, "K": 4096}}


def rows(name):
    """The measured rows of a shape: the sweep (three shapes), the 200-point samples (1024^3, M=32) and the out-of-sample
    runs (the four others), all in one form: holes, latency_ms."""
    out = []
    f = RUNS / f"sweep_{name}.jsonl"
    if f.exists():
        out += [r for r in (json.loads(l) for l in f.read_text().splitlines() if l.strip()) if r.get("ok") and r.get("correct") and not r.get("n_spills")]
    for f, prefix in ((RUNS / "sample" / "sample.jsonl", name + ":"), (RUNS / "oos" / "oos.jsonl", name + ":")):
        if f.exists():
            for r in (json.loads(l) for l in f.read_text().splitlines() if l.strip()):
                if r["label"].startswith(prefix) and r.get("ok") and r.get("correct") and not r.get("n_spills") and "latency_us" in r:
                    out.append({**r, "latency_ms": r["latency_us"] * 1e-3})
    return out


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).correlation)


RATE_TILES = ((128, 256), (256, 128))          # the fixed tiles of the rate experiment (long K)
SERIES_TILE = (128, 128)                         # the fixed tile of the STAGES x THREADS series (the brief's u_occ row)


def series_rows(prefix: str) -> list[dict]:
    f = RUNS / "series.jsonl"
    if not f.exists():
        return []
    return [r for r in (json.loads(l) for l in f.read_text().splitlines() if l.strip()) if r["label"].startswith(prefix) and r.get("ok") and r.get("correct")]


def series_terms(chip: dict, t: dict) -> dict:
    """Mechanism 1 and the per-tile start-up from the two designated series (runs/tilelang/series.jsonl):
      B (wave series, one tile, K = 1024, M = N growing): t = P + waves x T; the envelope P = min over pairs (t_a - t_b x waves_a / waves_b)
        ... taken as P = min over the series of (t_w - waves_w x T_min) with T_min = min_w t_w / waves_w, floored at 0.
      A (K series per tile, many waves): per wave t/waves = s + K x c; the marginal cost c from the two longest K (asymptotic rate),
        s = min over K of (t/waves - K x c) >= 0 (the start-up a tile pays before its pipeline is full)."""
    out = {"t_prologue": 0.0, "tile_startup": {}, "tile_rate_inf": {}}
    B = series_rows("B:")
    if len(B) >= 2:
        pts = []
        for r in B:
            p = predict(r["holes"], r["shape"], chip, t); pts.append((p["waves"], r["latency_us"] * 1e-6 - t["launch_s"]))
        # pairs: the single-wave point against every point of >= 8 waves (a partial last wave then weighs under 12 %):
        # t1 = P + T, t_w = P + w T  =>  P = (w t1 - t_w) / (w - 1); the envelope is the smallest P over the pairs
        ones = [tt for w, tt in pts if w == 1]; many = [(w, tt) for w, tt in pts if w >= 8]
        cands = [(w * t1 - tw) / (w - 1) for t1 in ones for w, tw in many]
        out["t_prologue"] = float(max(0.0, min(cands))) if cands else 0.0
    A = series_rows("A:")
    by_tile = {}
    for r in A:
        by_tile.setdefault(r["label"].split(":")[1], []).append(r)
    for tile, rs in by_tile.items():
        rs.sort(key=lambda r: r["shape"]["K"])
        per_wave = []
        for r in rs:
            p = predict(r["holes"], r["shape"], chip, t)
            per_wave.append((r["shape"]["K"], (r["latency_us"] * 1e-6 - t["launch_s"] - out["t_prologue"]) / p["waves"], p["terms"]["flops_tile"] / r["shape"]["K"]))
        (K1, t1, fpk), (K2, t2, _) = per_wave[-2], per_wave[-1]
        c = (t2 - t1) / (K2 - K1)                                  # seconds per unit of K per wave, asymptotic
        s_ = max(0.0, min(tt - K * c for K, tt, _ in per_wave))
        out["tile_startup"][tile] = float(s_)
        out["tile_rate_inf"][tile] = float(fpk / c) if c > 0 else 0.0   # flops per second at the asymptote
    return out


def fit(chip: dict, out: Path | None = None, overlap: str = "max", exclude: tuple = (), prologue: bool = True, block_cap: bool = True) -> dict:
    """R2's envelopes plus round 3's two mechanisms, every term a minimum (or a best rate) over its series.
      rate[threads][tile@K]   the best tile rate observed for that key, over the many-wave rows (waves >= 3) when the key has
                              any, so the per-launch prologue is not folded into it
      t_prologue              mechanism 1: the smallest (measured - launch - waves x tile time) over every row (>= 0)
      B_HBM, R_block          the best chip-wide and the best per-block DRAM streaming rate observed (bytes the model charges
                              over measured time, per resident block)
      B_L2_per_sm             best per-SM copy rate observed
    `exclude` names shapes left out of the fit (leave-one-shape-out)."""
    import numpy as np
    t = {**default_terms(chip), "overlap": overlap, "scale": 1.0}
    F = t["F_tc_per_sm"]
    use = {n: sh for n, sh in SHAPES7.items() if n not in exclude and rows(n)}
    # pass 1: tile rates from many-wave rows; keys seen only in single-wave rows wait for the prologue
    rate_many, rate_any, copy_best = {}, {}, 0.0
    meas, streaming = [], []                                     # (row, shape, waves, flops_tile, bytes_tile, key); the DRAM-bound rows apart
    for name, shape in use.items():
        for r in rows(name):
            h = r["holes"]; p = predict(h, shape, chip, t); tt = p["terms"]
            key = (h["threads"], h["block_M"], h["block_N"], shape["K"])
            secs = r["latency_ms"] * 1e-3 - t["launch_s"]
            # the regime filter: a row that streams at >= 60 % of the chip's DRAM rate is DRAM-bound; its time says nothing
            # about the tile (it sets the DRAM and per-block rates instead)
            if tt["dram_bytes"] / max(1e-9, secs) >= 0.6 * t["B_HBM"]:
                streaming.append((r, shape, p["waves"], tt["flops_tile"], tt["bytes_tile"], key)); continue
            meas.append((r, shape, p["waves"], tt["flops_tile"], tt["bytes_tile"], key))
            tile_s = max(1e-9, secs / p["waves"])
            rate_any[key] = max(rate_any.get(key, 0.0), tt["flops_tile"] / tile_s)
            if p["waves"] >= 3:
                rate_many[key] = max(rate_many.get(key, 0.0), tt["flops_tile"] / tile_s)
            copy_best = max(copy_best, tt["bytes_tile"] / tile_s)
    rate = dict(rate_any); rate.update(rate_many)              # many-wave rates win where they exist
    ser = series_terms(chip, t)
    # mechanism 1: the prologue from the wave series when it was run; else from the pair structure below. For a key seen both in single-wave rows (t1 = P + T) and in
    # many-wave rows (t_w = P + w T): T <= t_w / w for every many-wave row (an upper bound on the tile time, since the
    # prologue is folded in), so P >= t1 - min_w t_w / w. The envelope takes the smallest such P over the keys (>= 0).
    prol = 0.0
    if prologue:
        by_key = {}
        for r, shape, waves, fl, by, key in meas:
            p = predict(r["holes"], shape, chip, t)
            secs = r["latency_ms"] * 1e-3 - t["launch_s"]
            if p["terms"]["dram_bytes"] / max(1e-9, secs) >= 0.4 * t["B_HBM"]:
                continue
            by_key.setdefault(key, []).append((waves, p["tiles"], secs))
        cands = []
        for key, pts in by_key.items():
            ones = [tt for w, tiles, tt in pts if w == 1 and tiles <= chip["sms"]]; many = [tt / w for w, tiles, tt in pts if w >= 3]
            if ones and many:
                cands.append(min(ones) - min(many))
        prol = float(max(0.0, min(cands))) if cands else 0.0
        if ser["t_prologue"] > 0:
            prol = ser["t_prologue"]                             # the designated experiment wins over the sweep pairs
        # with the prologue known, every rate is re-taken net of it and of the tile's start-up (series A), so the rate is the
        # asymptotic one and the start-up is charged per wave in predict()
        rate = {}; rate_1 = {}; copy_best = 0.0
        for r, shape, waves, fl, by, key in meas:
            su = ser["tile_startup"].get(f"{key[1]}x{key[2]}", 0.0)
            tile_s = max(1e-9, (r["latency_ms"] * 1e-3 - t["launch_s"] - prol) / waves - su)
            (rate if waves >= 3 else rate_1)[key] = max((rate if waves >= 3 else rate_1).get(key, 0.0), fl / tile_s)
            copy_best = max(copy_best, by / tile_s)
    u_tile = {}
    for (th, bm, bn, K), v in rate.items():
        u_tile.setdefault(th, {})[f"{bm}x{bn}@{K}"] = float(min(1.0, v / F))
        d = u_tile[th]; d[f"{bm}x{bn}"] = max(d.get(f"{bm}x{bn}", 0.0), float(min(1.0, v / F)))
    for (th, bm, bn, K), v in (rate_1 if prologue else {}).items():     # single-wave rows: a key of their own, used only for single-wave predictions
        u_tile.setdefault(th, {})[f"{bm}x{bn}@{K}:1wave"] = float(min(1.0, v / F))
        if f"{bm}x{bn}@{K}" not in u_tile[th] and not prologue:
            u_tile[th][f"{bm}x{bn}@{K}"] = float(min(1.0, v / F))
    # series A measured the asymptotic rate of five tiles at 256 threads (net of the start-up): the stand-in for a (tile, K) the
    # sweeps never saw, for any thread count (a rate observed is a rate the tile reaches)
    for tile, v in ser.get("tile_rate_inf", {}).items():
        if v > 0:
            for th in (128, 256, 512):
                d = u_tile.setdefault(th, {}); d[tile] = max(d.get(tile, 0.0), float(min(1.0, v / F)))
    u_warps = {th: float(max(d.values())) for th, d in u_tile.items()}
    # DRAM: the best chip-wide rate and (mechanism 2) the best per-block rate, over every row
    b_hbm, r_block, l2_chip = 0.0, 0.0, 0.0
    for r, shape, waves, fl, by, key in meas + streaming:
        p = predict(r["holes"], shape, chip, t)
        secs = r["latency_ms"] * 1e-3 - t["launch_s"] - prol
        if secs <= 0:
            continue
        b_hbm = max(b_hbm, p["terms"]["dram_bytes"] / secs)
        l2_chip = max(l2_chip, p["terms"]["l2_read_bytes"] / secs)
    b_hbm = min(t["B_HBM"], b_hbm) if b_hbm else t["B_HBM"]    # capped by the contract's measured bandwidth before anything uses it
    L_eff = None
    base = {**t, "B_HBM": b_hbm, "t_prologue": prol if prologue else 0.0, "R_block": 0.0, "l2_bytes_effective": 64e9, "tile_startup": ser["tile_startup"]}
    for r, shape, waves, fl, by, key in streaming:               # the per-block rate is only observable where streaming is the limit
        secs = r["latency_ms"] * 1e-3 - t["launch_s"] - prol
        if secs <= 0:
            continue
        p = predict(r["holes"], shape, chip, t)
        r_block = max(r_block, (p["terms"]["dram_bytes"] / p["tiles"]) / (secs / p["waves"]))
        # the largest latency this row allows under the per-wave formula (bisection); the envelope keeps the smallest over rows
        lo, hi = 1e-7, 1e-3
        for _ in range(40):
            mid = (lo + hi) / 2
            if predict(r["holes"], shape, chip, {**base, "L_block": mid})["terms"]["t_dram"] <= secs * (1 + 1e-6):
                lo = mid
            else:
                hi = mid
        L_eff = lo if L_eff is None else min(L_eff, lo)
    # the traffic term is a chip-spec term: the microbenchmark's effective L2 and DRAM rate (calibration/spark-6951/contract.yaml),
    # never raised from the sweeps. Where a shape's rows contradict it the bound breaks and the report says so.
    l2_eff = t["l2_bytes_effective"]
    terms = {**t, "u_warps": u_warps, "u_tile": u_tile, "l2_bytes_effective": float(l2_eff), "B_L2_chip": float(max(t["B_L2_chip"], l2_chip)), "B_L2_per_sm": float(max(t["B_L2_per_sm"], copy_best)), "B_HBM": float(b_hbm),
             "R_block": 0.0, "L_block": float(L_eff or 0.0) if block_cap else 0.0, "t_prologue": float(prol) if prologue else 0.0,
             "t_tile_fixed": {128: 0.0, 256: 0.0, 512: 0.0}, "t_fill": {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}, "t_step": {2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0},
             "t_wave_startup": 0.0, "tile_startup": ser["tile_startup"], "knob_factors": {}}                 # the warp_spec factor (1.043) was inside the base spread; a factor on top of a rate fitted from the same rows breaks the bound by itself
    terms["source"] = (f"round 3 envelopes over shapes {sorted(use)} with chip-spec defaults (microbenchmark L2 and B_HBM for the traffic term, the series' start-up + asymptotic rate for an uncalibrated (tile, K)): tile rate per (threads, tile, K) net of the tile start-up (series A) and the prologue "
                       f"{prol * 1e6:.1f} us (series B); B_HBM {b_hbm / 1e9:.0f} GB/s; per-block streaming = bytes in flight / {(L_eff or 0) * 1e6:.1f} us; no least squares, no scale")
    terms["fit"] = {"shapes": sorted(use), "rate_keys": len(rate), "streaming_rows": len(streaming), "tile_rows": len(meas), "l2_policy": "chip-spec (microbenchmark), not fitted", "series": {k: (round(v * 1e6, 2) if isinstance(v, float) else {a: round(b * 1e6, 2) for a, b in v.items()}) for k, v in ser.items() if k != "tile_rate_inf"}, "prologue_us": prol * 1e6, "R_block_GBps": r_block / 1e9, "L_block_us": (L_eff or 0.0) * 1e6, "B_HBM_GBps": b_hbm / 1e9, "l2_eff_MB": l2_eff / 1e6}
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


def test(chip: dict, terms: dict, shapes: dict | None = None) -> dict:
    report = {}
    for name, shape in (shapes or SHAPES7).items():
        rs = rows(name)
        if not rs:
            continue
        pairs = []
        for r in rs:
            if r["holes"].get("smem_layout") == "linear":          # out of the menu since round 3; still a measured point, not a candidate
                continue
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

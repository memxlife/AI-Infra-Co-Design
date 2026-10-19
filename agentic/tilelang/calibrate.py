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


def fit(chip: dict, out: Path | None = None, one_sided: float = 0.95, overlap: str = "sum") -> dict:
    """The brief's calibration, term by term, each from a small designated experiment; the sweeps as a whole are the test.
      F_tc * u_warps[threads]   the tensor rate one block reaches: the best tile rate over the 4096^3 rows at a fixed
                                big tile (128x256 or 256x128), per thread count. Long K, so the per-tile costs are amortized.
      t_tile_fixed[threads], t_step[stages], B_L2_per_sm
                                the 1024^3 rows at the fixed tile 128x128 across block_K x stages x threads (the series);
                                least squares on the time left after the rate term.
      B_HBM                     the best streaming rate over the decode sweep (M = 32).
      scale                     the per-SM time deflated so `one_sided` of the calibration points lie at or above the model.
    Per tile on one SM: t_tile = t_tile_fixed + flops / (F_tc u) + bytes / B_L2_per_sm + k_steps * t_step."""
    import numpy as np
    t = {**default_terms(chip), "overlap": overlap}
    ths = [128, 256, 512]; sts = [2, 3, 4, 5]
    F = t["F_tc_per_sm"]

    def tile_s(r, shape):
        p = predict(r["holes"], shape, chip, t)
        return max(1e-9, (r["latency_ms"] * 1e-3 - t["launch_s"]) / p["waves"]), p

    # 1. the rate, per thread count
    rate = {th: 0.0 for th in ths}; rate_pts = 0
    for r in rows("4096"):
        h = r["holes"]
        if (h["block_M"], h["block_N"]) not in RATE_TILES:
            continue
        s_, p = tile_s(r, SHAPES["4096"]); rate_pts += 1
        rate[h["threads"]] = max(rate[h["threads"]], p["terms"]["flops_tile"] / s_)
    u = {th: float(min(1.0, rate[th] / F)) if rate[th] else 1.0 for th in ths}
    # 2. the per-tile costs from the fixed-tile series at two K (1024^3 and 4096^3, 128x128): with the rate term taken out,
    #    what is left is a cost per tile by thread count, a fill cost per tile by pipeline depth, one cost per k-step, and
    #    the copy time (when the overlap is a sum). Two K values separate the per-tile from the per-k-step costs.
    X, y = [], []
    for name in ("1024", "4096"):
        for r in rows(name):
            h = r["holes"]
            if (h["block_M"], h["block_N"]) != SERIES_TILE:
                continue
            s_, p = tile_s(r, SHAPES[name]); tt = p["terms"]
            rest = s_ - tt["flops_tile"] / (F * u[h["threads"]])
            row = np.zeros(len(ths) + len(sts) + 2)
            row[ths.index(h["threads"])] = 1.0                       # per tile, by threads
            row[len(ths) + sts.index(h["num_stages"])] = 1.0         # fill per tile, by stages
            row[-2] = tt["k_steps"]                                  # per k-step
            row[-1] = tt["bytes_tile"] if overlap == "sum" else 0.0  # 1 / copy rate
            X.append(row); y.append(max(0.0, rest))
    X = np.array(X); y = np.array(y)
    from scipy.optimize import nnls
    coef, _ = nnls(X, y)
    fixed = {th: float(coef[i]) for i, th in enumerate(ths)}
    fill = {st: float(coef[len(ths) + i]) for i, st in enumerate(sts)}
    step = {st: float(coef[-2]) for st in sts}
    copy_rate = float(1.0 / coef[-1]) if coef[-1] > 0 else t["B_L2_per_sm"]
    # 3. B_HBM from the decode sweep
    b_hbm = t["B_HBM"]
    if rows("m32"):
        rates = [predict(r["holes"], SHAPES["m32"], chip, t)["terms"]["dram_bytes"] / (r["latency_ms"] * 1e-3 - t["launch_s"]) for r in rows("m32")]
        b_hbm = min(t["B_HBM"], max(rates))
    terms = {**t, "u_warps": u, "t_tile_fixed": fixed, "t_fill": fill, "t_step": step, "B_L2_per_sm": copy_rate if overlap == "sum" else t["B_L2_per_sm"],
             "B_HBM": float(b_hbm), "overlap": overlap, "scale": 1.0}
    # 4. one-sided scale over the calibration points
    cal = [(r, SHAPES["4096"]) for r in rows("4096") if (r["holes"]["block_M"], r["holes"]["block_N"]) in RATE_TILES]
    cal += [(r, SHAPES["1024"]) for r in rows("1024") if (r["holes"]["block_M"], r["holes"]["block_N"]) == SERIES_TILE]
    cal += [(r, SHAPES["4096"]) for r in rows("4096") if (r["holes"]["block_M"], r["holes"]["block_N"]) == SERIES_TILE]   # the swizzle series (traffic)
    # the one-sided margin is set on every sweep point of every shape (in-sample for the bound test; the ranking terms
    # above come from the designated experiments only). Out-of-sample checks are the loop's real runs on new shapes.
    allpts = [(r, SHAPES[n]) for n in SHAPES for r in rows(n)]
    ratios = [r["latency_ms"] * 1e3 / predict(r["holes"], sh, chip, terms)["time_us"] for r, sh in allpts]
    terms["scale"] = float(min(1.0, np.quantile(ratios, 1 - one_sided)))
    terms["source"] = (f"term by term: rate from {rate_pts} long-K rows at {RATE_TILES}; per-tile, fill and per-step costs from {len(y)} series rows at {SERIES_TILE} "
                       f"(1024^3 and 4096^3); B_HBM from the decode sweep; the time scaled by {terms['scale']:.3f} so {int(one_sided * 100)} % of all {len(allpts)} sweep points lie at or above the model (in-sample margin)")
    terms["fit"] = {"calibration_points": len(cal), "rate_points": rate_pts, "series_points": int(len(y)), "rate_flops_per_s_by_threads": rate}
    if out:
        out.write_text(yaml.safe_dump({"chip": chip["name"], "family": "gemm_basic", "terms": {k: v for k, v in terms.items() if k != "fit"}, "fit_detail": terms["fit"]}, sort_keys=False))
    return terms


def load_terms(chip: dict) -> dict | None:
    f = ROOT / "calibration" / chip["provenance"]["host"] / "tilelang_terms.yaml" if "provenance" in chip else None
    if f and f.exists():
        y = yaml.safe_load(f.read_text()); tm = y["terms"]
        for k in ("u_warps", "swizzle_penalty", "t_tile_fixed", "t_step", "t_fill"):
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
        overlap = sys.argv[2] if len(sys.argv) > 2 else "sum"
        terms = fit(chip, out=out, overlap=overlap); print("wrote", out); print({k: v for k, v in terms.items() if k in ("u_warps", "B_L2_per_sm", "t_tile_fixed", "t_step", "scale", "source")})
    terms = load_terms(chip) or default_terms(chip)
    print("terms:", terms.get("source")); print(json.dumps(test(chip, terms), indent=1))

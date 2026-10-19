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


def fit(chip: dict, fit_shapes=("1024", "m32"), out: Path | None = None, one_sided: float = 0.95, overlap: str = "max") -> dict:
    """The per-SM tile model from the sweeps, by least squares, then shifted so that `one_sided` of the fit points lie
    above the model (J-hat stays a bound). Per tile on one SM:
        t_tile = t_tile_fixed[threads] + flops_tile / (F_tc * u_warps[threads]) + k_steps * t_step[num_stages]
    with the measured time per tile = (latency - launch) / waves. The DRAM and L2 terms keep the contract's values:
    the fit shapes do not reach them (1024^3 is L2-resident; the decode shape is the DRAM check, held as a test)."""
    import numpy as np
    t = {**default_terms(chip), "overlap": overlap}
    ths = sorted({128, 256, 512}); sts = sorted({2, 3, 4, 5})
    X, y, keys = [], [], []
    for name in fit_shapes:
        shape = SHAPES[name]
        for r in rows(name):
            h = r["holes"]
            if h["threads"] not in ths or h["num_stages"] not in sts:
                continue
            p = predict(h, shape, chip, t)
            if not p["ok"]:
                continue
            tile_s = max(1e-9, (r["latency_ms"] * 1e-3 - t["launch_s"]) / p["waves"])
            if p["terms"]["t_dram"] > 0.5 * r["latency_ms"] * 1e-3:      # a DRAM-bound point says nothing about the tile
                continue
            row = np.zeros(len(ths) * 2 + len(sts) + 1)
            row[ths.index(h["threads"])] = 1.0                                   # fixed per tile
            row[len(ths) + ths.index(h["threads"])] = p["terms"]["flops_tile"]   # 1 / rate
            row[2 * len(ths) + sts.index(h["num_stages"])] = p["terms"]["k_steps"]
            row[-1] = p["terms"]["bytes_tile"] if overlap == "sum" else 0.0      # 1 / copy rate
            X.append(row); y.append(tile_s); keys.append((name, h))
    X = np.array(X); y = np.array(y)
    from scipy.optimize import nnls
    coef, _ = nnls(X, y)
    pred = X @ coef
    # one-sided: scale the whole per-SM time down until `one_sided` of the points are at or above the model
    scale = float(min(1.0, np.quantile(y / np.maximum(pred, 1e-12), 1 - one_sided)))
    pred = pred * scale
    F = t["F_tc_per_sm"]
    u = {th: (1.0 / (F * coef[len(ths) + i]) if coef[len(ths) + i] > 0 else 1.0) for i, th in enumerate(ths)}
    u = {th: float(min(1.0, v)) for th, v in u.items()}
    copy_rate = (1.0 / coef[-1]) if coef[-1] > 0 else t["B_L2_per_sm"]
    # B_HBM: the bandwidth-bound shape (M = 32 decode) sets what a streaming kernel of this family reaches: the best
    # DRAM rate any configuration achieved (model bytes over measured time), a bound like the other terms
    b_hbm = t["B_HBM"]
    if "m32" in fit_shapes and rows("m32"):
        shape = SHAPES["m32"]
        rates = [predict(r["holes"], shape, chip, t)["terms"]["dram_bytes"] / (r["latency_ms"] * 1e-3 - t["launch_s"]) for r in rows("m32")]
        b_hbm = min(t["B_HBM"], max(rates))
    terms = {**t, "u_warps": u, "t_tile_fixed": {th: float(coef[i]) for i, th in enumerate(ths)},
             "t_step": {st: float(coef[2 * len(ths) + i]) for i, st in enumerate(sts)}, "scale": scale,
             "B_L2_per_sm": float(copy_rate) if overlap == "sum" else t["B_L2_per_sm"], "overlap": overlap, "B_HBM": float(b_hbm),
             "source": f"least squares on sweeps {list(fit_shapes)} ({len(y)} points), per-SM time scaled by {scale:.3f} so {int(one_sided * 100)} % of them lie at or above the model",
             "fit": {"points": int(len(y)), "above_model_fraction": float((y >= pred - 1e-12).mean()),
                     "median_measured_over_model": float(np.median(y / np.maximum(pred, 1e-12)))}}
    if out:
        out.write_text(yaml.safe_dump({"chip": chip["name"], "family": "gemm_basic", "terms": {k: v for k, v in terms.items() if k != "fit"}, "fit_detail": terms["fit"]}, sort_keys=False))
    return terms


def load_terms(chip: dict) -> dict | None:
    f = ROOT / "calibration" / chip["provenance"]["host"] / "tilelang_terms.yaml" if "provenance" in chip else None
    if f and f.exists():
        y = yaml.safe_load(f.read_text()); tm = y["terms"]
        for k in ("u_warps", "swizzle_penalty", "t_tile_fixed", "t_step"):
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

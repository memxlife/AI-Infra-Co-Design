"""The TileLang evaluator backend (the brief, section 6): render the family at the solver's point, compile and profile it
on the GPU host through agentic.tilelang.remote, check against torch, and put the measured time beside J-hat's
prediction. Selected by evaluator_backend = "tilelang:<ssh-alias>"."""
from __future__ import annotations

import hashlib

from ...tilelang.cost import default_terms, predict
from ...tilelang.remote import run_kernel
from ...tilelang.tools import render


def _terms_of(T: dict) -> dict | None:
    from ..solver import _terms_of as f
    return f(T)


def kernel_artifact(T: dict, x: dict) -> dict:
    family = T["scope"]["structure"]["family"]
    src = render(family, x["values"])
    return {"contract": T["id"], "assignment": x["id"], "template": T["template"], "language": "tilelang",
            "arch": "sm_121a", "source": src, "sha256": hashlib.sha256(src.encode()).hexdigest(),
            "provenance": {"generator": "agentic.tilelang.tools.render", "chip": T["scope"]["chip"].get("name"), "family": family, "holes": x["values"]}}


def measure(T: dict, x: dict, host: str, kernel: dict, reps: int = 100, ncu: bool = True, terms: dict | None = None) -> dict:
    shape = {k: T["scope"]["workload"][k] for k in ("M", "N", "K")}
    chip = T["scope"]["chip"]
    terms = terms or _terms_of(T) or default_terms(chip)
    p = predict(x["values"], shape, chip, terms)
    r = run_kernel(host, kernel["source"], shape["M"], shape["N"], shape["K"], {}, func=T["scope"]["structure"].get("func", "matmul"), reps=reps, ncu=ncu,
                   inout_c=T["scope"]["structure"].get("func_kind") == "inout_c")
    if not r.get("ok"):
        raise RuntimeError(f"kernel did not run on {host}: {str(r.get('error', '?'))[:400]}")
    measured_us = r["latency_ms"] * 1e3
    t = measured_us * 1e-6
    flops = 2.0 * shape["M"] * shape["N"] * shape["K"]
    prof = r.get("ncu") or {}
    dram_bytes = prof.get("dram_bytes") if prof.get("ok") else None
    util = {"tensor": flops / t / chip["tensor_flops_per_s_chip"]["fp16"],
            "DRAM": (dram_bytes if dram_bytes is not None else p["terms"]["dram_bytes"]) / t / chip["dram_bytes_per_s"],
            "L2": p["terms"]["l2_read_bytes"] / t / chip["l2_bytes_per_s"]}
    if prof.get("ok"):
        m = prof.get("metrics", {})
        util["tensor_pipe_counter"] = m.get("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", 0) / 100.0
    dom = max(("tensor", "DRAM", "L2"), key=lambda k: util[k])
    predicted = x["objective"]
    report = {k: v for k, v in r.items() if k not in ("cuda_src", "sass", "kernel_source", "ptxas")}
    return {"contract": T["id"], "assignment": x["id"], "design": x["values"],
            "predicted_cost": predicted, "measured_cost": measured_us,
            "discrepancy": (measured_us - predicted) / predicted if predicted else 0.0,
            "utilizations": util, "dominant_resource": dom, "simulator": f"tilelang:{host}", "seed": 0, "kernel": kernel["id"],
            "reference_agrees": bool(r.get("correct")), "ci95": None, "host": r.get("host"),
            "regime_source": "measured" if dram_bytes is not None else "modeled",
            "detail": {"reps": reps, "n_regs": r.get("n_regs"), "n_spills": r.get("n_spills"), "smem_bytes": r.get("smem_bytes"),
                       "threads": r.get("threads"), "tflops": r.get("tflops"), "compile_s": r.get("compile_s"),
                       "kernel_report": report, "prediction": p}}


def probe(T: dict, terms: dict | None = None, sample: int = 64) -> dict:
    """Does the assumed regime hold as a bound over the legal menu? J-hat's longest term on a sample of legal points."""
    from ...tilelang.tools import sample_legal
    shape = {k: T["scope"]["workload"][k] for k in ("M", "N", "K")}
    chip = T["scope"]["chip"]; family = T["scope"]["structure"]["family"]
    terms = terms or _terms_of(T) or default_terms(chip)
    menus = {k: v["domain"] for k, v in T["decisions"].items()}
    pts = sample_legal(family, shape, n=2000, menus=menus, chip=chip)["points"]
    step = max(1, len(pts) // sample)
    counts = {}
    for h in pts[::step]:
        b = predict(h, shape, chip, terms)["bound"]
        counts[b] = counts.get(b, 0) + 1
    regime = max(counts, key=counts.get) if counts else "?"
    name = {"tensor": "compute-bound", "copy": "copy-bound", "dram": "memory-bound", "l2": "memory-bound"}.get(regime, regime)
    return {"regime": name, "regime_holds_as_bound": len(counts) == 1, "feasible_designs": len(pts), "dominant_counts": counts,
            "note": f"J-hat's longest term on {len(pts[::step])} of {len(pts)} legal points"}


def feasible_designs(T: dict, terms: dict | None = None) -> list[dict]:
    """Every legal menu point with J-hat's time and its longest term, named as a resource for regime_of."""
    from ...tilelang.tools import sample_legal
    shape = {k: T["scope"]["workload"][k] for k in ("M", "N", "K")}
    chip = T["scope"]["chip"]; family = T["scope"]["structure"]["family"]
    terms = terms or _terms_of(T) or default_terms(chip)
    res = {"tensor": "tensor", "copy": "L2", "dram": "DRAM", "l2": "L2"}
    menus = {k: v["domain"] for k, v in T["decisions"].items()}
    out = []
    for h in sample_legal(family, shape, n=2000, menus=menus, chip=chip)["points"]:
        p = predict(h, shape, chip, terms)
        if p["ok"]:
            out.append({"design": h, "time_us": p["time_us"], "dominant_resource": res.get(p["bound"], p["bound"])})
    return out

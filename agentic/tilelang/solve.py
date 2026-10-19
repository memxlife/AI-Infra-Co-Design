"""The MILP for a TileLang family (the brief, section 5): one-hot menus, product variables for the terms that need them,
J-hat's terms as tables over those products, HiGHS through SciPy. The tables come from cost.py, so the solver and the
evaluator's prediction are the same function; the solver only adds the linking constraints and the epigraph."""
from __future__ import annotations

import itertools, math
from time import perf_counter

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .cost import blocks_per_sm, default_terms, predict
from .tools import legalize


def solve_family(T: dict, terms: dict | None = None, time_limit: float = 30.0) -> dict:
    t0 = perf_counter()
    family = T["scope"]["structure"]["family"]
    shape = {k: T["scope"]["workload"][k] for k in ("M", "N", "K")}
    chip = T["scope"]["chip"]
    terms = terms or default_terms(chip)
    menus = {k: v["domain"] for k, v in T["decisions"].items()}
    names = list(menus)
    # variables: one-hot per knob; z over (block_M, block_N, block_K, threads) for the per-SM tile time; w over
    # (num_stages, block_K) for the pipeline fill; v over (block_M, block_N, swizzle) for the DRAM term; T_ep the epigraph
    idx, var_names = {}, []
    def add(name):
        idx[name] = len(var_names); var_names.append(name); return idx[name]
    x = {(k, val): add(f"{k}={val}") for k in names for val in menus[k]}
    tile_keys = list(itertools.product(menus["block_M"], menus["block_N"], menus["block_K"], menus["threads"]))
    z = {key: add(f"z{key}") for key in tile_keys}
    pipe_keys = list(itertools.product(menus["num_stages"], menus["block_K"]))
    w = {key: add(f"w{key}") for key in pipe_keys}
    dram_keys = list(itertools.product(menus["block_M"], menus["block_N"], menus.get("swizzle", [0])))
    v = {key: add(f"v{key}") for key in dram_keys}
    smem_keys = list(itertools.product(menus["num_stages"], menus["block_M"], menus["block_N"], menus["block_K"]))
    s = {key: add(f"s{key}") for key in smem_keys}
    T_ep = add("time_s")
    n = len(var_names)
    rows, lo, hi = [], [], []
    def row(coefs, l, h):
        r = np.zeros(n)
        for i, c in coefs: r[i] += c
        rows.append(r); lo.append(l); hi.append(h)
    for k in names:                                                   # one value per knob
        row([(x[k, val], 1) for val in menus[k]], 1, 1)
    def link(prod, parts):                                            # z = AND of its one-hots; exactly one z
        for key, zi in prod.items():
            for (k, val) in zip(parts, key):
                row([(zi, 1), (x[k, val], -1)], -np.inf, 0)
            row([(zi, 1)] + [(x[k, val], -1) for (k, val) in zip(parts, key)], -(len(parts) - 1), np.inf)
        row([(zi, 1) for zi in prod.values()], 1, 1)
    link(z, ("block_M", "block_N", "block_K", "threads")); link(w, ("num_stages", "block_K"))
    link(v, ("block_M", "block_N", "swizzle")); link(s, ("num_stages", "block_M", "block_N", "block_K"))
    # tables: J-hat's pieces on each product, exactly as predict() combines them (so the solver's optimum is predict's)
    #   per-SM: scale * waves(bm,bn) * [ tile(bm,bn,bk,th) + k_steps(bk) * t_step(st) ] + launch   (z table + s table)
    #   DRAM and L2: traffic(bm,bn,swz)                                                           (v table)
    ub = np.ones(n)
    scale = terms.get("scale", 1.0)
    tile_part, step_part, dram_time, l2_time = {}, {}, {}, {}
    waves_of, ksteps_of = {}, {}
    st0, swz0 = menus["num_stages"][0], (menus.get("swizzle") or [0])[0]
    for (bm, bn, bk, th), zi in z.items():
        h = {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": st0, "threads": th, "swizzle": 0}
        lg = legalize(family, h, shape, chip)
        p = predict(h, shape, chip, terms)
        if not p["ok"] or (not lg["ok"] and ("registers" in lg["reason"] or "tile too small" in lg["reason"])):
            ub[zi] = 0; continue
        tt = p["terms"]
        combined = (tt["t_compute_tile"] + tt["t_copy_tile"]) if terms.get("overlap", "max") == "sum" else max(tt["t_compute_tile"], tt["t_copy_tile"])
        tile_part[zi] = scale * p["waves"] * (combined + tt["t_pipe"])     # scale on every term, as predict() applies it
        waves_of[bm, bn] = p["waves"]; ksteps_of[bk] = tt["k_steps"]
    from .cost import _lookup
    for (st, bm, bn, bk), si in s.items():                           # the shared-memory rule and the per-step cost, one table
        if st * (bm * bk + bk * bn) * 2 + 1024 > chip["smem_bytes_per_block"]:
            ub[si] = 0; continue
        if (bm, bn) in waves_of and bk in ksteps_of:
            step_part[si] = scale * waves_of[bm, bn] * (ksteps_of[bk] * _lookup(terms.get("t_step", {}), st) + _lookup(terms.get("t_fill", {}), st))
    for (bm, bn, swz), vi in v.items():
        h = {"block_M": bm, "block_N": bn, "block_K": menus["block_K"][0], "num_stages": st0, "threads": menus["threads"][0], "swizzle": swz}
        p = predict(h, shape, chip, terms)
        dram_time[vi] = scale * p["terms"]["t_dram"]; l2_time[vi] = scale * p["terms"]["t_l2"]
    launch = terms.get("launch_s", 0.0)
    row([(T_ep, 1)] + [(zi, -t) for zi, t in tile_part.items()] + [(si, -t) for si, t in step_part.items()], launch, np.inf)
    row([(T_ep, 1)] + [(vi, -t) for vi, t in dram_time.items()], launch, np.inf)
    row([(T_ep, 1)] + [(vi, -t) for vi, t in l2_time.items()], launch, np.inf)
    c = np.zeros(n); c[T_ep] = 1.0
    integrality = np.ones(n); integrality[T_ep] = 0
    lb = np.zeros(n); ub[T_ep] = np.inf
    res = milp(c, constraints=LinearConstraint(np.array(rows), lo, hi), integrality=integrality, bounds=Bounds(lb, ub),
               options={"time_limit": time_limit})
    if res.x is None:
        return {"contract": T["id"], "values": None, "objective": None, "bound": None, "gap": None, "status": res.message,
                "solver_time": perf_counter() - t0, "diagnostics": {"variables": n, "constraints": len(rows)}, "provenance": {"service": "agentic.tilelang.solve"}}
    values = {k: next(val for val in menus[k] if res.x[x[k, val]] > 0.5) for k in names}
    p = predict(values, shape, chip, terms)
    return {"contract": T["id"], "values": values, "objective": float(p["time_us"]),
            "bound": float(getattr(res, "mip_dual_bound", res.x[T_ep]) * 1e6), "gap": float(getattr(res, "mip_gap", 0.0) or 0.0),
            "solver_time": perf_counter() - t0, "status": "optimal" if res.success else res.message,
            "diagnostics": {"variables": n, "binary_variables": n - 1, "constraints": len(rows), "backend": "scipy.optimize.milp (HiGHS)",
                            "removed_by_tables": int((ub == 0).sum()), "milp_objective_us": float(res.x[T_ep] * 1e6), "predicted": p},
            "provenance": {"service": "agentic.tilelang.solve", "template": T["template"], "terms": terms.get("source")}}

"""A generic MILP over kernel configurations and cross-kernel resources.

Each kernel k has a list of candidate columns c (legal assignments of its tied variables, priced by the model: time
t[k,c] in microseconds, and resource use r[k,c,res], e.g. bytes pinned in the persisting L2). Separable variables
(knobs whose effect adds to the time without changing the rest) are one-hot choices with additive costs. Cross-kernel
constraints bound the sum of resource use (sum_k sum_c r[k,c,res] y[k,c] <= cap[res]). The objective is the sum over
kernels of count[k] * time. top_k() re-solves with a no-good cut on each solution returned, so the second solution is
the best one different from the first, and so on. brute_force() enumerates the same problem for small instances:
the two must agree (tests/test_decide.py)."""
from __future__ import annotations

import itertools
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


def _build(kernels: list[dict], caps: dict, cuts: list[list[int]]):
    """kernels: [{"name", "count", "columns": [{"id", "t", "res": {name: amount}}], "separable": {var: {value: dt}}}]"""
    names, idx = [], {}
    def add(key):
        idx[key] = len(names); names.append(key); return idx[key]
    for k in kernels:
        for c in k["columns"]: add((k["name"], "col", c["id"]))
        for var, table in (k.get("separable") or {}).items():
            for val in table: add((k["name"], var, val))
    n = len(names); cost = np.zeros(n); rows, lo, hi = [], [], []
    def row(coefs, l, h):
        r = np.zeros(n)
        for i, v in coefs: r[i] += v
        rows.append(r); lo.append(l); hi.append(h)
    for k in kernels:
        cnt = k.get("count", 1)
        row([(idx[(k["name"], "col", c["id"])], 1) for c in k["columns"]], 1, 1)
        for c in k["columns"]: cost[idx[(k["name"], "col", c["id"])]] = cnt * c["t"]
        for var, table in (k.get("separable") or {}).items():
            row([(idx[(k["name"], var, val)], 1) for val in table], 1, 1)
            for val, dt in table.items(): cost[idx[(k["name"], var, val)]] = cnt * dt
    for res, cap in caps.items():
        coefs = [(idx[(k["name"], "col", c["id"])], c.get("res", {}).get(res, 0.0)) for k in kernels for c in k["columns"]]
        row(coefs, -np.inf, cap)
    for cut in cuts:                                               # exclude an earlier solution: not all of its ones again
        row([(i, 1) for i in cut], -np.inf, len(cut) - 1)
    return names, idx, cost, rows, lo, hi


def solve(kernels: list[dict], caps: dict | None = None, cuts: list[list[int]] | None = None, time_limit: float = 60.0) -> dict | None:
    caps = caps or {}; cuts = cuts or []
    names, idx, cost, rows, lo, hi = _build(kernels, caps, cuts)
    n = len(names)
    cons = LinearConstraint(np.array(rows), lo, hi) if rows else None
    res = milp(cost, constraints=cons, integrality=np.ones(n), bounds=Bounds(np.zeros(n), np.ones(n)),
               options={"time_limit": time_limit, "mip_rel_gap": 1e-9})
    if res.x is None: return None
    on = [i for i in range(n) if res.x[i] > 0.5]
    pick = {}
    for i in on:
        kname, kind, val = names[i]
        pick.setdefault(kname, {})["column" if kind == "col" else kind] = val
    return {"objective_us": float(cost @ res.x), "pick": pick, "ones": on, "status": res.message, "variables": n, "constraints": len(rows)}


def top_k(kernels: list[dict], k: int = 5, caps: dict | None = None) -> list[dict]:
    out, cuts = [], []
    for _ in range(k):
        s = solve(kernels, caps, cuts)
        if s is None: break
        out.append(s); cuts.append(s["ones"])
    return out


def brute_force(kernels: list[dict], caps: dict | None = None) -> list[tuple[float, dict]]:
    """Every combination of every kernel's column and separable values; feasible ones sorted by objective."""
    caps = caps or {}; choices = []
    for k in kernels:
        sep = list((k.get("separable") or {}).items())
        opts = []
        for c in k["columns"]:
            for vals in itertools.product(*[list(t.items()) for _, t in sep]) if sep else [()]:
                opts.append((c, {var: v for (var, _), (v, _) in zip(sep, vals)}, sum(dt for _, dt in vals)))
        choices.append([(k, o) for o in opts])
    out = []
    for combo in itertools.product(*choices):
        use = {r: 0.0 for r in caps}; obj = 0.0; pick = {}
        for k, (c, sv, dt) in combo:
            obj += k.get("count", 1) * (c["t"] + dt)
            for r in caps: use[r] += c.get("res", {}).get(r, 0.0)
            pick[k["name"]] = {"column": c["id"], **sv}
        if all(use[r] <= caps[r] + 1e-9 for r in caps): out.append((obj, pick))
    return sorted(out, key=lambda t: t[0])

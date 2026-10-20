"""The generic decision MILP agrees with brute-force enumeration: random small problems with columns, separable knobs
and a shared resource budget (the shape of 'which layers' weights go in the persisting L2')."""
import random
from agentic.decide.mip import solve, top_k, brute_force


def _problem(seed):
    rnd = random.Random(seed); ks = []
    for i in range(3):
        cols = [{"id": f"c{j}", "t": rnd.uniform(5, 50), "res": {"l2": 0 if j == 0 else rnd.choice([0, 4, 8, 12])}} for j in range(rnd.randint(2, 5))]   # not pinning is always possible
        sep = {"store": {"direct": 0.0, "staged": rnd.uniform(-3, 2)}} if rnd.random() < 0.7 else {}
        ks.append({"name": f"k{i}", "count": rnd.randint(1, 4), "columns": cols, "separable": sep})
    return ks, {"l2": rnd.choice([8, 12, 18])}


def test_optimum_matches_enumeration():
    for seed in range(25):
        ks, caps = _problem(seed)
        best = brute_force(ks, caps)[0][0]
        s = solve(ks, caps)
        assert abs(s["objective_us"] - best) < 1e-6, seed


def test_top_k_matches_enumeration():
    for seed in range(10):
        ks, caps = _problem(seed)
        bf = [o for o, _ in brute_force(ks, caps)][:4]
        tk = [s["objective_us"] for s in top_k(ks, 4, caps)]
        assert len(tk) == len(bf) and all(abs(a - b) < 1e-6 for a, b in zip(tk, bf)), seed


def test_unified_template_renders_every_cover_point():
    import ast
    from agentic.decide.cover import cover
    from agentic.decide.template import render
    for base in ({}, {"copy_path": "cpasync", "warp_spec": 0, "threads": 256, "block_M": 64}):
        for a in cover(base):
            src = render(a); ast.parse(src)
            if a["split_k"] > 1: assert "T.atomic_add" in src and "accum_dtype)" in src
            elif a["c_store"] == "staged": assert "C_shared" in src
            if a["copy_path"] == "cpasync": assert "disable_tma=True" in src

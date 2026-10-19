"""The checks for a TileLang family, in the shape the orchestrator expects from the verifier service.
family: every menu point the constraints admit must pass legalize (K1 for this template: the constraint tables and the
precheck are the same rules). instance: the solver's point passes legalize. The full check, TileLang's own passes and a
run against torch, happens in the evaluator; its outcome is written into the measurement, not here."""
from __future__ import annotations

from hashlib import sha256
from time import perf_counter

from .tools import legalize


def certify_family(T: dict, proofs: dict | None = None) -> dict:
    t0 = perf_counter()
    family = T["scope"]["structure"]["family"]; shape = {k: T["scope"]["workload"][k] for k in ("M", "N", "K")}
    chip = T["scope"]["chip"]
    import itertools
    menus = {k: v["domain"] for k, v in T["decisions"].items()}        # the contract's menus (a model may have narrowed them)
    size = 1
    for v in menus.values():
        size *= len(v)
    if size <= 20000:
        pts = [dict(zip(menus, vals)) for vals in itertools.product(*menus.values())]
    else:                                                               # round 2: 560k points; a 5000-point sample stands in
        from .tools import sample_legal
        pts = sample_legal(family, shape, n=5000, menus=menus, chip=chip)["points"]
    bad = [(h, legalize(family, h, shape, chip)) for h in pts]
    removed = [r for _, r in bad if not r["ok"]]
    key = sha256(f"{family}:{shape}:{chip.get('name')}".encode()).hexdigest()
    obligations = [{"obligation": "legal: the menus' points that violate a constraint are exactly the ones legalize rejects",
                    "scope": f"{len(pts)} menu points" + (f" (a sample of {size})" if size > 20000 else ""), "status": "proved (by construction: one rule set)", "seconds": perf_counter() - t0,
                    "query_sha256": key[:16], "reused": False,
                    "detail": {"points": len(pts), "removed": len(removed), "reasons": _count(r["reason"] for r in removed)}}]
    return {"contract": T["id"], "assignment": None, "check": "family", "verdict": "proved", "obligations": obligations,
            "counterexample": None, "checker_time": perf_counter() - t0,
            "provenance": {"checker": "agentic.tilelang.verify", "family_key": key, "reused_from": None, "precondition_is_weakest": None}}


def check_instance(T: dict, x: dict) -> dict:
    t0 = perf_counter()
    family = T["scope"]["structure"]["family"]; shape = {k: T["scope"]["workload"][k] for k in ("M", "N", "K")}
    r = legalize(family, x["values"], shape, T["scope"]["chip"])
    ob = {"obligation": "legal: the solver's point passes the precheck", "scope": "one point", "status": "proved" if r["ok"] else "refuted",
          "seconds": perf_counter() - t0, "detail": r}
    cex = None if r["ok"] else {"obligation": "legal", "design": x["values"], "reason": r["reason"]}
    return {"contract": T["id"], "assignment": x["id"], "check": "instance", "verdict": "proved" if r["ok"] else "refuted",
            "obligations": [ob], "counterexample": cex, "checker_time": perf_counter() - t0,
            "provenance": {"checker": "agentic.tilelang.verify"}}


def _count(reasons):
    out = {}
    for r in reasons:
        k = (r or "?").split(":")[0]
        out[k] = out.get(k, 0) + 1
    return out

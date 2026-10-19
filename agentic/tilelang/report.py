"""The round's report: per shape, the three numbers the brief asks for (fraction removed by the constraints, fraction that
spill, gap between the solver's pick and the best measured), the model test, and the baselines when they exist.
Writes runs/tilelang/report.md. python -m agentic.tilelang.report"""
from __future__ import annotations

import json
from pathlib import Path

from ..chips import load_chip, scope_chip
from .analyze import load_rows, numbers
from .calibrate import SHAPES, load_terms, test
from .contract import family_contract
from .cost import default_terms
from .solve import solve_family

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs" / "tilelang"


def solver_pick(name: str, terms: dict) -> dict:
    T = family_contract("gemm_basic", SHAPES[name]); T["id"] = f"report-{name}"
    r = solve_family(T, terms=terms)
    rows = [x for x in load_rows(RUNS / f"sweep_{name}.jsonl") if x.get("ok") and x.get("correct")]
    hit = next((x for x in rows if x["holes"] == r["values"]), None)
    best = min(rows, key=lambda x: x["latency_ms"])
    return {"pick": r["values"], "predicted_us": r["objective"], "measured_us": hit and hit["latency_ms"] * 1e3, "in_sweep": hit is not None,
            "best_us": best["latency_ms"] * 1e3, "best": best["holes"], "gap": (hit["latency_ms"] / best["latency_ms"] - 1) if hit else None,
            "solver_s": r["solver_time"], "variables": r["diagnostics"]["variables"], "constraints": r["diagnostics"]["constraints"]}


def main():
    chip = load_chip("spark-6951-gb10")
    terms = load_terms(chip) or default_terms(chip)
    tst = test(chip, terms)
    out = ["# TileLang round 1: the numbers", "", f"Chip spark-6951 (GB10). Family gemm_basic. Terms: {terms.get('source')}", ""]
    out += ["| shape | grid | removed by constraints | compiled, ran, correct | spill | best measured | solver's pick | pick measured | gap pick vs best | Spearman (all) | bound broken | model pick gap |", "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    picks = {}
    for name, shape in SHAPES.items():
        f = RUNS / f"sweep_{name}.jsonl"
        if not f.exists():
            continue
        n = numbers(load_rows(f), json.load(open(RUNS / f"legalize_{name}.json")))
        p = solver_pick(name, terms); picks[name] = p
        t = tst.get(name, {})
        out.append(f"| {shape['M']}x{shape['N']}x{shape['K']} | {n['grid']} | {n['removed_by_constraints']} ({n['removed_fraction']:.0%}) | "
                   f"{n['compiled_and_timed'] - n['compile_or_run_failed'] - n['wrong_answer']} of {n['compiled_and_timed']} | {n['spill']} ({n['spill_fraction']:.1%}) | "
                   f"{n['best']['latency_us']:.1f} µs ({n['best']['tflops']:.1f} TF/s) | {_h(p['pick'])} | {p['measured_us'] and f'{p[chr(109)+chr(101)+chr(97)+chr(115)+chr(117)+chr(114)+chr(101)+chr(100)+chr(95)+chr(117)+chr(115)]:.1f} µs'} | "
                   f"{p['gap'] is not None and f'{p[chr(103)+chr(97)+chr(112)]:+.1%}'} | {t.get('spearman_all')} | {t.get('broken_bound_fraction', 0):.1%} | {t.get('model_pick_gap')} |")
    out += ["", "Removed by the constraints: the shared-memory rule, the register rule (launch bounds), the tile-per-thread rule. Spill: ptxas spill bytes > 0 among the configurations that ran. The solver's pick: the MILP over J-hat's tables with the calibrated terms, found in the sweep table (one compile and one profile instead of the grid). Bound broken: the share of points where J-hat predicted a slower time than measured (the one-sided test; 5 % allowed).", ""]
    out += ["", "## Two claims about TileLang's autotuner, kept apart (R4)", "",
            "**The menu claim.** TileLang's autotuner list (`example_gemm_autotune.py`, 144 configurations) has no traversal-panel knob. "
            "At 4096^3 its best is 1941.5 µs after 132 compiles; the best of our menu, which has the panel, is 1377.4 µs (exhaustive, 504 "
            "legal points) and the loop's one compile gave 1434.5 µs. The difference is the panel, a knob the list does not contain; it says "
            "nothing about the solver.", "",
            "**The solver claim.** Over our own menu, exhaustive timing of the 504 legal points at 4096^3 gives 1377.4 µs; the solver's one "
            "compile gives 1379.4 µs (the round-1 terms) and 1414.1 µs (the round-2 envelope terms, a model tie broken the other way). "
            "At 1024^3: 63.4 µs exhaustive against 64.4 µs; at M=32: 200.4 against 202.7 µs. These gaps are the solver's; the noise band "
            "within one session is 1.4 % at 1024^3 and 0.3-0.5 % at 4096^3 (R3).", "",
            "## Baselines", ""]
    for name, shape in SHAPES.items():
        f = RUNS / f"baseline_{name}.json"
        if not f.exists():
            out.append(f"- {shape['M']}x{shape['N']}x{shape['K']}: not run yet"); continue
        b = json.load(open(f))["kinds"]; p = picks.get(name, {})
        fx = RUNS / f"baseline_fix_{name}.json"                      # the rerun of carver (fixed mapping) and cuBLAS (second opinion)
        if fx.exists():
            b = {**b, **{k: v for k, v in json.load(open(fx))["kinds"].items() if v.get("latency_ms")}}
        # R1: torch.matmul was a fallback kernel (no sm_121 kernels in torch's cuBLAS); the baseline is cuBLASLt from the
        # Spark's CUDA 13 toolkit (agentic/tilelang/baselines/cublaslt_gemm.cu), best of its heuristic's candidates
        lt = cublaslt_results().get((shape["M"], shape["N"], shape["K"]))
        b = {k: v for k, v in b.items() if k != "cublas"}
        if lt:
            b["cublasLt"] = {"latency_ms": lt["ms"], "config": f"cuBLASLt CUDA 13, algo {lt['algo']} of {lt['candidates']} candidates"}
        parts = []
        for k in ("cublasLt", "autotune", "carver"):
            v = b.get(k, {})
            if v.get("latency_ms"):
                us = v["latency_ms"] * 1e3
                gap = f" (our pick {p['measured_us'] / us - 1:+.1%})" if p.get("measured_us") else ""
                extra = f", {v['compiled']} of {v['tried']} ran" if v.get("tried") else ""
                parts.append(f"{k} {us:.1f} µs{gap}" + (f" [{_h(v['config'])}]" if isinstance(v.get("config"), dict) else "") + extra)
            else:
                parts.append(f"{k}: {str(v.get('error', 'no result'))[:80]}")
        out.append(f"- {shape['M']}x{shape['N']}x{shape['K']}: " + "; ".join(parts))
    out += ["", "## The loop, live on the Spark (Claude Code as the Formulator, the MIP, TileLang, one compile and one run per shape)", ""]
    out += live_rows()
    out += ["", "## Solver", ""]
    for name, p in picks.items():
        out.append(f"- {name}: {p['variables']} variables, {p['constraints']} rows, {p['solver_s'] * 1e3:.0f} ms (HiGHS); predicted {p['predicted_us']:.1f} µs")
    out += ["", "## Error histograms (measured minus predicted, over measured)", ""]
    for name, t in tst.items():
        out.append(f"- {name}: {t.get('error_histogram')}; median measured/predicted {t.get('median_measured_over_predicted')}")
    (RUNS / "report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def cublaslt_results() -> dict:
    """RESULT lines of runs/tilelang/cublaslt.log: {(M, N, K): {ms, tflops, algo, candidates}}."""
    import re
    f = RUNS / "cublaslt.log"
    out = {}
    if f.exists():
        for m in re.finditer(r"RESULT M=(\d+) N=(\d+) K=(\d+) cublaslt_best_ms=([\d.]+) tflops=([\d.]+) algo=(\d+) candidates=(\d+)", f.read_text()):
            out[(int(m[1]), int(m[2]), int(m[3]))] = {"ms": float(m[4]), "tflops": float(m[5]), "algo": int(m[6]), "candidates": int(m[7])}
    return out


def live_rows() -> list[str]:
    """The measured events of the live run in runs/gemm-tilelang.db, with the baselines of the same shape when they exist."""
    from ..ledger import Ledger
    db = RUNS.parent / "gemm-tilelang.db"
    if not db.exists():
        return ["- not run yet"]
    L = Ledger(str(db))
    rows = ["| shape | our pick | measured | predicted | TFLOP/s | regs / spills | autotuner (compiles) | cuBLASLt | our time vs autotuner |", "|---|---|---|---|---|---|---|---|---|"]
    names = {(1024, 1024, 1024): "1024", (4096, 4096, 4096): "4096", (32, 4096, 4096): "m32", (2048, 2048, 2048): "new_2048", (16, 4096, 4096): "new_m16",
             (128, 4096, 4096): "new_m128", (4096, 4096, 1024): "new_k1024", (8192, 8192, 8192): "new_8192"}
    import sqlite3
    runs = [r[0] for r in sqlite3.connect(str(db)).execute("select distinct run from events order by seq").fetchall()] if hasattr(L, "events") else []
    for run in runs[-1:]:                                   # the latest live run
        for e in L.events(run, kinds=["measured"]):
            m = L.get(e["artifact"]); d = m.get("detail", {})
            T = L.get(L.get(e["artifact"])["contract"]); wl = T["scope"]["workload"]
            key = (wl["M"], wl["N"], wl["K"]); name = names.get(key)
            auto = cub = None
            for f in (RUNS / f"baseline_{name}.json", RUNS / f"baseline_{name.replace('new_', 'new_') if name else ''}.json"):
                if name and f.exists():
                    k = json.load(open(f))["kinds"]; auto = auto or k.get("autotune"); cub = cub or k.get("cublas")
            lt = cublaslt_results().get(key)
            cub = {"latency_ms": lt["ms"]} if lt else None
            flops = 2.0 * wl["M"] * wl["N"] * wl["K"]
            a_us = auto and auto.get("latency_ms") and auto["latency_ms"] * 1e3
            rows.append(f"| {wl['M']}x{wl['N']}x{wl['K']} | {_h(m['design'])} | {m['measured_cost']:.1f} µs | {m['predicted_cost']:.1f} µs | {flops / (m['measured_cost'] * 1e-6) / 1e12:.1f} | {d.get('n_regs')} / {d.get('n_spills')} | "
                        f"{f'{a_us:.1f} µs ({auto.get(chr(99)+chr(111)+chr(109)+chr(112)+chr(105)+chr(108)+chr(101)+chr(100))} of {auto.get(chr(116)+chr(114)+chr(105)+chr(101)+chr(100))})' if a_us else '–'} | "
                        f"{f'{cub[chr(108)+chr(97)+chr(116)+chr(101)+chr(110)+chr(99)+chr(121)+chr(95)+chr(109)+chr(115)] * 1e3:.1f} µs' if cub and cub.get('latency_ms') else '–'} | "
                        f"{f'{m[chr(109)+chr(101)+chr(97)+chr(115)+chr(117)+chr(114)+chr(101)+chr(100)+chr(95)+chr(99)+chr(111)+chr(115)+chr(116)] / a_us - 1:+.1%}' if a_us else '–'} |")
    return rows


def _h(h):
    return "–" if not h else f"{h.get('block_M')}x{h.get('block_N')}x{h.get('block_K')}, {h.get('num_stages')} st, {h.get('threads')} th, swz {h.get('swizzle')}"


if __name__ == "__main__":
    main()

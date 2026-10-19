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
    out += ["## Baselines", ""]
    for name, shape in SHAPES.items():
        f = RUNS / f"baseline_{name}.json"
        if not f.exists():
            out.append(f"- {shape['M']}x{shape['N']}x{shape['K']}: not run yet"); continue
        b = json.load(open(f))["kinds"]; p = picks.get(name, {})
        fx = RUNS / f"baseline_fix_{name}.json"                      # the rerun of carver (fixed mapping) and cuBLAS (second opinion)
        if fx.exists():
            b = {**b, **{k: v for k, v in json.load(open(fx))["kinds"].items() if v.get("latency_ms")}}
        parts = []
        for k in ("cublas", "autotune", "carver"):
            v = b.get(k, {})
            if v.get("latency_ms"):
                us = v["latency_ms"] * 1e3
                gap = f" (our pick {p['measured_us'] / us - 1:+.1%})" if p.get("measured_us") else ""
                extra = f", {v['compiled']} of {v['tried']} ran" if v.get("tried") else ""
                parts.append(f"{k} {us:.1f} µs{gap}" + (f" [{_h(v['config'])}]" if isinstance(v.get("config"), dict) else "") + extra)
            else:
                parts.append(f"{k}: {str(v.get('error', 'no result'))[:80]}")
        out.append(f"- {shape['M']}x{shape['N']}x{shape['K']}: " + "; ".join(parts))
    out += ["", "## Solver", ""]
    for name, p in picks.items():
        out.append(f"- {name}: {p['variables']} variables, {p['constraints']} rows, {p['solver_s'] * 1e3:.0f} ms (HiGHS); predicted {p['predicted_us']:.1f} µs")
    out += ["", "## Error histograms (measured minus predicted, over measured)", ""]
    for name, t in tst.items():
        out.append(f"- {name}: {t.get('error_histogram')}; median measured/predicted {t.get('median_measured_over_predicted')}")
    (RUNS / "report.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


def _h(h):
    return "–" if not h else f"{h.get('block_M')}x{h.get('block_N')}x{h.get('block_K')}, {h.get('num_stages')} st, {h.get('threads')} th, swz {h.get('swizzle')}"


if __name__ == "__main__":
    main()

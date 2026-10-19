"""Round 2 step 3: solve over the full 12-knob space at 1024^3 and M=32, against the 200-point random sample and round 1's pick.
  python -m agentic.tilelang.step3 picks     -> renders the solver's picks (and round 1's picks) into runs/tilelang/sample/pick_*.py + plan
  python -m agentic.tilelang.step3 report    -> after the picks ran on the Spark (sample_picks.jsonl): the comparison"""
from __future__ import annotations

import json, sys
from pathlib import Path

from ..chips import load_chip
from .calibrate import load_terms
from .contract import family_contract
from .cost import predict
from .solve import solve_family
from .tools import render

ROOT = Path(__file__).resolve().parents[2]; RUNS = ROOT / "runs" / "tilelang"; SAMPLE = RUNS / "sample"
SHAPES = {"1024": {"M": 1024, "N": 1024, "K": 1024}, "m32": {"M": 32, "N": 4096, "K": 4096}}
ROUND1_PICK = {"1024": {"block_M": 256, "block_N": 128, "block_K": 32, "num_stages": 3, "threads": 512, "swizzle": 0},
               "m32": {"block_M": 16, "block_N": 256, "block_K": 64, "num_stages": 2, "threads": 128, "swizzle": 0}}
ROUND2_DEFAULT = {"prefetch": 0, "warp_policy": 0, "vec_width": 0, "tma": 1, "warp_spec": 1, "smem_layout": "inferred"}


def picks():
    chip = load_chip("spark-6951-gb10"); terms = load_terms(chip)
    plan, out = [], {}
    for name, shape in SHAPES.items():
        T = family_contract("gemm_basic", shape, terms=terms); T["id"] = f"step3-{name}"
        r = solve_family(T, terms=terms)
        out[name] = {"pick": r["values"], "predicted_us": r["objective"], "solver_ms": r["solver_time"] * 1e3, "variables": r["diagnostics"]["variables"],
                     "round1_pick": {**ROUND1_PICK[name], **ROUND2_DEFAULT}, "round1_predicted_us": predict({**ROUND1_PICK[name], **ROUND2_DEFAULT}, shape, chip, terms)["time_us"]}
        for tag, h in (("pick", r["values"]), ("round1", out[name]["round1_pick"])):
            fn = SAMPLE / f"pick_{name}_{tag}.py"; fn.write_text(render("gemm_basic", h))
            plan.append({"label": f"{name}:{tag}", "shape": shape, "holes": h, "kernel": "sample/" + fn.name})
        print(name, "solver pick", r["values"], "predicted", round(r["objective"], 1), "us; round 1 pick predicted", round(out[name]["round1_predicted_us"], 1), "us")
    json.dump(plan, open(SAMPLE / "plan_picks.json", "w"), indent=1); json.dump(out, open(SAMPLE / "picks.json", "w"), indent=1)


def report():
    chip = load_chip("spark-6951-gb10"); terms = load_terms(chip)
    sample = [json.loads(l) for l in open(RUNS / "sample" / "sample.jsonl") if l.strip()]
    pk = [json.loads(l) for l in open(RUNS / "sample" / "sample_picks.jsonl") if l.strip()] if (RUNS / "sample" / "sample_picks.jsonl").exists() else []
    info = json.load(open(SAMPLE / "picks.json"))
    lines = ["## Step 3: the full 12-knob space, solver pick against 200 random legal points", ""]
    for name, shape in SHAPES.items():
        rs = [r for r in sample if r["label"].startswith(name + ":") and r.get("ok")]
        good = [r for r in rs if r.get("correct")]
        bad = [r for r in rs if not r.get("correct")]; fail = [r for r in sample if r["label"].startswith(name + ":") and not r.get("ok")]
        spill = [r for r in good if r.get("n_spills")]
        best = min(good, key=lambda r: r["latency_us"])
        pred = [(predict(r["holes"], shape, chip, terms)["time_us"], r["latency_us"]) for r in good]
        broken = sum(1 for p, m in pred if p > m) / max(1, len(pred))
        from scipy.stats import spearmanr
        rho = float(spearmanr([p for p, _ in pred], [m for _, m in pred]).correlation)
        got = {r["label"].split(":")[1]: r for r in pk if r["label"].startswith(name + ":")}
        pick = got.get("pick"); r1 = got.get("round1")
        used = {k: v for k, v in info[name]["pick"].items() if k in ROUND2_DEFAULT and v != ROUND2_DEFAULT[k]}
        lines.append(f"### {shape['M']}x{shape['N']}x{shape['K']}")
        lines.append(f"- sample: {len(sample) and sum(1 for r in sample if r['label'].startswith(name + ':'))} drawn, {len(fail)} failed to build or run, {len(bad)} wrong, {len(spill)} spill ({len(spill) / max(1, len(good)):.1%}); best measured {best['latency_us']:.1f} µs at {best['holes']}")
        lines.append(f"- model on the sample: Spearman {rho:.2f}, bound broken on {broken:.1%}")
        if pick:
            lines.append(f"- solver's pick ({info[name]['variables']} binaries, {info[name]['solver_ms']:.0f} ms): {info[name]['pick']} -> measured {pick['latency_us']:.1f} µs (predicted {info[name]['predicted_us']:.1f}); gap to sample best {pick['latency_us'] / best['latency_us'] - 1:+.1%}; new knobs it used: {used or 'none (all at the round-1 defaults)'}")
        if r1:
            lines.append(f"- round 1's pick re-timed in this run: {r1['latency_us']:.1f} µs; round 2 pick vs round 1 pick: {pick['latency_us'] - r1['latency_us']:+.1f} µs ({pick['latency_us'] / r1['latency_us'] - 1:+.1%})" if pick else "")
        lines.append("")
    txt = "\n".join(lines); print(txt)
    (RUNS / "step3.md").write_text(txt + "\n")


def oos_report():
    """R5: the four out-of-sample shapes, solver pick against 30 random legal points each."""
    from scipy.stats import spearmanr
    chip = load_chip("spark-6951-gb10"); terms = load_terms(chip)
    rows = [json.loads(l) for l in open(RUNS / "oos" / "oos.jsonl") if l.strip()]
    info = json.load(open(RUNS / "oos" / "picks.json"))
    shapes = {"2048": {"M": 2048, "N": 2048, "K": 2048}, "8192": {"M": 8192, "N": 8192, "K": 8192}, "k1024": {"M": 4096, "N": 4096, "K": 1024}, "m128": {"M": 128, "N": 4096, "K": 4096}}
    lines = ["## R5: out-of-sample shapes (terms calibrated on 1024^3, 4096^3 and M=32 only)", "",
             "| shape | solver pick | measured | predicted | sample best (of 30) | gap pick vs sample best | bound broken on the sample | Spearman on the sample | failed / wrong / spill in the sample |", "|---|---|---|---|---|---|---|---|---|"]
    for name, shape in shapes.items():
        rs = [r for r in rows if r["label"].startswith(name + ":")]
        pick = next((r for r in rs if r["label"].endswith(":pick")), None)
        smp = [r for r in rs if "sample" in r["label"]]
        good = [r for r in smp if r.get("ok") and r.get("correct")]
        fail = [r for r in smp if not r.get("ok")]; wrong = [r for r in smp if r.get("ok") and not r.get("correct")]; spill = [r for r in good if r.get("n_spills")]
        if not good or not pick or not pick.get("ok"):
            lines.append(f"| {shape['M']}x{shape['N']}x{shape['K']} | {_h(info[name]['pick'])} | {'failed: ' + (pick or {}).get('error', '')[-80:] if pick else '–'} | | | | | | {len(fail)} / {len(wrong)} / {len(spill)} |"); continue
        best = min(good, key=lambda r: r["latency_us"])
        pred = [(predict(r["holes"], shape, chip, terms)["time_us"], r["latency_us"]) for r in good]
        broken = sum(1 for p, m in pred if p > m) / len(pred)
        rho = float(spearmanr([p for p, _ in pred], [m for _, m in pred]).correlation)
        lines.append(f"| {shape['M']}x{shape['N']}x{shape['K']} | {_h(info[name]['pick'])} | {pick['latency_us']:.1f} µs | {info[name]['predicted_us']:.1f} µs | {best['latency_us']:.1f} µs ({_h(best['holes'])}) | "
                     f"{pick['latency_us'] / best['latency_us'] - 1:+.1%} | {broken:.0%} | {rho:.2f} | {len(fail)} / {len(wrong)} / {len(spill)} |")
    txt = "\n".join(lines); print(txt); (RUNS / "oos.md").write_text(txt + "\n")


if __name__ == "__main__":
    {"picks": picks, "report": report, "oos": oos_report}[sys.argv[1]]()

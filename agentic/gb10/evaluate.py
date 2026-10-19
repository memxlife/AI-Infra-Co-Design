"""Test the GB10 model on every measured kernel set, without refitting: per shape, the rank correlation between
predicted and measured times, the regret of the model's pick (its measured time over the best measured), how many of
the measured top 5 the model's top 5 holds, and the bottleneck the model names for the best kernels.
python -m agentic.gb10.evaluate  ->  runs/gb10/evaluate.json and a table"""
from __future__ import annotations
import collections, json, math
from pathlib import Path
from scipy.stats import spearmanr
from .model import predict

ROOT = Path(__file__).resolve().parents[2]; R = ROOT / "runs" / "tilelang"


def kern(shape, holes, launched, regs, smem, timing, uses_tma=None, spills=0):
    consumer = holes.get("threads", launched)
    path = "tma_ws" if launched == consumer + 128 else "cpasync"
    return {"shape": shape, "holes": holes, "path": path, "consumer_threads": consumer, "launched_threads": launched, "regs": regs or 128, "smem_bytes": smem, "timing": timing, "spill_bytes": spills or 0}


def load_sets() -> dict:
    sets = collections.defaultdict(list)
    for f, tag in (("sweep_1024.jsonl", "r1"), ("sweep_4096.jsonl", "r1"), ("sweep_m32.jsonl", "r1")):
        for r in map(json.loads, open(R / f)):
            if not r.get("ok"): continue
            s = r["shape"]; smem = (r.get("dynamic_smem_bytes") or 0) + (r.get("static_smem_bytes") or 0) or r.get("smem_bytes", 0)
            sets[f"{tag} {s['M']}x{s['N']}x{s['K']} (TMA, do_bench)"].append((kern(s, r["holes"], r["threads"], r["n_regs"], smem, "do_bench", spills=r.get("n_spills")), r["latency_ms"] * 1e3))
    for f, tag in (("sample/sample.jsonl", "r2 sample"), ("oos/oos.jsonl", "r3 oos")):
        for r in map(json.loads, open(R / f)):
            if not (r.get("ok") and r.get("correct") and r.get("latency_us")): continue
            s = r["shape"]; k = kern(s, r["holes"], r["threads"], r.get("n_regs"), r.get("dynamic_smem_bytes") or 0, "do_bench", spills=r.get("n_spills"))
            sets[f"{tag} {s['M']}x{s['N']}x{s['K']} (mixed, do_bench)"].append((k, r["latency_us"]))
    ab = ROOT / "calibration" / "spark-6951" / "ablation_cpasync"
    regs = dict(l.split() for l in open(ab / "regs.txt") if len(l.split()) == 2)
    for name, m in json.load(open(ab / "mechanisms.json")).items():
        h = m["pick"]; smem = h["num_stages"] * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
        sets["ablation 1024x1024x1024 (cp.async, warm)"].append((kern(m["shape"], h, h["threads"], int(regs[name]), smem, "warm"), m["full_event_us"]))
    return sets


def score(rows) -> dict:
    preds = [predict(k) for k, _ in rows]; meas = [t for _, t in rows]
    rho = spearmanr([p["us"] for p in preds], meas)[0] if len(rows) > 2 else float("nan")
    i_pick = min(range(len(rows)), key=lambda i: preds[i]["us"]); best = min(meas)
    top_m = set(sorted(range(len(rows)), key=lambda i: meas[i])[:5]); top_p = set(sorted(range(len(rows)), key=lambda i: preds[i]["us"])[:5])
    i_best = min(range(len(rows)), key=lambda i: meas[i])
    bn_best = collections.Counter(preds[i]["bottleneck"] for i in sorted(range(len(rows)), key=lambda i: meas[i])[:10])
    ape = sorted(abs(p["us"] - t) / t for p, t in zip(preds, meas))
    return {"n": len(rows), "spearman": rho, "regret": meas[i_pick] / best - 1, "top5_overlap": len(top_m & top_p),
            "pick": rows[i_pick][0]["holes"], "pick_measured_us": meas[i_pick], "pick_predicted_us": preds[i_pick]["us"], "best_measured_us": best,
            "best_holes": rows[i_best][0]["holes"], "best_predicted_us": preds[i_best]["us"], "bottleneck_of_best10": dict(bn_best), "median_abs_err": ape[len(ape) // 2]}


if __name__ == "__main__":
    sets = load_sets(); out = {}
    print(f"{'set':52s} {'n':>4} {'rho':>5} {'regret':>7} {'top5':>4} {'medAPE':>6}  bottleneck (best 10 measured)")
    for name in sorted(sets):
        if len(sets[name]) < 3: continue
        s = score(sets[name]); out[name] = s
        print(f"{name:52s} {s['n']:4d} {s['spearman']:5.2f} {s['regret']*100:6.1f}% {s['top5_overlap']:4d} {s['median_abs_err']*100:5.0f}%  {s['bottleneck_of_best10']}")
    (ROOT / "runs" / "gb10").mkdir(parents=True, exist_ok=True); json.dump(out, open(ROOT / "runs" / "gb10" / "evaluate.json", "w"), indent=1, default=str)

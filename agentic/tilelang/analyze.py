"""T3 tables: one CSV per shape from the sweep's JSON lines, and the three numbers the brief asks for.
python -m agentic.tilelang.analyze runs/tilelang/sweep_1024.jsonl runs/tilelang/legalize_1024.json"""
from __future__ import annotations

import csv, json, sys
from collections import Counter

KNOBS = ("block_M", "block_N", "block_K", "num_stages", "threads", "swizzle")


def load_rows(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def table(rows, legal, out_csv):
    cols = list(KNOBS) + ["ok", "correct", "latency_us", "tflops", "n_regs", "n_spills", "smem_bytes", "launched_threads", "compile_s", "error"]
    by_key = {tuple(r["holes"][k] for k in KNOBS): r for r in rows}
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(cols + ["precheck", "precheck_reason"])
        for l in legal:
            key = tuple(l[k] for k in KNOBS); r = by_key.get(key)
            smem = (r or {}).get("smem_bytes")
            if r and r.get("dynamic_smem_bytes", 0) == 0:              # the first sweep ran before the runner read the attribute
                smem = (r.get("static_smem_bytes") or 0) + l["num_stages"] * (l["block_M"] * l["block_K"] + l["block_K"] * l["block_N"]) * 2
            w.writerow(list(key) + [r.get("ok") if r else "", r.get("correct") if r else "", round(r["latency_ms"] * 1e3, 2) if r and r.get("latency_ms") else "",
                                    round(r["tflops"], 2) if r and r.get("tflops") else "", (r or {}).get("n_regs", ""), (r or {}).get("n_spills", ""), smem or "",
                                    (r or {}).get("threads", ""), (r or {}).get("compile_s", ""), (r or {}).get("error", "")[:80] if r else "",
                                    l["legal"]["ok"], (l["legal"]["reason"] or "")[:80]])


def numbers(rows, legal):
    n_grid = len(legal); removed = [l for l in legal if not l["legal"]["ok"]]
    ok = [r for r in rows if r.get("ok") and r.get("correct")]
    failed = [r for r in rows if not r.get("ok")]; wrong = [r for r in rows if r.get("ok") and not r.get("correct")]
    spill = [r for r in ok if r.get("n_spills")]
    best = min(ok, key=lambda r: r["latency_ms"]) if ok else None
    why = Counter((l["legal"]["reason"] or "?").split(":")[0] for l in removed)
    return {"grid": n_grid, "removed_by_constraints": len(removed), "removed_fraction": round(len(removed) / n_grid, 3), "removed_why": dict(why),
            "compiled_and_timed": len(rows), "compile_or_run_failed": len(failed), "wrong_answer": len(wrong),
            "spill": len(spill), "spill_fraction": round(len(spill) / max(1, len(ok)), 3),
            "best": best and {"holes": best["holes"], "latency_us": round(best["latency_ms"] * 1e3, 2), "tflops": round(best["tflops"], 2), "n_regs": best["n_regs"]},
            "worst_us": ok and round(max(r["latency_ms"] for r in ok) * 1e3, 2), "median_us": ok and round(sorted(r["latency_ms"] for r in ok)[len(ok) // 2] * 1e3, 2)}


if __name__ == "__main__":
    rows = load_rows(sys.argv[1]); legal = json.load(open(sys.argv[2]))
    out = sys.argv[1].replace(".jsonl", ".csv"); table(rows, legal, out)
    print(json.dumps(numbers(rows, legal), indent=1)); print("csv:", out)

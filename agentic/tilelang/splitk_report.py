"""Step 5: the split-K probe at M=32 against the gemm_basic base, and the exact-once check (correctness against torch)."""
from __future__ import annotations
import json
from pathlib import Path
RUNS = Path(__file__).resolve().parents[2] / "runs" / "tilelang"


def main():
    rows = [json.loads(l) for l in open(RUNS / "splitk.jsonl") if l.strip()]
    base = next(r for r in rows if "gemm_basic" in r["label"])
    lines = ["## Step 5: the split-K family at M=32", "",
             f"gemm_basic base (16x256x32, 4 stages, 256 threads): {base['latency_us']:.1f} µs, correct {base.get('correct')}.", "",
             "| kernel | split_k | measured | vs base | correct (exact-once) | regs | blocks |", "|---|---|---|---|---|---|---|"]
    best = None
    for r in rows:
        if "splitk" not in r["label"]:
            continue
        if not r.get("ok"):
            lines.append(f"| {r['label']} | {r['holes']['split_k']} | failed: {(r.get('error') or '').splitlines()[-1][:80]} | | | | |"); continue
        h = r["holes"]; blocks = (-(-32 // h["block_M"])) * (4096 // h["block_N"]) * h["split_k"]
        lines.append(f"| {h['block_M']}x{h['block_N']} st{h['num_stages']} | {h['split_k']} | {r['latency_us']:.1f} µs | {r['latency_us'] / base['latency_us'] - 1:+.1%} | {r.get('correct')} (err {r.get('max_abs_err', 0):.3f}) | {r.get('n_regs')} | {blocks} |")
        if r.get("correct") and (best is None or r["latency_us"] < best["latency_us"]):
            best = r
    if best:
        lines += ["", f"Best correct split-K kernel: {best['latency_us']:.1f} µs ({best['latency_us'] / base['latency_us'] - 1:+.1%} against the gemm_basic base; the noise band at M=32 is 0.5 %); DRAM floor for B once: 142.6 µs."]
    txt = "\n".join(lines); print(txt); (RUNS / "splitk.md").write_text(txt + "\n")


if __name__ == "__main__":
    main()

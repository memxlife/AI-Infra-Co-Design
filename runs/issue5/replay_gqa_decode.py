"""Issue 5's gqa_decode through our SM model with the contributor's inputs: where does the time go?"""
import sys, time
from pathlib import Path
sys.path.insert(0, "/Users/lishang/work/research/AI-Infra-Co-Design")
from agentic.gb10.sim.sm import SM, kernel_sass
from agentic.gb10.sim.sass import main_loop
sass = Path(sys.argv[1]) / "gqa_decode/bound-library/bound-0.sass"
ins = kernel_sass(sass, kernel="flashattn_gqa_decode_no_split_kernel"); lp = main_loop(ins)
print("main loop", lp, "length", lp[1] - lp[0] + 1)
marks = [(i, x.base, x.op[:28]) for i, x in enumerate(ins[lp[0]:lp[1] + 1], lp[0]) if x.base in ("LDGSTS", "LDGDEPBAR", "DEPBAR", "BAR", "HMMA", "MUFU", "LDG", "SHFL")]
# print runs: collapse consecutive same-base marks
run = []
for i, b, op in marks:
    if run and run[-1][1] == b: run[-1][2] += 1; run[-1][3] = i
    else: run.append([i, b, 1, i, op])
for s, b, n, e, op in run: print(f"  {s:4d}-{e:4d} {b:10s} x{n:3d}  {op}")
dram_clk = 512.0 / (237e9 / 32) * 2.54e9
def width(op):
    if op.endswith(".256"): return 32, 8
    if ".128" in op: return 16, 4
    if ".64" in op: return 8, 2
    return 4, 1
ldgs = [x for x in ins if x.base == "LDG"]; print("LDG ops:", sorted({x.op for x in ldgs}))
for label, hit in (("their inputs (mask loads from DRAM, 128 B each)", False), ("mask loads hit in the cache", True)):
    ent = [(width(x.op)[1], width(x.op)[0] * 32, hit) for x in ldgs]
    m = SM(ins, 128, 1, 256, active_sms=32, miss_frac=1.0, dram_clk_per_512B=dram_clk, copy_lines=[4] * 32, resident=1, grid_blocks=32,
           grid_dim=(4, 8, 1), ldg_lines=ent)
    t = time.time(); r = m.run(limit=40_000_000); cyc = r["cycles"]
    print(f"{label}: {cyc / 2540:.1f} us; per trip {cyc / 256:.0f} cycles; wall {time.time() - t:.0f} s")
    print("   busy fraction:", {k: round(v / cyc, 2) for k, v in r["busy"].items() if v})
print(f"DRAM cycles needed per trip at the fixed share: {32 * 1024 / 512 * dram_clk:.0f}; measured trip 10.8k cycles (1089 us)")

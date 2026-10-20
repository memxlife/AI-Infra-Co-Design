"""Replay the conversion probe (cvt.cu) on the simulator: one SM, the probe's 32 warps, its own machine code.
Prints cycles per element per sub-partition against the measured values in calibration/spark-6951/cvt.txt.
python -m agentic.gb10.sim.cvttest [mode ...]"""
import sys, time

from .sass import parse
from .sm import SM

ITERS, E, WARPS_PER_SMSP = 4096, 8, 8
MEASURED = {0: ("scaffold PRMT + IADD3", 4.04), 1: ("I2F.F16 from s32", 8.06), 2: ("I2F.F16 from u32", 8.05),
            3: ("I2FP.F32.S32 (int to fp32)", 4.03), 4: ("F2FP (fp32 to fp16)", 8.04), 5: ("MUFU.EX2 ruler", 8.23),
            6: ("I2F.F16 + MUFU.EX2 alternating", 8.13)}

if __name__ == "__main__":
    txt = open("calibration/spark-6951/sass/cvt.sass").read()
    for mode in [int(a) for a in sys.argv[1:]] or sorted(MEASURED):
        ins = parse(txt, kernel=f"_Z5probeILi{mode}EEvPKjPj"); t = time.time()
        r = SM(ins, 1024, 1, 1).run(limit=50_000_000)
        cyc = r["cycles"] / (ITERS * E * WARPS_PER_SMSP)
        label, meas = MEASURED[mode]
        print(f"mode {mode} {label:32s} simulated {cyc:5.2f}  measured {meas:5.2f}  ({time.time() - t:.0f} s)", flush=True)

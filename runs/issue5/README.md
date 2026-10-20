# Issue #5: an outside contributor's GB10 kernels against the simulator (Oct 20 2026)

Source: the bundle attached to issue #5, `gb10-simulator-calibration-20261020-v2.zip`, SHA256
`01f77fbadaa9ad61921d469edb62f61ac0c263b452a1307e5ad3dfab8207ad65`, from the contributor's fork (commit cc17474). Their
SASS and kernel sources were read as data. Their tools and binaries were not run. The kernel source for item 3 is
TileLang's dequantize-GEMV example as packaged in the bundle (`cases/w4_gemv/candidate.py`); it is not copied here.

Their three worst cases, simulated against measured (their numbers): `gemm` +82.0 %, `w4_gemv` +71.9 %,
`gqa_decode` +54.9 %.

## 1. `gemm` (+82 %): the swizzle never reached the simulator

The kernel calls `T.use_swizzle(10)`, but its case file has no swizzle, and their adapter reads `p.get("swizzle", 0)`.
The kernel also puts M tiles on blockIdx.x (`T.Kernel(ceildiv(M, bm), ceildiv(N, bn))`). Replayed with their SASS
(`replay_tma_gemm.py`, after commit 6126651 added `kernel=` and `h["grid_x"]`):

| Input, 4096 x 14336 x 4096 | Simulated (µs) | vs measured 5,081.9 |
|---|---:|---:|
| swizzle 0, as passed | 9,251.4 | +82 % |
| swizzle 10, grid x walks N | 5,189.2 | +2.1 % |
| swizzle 10, grid x walks M (the real launch) | 4,893.2 | −3.7 % |
| control `gemm_basic`, swizzle 8 | 5,038.0 | +3.3 % (vs 4,878.5) |

## 2. `I2F.F16` is not the GEMV's problem; `I2FP` was mischarged

`w4_gemv`'s loop runs 32 `I2F.F16` per pass, charged at the SFU rate without a measurement. Probe `cvt.cu`
(`calibration/spark-6951/cvt.txt`, commit ee6534c): `I2F.F16` takes 8.06 cycles per warp instruction per sub-partition and
shares the SFU with `MUFU`, as the simulator assumed. `I2FP.F32.S32` runs on the ALU at 2 cycles; the simulator charged
8, now fixed. The simulator replays the probe's own SASS within 2 % on all 7 modes (`agentic/gb10/sim/cvttest.py`).

## 3. `w4_gemv` (+72 %): the measurement was warm

1 x 14336 x 4096 with int4 weights (29.4 MB). Timed on our GB10 with `time_w4.py` (relative error 2e-4 against an
fp32 reference):

| Timing | µs per call | Weights at |
|---|---:|---:|
| their protocol: a CUDA graph of 30 calls on the same weights | 97.0 | 303 GB/s |
| back to back on the same weights | 100.7 | 292 GB/s |
| rotating 9 weight copies (264 MB), back to back | 125.3 | 234 GB/s |
| rotating 9 weight copies, in a graph | 122.9 | 239 GB/s |
| DRAM floor at 237 GB/s | 123.9 | |

Same weights each call is partly warm: the L2 keeps much of a re-read 29 MB set (our `stream_b2b.cu` gives 303 GB/s at
28 MiB, matching). Timed the way decode runs, the kernel sits at the DRAM floor, and their simulator result (130.2 µs)
is +4 to +6 %. Their own measurement, 75.7 µs (388 GB/s), we could not reproduce: the same protocol gives 97.0 µs here.

## 4. `gqa_decode` (+55 %): a known gap, not yet addressed

268 MB of K and V in 1,089 µs is 246 GB/s, the DRAM ceiling. The simulator's DRAM floor (1,133 µs) is close; its 1-stage
cp.async loop pays DRAM latency and queueing every step, where the hardware keeps DRAM busy across steps. This is the
"pipeline depth at the DRAM ceiling" gap in `notes/gb10_simulator_design.md`.

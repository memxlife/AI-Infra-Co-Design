# TileLang round 1: the numbers

Chip spark-6951 (GB10). Family gemm_basic. Terms: R2 lower envelopes: tile rate per (threads, tile) = best observed over the three sweeps (51 keys); per-SM copy rate = best observed; B_HBM = best streaming rate at M=32; wave start-up = smallest residual per wave at M=32; no scale, overlap max; round-2 knob factors from the 1024^3 curves (runs/tilelang/probe) against the mean of 7 base timings, clamped to 1 inside the base spread; the linear-layout factor withdrawn: that kernel computed wrong answers (legalize forbids linear + TMA)

| shape | grid | removed by constraints | compiled, ran, correct | spill | best measured | solver's pick | pick measured | gap pick vs best | Spearman (all) | bound broken | model pick gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024x1024x1024 | 1296 | 792 (61%) | 512 of 512 | 8 (1.6%) | 63.4 µs (33.9 TF/s) | 256x128x32, 3 st, 256 th, swz 0 | None | False | 0.776 | 0.8% | +0.0% |
| 4096x4096x4096 | 1296 | 792 (61%) | 510 of 512 | 6 (1.2%) | 1377.4 µs (99.8 TF/s) | 128x256x32, 2 st, 256 th, swz 16 | None | False | 0.85 | 1.0% | +0.0% |
| 32x4096x4096 | 1296 | 828 (64%) | 468 of 468 | 0 (0.0%) | 200.4 µs (5.4 TF/s) | 64x256x64, 2 st, 256 th, swz 0 | None | False | 0.097 | 0.9% | +0.0% |

Removed by the constraints: the shared-memory rule, the register rule (launch bounds), the tile-per-thread rule. Spill: ptxas spill bytes > 0 among the configurations that ran. The solver's pick: the MILP over J-hat's tables with the calibrated terms, found in the sweep table (one compile and one profile instead of the grid). Bound broken: the share of points where J-hat predicted a slower time than measured (the one-sided test; 5 % allowed).


## Two claims about TileLang's autotuner, kept apart (R4)

**The menu claim.** TileLang's autotuner list (`example_gemm_autotune.py`, 144 configurations) has no traversal-panel knob. At 4096^3 its best is 1941.5 µs after 132 compiles; the best of our menu, which has the panel, is 1377.4 µs (exhaustive, 504 legal points) and the loop's one compile gave 1434.5 µs. The difference is the panel, a knob the list does not contain; it says nothing about the solver.

**The solver claim.** Over our own menu, exhaustive timing of the 504 legal points at 4096^3 gives 1377.4 µs; the solver's one compile gives 1379.4 µs (the round-1 terms) and 1414.1 µs (the round-2 envelope terms, a model tie broken the other way). At 1024^3: 63.4 µs exhaustive against 64.4 µs; at M=32: 200.4 against 202.7 µs. These gaps are the solver's; the noise band within one session is 1.4 % at 1024^3 and 0.3-0.5 % at 4096^3 (R3).

## Baselines

- 1024x1024x1024: cublasLt 34.2 µs; autotune 60.2 µs [256x128x64, 2 st, 256 th, swz 0], 132 of 144 ran; carver 63.5 µs [128x64x64, 0 st, 128 th, swz 0], 20 of 20 ran
- 4096x4096x4096: cublasLt 1456.3 µs; autotune 1941.5 µs [128x256x64, 2 st, 256 th, swz 0], 132 of 144 ran; carver 2267.1 µs [128x128x64, 0 st, 128 th, swz 0], 20 of 20 ran
- 32x4096x4096: cublasLt 148.9 µs; autotune 204.6 µs [64x128x64, 3 st, 128 th, swz 0], 132 of 144 ran; carver 210.1 µs [16x64x128, 0 st, 128 th, swz 0], 7 of 20 ran

## The loop, live on the Spark (Claude Code as the Formulator, the MIP, TileLang, one compile and one run per shape)

| shape | our pick | measured | predicted | TFLOP/s | regs / spills | autotuner (compiles) | cuBLASLt | our time vs autotuner |
|---|---|---|---|---|---|---|---|---|
| 32x4096x4096 | 16x256x32, 3 st, 256 th, swz 0 | 205.8 µs | 247.7 µs | 5.2 | 168 / 0 | 204.6 µs (132 of 144) | 148.9 µs | +0.6% |
| 32x4096x4096 | 16x256x32, 3 st, 256 th, swz 0 | 207.9 µs | 201.7 µs | 5.2 | 168 / 0 | 204.6 µs (132 of 144) | 148.9 µs | +1.6% |

## Step 3: the full 12-knob space, solver pick against 200 random legal points

### 1024x1024x1024
- sample: 200 drawn, 0 failed to build or run, 13 wrong, 0 spill (0.0%); best measured 61.5 µs at {'block_M': 64, 'block_N': 256, 'block_K': 32, 'num_stages': 4, 'threads': 256, 'swizzle': 8, 'prefetch': 1, 'warp_policy': 2, 'vec_width': 0, 'tma': 0, 'warp_spec': 0, 'smem_layout': 'inferred'}
- model on the sample: Spearman 0.67, bound broken on 30.5%
- solver's pick (275 binaries, 20 ms): {'block_M': 256, 'block_N': 128, 'block_K': 32, 'num_stages': 3, 'threads': 256, 'swizzle': 0, 'prefetch': 0, 'warp_policy': 0, 'vec_width': 0, 'tma': 1, 'warp_spec': 1, 'smem_layout': 'inferred'} -> measured 62.5 µs (predicted 63.4); gap to sample best +1.6%; new knobs it used: none (all at the round-1 defaults)
- round 1's pick re-timed in this run: 64.5 µs; round 2 pick vs round 1 pick: -2.0 µs (-3.2%)

### 32x4096x4096
- sample: 200 drawn, 0 failed to build or run, 18 wrong, 0 spill (0.0%); best measured 203.8 µs at {'block_M': 64, 'block_N': 128, 'block_K': 64, 'num_stages': 3, 'threads': 256, 'swizzle': 16, 'prefetch': 2, 'warp_policy': 2, 'vec_width': 4, 'tma': 0, 'warp_spec': 1, 'smem_layout': 'swizzled'}
- model on the sample: Spearman -0.11, bound broken on 27.5%
- solver's pick (275 binaries, 38 ms): {'block_M': 64, 'block_N': 256, 'block_K': 64, 'num_stages': 2, 'threads': 256, 'swizzle': 0, 'prefetch': 0, 'warp_policy': 0, 'vec_width': 0, 'tma': 1, 'warp_spec': 1, 'smem_layout': 'inferred'} -> measured 206.8 µs (predicted 200.4); gap to sample best +1.4%; new knobs it used: none (all at the round-1 defaults)
- round 1's pick re-timed in this run: 203.7 µs; round 2 pick vs round 1 pick: +3.0 µs (+1.5%)

## R5: out-of-sample shapes (terms calibrated on 1024^3, 4096^3 and M=32 only)

| shape | solver pick | measured | predicted | sample best (of 30) | gap pick vs sample best | bound broken on the sample | Spearman on the sample | failed / wrong / spill in the sample |
|---|---|---|---|---|---|---|---|---|
| 2048x2048x2048 | 128x256x32, 4 st, 256 th, swz 8 | 242.7 µs | 189.6 µs | 257.0 µs (64x128x32, 5 st, 128 th, swz 16) | -5.6% | 0% | 0.20 | 0 / 2 / 0 |
| 8192x8192x8192 | 128x256x32, 4 st, 256 th, swz 16 | 11539.2 µs | 10755.1 µs | 13231.1 µs (128x256x32, 3 st, 512 th, swz 16) | -12.8% | 50% | 0.66 | 0 / 2 / 0 |
| 4096x4096x1024 | 256x128x64, 2 st, 256 th, swz 16 | 461.8 µs | 676.8 µs | 471.0 µs (64x128x32, 5 st, 128 th, swz 16) | -2.0% | 71% | 0.15 | 0 / 2 / 0 |
| 128x4096x4096 | 256x64x32, 2 st, 128 th, swz 0 | 258.9 µs | 209.6 µs | 207.9 µs (64x64x64, 3 st, 256 th, swz 8) | +24.6% | 14% | 0.03 | 0 / 2 / 0 |

## Step 5: the split-K family at M=32

gemm_basic base (16x256x32, 4 stages, 256 threads): 203.7 µs, correct True.

| kernel | split_k | measured | vs base | correct (exact-once) | regs | blocks |
|---|---|---|---|---|---|---|
| 16x256 st2 | 2 | 253.9 µs | +24.6% | True (err 0.117) | 168 | 64 |
| 16x256 st4 | 2 | 249.9 µs | +22.7% | True (err 0.122) | 168 | 64 |
| 16x256 st2 | 4 | 216.1 µs | +6.1% | True (err 0.119) | 168 | 128 |
| 16x256 st4 | 4 | 219.0 µs | +7.5% | True (err 0.090) | 168 | 128 |
| 16x256 st2 | 8 | 210.9 µs | +3.5% | True (err 0.109) | 168 | 256 |
| 16x256 st4 | 8 | 208.8 µs | +2.5% | True (err 0.123) | 168 | 256 |
| 32x128 st2 | 2 | 216.0 µs | +6.0% | True (err 0.125) | 168 | 64 |
| 32x128 st4 | 2 | 208.8 µs | +2.5% | True (err 0.105) | 168 | 64 |
| 32x128 st2 | 4 | 210.0 µs | +3.1% | True (err 0.115) | 168 | 128 |
| 32x128 st4 | 4 | 208.9 µs | +2.5% | True (err 0.123) | 168 | 128 |
| 32x128 st2 | 8 | 211.0 µs | +3.6% | True (err 0.109) | 168 | 256 |
| 32x128 st4 | 8 | 210.9 µs | +3.6% | True (err 0.124) | 168 | 256 |

Best correct split-K kernel: 208.8 µs (+2.5% against the gemm_basic base; the noise band at M=32 is 0.5 %); DRAM floor for B once: 142.6 µs.

## Solver

- 1024: 275 variables, 1146 rows, 20 ms (HiGHS); predicted 63.4 µs
- 4096: 275 variables, 1146 rows, 21 ms (HiGHS); predicted 1377.4 µs
- m32: 275 variables, 1146 rows, 38 ms (HiGHS); predicted 200.4 µs

## Error histograms (measured minus predicted, over measured)

- 1024: {'+0%': 455, '+10%': 41, '+20%': 4, '-10%': 4}; median measured/predicted 1.04
- 4096: {'+0%': 203, '+10%': 128, '+20%': 134, '+30%': 17, '+40%': 5, '+50%': 10, '+60%': 2, '-10%': 5}; median measured/predicted 1.14
- m32: {'+0%': 455, '+10%': 9, '-10%': 4}; median measured/predicted 1.03

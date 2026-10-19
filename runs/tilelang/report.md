# TileLang round 1: the numbers

Chip spark-6951 (GB10). Family gemm_basic. Terms: round 3 envelopes over shapes ['1024', '2048', '4096', '8192', 'k1024', 'm128', 'm32'] with chip-spec defaults (microbenchmark L2 and B_HBM for the traffic term, the series' start-up + asymptotic rate for an uncalibrated (tile, K)): tile rate per (threads, tile, K) net of the tile start-up (series A) and the prologue 21.6 us (series B); B_HBM 235 GB/s; per-block streaming = bytes in flight / 1.8 us; no least squares, no scale

| shape | grid | removed by constraints | compiled, ran, correct | spill | best measured | solver's pick | pick measured | gap pick vs best | Spearman (all) | bound broken | model pick gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024x1024x1024 | 1296 | 792 (61%) | 512 of 512 | 8 (1.6%) | 63.4 µs (33.9 TF/s) | 64x128x128, 2 st, 128 th, swz 8 | None | False | 0.134 | 9.4% | +0.1% |
| 4096x4096x4096 | 1296 | 792 (61%) | 510 of 512 | 6 (1.2%) | 1377.4 µs (99.8 TF/s) | 128x256x32, 3 st, 512 th, swz 16 | None | False | 0.856 | 26.6% | +1.8% |
| 32x4096x4096 | 1296 | 828 (64%) | 468 of 468 | 0 (0.0%) | 200.4 µs (5.4 TF/s) | 32x128x128, 2 st, 128 th, swz 4 | None | False | 0.228 | 7.1% | +0.0% |

Removed by the constraints: the shared-memory rule, the register rule (launch bounds), the tile-per-thread rule. Spill: ptxas spill bytes > 0 among the configurations that ran. The solver's pick: the MILP over J-hat's tables with the calibrated terms, found in the sweep table (one compile and one profile instead of the grid). Bound broken: the share of points where J-hat predicted a slower time than measured (the one-sided test; 5 % allowed).


## cuBLASLt at every shape (S1)

| shape | cuBLASLt (CUDA 13, best heuristic candidate) | our solver's pick, measured | our time vs cuBLASLt | best kernel we measured in the family |
|---|---|---|---|---|
| 1024x1024x1024 | 34.2 µs (62.8 TF/s) | 62.5 µs | +82.6% | 63.4 µs |
| 2048x2048x2048 | 183.9 µs (93.4 TF/s) | 242.7 µs | +32.0% | 242.7 µs |
| 4096x4096x4096 | 1456.3 µs (94.4 TF/s) | 1434.5 µs | -1.5% | 1377.4 µs |
| 8192x8192x8192 | 11198.8 µs (98.2 TF/s) | 11539.2 µs | +3.0% | 11539.2 µs |
| 4096x4096x1024 | 400.8 µs (85.7 TF/s) | 461.8 µs | +15.2% | 461.8 µs |
| 32x4096x4096 | 148.9 µs (7.2 TF/s) | 206.8 µs | +38.9% | 200.4 µs |
| 128x4096x4096 | 155.4 µs (27.6 TF/s) | 258.9 µs | +66.6% | 207.9 µs |

Same timing on both sides: CUDA events, median over the measured window after warm-up; cuBLASLt's number is the fastest of its heuristic's candidates (`agentic/tilelang/baselines/cublaslt_gemm.cu`). The within-session noise band is 1.4 % at 1024^3 and 0.3-0.5 % at 4096^3.

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

## Step 2 (round 3): the model before and after the two mechanisms

Before = round 2's envelopes (commit a690de4); after = round 3's: the per-launch prologue (21.6 µs, from the wave series), the tile start-up per tile class (from the K series), the per-block streaming cap (bytes in flight over 1.82 µs), the effective L2 as an envelope (39 MB). In-sample on all seven shapes; the leave-one-shape-out numbers are in the findings note.

| shape | bound broken, before | after | model pick gap, before | after | error histogram before (measured−predicted over measured) | after |
|---|---|---|---|---|---|---|
| 1024 | 7% | 0% | +0.1% | +0.1% | {'+0%': 538, '+10%': 52, '+20%': 4, '+30%': 1, '+40%': 4, '-10%': 39, '-20%': 4, '-30%': 4} | {'+0%': 269, '+10%': 180, '+20%': 187, '+30%': 8, '+40%': 1, '-10%': 1} |
| 4096 | 1% | 2% | +0.0% | +0.0% | {'+0%': 203, '+10%': 128, '+20%': 134, '+30%': 17, '+40%': 5, '+50%': 10, '+60%': 2, '-10%': 5} | {'+0%': 170, '+10%': 111, '+20%': 53, '+30%': 36, '+40%': 16, '+50%': 45, '+60%': 21, '+70%': 44, '-10%': 8} |
| m32 | 5% | 0% | +0.0% | +0.0% | {'+0%': 556, '+10%': 10, '-10%': 28, '-20%': 1} | {'+0%': 6, '+10%': 403, '+20%': 176, '+30%': 9, '-10%': 1} |
| 2048 | 0% | 0% | +0.0% | +0.0% | {'+0%': 2, '+10%': 8, '+20%': 6, '+30%': 4} | {'+0%': 15, '+10%': 4, '+20%': 1} |
| 8192 | 55% | 20% | +0.0% | +0.0% | {'+0%': 4, '+10%': 2, '+20%': 1, '+30%': 1, '+50%': 1, '-10%': 4, '-20%': 3, '-30%': 2, '-40%': 1, '-70%': 1} | {'+0%': 5, '+10%': 3, '+20%': 4, '+30%': 3, '+50%': 1, '-10%': 4} |
| k1024 | 95% | 15% | +36.1% | +0.0% | {'+10%': 1, '-110%': 2, '-120%': 1, '-40%': 3, '-50%': 3, '-60%': 2, '-70%': 2, '-80%': 4, '-90%': 2} | {'+0%': 9, '+10%': 6, '+20%': 2, '-10%': 3} |
| m128 | 15% | 0% | +0.0% | +0.0% | {'+0%': 14, '+10%': 2, '+20%': 1, '-10%': 1, '-20%': 1, '-50%': 1} | {'+10%': 10, '+20%': 7, '+30%': 3} |

## Step 2 (addition): leave-one-shape-out with chip-spec defaults, the solver's pick measured

| shape | bound broken (LOSO, defaults) | LOSO pick | measured | predicted | best measured (sweep / sample) | pick gap |
|---|---|---|---|---|---|---|
| 1024x1024x1024 | 16% | 64x128x64, 4 st, 256 th, swz 8 | 67.6 µs | 55.2 µs | 61.5 µs | +10.0% |
| 4096x4096x4096 | 38% | 128x256x32, 3 st, 256 th, swz 8 | 1437.5 µs | 1359.7 µs | 1377.4 µs | +4.4% |
| 32x4096x4096 | 26% | 16x256x64, 2 st, 128 th, swz 0 | 203.7 µs | 168.4 µs | 200.4 µs | +1.7% |
| 2048x2048x2048 | 0% | 128x256x32, 4 st, 512 th, swz 0 | 258.0 µs | 219.2 µs | 242.7 µs | +6.3% |
| 8192x8192x8192 | 25% | 128x256x32, 3 st, 512 th, swz 16 | 11953.2 µs | 10087.2 µs | 11539.2 µs | +3.6% |
| 4096x4096x1024 | 60% | 128x256x32, 4 st, 512 th, swz 0 | 456.7 µs | 431.0 µs | 461.8 µs | -1.1% |
| 128x4096x4096 | 10% | 256x64x32, 2 st, 128 th, swz 0 | 257.0 µs | 175.1 µs | 207.9 µs | +23.6% |

## Solver

- 1024: 274 variables, 1145 rows, 31 ms (HiGHS); predicted 54.9 µs
- 4096: 274 variables, 1145 rows, 26 ms (HiGHS); predicted 1437.6 µs
- m32: 274 variables, 1145 rows, 31 ms (HiGHS); predicted 168.4 µs

## Error histograms (measured minus predicted, over measured)

- 1024: {'+0%': 204, '+10%': 128, '+20%': 244, '+30%': 8, '+40%': 1, '-10%': 19, '-20%': 18, '-30%': 1, '-40%': 1, '-50%': 9, '-60%': 13}; median measured/predicted 1.22
- 4096: {'+0%': 219, '+10%': 98, '+20%': 27, '+30%': 13, '+40%': 1, '+50%': 10, '+60%': 2, '-10%': 80, '-20%': 37, '-30%': 17}; median measured/predicted 1.04
- m32: {'+0%': 7, '+10%': 372, '+20%': 174, '-10%': 3, '-40%': 19, '-50%': 18, '-60%': 2}; median measured/predicted 1.23
- 2048: {'+0%': 15, '+10%': 4, '+20%': 1}; median measured/predicted 1.0
- 8192: {'+0%': 8, '+10%': 3, '+30%': 1, '+50%': 1, '-10%': 5, '-20%': 1, '-40%': 1}; median measured/predicted 1.0
- k1024: {'+0%': 12, '+10%': 4, '+20%': 1, '-10%': 3}; median measured/predicted 1.0
- m128: {'+0%': 1, '+10%': 9, '+20%': 7, '+30%': 1, '-10%': 1, '-60%': 1}; median measured/predicted 1.23

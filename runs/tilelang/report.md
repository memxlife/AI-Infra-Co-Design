# TileLang round 1: the numbers

Chip spark-6951 (GB10). Family gemm_basic. Terms: R2 lower envelopes: tile rate per (threads, tile) = best observed over the three sweeps (51 keys); per-SM copy rate = best observed; B_HBM = best streaming rate at M=32; wave start-up = smallest residual per wave at M=32; no scale, overlap max; round-2 knob factors from the 1024^3 curves (runs/tilelang/probe) against the mean of 7 base timings, clamped to 1 inside the base spread

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
| 4096x4096x4096 | 256x128x64, 2 st, 256 th, swz 4 | 1434.5 µs | 1038.0 µs | 95.8 | 168 / 0 | 1941.5 µs (132 of 144) | 1456.3 µs | -26.1% |

## Solver

- 1024: 275 variables, 1145 rows, 20 ms (HiGHS); predicted 63.4 µs
- 4096: 275 variables, 1145 rows, 21 ms (HiGHS); predicted 1377.4 µs
- m32: 275 variables, 1145 rows, 37 ms (HiGHS); predicted 200.4 µs

## Error histograms (measured minus predicted, over measured)

- 1024: {'+0%': 455, '+10%': 41, '+20%': 4, '-10%': 4}; median measured/predicted 1.04
- 4096: {'+0%': 203, '+10%': 128, '+20%': 134, '+30%': 17, '+40%': 5, '+50%': 10, '+60%': 2, '-10%': 5}; median measured/predicted 1.14
- m32: {'+0%': 455, '+10%': 9, '-10%': 4}; median measured/predicted 1.03

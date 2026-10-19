# TileLang round 1: the numbers

Chip spark-6951 (GB10). Family gemm_basic. Terms: R2 lower envelopes: tile rate per (threads, tile) = best observed over the three sweeps (51 keys); per-SM copy rate = best observed; B_HBM = best streaming rate at M=32; wave start-up = smallest residual per wave at M=32; no scale, overlap max

| shape | grid | removed by constraints | compiled, ran, correct | spill | best measured | solver's pick | pick measured | gap pick vs best | Spearman (all) | bound broken | model pick gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024x1024x1024 | 1296 | 792 (61%) | 512 of 512 | 8 (1.6%) | 63.4 µs (33.9 TF/s) | 256x128x32, 3 st, 512 th, swz 0 | 64.4 µs | +1.7% | 0.776 | 0.8% | +0.0% |
| 4096x4096x4096 | 1296 | 792 (61%) | 510 of 512 | 6 (1.2%) | 1377.4 µs (99.8 TF/s) | 128x256x32, 4 st, 256 th, swz 8 | 1414.1 µs | +2.7% | 0.85 | 1.0% | +0.0% |
| 32x4096x4096 | 1296 | 828 (64%) | 468 of 468 | 0 (0.0%) | 200.4 µs (5.4 TF/s) | 16x256x64, 2 st, 128 th, swz 0 | 205.7 µs | +2.7% | 0.097 | 0.9% | +0.0% |

Removed by the constraints: the shared-memory rule, the register rule (launch bounds), the tile-per-thread rule. Spill: ptxas spill bytes > 0 among the configurations that ran. The solver's pick: the MILP over J-hat's tables with the calibrated terms, found in the sweep table (one compile and one profile instead of the grid). Bound broken: the share of points where J-hat predicted a slower time than measured (the one-sided test; 5 % allowed).

## Baselines

- 1024x1024x1024: cublasLt 34.2 µs (our pick +88.4%); autotune 60.2 µs (our pick +7.0%) [256x128x64, 2 st, 256 th, swz 0], 132 of 144 ran; carver 63.5 µs (our pick +1.5%) [128x64x64, 0 st, 128 th, swz 0], 20 of 20 ran
- 4096x4096x4096: cublasLt 1456.3 µs (our pick -2.9%); autotune 1941.5 µs (our pick -27.2%) [128x256x64, 2 st, 256 th, swz 0], 132 of 144 ran; carver 2267.1 µs (our pick -37.6%) [128x128x64, 0 st, 128 th, swz 0], 20 of 20 ran
- 32x4096x4096: cublasLt 148.9 µs (our pick +38.2%); autotune 204.6 µs (our pick +0.5%) [64x128x64, 3 st, 128 th, swz 0], 132 of 144 ran; carver 210.1 µs (our pick -2.1%) [16x64x128, 0 st, 128 th, swz 0], 7 of 20 ran

## The loop, live on the Spark (Claude Code as the Formulator, the MIP, TileLang, one compile and one run per shape)

| shape | our pick | measured | predicted | TFLOP/s | regs / spills | autotuner (compiles) | cuBLASLt | our time vs autotuner |
|---|---|---|---|---|---|---|---|---|
| 4096x4096x4096 | 256x128x64, 2 st, 256 th, swz 4 | 1434.5 µs | 1038.0 µs | 95.8 | 168 / 0 | 1941.5 µs (132 of 144) | 1456.3 µs | -26.1% |

## Solver

- 1024: 258 variables, 1138 rows, 20 ms (HiGHS); predicted 63.5 µs
- 4096: 258 variables, 1138 rows, 21 ms (HiGHS); predicted 1377.4 µs
- m32: 258 variables, 1138 rows, 33 ms (HiGHS); predicted 200.7 µs

## Error histograms (measured minus predicted, over measured)

- 1024: {'+0%': 455, '+10%': 41, '+20%': 4, '-10%': 4}; median measured/predicted 1.04
- 4096: {'+0%': 203, '+10%': 128, '+20%': 134, '+30%': 17, '+40%': 5, '+50%': 10, '+60%': 2, '-10%': 5}; median measured/predicted 1.14
- m32: {'+0%': 455, '+10%': 9, '-10%': 4}; median measured/predicted 1.03

# TileLang round 1: the numbers

Chip spark-6951 (GB10). Family gemm_basic. Terms: term by term: rate from 64 long-K rows at ((128, 256), (256, 128)); per-tile, fill and per-step costs from 144 series rows at (128, 128) (1024^3 and 4096^3); B_HBM from the decode sweep; the time scaled by 0.588 so 95 % of all 1476 sweep points lie at or above the model (in-sample margin)

| shape | grid | removed by constraints | compiled, ran, correct | spill | best measured | solver's pick | pick measured | gap pick vs best | Spearman (all) | bound broken | model pick gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024x1024x1024 | 1296 | 792 (61%) | 512 of 512 | 8 (1.6%) | 63.4 µs (33.9 TF/s) | 128x256x32, 4 st, 256 th, swz 4 | 64.5 µs | +1.8% | 0.489 | 9.5% | +1.7% |
| 4096x4096x4096 | 1296 | 792 (61%) | 510 of 512 | 6 (1.2%) | 1377.4 µs (99.8 TF/s) | 256x128x64, 2 st, 256 th, swz 4 | 1379.4 µs | +0.1% | 0.918 | 5.8% | +0.0% |
| 32x4096x4096 | 1296 | 828 (64%) | 468 of 468 | 0 (0.0%) | 200.4 µs (5.4 TF/s) | 32x64x128, 4 st, 128 th, swz 0 | 211.9 µs | +5.7% | 0.238 | 0.0% | +0.0% |

Removed by the constraints: the shared-memory rule, the register rule (launch bounds), the tile-per-thread rule. Spill: ptxas spill bytes > 0 among the configurations that ran. The solver's pick: the MILP over J-hat's tables with the calibrated terms, found in the sweep table (one compile and one profile instead of the grid). Bound broken: the share of points where J-hat predicted a slower time than measured (the one-sided test; 5 % allowed).

## Baselines

- 1024x1024x1024: cublas 201.5 µs (our pick -68.0%); autotune 60.2 µs (our pick +7.1%) [256x128x64, 2 st, 256 th, swz 0], 132 of 144 ran; carver 63.5 µs (our pick +1.6%) [128x64x64, 0 st, 128 th, swz 0], 20 of 20 ran
- 4096x4096x4096: cublas 10807.4 µs (our pick -87.2%); autotune 1941.5 µs (our pick -29.0%) [128x256x64, 2 st, 256 th, swz 0], 132 of 144 ran; carver 2267.1 µs (our pick -39.2%) [128x128x64, 0 st, 128 th, swz 0], 20 of 20 ran
- 32x4096x4096: cublas 491.2 µs (our pick -56.9%); autotune 204.6 µs (our pick +3.6%) [64x128x64, 3 st, 128 th, swz 0], 132 of 144 ran; carver 210.1 µs (our pick +0.9%) [16x64x128, 0 st, 128 th, swz 0], 7 of 20 ran

## The loop, live on the Spark (Claude Code as the Formulator, the MIP, TileLang, one compile and one run per shape)

| shape | our pick | measured | predicted | TFLOP/s | regs / spills | autotuner (compiles) | cuBLAS | our time vs autotuner |
|---|---|---|---|---|---|---|---|---|
| 4096x4096x4096 | 256x128x64, 2 st, 256 th, swz 4 | 1434.5 µs | 1038.0 µs | 95.8 | 168 / 0 | 1941.5 µs (132 of 144) | 10807.4 µs | -26.1% |

## Solver

- 1024: 258 variables, 1138 rows, 18 ms (HiGHS); predicted 31.5 µs
- 4096: 258 variables, 1138 rows, 25 ms (HiGHS); predicted 1038.0 µs
- m32: 258 variables, 1138 rows, 17 ms (HiGHS); predicted 110.3 µs

## Error histograms (measured minus predicted, over measured)

- 1024: {'+0%': 25, '+10%': 61, '+20%': 17, '+30%': 193, '+40%': 131, '+50%': 29, '-10%': 9, '-20%': 1, '-30%': 19, '-40%': 16, '-50%': 3}; median measured/predicted 1.52
- 4096: {'+0%': 42, '+10%': 156, '+20%': 201, '+30%': 71, '+40%': 5, '-10%': 11, '-20%': 10, '-30%': 5, '-40%': 3}; median measured/predicted 1.27
- m32: {'+40%': 453, '+50%': 15}; median measured/predicted 1.88

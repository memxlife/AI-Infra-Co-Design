# TileLang round 1: the numbers

Chip spark-6951 (GB10). Family gemm_basic. Terms: term by term: rate from 64 long-K rows at ((128, 256), (256, 128)); per-tile, fill and per-step costs from 144 series rows at (128, 128) (1024^3 and 4096^3); B_HBM from the decode sweep; the time scaled by 0.547 so 95 % of all 1476 sweep points lie at or above the model (in-sample margin)

| shape | grid | removed by constraints | compiled, ran, correct | spill | best measured | solver's pick | pick measured | gap pick vs best | Spearman (all) | bound broken | model pick gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024x1024x1024 | 1296 | 792 (61%) | 512 of 512 | 8 (1.6%) | 63.4 µs (33.9 TF/s) | 128x256x32, 4 st, 256 th, swz 4 | 64.5 µs | +1.8% | 0.489 | 7.9% | +1.7% |
| 4096x4096x4096 | 1296 | 792 (61%) | 510 of 512 | 6 (1.2%) | 1377.4 µs (99.8 TF/s) | 256x128x64, 2 st, 256 th, swz 8 | 1454.0 µs | +5.6% | 0.909 | 6.7% | +0.0% |
| 32x4096x4096 | 1296 | 828 (64%) | 468 of 468 | 0 (0.0%) | 200.4 µs (5.4 TF/s) | 64x256x64, 2 st, 512 th, swz 8 | 207.8 µs | +3.7% | 0.792 | 0.0% | +0.0% |

Removed by the constraints: the shared-memory rule, the register rule (launch bounds), the tile-per-thread rule. Spill: ptxas spill bytes > 0 among the configurations that ran. The solver's pick: the MILP over J-hat's tables with the calibrated terms, found in the sweep table (one compile and one profile instead of the grid). Bound broken: the share of points where J-hat predicted a slower time than measured (the one-sided test; 5 % allowed).

## Baselines

- 1024x1024x1024: cublas 221.3 µs (our pick -70.8%); autotune 60.2 µs (our pick +7.1%) [256x128x64, 2 st, 256 th, swz 0], 132 of 144 ran; carver: Traceback (most recent call last):
  File "/home/lishang/work/research/AI-Infra-
- 4096x4096x4096: not run yet
- 32x4096x4096: not run yet

## Solver

- 1024: 258 variables, 1138 rows, 18 ms (HiGHS); predicted 29.5 µs
- 4096: 258 variables, 1138 rows, 18 ms (HiGHS); predicted 967.0 µs
- m32: 258 variables, 1138 rows, 17 ms (HiGHS); predicted 102.9 µs

## Error histograms (measured minus predicted, over measured)

- 1024: {'+0%': 18, '+10%': 35, '+20%': 51, '+30%': 148, '+40%': 147, '+50%': 65, '-10%': 2, '-20%': 13, '-30%': 22, '-40%': 3}; median measured/predicted 1.63
- 4096: {'+0%': 31, '+10%': 101, '+20%': 177, '+30%': 146, '+40%': 15, '-10%': 10, '-20%': 21, '-30%': 3}; median measured/predicted 1.34
- m32: {'+40%': 181, '+50%': 287}; median measured/predicted 2.01

# TileLang round 1: the numbers

Chip spark-6951 (GB10). Family gemm_basic. Terms: least squares on sweeps ['1024', 'm32'] (504 points), per-SM time scaled by 0.877 so 95 % of them lie at or above the model

| shape | grid | removed by constraints | compiled, ran, correct | spill | best measured | solver's pick | pick measured | gap pick vs best | Spearman (all) | bound broken | model pick gap |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1024x1024x1024 | 1296 | 792 (61%) | 512 of 512 | 8 (1.6%) | 63.4 µs (33.9 TF/s) | 256x128x32, 3 st, 512 th, swz 0 | 64.4 µs | +1.7% | 0.851 | 5.2% | +1.6% |
| 32x4096x4096 | 1296 | 828 (64%) | 468 of 468 | 0 (0.0%) | 200.4 µs (5.4 TF/s) | 64x256x32, 4 st, 128 th, swz 0 | 202.7 µs | +1.1% | 0.792 | 0.0% | +0.0% |

Removed by the constraints: the shared-memory rule, the register rule (launch bounds), the tile-per-thread rule. Spill: ptxas spill bytes > 0 among the configurations that ran. The solver's pick: the MILP over J-hat's tables with the calibrated terms, found in the sweep table (one compile and one profile instead of the grid). Bound broken: the share of points where J-hat predicted a slower time than measured (the one-sided test; 5 % allowed).

## Baselines

- 1024x1024x1024: not run yet
- 4096x4096x4096: not run yet
- 32x4096x4096: not run yet

## Solver

- 1024: 258 variables, 1138 rows, 17 ms (HiGHS); predicted 57.2 µs
- m32: 258 variables, 1138 rows, 17 ms (HiGHS); predicted 186.3 µs

## Error histograms (measured minus predicted, over measured)

- 1024: {'+0%': 118, '+10%': 353, '+20%': 7, '-10%': 26}; median measured/predicted 1.14
- m32: {'+0%': 250, '+10%': 218}; median measured/predicted 1.11

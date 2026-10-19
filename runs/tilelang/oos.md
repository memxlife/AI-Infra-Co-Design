## R5: out-of-sample shapes (terms calibrated on 1024^3, 4096^3 and M=32 only)

| shape | solver pick | measured | predicted | sample best (of 30) | gap pick vs sample best | bound broken on the sample | Spearman on the sample | failed / wrong / spill in the sample |
|---|---|---|---|---|---|---|---|---|
| 2048x2048x2048 | 128x256x32, 4 st, 256 th, swz 8 | 242.7 µs | 189.6 µs | 257.0 µs (64x128x32, 5 st, 128 th, swz 16) | -5.6% | 0% | 0.20 | 0 / 2 / 0 |
| 8192x8192x8192 | 128x256x32, 4 st, 256 th, swz 16 | 11539.2 µs | 10755.1 µs | 13231.1 µs (128x256x32, 3 st, 512 th, swz 16) | -12.8% | 50% | 0.66 | 0 / 2 / 0 |
| 4096x4096x1024 | 256x128x64, 2 st, 256 th, swz 16 | 461.8 µs | 676.8 µs | 471.0 µs (64x128x32, 5 st, 128 th, swz 16) | -2.0% | 71% | 0.15 | 0 / 2 / 0 |
| 128x4096x4096 | 256x64x32, 2 st, 128 th, swz 0 | 258.9 µs | 209.6 µs | 207.9 µs (64x64x64, 3 st, 256 th, swz 8) | +24.6% | 14% | 0.03 | 0 / 2 / 0 |

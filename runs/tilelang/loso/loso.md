## Step 2: leave-one-shape-out, the solver's pick measured

| shape | bound broken (LOSO) | LOSO pick | measured | predicted | best measured (sweep / sample) | pick gap |
|---|---|---|---|---|---|---|
| 1024x1024x1024 | 16% | 64x128x32, 5 st, 128 th, swz 0 | 70.7 µs | 54.9 µs | 61.5 µs | +15.0% |
| 4096x4096x4096 | 38% | 128x256x32, 4 st, 256 th, swz 0 | 2025.4 µs | 1359.7 µs | 1377.4 µs | +47.0% |
| 32x4096x4096 | 22% | 16x256x64, 2 st, 256 th, swz 0 | 202.8 µs | 168.4 µs | 200.4 µs | +1.2% |
| 2048x2048x2048 | 0% | 128x256x32, 4 st, 512 th, swz 0 | 249.8 µs | 219.2 µs | 242.7 µs | +2.9% |
| 8192x8192x8192 | 5% | 128x256x64, 2 st, 512 th, swz 4 | 11631.6 µs | 10087.2 µs | 11539.2 µs | +0.8% |
| 4096x4096x1024 | 65% | 128x256x32, 4 st, 512 th, swz 0 | 476.2 µs | 451.1 µs | 461.8 µs | +3.1% |
| 128x4096x4096 | 0% | 256x64x32, 2 st, 128 th, swz 0 | 258.0 µs | 175.1 µs | 207.9 µs | +24.1% |

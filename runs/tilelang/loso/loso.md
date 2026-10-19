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

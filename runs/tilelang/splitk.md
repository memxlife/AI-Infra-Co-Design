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

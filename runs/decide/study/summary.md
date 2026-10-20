# Tier study: MIP-optimized kernels against TileLang and cuBLASLt (GB10)

Per call in a back-to-back sequence; warm for the compute shapes, cold (a 64 MB read before each call) for the decode shapes. Microseconds.

| Shape | Timing | TileLang default | TileLang autotuned | cuBLASLt | MIP T1 | MIP T1+T2 | MIP T1+T2+T3 | MIP T1-T4 (mode) | cuBLASLt with a graph |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x1024x1024 | warm | 45.1 | 32.9 | 32.8 | 32.8 | 28.9 | 28.9 | 27.6 (graph) | 30.3 |
| 2048x2048x2048 | warm | 238.1 | 187.5 | 185.5 | 187.0 | 192.8 | 192.8 | 192.8 (stream) | 175.1 |
| 4096x4096x4096 | warm | 3614.9 | 1506.5 | 1387.5 | 1537.5 | 1519.5 | 1519.5 | 1513.2 (graph) | 1361.9 |
| 8192x8192x8192 | warm | 36804.5 | 11192.0 | 11694.7 | 12674.1 | 12674.1 | 31043.1 | 30476.6 (graph) | 11667.7 |
| 4096x4096x1024 | warm | 442.1 | 416.4 | 412.4 | 376.9 | 373.6 | 373.6 | 373.6 (stream) | 400.5 |
| 32x4096x4096 | cold | 166.9 | 159.8 | 148.7 | 121.8 | 128.0 | 128.0 | 128.0 (stream) | 140.4 |
| 128x4096x4096 | cold | 157.6 | 192.7 | 155.6 | 136.9 | 142.0 | 142.0 | -2312.2 (pin) | 153.8 |

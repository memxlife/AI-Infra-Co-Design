# Tier study: MIP-optimized kernels against TileLang and cuBLASLt (GB10)

Per call in a back-to-back sequence; warm for the compute shapes, cold (a 64 MB read before each call) for the decode shapes. Microseconds.

| Shape | Timing | TileLang default | TileLang autotuned | cuBLASLt | MIP T1 | MIP T1+T2 | MIP T1+T2+T3 | MIP T1-T4 (mode) | cuBLASLt with a graph |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x1024x1024 | warm | 45.3 | 33.6 | 32.8 | 32.9 | 29.0 | 29.0 | 27.6 (graph) | 30.4 |
| 2048x2048x2048 | warm | 250.0 | 190.6 | 178.4 | 189.1 | 189.1 | 189.1 | 186.9 (graph) | 178.0 |
| 4096x4096x4096 | warm | 3650.6 | 1525.5 | 1508.0 | 1523.9 | 1469.0 | 1469.0 | 1469.0 (stream) | 1486.8 |
| 8192x8192x8192 | warm | 37464.8 | 11499.6 | 12016.9 | 11448.9 | 11448.9 | 11448.9 | 11448.9 (stream) | 11986.1 |
| 4096x4096x1024 | warm | 436.1 | 414.0 | 391.2 | 381.6 | 375.5 | 375.5 | 373.2 (graph) | 393.2 |
| 32x4096x4096 | cold | 217.8 | 164.8 | 140.8 | 118.0 | 118.0 | 118.0 | 118.0 (stream) | 140.6 |
| 128x4096x4096 | cold | 195.2 | 171.3 | 159.7 | 124.6 | 124.6 | 124.6 | 124.6 (stream) | 156.3 |

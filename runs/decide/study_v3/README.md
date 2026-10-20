# Tier study, run 3: fair decode timing and fair inputs (Oct 20 2026)

Same seven shapes, search and tiers as run 2 (`runs/decide/study_v2/README.md`). Two changes to the timing, both
found while writing `notes/gemm_design_space.md`:

- **Decode is timed with rotating weights.** Runs 1 and 2 read 64 MB before each call and subtracted that read's
  time. It was not cold: our kernels came out reading the 33.6 MB weight at 278-286 GB/s, faster than DRAM, and the
  same kernel measured 124.6 and 191.4 µs. The weight is read again on every call, and the L2 keeps re-read data
  through a one-time stream, so part of it stayed in the L2. Run 3 keeps 8 copies of the weight (268 MB) and each call
  reads the next one. Nothing is flushed or subtracted. This is how decode runs: every layer has its own weights.
- **cuBLASLt gets the same inputs as everyone else.** Its matrices were all zeros, and its flush used a different load
  instruction. Now: random inputs, the same rotating copies, the same flush.

The L2 pin was dropped from tier 4: the flush-based test never measured it. Tier 4 is a CUDA graph.

All 1,360 candidates compiled and passed the check. Times in microseconds: warm for the compute shapes, rot for
decode. One session.

| Shape | Timing | TileLang default | TileLang autotuned | cuBLASLt | MIP T1 | MIP T1+T2 | MIP T1+T2+T3 | MIP T1-T4 | cuBLASLt with a graph | T1-T3 speedup over the better of TileLang autotuned and cuBLASLt | T1-T4 speedup over cuBLASLt with a graph |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x1024x1024 | warm | 45.3 | 33.5 | 32.8 | 32.9 | 29.0 | 29.0 | 27.7 | 30.6 | 1.13x | 1.10x |
| 2048x2048x2048 | warm | 241.2 | 187.6 | 194.6 | 186.6 | 186.6 | 186.6 | 186.6 | 193.0 | 1.01x | 1.03x |
| 4096x4096x4096 | warm | 3,629.6 | 1,548.0 | 1,556.3 | 1,573.1 | 1,510.9 | 1,510.9 | 1,510.9 | 1,544.3 | 1.02x | 1.02x |
| 8192x8192x8192 | warm | 36,995.2 | 11,266.1 | 12,136.9 | 11,271.9 | 11,271.9 | 11,271.9 | 11,271.9 | 12,173.1 | 1.00x | 1.08x |
| 4096x4096x1024 | warm | 436.8 | 410.2 | 398.9 | 386.3 | 380.4 | 380.4 | 380.4 | 407.6 | 1.05x | 1.07x |
| 32x4096x4096 | rot | 147.9 | 146.6 | 149.3 | 137.1 | 137.1 | 137.1 | 137.1 | 148.4 | 1.07x | 1.08x |
| 128x4096x4096 | rot | 161.9 | 152.3 | 169.8 | 153.2 | 153.2 | 153.2 | 152.4 | 164.8 | 0.99x | 1.08x |

Raw per-tier best (only that tier's own finalists timed):

| Shape | T1 | T1+T2 | T1+T2+T3 |
|---|---:|---:|---:|
| 1024x1024x1024 | 32.9 | 29.0 | 29.0 |
| 2048x2048x2048 | 186.6 | 186.6 | 186.6 |
| 4096x4096x4096 | 1,573.1 | 1,510.9 | 1,510.9 |
| 8192x8192x8192 | 11,271.9 | 11,271.9 | 11,271.9 |
| 4096x4096x1024 | 386.3 | 380.4 | 380.4 |
| 32x4096x4096 | 137.1 | 137.1 | 137.1 |
| 128x4096x4096 | 153.2 | 153.2 | 153.2 |

Winning kernels:

- **1024x1024x1024**: T1 `tma_ws_256x128x32_st2_th256_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `cpasync_64x128x32_st2_th256_sw0_pf0_wp2_vw0_inf_staged_sk1`; T1-T4 `cpasync_64x128x32_st2_th256_sw0_pf0_wp2_vw0_inf_staged_sk1 (graph)`
- **2048x2048x2048**: T1 `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T4 `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1 (stream)`
- **4096x4096x4096**: T1 `tma_ws_128x256x32_st4_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `cpasync_128x128x32_st3_th128_sw10_pf0_wp2_vw0_inf_direct_sk1`; T1-T4 `cpasync_128x128x32_st3_th128_sw10_pf0_wp2_vw0_inf_direct_sk1 (stream)`
- **8192x8192x8192**: T1 `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T4 `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1 (stream)`
- **4096x4096x1024**: T1 `tma_ws_256x128x64_st2_th256_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `cpasync_128x128x32_st3_th128_sw0_pf0_wp2_vw0_inf_staged_sk1`; T1-T4 `cpasync_128x128x32_st3_th128_sw0_pf0_wp2_vw0_inf_staged_sk1 (stream)`
- **32x4096x4096**: T1 `tma_ws_32x128x64_st4_th128_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `tma_ws_32x128x64_st4_th128_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T4 `tma_ws_32x128x64_st4_th128_sw0_pf0_wp0_vw0_inf_staged_sk1 (stream)`
- **128x4096x4096**: T1 `tma_ws_128x128x32_st6_th256_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `tma_ws_128x128x32_st6_th256_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T4 `tma_ws_128x128x32_st6_th256_sw0_pf0_wp0_vw0_inf_staged_sk1 (graph)`

## What it shows

1. **MIP is never slower than the best baseline by more than noise, and wins on three shapes.** Between runs 2 and 3
   the same kernels moved 1-3 %, so differences under about 3 % are ties.
   - wins: 1024^3 1.13x, decode M=32 1.07x, 4096x4096x1024 1.05x;
   - ties: 2048^3 1.01x, 4096^3 1.02x, 8192^3 1.00x (the same kernel TileLang's AutoTuner picked), decode M=128 0.99x
     (TileLang autotuned 152.3, MIP 153.2).
   - Against cuBLASLt alone MIP is faster on every shape: 1.04x to 1.13x.
2. **The decode wins of run 2 were mostly the timing error.** M=32 went from 1.19x to 1.07x, M=128 from 1.28x to
   0.99x. The new times are physical: the M=32 winner reads at 247 GB/s, M=128 at 226 GB/s.
3. **cuBLASLt was faster on all-zero inputs.** At 2048^3 it went from 178.4 µs (zeros, run 2) to 194.6 µs (random,
   run 3), while our kernels and TileLang's moved 1-2 %. So run 2's 2048^3 loss was not a gap in our kernel code.
   Other shapes moved 0-3 %.
4. **What each tier buys** is unchanged: T2 (the cp.async path, warp policy, direct store) pays at 1024^3, 4096^3 and
   4096x4096x1024; T3 (split-K) never wins; T4 (a graph) saves 1-1.5 µs on the shortest kernel.
5. **Decode likes depth.** The new M=128 winner is TMA 128x128x32 with 6 stages, the deepest pipeline in the search.

## Still open

- 247 GB/s at M=32 is above the 237-242 GB/s the DRAM probes reached, so the probes do not find DRAM's ceiling.
- L2 pinning for decode needs its own experiment: which layers' weights to keep across a decode step.
- The study skips the simulator at 4096^3 and 8192^3 ("minutes per kernel"), but a 4096^3 kernel simulates in 13 s.

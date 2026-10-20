# Tier study: MIP-picked GEMM kernels against TileLang and cuBLASLt on the GB10 (Oct 20 2026)

> **Superseded for the decode shapes (Oct 20 2026).** The decode times here were timed "cold" by flushing the L2 and
> subtracting the flush. That was not cold: part of the weight stayed in the L2. cuBLASLt also got all-zero inputs. See
> `runs/decide/study_v3/README.md` for fair times.

Seven shapes, fp16 in, fp32 accumulate. Each tier adds decisions to the search:
- **T1**: TileLang's own tuning knobs (tile sizes, stages, threads, swizzle) on its TMA warp-specialized structure.
- **T1+T2**: adds the cp.async path, direct or staged C store, prefetch and warp policy.
- **T1+T2+T3**: adds split-K (1, 2, 4, 8).
- **T1-T4**: adds launch-level choices: a CUDA graph, and for the decode shapes an L2 persistence window on the weights.

For each tier, the price model scored every legal configuration and the MILP took its top 30, plus swizzle, prefetch
and warp-policy variants. Every kernel was compiled and checked for correctness (all 839 passed). The simulator
scored them, except at 4096^3 and 8192^3. The model's top 5 and the simulator's top 5 were then timed on the Spark.
TileLang's default kernel, TileLang's autotuned pick (its AutoTuner over 144 configs) and cuBLASLt were timed in the
same session with the same method. Times are per call in a back-to-back sequence. Compute shapes are warm. Decode
shapes are cold: a 64 MB read before every call, with its time subtracted. Microseconds.

A tier's number is the fastest kernel measured in that tier or any smaller tier, because each tier's search space
includes the smaller ones. The raw per-tier numbers are listed after the main table.

| Shape | Timing | TileLang default | TileLang autotuned | cuBLASLt | MIP T1 | MIP T1+T2 | MIP T1+T2+T3 | MIP T1-T4 | cuBLASLt with a graph | T1-T3 speedup over the better of TileLang autotuned and cuBLASLt | T1-T4 speedup over cuBLASLt with a graph |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x1024x1024 | warm | 45.1 | 32.9 | 32.8 | 32.8 | 28.9 | 28.9 | 27.6 | 30.3 | 1.13x | 1.10x |
| 2048x2048x2048 | warm | 238.1 | 187.5 | 185.5 | 187.0 | 187.0 | 187.0 | 187.0 | 175.1 | 0.99x | 0.94x |
| 4096x4096x4096 | warm | 3,614.9 | 1,506.5 | 1,387.5 | 1,537.5 | 1,519.5 | 1,519.5 | 1,513.2 | 1,361.9 | 0.91x | 0.90x |
| 8192x8192x8192 | warm | 36,804.5 | 11,192.0 | 11,694.7 | 12,674.1 | 12,674.1 | 12,674.1 | 12,674.1 | 11,667.7 | 0.88x | 0.92x |
| 4096x4096x1024 | warm | 442.1 | 416.4 | 412.4 | 376.9 | 373.6 | 373.6 | 373.6 | 400.5 | 1.10x | 1.07x |
| 32x4096x4096 | cold | 166.9 | 159.8 | 148.7 | 121.8 | 121.8 | 121.8 | 121.8 | 140.4 | 1.22x | 1.15x |
| 128x4096x4096 | cold | 157.6 | 192.7 | 155.6 | 136.9 | 136.9 | 136.9 | 136.9 | 153.8 | 1.14x | 1.12x |

Raw per-tier best (only that tier's own finalists timed):

| Shape | T1 | T1+T2 | T1+T2+T3 |
|---|---:|---:|---:|
| 1024x1024x1024 | 32.8 | 28.9 | 28.9 |
| 2048x2048x2048 | 187.0 | 192.8 | 192.8 |
| 4096x4096x4096 | 1,537.5 | 1,519.5 | 1,519.5 |
| 8192x8192x8192 | 12,674.1 | 12,674.1 | 31,043.1 |
| 4096x4096x1024 | 376.9 | 373.6 | 373.6 |
| 32x4096x4096 | 121.8 | 128.0 | 128.0 |
| 128x4096x4096 | 136.9 | 142.0 | 142.0 |

Winning kernels:

- **1024x1024x1024**: T1 `tma_ws_256x128x32_st2_th256_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `cpasync_64x128x32_st2_th256_sw0_pf0_wp2_vw0_inf_staged_sk1`; T1-T4 `cpasync_64x128x32_st2_th256_sw0_pf0_wp2_vw0_inf_staged_sk1 (graph)`
- **2048x2048x2048**: T1 `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T4 `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1 (stream)`
- **4096x4096x4096**: T1 `tma_ws_128x256x32_st4_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `cpasync_128x128x32_st3_th128_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T4 `cpasync_128x128x32_st3_th128_sw10_pf0_wp0_vw0_inf_staged_sk1 (graph)`
- **8192x8192x8192**: T1 `tma_ws_256x128x32_st4_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `tma_ws_256x128x32_st4_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T4 `tma_ws_256x128x32_st4_th256_sw10_pf0_wp0_vw0_inf_staged_sk1 (stream)`
- **4096x4096x1024**: T1 `tma_ws_256x128x32_st4_th256_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `cpasync_128x128x32_st3_th128_sw0_pf0_wp2_vw0_inf_staged_sk1`; T1-T4 `cpasync_128x128x32_st3_th128_sw0_pf0_wp2_vw0_inf_staged_sk1 (stream)`
- **32x4096x4096**: T1 `tma_ws_32x128x32_st4_th256_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `tma_ws_32x128x32_st4_th256_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T4 `tma_ws_32x128x32_st4_th256_sw0_pf0_wp0_vw0_inf_staged_sk1 (stream)`
- **128x4096x4096**: T1 `tma_ws_128x128x32_st4_th128_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `tma_ws_128x128x32_st4_th128_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T4 `tma_ws_128x128x32_st4_th128_sw0_pf0_wp0_vw0_inf_staged_sk1 (stream)`

## What it shows

1. **The MIP pick beats both TileLang autotuned and cuBLASLt on 4 of 7 shapes:** 1024^3 (1.13x), 4096x4096x1024
   (1.10x), decode M=32 (1.22x) and decode M=128 (1.14x). It ties at 2048^3 (0.99x). It loses at 4096^3 (0.91x) and
   8192^3 (0.88x).
2. **T2 is the tier that pays.** At 1024^3 the cp.async path with warp policy 2 takes 32.8 us down to 28.9 us. It
   gives small gains at 4096^3 and 4096x4096x1024. TileLang's AutoTuner can't reach this kernel at all.
3. **T3 (split-K) never won.** At 8192^3 the price model said split-8 would take 15.0 ms against 21.0 ms. It measured
   31-46 ms. Split-8 adds up an 8192x8192 fp32 output eight times with atomics, about 2 GB of extra traffic, and the
   model undercounts it.
4. **T4: a CUDA graph helps only the shortest kernel** (1024^3: 28.9 to 27.6 us). The L2 pin made decode slower with
   this method (M=32: 128.0 to 200.3 us). One pin run gave a negative time and was dropped. A likely cause is the
   64 MB flush read fighting the pinned window. So this result rules out "pin plus flush", not pinning itself.
5. **Why the big shapes lose: a bug in the study driver, not the model.** The driver priced every configuration
   with swizzle fixed at 0 (`run.py` line 52). It then gave the swizzle-10 copy the same price, with a wrong comment
   saying the model does not price swizzle. The model does price it. At 8192^3, 256x128x32 with 4 stages: the model
   gives 21.10 ms without swizzle and 11.48 ms with it; measured, 19.5 ms and 12.7 ms. TileLang's pick, 128x256x64
   with 2 stages: the model gives 28.70 ms and 11.99 ms; measured with swizzle, 11.19 ms. So the MIP ranked the big
   shapes on swizzle-0 prices, which all sit near 21 ms. The fix is to make swizzle a real decision in the columns
   and rerun 2048^3, 4096^3 and 8192^3.
6. **TileLang autotuned can be worse than TileLang's default.** At M=128 cold it takes 192.7 us against 157.6 us.
   Its AutoTuner times warm, with an L2 left dirty by do_bench, so it picks for the wrong case.

## Flaws in this run, to fix next

- Each tier timed only its own top picks. So raw T1+T2 came out slower than T1 at 2048^3, M=32 and M=128 (table
  above). Fix: carry the smaller tiers' finalists into each bigger tier.
- Tier 4 was tried on the raw T3 winner, not the overall best. At 8192^3 that was the slow split-K kernel.
- Price swizzle in the columns (the bug in item 5). Price model fix: the split-K atomic traffic.
- One session only. Back-to-back timing of 33-35 MB working sets varies about 10% between sessions.

Files: one JSON per shape here (every candidate: price, simulator time, registers, error, measured times). The tables
come from `python -m agentic.decide.study.report runs/decide/study`. The driver is `agentic/decide/study/run.py`. The
logs are `run_all.log`, `study.log`, `watchdog.log` and `swap.log`. The study stalled for 3 hours before 12:17 on a
broken wait loop, since fixed. At 12:27 it was switched at a shape boundary to parallel builds.

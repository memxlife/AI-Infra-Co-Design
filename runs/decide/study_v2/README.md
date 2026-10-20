# Tier study, run 2: swizzle priced (Oct 20 2026)

Same seven shapes, method and baselines as run 1 (`runs/decide/study/README.md`), with four changes to the driver
(commits 41bbf7e and 8fe2266):
- Swizzle {0, 10} is a decision the price model prices. Run 1 priced everything at swizzle 0 by mistake.
- Split-K pays for reading and writing C in DRAM when the fp32 C is larger than the L2.
- Each tier keeps the smaller tiers' finalists, so tier k is never slower than tier k-1, and tier 4 runs on the
  overall best.
- The MILP takes its top 60 (swizzle doubled the columns). Finalists are the model's best distinct kernels with all
  their prefetch and warp-policy variants.

All 1,360 candidates compiled and passed the check. Times are per call in a back-to-back sequence, warm for compute
shapes and cold for decode, in microseconds. Each row was timed in one session, so the comparisons within a row are
fair. Baselines move between sessions: cuBLASLt at 4096^3 measured 1,387.5 us in run 1 and 1,508.0 us here.

| Shape | Timing | TileLang default | TileLang autotuned | cuBLASLt | MIP T1 | MIP T1+T2 | MIP T1+T2+T3 | MIP T1-T4 | cuBLASLt with a graph | T1-T3 speedup over the better of TileLang autotuned and cuBLASLt | T1-T4 speedup over cuBLASLt with a graph |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x1024x1024 | warm | 45.3 | 33.6 | 32.8 | 32.9 | 29.0 | 29.0 | 27.6 | 30.4 | 1.13x | 1.10x |
| 2048x2048x2048 | warm | 250.0 | 190.6 | 178.4 | 189.1 | 189.1 | 189.1 | 186.9 | 178.0 | 0.94x | 0.95x |
| 4096x4096x4096 | warm | 3,650.6 | 1,525.5 | 1,508.0 | 1,523.9 | 1,469.0 | 1,469.0 | 1,469.0 | 1,486.8 | 1.03x | 1.01x |
| 8192x8192x8192 | warm | 37,464.8 | 11,499.6 | 12,016.9 | 11,448.9 | 11,448.9 | 11,448.9 | 11,448.9 | 11,986.1 | 1.00x | 1.05x |
| 4096x4096x1024 | warm | 436.1 | 414.0 | 391.2 | 381.6 | 375.5 | 375.5 | 373.2 | 393.2 | 1.04x | 1.05x |
| 32x4096x4096 | cold | 217.8 | 164.8 | 140.8 | 118.0 | 118.0 | 118.0 | 118.0 | 140.6 | 1.19x | 1.19x |
| 128x4096x4096 | cold | 195.2 | 171.3 | 159.7 | 124.6 | 124.6 | 124.6 | 124.6 | 156.3 | 1.28x | 1.25x |

Raw per-tier best (only that tier's own finalists timed):

| Shape | T1 | T1+T2 | T1+T2+T3 |
|---|---:|---:|---:|
| 1024x1024x1024 | 32.9 | 29.0 | 29.0 |
| 2048x2048x2048 | 189.1 | 189.1 | 189.1 |
| 4096x4096x4096 | 1,523.9 | 1,469.0 | 1,469.0 |
| 8192x8192x8192 | 11,448.9 | 11,448.9 | 11,448.9 |
| 4096x4096x1024 | 381.6 | 375.5 | 375.5 |
| 32x4096x4096 | 118.0 | 118.0 | 118.0 |
| 128x4096x4096 | 124.6 | 124.6 | 124.6 |

Winning kernels:

- **1024x1024x1024**: T1 `tma_ws_256x128x32_st2_th256_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `cpasync_64x128x32_st2_th256_sw0_pf1_wp2_vw0_inf_staged_sk1`; T1-T4 `cpasync_64x128x32_st2_th256_sw0_pf1_wp2_vw0_inf_staged_sk1 (graph)`
- **2048x2048x2048**: T1 `tma_ws_128x256x64_st2_th256_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `tma_ws_128x256x64_st2_th256_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T4 `tma_ws_128x256x64_st2_th256_sw0_pf0_wp0_vw0_inf_staged_sk1 (graph)`
- **4096x4096x4096**: T1 `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `cpasync_128x128x32_st3_th128_sw10_pf0_wp2_vw0_inf_staged_sk1`; T1-T4 `cpasync_128x128x32_st3_th128_sw10_pf0_wp2_vw0_inf_staged_sk1 (stream)`
- **8192x8192x8192**: T1 `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T4 `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1 (stream)`
- **4096x4096x1024**: T1 `tma_ws_256x128x64_st2_th256_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `cpasync_128x128x32_st3_th128_sw10_pf0_wp2_vw0_inf_staged_sk1`; T1-T4 `cpasync_128x128x32_st3_th128_sw10_pf0_wp2_vw0_inf_staged_sk1 (graph)`
- **32x4096x4096**: T1 `tma_ws_32x128x64_st4_th128_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `tma_ws_32x128x64_st4_th128_sw0_pf0_wp0_vw0_inf_staged_sk1`; T1-T4 `tma_ws_32x128x64_st4_th128_sw0_pf0_wp0_vw0_inf_staged_sk1 (stream)`
- **128x4096x4096**: T1 `tma_ws_128x256x32_st3_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T3 `tma_ws_128x256x32_st3_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`; T1-T4 `tma_ws_128x256x32_st3_th256_sw10_pf0_wp0_vw0_inf_staged_sk1 (stream)`

## What it shows

1. **MIP wins or ties on 6 of 7 shapes, against the better of TileLang autotuned and cuBLASLt:**
   - wins: 1024^3 1.13x, 4096x4096x1024 1.04x, decode M=32 1.19x, decode M=128 1.28x;
   - ties: 4096^3 1.03x and 8192^3 1.00x. At 8192^3 the MIP picked the same kernel TileLang's AutoTuner picked
     (128x256x64, 2 stages, swizzle 10). The 0.4% gap is noise.
   - loses: 2048^3, 0.94x. Here MIP and TileLang autotuned agree (189.1 and 190.6 us). cuBLASLt is 6% faster than
     every TileLang-template kernel tried. That gap is in the kernel code, not in the choice of settings.
2. **Pricing swizzle fixed the big shapes:**
   - 8192^3: 12,674 to 11,449 us, a 10% gain. cuBLASLt moved 3% the other way, so the gain is real.
   - 4096^3: 1,519.5 to 1,469.0 us, within this run's session-to-session change.
   - decode M=128: 136.9 to 124.6 us, a new kernel (128x256x32, 3 stages, swizzle 10).
3. **T2 (the cp.async path and warp policy) is the tier that pays:**
   - 1024^3: 32.9 to 29.0 us;
   - 4096^3: 1,523.9 to 1,469.0 us;
   - 4096x4096x1024: 381.6 to 375.5 us.
4. **T3 (split-K) still never wins.** It is now priced correctly, so it no longer pushes good kernels off the list.
5. **T4:** a CUDA graph saves 1-2 us on the short kernels (1024^3, 2048^3, 4096x4096x1024). The L2 pin is still
   slower with this cold method: M=32 118.0 to 162.6 us.

## Still open

- 2048^3: why cuBLASLt's kernel is 6% faster than any template kernel.
- The L2-pin test fights the 64 MB flush; it needs a method where the pin can work.
- `study_out_v2_top30` on the Spark holds the first try of this run (top 30, stopped after 1024^3). It is not
  copied here.

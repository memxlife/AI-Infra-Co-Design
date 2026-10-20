# Issue #5: an outside contributor's GB10 kernels against the simulator (Oct 20 2026)

Source: the bundle attached to issue #5, `gb10-simulator-calibration-20261020-v2.zip`, SHA256
`01f77fbadaa9ad61921d469edb62f61ac0c263b452a1307e5ad3dfab8207ad65`, from the contributor's fork (commit cc17474). Their
SASS and kernel sources were read as data. Their tools and binaries were not run. The kernel source for item 3 is
TileLang's dequantize-GEMV example as packaged in the bundle (`cases/w4_gemv/candidate.py`); it is not copied here.

Their three worst cases, simulated against measured (their numbers): `gemm` +82.0 %, `w4_gemv` +71.9 %,
`gqa_decode` +54.9 %.

## 1. `gemm` (+82 %): the swizzle never reached the simulator

The kernel calls `T.use_swizzle(10)`, but its case file has no swizzle, and their adapter reads `p.get("swizzle", 0)`.
The kernel also puts M tiles on blockIdx.x (`T.Kernel(ceildiv(M, bm), ceildiv(N, bn))`). Replayed with their SASS
(`replay_tma_gemm.py`, after commit 6126651 added `kernel=` and `h["grid_x"]`):

| Input, 4096 x 14336 x 4096 | Simulated (µs) | vs measured 5,081.9 |
|---|---:|---:|
| swizzle 0, as passed | 9,251.4 | +82 % |
| swizzle 10, grid x walks N | 5,189.2 | +2.1 % |
| swizzle 10, grid x walks M (the real launch) | 4,893.2 | −3.7 % |
| control `gemm_basic`, swizzle 8 | 5,038.0 | +3.3 % (vs 4,878.5) |

## 2. `I2F.F16` is not the GEMV's problem; `I2FP` was mischarged

`w4_gemv`'s loop runs 32 `I2F.F16` per pass, charged at the SFU rate without a measurement. Probe `cvt.cu`
(`calibration/spark-6951/cvt.txt`, commit ee6534c): `I2F.F16` takes 8.06 cycles per warp instruction per sub-partition and
shares the SFU with `MUFU`, as the simulator assumed. `I2FP.F32.S32` runs on the ALU at 2 cycles; the simulator charged
8, now fixed. The simulator replays the probe's own SASS within 2 % on all 7 modes (`agentic/gb10/sim/cvttest.py`).

## 3. `w4_gemv` (+72 %): the measurement was warm

1 x 14336 x 4096 with int4 weights (29.4 MB). Timed on our GB10 with `time_w4.py` (relative error 2e-4 against an
fp32 reference):

| Timing | µs per call | Weights at |
|---|---:|---:|
| their protocol: a CUDA graph of 30 calls on the same weights | 97.0 | 303 GB/s |
| back to back on the same weights | 100.7 | 292 GB/s |
| rotating 9 weight copies (264 MB), back to back | 125.3 | 234 GB/s |
| rotating 9 weight copies, in a graph | 122.9 | 239 GB/s |
| DRAM floor at 237 GB/s | 123.9 | |

Same weights each call is partly warm: the L2 keeps much of a re-read 29 MB set (our `stream_b2b.cu` gives 303 GB/s at
28 MiB, matching). Timed the way decode runs, the kernel sits at the DRAM floor, and their simulator result (130.2 µs)
is +4 to +6 %. Their own measurement, 75.7 µs (388 GB/s), we could not reproduce: the same protocol gives 97.0 µs here.

## 4. `gqa_decode` (+55 %): the mask loads were counted as DRAM traffic

The loop issues 16 one-byte mask loads (`LDG.E.U8`) per pass. The adapter described each as a 128-byte line read
from DRAM. The whole mask is 512 KB, 0.2 % of the 268 MB of K and V, and the 8 heads of a batch read the same lines,
so after the first touch they come from the cache. Replayed with our SM model and the contributor's other inputs
(`replay_gqa_decode.py`: one SM, 256 loop passes, all K/V copies from DRAM at 1/32 of 237 GB/s):

| Mask loads | Simulated (µs) | vs measured 1,089.3 | DRAM busy |
|---|---:|---:|---:|
| 128-byte DRAM reads (as passed) | 1,684.7 | +55 % (reproduces theirs) | 90 % |
| served from the cache | 1,134.9 | +4.2 % | 100 % |

So this is an input mistake, not a missing overlap mechanism. An earlier comment on the issue said the 1-stage loop
paid DRAM latency every step; that was wrong. With the right inputs the simulator keeps DRAM busy, as the hardware does
(246 GB/s). The separate gap in `notes/gb10_simulator_design.md` (2 against 4 stages of a TMA decode GEMM at M = 128)
is still open.

## 5. The two cases without a correct result

**`gqa_splitkv` (wrong answers on scaled inputs): the per-split log-sum-exp is kept in fp16.** The kernel stores each
split's log-sum-exp in `glse = T.alloc_global(..., dtype)` and the merge kernel reads it into
`lse_local = T.alloc_fragment(..., dtype)`, both fp16. The merge weights each split by `exp2(lse_split - lse_total)`.
With Q and K scaled by 20 (the contributor's "scaled" check), log-sum-exp values reach the thousands, where fp16 keeps
only every second integer, so a split's weight can be off by up to 2x. Checked on our GB10 (`check_gqa_splitkv.py`,
the contributor's inputs and pass rule, rtol = atol = 0.01):

| Kernel | random | zeros | Q, K x 20 |
|---|---:|---:|---:|
| as written | 0.000095 pass | 0 pass | 0.314453 FAIL (their number) |
| `glse` and `lse_local` in fp32 | 0.000031 pass | 0 pass | 0.001953 pass |

The same fp16 buffer is in TileLang's own examples (pinned commit): `examples/flash_decoding/example_gqa_decode.py`
line 72, `examples/flash_decoding/example_mha_inference.py` and `examples/deepseek_mla/example_mla_decode.py`.

**`block_sparse_attention` (timed out after 360 s on the contributor's machine): not reproduced.** With the
contributor's mask (diagonal blocks plus column 0), inputs (random, zeros, x3) and pass rule, the kernel passes all
three checks on our GB10 and runs at 624 µs per call (`check_block_sparse_attention.py`). TileLang's own block-sparse
example also passes here. Their log stops right after the compile with no output, so where it stalled is unknown;
their environment differs (driver 580.95.05, a PyTorch that reports support only up to compute capability 12.0).
One weakness we did see: with a mask that leaves some row blocks no visible key block, the kernel returns NaN
(0 / 0 in the final normalization). TileLang's example guards the related -inf case; the contributor's copy dropped
that guard. Their mask never triggers it.

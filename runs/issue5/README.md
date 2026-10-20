# Issue #5: an outside contributor's GB10 kernels against the simulator (Oct 20 2026)

Source: the bundle attached to issue #5, `gb10-simulator-calibration-20261020-v2.zip`, SHA256
`01f77fbadaa9ad61921d469edb62f61ac0c263b452a1307e5ad3dfab8207ad65`, from the contributor's fork (commit cc17474). Their
SASS and kernel sources were read as data. Their tools and binaries were not run. The kernel source for item 3 is
TileLang's dequantize-GEMV example as packaged in the bundle (`cases/w4_gemv/candidate.py`); it is not copied here.

Their three worst cases, simulated against measured (their numbers): `gemm` +82.0 %, `w4_gemv` +71.9 %,
`gqa_decode` +54.9 %.

## Status of every case (Oct 21 2026)

20 kernels plus 2 whole-model graphs (out of scope). By the contributor's checks (random, zeros and scaled inputs on
their GB10), 18 kernels were correct; with our fix and rerun, all 20 are correct now. 10 have a simulator comparison,
and all 10 are within 10 % once their inputs are right. 8 run correctly but the simulator cannot describe their
structure yet. Times are the contributor's measurements unless marked otherwise.

**With a simulator comparison (10)**

| Case | Measured | Their error | Status now |
|---|---:|---:|---|
| `gemm` | 5,081.9 µs | +82 % | the adapter dropped the kernel's swizzle; with its real launch order −3.7 % (section 1) |
| `w4_gemv` | 75.7 µs | +72 % | their timing re-read the same weights (partly warm); against rotating-weight timing on our GB10 (122.9-125.3 µs) +4 to +6 %; their 75.7 µs not reproduced, we get 97.0 with their protocol (sections 2, 3) |
| `gqa_decode` | 1,089.3 µs | +55 % | 1-byte mask loads counted as DRAM traffic; fixed +4.2 % (section 4) |
| `gemm_basic` | 4,878.5 µs | +3.3 % | reproduced exactly (5,038.0 µs) |
| `softmax_resident` | 578.2 µs | +8.3 % | not re-checked; mostly the memory floor |
| `rmsnorm` | 1,169.5 µs | +6.9 % | not re-checked; mostly the memory floor |
| `elementwise_add` | 857.1 µs | +6.1 % | not re-checked; mostly the memory floor |
| `rmsnorm_resident` | 1,187.3 µs | +5.3 % | not re-checked; mostly the memory floor |
| `softmax` | 595.9 µs | +5.1 % | not re-checked; mostly the memory floor |
| `gemv` | 498.6 µs | +0.2 % | not re-checked |

**Correct and timed, no simulator result yet (8)**

| Case | Measured | What the simulator lacks |
|---|---:|---|
| `gemm_splitk` | 25,008.9 µs (with zeroing and conversion) | split-K with its extra kernels |
| `gemm_persistent` | 5,327.8 µs | persistent blocks (48 blocks looping over tiles) |
| `conv2d` | 3,497.6 µs | convolution |
| `grouped_gemm` | 3,048.6 µs | grouped GEMM |
| `gqa_prefill` | 1,990.1 µs | prefill attention |
| `cumsum` | 726.6 µs | scan collectives |
| `gemv_reducer` | 518.5 µs | its copy, loop and reducer pattern |
| `w4a16` | 206.8 µs | a TMA int4 GEMM |

**Failed in the package, resolved here (2)**

| Case | Their result | Status now |
|---|---|---|
| `gqa_splitkv` | wrong on Q, K x 20 (max abs error 0.314) | the split log-sum-exp is kept in fp16; in fp32 it passes (0.002); the same bug is in TileLang's own examples, reported as tile-ai/tilelang#3585; not timed yet (section 5) |
| `block_sparse_attention` | timed out after 360 s | no hang on our GB10: passes all three checks, 624.2 µs per call (our measurement); returns NaN if a row block sees no keys, which their mask never does (section 5) |

**Out of scope (2):** `llama_network` and `matmul_bias_relu` are whole TVM graphs, included only to mark the boundary.

**Who checked what.** Correctness for 18 kernels comes from the contributor's checks on their machine. We re-ran 3 on
our GB10: `w4_gemv`, `gqa_splitkv` (as shipped and fixed) and `block_sparse_attention`.

**Open with the contributor (issue #5):** the 75.7 µs `w4_gemv` time, and where their `block_sparse_attention` run
stalled.

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
TileLang's own `example_gqa_decode.py` shows it too (`repro_tilelang_gqa_decode.py`: batch 4, 32 heads, 8 groups,
16,384 keys, block_N 64, 8 splits, 1 stage; block_N 128 with 2 stages needs 147 KB of shared memory, more than the
GB10's 99 KB). Against an fp32 reference at Q, K x 20: 0.1670 as shipped, 0.0013 with the two buffers in fp32. At
batch 1 and 8,192 keys the error did not show (0.0010): a row goes wrong only when two splits have similar maxima. The
example's own `ref_program` computes scores in fp16 and is itself off by 0.3958 at x 20, so its check cannot see the
bug. Reported upstream as tile-ai/tilelang#3585.

**`block_sparse_attention` (timed out after 360 s on the contributor's machine): not reproduced.** With the
contributor's mask (diagonal blocks plus column 0), inputs (random, zeros, x3) and pass rule, the kernel passes all
three checks on our GB10 and runs at 624 µs per call (`check_block_sparse_attention.py`). TileLang's own block-sparse
example also passes here. Their log stops right after the compile with no output, so where it stalled is unknown;
their environment differs (driver 580.95.05, a PyTorch that reports support only up to compute capability 12.0).
One weakness we did see: with a mask that leaves some row blocks no visible key block, the kernel returns NaN
(0 / 0 in the final normalization). TileLang's example guards the related -inf case; the contributor's copy dropped
that guard. Their mask never triggers it.

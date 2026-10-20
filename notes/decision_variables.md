# Decision variables for GEMM-like kernels on the GB10

Why this exists: the goal is a solver that can choose every decision that changes how fast a kernel runs, including
the ones that use the GB10's more advanced hardware. The microarchitecture model (`notes/gb10_model.md`) says what each
decision does to the hardware and prices it; this file lists the decisions. Each has a tier, a domain, where the
choice is made (its binder), the hardware mechanism it acts on, the evidence, the measured effect, and whether the
solver can choose it yet.

Four tiers:

1. **TileLang's tuning knobs**: what TileLang's own autotuner sweeps.
2. **TileLang controls not offered for tuning**: options in TileLang's language or pass configs that change the
   generated code but that no autotuner list enumerates. We lift them into the template.
3. **Skeleton structures**: program shapes TileLang can express, reached by choosing a different TileLang program
   (split-K with atomics, stream-K, GEMV, the cp.async pipeline without a producer group).
4. **Hardware features no TileLang skeleton can state**: they need our own CUDA/PTX, a runtime or launch setting, or a
   choice that spans kernels. TileLang itself is never changed. The survey moved several candidates down to tier 2:
   TileLang can already ask for L2 persistence (`annotate_l2_hit_ratio`), cluster multicast (`copy_cluster`), TMA
   cache hints and programmatic dependent launch. What stays in tier 4 is outside any kernel (CUDA graphs, streams,
   CPU memory traffic) and what TileLang hardcodes (the L2 promotion size of every TMA descriptor).

Machine-readable form: `agentic/decide/registry_tier34.yaml` (tiers 3 and 4); tiers 1 and 2 are the slots of the
unified template `agentic/tilelang/families/gemm/kernel.py.j2`.

## What each TileLang-level decision does (measured, Oct 20 2026)

Cover sets (`agentic/decide/cover.py`, `runs/decide/cover_1024`, `runs/decide/cover2`): every value of every decision
around a base point, compiled, checked against torch (all 70 correct), timed with the corrected method (launch hidden;
warm, and with the L2 cleared by reading). "Same code" means the machine code is identical to the base point's.

| Decision | Around the TMA base (1024³) | Around the cp.async base (1024³) | Around the TMA base (M = 128 decode, cold) |
|---|---|---|---|
| tile (block_M, block_N, block_K) | 36.7-116.5 µs | 32.5-53.1 µs | 150-449 µs |
| num_stages 2-6 | flat (45-47) | 2 stages best (32.5 vs 34.6) | 4+ best (166 → 150) |
| threads 128 / 256 | 256 faster (40.8 vs 46.9) | 256 faster | same |
| C store direct / staged | staged −9 % | staged −12 % | staged −1.4 % |
| split_k 1 / 2 / 4 | — | 1 best | 1 best cold |
| prefetch 0 / 1 | **same code** | 1: −6 % | same |
| warp_policy square / rows / cols | code changes, same time | cols −6 %, rows +12 % | same |
| vec_width (copy width) | **same code** | same | same |
| smem_layout inferred / swizzled | **same code** | same | same |
| swizzle panel 0-16 | same warm, −4 % cold | same | same |
| TMA with the producer group off | **same code as cp.async** | — | — |

Two consequences for the solver: (1) the TMA path needs TileLang's producer warp group (without it TileLang emits the
cp.async kernel), so "copy path" and "producer group" are one decision; (2) prefetch and warp policy matter only on the
cp.async path, so they cannot be priced apart from the path.

## Tiers 1 and 2: what TileLang offers (source survey of the pinned commit, Oct 20 2026)

**Tier 1, what TileLang's own tuning sweeps.** The AutoTuner has no search space of its own: it sweeps the list the
user writes (any kernel parameter, plus per-config `pass_configs`). The carver enumerates block tile, warp tile (hence
threads), block_K and copy vector width, and fixes the rest per architecture. On sm_120/121 the carver sets **no
pipelining** (`num_stages` 0) and knows nothing of TMA, warp specialization or clusters, so its configurations leave the
GB10's copy pipeline off. Every `T.gemm` on sm_120/121 lowers to `mma.sync` (WGMMA is Hopper-only, tcgen05 is
sm_100-only); the one sm_120-specific GEMM path is block-scaled fp4 (`mma.sync m16n8k64 kind::mxf4nvf4`).

**Tier 2, TileLang controls that change the code but no tuner offers** (file:line in the survey,
`runs/decide/tilelang_controls_survey.md`). Status: T = in the unified template; M = measured here; open = not yet.

| Control | Values | What it changes on the GB10 | Status |
|---|---|---|---|
| copy path: `T.copy(disable_tma=)`, `prefer_instruction=` | tma / cp_async / sync | TMA loads need the producer warp group (warp specialization on); without it TileLang emits cp.async | T, M |
| `tl.disable_warp_specialized` | on / off | the 128-thread producer warp group and its mbarriers; the register budget | T, M |
| `T.Pipelined(num_stages, order, stage)` | depth; manual order | buffers in flight; the trigger for warp specialization | T, M (prefetch: −6 % on cp.async, no-op on TMA) |
| `T.gemm(policy=)` | square / full rows / full columns | how warps split the C tile: ldmatrix traffic per warp | T, M (cols −6 %, rows +12 % on cp.async) |
| C store through `C_shared` | direct / staged | stmatrix + TMA store instead of half-sector STG | T, M (−9 to −12 % at 1024³) |
| `T.use_swizzle(panel, order)` | panel 0-16, row / column | block traversal for L2 reuse | T (row), M; column order open |
| `T.annotate_layout` | inferred / swizzled / linear | bank conflicts; linear with TMA is wrong | T, M |
| `eviction_policy` on TMA copies | normal / evict_first / evict_last | L2 cache hint per TMA box (cp.async ignores it) | open: streamed weights could be evict_first |
| `T.annotate_l2_hit_ratio({buf: r})` | ratio per buffer | the host sets an access-policy window and the persisting L2 size | open: the tier-4 L2 pinning through TileLang (3.4-5x measured with the runtime API) |
| `-DTL_ENABLE_L2_PREFETCH` (compile flag) | on / off | `.L2::128B` prefetch on cp.async and predicated loads | open |
| `T.Kernel(cluster_dims=)` + `T.copy_cluster(cluster_mask=)` | cluster 1-8 | TMA multicast: one L2 read for several SMs (clusters and distributed shared memory work on the GB10) | open |
| `T.tma_copy(barrier=)` split-phase, `T.ws`, `T.annotate_ws_schedule` | manual schedules | how many TMA boxes the producer issues before waiting | open |
| `T.annotate_producer_reg_dealloc` / `consumer_reg_alloc` | 24-240 | setmaxnreg split between producer and consumer warps | open |
| `T.annotate_min_blocks_per_sm(n)` | 1, 2, ... | `__launch_bounds__` minimum blocks: caps registers, raises occupancy | open |
| `T.Persistent` / `T.PersistentTileScheduler` | wave size, swizzle, cluster | persistent tile loop instead of one block per tile | open |
| `T.pdl_trigger` / `T.pdl_sync` | on / off | programmatic dependent launch: the next kernel's start overlaps this one's end | open (launch costs 2.6 µs per kernel) |
| `T.atomic_add(use_tma=)` | off / on | bulk TMA reduce-add of C from shared memory for split-K | open |
| `T.mma_gemm_blockscaled` | nvf4 m16n8k64 | block-scaled fp4 | open (rate not measured) |
| `tl.disable_vectorize_256`, `tl.enable_fast_math`, `tl.ptxas_register_usage_level`, `tl.layout_cost_model` | flags | 256-bit loads, fast math, ptxas register use, fragment layout choice | open |

## Tier 3: skeletons

| Decision | Domain | Mechanism | Effect measured | Solver |
|---|---|---|---|---|
| skeleton | TMA + producer group, cp.async pipelined, split-K atomic, stream-K, GEMV split-K, fp4 fragment | who moves the data, how K is divided, how C is reduced | 1024³: cp.async staged 30.5 µs vs the TMA kernels' 41-47 | path and split-K in the pipeline |
| split_k | 1, 2, 4, 8 | more blocks over K; C added with fp32 atomics (8 lines per warp instruction, chip 225 B/clk; cold: each C line read first) | split-K decode within 5 % in the simulator | in the pipeline |
| stream-K programs | 48, 96, 144 | persistent blocks share the last wave | within 10 % of cuBLASLt (old timing) | example does not compile at the pinned commit |
| GEMV rows × lanes | BLOCK_N 2-8, reduce_threads 32-64 | lines per load, the shared-memory CAS reduction | 8.3-12.4 µs at N = 2048 | simulator only |

## Tier 4: hardware features beyond TileLang

| Decision | Binder | Mechanism | Evidence | Effect |
|---|---|---|---|---|
| launch mode: stream / CUDA graph | runtime | launch cost per kernel | `launch.cu` | 2.6 → 0.43 µs per kernel |
| L2 pinning per tensor (≤ 18 MB in all) | runtime: persisting L2 set-aside + access-policy window | the L2 keeps a tensor across other traffic | `tier4_query.cu`, `l2persist.cu` | 8 MB of weights after 64 MB of other traffic: 37.7 → 8.0 µs; 4-16 MB: 3.4-5x |
| thread-block clusters with TMA multicast | our CUDA | one L2 read feeds up to 8 SMs | clusters up to 8 and distributed shared memory work (`tier4_query.cu`) | not measured |
| TMA producer issue pattern | our CUDA | bytes in flight on the TMA path against DRAM latency under load | `dramlat.cu`, decode sweep | open |
| prefetch the next kernel's weights into L2 | our CUDA (cp.async.bulk.prefetch.L2) | later DRAM reads moved under this kernel's compute | — | not measured |
| two kernels at once (streams, priorities) | runtime | SMs, L2 and DRAM shared by two kernels | `concurrency.py` | decode GEMM + 1024³ GEMM: 180 µs serial, 192 together (worse) |
| CPU memory traffic during GPU phases | runtime / scheduler | CPU and GPU share the LPDDR5X | `cpuhog.c` | GPU DRAM 237 → 138 GB/s with 20 CPU threads copying |
| mma input format | precision decision | tensor work per 16-cycle instruction | `mma_types.cu` | fp8 / int8 / fp4 (f8f6f4) 2x fp16; tf32 0.5x |

L2 pinning is the clearest case for the MILP across kernels: 18 MB to share among a decode step's tensors is a
knapsack, and the solver's resource budgets (`agentic/decide/mip.py`, `caps`) are built for it.

## The solver path

`agentic/decide/pipeline.py`: legal columns of the tied decisions → the analytical model plus measured terms
(`agentic/decide/price.py`) → the MILP's top 20 (`agentic/decide/mip.py`, checked against brute force in
`tests/test_decide.py`) → expanded with the path-dependent knobs → compiled and checked on the Spark → re-ranked by the
simulator → the union of both top lists timed against cuBLASLt in one session.

## First end-to-end runs (Oct 20 2026)

`python -m agentic.decide.pipeline M N K [--cold]`: the analytical model prices every legal configuration, the MILP's
top 30 are expanded with the path-dependent knobs, compiled and checked on the Spark, re-ranked by the simulator, and
the finalists are timed against cuBLASLt in the same session.

| Shape | Configurations priced | Compiled, all correct | Best found | cuBLASLt | Total time |
|---|---:|---:|---|---:|---:|
| 1024³, warm | 1078 | 96 | 30.5 µs: cp.async 64x128x32, 2 stages, 256 threads, full-column warps, staged C | 34.8 µs (found kernel 12.4 % faster) | 263 s |
| 128×4096×4096, cold | 946 | 72 | 150.5 µs: cp.async 128x256x32, 4 stages, 256 threads, staged C | 151.4 µs (level; both near the 146 µs DRAM floor) | 168 s |

What the first runs taught:
- **A register bug hid the best kernel.** The pipeline first passed the cp.async register budget (255) as the register
  count. That made the model think one block fits per SM. The kernel that measured 30.5 µs was priced at 50 µs and
  ranked 299th. With a register estimate from compiled kernels (accumulators + 70) it ranked 26th and was found.
- **The simulator ranks the cp.async region well at 1024³**, but misses the prefetch penalty at M = 128: it scored
  the prefetch kernels at 151.7 µs, and they measured 164.
- **At M = 128 the finalists were all cp.async.** The cover set's TMA 128x128x32 with 4 stages and the staged store
  measured 148.4 µs cold, slightly better. Both models prefer cp.async there, which is a known gap.

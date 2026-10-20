# TileLang controls for GEMM-like kernels (pinned commit fe6d319c, source survey Oct 20 2026)

Read-only survey of the TileLang source (no changes made). Paths are relative to the TileLang tree. "Enum" says
whether TileLang's own AutoTuner or carver sweeps the control.

**The AutoTuner** (`tilelang/autotuner/tuner.py:1347`) has no search space of its own. It sweeps the list or callable
of configs the user writes: any kernel parameter, plus a reserved per-config `pass_configs` key (`tuner.py:52, 932-944`).
`compile_flags` and `target` cannot vary per config.

**The carver** (`carver/template/matmul.py:40-52`, `carver/utils.py:29-55`, TensorCorePolicy):
- It enumerates the block tile, the warp tile (which gives threads), `rstep` (block_K) and per-tensor vector width.
- It fixes `pipeline_stage`, `use_async` and the rasterization per architecture (`roller/policy/tensorcore.py:31-48, 342-364`).
- On sm_100/sm_120 it sets no pipelining: `pipeline_stage` 1 maps to `num_stages=0` (`example_gemm_autotune.py:74`).
- It knows nothing of WGMMA, tcgen05, TMA, warp specialization or clusters.
- In `example_gemm_autotune.py:76` the rasterization test compares an instance to a class, so it is always true.

## (A) Tile / schedule shape
- **Block tile** (`block_M`, `block_N`, `block_K`): set the tile shapes, the mma count, shared memory and TMA box sizes. The carver sweeps them.
- **`T.Kernel(threads=)`** (`tilelang/cuda/language/kernel.py:15`): warp count. Auto warp specialization adds a 128-thread producer group (`src/cuda/transform/producer_consumer_ws.cc:1189`). The carver sweeps it through the warp tiles.
- **`T.gemm(policy=)`** (`tilelang/tileop/base.py:5-12`; `src/cuda/op/gemm.cc:447-460`): Square, FullRow or FullCol. Sets the warp split of the C tile.
- **`T.gemm(transpose_A/B)`** (`cuda/language/gemm_op.py:36-37`): operand majorness, `ldmatrix.trans`, instruction eligibility.
- **`T.gemm(clear_accum=)`** (`gemm_op.py:39`; `src/op/gemm.cc:142-160`): zero-initializes C inside the gemm.
- **Accumulator scope** (`allocate.py:81, 254`; `src/cuda/op/gemm.cc:393-445`): on sm_120 every `T.gemm` lowers to `mma.sync`. WGMMA requires Hopper (`gemm.cc:94-96`); tcgen05 requires sm_100-110 with C in tensor memory (`target_utils.cc:71-76`).
- **`T.gemm_blockscaled` / `T.mma_gemm_blockscaled(sf_layout=)`** (`gemm_op.py:84, 398`): on sm_120, `mma.sync.m16n8k64.kind::mxf4nvf4.block_scale` (`src/tl_templates/cuda/instruction/mma_block_scale.h:44-103`).
- **Split-K and stream-K** have no primitive. Split-K is a third grid dimension plus atomics; stream-K is a 1-D grid with a manual loop. The carver defines `Hint.split_k_factor` (`carver/roller/hint.py:180`) but never sets it.

## (B) Data movement
- **Default copy choice** (`src/cuda/op/copy_analysis.cc:716-890`): without warp specialization, loads become cp.async in a pipelined loop (TMA only for shared-to-global stores); with it, loads become TMA. LDSM/STSM handle shared to and from fragments.
- **`T.copy(disable_tma=, prefer_instruction=)`** (`cuda/language/copy_op.py:24-26`; `copy_analysis.cc:103-131`): tma / cp_async / sync; fails if the choice cannot be honored.
- **`eviction_policy`** (`language/copy_op.py:56`; also on `copy_cluster`, `tma_copy`, gather/scatter): evict_normal / first / last, sent as `.L2::cache_hint` (`copy_sm90.h:11-15`). Only TMA and bulk copies honor it (`src/cuda/op/copy.cc:2222, 2465, 2586, 2762`).
- **L2 promotion on TMA descriptors**: hardcoded to `CU_TENSOR_MAP_L2_PROMOTION_L2_128B` (`copy.cc:2173, 2424, 2680`; `atomic_add.cc:402`). Not user-controllable.
- **`coalesced_width`, `loop_layout`, `T.Parallel(prefer_async=)`, `T.async_copy`** (`copy_op.py:63-65, 188`; `loop.py:15-22`): SIMT copy width and mapping, explicit cp.async.
- **`T.tma_copy(barrier, cluster_mask, ...)`** (`copy_op.py:232-324`): split-phase TMA. `cluster_mask` makes it a multicast.
- **`T.copy_cluster(dst_block, cluster_mask, remote_barrier)`** (`copy_op.py:133`; `copy.cc:2238-2250`): TMA multicast or SM-to-SM distributed shared memory copy. Gated on arch ≥ 90 only.
- **`T.annotate_layout`** with the swizzle helpers (`annotations.py:45`; `layout/swizzle.py:66-129`): shared-memory swizzle, TMA swizzle mode, bank conflicts.
- **`T.annotate_l2_hit_ratio({buf: r})`** (`annotations.py:77`; `src/cuda/transform/lower_l2_persistent_annotation.cc:21-54`): the host sets a stream access-policy window and the persisting L2 size.
- **`-DTL_ENABLE_L2_PREFETCH=1`**, a compile flag (`src/tl_templates/cuda/copy.h:26-86`): `.L2::128B` prefetch on cp.async and predicated loads.
- **Pass configs**:
  - `tl.disable_tma_lower` (deprecated; the warp-specialization classifier ignores it, `copy_analysis.cc:880`)
  - `tl.enable_async_copy`
  - `tl.disable_vectorize_256` (256-bit loads are on by default for arch ≥ 100, including sm_120)
  - `tl.enable_lower_ldgstg`
  - `tl.disable_safe_memory_legalize`
  - `tl.layout_cost_model`
  - `tl.config_index_bitwidth`
  - shared-memory merge and reuse keys (`pass_config.py:114-346`)

## (C) Pipeline and synchronization
- **`T.Pipelined(num_stages, order, stage)`** (`loop.py:112-117`; `src/ir.cc:141-146`): one or more stages enable warp specialization when TMA is eligible (`producer_consumer_ws.cc:2479-2486`). `order` and `stage` set a manual schedule with depth max(stage) + 1.
- **`T.unroll(explicit, unroll_factor)`** (`cuda/language/loop.py:57-63`) and the `tl.UnrollLoop` thresholds (`src/transform/unroll_loop.cc`).
- **Barrier, fence and mbarrier primitives** (`allocate.py:194, 227`; `builtin.py:346-1413`; `cuda/language/cluster.py:27-44`). The WGMMA and tcgen05 sets do not apply to sm_120.
- **Pass configs**: `tl.disable_thread_storage_sync`, `tl.disable_shuffle_elect`, loop unswitching keys.

## (D) Warp specialization and registers
- **Auto producer/consumer split**: on when arch ≥ 90, `num_stages` ≥ 1 and a copy is TMA-eligible (`producer_consumer_ws.cc:2544-2575`). Runs on sm_120.
- **`tl.disable_warp_specialized`** (`pass_config.py:73`); `tl.enable_auto_warp_specialization="role_based"`.
- **`T.ws`, `T.annotate_ws_schedule`, `T.annotate_ws_pipeline_depth`, `T.ws_op`** (`warpgroup.py:19, 59`; `annotations.py:136-167`): manual warp roles and schedules.
- **setmaxnreg**: defaults are 24 for the producer and 240 for consumers, auto-adjusted (`annotate_warp_group_reg_alloc.cc:248-313`). Overrides: `T.annotate_producer_reg_dealloc(n)` and `T.annotate_consumer_reg_alloc(n)` (24-240), plus `set/inc/dec_max_nreg` and `no_set_max_nreg` (`builtin.py:389-440`). It needs an "a" target such as sm_121a.
- **`T.annotate_min_blocks_per_sm(n)`** (`annotations.py:86`): the second argument of `__launch_bounds__`.

## (E) Epilogue / output
- **Direct vs staged C store** (`copy_analysis.cc:798-805`; `copy.cc:2286-2296`): staging through `C_shared` gives STSM plus a TMA store with `wait_group.read 0`.
- **`T.atomic_add(memory_order, return_prev, use_tma)`** (`cuda/language/atomic.py:12-17`; `atomic_add.cc:36-95`): `use_tma` is a bulk reduce-add from shared memory; tiles vectorize to x2/x4.
- **`T.reduce_*`** (`batch`, `nan_propagate`), the fp32x2 reduction (on by default for arch ≥ 100), and `T.alloc_reducer`.

## (F) Launch / grid
- **`T.Kernel(cluster_dims=)` / `T.ClusterKernel`** (`cuda/language/kernel.py:17`): cluster launch through `cudaLaunchKernelEx`.
- **`T.use_swizzle(panel_size, order="row"|"column")`** (`annotations.py:30-42`; `threadblock_swizzle.h:11-46`).
- **`T.Persistent`, `T.PersistentTileScheduler`** (`loop.py:82-109`; `tile_schedule.py:93-290`): persistent tile loops with swizzle and cluster grouping.
- **`T.sync_grid`** (cooperative launch).
- **`T.pdl_trigger` / `T.pdl_sync`** (`cuda/language/pdl.py:10-17`): programmatic dependent launch.
- **Cluster launch control `T.clc_*`**: Blackwell.
- **`T.Kernel(prelude=)`, `T.import_source`, `T.annotate_restrict_buffers`**: raw CUDA injection.

## (G) Compile options
- **`target`** (`jit/__init__.py:594`): auto-detect gives `sm_XXa`.
- **`execution_backend`**: tvm_ffi / cython / nvrtc / cutedsl / ...
- **`compile_flags`** (the only route to `-maxrregcount` and `-DTL_ENABLE_L2_PREFETCH`) and `pass_configs`.
- **`tl.enable_fast_math`, `tl.ptxas_register_usage_level` (0-10), `tl.device_compile_flags`**.
- **Fixed flags**: `-O3 -lineinfo -std=c++20`.
- **No effect on code**: several TIR keys have no reader at this commit, and the diagnostic-only keys.

## Missing, a no-op, or a caveat on sm_120/121
1. **Hopper-only**: WGMMA and its knobs, and TMA im2col. **sm_100-only**: tcgen05, TMEM, 2-CTA MMA, gather4/scatter4.
2. **Every `T.gemm` is `mma.sync`.** The one sm_120-specific path is block-scaled nvf4.
3. **The carver** turns pipelining off and knows no TMA, warp specialization or clusters.
4. **setmaxnreg** needs the `a` target.
5. **Checked only as arch ≥ 90/100**: clusters and multicast, 256-bit loads, fp32x2 reduction, stmatrix m16n8. Hardware support has to be confirmed (here: clusters and distributed shared memory work, `calibration/spark-6951/tier4_query.txt`).
6. **Warp specialization, TMA, mbarriers and PDL** do apply on sm_120.

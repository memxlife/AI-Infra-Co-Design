# TileLang on the GB10: what the first round found

*Round 1 of the TileLang brief (Oct 2026). Family `gemm_basic` (TileLang's `examples/gemm/example_gemm.py` with six
integer holes), FP16 in, FP32 accumulate, on spark-6951. Pinned versions: `agentic/tilelang/PINNED.md`. Tables:
`runs/tilelang/sweep_*.csv`. Terms: `calibration/spark-6951/tilelang_terms.yaml`.*

## Where TileLang differs from the brief (file and line at the pinned commit fe6d319c)

| the brief says | what TileLang does | where |
|---|---|---|
| `kernel.profile` returns `n_regs`, `n_spills` from the JIT kernel | `JITKernel.resource_usage` is "HIP only now"; on CUDA `n_regs`, `n_spills`, `n_max_threads` are None | `tilelang/jit/kernel.py:759-761` |
| time with the profiler's `cupti` backend | `do_bench(backend="cupti")` returns 0.0 ms on this host; the `event` backend (CUDA events, median) works | `tilelang/profiler/__init__.py:251` lists the backends; measured here |
| `T.Kernel(threads=T)` runs T threads | on `sm_121a` TileLang's warp specialization adds one producer warp group of 128 threads: `__launch_bounds__(T + 128, 1)`; 128 asks launch 256, 256 launch 384, 512 launch 640 | `tilelang/transform/__init__.py:201` (`DEFAULT_SIMT_THREADS = 128`), the generated CUDA |
| registers per thread follow the tile and the thread count | ptxas fills the launch-bounds budget: 65536 / (T + 128) rounded down to 8, so 240, 168, 96. A 256 x 256 tile at 512 threads needs 128 accumulators and spills (1.6–2 KB per thread, 5x slower) | the sweep, `runs/tilelang/sweep_1024.csv`; the rule is in `agentic/tilelang/tools.py` legalize |
| the `gemm_ws` family (`tcgen05`) for Blackwell tensor cores | `wgmma`, `tcgen05`, `setmaxnreg` are rejected by ptxas on `sm_121` (the GB10 book, chapter 15); the family cannot build. The plain family already lowers to TMA + `mma` with a producer warp group | `notes/gb10_microarchitecture.md` |
| `dram__bytes` from ncu for B_HBM | no `dram__*` metrics on this chip; `lts__t_sectors_op_read_lookup_miss.sum` x 32 is the DRAM read estimate | the GB10 book, `calibration/spark-6951/coverage.md` |
| the copies are `cp.async` or TMA by a pass config | on `sm_121a` every configuration of the family lowered to TMA (`CUtensorMap`); none used `cp.async` | the sweep's `uses_tma` column |
| the transform passes "accept or reject a knob assignment" (shared memory among the rules) | a point whose pipeline exceeds the 99 KB a block may have compiles fine and fails at launch: "Failed to set the allowed dynamic shared memory size to 122880". 12 of the 144 configs in TileLang's own autotune list fail this way at 1024^3; the Mac's legalize removes them before compiling | `runs/tilelang/baseline_1024.json`; the attribute is set by the adapter at launch |
| the carver's chip model as the first draft of the GB10 chip cross-check | on `sm_121a` `CUDA("cuda")` reports `smem_cap` 49152 (the 48 KB default, not the 99 KB opt-in), `sm_partition` 4, `compute_max_core` 48; its hints carry no warp split (`hint.warp == []`), `thread` holds the block's thread grid, every hint has `pipeline_stage` 1 and no rasterization | `tilelang/carver/arch/cuda.py`; `spark_baseline.py` |
| the dynamic shared-memory size is in the host source | the host code has no number; it is the device function's `dyn_shared_memory_buf` attribute, present only when the kernel was compiled in this process (a cache hit has no artifact) | `tilelang/jit/adapter/wrapper.py:488`; `spark_runner.dynamic_smem` |

## The loop, live

Claude Code is the Formulator: it reads `knobs.inventory` and the shape, writes the family, the menus to search and the
regime it expects (`tests/fixtures/tilelang_model_replies.json`, one entry per shape, written in the session). The MIP
(`agentic/tilelang/solve.py`, HiGHS, ~20 ms) picks the six numbers from the calibrated time model. `template.render`
fills the holes, TileLang compiles the kernel on the Spark, the kernel runs once against torch, and the time lands in
the ledger beside the prediction. One compile and one run per shape. TileLang's own autotuner compiles its list of 144
configurations and times each (12 of them fail at launch). The numbers, per shape, are in `runs/tilelang/report.md`
(written by `agentic/tilelang/report.py`): the three numbers of the brief for the sweep shapes, and the live loop
against the autotuner, the carver and cuBLAS for all eight shapes.

Two measurement notes. The same configuration measures 5 % apart between runs minutes apart (63.4 µs in the sweep,
60.2 µs in the autotuner's run, 62.5 µs in the loop, at 1024^3): differences under 5 % are noise on this box.
`torch.matmul` in FP16 (cuBLAS through torch 2.11 cu128) is 3 to 8x slower than the TileLang kernels here, checked
three ways (`do_bench`, an `out=` variant, a plain CUDA-event loop); it looks like torch's cuBLAS has no tuned kernels
for `sm_121` rather than a timing artifact, but that is a guess until someone profiles it.

## Cliffs the constraints must know

1. **Registers under launch bounds.** The accumulators plus about 24 registers must fit `floor(65536 / (threads + 128))`
   rounded down to 8. Over it: spills, 5x. 8 of 512 compiled points at 1024^3; the rule now removes exactly those 8.
2. **The traversal panel at 4096^3.** With A and B over L2 (64 MB), the default N-fastest order re-reads B once per
   row of tiles: 6.1 ms best. Panels of 8 or 16 (`use_swizzle`) share B across the panel's rows: 1.73 ms best, 79.5
   TFLOP/s, 86 % of the measured tensor peak. J-hat's traffic term models the panel (`cost.traffic_bytes`).
3. **One block per SM, always.** Because of the register budget above, no configuration of this family runs two
   blocks on an SM; occupancy is the thread count, not the block count. The per-tile model is fitted by thread count.

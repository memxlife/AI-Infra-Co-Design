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
| the dynamic shared-memory size is in the host source | the host code has no number; it is the device function's `dyn_shared_memory_buf` attribute, present only when the kernel was compiled in this process (a cache hit has no artifact) | `tilelang/jit/adapter/wrapper.py:488`; `spark_runner.dynamic_smem` |

## The three numbers of the brief, per shape

See `runs/tilelang/report.md` (written by `agentic/tilelang/report.py`) for the table.

## Cliffs the constraints must know

1. **Registers under launch bounds.** The accumulators plus about 24 registers must fit `floor(65536 / (threads + 128))`
   rounded down to 8. Over it: spills, 5x. 8 of 512 compiled points at 1024^3; the rule now removes exactly those 8.
2. **The traversal panel at 4096^3.** With A and B over L2 (64 MB), the default N-fastest order re-reads B once per
   row of tiles: 6.1 ms best. Panels of 8 or 16 (`use_swizzle`) share B across the panel's rows: 1.73 ms best, 79.5
   TFLOP/s, 86 % of the measured tensor peak. J-hat's traffic term models the panel (`cost.traffic_bytes`).
3. **One block per SM, always.** Because of the register budget above, no configuration of this family runs two
   blocks on an SM; occupancy is the thread count, not the block count. The per-tile model is fitted by thread count.

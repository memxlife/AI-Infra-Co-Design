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

## Round 2

### Step 1: R1, R2, R3

**R1, the baseline was a fallback kernel.** `torch.matmul` in FP16 (torch 2.11, cu128) measured 201 / 491 / 10807 µs at
1024³ / M=32 / 4096³: torch's cuBLAS has no `sm_121` kernels. The baseline is now cuBLASLt from the Spark's CUDA 13
toolkit (`agentic/tilelang/baselines/cublaslt_gemm.cu`: FP16 in, FP32 accumulate, row-major, the heuristic's
candidates each timed with CUDA events, best kept): **34.2 µs (62.8 TFLOP/s), 148.9 µs (7.2), 1456 µs (94.4)**. That
changes the story at the small and skinny shapes: cuBLASLt is 1.8x faster than the best kernel in the `gemm_basic`
family at 1024³ and 1.35x at M=32 (it sits at the DRAM floor, 143 µs); at 4096³ the family's best (1377 µs) is 5 %
faster than cuBLASLt. The torch row is removed from `report.md` and the console. 118 lines for the harness.

**R2, the scale is gone.** `scale: 1.0`, `overlap: max`. Every term is now a lower bound on its own: the tensor rate
is the best any configuration reached, per thread count, tile and K (a table of 65 keys from the three sweeps; a K
not in the table takes the best over the K's seen); the per-SM copy rate and the DRAM rate are the best observed;
the DRAM traffic charges A once and B once per panel less the fraction L2 keeps (an envelope under the four ncu
counts); the per-tile fixed, fill and per-step costs are zero (an envelope cannot carry them). The decode term the
brief asked for came out empty: the smallest residual per wave over the M=32 rows is 0 once the DRAM rate is the
best observed, so no start-up term is added (`t_wave_startup: 0`). One-sided test with no scale: **bound broken on
0.8 % / 1.0 % / 0.9 %** of the sweep points at 1024³ / 4096³ / M=32. Ranking: Spearman 0.78 / 0.85 / 0.10 (the M=32
sweep is 468 points within 30 % of each other, mostly ties for the model). What changed in the model when the scale
came out: the rate became a table over (threads, tile, K) instead of one number per thread count, and the fitted
per-tile costs that the scale had been cancelling disappeared.

**R3, the noise band.** One configuration, five timings in a row (`agentic/tilelang/spark_noise.py`), SM clock and
temperature recorded each time: **spread 1.4 % at 1024³ (62.5–63.4 µs), 0.3 % at 4096³ (1351.6–1355.8 µs)** with the
clock at 2405 MHz throughout, 41–42 °C. Pinning the clock (`nvidia-smi -lgc 2418,2418`, allowed without a password)
does not shrink it (1.3 % / 0.5 %) and shifts 4096³ up by 2 %, so the sweeps stay unpinned. The larger differences
seen between sessions (60.2 / 62.5 / 63.4 µs for one config) are between processes minutes apart, not within one. Every
`kernel_report` now carries the clock and temperature before and after timing. The ranking test is restated: the
solver's pick is inside the noise band of the sweep's best when its gap is under 1.4 % (1024³) or 0.5 % (4096³);
today the gaps are +1.7 % / +2.7 % / +2.7 %, so the picks are just outside the within-session band and inside the
between-session one. The gaps at 4096³ and M=32 are ties in the model broken the wrong way: the MILP returns one of
several points with the same bound.

### Step 2: opening Group A and Group B

Every name was checked against the TileLang source at fe6d319c (the file and line are in `knobs.yaml`, column
`binds`). Each knob was then varied alone around round 1's best at 1024³ (256x128x64, 2 stages, 256 threads) and at
M=32 (16x256x32, 5 stages, 256 threads), one compile and one timing per value (`runs/tilelang/probe/probe.jsonl`).
The base configuration itself was timed seven times in that run: mean 61.8 µs, spread 3.4 % at 1024³ (the timer
steps in whole microseconds there: 60.4 / 61.4 / 62.4 µs appear verbatim), 202.2 µs, spread 0.5 % at M=32. A knob
value inside that spread gets factor 1.0 in `tilelang_terms.yaml`.

- **PREFETCH** (`T.Pipelined(order=, stage=)`, `tilelang/language/loop.py:112-117`). Copies at stage 0, the multiply at
  stage P. TileLang infers the pipeline depth as max(stage)+1, so P needs P+1 buffers: at 256x128x64 a prefetch of 2 or
  3 asks for 147 or 196 KB of shared memory and fails at launch (the rule is now in legalize). P = 1 against the
  inferred schedule: −1.6 % at 1024³, −0.5 % at M=32, inside the spread. Factor 1.0.
- **WARP_POLICY** (`T.gemm(policy=GemmWarpPolicy)`, `gemm_op.py:160-166`, `tileop/base.py:5`: Square, FullRow, FullCol).
  −1.6 / 0 / −3.2 % at 1024³, within ±0.5 % at M=32. Inside the spread. Factor 1.0.
- **VEC_WIDTH** (`T.copy(coalesced_width=)`, `copy_op.py:59-63`). 4 / 8 / 16 elements: 0 / −3.3 / 0 % at 1024³, within
  ±0.5 % at M=32. Inside the spread (the copies are TMA; the width hint does not reach them), and 16 is silently
  clamped to 8 by the compiler ("Requested coalesced_width=16 is incompatible with the geometry-supported vector
  width 8; using 8", `src/op/parallel.cc:1049`), so the menu is really {4, 8}. Factor 1.0.
- **MAXRREG dropped.** `tilelang.jit(compile_flags=['-maxrregcount=N'])` is accepted (`tilelang/jit/__init__.py:104`) and
  stored on the kernel, but the compiled cubin keeps 168 registers at N = 128 (`cuobjdump --dump-resource-usage` on
  the cached `kernel.cubin`): the flag never reaches nvcc with the tvm_ffi backend at this commit. No way to move the
  register cliff from outside; removed from `knobs.yaml`.
- **TMA_LOWER** (`tilelang.cuda.language.copy(disable_tma=True)` per copy, `tilelang/cuda/language/copy_op.py:24`; the
  pass config `tl.disable_tma_lower` is deprecated, `pass_config.py:378`). Off: the copies become `cp.async`, TileLang
  drops the producer warp group (256 threads launched instead of 384) and ptxas uses 201 registers; −3.3 % at 1024³,
  +0.5 % at M=32. Inside the spread; factor 1.0. The register rule in legalize follows it (no +128 threads).
- **WARP_SPEC** (`pass_configs={'tl.disable_warp_specialized': True}`, `pass_config.py:73`). Off: no producer group and,
  as a consequence, no TMA either (the kernel falls to `cp.async`, 256 threads, 201 registers); +3.2 % at 1024³ (factor
  1.043), +0.5 % at M=32. So the T+128 register cliff of round 1 can be removed, at a small cost on the big shape.
- **SMEM_SWIZZLE** (`T.annotate_layout({A_s: make_swizzled_layout(A_s), ...})`, `annotations.py:45`,
  `layout/swizzle.py:66,129`). The explicit swizzled layout equals the inferred one (+1.3 %, inside the spread). The
  linear (row-major) layout first looked like a cliff (+36 % at 1024³) and turned out to be a **wrong answer**: with
  TMA copies on, the kernel computes garbage (max error 43 at 1024³, 25 at M=32; 31 of 35 such points in the step-3
  sample). TMA writes the tile in its own swizzled order and the explicit row-major layout reads it as if it were not.
  With the `cp.async` path (TMA off, or the producer group off) the linear layout is correct and slower (the sample's
  medians: 109 µs against 74 µs at 1024³). legalize and the MILP now forbid linear + TMA; the factor is withdrawn.
  Correctness, not time, is what this knob exposed, and the one-knob probe at step 2 missed it because the probe only
  printed the time; the step-3 sample caught it.

Net: of the seven candidate knobs, one does not work (MAXRREG), one is a correctness rule rather than a cost (linear
layout with TMA), one trades the register cliff for 3–4 % (WARP_SPEC), and four are inside the noise at these two
shapes. The full
space is 559,872 points; the constraints admit about 31–33 % of random draws (612 / 663 draws for 200 legal points).
The solver now has 279 binaries over 12 knobs and still answers in ~20 ms. Lines changed to expose the seven knobs
(`agentic/tilelang/`): the template 18, `knobs.yaml` 48, `tools.py` 41, `cost.py` 15, `solve.py` 14, the probe runner
78 — about 215 lines, of which the family-specific part (template and knobs.yaml) is 66.


### Step 3: the solver over the full space against 200 random legal points

Full space 559,872 points per shape; 200 random legal points each, every one compiled and run (none failed to
build; 31 computed wrong answers, all of them linear layout + TMA, see step 2). At **1024³** the sample's best is
61.5 µs (64x256x32, 4 stages, 256 threads, panel 8, cp.async, no producer group); the solver's one compile over the
full space, 275 binaries in 20 ms, gives 62.5 µs, **+1.6 %**, and it used none of the new knobs: every one at
TileLang's default. Round 1's pick re-timed in the same session: 64.5 µs, so the round-2 pick is 2.0 µs (3.2 %)
better, which is one or two timer ticks. At **M=32** the sample's best is 203.8 µs, the solver's pick 206.8 µs,
**+1.4 %**, again all new knobs at their defaults; round 1's pick re-timed 203.7 µs (+1.5 % for round 2's, inside
the 0.5 % band's noise once the tie is counted). The bigger space found nothing better than the six-knob space at
these two shapes: the new knobs move the time by less than the timer's spread, and the model says so (factors 1.0).
Model on the sample: Spearman 0.67 at 1024³, −0.11 at M=32 (182 points within 10 % of each other), bound broken on
30 % / 28 %: the envelopes were fitted on the six-knob sweeps and the new knobs' combinations (cp.async paths, other
thread counts' register budgets) sit below them. The honest summary for the paper: on this family and chip, the
six knobs TileLang's own autotuner sees are the ones that matter; the extra seven cost 215 lines to expose, and
what they bought was one correctness rule and one register-cliff escape, not a faster kernel.

### Step 4: R4 and R5

**R4.** `report.md` now carries the two autotuner claims under separate headings and no merged percentage: the menu
claim (TileLang's 144-config list has no traversal panel; 1941.5 µs after 132 compiles against our menu's best of
1377.4 µs at 4096³) and the solver claim (over our own menu, 1377.4 µs exhaustive against 1379.4 µs from one compile
with the round-1 terms, 1414.1 µs with the round-2 envelopes; 63.4 vs 64.4 µs at 1024³; 200.4 vs 202.7 µs at M=32).

**R5, out of sample.** Terms fitted on 1024³, 4096³ and M=32 only; four new shapes, the solver's pick over the full
space plus 30 random legal points each (`runs/tilelang/oos.md`):

| shape | pick measured | sample best of 30 | pick vs sample best | predicted | bound broken on the sample |
|---|---|---|---|---|---|
| 2048³ | 242.7 µs | 257.0 µs | −5.6 % | 189.6 µs | 0 % |
| 8192³ | 11539 µs | 13231 µs | −12.8 % | 10755 µs | 50 % |
| 4096×4096×1024 | 461.8 µs | 471.0 µs | −2.0 % | 676.8 µs | 71 % |
| 128×4096×4096 | 258.9 µs | 207.9 µs | **+24.6 %** | 209.6 µs | 14 % |

The pick beats thirty random legal points on three of the four shapes without a single extra compile. It loses at
M=128 by a quarter: the pick pads M to 256 rows (one row of 64 blocks) and streams B at 130 GB/s, while the sample's
best (64x64 tiles, 128 blocks) reaches 165 GB/s; the model's DRAM rate is the 235 GB/s the big square reaches, and it
has no term for the fewer, bigger blocks of a skinny shape pulling less. The bound breaks on the two shapes whose K
is not 1024 or 4096 in the way the table expects: at K=1024 with 512 tiles (11 waves) the rate table's entry comes
from the 1024³ sweep, where that tile ran one wave with its start-up unamortized, so the model is 47 % too slow;
at 8192³ the "best over the K's seen" fallback is 7 % too slow for half the sample. The envelopes are bounds on the
shapes they were fitted on and not far outside them; a per-wave start-up term and a per-block streaming cap are the
two terms R5 says are missing. Each sample had 2 wrong answers: linear layout + TMA again (the sample was drawn
before step 3's rule).

### Step 5: the split-K family, the live Formulator call, and R7

**R6, refuted.** Family `gemm_splitk` (TileLang's `examples/gemm_splitk` with the six holes plus `split_k`; the grid's
z axis over parts of K, FP32 atomics into a zeroed C; exact-once as a legalize rule, `K mod split_k == 0` and
`(K / split_k) mod block_K == 0`, and as a contract obligation). The Formulator (Claude Code, live in the session)
read the probe at M=32, 32 tiles for 48 SMs, one block per SM, and chose `gemm_splitk` with split_k in {2, 4, 8},
predicting 150–180 µs. Every split-K kernel is correct (exact-once holds; max error 0.09–0.125 against torch, the
same as `gemm_basic`), and every one is **slower**: the best, 32x128 tiles with 4 stages and split 2 or 4, takes
208.8 µs against 203.7 µs for the `gemm_basic` base, +2.5 %, outside the 0.5 % band in the wrong direction; split 8
with 256 blocks is +3.5 %, split 2 with 16x256 tiles +23 %. The model had the order right (200 / 211 / 223 / 248 µs
for split 1 / 2 / 4 / 8). So at M=32 the limit is not the block count: 32 blocks already pull 160 GB/s and 256 blocks
pull no more, against cuBLASLt's 215 GB/s. The decode shape's remaining 40 % is in how B is streamed (the TMA box
geometry or the per-SM copy path), not in how many blocks stream it; that is the next structural choice to open,
and it is not split-K. The loop recorded all of it: run `tilelang-splitk-m32` in `runs/gemm-tilelang.db` has the
proposal choosing the family, the family check with exact-once, the solver's pick (16x256, 3 stages, split 8,
predicted 247.7 µs), the kernel correct at 205.8 µs, and then R7 firing live (below).

**R7, the loop reacts once.** Mechanism: when a kernel runs faster than the model's bound at the solver's point, the
orchestrator names the longest term, re-fits that one term from the measurement (the ncu DRAM bytes when the
report has them, otherwise the rate the measurement implies), derives a contract that carries the new terms, solves
again and runs the new pick (`agentic/tilelang/recalibrate.py`, `Orchestrator._recalibrate_and_resolve`). Replay
without the GPU (`tests/test_tilelang_r7.py`, from the 4096³ sweep): the brief's injected error, the effective L2
halved, cannot show it, because every panel pick at 4096³ is tensor-bound in the model and a wrong L2 term changes
neither the pick nor its prediction; a DRAM rate at a quarter of its value does: the first pick is predicted 2346 µs
and measured 2222 µs, `diagnose(cost_model)` fires, B_HBM moves from 43 to 46 GB/s (what that one measurement
proves), the re-solve changes the pick, the new pick measures 2063 µs. The ledger shows measured → diagnosis →
recalibration → new pick. It also fired live in the split-K run: predicted 247.7 µs, measured 205.8 µs, the per-SM
rate entry for that tile re-fitted, the re-solve kept the same pick (the menus the Formulator left had no better
point), the second run measured 207.9 µs. Round 3 note: under the round-3 envelopes a single measurement moves a
shared rate only to what it proves (B_HBM/16 injected: 15 to 37 GB/s), and the least-traffic point stays the pick; the
replay test now asserts the chain (measured, diagnosis, recalibration, re-solve, re-run) and reports whether the pick
moved rather than requiring it.

## Round 3

### The linear-layout + TMA wrong-answer bug (filed upstream)

`T.annotate_layout({A_s: make_linear_layout(A_s), B_s: make_linear_layout(B_s)})` on the GEMM operands computes wrong
results when the copies are lowered to TMA, which is TileLang's default on `sm_121a` at fe6d319c. Evidence: the step-2
one-knob probe (max error 42.7 at 1024³, 24.5 at M=32, against 0.06 / 0.125 for the inferred layout) and the step-3
random sample (31 of 35 points with linear layout + TMA + the producer group wrong; 0 of 96 linear points on the
`cp.async` path wrong). `make_swizzled_layout` and the inferred layout are correct with TMA. The likely cause: the TMA
tensor map is built with the swizzle the inferred layout assumes, and the explicit row-major layout reads the tile as
if it were not swizzled. Minimal repro: `agentic/tilelang/repro/linear_layout_tma_wrong.py` (six variants: three
layouts x TMA on/off). What we did about it here: legalize and the MILP forbid linear + TMA (step 3), and the
round-2 factor for the linear layout, which came from a wrong kernel, was withdrawn. What the compiler should do:
honour the layout in the tensor map, or refuse the combination at lowering time rather than run and be wrong. Filed upstream as tile-ai/tilelang#3571 (https://github.com/tile-ai/tilelang/issues/3571) with the repro, its
output and the generated CUDA of the failing point (`agentic/tilelang/repro/`). The lesson for the method: the
correctness check caught a compiler bug that a performance sweep alone would have recorded as a cliff.

### S1: cuBLASLt at every shape

Same timing on both sides (CUDA events, median after warm-up; cuBLASLt's fastest heuristic candidate, 3–7 per shape):

| shape | cuBLASLt | our pick | ours vs cuBLASLt |
|---|---|---|---|
| 1024³ | 34.2 µs (62.8 TF/s) | 62.5 µs | +83 % |
| 2048³ | 183.9 µs (93.4) | 242.7 µs | +32 % |
| 4096³ | 1456 µs (94.4) | 1434 µs (sweep best 1377) | −1.5 % (−5 %) |
| 8192³ | 11199 µs (98.2) | 11539 µs | +3 % |
| 4096×4096×1024 | 400.8 µs (85.7) | 461.8 µs | +15 % |
| 32×4096×4096 | 148.9 µs | 206.8 µs | +39 % |
| 128×4096×4096 | 155.4 µs | 258.9 µs (sample best 207.9) | +67 % (+34 %) |

The two big squares are within a few percent either way (4096³ inside the noise band in our favour; 8192³ 3 % behind,
outside the 0.5 % band but close): that part is closed. Everything smaller or skinnier loses, by 15 % at short K and
by 1.3–1.8x at 1024³, 2048³ and the decode shapes. cuBLASLt holds 85–98 TFLOP/s from 2048³ up and 62.8 at 1024³; our
family's best at 1024³ is 34 TFLOP/s and at 2048³ 70. The gap is the family, not the solver: at 2048³ the solver's pick
is the best of the 31 kernels we measured there, and at 1024³ it is 1.6 % from the sweep's best of 504. S2 is about
the family.

### Step 0: cuBLASLt under ncu, beside our pick

Same sections and metrics for both, one launch each (`runs/tilelang/ncu/`, `agentic/tilelang/ncu_table.py`). Durations
are ncu's (clock control on), so they sit above the event timings; the ratios hold.

| | 1024³ cuBLASLt | 1024³ ours | M=32 cuBLASLt | M=32 ours | 4096³ cuBLASLt | 4096³ ours |
|---|---|---|---|---|---|---|
| kernel | CUTLASS `cutlass_80_tensorop_f16_s16816gemm_relu_f16_256x128_32x3_nn_align8` | TileLang 256x128x32, 3 st | `nvjet_sm121_hsh_mma_192x24x32_7_64x24x32_tmaAB_bz_NNNN` | TileLang 64x256x64, 2 st | CUTLASS `..._128x256_32x3_nn_align8` | TileLang 256x128x64, 2 st, panel 4 |
| grid x block | 32 x 256 | 32 x 384 | 48 x 96 | 16 x 384 | 512 x 256 | 512 x 384 |
| registers | 218 | 168 | 255 | 168 | 228 | 168 |
| shared memory | 73.7 KB | 74.8 KB | 101.4 KB | 82.9 KB | 73.7 KB | 99.3 KB |
| blocks per SM (limit) | 1 | 1 | 2 (regs) / 1 (smem) | 1 | 1 | 1 |
| achieved occupancy | 16.6 % | 17.7 % | 6.2 % | 18.7 % | 16.7 % | 18.5 % |
| waves | 0.67 | 0.67 | 1.0 | 0.33 | 10.7 | 10.7 |
| duration (ncu) | 47.3 µs | 70.5 µs | 161 µs | 203 µs | 1700 µs | 1690 µs |
| tensor pipe active | 75.8 % | 68.7 % | 12.0 % | 42.2 % | 77.8 % | 82.2 % |
| DRAM bytes | 6.4 MB | 4.8 MB | 33.9 MB | 33.9 MB | 245 MB | 203 MB |
| L2 read bytes | 25.4 MB | 25.3 MB | 77.9 MB | 42.1 MB | 1611 MB | 1611 MB |
| L1 hit | 0 % | 83 % | 87.5 % | 87.5 % | 0 % | 85 % |
| top stall | fixed-latency execution, 4.2 cycles | fixed-latency execution, 4.8 cycles | scoreboard (memory), 16.2 cycles | all threads in the warp, 16.1 cycles | fixed-latency, 4.4 | fixed-latency, 5.7 |

What it says. **1024³:** cuBLASLt picked the same tile as our solver (256x128, K slice 32, 3 stages, 8 warps, 32
blocks, two-thirds of a wave), with the same L2 traffic, and runs it in two-thirds of the time. The difference is
inside the tile: a CUTLASS sm80-style kernel (`cp.async` multistage, every warp computes, 218 registers, L1 bypassed)
against TileLang's TMA + producer-warp-group kernel (384 threads, 168 registers). The tensor pipe is 76 % against 69 %
busy while active; the rest of the gap is time the pipe is not active at all: the start-up of each tile (tensor-map
and barrier setup, the producer hand-off) and its epilogue, which a 32-step K loop does not amortize. At 4096³ the
same two kernels tie (1700 vs 1690 µs), so the per-tile fixed cost is the whole 1024³ story. **M=32:** cuBLASLt uses a
decode kernel, 48 blocks of 96 threads, one per SM, 255 registers and 101 KB of shared memory each, 12 % tensor
activity, stalled on memory (scoreboard) as a streaming kernel should be; it moves the same 33.9 MB from DRAM in
161 µs against our 203. Our pick runs 16 blocks (64-row tiles over M=32: a third of the SMs), and the split-K probe
showed 256 blocks do no better. The design hint for S2 is not block count but bytes in flight per SM: cuBLASLt keeps
101 KB of B staged per SM with 3 warps, ours keeps 83 KB with 12 warps and a producer that issues one TMA box at a
time. **4096³:** tie; closed. Not done at 2048³ in step 0; the step-2 profiles cover it.

### Step 2: the model round (S3 widened)

The per-term breakdown of each solver pick against the counters (`runs/tilelang/ncu/`, all seven shapes):

| shape | measured | predicted | longest term | model DRAM MB / ncu | model L2 MB / ncu | tensor active | waves | top stall |
|---|---|---|---|---|---|---|---|---|
| 1024³ | 62 µs | 63 | tensor | 6 / 5 | 25 / 25 | 69 % | 1 | fixed latency |
| M=32 | 207 | 200 | DRAM | 34 / 34 | 38 / 42 | 42 % | 1 | sleeping (barrier) 14.8 |
| 4096³ | 1434 | 1379 | tensor | 151 / 203 | 1611 / 1611 | 82 % | 11 | fixed latency |
| 2048³ | 243 | 190 | tensor | 25 / 20 | 201 / 201 | 65 % | 3 | fixed latency |
| 8192³ | 11539 | 10755 | tensor | 755 / 2080 | 12885 / 12887 | 81 % | 43 | fixed latency |
| 4096×4096×1024 | 462 | **677** | tensor | 50 / 30 | 403 / 403 | 61 % | 11 | sleeping 7.4 |
| 128×4096×4096 | 259 | 210 | DRAM | 36 / 36 | 101 / 168 | 36 % | 2 | sleeping 13.8 |

The traffic terms agree with the counters where they matter (DRAM bytes within 20 % on the DRAM-bound shapes, L2
bytes exact except the padded M=128 tile). The two disagreements are in the per-SM term and in how the DRAM term is
spent, and they are the two mechanisms named here before coding:

1. **Per-launch prologue.** At 4096×4096×1024 the pick's tile (256x128, K=1024) is the same work as at 1024³, where the
   sweep measured it as one wave of 63 µs; here it runs 11 waves in 462 µs, 42 µs per wave. The single-wave sweep rows
   folded a once-per-launch cost (descriptor fetch, first-wave fill, last-wave drain, about 20 µs) into the tile
   rate. Mechanism: a kernel's time is a prologue paid once plus waves times the tile time; the prologue is the
   smallest value consistent with every row, the tile time comes from the many-wave rows.
2. **Per-wave DRAM with a per-block cap.** At M=128 the DRAM bytes match the model but stream at 139 GB/s against the
   235 GB/s the model charges; the kernel is 64 blocks on 48 SMs, so its second wave is 16 blocks streaming alone,
   and the top stall is warps sleeping on the TMA barrier. Mechanism: each wave streams its share of the bytes at
   min(chip rate, blocks in the wave times a per-block rate), with the per-block rate the best any block was seen to
   reach; a partial last wave therefore costs more than its share of bytes.

Rules kept: every term a minimum (or maximum rate) over its series, no least squares, no global scale; validation
leave-one-shape-out over the seven shapes.

**What was built.** Two designated experiments, 29 kernels in three minutes (`runs/tilelang/series.jsonl`): a K
series per tile at 4096² (K = 256 … 4096, five tiles) and a wave series at one tile and K = 1024 (M = N = 1024 … 8192,
1 to 43 waves). From them, as envelopes: the per-launch prologue, **21.6 µs** (the one-wave point against the
11- and 43-wave points: P = (w·t₁ − t_w)/(w − 1), the smallest over the pairs); the start-up a tile pays before its
pipeline is full, per tile class (2.0 / 6.8 / 8.9 / 1.7 / 1.7 µs for 128x128 / 256x128 / 128x256 / 64x128 / 64x64), charged
per wave; the tiles' asymptotic rates (1.4–2.4 TFLOP/s per SM). Mechanism 2 in the brief's form: a block streams at
bytes-in-flight / 1.82 µs, the latency being the smallest any streaming row allows under the per-wave formula; a
wave streams its bytes at min(chip rate, blocks × that). The sweep rates are then re-taken net of the prologue and
the start-up, rows that stream at over 60 % of the DRAM rate set only the DRAM terms, and the effective L2 is the
smallest value under which no row's DRAM term exceeds its time (39 MB). No least squares, no scale.

**In-sample (all seven shapes in the fit):** bound broken 0 / 2 / 0 / 0 / 20 / 15 / 0 % at 1024³ / 4096³ / M=32 /
2048³ / 8192³ / 4096×4096×1024 / M=128; the model's favourite among the measured rows is the best row on every
shape (gap ≤ 0.1 %). Before the mechanisms (round 2's terms) it was 0 / 1 / 1 / 0 / 20 / 20 / 0 % and the favourite was
97 % off at 2048³ and K=1024 (`runs/tilelang/step2_model.md` has the histograms side by side).

**Leave-one-shape-out, the solver's pick measured on the Spark** (`runs/tilelang/loso/`):

| shape | bound broken (LOSO) | LOSO pick | measured | predicted | best measured (sweep / sample) | pick gap |
|---|---|---|---|---|---|---|
| 1024x1024x1024 | 16% | 64x128x32, 5 st, 128 th, swz 0 | 70.7 µs | 54.9 µs | 61.5 µs | +15.0% |
| 4096x4096x4096 | 38% | 128x256x32, 4 st, 256 th, swz 0 | 2025.4 µs | 1359.7 µs | 1377.4 µs | +47.0% |
| 32x4096x4096 | 22% | 16x256x64, 2 st, 256 th, swz 0 | 202.8 µs | 168.4 µs | 200.4 µs | +1.2% |
| 2048x2048x2048 | 0% | 128x256x32, 4 st, 512 th, swz 0 | 249.8 µs | 219.2 µs | 242.7 µs | +2.9% |
| 8192x8192x8192 | 5% | 128x256x64, 2 st, 512 th, swz 4 | 11631.6 µs | 10087.2 µs | 11539.2 µs | +0.8% |
| 4096x4096x1024 | 65% | 128x256x32, 4 st, 512 th, swz 0 | 476.2 µs | 451.1 µs | 461.8 µs | +3.1% |
| 128x4096x4096 | 0% | 256x64x32, 2 st, 128 th, swz 0 | 258.0 µs | 175.1 µs | 207.9 µs | +24.1% |

Pass both criteria (bound ≤ 10 %, pick ≤ 10 %): **2048³ and 8192³**. Pass the pick only: M=32 (+1.2 %), K=1024
(+3.1 %). Fail both: 1024³, 4096³, M=128.

**Why the rest fail, and the stop.** Two mechanisms are in; the brief says stop here and report. The failures have one
shape each. 4096³ left out: the fit has no row where the traversal panel matters (the other six shapes are either
L2-resident or already panelled), so the effective-L2 envelope says panels are free, the solver picks panel 0, and the
kernel takes 47 % longer: a term the other shapes cannot teach. 1024³ left out: the single-wave rate for its tiles is
not in the fit and the prologue is charged with a tile rate from eleven-wave rows; the pick is a small-tile kernel
15 % off. M=128 left out: the padded 256-row tile is priced as its full-tile rate from 4096³; the M=32 rows do not
contain it (block_M ≤ 64 there), and the per-block cap does not bite at 64 blocks; +24 % again. K=1024 left out: the
bound breaks on 65 % of its rows because the K=1024 tile rates come from the single-wave 1024³ sweep even net of the
prologue, yet the pick is within 3 %. The pattern: the model transfers across shapes that share a regime and a K,
and not across a regime boundary; each failing fold is the one shape that alone holds its mechanism. That is the
finding, not a third term.

**Addition: chip-spec defaults for every term** (no GPU but the picks). Each term now has a value that stands in when
no fitting shape calibrates it: the traffic term is the microbenchmark's (effective L2 16.8 MB, B_HBM 235 GB/s, B
re-read once per panel of rows and nothing kept between panels: ncu's 770 MB at 4096³ panel 0 against 783 from
the count), never raised from the sweeps; a padded tile is priced at its tile class's rate; a (tile, K) the sweeps never
saw takes the series' start-up plus asymptotic rate; and a single-wave row's rate serves only single-wave predictions.
The "kept fraction of L2" form was withdrawn: it made the panel look free in the 4096³ fold. Leave-one-shape-out with
the defaults, picks run on the Spark (`runs/tilelang/loso/`):

| shape | bound broken (LOSO, defaults) | LOSO pick | measured | predicted | best measured (sweep / sample) | pick gap |
|---|---|---|---|---|---|---|
| 1024x1024x1024 | 16% | 64x128x64, 4 st, 256 th, swz 8 | 67.6 µs | 55.2 µs | 61.5 µs | +10.0% |
| 4096x4096x4096 | 38% | 128x256x32, 3 st, 256 th, swz 8 | 1437.5 µs | 1359.7 µs | 1377.4 µs | +4.4% |
| 32x4096x4096 | 26% | 16x256x64, 2 st, 128 th, swz 0 | 203.7 µs | 168.4 µs | 200.4 µs | +1.7% |
| 2048x2048x2048 | 0% | 128x256x32, 4 st, 512 th, swz 0 | 258.0 µs | 219.2 µs | 242.7 µs | +6.3% |
| 8192x8192x8192 | 25% | 128x256x32, 3 st, 512 th, swz 16 | 11953.2 µs | 10087.2 µs | 11539.2 µs | +3.6% |
| 4096x4096x1024 | 60% | 128x256x32, 4 st, 512 th, swz 0 | 456.7 µs | 431.0 µs | 461.8 µs | -1.1% |
| 128x4096x4096 | 10% | 256x64x32, 2 st, 128 th, swz 0 | 257.0 µs | 175.1 µs | 207.9 µs | +23.6% |

Pass both: ['2048']. The defaults fix what they were asked to fix: the 4096³ fold now picks a panel (8) and the pick
is 4.4 % off the best, and 1024³'s pick moves to a panelled 64x128 kernel 10 % off. They cost bound coverage: the plain
re-read count over-charges small-tile, low-panel rows at 4096³ and 8192³ where L2 does absorb part of the re-reads,
so those folds' bound-broken rates rise to 38 % and 25 %. M=128 stays at +24 %: the padded 256-row tile is priced
right as work, and the kernel is slow for a reason the model does not hold (64 blocks streaming 36 MB at 139 GB/s with
the warps asleep on the TMA barrier). K=1024's fold still breaks the bound on 60 %: its single-wave twin, 1024³, is
the only source of K=1024 tile rates once the series' asymptotic rate is a bound rather than a fit.

**Absolute error, for the record** (asked after step 2; all timings from disk, no GPU). Mean absolute percent error of
predicted against measured over every timed kernel at the shape, the mean sign, and the same on the solver's pick
alone. "Before" is round 2's terms (fitted on the three sweep shapes, so the last four rows are already out of
sample for it); "after" is round 3's, fitted on all seven and again with the shape held out. A pick marked "–" is one
the round-3 in-sample solver chose that was never timed.

| shape | timed | before: all kernels | before: pick | after, all seven: all kernels | pick | after, held out: all kernels | pick |
|---|---|---|---|---|---|---|---|
| 1024³ | 692 | 7 %, 6 % under | 0 % | 18 %, 14 % under | 26 % under | 20 %, 16 % under | 21 % under |
| 4096³ | 505 | 24 %, 10 % over | 11 % under | 11 %, 6 % under | 8 % under | 18 %, 2 % over | 7 % under |
| 32×4096×4096 | 651 | 4 %, 3 % under | 5 % under | 20 %, 15 % under | 20 % under | 20 %, 4 % under | 18 % under |
| 2048³ | 30 | 32 %, 32 % under | 22 % under | 16 %, 16 % under | – | 28 %, 28 % under | 15 % under |
| 8192³ | 30 | 30 %, 0 % over | 7 % under | 14 %, 9 % under | – | 23 %, 18 % under | 16 % under |
| 4096×4096×1024 | 30 | 51 %, 38 % over | 47 % over | 16 %, 16 % under | – | 27 %, 13 % under | 6 % under |
| 128×4096×4096 | 30 | 14 %, 1 % under | 19 % under | 22 %, 12 % under | 32 % under | 23 %, 16 % under | 32 % under |

Reading it: round 2's model was tight where it was fitted (4–7 % on the sweep shapes) and wrong in both directions out
of sample (38 % over at K=1024, 32 % under at 2048³). Round 3's is a looser bound everywhere (11–22 % in-sample, 18–28 %
held out) and under on every row but one: that is what making every term an envelope buys and costs. The pick's own
error is the number the solver acts on: 6–32 % under, held out, which is why the picks are right to within a few
percent on four shapes and wrong by 24 % at M=128, where the model under-predicts the most.

### Step 3 (S2): the third family for 1024³ and M=32, both refuted

The Formulator (Claude Code, live, from the step-0 table; replies in `tests/fixtures/tilelang_model_replies.json`)
asked for the two structures the counters pointed at, and the solver's sub-spaces were run in full (58 kernels,
`runs/tilelang/s2/`), every one correct.

**1024³: every warp computes, copies through `cp.async`, no producer group.** This is cuBLASLt's own structure here
(CUTLASS sm80, 256x128x32, 3 stages, 8 warps, 218 registers, 47 µs under ncu). TileLang builds the same shape of
kernel from `gemm_basic` with warp specialization off: 128x256x32, 3 stages, 256 threads, 202 registers, no TMA, no
producer. It runs in **61.5 µs**, the same as the TMA kernel (62.5), **+80 % against cuBLASLt's 34.2 µs**. Twenty-two
configurations over 3–5 stages and 128–256 threads, all between 61.5 and 75 µs. Same tile, same waves, same bytes,
same warp count, same register count, same copy instruction: what is left is the code inside the k-loop. TileLang's
pipeline on this path issues one `cp.async` group and one barrier per stage per operand and lowers `T.gemm` through
its `mma` template; CUTLASS's mainloop keeps more fragments in flight and overlaps the next stage's loads with the
current `mma`s at the instruction level. That is codegen, not a knob the family exposes, and not something the brief
lets us fork. **Finding: 1024³ cannot be matched from TileLang on this path; the 1.8x is in the generated
mainloop.** The gap closes with K: at 4096³ the two tie, because the per-tile cost that the mainloop pays is
amortized over 128 steps instead of 32.

**M=32: ~100 KB of B staged per SM with few warps, split-K so every SM has a block.** cuBLASLt's kernel: 48 blocks of
96 threads, 7 stages, 101 KB, split along K, 149 µs (161 under ncu). TileLang: 36 configurations at 60–101 KB staged,
3/5/7 stages, 64 or 128 threads, split 1/2/4, TMA on. Best **201.7 µs** (32x128x32, 7 stages, 64 threads, 70 KB),
**+35 % against cuBLASLt**; the whole set sits between 201 and 230 µs, exactly where the round-1 sweep and the
round-2 split-K probe sat. Staging depth, warp count and split-K each move the time by less than the noise band.
The counters say why: our streaming kernels stall with warps asleep on the TMA barrier, and the per-block streaming
rate is what the K-series and the per-block cap measured, bytes in flight over 1.8 µs. cuBLASLt's kernel streams
the same 34 MB with the same tile bytes in flight per block at a 35 % higher rate. The difference is in how the
copies are issued and waited for, which TileLang's TMA path fixes per copy. **Finding: M=32 cannot be matched from
TileLang's TMA path by any menu or structure we can name; the 35 % is in the copy pipeline's issue and wait.**

Both findings have the same shape as step 0's: the family picks the right tile, the compiler's mainloop and copy
pipeline set the per-tile and per-block rates, and those are the terms the model carries as envelopes rather than
explains. For the paper, that is the boundary of what a wrapped compiler can reach, measured.

### Step 4 (S4): knob importance, one figure

At each sweep shape's best point, one knob at a time over its menu, the spread of latency (max/min) from the points
the tables already hold (`runs/tilelang/figures/s4_knob_importance.png`, numbers in the `.json` beside it; the six
round-1 knobs from the sweeps, the round-2 knobs from step 2's one-knob curves at 1024³ and M=32).

| knob | 1024³ | 4096³ | M=32 |
|---|---|---|---|
| swizzle (traversal panel) | 0 % | **43 %** | 0 % |
| block_N | **15 %** | **35 %** | 0 % |
| block_M | **13 %** | **19 %** | 1 % |
| block_K | 3 % | 3 % | – |
| threads | 0 % | 2 % | 0 % |
| num_stages | – | – | 1 % |
| smem_layout (inferred vs swizzled) | 5 % | – | 0 % |
| tma, warp_spec, warp_policy, vec_width, prefetch | 2–3 % | – | 1 % |

A dash means the other menu values are illegal at that point (shared memory), so the tables hold no second point.
The space is low-dimensional and the dimension changes with the shape: at 4096³ two knobs (the panel and the tile's
column count) carry everything, at 1024³ only the tile shape moves the time and by a tenth, and at M=32 nothing in
the family moves it at all, which is S2's finding seen from the knob side. The seven round-2 knobs never exceed 5 %
on any shape.

### Step 4 (S5): the price of one wall, measured

The shared-memory cap the contract hands the solver, from 99 KB down in 16 KB steps; the solver re-run at 4096³ under
each cap, its pick compiled and run (`runs/tilelang/s5/`, figure `runs/tilelang/figures/s5_smem_wall.png`):

| cap | solver's pick | staged | measured | predicted |
|---|---|---|---|---|
| 99 KB | 128x256x32, 3 stages, 512 threads, panel 16 | 72 KB | 1573 µs | 1438 |
| 83 KB | 128x256x32, 3 stages, 512 threads, panel 8 | 72 KB | 1432 µs | 1438 |
| 67 KB | 128x256x32, 2 stages, 512 threads, panel 16 | 48 KB | 1587 µs | 1438 |
| 51 KB | 128x256x32, 2 stages, 512 threads, panel 16 | 48 KB | 1581 µs | 1438 |
| 35 KB | 128x128x32, 2 stages, 256 threads, panel 16 | 32 KB | 1819 µs | 1468 |
| 19 KB | 64x64x32, 2 stages, 128 threads, panel 16 | 16 KB | 3696 µs | 2264 |

The staircase has two steps and a flat. From 99 down to 51 KB the pick keeps its tile (128x256) and gives up one
pipeline stage at 67 KB; the measured time moves between 1432 and 1587 µs, and that spread is not the cap: it is the
model pricing panels 8 and 16 the same (both tensor-bound in J-hat at this shape) and breaking the tie by menu
order, while the chip prefers panel 8 by 10 %. The first real step is at 35 KB, where the tile shape changes to
128x128 (the pick no longer fits a 256-wide tile with two stages): +250 µs, 17 %. The second is at 19 KB, 64x64
tiles: +2.1 ms, 2.6x. So the shadow price of shared memory at 4096³ on this chip is zero above 48 KB, about
15 µs per KB between 48 and 32 KB, and about 115 µs per KB below that, as the family's tile menu falls away one
column at a time. The model's prediction tracks the steps' order and under-prices both (1468 and 2264 against 1819
and 3696), since the rate table never saw 128x128 or 64x64 at 4096³ with panel 16 under a stage shortage. The
10 % panel tie is the one thing this experiment adds to the model's to-do list: a traffic term that separates
panel 8 from 16 at 4096³ would need the L2's actual reuse across panels, which the microbenchmark default sets
to zero.

### S2 revised: three skeletons from the TileLang repo, run unchanged first

*Round 3, Oct 19 2026. Scripts in `runs/tilelang/skel/`, generated CUDA in `runs/tilelang/skel/src/`, families under
`agentic/tilelang/families/{gemm_pipelined,gemm_streamk,gemv_splitk,gemm_frag_w4a16}/` with the upstream path and
commit in each `UPSTREAM.md`. All times warm (back-to-back, as cuBLASLt was timed in S1) unless marked cold
(TileLang's `do_bench`, which flushes the L2 between calls: Finding 0 of step 2c). cuBLASLt warm: 1024³ 34.2 µs,
4096x4096x1024 400.8, M=32 148.9, M=128 155.4.*

**Skeleton 1, `examples/gemm/example_gemm.py` with warp specialization off (cp.async, every warp computes).** The
example's kernel body, unchanged, compiled once as the example (TMA + the producer warp group) and once with
`tl.disable_warp_specialized`. Both correct (max error 0.062 against torch, fp16 rounding). At 1024³ with the example's
own 128x128x32 tile, 3 stages, 128 threads: as-is 45.2 µs warm (74.7 cold), warp specialization off 38.9 µs warm
(63.4 cold). So the skeleton the brief asked for is already 14 % faster than the round-1 pick at this shape and sits
**+14 % against cuBLASLt** before any knob is touched; the generated code is `cp_async_gs` + `ldmatrix` + `mma_sync`,
128 threads, `__launch_bounds__(128, 1)`. The other tiles tried unchanged, as-is / off: 64x128x32 38.9 / 39.7,
64x64x32 43.0 / 37.7, 128x128x64 45.2 / 45.9, 256x128x32 117 / 101, 128x256x32 116 / 91. The knob exposure and the
solver's pick follow below once the cp.async constants are measured.

**Skeleton 2, `examples/gemm_streamk/`.** As written the example does not compile at the pinned commit: the
warp-specialization pass fails with `ProducerConsumerWS: failed to replace pipeline loop`
(`src/cuda/transform/producer_consumer_ws.cc:1789`). With `tl.disable_warp_specialized` it compiles, lowers to
cp.async, and is correct at its own shape (256x1024x512, 14.4 µs) and at ours. Its kernel closes over module-level
numbers (shape, `total_sm = 108` for an A100, the 16x128x32 tile, 64 threads, 2 stages in the stream-K loop and 1 stage
hard-coded in the data-parallel loop); "unchanged" means those numbers set to our shape and 48 SMs, the body as is.
1024³: 512 tiles, 80 stream-K tiles + 432 data-parallel, **114.7 µs, +235 % against cuBLASLt** (C zeroed outside the
timed region; the fp32 atomic output is the example's). 4096x4096x1024: 8192 tiles, **2444 µs, +510 %**. The gap is
the example's choices, not stream-K: a 16-row tile moves 9x the bytes per flop of a 128x128 tile, 64 threads hold
one warp's worth of MMA issue, and the data-parallel loop, which does 98 % of the work at 4096x4096x1024, has no
pipelining at all. Those are exactly the holes the family exposes (`gemm_streamk/knobs.yaml`: tile, stages of both
loops, threads, programs).

**Skeleton 3, `examples/gemv/` and the fragment kernel of PR #3290.** The GEMV example computes one row; at M=1,
N=K=4096 fp16 its vectorized split-K kernels stream B at 239 to 247 GB/s warm (136 to 140 µs; 211 cold, 159 GB/s),
torch's own GEMV 191 µs. That is the weight-stream floor for the decode shapes: the 32 MB of B at the measured DRAM
rate is 124 µs, our M=32 pick already runs at 124, cuBLASLt at 149. The PR #3290 fragment kernel ran unchanged on
the pinned commit (its imports exist; its compute-capability check wants 12.0 and the GB10 reports 12.1, so it was
launched through its own `launch()`): relative error 1.7e-3 against the dequantized reference, deterministic. It is a
W4A16 kernel, 0.56 B per weight against 2 B for the fp16 baseline, so its times are not the S1 comparison: M=32
24.8 µs warm / 94.0 cold with its heuristic config (32-row pass, 4 warps, split-K 2), M=128 69.6 warm / 149.2 cold
(two 64-row passes). Cold, its 9.4 MB (M=32) and 18.9 MB (M=128) of weights move at 100 to 127 GB/s; warm the M=32
weights are L2-resident (9.4 MB fits), which is why warm and cold differ 4x there and 2x at M=128. Read for its
loader: a hand-written cp.async stage ring (`ptx_commit_group`/`ptx_wait_group`, `call_extern` into a `prelude`),
a run-time persistent grid, 64-bit iteration partition and a ticketed fixed-order fix-up. None of it is TileLang
primitives we could lift into a family without writing a kernel, which the brief rules out; the family here is the PR
module unchanged with its `FragConfig` knobs.

**Race check, before trusting any warp-specialized kernel (issue #3451).** Two checks on every skeleton:
`compute-sanitizer --tool racecheck` on one launch and 30 random inputs against torch; plus a static check of the
generated CUDA (`agentic/tilelang/skeletons/release_check.py`) that finds the stage buffers the producer fills and
verifies no consumer read of them follows the back-pressure `arrive` in the loop body, the exact shape of #3451
(tested on a synthetic late read). Results: (1) the TMA + warp-specialized kernel of our family releases each stage
after its last ldmatrix: OK, no hazards. (2) **On this commit, TileLang warp-specializes only when the copies go
through TMA.** Our family's `tma: 0, warp_spec: 1` point renders a kernel with 128 threads, no producer group and no
mbarrier, byte for byte the `warp_spec: 0` kernel; so round 2's 112 "cp.async + warp specialization" points were the
plain kernel, and the #3451 pattern (a cp.async-filled stage under warp specialization) cannot be produced by any
of these families here. The knob is a no-op on the cp.async path; `gemm_pipelined` pins it. (3) The cp.async
skeleton and the fragment kernel: no hazards, correct on every input. (4) **The stream-K example has a real
write-after-read hazard**: racecheck reports it on every launch (`__shared__ 0x4600`, inside `A_shared`). The
generated CUDA has no `__syncthreads()` between the last multiply of one `while` iteration (the pipeline epilogue
reading the last stage) and the first cp.async of the next (the prologue writing stage 0); the data-parallel loop
that follows does get that barrier. Thirty random inputs were still correct, so on this GPU it is a timing race.
Draft report with the repro in `agentic/tilelang/repro/streamk_war/ISSUE.md`, to show Li before filing.

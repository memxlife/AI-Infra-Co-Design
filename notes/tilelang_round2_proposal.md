# Work Proposal: TileLang as the Tool Surface for Agentic Kernel Optimization

Oct 19, 2026 · @Li Shang

## 1. What to build, and the boundary conditions

Build a loop in which a large model writes a TileLang kernel with holes, a MIP solver fills the holes, TileLang compiles the result to SASS, and the kernel runs on the DGX Spark (GB10). The loop plugs into the multi-agent system already in `agentic/` and shows up in the architect console. This document is the brief for Claude Code; it should be executable without asking Li questions.

**Read these first, in this order.** They are the fixed points; extend them, do not reinvent them.

1. `agentic/` in the repo `https://github.com/memxlife/AI-Infra-Co-Design` (local clone: `/Users/lishang/work/research/AI-Infra-Co-Design`): the ledger, the contract schema (`agentic/schema/contract.json`, `chip.json`), the solver and verifier services, `agentic/templates/smem_tiling/smem_template.py` (the GEMM tiling template with its `cost_model` and `HardwareContract`), tests in `tests/`, and the demo ledger `runs/demo.db`.
2. `notes/multi_agent_design.md` (agents, contract-as-message, ledger, build order S0–S7) and `notes/architect_console_design.md` (the UI). The new work is one more Formulator tool family plus one more Evaluator backend inside that design.
3. `notes/mlir_codesign_framework_design.md`, for the dialect ladder and the `tmpl` contract. TileLang is not MLIR (it sits on TVM TensorIR); here it is a printed target below `tmpl`, not the IR.
4. The TileLang source: clone `https://github.com/tile-ai/tilelang` (MIT, \~125k lines C++ in `src/`, \~52k lines Python in `tilelang/`). Section 3 says which directories matter.
5. The partially built UI in Li's local directory (he will name it when handing this over). Reuse its screens; add the kernel view described in Section 6.

**Boundary conditions.**

- Do not fork TileLang wholesale. Wrap it. The repo is a dependency installed from source (`pip install -e .` with CUDA 13 and the Blackwell target), pinned to one commit recorded in the ledger. The one exception is Section 10's second kind of hidden decision: a pass may be patched to accept a choice it currently computes, as a contained change to one file in `src/transform/`, kept as a patch file under `agentic/tilelang/patches/` with the default path untouched.
- The only large model in the loop is the Formulator, as in `multi_agent_design.md` §2.1. The solver, the compile step, the profiler and the verifier are code.
- Target hardware is the Spark (`spark-6951`, reachable over SSH from Li's Mac; GB10, Blackwell, `sm_121`). Everything that touches a GPU runs there; everything else runs on the Mac.
- First workload is GEMM (FP16 in, FP32 accumulate), shapes 1024³ to 8192³ plus the skinny decode shapes M in {16, 32, 64, 128}. Attention comes second and only after the GEMM milestones pass.
- Out of scope for this round: MLIR integration, RTL or Ventus, Triton or Gluon backends, multi-GPU.

**What done looks like.** From the console, Li picks a GEMM shape, watches the Formulator emit a TileLang template with holes, sees the solver's assignment and predicted time, sees the compiled kernel's registers, spills and shared memory, sees the measured time on the Spark next to the prediction, and can compare it against TileLang's own autotuner and against cuBLAS on the same shape.

## 2. The idea in one page

TileLang already did the hard part of our template idea: every performance decision is a primitive in the kernel source. Tile sizes, pipeline depth, thread count, shared-memory layout and swizzle, pipeline stage order, warp roles, copy width, tile traversal order. In Triton most of these are inside the compiler and invisible; in TileLang they are arguments and annotations a program can read and set. So the agent does not need to take TileLang's compiler apart. It needs to write a TileLang kernel in which some of those arguments are holes, and the holes are the MIP's decision variables.

The loop, per workload shape:

1. The Formulator (large model) picks a template family and writes a TileLang kernel with holes, plus a contract: the holes with their menus, the constraints that tie them to the chip, and a cost model. This is the paper's template (O, X, C, Pre, P, π, Σ, Ĵ, Γ) with P being TileLang source.
2. The solver fills the holes: a MILP over the integer knobs, minimizing predicted time subject to shared memory, registers, occupancy and divisibility.
3. The compile tool prints the finished kernel, calls TileLang's JIT, and reads back what ptxas says: registers per thread, spill bytes, shared bytes. A spill or an over-budget allocation is a hard rejection, fed back as a tightened constraint.
4. The profile tool runs the kernel on the Spark, checks the output against torch, and returns latency plus Nsight counters. Measured versus predicted goes into the ledger as the calibration record, as `multi_agent_design.md` §2.3 already specifies.
5. The verifier checks the obligations the contract states (index coverage, exact-once for split-K, lifetime for the pipeline) with the existing bounded checker.

What this gives that the existing GEMM template does not: a real compiler underneath, a real GPU to calibrate against, and a built-in baseline. TileLang's own autotuner is exhaustive timing over a hand-written config list; its Carver module is a heuristic tile recommender with an analytic shared-memory and register estimate. Both run on the same kernel source, so the comparison is clean: same kernel, same shape, same chip; solver-over-model versus list-timing versus heuristic versus cuBLAS. That comparison is the first result the paper needs, the one that quantifies the structure of the solution space.

Two limits to state up front, so the design does not over-promise. The knobs split into two kinds: integers from menus (tile sizes, stages, warps, split-K, swizzle stride), which the solver owns, and structural choices (which layout, which pipeline stage assignment, warp-specialized or not, which fusion), which change what the program is and belong to the agent, one template family each. And the MIP is only as good as its cost model: the thing TileLang gives us is that every observable lines up with a decision in the source, so the model can be calibrated knob by knob on the Spark. The calibration loop is the centerpiece; the solver is the cheap part.

## 3. Factoring TileLang: four layers, no fork

TileLang is already four layers. We do not split the source; we wrap each layer as one tool the agent can call. Paths are relative to the TileLang clone (commit of 16 Oct 2026; pin whatever commit is current when the work starts).

| Layer | What it is | Where in the clone | Becomes which tool |
| --- | --- | --- | --- |
| L1 Frontend | The Python primitives the kernel author writes: `T.Kernel`, `alloc_shared`, `alloc_fragment`, `copy`, `gemm`, `Pipelined`, `annotate_layout`, `use_swizzle`, warp specialization. The knob surface. | `tilelang/language/` (`kernel.py`, `allocate.py`, `copy_op.py`, `gemm_op.py`, `loop.py`, `annotations.py`, `warp_specialize.py`, `tile_schedule.py`); worked kernels in `examples/gemm/`, `examples/gemm_tcgen05/`, `examples/flash_attention_sm100/` | `knobs.inventory`, `template.render` |
| L2 Transform passes | The TIR passes that turn the frontend into a legal schedule: layout inference, pipeline planning, warp specialization, bank-conflict legalization, vectorization. These passes accept or reject a knob assignment and decide what the source left open. | `src/transform/` (22k lines): `layout_inference/` (with its own `layout_cost_model.cc`), `pipeline_planning.cc`, `inject_pipeline.cc`, `loop_vectorize.cc`, `thread_storage_sync.cc`, `merge_shared_memory_allocations.cc`, `common/warp_specialize.h`; Python driver `tilelang/engine/lower.py`; pass events in `tilelang/instrumentation/events.py` | `template.legalize` |
| L3 Codegen + ptxas | CUDA C++ emission, nvcc, ptxas; the resource report. | `src/cuda/`, `src/tl_templates/cuda/` (header templates over CUTLASS); `tilelang/jit/kernel.py` (`JITKernel`: `get_kernel_source`, `resource_usage`, `n_regs`, `n_spills`, `_get_ptx`, `_get_sass`) | `kernel.compile` |
| L4 Runtime + profiler | Launch on the GPU, correctness check against torch, timing, counters. | `tilelang/profiler/` (`do_bench` with the `cupti` backend, `assert_allclose`), `tilelang/jit/adapter/`; autotuner for the baseline in `tilelang/autotuner/tuner.py` | `kernel.profile`, `baseline.autotune` |

Two modules sit beside the four layers and are worth reusing rather than rebuilding:

- `tilelang/carver/`: a tile-configuration recommender inherited from BitBLAS. `carver/arch/cuda.py` holds a chip model (`smem_cap`, `reg_cap`, `warp_size`, `sm_partition`, L2 size); `carver/roller/policy/` enumerates tiles under those caps with an analytic shared-memory and register estimate; `carver/template/matmul.py` and `flashattention.py` are its operator templates. This is a heuristic cousin of our template: same constraints, no solver, no calibration. Use its arch model as the first draft of the GB10 `chip.json` cross-check, and use `recommend_hints(topk=20)` as a second baseline.
- `tilelang/analysis/` and `tilelang/tools/`: `layout_visual.py`, `plot_layout.py`, `pass_visualizer`, `lower_trace`, `pass_timing.py`. These produce the pictures of layouts and pass sequences the console can show.

What the layers give back, and what the loop does with it:

| From | Signal | Used as |
| --- | --- | --- |
| L2 | pass accepted or threw; inferred layout; pipeline plan | legality constraint C (mined, not guessed); frozen choices made visible |
| L3 | registers per thread, spill bytes, shared bytes, max threads | hard constraints (spill = reject; smem ≤ cap), occupancy term in Ĵ |
| L4 | latency (median of N), correctness verdict, Nsight counters (occupancy, tensor-pipe utilization, HBM GB/s, bank conflicts, stall reasons) | measured cost; calibration of Ĵ term by term; optimism record |

The boundary of what we can see is ptxas: TileLang stops at CUDA C++ source, so everything down to that point is readable and attributable to a source line; register allocation and instruction scheduling below it are NVIDIA's and show up only as the L3 report and the L4 counters.

## 4. The tools the agent calls

Six tools, all code, all returning typed JSON that goes into the ledger. The Formulator sees the first three; the Orchestrator calls the rest. Signatures are the spec; Claude Code picks the implementation.

| Tool | Input | Output | Wraps |
| --- | --- | --- | --- |
| `knobs.inventory(family, chip)` | template family name (`gemm_basic`, `gemm_ws`, `gemm_splitk`, later `attn_fwd`), chip id | the knob list for that family: name, kind (`int_menu` / `structural`), menu, which primitive it binds, which constraint it enters | a hand-written YAML per family in `agentic/tilelang/families/`, checked against L1 at import |
| `template.render(family, holes)` | family, dict of hole values (may be partial) | TileLang source text; unfilled holes left as named `constexpr` parameters | Jinja over the family's kernel file |
| `template.legalize(source, holes)` | rendered source, full assignment | `{ok, reason, inferred_layout, pipeline_plan, smem_bytes_est}` | L2: run `tilelang.engine.lower` with pass instrumentation, catch the throw, read the layout and pipeline attributes off the lowered module |
| `kernel.compile(source, shape, chip)` | source, M N K, target | `{kernel_id, n_regs, n_spills, smem_bytes, max_threads, cuda_src, sass_path}` | L3: `JITKernel` + `resource_usage`, `export_sources`, `export_sass` |
| `kernel.profile(kernel_id, shape, counters)` | compiled kernel, shape, which Nsight metrics | `{correct, latency_ms, tflops, counters{...}}` | L4: `Profiler.assert_allclose` + `do_bench(backend="cupti")`; `ncu --metrics ...` on a one-shot run |
| `baseline.run(kind, family, shape)` | `autotune` / `carver` / `cublas` | latency and the config chosen | `AutoTuner` over the family's config list; `carver.recommend_hints`; `torch.matmul` |

Rules that keep the loop honest:

- The Formulator never calls `kernel.compile` or `kernel.profile`. It proposes; the Orchestrator measures. Otherwise the model tunes by trial and the solver is decoration.
- Every tool call is a ledger event with its inputs hashed, so a run replays without the GPU.
- `kernel.compile` and `kernel.profile` execute on the Spark over SSH. Keep the transport trivial: `rsync` the rendered source and a runner script, run it under the Spark's Python environment, bring back one JSON file. No daemon in the first round.
- `template.legalize` is the cheap check and runs on the Mac; it needs TileLang importable but no GPU.

What `template.render` produces for the first family, so the hole convention is fixed:

```python
@tilelang.jit
def gemm(A, B, BM, BN, BK, STAGES, THREADS, SWZ):          # holes = constexpr params
    M, N, K = T.const("M, N, K")
    A: T.Tensor((M, K), T.float16); B: T.Tensor((K, N), T.float16)
    C = T.empty((M, N), T.float16)
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=THREADS) as (bx, by):
        A_s = T.alloc_shared((BM, BK), T.float16)
        B_s = T.alloc_shared((BK, BN), T.float16)
        C_f = T.alloc_fragment((BM, BN), T.float32)
        T.use_swizzle(panel_size=SWZ)
        T.clear(C_f)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=STAGES):
            T.copy(A[by*BM, k*BK], A_s); T.copy(B[k*BK, bx*BN], B_s)
            T.gemm(A_s, B_s, C_f)
        T.copy(C_f, C[by*BM, bx*BN])
    return C
```

This is `examples/gemm/example_gemm.py` with the six integer knobs lifted to parameters. The second family (`gemm_ws`) is `examples/gemm_tcgen05/gemm_tcgen5mma_ws.py` treated the same way; it adds the producer/consumer warp split as a structural choice and is the right one for GB10's tensor cores.

## 5. Which knobs the solver owns, which the agent owns

The solver gets the knobs that are integers from a menu and that leave the program's structure unchanged. The agent gets the knobs that change what the program is. A reviewer who knows TileLang will check this split first.

| Knob | TileLang primitive | Kind | Owner | Menu or choice |
| --- | --- | --- | --- | --- |
| BM, BN | `alloc_shared`, `Kernel` grid | int menu | solver | {64, 128, 256} |
| BK | `alloc_shared`, `Pipelined` bound | int menu | solver | {32, 64, 128} |
| STAGES | `Pipelined(num_stages=)` | int menu | solver | {2, 3, 4, 5} |
| THREADS | `Kernel(threads=)` | int menu | solver | {128, 256, 512} |
| SWZ (swizzle panel) | `use_swizzle(panel_size=)` | int menu | solver | {0, 4, 8, 16} |
| SPLIT\_K | grid z + `atomic_add` | int menu | solver | {1, 2, 4, 8}; only in family `gemm_splitk` |
| RASTER (tile traversal group) | `tile_schedule` | int menu | solver | {1, 4, 8, 16} |
| layout of A\_s, B\_s | `annotate_layout` | structural | agent | row-major vs swizzled vs TMA-canonical |
| pipeline stage assignment | `Pipelined(order=, stage=)` | structural | agent | uniform vs load-ahead-compute-behind |
| warp roles | `warp_specialize` | structural | agent | none vs producer/consumer; family `gemm_ws` |
| copy lowering | pass config | structural | agent | `cp.async` vs TMA |
| epilogue fusion | fragment ops before the final `copy` | structural | agent | none vs bias+activation |

The MILP for family `gemm_basic`, as the contract's search face should state it. One-hot menus keep it linear; `b_m,i` is 1 when BM takes its i-th menu value, and so on.

```latex
\min_{x}\; \hat J(x) = \max\!\left(\frac{2MNK}{F_{\mathrm{tc}}\cdot u_{\mathrm{occ}}(x)},\; \frac{\mathrm{bytes}(x)}{B_{\mathrm{HBM}}}\right) + t_{\mathrm{pipe}}(x)
```

subject to, for a chip contract with shared-memory cap S\_cap, register file R\_cap, SM count P:

- shared memory: STAGES · (BM·BK + BK·BN) · 2 bytes ≤ S\_cap (the lifetime constraint C3 of the existing template, now with real bytes);
- registers: BM·BN·4 / THREADS + overhead(THREADS) ≤ 255, and the resulting blocks per SM ≥ 1;
- occupancy: blocks per SM from the two caps above, giving u\_occ as a piecewise-constant lookup, which is linear in the one-hot variables;
- divisibility: BM | M, BN | N, BK | K, or the padding term enters bytes(x);
- wave quantization: ceil(M/BM)·ceil(N/BN) tiles over P SMs, the tail fraction entering Ĵ as a lookup;
- threads: THREADS ≥ 4 warps for tensor-core issue; BM·BN / THREADS ≥ 32 so each thread owns a fragment.

All of these are products of menu values, so each is a table over the one-hot variables and the whole thing is a small MILP, well under 100 variables, which HiGHS or OR-Tools solves in milliseconds. The existing solver service in `agentic/agents/solver/` already does this shape of problem for the paper's template; the new part is the constraint tables generated from `knobs.inventory` plus the chip contract, not a new solver.

The cost model terms and how each is calibrated on the Spark:

| Term | First guess | Calibrated by |
| --- | --- | --- |
| F\_tc (tensor-core FLOP/s per SM) | GB10 spec | one large square GEMM at the best autotuned config, tensor-pipe utilization from `ncu` |
| B\_HBM | GB10 spec (LPDDR5X, \~273 GB/s) | a bandwidth-bound shape (M = 16 decode), `dram__bytes` from `ncu` |
| u\_occ(blocks/SM) | 1 − e^{−k·blocks} | latency across STAGES × THREADS at fixed tiles |
| t\_pipe (fill/drain, barrier) | STAGES · BK-slice time | small-K shapes |
| bank-conflict penalty on SWZ | 0 | `l1tex__data_bank_conflicts` across SWZ values |

Each row is one experiment of a few dozen runs, and each fixes one term. That is the knob-by-knob calibration that TileLang's explicit source makes possible and that a Triton kernel would not allow.

## 6. Where it plugs into what exists

Nothing in `multi_agent_design.md` changes. TileLang enters as one more Formulator tool family, one more Evaluator backend, and one more view. Proposed layout inside the repo:

```
agentic/
  tilelang/                      # new
    families/
      gemm_basic/   kernel.py.j2  knobs.yaml  contract.json
      gemm_ws/      kernel.py.j2  knobs.yaml  contract.json
      gemm_splitk/  ...
    tools.py                     # knobs.inventory, template.render, template.legalize
    remote.py                    # rsync + ssh runner for the Spark
    spark_runner.py              # runs ON the Spark: compile, profile, ncu, emit JSON
    calibrate.py                 # the five experiments of Section 5, writes chip.json terms
  agents/evaluator/tilelang.py   # new backend beside roofline.py / ventus.py
  agents/formulator/prompts/tilelang_family.md   # new prompt: pick family, fill contract
  schema/kernel_report.json      # new artifact: compile + profile report
  examples/gemm_tilelang/        # S0 fixture: one rendered kernel, one report, one ledger
```

What to reuse as is:

- The contract schema (`agentic/schema/contract.json`) and the ledger. A TileLang template is a contract whose `program` field holds the rendered source and whose `decisions` are the knobs.yaml menus. Add one artifact type, `kernel_report`, for the L3 + L4 output.
- The solver service. Feed it the constraint tables; do not write a second solver.
- The verifier. The obligations it already checks on the GEMM template (index coverage, lifetime) apply unchanged; `gemm_splitk` adds exact-once, which the existing checker also knows.
- The chip contract (`agentic/schema/chip.json`). Add a `tilelang` block with target string (`cuda` with `sm_121`), the ptxas caps, and the calibrated terms from `calibrate.py`. Cross-check against `tilelang/carver/arch/cuda.py` for the same chip.
- The hardware calibration work already in the repo for the GB10 (the microarchitecture notes and the calibration suites). The five experiments in Section 5 should reuse that harness and its ledger conventions, not start a parallel one.

What to add to the console (`architect_console_design.md`):

- In the inspector, a fourth face next to formulation / optimization / verification: **Kernel**. It shows the rendered TileLang source with the filled holes highlighted, the ptxas line (registers, spills, shared bytes, occupancy), and the measured-versus-predicted bar with the dominant resource named. One click opens the CUDA source TileLang generated, and the layout picture from `tilelang/analysis/layout_visual.py`.
- On the timeline, one more column: the baseline gap, i.e. our latency over the best of autotune / carver / cuBLAS for that shape.
- Li's local UI directory already has some of these screens. Claude Code should read it before touching `agentic/ui/`, keep whatever renders the ledger already, and add the Kernel face into that structure rather than beside it.

What runs where:

| Piece | Runs on | Needs |
| --- | --- | --- |
| Formulator, solver, verifier, ledger, console | Mac | Python, TileLang importable (CPU only is fine for `legalize`) |
| `kernel.compile`, `kernel.profile`, `calibrate.py`, baselines | Spark over SSH | CUDA 13, TileLang built for `sm_121`, torch, `ncu` |

## 7. Build order

Seven steps, each with a test that must pass before the next starts. Steps 1 to 3 need no large model and no solver; they are the plumbing and they are where most of the time goes.

| Step | Deliverable | Test |
| --- | --- | --- |
| T1 TileLang on the Spark | TileLang installed from source on `spark-6951` for `sm_121`; `spark_runner.py` compiles and profiles `examples/gemm/example_gemm.py`; `remote.py` round-trips from the Mac | From the Mac, one command returns a `kernel_report` JSON for 1024³: correct, latency, n\_regs, n\_spills, smem bytes. Record the TileLang commit and CUDA version in the ledger. |
| T2 First family | `gemm_basic`: `kernel.py.j2`, `knobs.yaml`, `contract.json`; `knobs.inventory`, `template.render`, `template.legalize` | Rendering with the example's values reproduces the example kernel byte for byte. `legalize` rejects STAGES=8 at BM=BN=256 with a shared-memory reason, and accepts the example. |
| T3 Sweep and baselines | Full grid of the six integer knobs at 1024³, 4096³ and M=32 decode, run on the Spark; `baseline.run` for autotune, carver and cuBLAS | A table per shape: every config with latency, regs, spills, smem, counters. This table is the first measurement of the solution space: how many configs the constraints remove, how many spill, where the cliffs are. Save it; it is a paper figure. |
| T4 Cost model and calibration | `calibrate.py` runs the five experiments of Section 5; `Ĵ` implemented as a function of the knobs and the chip terms; the optimism record wired | On the T3 sweep, Ĵ ranks the top-10 configs with Spearman ≥ 0.8, and never predicts faster than measured on more than 5% of configs (one-sided). Report the error histogram. |
| T5 Solver in the loop | Constraint tables from knobs.yaml + chip.json into the existing solver service; Orchestrator runs propose → legalize → solve → compile → profile with a scripted Formulator | For each of the three shapes, the solver's pick is within 5% of the best config in the T3 sweep, found with one compile and one profile instead of the full grid. Report solver time, and the gap to autotune, carver and cuBLAS. |
| T6 Large-model Formulator | Prompt around the family list and the knob inventory; the model emits family + contract, never source by hand | On five new shapes, the Formulator's contract passes `legalize` first time ≥ 80%, and the loop's result stays within 5% of a fresh autotune. Every contract carries a `structure` block. |
| T7 Console | Kernel face in the inspector, baseline-gap column on the timeline, layout picture | Li can pick a shape, watch one iteration, open the kernel, read the ptxas line and the measured-vs-predicted bar, and see the three baselines, without the CLI. |

After T7: `gemm_ws` as the second family (warp-specialized, `tcgen05` path on Blackwell), then `gemm_splitk` for the decode shapes, then `attn_fwd` from `examples/flash_attention_sm100/`. Each new family is T2 + T3 + T5 again with the same tools; nothing else changes.

Time guess, one person: T1 two to three days (the Blackwell build is the risk), T2 two days, T3 one day of machine time plus a day, T4 three days, T5 two days, T6 three days, T7 three days. About three weeks to a demo on GEMM.

## 8. Risks, open questions, what to report back

Risks, with the fallback for each:

- **TileLang does not build for `sm_121`.** GB10 is a new target; TileLang's Blackwell paths are written for `sm_100a`/`sm_103a` (see the arch prefix note in `tilelang/carver/arch/cuda.py`) and the `tcgen05` examples. Round 1 confirmed this: `tcgen05` and `wgmma` are rejected by ptxas on `sm_121`. Stay on `gemm_basic` (plain `mma`, which does build) for the GEMM round and report the gap to the baseline honestly; do not try to make the `tcgen05` path build.
- **`ncu` is unavailable or too slow on the Spark.** Fall back to `do_bench(backend="cupti")` for latency and to the ptxas report for resources; calibrate only F\_tc, B\_HBM and u\_occ from latencies. Bank-conflict calibration waits.
- **The cost model is not one-sided.** If T4's optimism test fails, do not tune the model by hand. Record which term is off, add its calibration experiment, rerun. The paper's claim depends on the model being honest, not accurate.
- **The solver's pick is far from the sweep's best.** This is a finding, not a failure. It means the space has structure the model misses (a slope, not a cliff). Report which configs beat the pick and by how much; that is the long-tail measurement the lecture's limits slide already predicts.
- **Dynamic shapes.** First round is static shapes only (`T.const`). Serving needs symbolic M; TileLang supports it (`examples/dynamic_shape/`) but the knob menus change with M, so it is a second round.

Open questions for Li, to settle during T1–T3 rather than before:

1. Latency metric: median of 100 after 10 warm-ups, or TileLang's `do_bench` default? Pick one and write it into the ledger schema.
2. Should `legalize` run the full L2 pass pipeline (slow, exact) or a Python precheck of the shared-memory and divisibility rules (fast, approximate), with the full pass only at compile time? Recommendation: both; the precheck is what the solver's constraint tables encode, and disagreement between the two is itself a finding.
3. Where the T3 sweep table lives: ledger only, or also a CSV under `runs/` for the paper's figures. Recommendation: both.

What to report back after each step: the test result in one line, the ledger run id, any TileLang behavior that contradicts this document (file and line), and, from T3 onward, the three numbers that matter: fraction of configs removed by constraints, fraction that spill, and the gap between the solver's pick and the best measured.

## 9. Round 2: corrections to round 1 and the next two experiments

Round 1 (`notes/tilelang_integration.md`, `notes/tilelang_findings.md`, `runs/tilelang/report.md`, commit `807be1e`) delivered T1–T5 and T7 on `gemm_basic`. The sweep tables stand: 61–64% of 1,296 points removed by the constraints, 1–2% of survivors spill, the solver's single compile within 0.1% / 1.8% / 5.7% of the grid's best at 4096³ / 1024³ / M=32. Two reported numbers are not yet trustworthy and come first.

| # | Item | What to do | Done when |
| --- | --- | --- | --- |
| R1 | cuBLAS baseline is a fallback kernel | `torch.matmul` (cu128) is 3–8× slower than a 28-line TileLang kernel: torch has no `sm_121` kernels. Replace with cuBLASLt from the Spark's CUDA 13 toolkit (small C++ harness, same CUDA-event timing) or the CUTLASS profiler. If neither runs, drop the row and say why. | cuBLASLt within 2× of our best at 4096³, or the row removed from `report.md` and the console. |
| R2 | `scale: 0.588` in `tilelang_terms.yaml` | Remove the global deflation. Make each term a lower bound on its own (`overlap: max`, since TMA + producer warps overlap by construction). Fit the missing decode term: at M=32, 64 tiles on 48 SMs is two waves with the second one-third full plus a per-tile start-up the long-K calibration never saw; every M=32 point sits at +40–50%. | One-sided test passes (≤ 5% bound-broken) at all three shapes with `scale: 1.0`. 1024³ currently fails at 9.5%. |
| R3 | Noise band | Same config measures 60.2 / 62.5 / 63.4 µs minutes apart. Pin clocks if the driver allows; three interleaved runs per point; record clock and temperature in `kernel_report`. State the ranking test as "pick inside the measured noise band of the best," not Spearman over all points. | Noise band measured per shape and written into `report.md`; T5's test restated against it. |
| R4 | Two autotuner claims, kept apart | The 26–29% win at 4096³ is the panel knob, absent from TileLang's 144-config list: a claim about the menu. The solver claim is: over our own menu, exhaustive timing of 504 legal points gives 1377 µs, the solver's one compile gives 1379 µs. | `report.md` shows both numbers under separate headings; no merged percentage. |
| R5 | Out-of-sample shapes | The sweep shapes calibrated the terms and then graded them. Add 2048³, 8192³, 4096×4096×1024, M=128×4096×4096: solver pick plus a 30-point random sample of the legal grid each. | Gap from pick to sample-best reported per shape; bound-broken rate on the sample. |
| R6 | Second family: `gemm_splitk` | The Formulator's structural choices are all "as inferred" and its replies are fixtures; the loop runs the same without it. Add split-K (grid z + `atomic_add`, exact-once obligation to the verifier), aimed at the decode shape where tiles < SMs. The Formulator must pick the family from the probe's tiles-per-SM; the solver picks SPLIT\_K ∈ {2, 4, 8}; the verifier discharges exact-once. | At M=32, a live Formulator call chooses `gemm_splitk`, the kernel is correct, and the time beats the `gemm_basic` pick (200 µs) by a margin outside the noise band. |
| R7 | The loop reacts once | Halve `l2_bytes_effective` in the chip terms and run 4096³: the prediction misses, `diagnose("model")` fires, the term is recalibrated, the re-solve changes the pick. Keep the ledger as the fixture. | Ledger shows measured → diagnosis → recalibration → new pick, replayable without the GPU. |

Order: R1, R2, R3 first (they correct round 1), then R4 and R5 (reporting), then R6 and R7 (round 2 proper). `tcgen05` stays out: it does not build on `sm_121`.

What to report back: the same three numbers as before per shape, now out of sample, plus the noise band, plus one sentence on what changed in the model when the scale came out.

## 10. Round 2 brief: the goal, the search space, and how to work

*Added after round 1. This section is the brief for round 2. Where it conflicts with Sections 1–9 (test thresholds, sweep sizes, repetition counts, the no-fork rule), this section wins.*

### 10.1 The goal

Use an existing open-source compiler as the tool the large model formulates with. The compiler is taken apart into pieces that serve two purposes. First, it **exposes the search space**: the knobs, the rules that make a setting illegal, and the costs are read out of the compiler rather than invented. Second, it **builds and runs the answer**: once the solver has filled the holes, the same compiler lowers the kernel to machine code and runs it on the real chip. The large model sits between: it reads what the tools say, writes the contract (the template), calls the MIP solver, and hands the result back to the compiler. Nothing is written from scratch. Nothing is searched by timing.

TileLang is the first compiler we do this to, because it already exposes more at the user level than Triton or CUDA. It is a starting point, not the target. Round 1 showed the loop works on the six knobs TileLang's own autotuner sees. **Round 2 opens the space well beyond that: every decision we can name in the TileLang GEMM path becomes a hole the solver or the agent can set.** The list is in 10.2. It is not final; it is what we can reach now without rewriting TileLang.

Round 1 did the exposing by hand: `knobs.yaml` was written from the kernel source and `legalize` copies the capacity rules in Python. That was the right shortcut. Round 2 keeps the shortcut wherever reading the compiler directly would cost more than a day; the point is to show the recipe, not to finish it.

### 10.2 The round 2 search space

Sixteen knobs in three groups, against six in round 1. Every name below is from a reading of the TileLang clone; **Claude Code confirms each against the source (file and line) before it enters `knobs.yaml`, and drops any that does not exist or does not build on `sm_121`, with one line in the findings note saying so.**

**Group A: integer menus, solver-owned.** Six from round 1 unchanged: BM, BN, BK, STAGES, THREADS, SWZ. Five new:

| Knob | Where | Menu | Why it matters |
| --- | --- | --- | --- |
| PREFETCH (k-tiles ahead) | `T.Pipelined(order=, stage=)` | {1, 2, STAGES−1} | the lifetime rule P ≤ S−2+β of the paper's template, now live |
| WARP\_POLICY | `T.GemmWarpPolicy` on `T.gemm` | square / full-row / full-column | how warps split the tile; changes fragment shape and smem reads |
| VEC\_WIDTH | `coalesced_width=` on `T.copy` | {4, 8, 16} elements | copy instruction width; bank-conflict and coalescing term |
| MAXRREG | `maxrregcount` passed to nvcc/ptxas | {none, 128, 168, 255} | moves the register cliff found in round 1 |
| SPLIT\_K | grid z + `T.atomic_add` | {1, 2, 4, 8} | fills the machine at skinny M; needs exact-once (family `gemm_splitk`) |

**Group B: on/off switches TileLang already has as pass configs, never touched in round 1.** Binary, solver-owned, reachable through `tilelang.jit(pass_configs=...)` or an annotation; no fork needed.

| Switch | Where | Effect |
| --- | --- | --- |
| TMA\_LOWER | pass config `tl.disable_tma_lower` | off → copies become `cp.async`; round 1 saw only TMA |
| WARP\_SPEC | pass config `tl.disable_warp_specialized` | off → no 128-thread producer group; removes the `T+128` register cliff |
| SMEM\_SWIZZLE | `T.annotate_layout` swizzled vs row-major on `A_s`, `B_s` | bank-conflict penalty on or off |

**Group C: structural, agent-owned.** Each is a different kernel text, one template family each, verified once.

| Choice | Options | Obligation |
| --- | --- | --- |
| operand path | smem → tensor core directly, or smem → register fragment → tensor core | readiness |
| epilogue path | fragment → global directly, or fragment → smem → global | lifetime of the epilogue buffer |
| split-K | family `gemm_splitk` vs `gemm_basic` | exact-once |

The honest count: eleven integers, three switches, three structural choices. The solver's MILP grows from 93 to a few hundred binaries; still milliseconds. The first grid at 1024³ over Groups A and B is about 10⁵ points before constraints; do **not** sweep it. Sample 200 random legal points to grade the model and let the solver pick; that is the whole measurement.

### 10.3 How to work: exploration, not engineering

We are testing whether mechanisms work, not shipping a system. Every item is a quick try-and-see.

- Time cap per item: an hour or two of machine time. Two shapes only: 1024³ and M=32×4096×4096. One run per point unless the spread matters for the claim.
- A result that holds on three points is enough to move on. Build no protocol before the mechanism has been seen to work once.
- One paragraph per item in `notes/tilelang_findings.md`, then the next item. Dead ends are findings.
- The one strict item is R1, the baseline: a wrong baseline poisons every comparison after it.
- R3 shrinks to: time one config five times, report the spread. R7 shrinks to: one injected wrong term, one ledger showing diagnose → recalibrate → re-solve.
- Test gates in Section 7 (80% first-time legalize, 5% gap, Spearman 0.8) are targets to report against, not gates to stop at.

### 10.4 What to do, in order

1. **R1, R2, R3** from Section 9: fix the baseline, remove `scale`, measure the noise band. One day.
2. **Open Group A and Group B.** Add the rows to `knobs.yaml`, extend `kernel.py.j2`, confirm each knob builds. For each new knob, one small experiment: hold round 1's best fixed, vary only that knob, record the latency curve. That curve is the cost term for the knob; fit it, write it into `tilelang_terms.yaml`. Two or three days.
3. **Solve over the full space** at both shapes. Compare the solver's pick with 200 random legal samples and with round 1's pick. Report: did the bigger space find a better kernel, by how much, and which new knob did it.
4. **R4, R5** reporting from Section 9.
5. **Group C: `gemm_splitk`** (R6) with a live Formulator call at M=32. Then **R7**.
6. If time remains: the operand-path and epilogue-path choices as a third family.

### 10.5 What to report

Three numbers per shape as before (fraction removed by constraints, fraction that spill, gap from solver pick to sample best), plus for round 2: which new knobs the solver's pick actually used, the gain over round 1's pick in microseconds and percent, and for each knob the number of lines changed in `agentic/tilelang/` to expose it. That last number is the cost of the recipe, and it is what tells us whether to repeat it on Triton.

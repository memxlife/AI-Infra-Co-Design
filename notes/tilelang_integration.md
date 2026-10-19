# How TileLang is wired into the agentic flow

*Round 1, Oct 2026. Code under `agentic/tilelang/`; the hooks into the existing agents are listed at the end.*

## The idea in one paragraph

TileLang is used as a tool, never changed. Every performance decision in a TileLang kernel is a number in the
source: tile sizes, the K slice, the pipeline depth, the thread count, the block walking order. We keep TileLang's own
example GEMM kernel and leave those six numbers open ("holes"). The agent decides which numbers are worth searching
and why; a MIP solver fills them from a calibrated time model; TileLang compiles the filled-in kernel; the GPU runs it;
the measured time comes back next to the prediction. One compile and one run per shape.

## The six steps, with what each one touches

### 1. The family: TileLang's kernel with holes

`agentic/tilelang/families/gemm_basic/`

- `kernel.py.j2`: `examples/gemm/example_gemm.py` from TileLang, byte for byte, with six Jinja holes:
  `block_M`, `block_N`, `block_K` (the shared-memory tiles), `num_stages` (`T.Pipelined`), `threads` (`T.Kernel`),
  `swizzle` (`T.use_swizzle(panel_size=)`, omitted when 0). Filled with the example's values it renders the example
  exactly; TileLang then generates the same CUDA and the same SASS (checked).
- `knobs.yaml`: for each hole its menu, the TileLang primitive it binds, and which rule it enters (shared memory,
  registers, threads, waves). A skinny menu for `block_M` when M < 64.
- `contract.json`: the family as a contract for the ledger (see step 3).

### 2. The three tools the Formulator sees

`agentic/tilelang/tools.py`

- `inventory(family)`: the knob list above, as data.
- `render(family, holes)`: the TileLang source with the holes filled; an unfilled hole stays a function parameter.
- `legalize(family, holes, shape)`: the fast check on the Mac, no GPU: shared memory
  (`stages x (BM x BK + BK x BN) x 2 B + 1 KB <= 99 KB`), registers (accumulators `BM x BN / threads` plus about 24 must
  fit `65536 / (threads + 128)` rounded down to 8, because TileLang adds a 128-thread producer group and ptxas fills
  that budget), at least 16 values per thread, one block per SM. These are the same rules the MIP uses as constraints.
  TileLang's own passes are the full check, at compile time.

### 3. The Formulator: the large model, which is Claude Code in the session

`agentic/agents/formulator/model.py`, prompt in `agentic/agents/formulator/prompts/tilelang_family.md`

The model reads the shape, the chip and the inventory and returns, as JSON: the family, the menus to search (it may
only narrow the inventory's menus, never add values), the regime it expects and why, and a rationale. It never writes
kernel source and never picks knob values. In this round the replies were written by Claude Code in the session and
stored in `tests/fixtures/tilelang_model_replies.json` (eight shapes); the same class can call the Anthropic SDK or the
`claude` CLI, but no key is in the repo. `check_reply` refuses unknown families, menu values outside the inventory,
structures not in the catalog, and any source code.

`agentic/tilelang/contract.py` turns the reply into a contract of the existing schema: `program` = the rendered source
with every hole open, `decisions` = the menus, `constraints` = the rules of step 2 in words and formulas, `cost` = the
time model's terms and the calibrated numbers, `certificate_schema` = legal, compiles, agrees with torch. Template
name `gemm_tilelang`, level 5.

### 4. The MIP

`agentic/tilelang/solve.py`, HiGHS through SciPy, about 260 binaries, 10 to 25 ms.

One-hot variables per knob; product variables where a term needs them: `z` over (BM, BN, BK, threads) for the per-SM
tile time, `s` over (stages, BM, BN, BK) for the shared-memory rule and the fill cost, `v` over (BM, BN, swizzle) for the
DRAM and L2 traffic; an epigraph `T` over the three bounds. The tables come from `cost.predict`, so the solver's
optimum is the model's optimum (checked by enumeration). Registered as the `gemm_tilelang` backend of the existing
solver service.

### 5. The time model and its calibration

`agentic/tilelang/cost.py`, `agentic/tilelang/calibrate.py`, terms in `calibration/spark-6951/tilelang_terms.yaml`

Per tile on one SM: a fixed cost by thread count, the tensor work at the rate that thread count reaches, the copy time,
a fill cost by pipeline depth, a small cost per K step. Rounds of blocks over 48 SMs. DRAM traffic from which blocks run
together (the swizzle panel, or TileLang's default order), so B is re-read once per panel of rows; checked against ncu
DRAM bytes on the Spark (within 10 % for the default order and the decode shape). Each term comes from a designated
experiment on the Spark (long-K rows at a fixed big tile for the rate; a fixed-tile STAGES x THREADS series at two K
for the fixed costs; the decode shape for the DRAM rate; ncu counters for tensor utilization and DRAM bytes), and the
whole time is scaled so 95 % of the sweep points lie at or above the model (one-sided, in-sample). The structural
facts the model takes as given, all measured: TileLang adds 128 producer threads; ptxas uses the launch-bounds budget;
every configuration of this family runs one block per SM; every configuration lowers to TMA + mma.

### 6. Compile, run, measure

`agentic/tilelang/remote.py` -> `agentic/tilelang/spark_runner.py` (runs on the Spark), `agentic/agents/evaluator/tilelang.py`

The evaluator renders the solver's point, rsyncs the source and the runner to the Spark, and runs it under the Spark's
venv: `fn.compile(M, N, K, **holes)`, a run against `A @ B` (1e-2 tolerance), the median of CUDA-event timing over
100 ms of repetitions, the exported CUDA source compiled again with `nvcc -Xptxas -v` for registers, spills and static
shared memory (TileLang's own `n_regs`/`n_spills` are HIP-only), the dynamic shared memory from the device function's
attribute, and optionally ncu on that one kernel (tensor-pipe utilization, L2-miss sectors as DRAM bytes, bank
conflicts). One JSON comes back, the `kernel_report` artifact kind, and goes into the ledger beside the prediction.

## Where it hooks into the existing agents

- `agentic/agents/orchestrator.py`: `gemm_tilelang` is a checked template. For it the loop is propose -> family check
  (`agentic/tilelang/verify.certify_family`: every menu point the constraints admit passes legalize) -> solve
  (`solve_family`) -> instance check (the solver's point passes legalize) -> compile + profile (the TileLang evaluator,
  `evaluator_backend = "tilelang:<ssh-alias>"`) -> optimism record -> regime probe -> admitted. The calibrated terms
  travel inside the contract (`cost.terms_values`), so the solver, the probe and the evaluator use the same numbers.
- `agentic/agents/solver/__init__.py`: the `gemm_tilelang` backend.
- `agentic/schema/kernel_report.json`: the new artifact kind.
- The console: `agentic/templates/gemm_tilelang/describe.yaml` (plain words for the six knobs and the rules) and a
  Kernel tab (`agentic/ledger/plain.kernel`, `agentic/ui/static/inspector.js`) showing registers, spills, shared memory,
  the Spark's time against the prediction, the baselines of the same shape, and the layout in words.
- Baselines, same kernel, same shape, same timing (`agentic/tilelang/spark_baseline.py`): TileLang's autotuner list
  (144 configs from `example_gemm_autotune.py`, each compiled and timed), the carver's `recommend_hints(topk=20)`,
  and `torch.matmul` (cuBLAS).

## What the quick end-to-end test showed (4096^3, 16 s wall)

Proposal (big tiles, panels only) -> MIP in 8 ms: 256x128 tile, K slice 64, 2 stages, 256 threads, panel 4 ->
TileLang compile 1.4 s: 168 registers, no spills, 97 KB shared -> Spark: 1434 µs, correct, 95.8 TFLOP/s. TileLang's
autotuner: 1942 µs after 132 compiles (12 more crashed at launch). Carver: 2267 µs. cuBLAS via torch: 10807 µs.
The prediction was 1038 µs, 38 % optimistic: the model ranks well, its absolute times are loose.

## Where it falls short, for the revision

- The model's absolute time is loose (one global scale keeps it one-sided). Its ranking is good on the big shapes
  (Spearman 0.92 at 4096^3) and weak where the candidates are within noise of each other (1024^3, M = 32).
- The Formulator's replies are recorded, not live calls. The loop runs them, but nothing in the round shows the model
  reacting to a measurement it did not expect.
- One family only (`gemm_basic`). The warp-specialized `tcgen05` family does not build on `sm_121`; split-K and
  attention are not started.
- The regime probe and the optimism record run, but the round did not use them to change a proposal.

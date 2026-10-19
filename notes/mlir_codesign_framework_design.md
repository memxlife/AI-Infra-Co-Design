# Templates in MLIR: One Ladder for Hardware–Software Co-Design

**Design document, v0.2 · October 18, 2026**
*Li Shang, drafted with Claude. Follows `describing_templates.md` (console round 3) and `hardware_calibration_design.md`. Companion to `papers/learning_to_formulate/paper.tex` §2–§5. Written to be executed by Claude Code in stages; §8 gives the order.*

---

## 0. Why

Today the flow holds a template in three places that nothing keeps consistent. The contract is a JSON object (`agentic/schema/contract.json`) whose `program` field is prose and whose constraints are strings the solver bridge parses; the chip is a second JSON object (`chip.json`) with sixty calibrated fields and a `provenance` block; and the executable meaning of a design lives in a Python interpreter inside the checker, which is not the code that would run on a GPU. The solver reads the first, the cost model reads the second, the verifier reads the third, and the agent writes all three.

MLIR is a framework for exactly this situation: several levels of abstraction, each a legal program, with passes between them. The software stack already lives in it (Triton, TileLang, IREE and XLA-GPU all lower through MLIR dialects), and the hardware world has the same infrastructure in CIRCT. What neither side has is the thing this project is about: an object that says what the software may assume about the hardware, what the solver may choose, and what the verifier must prove, and that both sides read.

This document proposes to make the **template an MLIR object**, with three small dialects:

| Dialect | What it holds | Replaces |
|---|---|---|
| `tmpl` | the template tuple (O, X, C, Pre, P, π, Σ, Ĵ, Γ) as operations with regions | `contract.json` and the Python template classes |
| `hwm` | the hardware model library: components as ops with configuration attributes and performance/power/area model functions, with provenance and calibration range | `chip.json` and `HardwareContract` |
| `evm` | the event model the certificate's relation and rank are stated over | the checker's `EventModel` in Python |

and three passes over them: **export** (template → MILP), **decode** (solver assignment → `gpu`-dialect program), **verify** (template → SMT obligations, including K1–K3). Everything else — the ladder below `gpu` dialect to PTX, the chip-side ladder in CIRCT, the parser, printer and pass manager — is reused.

**The organizing idea: two levels for the hardware, one contract between them.** The chip appears in the flow at two different heights, and keeping them apart is what makes the framework simple.

- *Microarchitecture configuration* — shared-memory capacity, bank count, register-file size, queue depth, tensor-unit shape — is a set of integers from small menus. It belongs at the **operator-compiler rung (tensor program, L3), beside tiling**, because that is where it binds: a capacity and a tile size are the same kind of object and appear together in one inequality (C1). The only difference between a software knob and a hardware knob is which side of the inequality the solver is allowed to move.
- *Microarchitecture implementation* — the RTL — sits at the **bottom of the ladder, in CIRCT**, below PTX on the software side. It changes only when a chip is generated, never inside the search loop.
- The **PPA model library (`hwm`) is the contract between the two**: a configuration is an argument to the model, and the RTL is what the model is calibrated against. Nothing else crosses.

So the ladder has two walls, not one. The upper wall is the model library, and the solver moves it. The lower wall is the RTL, and it moves only when a new chip is built. Verification follows the same split: the template's obligations stop at the model (the design computes the workload *given* the model's semantics and limits); whether the RTL implements the model is a separate hardware-verification obligation, in the same way ptxas sits below the software proofs. Figure 1 shows the two walls.

```
                 software ladder                          hardware
   L2  linalg on tensor  (graph)                            │
   L3  tiled linalg / scf / memref  (tensor program)  ◄──►  hwm: microarchitecture CONFIGURATION
         tile sizes, slots, prefetch  ─── C1 ───  smem bytes, banks, regs, queue depth   (solver moves this wall)
   L4  vector / gpu / nvgpu           ─ reads ─►  hwm model functions (cycles, energy, area)
   L4  llvm → PTX                                           │   calibrated against
   ─── ptxas → SASS (closed) ───────────────────────────────┼───────────────────────────────
                                                  CIRCT: firrtl / hw / comb / seq  = microarchitecture IMPLEMENTATION (RTL)
                                                           (moves only when a chip is generated)
```
*Figure 1. Configuration lives at the operator-compiler rung; implementation lives below the line; the model is the only thing that crosses.*

The claim is modest: MLIR replaces the data structures, not the solver, the verifier or the cost model. Section 6 says what remains to be built.

---

## 1. The software ladder, and two policies over it

The six software layers of the lecture are rungs of one MLIR ladder:

| Lecture layer | Dialects | What is concrete here | Decisions made here |
|---|---|---|---|
| Graph (L2) | `linalg` on `tensor` | operators as structured ops; tensors are SSA values, the graph is use–def | fusion boundaries, horizontal fusion, pattern replacement, layouts |
| Tensor program (L3) | tiled `linalg`, `scf`/`affine`, `memref` **+ `hwm` configuration** | tile loops explicit; buffers in memory spaces; **the chip's capacities, counts and depths as attributes beside them** | tile sizes, loop order, placement, buffer reuse, unrolling; **in co-design, the hardware configuration the tiles must fit** |
| Hardware IR, upper (L4) | `vector`, `gpu`, `nvgpu` | blocks, threads, shared memory, async copies, tensor-core ops | thread/block binding, vector width, instruction choice, pipeline depth, swizzle |
| Hardware IR, exit (L4) | `llvm` → LLVM IR → PTX | — | instruction selection |
| below MLIR | `ptxas` → SASS | — | registers, instruction scheduling (closed) |

**Triton and TileLang are two policies over the same knobs.** Both compile through this ladder; they differ only in *who sets each attribute*:

| Knob (Part 1) | MLIR attribute / op | Triton | TileLang | This framework |
|---|---|---|---|---|
| Tiling | `transform.structured.tile_using_for` sizes; `BLOCK_*` constexpr | programmer (constexpr), `autotune` list | programmer | **solver**, from the template's menu |
| Memory placement | `memref` memory space; `gpu.alloc` shared; TMEM | compiler layout pass | programmer (`T.alloc_shared`, `alloc_fragment`) | solver / frozen by template |
| Loop permutation | `transform.structured.interchange` | compiler | programmer | solver |
| Fusion | `linalg` fusion / epilogue region | Inductor above Triton | programmer | graph-level template |
| Unrolling | `scf` unroll attribute, `num_stages` | compiler, `num_stages` | programmer (`T.Pipelined(num_stages)`) | solver (S, P) |
| Thread/block mapping | `gpu.launch` dims, `triton_gpu` layouts, warp roles | compiler (one program ≈ one block) | programmer (`T.Parallel`, warp specialization) | frozen (n_t) or solver |
| Synchronization | `gpu.barrier`, `nvgpu.device_async_wait`, mbarrier ops | compiler | programmer | solver (β), frozen (leading barrier) |

So the design space the framework searches is not new: it is the set of attributes these two compilers already set, by heuristic or by hand. What the template adds is a declared menu for each, a constraint set over them, and a proof obligation attached to each choice.

---

## 2. The hardware side: configuration at the tile level, implementation below

The chip enters the search loop as **performance, power and area models of components**, parameterized by a configuration, which is what the solver and the cost model consume. The configuration is a handful of integers and lives at the operator-compiler rung next to the tiling decisions (§0, Figure 1). The RTL (CIRCT's `firrtl`/`hw` dialects) is the implementation of that configuration: the ground truth the models are calibrated against, referenced from the `hwm` chip module, never embedded in it and never visible to the solver.

### 2.1 The `hwm` dialect

One operation per component type. Attributes carry the configuration; model functions are regions over the configuration attributes that return a cost; provenance says where each number came from.

```mlir
hwm.chip @gb10_model {
  hwm.sm @sm {
    count = 48 : i32,
    schedulers = 4 : i32, issue_width = 1 : i32,
    fp32_flops_per_s = 1.31e12 : f64,          // per SM, measured
    regs_per_sm = 65536 : i32, regs_per_thread_max = 255 : i32,
    model.perf = @sm_cycles,                     // region: (ops, threads) -> cycles
    model.area_mm2 = 0.0 : f64,                  // not modeled on a fixed chip
    provenance = #hwm.prov<kind = "measured", source = "calibration/spark-6951/raw/b17.json",
                           range = {warps = [1, 64]}, ci95 = 0.02>
  }
  hwm.shared_memory @smem {
    bytes_per_block = 101376 : i32, banks = 32 : i32, bank_width_B = 4 : i32,
    model.perf = @smem_access_cycles,            // region: (stride, warps) -> cycles, bank-conflict aware
    model.energy_pJ = @smem_energy,
    provenance = #hwm.prov<kind = "measured", source = "...b6.json, b26.json", range = {stride = [1, 128]}>
  }
  hwm.dram @dram {
    bytes_per_s = 2.35e11 : f64, channels_hash_bits = 7 : i32,
    latency_ns = 580 : i32,
    model.perf = @dram_time,                     // region: (bytes, warps_in_flight) -> seconds
    provenance = #hwm.prov<kind = "measured", source = "...b4.json, b18.json", range = {bytes = [1e6, 1e10]}>
  }
  hwm.copy_engine @tma { latency_dist = @cp_async_latency, groups_in_flight = 8 : i32,
    provenance = #hwm.prov<kind = "measured", source = "...b8.json, b28.json", note = "~200 cycles/4KB unexplained"> }
  hwm.barrier @bar { cycles = 40 : i32, provenance = #hwm.prov<kind = "illustrative"> }
  hwm.queue @lsu_queue { depth = 16 : i32, provenance = #hwm.prov<kind = "measured", source = "...b30.json"> }
}
```

Rules:

- **Every attribute the software may read is declared here, once.** The template's scope Γ (§3) is a list of `hwm` symbols; the `gpu`-dialect program produced by decode reads capacities and counts from the chip module, never from a literal. Today `96 KiB` appears in three files.
- **Provenance is mandatory** and has four kinds: `illustrative`, `analytical`, `fitted` (to synthesis or simulation points) and `measured` (with the raw record and a 95% interval). A pass can therefore answer "which numbers in this design's cost are measured?" mechanically, which is the evidence labeling the paper requires (proved / simulated / estimated / modeled / measured).
- **Calibration range is mandatory for fitted and measured models.** When the solver selects a configuration outside the range (a 128 KiB shared memory on a chip calibrated at 96), the `hwm-check-range` pass marks the cost `extrapolated`, and the orchestrator records the regime assumption as *observed, never proved* (paper §4.7, `hardware_calibration_design.md` §0).
- **Fidelity is a lowering.** Several `model.perf` regions may exist for one component at different fidelity (roofline bandwidth; a queueing model with `warps_in_flight`; the mechanistic simulator's stage model). Choosing a higher-fidelity model is a refinement step on the hardware side, done by a pass, with the same bookkeeping as a software refinement: which obligations it reopens, which proofs it keeps.
- **A configurable chip is a module with free attributes.** For co-design, `bytes_per_block` is not a literal but an SSA value bound by a chip-level template's decision (§5). The `chip.json` fields map one-to-one onto `hwm` attributes (Appendix A).

### 2.2 What `hwm` is not

It is not a simulator. The `model.perf` regions are closed-form or table-driven functions the cost evaluator pass calls; the mechanistic simulator (`mechanistic_simulator_design.md`) remains a separate evaluator backend that produces measurements, and those measurements calibrate `hwm`. It is also not RTL: when Ventus is the target, the `hwm` chip module carries a `rtl = @ventus_firrtl` reference into CIRCT for area/timing regeneration, nothing more.

### 2.3 The lower level: implementation in CIRCT, later

When an actual hardware implementation exists (Ventus today; a generated chip later), it is held below the line in CIRCT: Chisel emits FIRRTL, FIRRTL is a CIRCT dialect, and `hw`/`comb`/`seq` carry the RTL to Verilog. The `hwm` configuration becomes the parameter set of those modules (`SharedMemory(bytes = 98304, banks = 32)`), so a configuration the solver picks can be instantiated, synthesized, and its area and timing fed back into the `hwm` model as `fitted` provenance. Two things are deliberately **not** done in v0.1: the solver never reads RTL, and the template's obligations never descend into it. The proof that the RTL implements the model's semantics (the barrier axioms, the load-completion axiom, the queue's arbitration contract) is a hardware-verification obligation below the line, owned by the chip, in the same position as ptxas on the software side. That is the extension point when the RTL work begins.

---

## 3. The template as an MLIR object: the `tmpl` dialect

One operation per field of the tuple **T = (O, X, C, Pre, P, π, Σ, Ĵ, Γ)**, inside a `tmpl.template` container. Decisions are SSA values of a `!tmpl.decision` type so that constraints, cost and program all refer to the same symbol. Listing 1 of the paper, written out:

```mlir
tmpl.template @smem_tiling
    refines(level = 1 -> 2, object = "shared-memory slot: one Tm x Tk tile of A and one Tk x Tn tile of B") {

  // ---- Γ: scope. Facts nothing at this step can change. ----
  tmpl.scope {
    workload = #tmpl.workload<"C[M,N] += A[M,K] B[K,N]", dims = ["M","N","K"], min = 1, dtype = f32, elem_B = 4>
    chip     = [@gb10_model::@sm, @gb10_model::@smem, @gb10_model::@dram, @gb10_model::@bar]   // hwm symbols the program may read
    semantics = [#tmpl.axiom<"barrier: no thread passes until all arrive; smem writes before it are visible after it">,
                 #tmpl.axiom<"load: a tile load completes before the next barrier">]
  }

  // ---- X: decisions. searched | frozen | derived ----
  %Tm = tmpl.decision "Tm" searched menu [16, 32, 64, 128, 256] knob "tiling"
  %Tn = tmpl.decision "Tn" searched menu [16, 32, 64, 128, 256] knob "tiling"
  %Tk = tmpl.decision "Tk" searched menu [8, 16, 32, 64]        knob "tiling"
  %S  = tmpl.decision "S"  searched menu [1, 2, 3]              knob "memory placement"
  %P  = tmpl.decision "P"  searched menu [0, 1, 2]              knob "synchronization"
  %b  = tmpl.decision "beta" searched menu [0, 1]               knob "synchronization"
  %nt   = tmpl.decision "n_t"   frozen 256   meaning "threads per block"
  %amax = tmpl.decision "a_max" frozen 128   meaning "accumulator registers per thread"
  %grid = tmpl.decision "grid"  frozen #tmpl.rule<"ceil(M/Tm) x ceil(N/Tn)">
  %disj = tmpl.decision "tiles_disjoint" frozen true
  %grd  = tmpl.decision "guard"  frozen true  meaning "skip any index past the matrix edge"
  %lead = tmpl.decision "leading_barrier" frozen true
  %KT   = tmpl.derived "KT" = affine_map<()[K, Tk] -> (ceildiv(K, Tk))>
  %acc  = tmpl.derived "acc" = affine_map<()[Tm, Tn, nt] -> (Tm * Tn floordiv nt)>

  // ---- C: search constraints (read by the solver) ----
  tmpl.constraint "C1" resource  : S * 4 * (Tm*Tk + Tk*Tn) <= @gb10_model::@smem.bytes_per_block
  tmpl.constraint "C2" resource  : 1 <= Tm*Tn / nt <= amax
  tmpl.constraint "C3" correctness : P <= S - 2 + beta        discharges @lifetime

  // ---- Pre: preconditions (read by the verifier), each naming what discharges it ----
  tmpl.precondition @lifetime  : P <= S - 2 + beta                       by %C3
  tmpl.precondition @readiness : "a slot is read only after its load's barrier"   by %lead
  tmpl.precondition @coverage  : "every output element lies in exactly one tile"  by %grid
  tmpl.precondition @exact_once: "tiles are disjoint"                            by %disj
  tmpl.precondition @no_phantom: "no access outside [M,N,K]"                     by %grd

  // ---- Ĵ: cost, with its assumptions marked steer-only ----
  tmpl.cost minimize %T {
    %W     = ceildiv(Gm * Gn, @sm.count)
    %Tcomp = W * (2 * Tm * Tn * K / @sm.fp32_flops_per_s + KT * (1 + beta) * @bar.cycles / f_clk)
    %Tmem  = 4 * (Gm * Gn * K * (Tm + Tn) + M * N) / @dram.bytes_per_s
    %T     = select(P >= 1, max(%Tcomp, %Tmem), %Tcomp + %Tmem)
  } assumptions ["one block per SM at a time", "no L2 reuse", "tails cost as full tiles"]  // steer, never proved

  // ---- P: the program at level ℓ+1, decisions symbolic. Lowers to gpu dialect on decode. ----
  tmpl.program {
  ^bb0(%A: memref<?x?xf32>, %B: memref<?x?xf32>, %C: memref<?x?xf32>):
    gpu.launch blocks(%bi, %bj) in (Gm, Gn) threads(%t) in (nt) {
      %slots = tmpl.slots S of (memref<Tm x Tk x f32, 3>, memref<Tk x Tn x f32, 3>)
      scf.for %u = 0 to P      { tmpl.load_tile %u -> slot(%u mod S) }
      scf.for %p = 0 to KT {
        scf.if (%p + P < KT)   { tmpl.load_tile (%p + P) -> slot((%p + P) mod S) }
        gpu.barrier                                     // leading, frozen
        tmpl.compute_tile %C[%bi, %bj] += slot(%p mod S) guard %grd
        scf.if (beta == 1)     { gpu.barrier }          // trailing, searched
      }
    }
  }

  // ---- π: backward map ----
  tmpl.backward_map "forget grid, slots and schedule" -> @gemm_operator_level

  // ---- Σ: certificate schema over the event model ----
  tmpl.certificate {
    relation R : evm.holds(slot(u mod S), tile u) from evm.complete(load u) until evm.last_read(u)
    rank     ρ : evm.count(tiles_not_consumed)
    obligations [@coverage, @exact_once, @no_phantom] kind index    extents symbolic
    obligations [@readiness, @lifetime]               kind schedule model = #evm.model<warps = 2, phases = 5>
    obligation  @agreement                            kind replay
  }
}
```

Three properties the dialect enforces by construction, which the JSON schema cannot:

1. **A constraint and a precondition that are the same predicate are the same SSA value.** C3 *is* `@lifetime`; the `discharges`/`by` edges make K1 a graph property, and a searched decision that appears in a precondition with no discharging constraint is a verifier error at parse time.
2. **The program can only read declared scope.** `tmpl.program` may reference `hwm` symbols listed in `tmpl.scope` and decisions declared in the template. A literal `96 KiB` in the program region does not verify.
3. **Frozen decisions are decisions.** They have the same type as searched ones, so a later template can `tmpl.unfreeze %amax menu [64, 128, 256]` and the passes know which preconditions that reopens (the ones whose `by` edge points at it).

The agent's output is this text. The population (`paper.tex` §3.1) is a directory of `.mlir` files, each with its certification status and evidence as attributes.

---

## 4. The passes: how the two faces are read

### 4.1 `tmpl-export-milp`: the solver's face

Walks `tmpl.decision` (searched), `tmpl.constraint` and `tmpl.cost`; emits the MILP. Each menu becomes a one-hot block; products of decisions are expanded over the menus (exact, as today's `smem_search.py`); the cost's `max`/`select` become epigraph rows. Output: an LP file plus a **decoder map** (which binary corresponds to which decision value). The pass also emits the *exported predicate* Ĉ of paper §4.5 so that the translation check (Ĉ ⇒ C) can be run as an SMT query, which today is done by trust.

### 4.2 `tmpl-decode`: from assignment to program

Takes the solver's assignment, performs the **instance check** of paper §4.5 in order — in scope (every `hwm` symbol read has a value in the target chip), decodes cleanly (one value per menu), constraints hold (C1–C3 re-evaluated on the exact values, not through the encoding), frozen choices kept — then substitutes the values into `tmpl.program` and lowers `tmpl.*` ops to `gpu`/`nvgpu`/`memref` ops. The result is an ordinary MLIR module that the standard pipeline takes to PTX. The frozen choices are checked *again* after lowering by a small pattern (`gpu.barrier` precedes every `compute_tile`; the grid uses `ceildiv`), which is the "generated code keeps the frozen choices" check that today is a comment.

### 4.3 `tmpl-verify`: the verifier's face

Emits one SMT query per obligation:

- **Index obligations** (`coverage`, `exact_once`, `no_phantom`): quantify over symbolic extents M, N, K and tile indices; the hypothesis is the program's index expressions and the guard; the negated conclusion is the bug. These are the queries `smem_verify.py` builds by hand.
- **Schedule obligations** (`readiness`, `lifetime`): instantiate the `evm` model (§6.1) with the template's decisions symbolic, add program order and the barrier axioms from `tmpl.scope.semantics`, and ask for an interleaving in which a slot is read before completion or overwritten before last read.
- **K1**: for every precondition with `by %Cj`, the query `C ∧ frozen ∧ ¬Pre` must be unsat; for preconditions discharged by a frozen decision, the query is over the frozen value only. One query per edge.
- **K2**: the family lemma: `Pre ⇒ obligations` under Σ, over the declared family (menus) and the event model.
- **K3**: `∃ x ∈ menus : C(x)` for the symbolic workload — a one-line SAT query.

Counterexamples are returned as `evm` traces and replayed by the interpreter pass (§6.1); an unsat result is stored as a `tmpl.proof` attribute with the query hash and the premises (the `hwm` symbols and semantics axioms it used), which is what incremental reuse (paper §4.6) keys on.

### 4.4 Composition down the ladder

A refinement is a pair (`tmpl.template` at level ℓ+1, `tmpl.backward_map` to the template at ℓ). The `tmpl-compose` pass checks feasibility preservation (C_{ℓ+1} ⇒ C_ℓ ∘ π) as an SMT query and chains the certificates. Because every template's program is a region at a real rung, the composed proof talks about the same program the decoder lowers; the Python interpreter and the real pipeline no longer diverge.

---

## 5. Co-design: when a hardware attribute becomes a decision

A co-design template is a `tmpl.template` at the **same rung as the tiling template** whose decisions bind `hwm` attributes. It is not a lower level; it is the hardware half of the tensor-program level, which is why the two share one MILP without any cross-level machinery:

```mlir
tmpl.template @smem_capacity refines(level = 1 -> 2, object = "shared-memory block configuration") {   // same rung as @smem_tiling
  tmpl.scope { chip = [@chip::@smem], area_budget_mm2 = 1.11 }
  %cap   = tmpl.decision "smem_bytes" searched menu [49152, 65536, 98304, 131072]   binds @chip::@smem.bytes_per_block
  %banks = tmpl.decision "banks"      searched menu [16, 32, 64]                    binds @chip::@smem.banks
  tmpl.constraint "A1" resource : hwm.area(@chip::@smem{bytes=%cap, banks=%banks}) + area_rest <= area_budget
  tmpl.cost minimize %T_total { ... }     // shared with the software templates active at this step
}
```

When this template is active, every software template whose scope lists `@chip::@smem` sees `bytes_per_block` as a variable, and the joint MILP contains C1 as `S*4*(TmTk+TkTn) <= smem_bytes` with both sides decided: the product is expanded over both menus. This is the paper's `(n_c, θ_c)` and boundary variable `b` made concrete, and it is the mechanism by which "hardware moves the cliffs": the wall C1 is now a function of a hardware decision.

Two obligations attach to a chip-level decision, and the passes make them visible:

- **Range.** If the selected `bytes_per_block` is outside every `hwm` model's calibration range, `hwm-check-range` labels the cost `extrapolated` and the regime assumption `observed`. On the GB10 this always happens (the card cannot change); on Ventus it does not, because the model is fitted to synthesis points across the menu.
- **Reopened proofs.** A software precondition that was discharged by a frozen choice depending on the moved attribute is re-proved; the dependency is explicit in the `by` edges and the `tmpl.proof` premises, so `tmpl-verify` recomputes only those.

The ladder slide of the lecture should show this picture: the software templates are rungs on the left, the `hwm` configuration is the upper wall at the tensor-program rung, a co-design template moves part of that wall, and the RTL is the lower wall, below the line, that only a new chip moves.

---

## 6. What MLIR does not provide, and what we build

| Missing | Why it matters | What we build |
|---|---|---|
| **Execution state** for the certificate | R and ρ are statements about events (loads completing, reads, barriers) in an interleaving; MLIR has no notion of a run | `evm` dialect (§6.1): a tiny event model with warps, phases, loads, barriers, reads; an interpreter pass that replays a trace; the SMT encoding of its interleavings |
| **Search** | passes are heuristics; nothing enumerates or optimizes | the solver bridge (`tmpl-export-milp` → HiGHS/CBC; `tmpl-verify` → Z3). Unchanged from today, reading MLIR instead of JSON |
| **Correctness of lowerings** | `tmpl-decode` and the standard `gpu` → PTX pipeline are trusted | keep the trusted base explicit (paper §4.5): the decoder is small and reviewed; the post-lowering frozen-choice pattern check; optionally translation validation of `tmpl-decode` per instance (Alive2-style) |
| **Cost evaluation** | `hwm` carries model regions, but nothing calls them | `hwm-evaluate` pass: walks a decoded `gpu` program and the `hwm` chip, applies the roofline / queueing / stage models, returns a `measurement` artifact labeled by provenance (`modeled`), beside the real measurement from the Spark or the simulator |
| **Blackwell-class ops** upstream | `nvgpu` covers Hopper (WGMMA, TMA); tcgen05/TMEM are partial | target Hopper-class and Ventus first; for Blackwell, decode to CuTe-DSL text as a second backend |
| **Numerical contracts** | obligations are exact-arithmetic; emulated exp, FP8 scaling need error bounds | out of scope for v0.1; reserve `tmpl.precondition ... kind numeric` with a bound attribute, discharged by a separate checker |

### 6.1 The `evm` dialect

The checker's `EventModel(warps=2, phases=5)` becomes a dialect so that the certificate is written in the same file as the program:

```mlir
evm.model @two_warp attributes { warps = 2, phases = 5 } {
  evm.event @load(u, slot)      // issue of a tile load
  evm.event @complete(u, slot)  // its completion; axiom: complete(u) happens-before the next barrier
  evm.event @barrier(p)         // all warps arrive; axiom: writes before are visible after
  evm.event @read(w, u, slot)   // warp w reads tile u from slot
  evm.order program : load(p+P) < barrier(p) < read(*, p) < [barrier_trailing(p)]
  evm.order axioms  : complete(u) < barrier(after load(u));  write visibility across barrier
}
```

`tmpl-verify` instantiates this with the template's decisions symbolic and asks Z3 for an interleaving violating `R`; the replay pass executes the returned trace on the `tmpl.program` region with concrete small extents and reports the wrong term (`C[0,1] gets A[0,2]·B[4,1]`). The two-warp sufficiency argument (any race is one writer and one reader) is stated as a comment in v0.1 and is a candidate for a mechanized proof later.

---

## 7. What this changes for the agent, the solver and the paper

- **The agent writes one artifact in a language with a parser and a verifier.** The repair case in the paper (§2.4) becomes a diff between two `.mlir` files; the "scope rule" is enforced by the dialect (a literal where a scope symbol belongs does not parse). The console's three lenses (`describing_templates.md`) render from the same ops.
- **The solver's input is generated, not transcribed**, and the translation check Ĉ ⇒ C is a query, not an assumption.
- **The verifier's program is the program that runs.** The evaluator's measurement and the verifier's obligation refer to the same `gpu`-dialect module; the "12 obligations in 25 ms" gate runs on the decoded design.
- **Evidence labels are mechanical.** Every cost term's provenance (`illustrative` / `fitted` / `measured`, in or out of range) is readable from the `hwm` attributes, which is what §7 of the paper needs for the two-hardware-model study.
- **The population is a directory**, and template reuse is `tmpl.scope` matching plus a `tmpl-compose` check.

---

## 8. Build order

Each stage is a day or two of Claude Code work and ends with a check against the current Python results, so nothing regresses.

1. **`hwm` for the two chips.** Write `gb10_model.mlir` and `ventus_ref.mlir` from `chip.json` and the calibration records; a Python loader that round-trips them to the current `HardwareContract`. Check: identical cost-model outputs on the 722 designs.
2. **`tmpl` for Listing 1.** Write `smem_tiling.mlir`; `tmpl-export-milp` producing the same 93-variable / 248-constraint MILP; HiGHS returns the same optimum on the three GEMM shapes.
3. **`tmpl-decode` to `gpu` dialect**, lower through `nvgpu` to PTX, run on the Spark. Check: the kernel computes C = AB; measured time beside the modeled 53.1 µs (this is the first real measurement of the paper's anchor design).
4. **`evm` + `tmpl-verify`.** Reproduce the 24-combination lifetime table and the six injected faults; proofs stored with premises.
5. **Co-design template** (`smem_capacity`) on the Ventus model, at the tiling rung; joint MILP; range labeling. Check: the ladder baseline → joint for study 1 reproduces the report numbers.
6. **Register-tiling template** written by the agent in `tmpl` syntax (paper §2.4 experiment), certified by stage 4's passes.
7. **Below the line, when the RTL work starts:** Ventus FIRRTL into CIRCT; `hwm` configurations as module parameters; synthesis-derived area/timing written back as `fitted` provenance; the model-vs-RTL semantics obligation stated (not yet discharged).

Risks, in order: engineering weight of an MLIR build (mitigate: Python bindings and a textual dialect first, C++ only for the lowering to `gpu`); `nvgpu` coverage on the Spark's Blackwell-class SM (mitigate: CuTe-DSL backend for decode); the certificate's event model staying small enough for exhaustive SMT (today's two warps, five phases; grows with new objects).

---

## Appendix A. `chip.json` → `hwm` attribute map (excerpt)

| `chip.json` field | `hwm` op · attribute | provenance kind today |
|---|---|---|
| `sms` | `hwm.sm.count` | measured (B1) |
| `smem_bytes_per_block` | `hwm.shared_memory.bytes_per_block` | measured / declared |
| `dram_bytes_per_s`, `dram_bytes_per_s_under_cpu_load` | `hwm.dram.bytes_per_s`, `.under_cpu_load` | measured (B4, B18) |
| `fp32_flops_per_s_per_sm`, `tensor_flops_per_s_per_sm` | `hwm.sm.fp32_flops_per_s`, `.tensor_flops_per_s` | measured (B17, B31–34) |
| `barrier_s`, `mbarrier_s` | `hwm.barrier.cycles`, `hwm.mbarrier.cycles` | measured / illustrative |
| `cp_async_latency_dist`, `cp_async_groups_in_flight`, `tma_2d_latency_s` | `hwm.copy_engine.*` | measured (B8, B15, B28, B35–36); open gap noted |
| `l2_*` (line, sets, ways, slices, hash bits, latency near/far) | `hwm.l2.*` | measured (B4, B13) |
| `warps_to_hide_dram_latency`, `per_sm_outstanding_requests`, `bandwidth_vs_warps` | `hwm.dram.model.perf` inputs | measured (B18–B21) |
| `spill_penalty_curve`, `local_mem_latency_ns` | `hwm.register_file.spill_model` | measured (B9, B19) |
| `block_dispatch_order`, `block_dispatch_rate_per_s`, `wave_staircase` | `hwm.scheduler.*` | measured (B22–B24) |
| `fixed` (list of fields that cannot change on this chip) | `hwm.chip` attribute `fixed = [...]`; a chip-level template may not `bind` these | — |
| `provenance` | `#hwm.prov` on every op | — |

## Appendix B. `contract.json` → `tmpl` op map

| `contract.json` field | `tmpl` construct |
|---|---|
| `level`, `object`, `template`, `parent` | `tmpl.template` attributes `refines`, `object`, `parent` |
| `scope.workload`, `scope.chip`, `scope.semantics` | `tmpl.scope` |
| `decisions` (with `searched`/`frozen`/`derived`) | `tmpl.decision`, `tmpl.derived` |
| `constraints[]` (`kind`, `expr`, `meaning`) | `tmpl.constraint` |
| `preconditions[]` (`discharged_by`) | `tmpl.precondition ... by` |
| `cost` (`objective`, `terms`, `assumptions`) | `tmpl.cost` region + `assumptions` |
| `program` (prose today) | `tmpl.program` region |
| `backward_map` | `tmpl.backward_map` |
| `certificate_schema` (`relation`, `rank`, `obligations`) | `tmpl.certificate` over `evm` |
| `structure`, `signature`, `rationale` | attributes on `tmpl.template` (rationale is never parsed) |
| `status`, `certificate`, `provenance`, `methods` | `tmpl.proof` attributes written by `tmpl-verify` |

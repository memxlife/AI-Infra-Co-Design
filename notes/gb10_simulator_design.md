# Design of the GB10 instruction-level simulator

*Oct 20 2026. This note describes the code in `agentic/gb10/sim/` as it stands after commit 8ddfa04, the last change to
it. Its constants are the `sim` block of `calibration/spark-6951/gb10.json`. The probes that measured them are in
`agentic/tilelang/microbench/`. Every number below comes from that code, that file, the saved results in `runs/gb10/`,
or the project's running write-up `notes/gb10_model.md`. Each number says where it came from. Numbers marked "re-run for
this note" were produced on a laptop from the current code and the saved machine code; nothing was run on the GPU.*

## 1. What it is for

The simulator takes a GPU kernel after compilation and plays its machine code on a model of one SM of the GB10. Then it
scales that SM up to the whole chip. The GB10 is the chip in the NVIDIA DGX Spark. Its GPU has 48 SMs and code for it is
compiled for `sm_121a`. The kernels are TileLang's matrix multiplies (C = A × B with fp16 inputs, fp32 accumulation and
fp16 output) and two relatives: split-K matrix multiply and matrix-vector multiply (GEMV).

It answers two questions:

1. **Which candidate is fastest?** Given several compiled kernels for one shape, put them in the order the GPU would.
2. **What limits this kernel?** The tensor pipe, shared memory, the path to the L2, DRAM, the store path, or waiting (for
   data, at a barrier, or for a hand-off between warps).

What counts as good: rank candidates the way the GPU does and name the unit the GPU really spends its time on. Absolute
error is reported, but it is not the goal.

### Words used in this note

| Word | Meaning |
|---|---|
| SM | Streaming multiprocessor, one of the GPU's 48 cores. It holds several blocks at once. |
| Sub-partition | A quarter of an SM. Each has its own warp scheduler, tensor pipe, FMA unit, ALU unit and special function unit. |
| Warp | 32 threads that issue each instruction together. The scheduler picks warps, not threads. |
| Block | A group of warps that run on one SM and share its shared memory. TileLang gives each block one tile of C. |
| Tile, k-tile | A block computes a block_M × block_N tile of C. It walks K in steps of block_K; each step is a k-tile. |
| Stages | How many k-tiles a block keeps in shared memory at once (the pipeline depth). |
| SASS | The GPU's machine code, as printed by NVIDIA's disassembler `nvdisasm`. |
| mma | A warp-wide matrix multiply-add on the tensor pipe (`HMMA` in SASS for fp16). One m16n8k16 mma does 2048 multiply-adds. |
| ldmatrix, stmatrix | Warp-wide load from / store to shared memory in the layout mma wants (`LDSM`, `STSM`). |
| cp.async | A copy from global memory straight into shared memory (`LDGSTS`). Copies are grouped by a commit (`LDGDEPBAR`) and waited on by group (`DEPBAR`). |
| TMA | The SM's tensor memory accelerator. One thread asks it to move a whole box of a matrix (`UTMALDG` to load, `UTMASTG` to store). |
| Warp specialization | Some warps (producers) only issue TMA loads; the rest (consumers) only compute. They hand off k-tiles through mbarriers. |
| mbarrier | A barrier object in shared memory. Warps arrive on it and wait for its phase to flip (`SYNCS.*` in SASS). |
| Dependency counter | Six per warp. A slow instruction raises one until its result is ready (or its inputs are read). A later instruction lists the counters it must wait for. The compiler sets all of this. |
| L2, DRAM | The chip's 24 MB last-level cache, and the LPDDR5X memory it shares with the CPU. |
| SM pair | Two neighbouring SMs (NVIDIA calls it a TPC). On the GB10 they share one store path. |
| Warm, cold | Warm: the kernel is timed in a back-to-back loop, so data that fits stays in the L2. Cold: the L2 is flushed before each run. |
| Probe | A small CUDA kernel that isolates one unit and measures it. |
| Ablation | A real TileLang kernel with one part removed: `noload` (no copies, mma only) or `nocompute` (copies only). |

### Why instruction level

The project also has an analytical model, `agentic/gb10/model.py`. Per k-tile it works out how long each unit needs
(tensor pipe, shared memory, copy port, latency) and joins compute and loads with a measured rule,
(compute^p + loads^p)^(1/p), with p about 2 (median 1.97 from the ablation, range 1.5 to 6, per
`notes/gb10_model.md`). It is fast and ranks large shapes well. But it works on totals, so it cannot see three things:

- **How instructions interleave.** The analytical model needs a measured tensor efficiency for a GEMM loop (83 % with one
  warp per sub-partition, 90 % with two, from the ablation). In the simulator that number falls out of issuing the real
  loop: ldmatrix and mma from the same warps, 16-cycle gaps between mma, barriers between k-tiles.
- **Register-dependency stalls.** In TileLang's cp.async loop each warp issues its copies, then updates the addresses
  for the next k-tile. The update must wait until the copies have read their address registers, and that happens only
  when the SM's load unit reaches them. The SASS shows it: the copy sets a read counter and the update waits on it. ncu
  (NVIDIA's profiler) on a whole kernel against its mma-only version, one block: long-scoreboard stalls 0.80 against
  0.00, tensor pipe active 68 % against 84 %. Modelled from the machine code, the loss per k-tile for one block came out
  320 / 160 / 96 cycles for three tile shapes; measured 390 / 174 / 79.
- **Barrier and hand-off timing.** In a 2-stage TMA pipeline each k-tile can wait on a chain: consumers free a stage,
  134 cycles later the producer sees it, the TMA fetch takes its latency, the bytes land, 134 cycles later the consumers
  see them. Whether that chain is hidden depends on the stage count and the latencies, not on totals. The analytical
  model calls the M=128 decode TMA pick DRAM-bound, yet after a flush it moves only 138 GB/s and waits on load latency;
  `tests/test_gb10_model.py` records this and names the simulator as the tool for that case.

### What it does not do

- It computes no data. The only values it tracks are warp-wide integers (loop counters, thread and block indices) so it
  can decide branches. Lane 0 stands for the whole warp; warps never split at a branch.
- It simulates one SM in detail: the busiest one. The rest of the chip enters only as this SM's share of chip-wide
  limits (L2, DRAM, stores, TMA).
- Latencies are fixed numbers. Real latencies vary; the simulator has no spread.
- It does not model the instruction cache or register bank conflicts (Section 8 says why that is safe here).
- Only the instructions TileLang's GEMM and GEMV code needs have measured costs. Anything else gets an assumed 20-cycle
  result latency and no unit time.
- It turns cycles into time at one clock rate, 2.54 GHz.

There is an older, separate simulator in `agentic/sim/` (`warp.py`) that is fed instruction counts rather than SASS.
`notes/sim_gaps.md` logs that one's gaps. It is not described here and nothing here depends on it.

## 2. Inputs and outputs

### Inputs

| Input | What it is | Where it comes from |
|---|---|---|
| SASS file | `nvdisasm -hex` output of the kernel's cubin. `-hex` prints both 64-bit words of every instruction, so the compiler's control bits can be read. | The pipeline compiles with `nvcc ... -gencode arch=compute_121a,code=sm_121a -cubin`, then runs `nvdisasm -hex` (`compile_one` in `agentic/decide/study/run.py`). |
| Kernel name | Which `.text.<name>` section to read. | Default `matmul_kernel`; the pipeline takes it from the SASS. |
| Tile config `h` | `block_M`, `block_N`, `block_K`; `num_stages`; `threads` per block (on the TMA path: consumer threads only); `swizzle`, the block launch order (0 = row-major; n = TileLang's panels of n tile rows, walked back and forth); optional `smem_layout` (`"linear"` turns on shared-memory bank conflicts). | The candidate's configuration. Note `swizzle` here is launch order, not a shared-memory layout. |
| Copy path | cp.async, or warp-specialized TMA. | Chosen by which entry point you call (Section 10). |
| Shape | M, N, K. | |
| Registers | Registers per thread. Used only to work out blocks per SM on the cp.async path. | `cuobjdump -res-usage`. |
| Warm or cold | `warm=True`: back-to-back runs. `warm=False`: after an L2 flush. | The timing protocol you want to predict. |
| Split-K | `split` (K is cut into this many pieces, each a block) and `atomic_c` (C is added with atomics). | |
| Shared memory | Bytes per block. | Default `num_stages × (block_M + block_N) × block_K × 2`. |

### Outputs

`simulate_kernel` (cp.async path) returns a dict. Real output for the 2048³ kernel with 128×128×32 tiles, 3 stages,
128 threads and 202 registers (re-run for this note, numbers rounded; it took 1.2 s):

```python
{"us": 182.8, "tiles": 256, "blocks_per_sm": 2, "waves": 3, "dram_MB": 16.8,
 "parts": [{"counts": {}, "blocks": 256, "per_sm": 2, "miss_frac": 0.062, "dram_GBs": 140.9, "us": 180.22,
            "busy": {"tensor": 3.44, "smem": 0.32, "fill": 0.72, "alu": 0.11, "fma": 0.04,
                     "lsu": 0.16, "dram": 0.66, "stg": 0.13}}]}
```

`us` is the predicted kernel time including launch (measured warm: 217.2 µs). `miss_frac` is the share of copies that
miss the L2. `dram_GBs` is read-miss bytes over the DRAM time, which also includes writing C. `busy` is how many cycles
each unit was in use, over the run's cycles.

`kernel_us` (TMA path, in `ws_validate.py`) returns just the time in µs. `SM.run()` underneath both returns
`{"cycles", "issued", "HMMA", "LDSM", "LDGSTS", "BAR", "busy"}`, plus `STSM`, `UTMASTG` and `LDG` counts when the code
has them. On the TMA path `busy` is in raw cycles.

**How to read `busy` for the limit.** The code does not print a one-word bottleneck. You read it from `busy`. Tensor,
FMA, ALU and SFU are per sub-partition, so they add up to at most 4.0. Everything else (shared-memory array `smem`, L2
return path `fill`, address stage `lsu`, DRAM queue `dram`, store port `stg`, atomics `red`, TMA engine `tma`) is per SM
and tops out at 1.0. The unit closest to full is the limit. If none is close, the kernel is waiting on latency. Two real
runs:

| Kernel (128×128×32, 3 stages) | Simulated | Measured | Tensor | L2 return | DRAM | Reading |
|---|---:|---:|---:|---:|---:|---|
| 2048³ | 182.8 µs | 217.2 µs | 3.44 / 4 = 86 % | 72 % | 66 % | tensor-bound |
| 4096³ | 1976.3 µs | 2015.4 µs | 2.30 / 4 = 58 % | 52 % | 96 % | DRAM-bound |

To find what a waiting kernel waits for, the TMA model keeps a time-stamped log of every wait passed, arrive, release
and TMA box (`WSSM.log`); `notes/gb10_model.md` used it to trace the per-k-tile chain of the M=128 pick.

## 3. Structure

| File | Lines | Role |
|---|---:|---|
| `sass.py` | 82 | Parse `nvdisasm -hex` into instructions with control bits; find the main k-loop. |
| `sm.py` | 511 | The SM: warps, issue, dependency counters, units, memory paths, barriers, branches, block replacement, chip shares. |
| `ws.py` | 185 | `WSSM`, a subclass of `SM` for warp-specialized TMA kernels: roles, TMA engine, mbarriers, register hand-off. |
| `kernel.py` | 148 | The whole kernel on the cp.async path: blocks per SM, DRAM traffic, warm or cold, launch. |
| `ws_validate.py` | 70 | The whole kernel on the TMA path (`kernel_us`), and its check against the TMA ablation. |
| `audit_values.py` | 80 | Checks that branches decided by computed values run every mma exactly once. |
| `validate.py`, `shapes.py`, `epilogue.py`, `layout.py`, `gemv.py`, `decode_sweep.py`, `schedtest.py` | | Drivers: each runs one measured set and prints simulated against measured. |

```
  config h, shape, regs, warm/cold                           SASS text (nvdisasm -hex)
               |                                                       |
               v                                                       v
  kernel.simulate_kernel / ws_validate.kernel_us             sass.parse -> list of Ins
    blocks per SM (occupancy)                                  opcode, operands, predicate,
    model.dram_split_kslice: L2 misses per wave                stall, yield, counters, wait mask,
    wave_dram: DRAM seconds -> miss fraction, DRAM queue rate  branch target; main_loop()
               |                                                       |
               +--------------------------+----------------------------+
                                          v
           sm.SM (cp.async)  or  ws.WSSM (warp-specialized TMA): one SM, its share of the grid
  +----------------------------------------------------------------------------------------+
  | each cycle, each of 4 sub-partitions picks one eligible warp and issues its instruction |
  |   per sub-partition: tensor pipe, FMA unit, ALU unit, SFU, memory-instruction slot      |
  |   per SM: shared-memory array, address stage, L2 return path, DRAM queue, L1 path,      |
  |           store port (shared with the partner SM), store queue, TMA engine (WSSM)       |
  | time-ordered events:                                                                    |
  |   sb        lower a dependency counter                                                  |
  |   write     a copy's data reached the SM: write it into the shared-memory array         |
  |   landed    a copy is in shared memory: count down its group                            |
  |   bar       release the warps of a block barrier                                        |
  |   tmawrite, tmalanded, wake    TMA boxes and mbarrier hand-offs (WSSM)                 |
  | a block exits -> the next block starts on its slot; the chip shares are recomputed      |
  +----------------------------------------------------------------------------------------+
                                          |
                                          v
          cycles, instruction counts, busy cycles per unit;  time (us) = 2.6 + cycles / 2540
```

One cp.async copy, end to end: issue (memory slot) → address stage → L2 (350 cycles) or DRAM (1560 cycles plus a slot
in the DRAM queue) → L2 return path → `write` event → shared-memory array (4 cycles) → `landed` event → when the last
copy of its group lands, the group's counter drops → the warp's `DEPBAR` lets it pass.

One TMA box: issue → TMA engine → lands after 440 cycles (L2) or 1170 plus the DRAM queue → `tmawrite` → shared-memory
array → `tmalanded` → when every box of the k-tile is in, a `wake` event 134 cycles later → the consumers' mbarrier wait
passes.

## 4. The SM model (`sm.py`)

### 4.1 Reading the machine code (`sass.py`)

Each instruction is 128 bits. `nvdisasm -hex` prints it as two 64-bit words. The compiler's scheduling control sits in
bits 41-61 of the second word:

```python
def decode_ctrl(hi: int) -> dict:
    c = hi >> 41
    stall = c & 0xF; yld = (c >> 4) & 1; wb = (c >> 5) & 7; rb = (c >> 8) & 7; wait = (c >> 11) & 0x3F
    return {"stall": stall, "yld": yld, "wbar": None if wb == 7 else wb, "rbar": None if rb == 7 else rb, "wait": wait}
```

- **stall**: cycles before this warp may issue again.
- **yield**: a hint to let another warp go. Parsed, but the default scheduler ignores it.
- **write counter** (0-5, or none): raised until this instruction's result is ready.
- **read counter** (0-5, or none): raised until this instruction has read its source registers.
- **wait mask**: the counters that must be zero before this instruction issues.
- The four register-reuse bits are not used.

A real pair of lines from the 128×128×32 kernel's main loop:

```
/*1120*/  HMMA.16816.F32 R136, R44, R36, R136 ;   /* 0x000000242c88723c */
                                                    /* 0x002fde0000001888 */
```

The second word decodes to stall 15, yield 0, no counters, wait mask `0b000010`: this mma waits for counter 1 (set by
the ldmatrix that loaded R44), then holds its warp 15 cycles. A `NOP` with stall 1 follows, so mma issue every 16 cycles.

The parser also keeps the predicate (`@P0`, `@!PT`, `@UP0`), the opcode with modifiers (`HMMA.16816.F32`), the base
opcode (`HMMA`), the operands, and branch targets (labels resolved to addresses). It reads only the named kernel's
`.text` section.

`main_loop()` finds the k-loop: among backward branches that are conditional (a predicate, or `BRA.U` on a uniform
predicate), the one whose body holds the most mma, then the most copies (`LDGSTS` or `UTMALDG`), then the longest.
Unconditional backward jumps are retry stubs of mbarrier waits, not loops. In the 128×128×32 kernel the loop body has
64 HMMA, 16 LDSM and 8 LDGSTS: one k-tile per pass.

### 4.2 Warps, sub-partitions and issue

The SM holds every block it will run, but only `resident` blocks are live at once (Section 6). Warp i of a block goes to
sub-partition i mod 4.

Each cycle, each sub-partition issues at most one instruction. A warp is eligible when all of these hold:

- it is not done, not held at a barrier, and its stall count has run out;
- every counter in the instruction's wait mask is zero; for `DEPBAR.LE SBn, k`, counter n is at most k;
- the unit the instruction needs is free: tensor pipe for mma; SFU for `MUFU` and integer/float conversions; FMA or ALU
  unit for arithmetic; the sub-partition's memory slot for `LDSM`, `LDGSTS`, `STSM`, `LDG`, `REDG`;
- for stores and atomics: the SM's store queue has room.

Among eligible warps the scheduler is greedy: it keeps the warp it issued from last, if that warp is still eligible.
Otherwise it takes the oldest, which here means the lowest warp number (the fallback loop rule in 4.4 also counts loop
passes, and fewer passes goes first). After issue, the warp's next issue is at `now + max(1, stall)`.

This policy was checked against probe `sched.cu`: warp 0 runs a long stream of independent FFMAs, warp 4 on the same
sub-partition runs dependent integer adds. `schedtest.py` replays the probe's own machine code under four policies
(re-run for this note):

| Policy | Cycles per add, alone | Next to the FFMA warp |
|---|---:|---:|
| Measured | 5.67 | 15.10 |
| Greedy, then oldest (the simulator's) | 5.31 | 14.91 |
| Round-robin | 5.31 | 5.33 |
| Leave a warp when its yield bit is set | 5.31 | 5.31 |
| Leave a warp when its yield bit is clear | 5.31 | 14.91 |

Time moves one cycle when anything issued. When nothing did, it jumps to the next moment something can change: a warp's
stall running out, a unit coming free, the next event, or the store queue draining.

### 4.3 Dependencies

The simulator keeps the six counters per warp as numbers. An instruction with a write counter adds 1 and schedules an
`sb` event that subtracts it when the result is ready; the same for a read counter and the moment its inputs are read.
Instructions with a fixed latency (FFMA, IMAD, HMMA and so on) carry no counter. The compiler covers their latency with
stall counts, and the simulator obeys those. So the 29-cycle mma latency in `gb10.json` (`hmma_latency_clk`, probe
`mma_rate.cu`) is not read by the code: the stall counts already carry it.

When the counters clear:

| Instruction | Write counter clears | Read counter clears |
|---|---|---|
| `LDSM` (ldmatrix) | start on the array + 34 + 4 × (ways − 1) | issue + 8 |
| `LDG` (plain load), L1 hit | end of its L1 time + 45 | end of the address stage |
| `LDG`, L1 miss | end of its return-path time | end of the address stage |
| `LDL`, `STL` (spills) | end of its L1 time + 45 | end of its L1 time |
| `MUFU`, `F2I`, `I2F`, `F2F`, `FRND`, `I2FP`, `F2IP` | issue + 18 | issue + 8 |
| `LDGSTS` (cp.async) | none (completion goes through its group) | end of the address stage |
| `LDGDEPBAR` (commit) | when the group's last copy is in shared memory | |
| `STG`, `RED` | | when the store port starts it |
| `STSM` | | start on the array + 4 |
| `UTMACMDFLUSH` | | later of issue + 8 and last TMA store box sent + 457 |
| Arithmetic with a write counter, and anything else | issue + 20 (assumed) | issue + 8 (assumed) |

The 20 and 8 are the two values in `gb10.json` marked as assumptions, not measured.

An instruction whose predicate is known to be false (or is `@!PT`) uses an issue slot and its stall count, and nothing
else. One whose predicate is unknown runs.

### 4.4 Branches and the value evaluator

To follow the code's own control flow, the simulator tracks warp-wide integer values per warp. Lane 0 stands for the
warp. It knows:

- `RZ` = 0; `S2R SR_TID.X` = the warp's first thread (32 × warp number in the block); `SR_LANEID` = 0.
- `SR_CTAID` (block index): blocks go round the 48 SMs in launch order, so this SM's j-th block is grid block 48j,
  split into x (fastest), y, z.
- Results of `MOV`, `IADD3`, `VIADD`, `IMAD` (not `.WIDE` or `.HI`), `LEA`, `SHF` (left, and right in the simple forms),
  `LOP3` (through its 8-bit lookup table), `SEL`, `R2UR`, and their uniform (`U`) forms, when the inputs are known.
  Any other write to a register makes it unknown.
- Predicates from `ISETP` (signed or unsigned compare, with its AND/OR combine), `ELECT` (true: lane 0 is elected),
  `LOP3` writing a predicate from the thread index, and `PLOP3`. Any other write makes the predicate unknown.

A branch whose predicate is known is followed. When it is not known, older rules decide: the main loop's back edge is
taken `trips − 1` times; other loops use trip counts a driver passes in; an unconditional jump is taken; `BRA.DIV` (taken
only if lanes diverged) never is; anything else falls through. The trip count comes from the source: the main loop runs
`KT − (stages − 1)` times (the first `stages − 1` k-tiles are loaded before it), divided by how many k-tiles one pass
of the compiled loop covers (mma in the body against mma per k-tile per warp).

A guard: if one backward branch is taken more than 1,000,000 times, the evaluator turns itself off.

**Check** (`audit_values.py`, re-run for this note): one block of each of 46 kernels (25 at five shapes, 6 C-store
kernels, 8 split-K, 7 TMA picks) executes exactly block_M × block_N × K / 2048 mma instructions (K per split) with
branches decided by computed values. 46 of 46 pass. `notes/gb10_model.md` adds that with a deliberately wrong trip
count the count stays right, so the loops really are decided by the values.

### 4.5 Arithmetic units

| Unit (one per sub-partition) | Instructions | Holds the unit | Probe |
|---|---|---|---|
| Tensor pipe | `HMMA`, `IMMA`, `QMMA`, `OMMA`, every shape and type | 16 cycles | `mma_rate.cu`, `mma_types.cu` (16.05-16.10) |
| FMA unit | `FFMA`, `FADD`, `FMUL` | 1 cycle | `alu.cu`, `alumix.cu` |
| FMA unit | `IMAD`, `IMUL`, `HFMA2`, `HADD2`, `HMUL2` | 2 cycles (FP32 waits too) | `alu.cu`, `alumix.cu` |
| ALU unit | `LOP3`, `PRMT`, `SHF`, `IADD3`, `ISETP`, `SEL`, `LEA`, `F2FP`, `IMNMX`, `FMNMX`, `FSETP`, `FSEL`, `PLOP3`, `LOP`, `IABS`, `SHL`, `SHR` | 2 cycles | `alu.cu`, `alumix.cu` |
| FMA or ALU, whichever is free | `HADD2.F32` (half to float) | 2 cycles | `alumix.cu` |
| SFU | `MUFU` (exp2, rsqrt, ...), `F2I`, `I2F`, `F2F`, `FRND`, `I2FP`, `F2IP` | 8 cycles; result after 18 | `sfu.cu` |

Every mma type holds the pipe the same 16 cycles, so the type sets the rate: fp16 and bf16 give 1024 FLOP per clock per
SM, tf32 512, fp8, int8 and fp4 (through `kind::f8f6f4`) 2048 (`mma_types.cu`). Instructions not in this table (`MOV`,
`IADD.64`, `S2R`, `NOP`, most uniform ops) take only an issue slot, plus the assumed result time of 4.11 if they set a
write counter.

From `notes/gb10_model.md`: the 20 arithmetic probe kernels (single kinds and pairs), replayed from their own machine
code, come out with median error 3 %, worst 13 %. The SFU probe's code reproduces 4.0 / 15.8 / 16.0 / 16.0 results per
clock at 1 / 4 / 8 / 16 warps, FFMA within 4 %. One measured mix is unexplained: LOP3 with FFMA runs about 23 % below
what two separate units predict.

### 4.6 Shared memory: ldmatrix, stmatrix and bank conflicts

Each SM has one shared-memory array that moves 128 bytes per cycle, so 512 bytes (one warp-wide x4 access) take 4 cycles
(`smem_bw.cu`: conflict-free ldmatrix, LDS.128, STS.128 and stmatrix all move 512 B in 4 cycles per SM). The array is a
single queue (`smem_free`) shared by ldmatrix reads, stmatrix writes, cp.async data arriving, TMA boxes arriving and TMA
store reads.

- **ldmatrix** holds the sub-partition's memory slot 11.7 cycles (one warp alone issues one per 11.7, `instlat.cu`),
  then the array 4 × ways cycles. Its result is ready 34 cycles after it starts on the array, plus 4 per extra way.
- **Bank conflicts.** Shared memory has 32 banks of 4 bytes. ldmatrix reads 8 rows of 16 bytes per 8×8 matrix, so each
  row covers a quarter of the banks. Rows that land on the same quarter take turns. Ways = 8 / quarters touched
  (`bank_conflict_ways` in `model.py`): row stride 16 B → 1 way, 32 B → 2, 64 B → 4, 128 B or more → 8. `smem_bw.cu`
  measured exactly 4 cycles × ways. A tiles are read with rows `block_K × 2` bytes apart (plain `LDSM.M88`), B tiles
  transposed with rows `block_N × 2` apart (`LDSM.MT88`). This applies only when `h["smem_layout"] == "linear"`.
  TileLang's inferred and swizzled layouts are conflict-free, so they get 1 way.
- **stmatrix** holds the memory slot 10.8 cycles (`tmastore.cu`: one warp every 10.77 cycles) and the array 4 cycles.
  Its source registers are free once the array has taken them.

Plain shared loads and stores (`LDS`, `STS`) are not tied to the array; they get the assumed defaults. TileLang's GEMM
loops read shared memory almost only through ldmatrix (the 128×128×32 loop has 16 LDSM and 3 LDS per pass, and 3 of
those LDS are predicated off).

### 4.7 The cp.async copy path

Take the 128×128×32 tile with 128 threads. Each k-tile moves (128 + 128) × 32 × 2 = 16 KB. Each warp instruction moves
32 lanes × 16 B = 512 B, so each thread issues 4 copies of A and 4 of B per k-tile. An A row is 32 × 2 = 64 B, so one
512-byte copy covers 8 rows in 8 different 128-byte lines. A B row is 256 B, so a copy covers 2 rows in 4 lines.
`copy_lines(h)` gives that list, `[8, 8, 8, 8, 4, 4, 4, 4]`, and each warp cycles through it in program order.

The path of one copy:

1. **Issue.** The sub-partition's memory slot is held 6 cycles (`cpq.cu`: one warp issues a copy every 6 cycles).
2. **Address stage.** The SM's load unit looks up one 128-byte line per cycle. This is a design assumption, checked:
   `cpq.cu` with contiguous 512 B (4 lines) gives 4 cycles per copy with 4 warps, and the rule predicts the one-block
   loss of three ablated kernels within 20 %. The copy's read counter clears here, which is what TileLang's address
   update waits for (Section 1).
3. **L2 or DRAM.** A hit arrives 350 cycles later (`instlat.cu`: 16-byte cp.async round trip from L2). A miss waits
   1560 cycles (0.62 µs, from the DRAM stream probe `mlp_stream`) and then for its turn in this SM's DRAM queue, which
   takes one copy per `dram_clk` cycles (Section 6.4). Misses are spread evenly: each copy adds the miss fraction to an
   accumulator, and the copy that takes it past 1 misses.
4. **Return path.** The data comes back through the SM's port at
   `max(16.7, 512 × share / 817.7)` cycles per 512 B. 16.7 cycles is 30.7 B per clock, one SM's port (`loopsteps.cu`:
   TileLang's loop moves 78 GB/s per SM). 817.7 B per clock is the chip's L2 for this traffic, 2.08 TB/s
   (`loopsteps.cu` 2.07-2.10, `l2share.cu` 2.08-2.14). `share` is how many SMs' worth of blocks are pulling at the same
   time (Section 6.3). At a share of 32 the return path takes 20.0 cycles per copy; at 48, 30.1.
5. **Into shared memory.** A `write` event puts the 512 B into the array (4 cycles), then a `landed` event counts the
   copy off its group.
6. **Groups.** A commit (`LDGDEPBAR`) closes the current group and raises its write counter. The counter drops when
   the group's last copy has landed. `DEPBAR.LE SB0, 2` then lets the warp pass when at most 2 groups are still out.

The code for steps 2 to 4 (the `busy` bookkeeping left out):

```python
lines = self.copy_lines[w.ncopy % len(self.copy_lines)]; w.ncopy += 1
t_lsu = max(now, self.lsu_free); self.lsu_free = t_lsu + lines * P["lsu_clk_per_line"]
lsu_done = self.lsu_free
self.miss_acc += self.miss_frac
if self.miss_acc >= 1.0:
    self.miss_acc -= 1.0
    t_d = max(lsu_done + P["ldgsts_dram_latency_clk"], self.dram_free); self.dram_free = t_d + self.dram_clk
    arrive = t_d
else:
    arrive = lsu_done + P["ldgsts_l2_latency_clk"]
ret = max(arrive, self.fill_free); self.fill_free = ret + self.ret_clk
```

And the address-stage effect in a real loop (128×128×32 kernel): the copy at `0xfb0` sets read counter 1; the address
update at `0x1030` (`IADD.64 R198, R198, 0x10000`, wait mask `0b10`) cannot issue until the load unit has processed it.

### 4.8 Plain loads, the L1 and spills

`LDG` uses a per-instruction list of (lines the warp touches, bytes it uses, whether the L1 already holds them), cycled
in program order. A driver supplies it; the default is (4, 512, miss). Each load goes through the address stage like a
copy. An L1 hit then takes 2 cycles per line on the L1 path and 45 cycles of latency (`ldg_atomic.cu`: L1 hits 7.4-8.5
cycles per 512 B; a dependent load 50.8 cycles per hop, about 45 after the loop's own instructions). A miss goes like a
copy, with 342 cycles to L2 (`ldg_atomic.cu`: 342.5 per hop in a 4 MB ring), and return-path and DRAM time scaled by its
bytes. Spills (`LDL`, `STL`) take 4 cycles per line on the L1 path, then 45 (`ldg_atomic.cu`: one slot per warp, 11.9
cycles per two loads and a store).

### 4.9 Stores, atomics and the store queue

**The store port is shared by an SM pair.** Probe `storepat.cu`: every store pattern doubles in time from 1 to 2 busy
SMs, then stays flat up to 8. So when the chip runs at least 2 SMs' worth of blocks, each store costs twice the pair
time. The chip also has its own store limit. This is why a TileLang kernel's per-block time rose about 8 % as soon as a
second SM was busy: the C store at the end.

- **Plain stores (`STG`).** Each instruction holds the port for the larger of the pair time (doubled when the partner
  SM is busy) and bytes / this SM's share of the chip limit. TileLang's default C store writes 4 bytes per lane, 8 rows × 16 bytes per instruction, so
  each 32-byte sector is filled in two halves: 15 cycles per instruction per pair, chip 137.8 B per clock (0.35 TB/s).
  Coalesced 16-byte stores: 14 cycles per 512 B, chip 570.9 B per clock (1.45 TB/s). Example: a 4-byte store with 32
  SMs' worth of blocks costs max(15 × 2, 128 × 32 / 137.8) = max(30, 29.7) = 30 cycles.
- **Atomics without a return value (`RED`, `REDG`).** 128, 256 or 512 bytes per instruction (F32, F32x2, F32x4). Each
  costs the largest of: bytes / 25.6 (the SM's rate), lines × 2 cycles × pair share (the pair's port), and bytes × share
  / chip rate. The chip rate is 225 B per clock when the atomics come straight from mma accumulators (8 lines of 32 B per
  instruction) and 330 otherwise. The sub-partition's memory slot is held 2.65 cycles per line. All from `ldg_atomic.cu`
  and `stq.cu`: RED.F32x2 in the accumulator layout takes 16 cycles per instruction for one block, 32 each for two
  blocks on a pair, and one warp alone issues one every 21.25 cycles.
- **Store queue.** An SM takes about 16 store or atomic instructions, then issue waits for the drain (`stq.cu`). The
  simulator keeps the finish times of queued `STG`/`RED` and blocks issue while 16 are pending. Without it a block could
  exit with 10 µs of atomics still queued and the next block would start too early.
- **Exit.** A block's slot frees when its last warp exits, even with stores still draining. The kernel's end time is at
  least when the store port drains.
- **TMA stores (`UTMASTG`).** The C tile is split evenly over the store boxes in the code. Each byte costs the larger of
  pair share / 33.1 and SM share / 482 cycles; the box is also read out of the shared-memory array. `UTMACMDFLUSH`'s
  read counter clears 457 cycles after the last box is sent. From `tmastore.cu`: a 64 KB tile takes 0.96 µs for one
  block, 1.74 µs each for two blocks on a pair, 2.75 µs with 48 blocks; waiting for the read out of shared memory (what
  TileLang's store wait does) costs the same as waiting for the write.

The staged C store, as TileLang compiles it: every warp writes its accumulators with 16 stmatrix x4; a block barrier;
every thread but one exits; that thread fences, issues two TMA store boxes, flushes and waits. The direct store is 64
four-byte `STG.E` per warp (`runs/tilelang/epilogue/epi_1024.txt`).

### 4.10 Block barriers

`BAR.SYNC id[, count]` holds the warp. It waits for `count / 32` warps, or, with no count, for the whole block minus
warps that already exited. When the last one arrives, all are released `13 + 2 × warps` cycles later (`instlat.cu`
measured 14.6 plus about 2 per warp; the stored base is 13 and the file does not say how it was derived). Example:
8 warps release 29 cycles after the last arrives. Named barriers with a count let the consumer warps of a TMA kernel
sync among themselves.

### 4.11 Everything else

Any other instruction with a write counter gets its result 20 cycles after issue, and its read counter clears after
8. Both are assumptions. Instructions that land here include `LDS`, `STS`, `LDC`, `SHFL`, `S2R` and shared-memory
atomics (`ATOMS`). In the base `SM` class `UTMALDG` and the mbarrier instructions also land here; only `WSSM` models
them.

### 4.12 The constants

All from the `sim` block of `gb10.json`. Probe files are in `agentic/tilelang/microbench/`.

| Constant | Value | Meaning | Source |
|---|---:|---|---|
| `hmma_16816_pipe_clk` | 16 | Cycles one mma holds the tensor pipe, any type | `mma_rate.cu`, `mma_types.cu` |
| `hmma_latency_clk` | 29 | mma result latency | `mma_rate.cu`; **not read by the code** |
| `fma_fp32_clk` | 1.0 | FFMA/FADD/FMUL on the FMA unit | `alu.cu`, `alumix.cu` |
| `alu_half_rate_clk` | 2.0 | IMAD, half math (FMA unit); logic, shifts, int add (ALU unit) | `alu.cu`, `alumix.cu` |
| `ffma_lanes_per_clk_per_sm` | 124 | FFMA lanes per clock per SM | `sfu.cu`; **not read by the code** |
| `mufu_smsp_clk_per_instr` | 8.0 | SFU cycles per instruction per sub-partition | `sfu.cu` |
| `mufu_latency_clk` | 18 | SFU result latency | `sfu.cu` |
| `ldsm_latency_clk` | 34 | ldmatrix result latency | `instlat.cu` (38.4 per dependent hop including the address add; how 34 was derived is not recorded) |
| `ldsm_sm_clk_per_x4` | 4.0 | ldmatrix x4 on the array | `instlat.cu`; **not read**: the code uses `smem_array_clk_per_512B`, same value |
| `smem_array_clk_per_512B` | 4.0 | Shared-memory array time per 512 B (128 B/clock) | `instlat.cu`, `cpq.cu`, `smem_bw.cu` |
| `mio_ldsm_clk` | 11.7 | Memory slot per ldmatrix | `instlat.cu` (one warp alone) |
| `mio_stsm_clk` | 10.8 | Memory slot per stmatrix | `tmastore.cu` |
| `mio_ldgsts_clk` | 6.0 | Memory slot per cp.async | `cpq.cu` |
| `lsu_clk_per_line` | 1.0 | Address stage per 128-byte line | design assumption, checked with `cpq.cu` |
| `ldgsts_l2_latency_clk` | 350 | cp.async round trip from L2 | `instlat.cu` |
| `ldgsts_dram_latency_clk` | 1560 | Copy or load latency from DRAM (0.62 µs) | `mlp_stream` |
| `l2_return_clk_per_512B` | 16.7 | One SM's return port (30.7 B/clock, 78 GB/s) | `loopsteps.cu` |
| `l2_aggregate_B_per_clk_cpasync` | 817.7 | Chip L2 rate for copy traffic (2.08 TB/s) | `loopsteps.cu`, `l2share.cu` |
| `ldg_l2_latency_clk` | 342 | Plain load from L2 | `ldg_atomic.cu` |
| `l1_clk_per_line` | 2.0 | L1 hit path per 128-byte line | `ldg_atomic.cu` |
| `l1_hit_latency_clk` | 45 | L1 hit latency | `ldg_atomic.cu` |
| `local_clk_per_line` | 4.0 | Spill path per line | `ldg_atomic.cu` |
| `stg_tpc_clk_per_instr_4B` | 15.0 | Pair port per 4-byte-per-lane store | `storepat.cu` |
| `stg_tpc_clk_per_instr_16B` | 14.0 | Pair port per 16-byte-per-lane store | `storepat.cu` |
| `stg_chip_B_per_clk_4B` | 137.8 | Chip store limit, 4-byte stores | `storepat.cu` (0.31-0.38 TB/s) |
| `stg_chip_B_per_clk_16B` | 570.9 | Chip store limit, 16-byte stores | `storepat.cu` (1.4-1.5 TB/s) |
| `st_queue_instr` | 16 | Store/atomic instructions an SM takes before issue waits | `stq.cu` |
| `red_sm_B_per_clk` | 25.6 | Atomics per SM | `ldg_atomic.cu` |
| `red_chip_B_per_clk` | 330.0 | Chip atomics, contiguous | `ldg_atomic.cu` (323-340) |
| `red_chip_B_per_clk_frag` | 225.0 | Chip atomics, accumulator layout | `ldg_atomic.cu` redfrag |
| `red_pair_clk_per_line` | 2.0 | Pair port per atomic line | `ldg_atomic.cu` redfrag |
| `red_smsp_clk_per_line` | 2.65 | Memory slot per atomic line | `stq.cu` |
| `tma_st_tpc_B_per_clk` | 33.1 | TMA store, pair rate | `tmastore.cu` |
| `tma_st_chip_B_per_clk` | 482.0 | TMA store, chip rate (1.22 TB/s) | `tmastore.cu` |
| `tma_st_fixed_clk` | 457 | TMA store fixed delay | `tmastore.cu` |
| `bar_base_clk`, `bar_per_warp_clk` | 13, 2 | Barrier release | `instlat.cu` (14.6 + about 2 per warp) |
| `mbar_wake_clk` | 134 | mbarrier arrive to waiter through | `mbar.cu` |
| `tma_l2_latency_clk` | 440 | TMA box from L2, before transfer | `dramlat.cu` |
| `tma_dram_latency_clk` | 1170 | TMA box from DRAM, before transfer | `dramlat.cu` |
| `default_var_latency_clk` | 20 | Result of any other counted instruction | **assumption** |
| `rbar_release_clk` | 8 | Read counter of non-memory instructions | **assumption** |

Values written into the code rather than read from this block: the TMA load rates 81 GB/s alone and 1.86 TB/s for the
chip (`ws.py`; the same as `tma_r_GBs_alone` and `tma_l2_aggregate_GBs` elsewhere in `gb10.json`, from the TMA loads-only
ablation); the shared-memory write of a TMA box at 128 B per clock; 48 SMs; the clock (2.54 GHz in `kernel.py`,
`ws.py` and `ws_validate.py`, but 2.52 GHz in `validate.py`); the 2.6 µs launch; the driver's occupancy limits.

## 5. The warp-specialized TMA path (`ws.py`)

This is TileLang's default lowering on `sm_121a`. A block launches its consumer threads plus 128 more: a producer group
of 4 warps. In the 1024³ pick (64×128×128 tiles, 2 stages, 128 consumer threads) warps 0-3 become producers and warps
4-7 consumers; warps 1-3 exit at once, and warp 0 issues the TMA loads. The simulator does not assume this split. It
follows the code: thread-index tests and `ELECT` are evaluated per warp (4.4), and each warp's role is fixed by where it
first waits on an mbarrier (inside the mma loop: consumer; elsewhere: producer) or by `USETMAXREG` (`DEALLOC`: producer,
`TRY_ALLOC`: consumer).

**Register hand-off.** Producers give registers back with `USETMAXREG.DEALLOC` so consumers can grow theirs. A consumer's
`USETMAXREG.TRY_ALLOC` waits until 4 producer-group warps have deallocated; its retry loop is then skipped.

**The TMA engine.** One per SM, moving `min(81 GB/s, 1.86 TB/s / share)`. At 2.54 GHz that is 31.9 B per clock alone and
15.3 at a share of 48. The rate is recomputed when the share changes.

**One box.** The k-tile's bytes (only rows and columns inside the matrix: `(min(block_M, M) + min(block_N, N)) ×
block_K × 2`; the TMA unit fills the rest with zeros without moving data) are split evenly over the `UTMALDG`
instructions between two waits. For a box issued at time t:

```
start   = max(t, engine free);      engine busy for box bytes / rate
no DRAM:   lands at max(start + 440, engine done)
DRAM part  (box bytes x miss fraction):
           DRAM slot starts at max(start + 1170, DRAM queue free) and lasts miss bytes x dram_clk / 512
           lands at max(start + 440, engine done, end of DRAM slot)
```

Then a `tmawrite` event writes it into the shared-memory array at 128 B per clock, and `tmalanded` counts it in. When
every box of the k-tile is in, the k-tile is landed. Example from the 1024³ pick alone on the chip: a k-tile is
49,152 B in 4 boxes of 12,288 B; each holds the engine 385 cycles; the first box starts at cycle 826 and lands at 1266
(826 + 440), the second starts at 1211.

The 1170 and 440 come from `dramlat.cu`: one box loaded alone took 512 / 608 / 704 / 896 ns from DRAM for 4 / 8 / 16 /
32 KB (about 460 ns, 1170 cycles, plus the engine's transfer) and 224 ns from L2 for 4 KB (about 440 cycles plus
transfer). Under load a box waits behind everyone's queued bytes; the DRAM queue gives that form. Before the decode fix
(below) the TMA path used cp.async's 1560 and 350.

**mbarrier waits.** TileLang waits with `SYNCS.PHASECHK.TRANS64.TRYWAIT` in a spin loop with a retry stub. The
simulator treats each as one blocking wait, matched by order, and sets its predicate so the retry stub is never taken:

- A consumer's n-th wait passes when the producer has arrived on k-tile n's barrier and every byte of k-tile n has
  landed, both at least 134 cycles ago (`mbar.cu`: a waiter gets through 134 cycles after the arrive with try_wait).
- The producer's wait before loading k-tile k passes when every consumer warp has released k-tile k − stages (its
  `SYNCS.ARRIVE`), at least 134 cycles ago.

```python
k = w.n_full; arr = self.arrived.get((b, k)); wake = P["mbar_wake_clk"]
if arr is None or arr + wake > now: return False
if (self.landed.get((b, k)) or 0.0) + wake > now: return False
```

**Pipeline depth.** With S stages the producer can run S k-tiles ahead. When each k-tile comes from DRAM, the chain
"consumers release k-tile k − S → 134 → producer passes → TMA (1170 + transfer + queue) → lands → 134 → consumers pass"
is on the critical path unless S k-tiles of mma cover it. That is why depth matters for decode shapes. Measured at
M=128 with 128×128×32 tiles: 164 → 152 → 148.5 µs for 2 → 3 → 4 stages, at 233 GB/s.

**The wake events.** A wait can pass only at "landed + 134" or "arrive + 134", but nothing in the simulation used to
happen at that moment. A waiting warp was re-checked only at the next unrelated event, often thousands of cycles later,
and 2-stage kernels came out far too slow. Now every arrive, release and landed k-tile schedules a `wake` event 134
cycles later. This fix, the in-matrix byte count, TMA's own latencies and the cold C rule (6.5) together moved the
decode sweep (Section 7) from 3 % / 35 % median / worst error to 1 % / 3 % at M=32, and from 7 % / 67 % to 3 % / 9 % at
M=128.

**k-loops.** The producer's and consumers' loops are decided by k-tiles left (`kt`): the producer continues while
`k_prod + 1 < kt`, a consumer while `n_full < kt`. The compiler emits either a conditional back edge, or a conditional
forward exit followed by an unconditional jump back; the exit test may sit a few instructions before the jump (the
4096³ and 8192³ picks). Before that look-ahead was added, those producers never ended.

**Limits of this path.** It assumes one block per SM (`resident=1`) and ignores the register count. On the TMA path
ptxas fills TileLang's launch-bounds register budget (a comment in `agentic/decide/pipeline.py` gives 240 and 168
registers per thread for 128 and 256 consumer threads), so a second block would not fit anyway. A TMA kernel without
warp specialization has no model: the base `SM` would treat its `UTMALDG` as a 20-cycle instruction. The decision
pipeline never sends one.

## 6. The whole kernel (`kernel.py`)

### 6.1 Blocks per SM

`occupancy()` applies the driver's limits:

```
R = max(1, min(65536 // (warps × regs_per_warp), 102400 // (smem + 1024), 1536 // threads, 24))
regs_per_warp = ceil(regs × 32 / 256) × 256
```

Example: 128×128×32, 3 stages, 128 threads, 202 registers. Registers: 4 warps × 6656 → 2 blocks. Shared memory:
49,152 + 1024 → 2 blocks. So R = 2.

### 6.2 The busiest SM, with block replacement

Blocks go round the SMs in launch order, so the busiest SM gets `ceil(tiles / 48)` blocks. The simulator runs exactly
those, at most R at a time. When all warps of a block have exited, the next block starts on the freed slot the next
cycle. This is "FLOW mode" (`FLOW = True` in `kernel.py`). It replaced simulating separate waves, which could not overlap
one block's atomic tail with the next block's loads. The wave code is still there behind the flag.

### 6.3 Shares that follow the grid

The chip-wide limits (L2 rate, DRAM, chip store limits, TMA rate) are shared by blocks in flight, not by SMs. Probe
`l2share.cu`: the L2 delivers 2.08-2.14 TB/s with 1, 2, 3 or 4 blocks resident per SM. So the simulator's share is
counted in SM-equivalents: blocks in flight on the whole chip divided by blocks in flight on this SM. It assumes every
SM progresses alike:

```python
def share(self, finished):
    q, r = divmod(self.grid_blocks, 48); R = self.resident
    counts = [q + 1] * r + [q] * (48 - r); mine = max(counts)
    chip = sum(max(0, min(R, c - finished)) for c in counts); own = max(1, min(R, mine - finished))
    return max(1.0, chip / own)
```

Example: 1024³ with 128×128 tiles is 64 blocks, R = 2. 16 SMs get 2 blocks and 32 get 1. At the start 64 blocks are in
flight and this SM holds 2, so the share is 32 SM-equivalents and the return path takes 20.0 cycles per copy. When this
SM has finished one block, every SM is assumed to have finished one: 16 blocks remain, this SM holds 1, the share drops
to 16, and the return path is back at its own port's 16.7 cycles. The share is recomputed at every block exit; it sets
the return path, the DRAM queue rate (scaled by the share), the chip store and atomic limits, and the TMA rate. The
pair's store-port share is fixed at the start: 2 whenever the share is at least 2.

Using blocks in flight instead of a wave split fixed the two worst 1024³ kernels (+16 % to +4 %, +12 % to +1 %, per
`notes/gb10_model.md`).

### 6.4 DRAM traffic

**Which copies miss.** `dram_split_kslice()` in `model.py` replays the L2. The blocks of a wave (48 × R in launch order)
walk k in lockstep. At each k-step each block reads its A slice (block_M × block_K) and B slice (block_K × block_N).
Slices pass through an L2 of 16.78 MB, the measured effective size (about 2/3 of the 24 MB, where a re-read stops
hitting). The L2 keeps two lists, slices seen once and slices hit again; a hit promotes, and eviction takes seen-once
slices first, because the real L2 protects reused data (`l2policy.cu`: a re-read 4-8 MB set survives a 64 MB stream).
Launch order follows TileLang's: row-major, or panels of `swizzle` tile rows walked back and forth. Rows past M
are never read. Against ncu's L2 read misses on 18 kernels (`notes/gb10_model.md`): 403/403, 409/413, 749/772 and
725/767 MB at 4096³, 3138/3219 at 8192³; median within 6 %.

**How fast DRAM serves them, by operand.** A misses come as `block_K × 2`-byte pieces of rows far apart. B misses do
not: the column tiles in flight read neighbouring pieces of the same B rows at the same k-step, so DRAM sees one run of
`columns in flight × block_N × 2` bytes. C goes back to DRAM at the write rate. The read rate depends on run length
(`tileload.cu`, `mlp_stream`, interpolated on a log scale): 64 B → 204 GB/s, 256 B → 233, 4096 B and up → 237. Writes:
197 GB/s. A mix is served as each part at its own rate (`dram.cu`: 1 / (f_read / 242 + f_write / 197) within 4 %). So
one wave's DRAM time is

```
A_bytes / rate(block_K × 2) + B_bytes / rate(cols × block_N × 2) + C_bytes / 197 GB/s
```

Example, 4096³ with 128×128×32 tiles (re-run for this note): 11 waves of up to 96 blocks; the first covers 3 tile rows
and 32 tile columns. Misses: A 33.6 MB at 204 GB/s (64-byte pieces), B 369.1 MB at 237 GB/s (8192-byte runs), 403 MB in
all, the same as ncu's 403.

**Into the SM.** The miss fraction is read-miss bytes over all bytes copied (0.1875 in that example). The DRAM time is
spread over the read misses, writes included, and shared by the SM-equivalents:
`dram_clk = 512 × (DRAM seconds / read-miss bytes) × 2.54e9 × share` cycles per 512 B. In the example that is 293 cycles
per 512-byte miss per SM.

**The CPU.** The CPU shares the LPDDR5X. `CPU_DRAM_GBs` in `kernel.py` takes the CPU's DRAM use away from these rates. It
is 0 in all measurements here. Probe `cpuhog.c`: while the CPU copies with 2, 8 or 20 threads, the GPU's read rate drops
from 237 to 214, 171 or 138 GB/s.

### 6.5 Warm and cold

- **Warm** (back-to-back runs): if A, B and C together fit in 16.78 MB, there is no DRAM traffic at all. Otherwise the
  misses above, and every C tile is written back at the write rate: a line written in one run is evicted before the next
  run writes it.
- **Cold** (after a flush): the same read misses. C ends the kernel dirty in the L2 and is written back later, during the
  next flush, so the kernel pays only for the part of C that does not fit beside the stream. But atomics on C lines not
  in the L2 read them from DRAM first, so cold split-K pays a read of its fp32 C.

### 6.6 Time

Kernel time in µs = 2.6 + cycles / 2540. The 2.6 µs is the launch cost from the ablation harness. Probe `launch.cu`:
2.05 µs per empty kernel back-to-back from one stream (the host's submission rate), 3.1 µs between kernels that do work,
0.43 µs per kernel inside a CUDA graph. `launch_us(graph=True)` returns the graph figure, but `simulate_kernel` always
adds 2.6. The 2.54 GHz clock comes from short probes. Probe `sustain.cu`: a 20 s tensor load ran at 2496 MHz, then
2476, so long runs are about 2 % slower than the simulator assumes.

### 6.7 Split-K and GEMV

**Split-K** (`split > 1`) adds a third grid dimension, outermost in launch order. Each block runs K / split and adds its
C tile with fp32 atomics, which are in the machine code. Each split is its own GEMM over K / split for the traffic
replay. The fp32 C stays in the L2 while the splits add into it and goes back to DRAM once per run.

**GEMV** (`gemv.py`, TileLang's vectorized split-K GEMV) has no shared-memory tiles. Each thread issues 16-byte loads of
the vector A and of its row of B, multiplies, then adds its partial sum into a shared-memory slot with a compare-and-swap
retry loop. The driver reads from the code which loads read what: a load whose address depends on the block index reads
B (each block its own rows; misses the L1), the others read A (the same for every block; hit the L1 after the first
touch). Lines per load come from the lane layout. The compare-and-swap loop runs `reduce_threads` rounds (ncu: 30 rounds
per warp at BLOCK_N 2, 19 at 8). Warm runs use the measured fraction of B that survives in the L2 between back-to-back
runs (`stream_b2b.cu`: all of it up to 16 MiB, 82 % at 20, 60 % at 24, 26 % at 28, 13 % at 32, none from 36).

## 7. Validation

Each set is real TileLang kernels measured on the GB10, compared with no refitting. Sources: the saved results in
`runs/gb10/sim_*.json` (written by the drivers), `notes/gb10_model.md`, and re-runs for this note. Where the note and a
saved file differ, the saved file is newer and is used.

| Set | Kernels | Driver | Result |
|---|---:|---|---|
| cp.async GEMMs at 1024³, whole kernel, warm | 22 | `validate.kernel_level()` | median 5 %, worst −16 %, 21 within 10 %, rank correlation 0.86; loads-only variants median 6 % |
| cp.async GEMMs at 2048³, 4096³, 4096×4096×1024, M=32, M=128 (N = K = 4096), warm | 25 | `shapes` | median 5 %, 21 within 10 %, the simulator's pick at most 1.8 % slower than the best |
| C store, direct against staged, 1024³, warm | 10 | `epilogue` | worst −10.4 %; staged picked in 5 of 5 pairs, as measured |
| Shared-memory layout, inferred against linear, 1024³, warm | 10 | `layout` | linear kernels within 8 %; cost of linear simulated / measured: 1.45 / 1.41, 1.40 / 1.38, 2.04 / 2.23, 1.77 / 1.62, 2.14 / 2.15 |
| GEMV and split-K, after an L2 flush | 16 | `gemv` | −7 to +5 % |
| The same, back-to-back | 16 | `gemv` | GEMV N=4096 −1 to −6 %; N=2048 −7 to −32 %; split-K 1024³ −8 to −17 %; split-K 128×4096×4096 +9 to +29 % |
| TMA picks at their grid | 7 | `ws_validate` | warm: 1024³ −11 %, 2048³ −6 %, 4096³ +4 %, 8192³ +4 %, 4096×4096×1024 −11 %; decode picks cold: M=32 +1 %, M=128 +7 % |
| TMA picks, one block, mma only | 7 | `ws_validate` | −1 to +15 % (re-run for this note) |
| Decode TMA sweep, after a flush | 26 | `decode_sweep` | M=32 (10 kernels): median 1 %, worst 3 %. M=128 (16): median 3 %, worst 9 %, rank 0.82, pick 10 % slower than the best |
| Branches from computed values | 46 | `audit_values` | 46 of 46 run exactly the expected mma count |
| Warp scheduler | 1 probe | `schedtest` | 14.91 against 15.10 cycles per add (Section 4.2) |

Details per set:

- **22 cp.async kernels at 1024³** (`runs/gb10/sim_kernel_level.json`). These are the ablated kernels: tiles from
  64×64 to 256×128 and 128×256, block_K 32 to 128, 2 or 3 stages, 128 or 256 threads, measured at 35 to 52 µs. The
  one kernel beyond 10 % is 128×128×32 with 256 threads (−16 %).
- **One block at a time** (`validate.py` run as a module). Re-run for this note, without writing files: mma alone (the
  `noload` variant, one block) median 3 %, 20 of 22 within 10 %. The whole-block and loads-only rows of this mode are not
  a fair test of today's model. This mode gives each SM a share of `min(48, tiles)` SMs whatever the blocks per SM, which
  over-counts the sharing when an SM holds two or more blocks (64 blocks at 2 per SM: 48 instead of 32). With one block
  per SM its whole-block error is −4 to +7 % (14 kernels); with two or more it reads +16 to +38 % (8 kernels).
  The kernel-level mode above uses blocks in flight and is the one the pipeline relies on.
- **25 kernels at five shapes** (`runs/gb10/sim_shapes.json`): per shape, median error 6 % (2048³), 4 % (4096³, 7
  kernels, −5 to +4 %), 5 % (4096×4096×1024), 1 % (M=32), 1 % (M=128). Off: 2048³ 128×128×32 at −16 % (not explained
  in the notes) and M=128 with 64×64×64 tiles at +35 %. That kernel runs 3 waves, and the L2 replay over-counts its DRAM
  traffic: the notes record 69 MB simulated against 51 MB measured for it, and say panelled orders and multi-wave decode
  kernels come out 20-35 % high because the real L2 keeps reused lines better than strict LRU over lockstep waves.
- **C store** (`runs/gb10/sim_epilogue.json`). Time the staged store saves, simulated / measured: 0.8 / 4.2, 3.3 / 4.4,
  3.5 / 2.7 µs for cp.async 128×128, 256×128, 64×128 tiles; 3.0 / 3.9 and 3.0 / 2.8 µs for TMA 128×128 and 256×128.
  The first is a known gap (Section 8, item 3).
- **GEMV and split-K** (`runs/gb10/sim_item5.json`). Back-to-back timing of working sets just above the L2 changes from
  session to session (the same split-K kernel took 168, 135 and 146 µs in three sessions), so the flushed column is the
  stable check. The warm GEMV at N=2048 is L2-resident: there more rows per block cost more on the hardware (8.3 → 11.9 µs
  from 2 to 8 rows) and nothing in the simulator.
- **TMA picks.** Re-run for this note: 1024³ −11 %, 2048³ −6 %, 4096×4096×1024 −11 % (4096³ and 8192³ were not
  re-run). The decode picks read +20 % and +37 % against older measurements (124 and 162 µs) from a session where part
  of B stayed in the L2; against the 146 and 203 µs measured after a flush on Oct 20 they are +1 % and +7 %. One block,
  mma only: 1024³ +11 %, 2048³ +6 %, 4096³ +2 %, 4096×4096×1024 +15 %, 8192³ 0 %, M=32 +4 %, M=128 −1 %. Why the
  1024³ and 4096×4096×1024 picks come out too fast at full grid is not explained in the notes.
- **Decode sweep** (`runs/gb10/sim_decode_sweep.json`): 26 TMA GEMMs, tiles × pipeline depth, timed after a flush. At
  M=32 every kernel already runs at 226-236 GB/s. At M=128 the measured time falls with depth until DRAM saturates,
  and the simulator misses part of that gain (Section 8, item 1).

## 8. Known gaps and limits

What is wrong today, with the evidence:

1. **Pipeline depth at the DRAM ceiling.** At M=128 the simulator sees no difference between 2 and 4 stages for
   128×128×32 (153.5 µs flat). The hardware gains 10 % (163.8 → 148.5 µs). A lone 4 KB box under load takes about twice a
   single line's latency, because it waits for the slowest of its 32 lines. The simulator's even fair-share DRAM queue
   has no such spread, so it thinks 2 stages keep DRAM busy. The next mechanism to add is a spread in box latency under
   load.
2. **The L2 replay over-counts some patterns.** Panelled launch orders and multi-wave decode kernels come out 20-35 %
   high in DRAM bytes. No textbook policy (LRU, random, segmented LRU, insertion at the bottom, RRIP at 24 MB) reproduces
   all of `l2policy.cu`; the two-list LRU over 16.78 MB still matches ncu best for GEMM traffic.
3. **Warps drifting apart at the loop-top barrier.** The staged-store cp.async 128×128 kernel's main loop, identical to
   the direct one apart from instruction order, runs 190 cycles per k-tile slower in the simulator: its warps drift apart
   and wait at the barrier. The hardware's loops run alike (the measured saving is 4.2 µs; the simulator gives 0.8).
4. **Fixed latencies.** Memory latency varies on the hardware and not here. The notes' guess for the L2-resident GEMV
   (more rows per block costs more on the hardware) is the block barrier waiting for its slowest warp.
5. **Back-to-back L2 carry-over.** Only the GEMV driver uses the measured carry-over curve. For tiled kernels, warm means
   "everything fits, no DRAM" or "replay from an empty L2". Working sets just above 16.78 MB are where warm predictions
   are weakest.
6. **Instructions with no measured cost.** `LDS`, `STS`, `LDC`, `SHFL`, `ATOMS`, `S2R` and the rest get the assumed
   20-cycle result and no unit time. That is fine for TileLang's GEMM and GEMV loops. A kernel that leans on shuffles or
   plain shared loads (softmax, attention) needs these measured first.
7. **What the code ignores on purpose.** The yield and reuse bits; mma result latency beyond stall counts; register bank
   conflicts (they cannot be set up from CUDA because the compiler picks registers; the math-only checks, within about
   4 %, bound them); the instruction cache (`icache.cu`: full rate up to 32 KB of loop code, −10 % at 64 KB; TileLang's
   kernels here are at most about 24 KB with main loops of about 3 KB); divergence inside a warp.
8. **One SM stands for all.** Other SMs are assumed to progress alike, and blocks are assumed to go round the SMs in
   launch order. The pair's store port is assumed shared whenever 2 or more SMs' worth of blocks are running.
9. **TMA path limits.** One block per SM; no TMA without warp specialization; every hand-off costs 134 cycles, though
   `mbar.cu` shows an arrive that follows a global store waits for it (217 cycles).
10. **Clock and launch.** One clock for all runs (long runs are about 2 % slower); a fixed 2.6 µs launch.
11. **The CPU.** DRAM taken by the CPU is an input (`CPU_DRAM_GBs`), not something the simulator can know.

What was checked and found not to matter (from `notes/gb10_model.md`), so it is not modelled:

- Contention on L2 lines shared by neighbouring blocks: `shareload.cu` loads GEMM-shaped tiles with neighbours sharing A
  and B, in lockstep and staggered: identical times.
- Clock throttling: clock64 against globaltimer inside real kernels reads 2.39-2.53 GHz at full grid.
- Issue interference: `interfere.cu`, a copy from another warp on the same sub-partition costs the mma stream 2.3 cycles
  (global load 1.9, ldmatrix 0.4, integer add 0).
- Commit, wait and barriers in the copy loop: `loopsteps.cu` adds them one at a time; they cost nothing measurable.

## 9. Rules for extending it

1. **Every constant names its source.** `gb10.json` says it: spec, probe or ablation; nothing is fitted to kernel
   times. `tests/test_gb10_model.py` checks that every entry's source starts with one of those words. A value later
   found wrong stays in the file marked `WITHDRAWN`, still naming its probe.
2. **A new mechanism comes with the probe that measures it.** Write a probe in `agentic/tilelang/microbench/` that
   isolates the unit. Put the number and its source in the `sim` block. Then check in this order: the simulator replays
   the probe's own machine code and reproduces it (as `schedtest.py` does for `sched.cu`, and as was done for `sfu.cu`,
   `alu.cu` and `alumix.cu`); then re-run every driver in Section 10 and compare with the table in Section 7.
3. **Do not tune to pass.** If a set gets worse with a measured mechanism, keep the mechanism and log the miss with what
   you suspect. The 134-cycle hand-off made two TMA picks worse when it went in (M=128 from +21 % to +55 %) and stayed,
   because it is measured. Later work on the same miss found the wake-event bug (Section 5).
4. **Read the probe's SASS before believing it.** The compiler has fooled these probes four ways:
   - **clock64 moves.** The compiler does not keep a clock read after register-only instructions. The first
     `mma_types.cu` read 8 times the true mma rate. Time such probes with events around the whole kernel.
   - **Repeated loads get merged or dropped.** ptxas removes loads it can prove repeat. The first `smem_bw.cu` read 2
     cycles per 512 B; the compiler had dropped half its loads. The fix: each address adds (previous result & zero),
     with zero = 0 only at run time, so every load depends on the one before.
   - **Work gets folded away.** The compiler computes what it can at compile time and removes the rest; the rcp row of
     `sfu.cu` was folded away (and not checked further).
   - **The probe's own code costs time.** `tileload.cu` did run-time integer division in its address code and read
     37.8 GB/s for one block (TileLang's real loop moves 78); `instlat.cu` put lanes 128 bytes apart and measured its
     own bank conflicts as a 16 B per clock copy limit. Both values are withdrawn.
5. **Confirm on a real kernel.** A probe number should match TileLang's own ablated kernels where possible: the 78 GB/s
   copy port was confirmed by TileLang's loads-only kernels at 76-80 GB/s with one block.
6. **Prefer ncu stall reasons to guesses.** The address-stage mechanism was found by comparing ncu stall counts of a
   kernel with and without its loads, then reading the machine code.

## 10. How to run it

The simulator needs only the Python standard library. The drivers that print rank correlations (`validate`, `shapes`,
`decode_sweep`) also import scipy. Run from the repository root.

**One cp.async kernel:**

```python
from pathlib import Path
from agentic.gb10.sim.kernel import simulate_kernel

h = {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 3, "threads": 128, "swizzle": 0}
r = simulate_kernel(Path("runs/gb10/shapes/s2048x2048x2048_128x128x32_st3_th128_p0.sass"), h,
                    {"M": 2048, "N": 2048, "K": 2048}, regs=202, warm=True)
print(r["us"], r["parts"][0]["busy"])          # 182.8 us
```

Split-K: add `split=4, atomic_c=True, kernel="<name from the SASS>"`. A linear shared-memory layout: add
`"smem_layout": "linear"` to `h`.

**One warp-specialized TMA kernel:**

```python
from pathlib import Path
from agentic.gb10.sim.ws_validate import kernel_us

h = {"block_M": 64, "block_N": 128, "block_K": 128, "num_stages": 2, "threads": 128, "swizzle": 8}
us = kernel_us("1024", h, {"M": 1024, "N": 1024, "K": 1024}, warm=True,
               sass_path=Path("calibration/spark-6951/sass/tma_1024__full.sass"))   # 40.1 us; measured 45.1
```

`threads` is the consumer threads; the producer group's 128 are added inside. `ws_sim()` in the same file runs a single
block and returns the full `run()` result with `warps_done`, which shows whether every warp finished.

**The drivers.** Each prints simulated against measured. Most also overwrite their saved results in
`runs/gb10/sim_*.json`.

| Command | What it runs | Data it reads |
|---|---|---|
| `python -m agentic.gb10.sim.validate [name ...]` | 22 ablated cp.async kernels at 1024³, one block per variant (full, noload, nocompute) | `calibration/spark-6951/ablation_cpasync/`, `calibration/spark-6951/sass/` |
| `python -c "from agentic.gb10.sim.validate import kernel_level; kernel_level()"` | The same 22, whole kernel | same |
| `python -m agentic.gb10.sim.ws_validate [name ...]` | 7 TMA picks: one block mma-only, one block whole, whole kernel | `calibration/spark-6951/ablation/`, `sass/tma_*` |
| `python -m agentic.gb10.sim.shapes` | 25 cp.async kernels at five shapes | `runs/gb10/shapes/` |
| `python -m agentic.gb10.sim.epilogue` | 10 kernels, direct against staged C store | `runs/tilelang/epilogue/` |
| `python -m agentic.gb10.sim.layout` | 10 kernels, inferred against linear layout | `runs/tilelang/layout/pairs/` |
| `python -m agentic.gb10.sim.gemv` | 8 GEMV and 8 split-K kernels, warm and cold | `runs/tilelang/item5/` |
| `python -m agentic.gb10.sim.decode_sweep [name ...]` | 26 decode TMA kernels, cold | `runs/tilelang/decode_stages/` |
| `python -m agentic.gb10.sim.audit_values` | 46 kernels, one block, mma count with computed branches | all of the above |
| `python -m agentic.gb10.sim.schedtest` | Scheduler policies on `sched.cu`'s machine code | `calibration/spark-6951/sass/sched.sass` |

Run times on a laptop (re-run for this note): about 1 s for the 2048³ kernel above, 13 s for the 4096³ one, 4 s for
`schedtest`.

## 11. How the decision pipeline uses it

The decision pipeline (`agentic/decide/pipeline.py`, `python -m agentic.decide.pipeline M N K [--cold]`) searches for the
fastest TileLang kernel for one shape:

1. List every legal assignment of the template's choices: copy path, tile, depth, threads, C store, split-K, launch
   order.
2. Price each with the analytical model.
3. Keep the MILP's top 30.
4. Add variants the model cannot price (prefetch and warp policy on the cp.async path); compile all on the GPU host,
   check correctness, save SASS and register counts.
5. **The simulator re-ranks the compiled candidates.** The simulator's top 5 and the model's top 3 are then timed on the
   GPU against cuBLASLt in the same session.

The call, in `simulate()`:

```python
if a.get("copy_path") == "tma" and a.get("warp_spec", 1):
    return kernel_us(rec["name"], h, shape, warm=warm, sass_path=path)
return simulate_kernel(path, h, shape, rec["regs"], warm=warm, split=a.get("split_k", 1),
                       kernel=rec["kernel_name"], atomic_c=a.get("split_k", 1) > 1)["us"]
```

`h` carries the tile, depth, threads and launch order; the C store, prefetch and warp policy reach the simulator only
through the machine code. `smem_layout` is not passed, so every candidate is treated as conflict-free, which matches
TileLang's default layouts. A simulation that raises an error gets an infinite time, and the error is kept with the
record.

The tier study (`agentic/decide/study/run.py`, run entirely on the GPU host) does the same per tier. `sim_worker()`
makes the same two calls in 12 worker processes; any error gives an infinite time. Per tier the finalists are the
simulator's top 5, the model's 5 best distinct kernels with their prefetch and warp-policy variants, and the smaller
tiers' finalists. The study skips the simulator for 4096³ and 8192³; there the model keeps its 8 best instead. The
driver's comment says one simulation takes minutes at those shapes, but the 4096³ kernel in Section 10 took 13 s on a
laptop, so the skip may no longer be needed. It matters: at 8192³ the model ranked the copy path wrong, and the winning
TMA kernel reached the timing step only because the smaller tier's finalists carry forward.

## 12. Where the code's own comments are out of date

Readers of the code will meet these; the code's behaviour is what this note describes.

- `ws.py`'s docstring says the chip's TMA rate is 1.58 TB/s. The code uses 1.86 TB/s. The 1.58 figure included the C
  store and was superseded.
- `sm.py`'s docstring says cp.async moves one warp instruction per 32 cycles through the fill path. That 16 B per clock
  figure was withdrawn (a probe's own bank conflicts); the code uses the 30.7 B per clock port and the chip share.
- `sm.py`'s docstring says the k-loop's back edge is taken trip-count times. Branches are now decided by computed
  values first; the trip count is the fallback.
- `kernel.py`'s docstring says it times warm runs. It handles both warm and cold.
- `validate.py` converts cycles at 2.52 GHz; `kernel.py`, `ws.py` and `ws_validate.py` use 2.54 GHz.

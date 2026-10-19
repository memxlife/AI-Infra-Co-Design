# A functional timing model of the GB10's GPU

*Oct 20 2026. Code: `agentic/gb10/model.py`. Constants: `calibration/spark-6951/gb10.json`, each with its source.
Tests: `tests/test_gb10_model.py`. Plan and acceptance: `notes/gb10_model_plan.md`.*

## What it is

The model describes a tiled GEMM kernel the way the chip runs it. Each term is one hardware unit doing its job at
a rate we know from the chip's design or measured with a small probe. Nothing is fitted to kernel times. The model
answers three questions for a kernel: how long it takes, which unit sets that time, and what to change.

What "good" means here (Li): rank candidate kernels the way the GPU does, and name the unit the GPU really spends
its time on. Absolute error is reported but not the goal.

## The chip, as the model sees it

| Unit | Number | Where it comes from |
|---|---|---|
| SMs | 48, 2.52 GHz under load | driver; clock probe |
| Tensor pipe | 1024 FLOP per clock per SM (2.58 TF/s) | design; mma probe reached it with 8 warps |
| Tensor pipe in a real GEMM loop | 83 % with 1 warp per sub-partition, 90 % with 2 | ablation of 22 kernels (ldmatrix shares the warps) |
| Shared memory | 128 B per clock; a linear layout makes ldmatrix rows collide 4 or 8 ways | design (32 banks x 4 B) |
| Occupancy | registers, shared memory, 1536 threads, 24 blocks per SM | driver |
| L2 | 24 MB; about 16.8 MB holds reused data | driver; reuse probe |
| L2 latency, DRAM latency | 136 ns; about 0.44-0.62 µs | pointer chase; Little's law at low load |
| One SM's port to L2 | 80.7 GB/s (32 B per clock) for a plain stream | stream probe, 1 to 32 blocks |
| TileLang's cp.async loop | 0.22 µs per k-tile + bytes at 71 GB/s for one block; 47 GB/s per SM with two | loader probe; stages don't change it |
| TileLang's TMA loop | 81 GB/s for one block; 1.58 TB/s chip-wide at full grid | loads-only ablation of the real kernels |
| Overlap in the cp.async loop | (compute^2 + loads^2)^(1/2) inside a block; full across blocks | ablation: whole against parts, median exponent 1.97 |
| DRAM | 237 GB/s for a long stream; 204 GB/s for 64-byte rows; few SMs need enough bytes in flight | stream and tile probes |
| Dirty L2 | after a write flush, reading costs extra write-back (12 µs at 2 MB, 38 µs at 64 MB) | flush probe; TileLang's do_bench leaves the L2 this way |
| Fixed costs | launch 2.6 µs, block setup 0.1 µs, store 28 ns per KB | ablation |

## How the parts combine

Per SM and per k-tile, with R blocks on the SM: the mma time, the shared-memory time, the copy time through the
loop or the port, and the wait for data that streams from DRAM (one round trip plus the transfer at this SM's share,
spread over stages minus one tiles in flight). A wave's first k-tile is a pipeline fill. Waves are tiles over SMs
times blocks per SM; the last wave runs on the SMs it needs. DRAM traffic comes from a simulation: blocks run in
TileLang's launch order, a wave at a time, through an L2 of 16.8 MB; a wave takes the longer of its SM time and its
own DRAM traffic. Spilled registers go through the port, or to DRAM when they don't fit the L2.

## How well it ranks (no refit)

Spearman rank correlation between predicted and measured times; regret = how much slower the model's pick ran than
the best measured kernel; top-5 = how many of the measured best five the model also puts in its best five.

| Set | Kernels | Rank corr. | Regret | Top-5 |
|---|---:|---:|---:|---:|
| 8192³, mixed knobs | 29 | 0.95 | 0.0 % | 5 |
| 4096×4096×1024, mixed | 29 | 0.88 | 0.0 % | 4 |
| 2048³, mixed | 29 | 0.84 | 0.0 % | 4 |
| 4096³, TMA sweep | 512 | 0.81 | 0.0 % | 2 |
| 1024³, mixed | 187 | 0.87 | 4.8 % | 1 |
| 1024³, TMA sweep | 512 | 0.73 | 1.7 % | 0 |
| 1024³, the 22 ablated cp.async kernels (warm) | 22 | 0.75 | 12.7 % | 2 |
| M=32, TMA sweep | 468 | 0.69 | 1.6 % | 0 |
| M=32, mixed | 182 | 0.64 | 1.5 % | 0 |
| M=128, mixed | 29 | 0.26 | 29.5 % | 0 |

Where it started this morning, with the same sets: 8192³ regret 140 %, 2048³ 94 %, 1024³ mixed 113 %, the
4096×4096×1024 rank correlation 0.13.

## Does it name the right bottleneck?

- **Against ncu, seven TMA picks: 6 of 7.** Tensor-bound at 4096³, 2048³, K=1024; DRAM-bound at M=32 and 8192³;
  load and latency at 1024³. The miss is M=128 (below).
- **Against the ablation, 22 cp.async kernels: 16 of 22.** All six misses are near-ties: loads beat compute by
  2 to 17 %.
- **DRAM traffic against ncu: 0.70 to 1.07 at all seven picks.** The old formula said 537 MB at 8192³; ncu counted
  3256; the simulation gives 2754. The cause: a wave of 48 blocks in panels of 16 rows touches 64 MB of A, more
  than the L2 holds, so A is read again every wave.

## What each mechanism bought

Each line is one change, in the order made, and the set it moved most.

1. Bank conflicts of the linear layout (design, no measurement): 1024³ mixed, regret 113 % to 0.1 %.
2. The launch-order L2 simulation: 8192³ traffic within 15 % of ncu (was 6x off).
3. A wave's DRAM traffic belongs to that wave: M=32 rank 0.27 to 0.71.
4. The cp.async loop's per-k-tile cost and real tile rates (loader probe): ablation bottleneck 5/22 to 16/22;
   8192³ regret 22 % to 0.
5. Spilled registers: 4096³ rank 0.69 to 0.76 (256x256 tiles spill and run 27x slower).
6. Streamed k-tiles wait a DRAM round trip per stage: ncu bottleneck 6/7; 4096³ regret to 0.

## What is still open

- **M=128, few SMs on a DRAM stream.** Kernels with 16 tiles (16 SMs) run 233 to 269 µs; the model says 186. Their
  measured time matches compute plus the DRAM wait added, not overlapped. So with data streaming from DRAM the mma
  and the wait seem to overlap less than with L2 data. The probe that would settle it: the loader plus an mma loop, data
  from DRAM, at 16 and 48 SMs.
- **Near-ties at 1024³.** In the 22 ablated kernels the model's top pick is 12.7 % off the best; loads and compute
  are within 20 % of each other there, and the overlap law (median exponent 2, range 1.5 to 6) is the loosest number.
- **Absolute times under do_bench are 10 to 27 % off.** The dirty-L2 cost is measured with a plain stream; how it
  spreads over a kernel's waves is assumed proportional to each wave's reads.
- **TMA with small tiles** moves about 40 GB/s per SM, not the 33 measured on big tiles. It only affects kernels that
  lose anyway.

## Oct 20, later: the instruction-level SM simulator

`agentic/gb10/sim/` runs a kernel's real machine code (nvdisasm with encodings) on one simulated SM. It computes no
values. Each warp issues under the compiler's own control bits (stall count, the six dependency counters it sets and
waits on); four sub-partitions issue one instruction per cycle; the tensor pipe, the shared-memory array, the memory
issue slot, the copy path, copy groups and the block barrier are modeled as resources and events. Latencies and
throughputs come from probes (`instlat.cu`, `cpq.cu`, `loopsteps.cu`); two small values are assumptions and marked.

Against the 22 ablated cp.async kernels, each in three variants (`python -m agentic.gb10.sim.validate`):

| Variant | Median error | Within 10 % |
|---|---:|---:|
| mma alone (one block) | 6 % | 20 of 22 |
| Loads alone (full grid) | 15 % | 6 of 22 |
| Whole kernel | 19 % | 6 of 22 |

With two or more blocks per SM the whole kernel is within 13 %; with one block per SM the simulator is 12 to 28 %
too fast. The probes behind that split:

- One block alone, the copy loop TileLang generates moves 78 GB/s, the SM's port. Commit, wait and both barriers cost
  nothing measurable (`loopsteps.cu`). TileLang's own loads-only kernels confirm it (76-80 GB/s with one block).
- With every SM loading, the L2 delivers 2.07-2.10 TB/s for this traffic; each SM gets its share past about 24 SMs.
- Withdrawn: the 0.22 µs fixed cost per k-tile (`cpasync_t0_us`) and the 16 B per clock copy limit. Both came from
  my own probes: run-time integer division in `tileload.cu`'s address code, and lanes 128 bytes apart in
  `instlat.cu`'s writes (bank conflicts). The analytical model still carries t0; its ranking was tested with it, but
  its explanation was wrong.
- Open: at 32 SMs TileLang's loads-only kernels get 42 GB/s per SM where the lean probe gets 65. TileLang's blocks
  read the same A rows and B columns as their neighbours at the same time; the probe's blocks don't. The hypothesis is
  contention on the L2 slices holding those shared lines. The probe to test it: the lean loop with blocks sharing
  panels the way a GEMM grid does.

### Oct 20, the copy-path interference and what is left

- **Same-line contention: refuted.** `shareload.cu` loads GEMM-shaped tiles with neighbours sharing A and B, once in
  lockstep (as the kernels do) and once staggered in k: identical times for all six tile shapes. It also matches the
  simulator's loads-only time (12.8 µs per block at 32 blocks, simulator 12.5).
- **Clock throttling: refuted.** clock64 against globaltimer inside the real kernels: 2.39-2.53 GHz at full grid.
- **Issue interference from other warps: small.** `interfere.cu`: a copy from another warp on the same sub-partition
  costs the mma stream 2.3 cycles (global load 1.9, ldmatrix 0.4, integer add 0).
- **Found: the load unit's address stage.** ncu on the whole kernel against its mma-only version (one block):
  long_scoreboard 0.80 vs 0.00, mio throttle 1.12 vs 0.28, lg throttle 1.78 vs 0.98, barrier 1.19 vs 0.28, tensor
  active 68 % vs 84 %, no bank conflicts. The machine code explains it: after its copies, each warp updates addresses
  with instructions that wait for the copies to read their address registers, which happens only when the load unit
  processes them. Rule: one 128-byte line per cycle per SM; a copy with 64-byte rows touches 8 lines, wider rows 4.
  Predicted loss per k-tile for one block: 320 / 160 / 96 cycles; measured 390 / 174 / 79. In the simulator: one block
  alone, whole kernels now within −13 to +2 % (five kernels timed alone on an idle chip).
- **Open: a chip-level step.** A TileLang kernel's per-block time rises about 8 % as soon as a second SM is busy (even
  with no loads), then stays flat to 24 blocks; the final C store also slows from 2.4 to 5.4 µs with 32 blocks storing
  at once. The pure-mma probe shows no such step, so it is something else TileLang's loop shares across SMs. Not
  identified.

### Oct 20, the store path; kernel-level result

- **The chip-level step is the epilogue store of C.** The store alone takes 2.37 µs with one block and 4.64 µs with
  2 to 24 blocks (flat), 5.5 at 32. `storepat.cu`: the two SMs of a pair (TPC) share one store port (every pattern
  doubles from 1 to 2 busy SMs, blocks 0 and 1 land on SMs 0 and 1), and the chip has a store limit of its own:
  0.31-0.38 TB/s for TileLang's stores, 1.4-1.5 TB/s for 16-byte coalesced stores.
- **TileLang's C store is a real cost on this chip.** It writes 4 bytes per lane, 8 rows x 16 bytes per instruction,
  so each 32-byte sector is filled in two halves by different instructions: 3.0 µs per 64 KB tile alone against 0.7
  for coalesced 16-byte stores. At 1024³ the 2 MB of C at ~0.35 TB/s is ~6 µs of a 35-40 µs kernel. Vectorized
  stores would cut it ~4x (from the probe; not yet tried in a TileLang kernel).
- **Kernel-level validation** (`kernel_level()` in `agentic/gb10/sim/validate.py`): each wave simulated with its own
  blocks per SM and busy SMs, plus launch, against the measured kernel time.

| Kernel time, 22 cp.async kernels at 1024³ | Median error | Worst | Within 10 % |
|---|---:|---:|---:|
| Whole kernel | 5 % | 16 % | 20 of 22 |
| Loads only | 6 % | 31 % | 12 of 22 |

  Rank correlation of whole-kernel times 0.80 (the 22 lie within 35-52 µs). The two misses are the smallest tiles
  with 3-4 blocks per SM (+12 %, +16 %), where the loads-only kernel is also slow in the simulator: the L2 share per
  SM with 4 resident blocks is the next number to measure.

What the simulator now contains, each piece with its probe: the compiler's issue control bits; tensor pipe 16
cycles per mma; ldmatrix 34-cycle latency, 4 cycles per x4 on the shared-memory array, 11.7 per warp; barrier 13 + 2
per warp; copies: one 128-byte line per cycle in the load unit's address stage (register read released there), 350
cycles to L2, 30.7 B per clock per SM port, 2.08 TB/s chip L2, writes into the shared array; stores: a port per SM
pair and a chip limit, by store width. Two values are assumptions (default latency of other variable-latency
instructions, 20 cycles; read-barrier release of non-memory instructions, 8 cycles).

### Oct 20, beyond 1024³ on the cp.async path

- **Blocks per SM do not change the L2's rate.** `l2share.cu`: 2.08-2.14 TB/s with 1, 2, 3 or 4 blocks resident on
  every SM, for 8-16 KB tiles. The two earlier misses came from my wave split (a 64-block second wave simulated as 48
  SMs x 2); sharing the L2 and store limits by blocks in flight fixes both (+16 % to +4 %, +12 % to +1 %): 21 of 22
  kernels at 1024³ within 10 %.
- **DRAM in the copy path, and traffic at k-slice level.** Copies miss the L2 in the proportion the traffic
  simulation gives for their wave; a miss pays the DRAM round trip and this SM's share of the DRAM rate. The traffic
  simulation now walks each wave's k-steps in lockstep through an LRU of 16.8 MB (`dram_waves_kslice` in
  `agentic/gb10/model.py`). Against ncu's L2 read misses on 18 kernels: 403/403, 409/413, 749/772, 725/767 MB at 4096³
  (the old panel-level count said 250-441), 3138/3219 at 8192³; median within 6 %. Panelled orders and one multi-wave
  decode kernel come out 20-35 % high: the real L2 keeps reused lines better than strict LRU over lockstep waves.
  Capacity is not the cause (16.8, 20 and 24 MB all give a median near 6 %).
- **25 new kernels at five shapes**, warm (`python -m agentic.gb10.sim.shapes`):

| Shape | Kernels | Median error | Rank correlation | Regret of the simulator's pick |
|---|---:|---:|---:|---:|
| 4096³ (with and without panels) | 7 | 7 % | 0.93 | 0 % |
| 4096×4096×1024 | 5 | 5 % | ties | 0 % |
| 2048³ | 5 | 10 % | ties | 0 % |
| M=32 | 3 | 3 % | 0.50 | 0.6 % |
| M=128 | 5 | 3 % | 0.70 | 0.6 % |

  All 25: median 7 %, 20 within 10 %. Outlier: M=128 with 64x64 tiles (+43 %), the LRU over-count above (69 MB
  simulated, 51 MB measured).

### Oct 20, the TMA path and the SFU

- **Warp-specialized TMA kernels** (`agentic/gb10/sim/ws.py`, `python -m agentic.gb10.sim.ws_validate`): roles from
  thread-index comparisons evaluated per warp; TMA boxes through the SM's TMA engine (81 GB/s alone, 1.86 TB/s chip,
  from the 1024³ pick's loads-only variant without the store; the earlier 1.58 included the store); mbarrier waits as
  blocking waits matched by order; the register hand-off; k-loops of either shape (conditional back edge, or forward
  exit plus jump back) decided by the k-tiles remaining; out-of-range rows of a box move no data.

| TMA pick | mma only, one block | Whole kernel at its grid |
|---|---:|---:|
| 1024³ | +10 % | −10 % |
| 2048³ | +3 % | −4 % |
| 4096³ | 0 % | −16 % |
| 4096×4096×1024 | +14 % | −15 % |
| 8192³ | 0 % | −38 % |
| M=32 | +4 % | +21 % |
| M=128 | −2 % | +20 % |

  The compute side holds; the misses are on the memory side at large and decode shapes (DRAM traffic and its rate on
  the TMA path).
- **The special function unit** (`sfu.cu`): ex2, rsqrt, lg2, tanh 18-cycle latency (sin 22), 16 lane results per clock
  per SM, one warp instruction per 8 cycles per sub-partition; FFMA 124 per clock. Added to the simulator as a pipe
  per sub-partition; on the probe's own machine code it reproduces 4.0 / 15.8 / 16.0 / 16.0 at 1 / 4 / 8 / 16 warps
  (FFMA within 4 %). For attention at head dimension 128 the exp needs about one eighth of the tensor time.
- **cp.async at five shapes after the in-range fix:** M=32 rank 1.00, regret 0; all 25 median 7 %.

### Oct 20, the L2 policy, DRAM in detail, the warp scheduler

- **The L2 keeps reused data** (`l2policy.cu` with ncu). A 4-8 MB set that is read again survives a 64 MB stream
  passing through. A cyclic scan loses reuse slowly, from all hits at 14 MB to none at 40 MB, not all at once as LRU
  would. Streams that are never reused still hit half the time, because each fill brings two 32-byte sectors. No
  textbook policy (LRU, random, segmented LRU, insertion at the bottom, RRIP) reproduces all three at 24 MB
  (`agentic/gb10/l2policy_fit.py`). For GEMM traffic the k-slice LRU over 16.8 MB still matches ncu best (median 6 %
  over 18 kernels), so it stays; a variant that protects reused slices is the default and gives the same numbers.
- **DRAM in detail** (`dram.cu`, `cpuhog.c`):
  - Reads alone 242 GB/s, writes alone 197. A mix is served as each part at its own rate, within 4 %
    (1:1 → 220, 3:1 → 222, 1:3 → 217).
  - Latency grows with load: 419 ns idle, 1.35 µs with 4 SMs streaming, about 6 µs with 24 or more.
  - The CPU shares the LPDDR5X. While it copies with 2, 8 or 20 threads, the GPU's read rate drops from 237 to
    214, 171 or 138 GB/s. This is an input of the simulator (`CPU_DRAM_GBs`, zero for all measurements here).
- **Which DRAM rate a miss gets.** A misses come as short pieces of rows far apart (64 bytes at block_K 32): about
  204 GB/s. B misses are not like that: the column tiles in flight read neighbouring pieces of the same B rows at the
  same k-step, so DRAM sees one long run and streams it near 237. At 4096³ with 128x128 tiles, 34 MB of the 403 MB of
  misses are A and 369 MB are B. The simulator used to charge every miss the short-piece rate; now each wave's DRAM
  time is A at its run length, B at its run length, and C at the write rate when operands and output do not stay in
  the L2 together. The same C used to be counted as reads too; that double count is gone.
- **The warp scheduler is greedy** (`sched.cu`, `agentic/gb10/sim/schedtest.py`): a warp keeps issuing until it
  stalls, then the oldest ready warp goes. Replayed on the probe's machine code, greedy gives 14.91 cycles per add
  against 15.10 measured; round-robin is far off. The simulator already did this.
- **A producer loop that never ended.** In the 4096³ and 8192³ TMA kernels the producer's exit test sits two
  instructions before its jump back. The simulator only recognised an exit right next to the jump, so the producer
  spun until the cycle limit (the earlier −16 % and −38 % were from runs cut short at the old limit). Fixed by looking
  ahead to the jump.

Results after these changes:

| Set | Before | Now |
|---|---|---|
| 25 cp.async kernels, five shapes | median 7 %, 20 within 10 % | median 4 %, 22 within 10 % |
| the 7 kernels at 4096³ | +3 to +16 % | −6 to +4 % |
| 22 cp.async kernels at 1024³ | median 5 %, 21 within 10 % | unchanged |

| TMA pick | Whole kernel at its grid, before | Now |
|---|---:|---:|
| 1024³ | −10 % | −10 % |
| 2048³ | −4 % | 0 % |
| 4096³ | did not finish | +3 % |
| 4096×4096×1024 | −15 % | −3 % |
| 8192³ | did not finish | +5 % |
| M=32 | +21 % | +20 % |
| M=128 | +20 % | +19 % |

The two decode-shape TMA picks are still 20 % slow; their one-block math is right (+4 %, −2 %), so the gap is in how
few blocks share DRAM there. The analytical model's ncu bottleneck test now expects 5 of 7: the M=128 pick matched
before only because A rows past M were counted twice.

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

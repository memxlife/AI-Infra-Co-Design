# A mechanistic timing model for TileLang GEMM on the GB10 (round 3, step 2c)

Every constant below was measured alone, by removing one mechanism from the solver's own kernel at each of the seven shapes and timing what was left (`agentic/tilelang/ablation/`: the harness, the variant patcher, the Spark driver; raw timings in `calibration/spark-6951/ablation/ablation.jsonl`, the per-shape mechanism numbers in `mechanisms.json`, the constants with their sources in `calibration/spark-6951/chip_tilelang.json`, the model in `agentic/tilelang/model/mech.py`). Nothing is fitted. Error is a residual with a name.

## The method, as run

The TileLang-generated CUDA of each pick is compiled by nvcc with TileLang's own flags into a harness that builds the same two TMA tensor maps once, fills A and B with random halves, and launches with the same grid, block and dynamic shared memory. Variants of that kernel: `kt0` (both k-loops run 0 tiles: launch, block setup, store), `kt0_nostore` (the store loop runs 0 times), `noload` (TMA loads and expect-tx removed, the producer arrives on the barrier so the consumer runs on stale shared memory), `nocompute` (ldmatrix + mma removed, barrier protocol kept), and `full`. Every variant stamps `%globaltimer` at block entry and after the store, so a block's own time is known beside the CUDA-event time of the launch. Grids: 1 block, 48, 96, the full grid. One finding changed how the acceptance is read, so it comes first.

## Finding 0: TileLang's `do_bench` times a cold L2

For the 1024³ pick, `do_bench` reports 73.6 µs; the same call back-to-back takes 47.8 µs of wall-clock and the harness measures 45.1 µs. `tilelang/profiler/bench.py` writes a 256 MB flush buffer between timed calls (`fast_flush`, `cache_size=256`), so each timed kernel starts with a cold L2 (6.3 MB of A and B from DRAM: 27 µs at 235 GB/s) and with dirty lines draining. Encoding a tensor map costs 0.47 µs and is not the cause. Every sweep, sample and autotuner number in this project was taken through `do_bench`; the cuBLASLt harness and this harness time warm, back-to-back launches. The two conditions are both legitimate and were mixed in every comparison so far. Warm against warm, the picks against cuBLASLt: 1024³ 45.1 vs 34.2 µs (+32 %, was +83 %); M=32 124.0 vs 148.9 µs (**−17 %**, was +39 %); 4096³ 1511 vs 1456 µs (+4 %); 2048³ 203 vs 184 µs (+10 %); M=128 162 vs 155 µs (+4 %); 4096×4096×1024 449 vs 401 µs (+12 %); 8192³ 15136 vs 11199 µs (+35 %, a different pick from the one timed earlier). The decode shape, which three rounds called unmatchable, is ahead of cuBLASLt when both are timed the same way.

## The constants, each with its probe

| constant | value | probe | B-suite cross-check |
|---|---|---|---|
| launch (1 empty block, events) | 2.59 µs | kt0_nostore, 1 block, CUDA-event median | contract launch_s 2.02 µs: agrees |
| block setup (descriptor prefetch, barrier init, register reconfig) | 0.10 µs | kt0_nostore, 1 block, %globaltimer span | — |
| block issue rate | 8 ns per block | kt0_nostore full grid at 8192³ (2048 empty blocks) minus 1 block | — |
| C store | 0.028 µs per KB of tile per block (0.83 µs for 64x128, 2.78 for 256x128) | kt0 minus kt0_nostore, 1 block | — |
| tensor rate, one block alone | 2.48 TF/s per SM (picks 2.16–2.70) | noload, 1 block: (span − setup) / KT over the k-tile's FLOPs | B31 2.52: agrees (86–107 %) |
| TMA load rate, one block alone | 81 GB/s per block (77–87 on every pick) | nocompute, 1 block | B8 31 GB/s per SM for one resident tile: **disagrees by 2.6x**; B8 measured a single un-pipelined tile, the kernel keeps 2–4 in flight |
| L2 → SM rate, chip, 48 blocks, data in L2 | 1580 GB/s | nocompute at 48 blocks, grid bytes over span | contract l2_bytes_per_s 1.02 TB/s streaming, tma_mix 1.74–1.94: within the TMA range |
| DRAM rate the pipeline reaches | 270 GB/s | full kernel at M=32, warm: 33.5 MB once in 124 µs | contract B_HBM 235 (B13 streaming): **disagrees by 15 %**, datasheet 273: the TMA pipeline reaches the datasheet |
| sync per k-tile (barrier hand-off when the pipeline overlaps) | 90 ns (picks 16–306 ns) | full − max(compute alone, loads under the full grid), per block, over KT | B7 barrier 15 ns, mbarrier phase 95 cycles ≈ 40 ns: 1–7 barrier costs per k-tile |
| do_bench offset | 25.8 µs at 1024³ | do_bench vs wall-clock back-to-back | — (a measurement artefact, named) |

## The mechanisms, one by one

1. **Launch and prologue.** An empty 1-block kernel: 2.59 µs by events, of which the block itself is 0.10 µs. 48 empty blocks: 2.53 µs; 2048 empty blocks (8192³): 18.9 µs, so blocks issue at about 8 ns each. The fitted 21.6 µs "prologue" is none of this: it is the do_bench offset (finding 0) that the one-wave 1024³ rows carried and the pair fit attributed to the launch.
2. **Epilogue.** The C store per block: 0.83 µs for a 64x128 tile (16 KB), 2.78 µs for 256x128 (64 KB), about 28 ns per KB; the k1024 pick's store did not register above its setup (0.00 µs by stamps, 167 µs across the grid by events: the stamps bracket the consumer's loop and the store is overlapped with the next block's start).
3. **Steady-state compute.** Per k-tile, one block alone: 0.936 µs for 64x128x128 (2.24 TF/s per SM), 0.847 µs for 128x256x32 (2.48), 1.637 µs for 256x128x64 (2.56), 0.777 µs for 256x128x32 (2.70). Pipe utilisation against B31's 2.52: 86 % to 107 %; the kernel's mma issue is at the benchmark's rate, and above it for the 256x128 tiles (the B31 loop was not the best possible).
4. **Data movement, one block alone.** 0.636 µs per 48 KB k-tile at 2 stages (1024³), 0.299 µs per 24 KB at 3 stages (4096³), 0.514 µs per 40 KB (M=32): 77–87 GB/s per block whatever the depth, which says the single block is bound by TMA issue-and-land, not by bytes in flight.
5. **Shared rate under load**, loads only, chip GB/s and per-block GB/s by resident blocks:

   - 1024: 1 blocks: 65 GB/s chip, 77 per block, 48 blocks: 1486 GB/s chip, 36 per block, 96 blocks: 1325 GB/s chip, 32 per block, 128 blocks: 1314 GB/s chip, 34 per block
   - 4096: 1 blocks: 81 GB/s chip, 82 per block, 48 blocks: 905 GB/s chip, 19 per block, 96 blocks: 1324 GB/s chip, 30 per block, 512 blocks: 1187 GB/s chip, 28 per block
   - m32: 1 blocks: 77 GB/s chip, 80 per block, 32 blocks: 355 GB/s chip, 11 per block
   - 2048: 1 blocks: 71 GB/s chip, 81 per block, 48 blocks: 1580 GB/s chip, 36 per block, 96 blocks: 1632 GB/s chip, 37 per block, 128 blocks: 1492 GB/s chip, 39 per block
   - 8192: 1 blocks: 81 GB/s chip, 81 per block, 48 blocks: 657 GB/s chip, 14 per block, 64 blocks: 639 GB/s chip, 17 per block, 2048 blocks: 854 GB/s chip, 19 per block
   - k1024: 1 blocks: 86 GB/s chip, 87 per block, 32 blocks: 2253 GB/s chip, 74 per block, 96 blocks: 1551 GB/s chip, 49 per block, 512 blocks: 1216 GB/s chip, 43 per block
   - m128: 1 blocks: 77 GB/s chip, 78 per block, 48 blocks: 1719 GB/s chip, 37 per block, 64 blocks: 1197 GB/s chip, 34 per block

   The L2-resident shapes (1024³, 2048³, K=1024, M=128) hold 1.2–1.7 TB/s chip-wide from 48 blocks up. The DRAM-bound ones fall to what DRAM gives: M=32 with 32 blocks moves 355 GB/s of L2 traffic (B at 270 GB/s plus A re-reads from L2); 8192³ with 2048 blocks 854 GB/s of L2 traffic over 211 GB/s of DRAM. Per block the rate drops from 80 GB/s alone to 11–19 GB/s when the chip is saturated: the M=32 "per-block cap" of round 3 was the DRAM ceiling divided by the block count, and the 1.82 µs latency it implied measured nothing.
6. **L2 reuse and traversal order** (4096³, the pick with panel 16 and with panel 0, ncu): DRAM 232 MB and L2 hit 86.2 % against 407 MB and 75.8 %; time 1650 against 2090 µs. The panel saves 175 MB and 440 µs; the time saved is 2.5x the bytes saved at the DRAM rate, so the cost of the plain order is mostly the latency of the extra misses stalling the loads, not their bandwidth. The model counts the bytes (the traffic count gives 232 and 407 MB from the plain re-read rule: within 6 % of ncu here); it does not yet price the latency.
7. **Overlap.** Per block, full kernel against max and sum of its parts (setup + compute alone + loads under the full grid):

   - 1024: measured 14.2 µs; max(parts) 12.4, sum(parts) 19.9; the gap to max is 1.7 µs = 218 ns per k-tile
   - 4096: measured 128.0 µs; max(parts) 112.9, sum(parts) 221.3; the gap to max is 15.1 µs = 118 ns per k-tile
   - m32: measured 118.0 µs; max(parts) 116.4, sum(parts) 131.9; the gap to max is 1.6 µs = 49 ns per k-tile
   - 2048: measured 65.1 µs; max(parts) 55.3, sum(parts) 96.1; the gap to max is 9.8 µs = 306 ns per k-tile
   - 8192: measured 341.6 µs; max(parts) 339.6, sum(parts) 550.8; the gap to max is 2.0 µs = 16 ns per k-tile
   - k1024: measured 27.9 µs; max(parts) 25.0, sum(parts) 43.2; the gap to max is 2.9 µs = 90 ns per k-tile
   - m128: measured 85.5 µs; max(parts) 78.7, sum(parts) 139.2; the gap to max is 6.7 µs = 53 ns per k-tile

   Every pick sits at max, not sum: the pipeline overlaps. What is left over max is 1–15 % of the block, 16–306 ns per k-tile, one to seven barrier costs; largest where the block is short and the tile big (2048³: 306 ns).
8. **Wave quantisation and the scheduler.** TileLang compiles of the 1024³ tile at one row of 47 / 48 / 49 tiles: 109.5 / 112.6 / 120.0 µs, and 95 / 96 / 97: 184.1 / 184.3 / 184.3 µs (do_bench, so with the offset). The 49th block adds 7 µs, half a block; the 97th adds nothing: no ceil() step. But the per-block stamps of the full K=1024 kernel (512 blocks of 27.5 µs) show the SMs draining to zero between waves (running blocks sampled over time: 45, 48, 0, 48, 48, 4, 48, 0, 48, …; wave period 42 µs for 27.5 µs blocks) and only 35 of 48 SMs busy on average, while the 1024³ kernel (128 blocks of 13 µs) keeps 41.5 busy. Empty blocks show no such gap (0.6 µs per wave). This is the K=1024 residual below.

## Assembly

`mech.predict(h, shape)`: expected = launch + blocks × issue + waves_eff × T_block, T_block = setup + KT × (max(compute, load under load) + sync) + store, waves_eff = tiles / 48 (fractional, from mechanism 8), the load under load = max(one block alone, L2 chip rate / resident blocks, DRAM share / resident blocks), DRAM bytes from the plain re-read count (mechanism 6); floor = max(expected, DRAM bytes / DRAM rate). Constants from `chip_tilelang.json`; for a pick with its own ablation its tile's numbers are used, otherwise the chip medians.

## Acceptance

**The seven picks, warm (harness) timings, no fitting:**

| shape | measured | expected (own ablation) | residual | expected (chip medians) | residual |
|---|---|---|---|---|---|
| 1024 | 45.1 | 42.6 | -5.6% | 38.8 | -13.9% |
| 4096 | 1510.6 | 1329.5 | -12.0% | 1306.9 | -13.5% |
| m32 | 124.0 | 131.1 | +5.7% | 132.2 | +6.6% |
| 2048 | 203.3 | 177.2 | -12.8% | 160.9 | -20.8% |
| 8192 | 15136.1 | 12209.7 | -19.3% | 12678.4 | -16.2% |
| k1024 | 449.0 | 304.5 | -32.2% | 346.7 | -22.8% |
| m128 | 162.1 | 144.7 | -10.8% | 151.8 | -6.4% |

Under 10 % on 1024³, M=32 and M=128 with the pick's own numbers; 12–13 % on 4096³ and 2048³; 19 % on 8192³; 32 % on K=1024.

**The sweeps and samples, do_bench (cold) timings, chip-median constants:** mean signed error and mean absolute error against do_bench, then the same with the 25.8 µs offset removed, then the share of points where the expectation exceeds the measurement:

| shape | kernels | vs do_bench | offset removed | expected > measured |
|---|---|---|---|---|
| 1024³ | 646 | −44 %, 44 % | −13 %, 19 % | 0 % |
| 4096³ | 504 | +12 %, 19 % | +13 %, 19 % | 67 % |
| M=32 | 595 | −33 %, 33 % | −24 %, 24 % | 0 % |
| 2048³ | 20 | +7 %, 28 % | +17 %, 33 % | 60 % |
| 8192³ | 20 | +13 %, 28 % | +13 %, 28 % | 50 % |
| K=1024 | 20 | +18 %, 33 % | +24 %, 37 % | 70 % |
| M=128 | 20 | −26 %, 33 % | −16 %, 27 % | 10 % |

The samples were timed cold and the constants were measured warm; the offset is a per-call constant only where the kernel is L2-resident and short (1024³), and at M=32 the dirty-line drain adds something the single offset does not cover (−24 % after it). On the big shapes the expectation sits above most sample points: the chip-median sync and load constants come from the picks' tiles, and the sample's smaller tiles (64-wide, 1-2 stages) run their loads faster per byte than the medians say.

## Residuals, named

- **4096×4096×1024, −32 % (warm).** The block is right (27.5 µs measured, 27.9 modelled); the kernel is 449 µs against 298 for continuous packing because the SMs drain between waves: 48 blocks start together, end within 5 µs of each other, and the next 48 start about 14 µs later. Empty blocks do not show it. Hypothesis: the gap is the block's teardown and the next block's shared-memory carve-out at 98 KB per block and 640 threads (the K=1024 pick has the largest smem and thread count of the seven); probe: kt0 with loads only at full grid, and the same pick at 2 stages.
- **8192³, −19 %.** The traffic count gives 537 MB where ncu counted 3256 MB (step 2b): with panel 16 a panel's A rows are 64 MB and are re-read about 21 times; the "A once" rule was measured at 4096³ only. Mechanism 6 at 8192³ is the probe (two ncu runs).
- **4096³ and 2048³, −12 %.** Both have 11 and 3 waves of blocks that finish unevenly (block max 16–31 µs above the mean in the stamps), and the 4096³ pick pays the panel-0-like misses on A at panel 16 that mechanism 6 priced only as bytes. Hypothesis: the load latency stall per k-tile under L2 misses, visible in the panel experiment as 2.5x the byte cost; probe: nocompute at the full grid with panel 0 vs 16.
- **M=32, +6 %; M=128, −11 %.** The DRAM rate is the datasheet's and the kernel reaches it; the remaining 6–11 % is the partial last wave (M=128: 64 blocks, the second wave 16) and the A re-reads at M=128 (169 MB of L2 traffic, 101 counted).
- **The fitted constants of round 3, in the light of the probes:** 21.6 µs prologue = the do_bench offset; 1.82 µs per block = the DRAM ceiling divided by the block count; the per-tile start-up of series A (2–9 µs) = what the single-wave 1024³ rows pay to warm a cold L2 plus the real 0.1–0.2 µs setup.

## What the paper gets

A model whose every term names its probe and its counter cross-check; two measurements that overturn earlier conclusions (the decode shape is 17 % faster than cuBLASLt warm; the B-suite's DRAM rate is 15 % under what the TMA pipeline reaches); a residual table with a hypothesis and a probe per shape; and a timing-method finding that applies to every TileLang number ever published with `do_bench`.

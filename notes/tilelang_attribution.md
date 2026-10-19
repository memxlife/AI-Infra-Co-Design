# Attribution: the model's terms against the counters, per shape (round 3, step 2b)

The solver's in-sample pick at each shape, one launch under ncu at real clocks (`--clock-control none`), the seven profiles in `runs/tilelang/attr/`. Left column: the model's terms as `predict()` computes them, in microseconds. Right column: the counter each term should match, converted the same way: tensor time = tensor-pipe-active fraction × SM active time; DRAM time = (L2 lookup-miss sectors × 32 B) / 235 GB/s; L2 time = (L2 read + write sectors × 32 B) / 1.94 TB/s; prologue = (max elapsed cycles − mean active cycles) / clock. "Attributed" = prologue counter + the largest of tensor, DRAM and L2 counter times, which is how the model combines them; "unexplained" = ncu duration − attributed. Stall shares split the active time per warp by `smsp__average_warps_issue_stalled_*` (per issued instruction). No term was refitted for this note.

## 1024³: pick 64x128x128, 2 stages, 128 threads, panel 8

ncu duration 51.8 µs (event timing 73.7 µs)
; model 54.9 µs, longest term tensor; 3 waves of 128 blocks; clock 2395 MHz; SM active 45.5 µs; tensor pipe 40 % of active; DRAM 5 MB at 92 GB/s (model charges 6 MB); L2 55 MB, hit 91 %; occupancy 10 % (4.9 warps/SM, 1 block/SM); issue slots used 7 %; shared-memory bank conflicts 0.

| term | model (µs) | counter (µs) | model vs counter |
|---|---|---|---|
| prologue (once per launch) | 21.6 | 6.3 | +243% |
| tile start-up x waves | 5.0 | none |  |
| compute (tensor) x waves | 26.2 | 18.2 | +44% |
| copy x waves | 19.6 | none |  |
| DRAM (bytes / B_HBM, per-block cap) | 26.7 | 20.2 | +32% |
| L2 (bytes / B_L2) | 24.4 | 26.5 | -8% |
| launch | 2.0 | none |  |
| **attributed** (prologue counter + max of tensor, DRAM, L2 counters) | | 32.8 | |
| **unexplained** (duration − attributed) | | 19.0 | |

Stall shares of the active time, per warp: sleeping 20 µs, wait 13 µs, selected 3 µs, lg_throttle 2 µs, drain 2 µs, long_scoreboard 1 µs.

## 4096³: pick 128x256x32, 3 stages, 512 threads, panel 16

ncu duration 1720.0 µs (event timing 1568.7 µs)
; model 1437.6 µs, longest term tensor; 11 waves of 512 blocks; clock 2398 MHz; SM active 1936.4 µs; tensor pipe 60 % of active; DRAM 234 MB at 136 GB/s (model charges 134 MB); L2 1677 MB, hit 86 %; occupancy 30 % (14.4 warps/SM, 1 block/SM); issue slots used 21 %; shared-memory bank conflicts 7074.

| term | model (µs) | counter (µs) | model vs counter |
|---|---|---|---|
| prologue (once per launch) | 21.6 | -215.5 |  |
| tile start-up x waves | 97.7 | none |  |
| compute (tensor) x waves | 1316.2 | 1166.3 | +13% |
| copy x waves | 576.1 | none |  |
| DRAM (bytes / B_HBM, per-block cap) | 570.3 | 995.8 | -43% |
| L2 (bytes / B_L2) | 781.7 | 813.7 | -4% |
| launch | 2.0 | none |  |
| **attributed** (prologue counter + max of tensor, DRAM, L2 counters) | | 950.8 | |
| **unexplained** (duration − attributed) | | 769.2 | |

Stall shares of the active time, per warp: math_pipe_throttle 767 µs, wait 371 µs, sleeping 346 µs, selected 96 µs, not_selected 67 µs, long_scoreboard 61 µs.

## 32×4096×4096: pick 32x128x128, 2 stages, 128 threads, panel 4

ncu duration 176.1 µs (event timing 210.9 µs)
; model 168.4 µs, longest term dram; 1 waves of 32 blocks; clock 2519 MHz; SM active 95.0 µs; tensor pipe 9 % of active; DRAM 34 MB at 193 GB/s (model charges 34 MB); L2 43 MB, hit 20 %; occupancy 13 % (6.1 warps/SM, 1 block/SM); issue slots used 2 %; shared-memory bank conflicts 0.

| term | model (µs) | counter (µs) | model vs counter |
|---|---|---|---|
| prologue (once per launch) | 21.6 | 81.2 | -73% |
| tile start-up x waves | 0.0 | none |  |
| compute (tensor) x waves | 14.1 | 8.7 | +62% |
| copy x waves | 21.8 | none |  |
| DRAM (bytes / B_HBM, per-block cap) | 144.8 | 144.1 | +0% |
| L2 (bytes / B_L2) | 20.4 | 20.7 | -2% |
| launch | 2.0 | none |  |
| **attributed** (prologue counter + max of tensor, DRAM, L2 counters) | | 225.3 | |
| **unexplained** (duration − attributed) | | -49.2 | |

Stall shares of the active time, per warp: sleeping 82 µs, wait 7 µs, selected 1 µs, long_scoreboard 1 µs, drain 1 µs, short_scoreboard 0 µs.

## 2048³: pick 256x128x64, 2 stages, 256 threads, panel 0

ncu duration 293.7 µs
; model 225.5 µs, longest term tensor; 3 waves of 128 blocks; clock 2444 MHz; SM active 239.8 µs; tensor pipe 60 % of active; DRAM 20 MB at 69 GB/s (model charges 25 MB); L2 218 MB, hit 91 %; occupancy 20 % (9.7 warps/SM, 1 block/SM); issue slots used 10 %; shared-memory bank conflicts 1420.

| term | model (µs) | counter (µs) | model vs counter |
|---|---|---|---|
| prologue (once per launch) | 21.6 | 54.1 | -60% |
| tile start-up x waves | 20.3 | none |  |
| compute (tensor) x waves | 181.6 | 143.0 | +27% |
| copy x waves | 78.6 | none |  |
| DRAM (bytes / B_HBM, per-block cap) | 106.9 | 86.0 | +24% |
| L2 (bytes / B_L2) | 97.7 | 105.7 | -8% |
| launch | 2.0 | none |  |
| **attributed** (prologue counter + max of tensor, DRAM, L2 counters) | | 197.1 | |
| **unexplained** (duration − attributed) | | 96.6 | |

Stall shares of the active time, per warp: sleeping 73 µs, wait 57 µs, math_pipe_throttle 53 µs, lg_throttle 22 µs, selected 10 µs, misc 3 µs.

## 8192³: pick 256x128x64, 2 stages, 512 threads, panel 16

ncu duration 15430.0 µs
; model 10724.1 µs, longest term tensor; 43 waves of 2048 blocks; clock 2509 MHz; SM active 15803.4 µs; tensor pipe 56 % of active; DRAM 3256 MB at 211 GB/s (model charges 537 MB); L2 13153 MB, hit 75 %; occupancy 34 % (16.4 warps/SM, 1 block/SM); issue slots used 11 %; shared-memory bank conflicts 15647.

| term | model (µs) | counter (µs) | model vs counter |
|---|---|---|---|
| prologue (once per launch) | 21.6 | -369.4 |  |
| tile start-up x waves | 291.4 | none |  |
| compute (tensor) x waves | 10409.1 | 8917.8 | +17% |
| copy x waves | 4504.4 | none |  |
| DRAM (bytes / B_HBM, per-block cap) | 2281.1 | 13832.7 | -84% |
| L2 (bytes / B_L2) | 6253.6 | 6383.9 | -2% |
| launch | 2.0 | none |  |
| **attributed** (prologue counter + max of tensor, DRAM, L2 counters) | | 13463.3 | |
| **unexplained** (duration − attributed) | | 1966.7 | |

Stall shares of the active time, per warp: sleeping 6132 µs, math_pipe_throttle 5318 µs, wait 2073 µs, short_scoreboard 557 µs, selected 415 µs, not_selected 316 µs.

## 4096×4096×1024: pick 256x128x32, 4 stages, 512 threads, panel 0

ncu duration 466.9 µs
; model 431.0 µs, longest term tensor; 11 waves of 512 blocks; clock 2419 MHz; SM active 472.1 µs; tensor pipe 61 % of active; DRAM 31 MB at 67 GB/s (model charges 50 MB); L2 468 MB, hit 94 %; occupancy 32 % (15.2 warps/SM, 1 block/SM); issue slots used 13 %; shared-memory bank conflicts 84804.

| term | model (µs) | counter (µs) | model vs counter |
|---|---|---|---|
| prologue (once per launch) | 21.6 | -4.9 |  |
| tile start-up x waves | 74.5 | none |  |
| compute (tensor) x waves | 332.8 | 289.0 | +15% |
| copy x waves | 144.0 | none |  |
| DRAM (bytes / B_HBM, per-block cap) | 213.9 | 132.7 | +61% |
| L2 (bytes / B_L2) | 195.4 | 227.0 | -14% |
| launch | 2.0 | none |  |
| **attributed** (prologue counter + max of tensor, DRAM, L2 counters) | | 284.1 | |
| **unexplained** (duration − attributed) | | 182.9 | |

Stall shares of the active time, per warp: math_pipe_throttle 167 µs, sleeping 95 µs, wait 69 µs, lg_throttle 52 µs, short_scoreboard 21 µs, selected 15 µs.

## 128×4096×4096: pick 256x64x32, 2 stages, 128 threads, panel 0

ncu duration 297.3 µs (event timing 258.9 µs)
; model 175.1 µs, longest term dram; 2 waves of 64 blocks; clock 2512 MHz; SM active 221.9 µs; tensor pipe 31 % of active; DRAM 36 MB at 120 GB/s (model charges 36 MB); L2 169 MB, hit 79 %; occupancy 12 % (5.5 warps/SM, 1 block/SM); issue slots used 5 %; shared-memory bank conflicts 0.

| term | model (µs) | counter (µs) | model vs counter |
|---|---|---|---|
| prologue (once per launch) | 21.6 | 75.4 | -71% |
| tile start-up x waves | 0.0 | none |  |
| compute (tensor) x waves | 143.5 | 69.6 | +106% |
| copy x waves | 87.3 | none |  |
| DRAM (bytes / B_HBM, per-block cap) | 151.5 | 152.2 | -0% |
| L2 (bytes / B_L2) | 48.9 | 82.1 | -41% |
| launch | 2.0 | none |  |
| **attributed** (prologue counter + max of tensor, DRAM, L2 counters) | | 227.6 | |
| **unexplained** (duration − attributed) | | 69.7 | |

Stall shares of the active time, per warp: sleeping 137 µs, wait 52 µs, selected 9 µs, long_scoreboard 7 µs, misc 3 µs, short_scoreboard 2 µs.

## What the tables say, per shape

Rule used: a term "disagrees" when model and counter differ by more than 10 %. "Prologue counter" is the idle time
(max elapsed − mean active); it is not measurable this way on the many-wave shapes (mean active exceeds max elapsed
there, so the row reads negative) and on the skinny shapes it measures idle SMs, not a prologue.

- **1024³.** Disagree: prologue (21.6 vs 6.3), compute (26.2 vs 18.2: the tensor pipe is active 40 % of the SM's
  time; the rate table says 56 %), DRAM (26.7 vs 20.2: the model charges 6 MB, the chip reads 5). Unexplained 19 µs;
  the dominant stall is *sleeping* (19.5 µs per warp of 45 active: warps parked on the TMA barrier), then *wait*
  (13.4, fixed-latency dependencies). Missing mechanism: the per-k-step wait for a TMA tile with two stages, which
  neither the compute nor the copy term holds.
- **4096³.** Disagree: DRAM (570 vs 996: the model charges 134 MB, the chip reads 234; with panel 16 the A row-panel
  of 16 × 128 rows × 4096 × 2 B = 16 MB does not stay in L2, so A is re-read, which the "A once" rule denies);
  compute agrees within 13 %. Unexplained 769 µs; dominant stall *math_pipe_throttle* (767 µs: warps queue for the
  tensor pipe, which is the compute term seen from the warp's side), then *wait* and *sleeping*. Missing: the DRAM
  time is not hidden under compute as max() assumes; part of the 996 adds.
- **M=32.** DRAM agrees (144.8 vs 144.1; 34 MB both sides at 193 GB/s). Prologue counter 81 µs is the 16 idle SMs
  (32 blocks on 48), not a prologue. Unexplained −49: double counted by that row. Dominant stall *sleeping* 82 of 95
  active µs: the block waits for TMA almost all the time. The per-block cap (1.82 µs) is not active at this pick
  (32 blocks × 80 KB in flight would allow 1.4 TB/s); the counters put the limit in the issue-and-wait of the copies.
- **2048³.** Disagree: prologue (21.6 vs 54), compute (182 vs 143), DRAM (107 vs 86). Unexplained 97 µs; stalls
  *sleeping* 73, *wait* 57, *math_pipe_throttle* 53. Same missing mechanism as 1024³: TMA waits per k-step.
- **8192³.** Disagree: DRAM by 6× (2281 vs 13833: the model charges 537 MB, the chip reads 3256; a panel's A rows
  are 64 MB, so A is re-read about 21 times), compute by 17 %. Unexplained 1967 µs; stalls *sleeping* 6132,
  *math_pipe_throttle* 5318. Missing: A re-reads when the panel's A does not fit L2, and the un-hidden part of them.
- **4096×4096×1024.** Disagree: compute (333 vs 289), DRAM (214 vs 133: 50 MB charged, 31 read: the
  re-read count over-charges here, the opposite error from 8192³). Unexplained 183 µs; stalls *math_pipe_throttle*
  167, *sleeping* 95, *wait* 69; 84,804 shared-memory bank conflicts, the only pick with any.
- **M=128.** DRAM agrees (151.5 vs 152.2). Disagree: compute (143.5 vs 69.6: the padded 256-row tile's flops are
  charged, but the pipe is active 31 %), L2 (49 vs 82: 169 MB moved, the padded rows). Prologue counter 75 µs is the
  16-block second wave's idle SMs. Unexplained 70 µs; *sleeping* 137 of 222.

**The two fitted constants.** 21.6 µs "prologue": the idle counter is 6 µs at 1024³ and ≤ 0 on every many-wave
shape; the constant is not a launch cost. It is what the one-wave 1024³ kernel pays in un-hidden pipeline fill and
tail (the *sleeping* and *wait* stalls at the start and end of each block), which the pair fit attributed to the
launch. 1.82 µs per block: at no pick is the per-block cap the binding term, and at the two skinny shapes the DRAM
rate (120–193 GB/s) comes with *sleeping* as the dominant stall, so the constant measures the copy pipeline's
issue-and-wait, not bytes in flight over a latency.

**The one mechanism every shape is missing**: time spent waiting on the TMA barrier per k-step (*sleeping*), which
is neither compute nor copy and which max(compute, copy) hides. Second: A re-reads at large panels (8192³), and the
mirror over-count at short K. Nothing was refitted; this is the list for step 2c.

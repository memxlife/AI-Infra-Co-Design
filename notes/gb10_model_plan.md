# GB10 GPU model: plan

*Oct 20 2026. Goal: a microarchitecture-level model of the GB10's GPU, grounded in how the chip works, measured by
microbenchmarks, no regression.*

## Start from what the chip is

Driver query and one clock probe (`agentic/tilelang/microbench/devquery.cu`, `clock.cu`):

| Part | Value | Source |
|---|---|---|
| SMs | 48, compute capability 12.1 | driver |
| SM clock under load | 2.52 GHz (driver max 2.418) | clock probe: clock64 against globaltimer, 48 blocks |
| Tensor pipe | mma.sync m16n8k16 fp16 in, fp32 accumulate: 1024 FLOP per clock per SM, so 2.58 TF/s per SM, 124 TF/s chip | Blackwell sm_12x design; no tcgen05, no wgmma |
| Shared memory | 100 KB per SM, 99 KB per block; 32 banks x 4 B = 128 B per clock | driver; design |
| Registers | 64 K per SM | driver |
| Occupancy caps | 1536 threads, 24 blocks per SM | driver |
| L2 | 24 MB | driver |
| DRAM | LPDDR5X, 256 bits at 8533 MT/s = 273 GB/s, shared with the CPU | driver |

## Measure only what the design does not state

1. Memory: unloaded latency, the per-SM port, chip L2 bandwidth, achievable DRAM read, the per-warp limit on
   loads in flight, and what a dirty L2 costs (write-back). One streaming kernel and one pointer chase.
2. Tensor pipe: the rate against warps per SM (how many warps hide the mma latency).
3. Fixed costs: launch, block setup, barrier, store (already measured by ablation, kept).

## Compose

Per SM and per k-tile: tensor time from FLOPs at the measured issue efficiency; shared-memory time from bytes at
128 B per clock; load time from Little's law (bytes in flight over latency) capped by the port, the SM's share of
L2, and its share of DRAM; the barrier. Pipeline overlap from the stage count: what the stages keep in flight
decides whether the load latency is hidden. Occupancy from the driver's caps. Waves from tiles over SMs times
blocks per SM. Every constant names its unit and its probe.

## What "good" means (Li, Oct 20)

Not absolute accuracy. Two things:

1. **Relative accuracy.** Within a shape, the model orders candidate kernels the way the GPU does. Measured by
   Spearman rank correlation over the measured candidates, and by regret: the measured time of the model's pick
   over the best measured time.
2. **Functional fidelity.** For each kernel the term the model says dominates (tensor, shared memory, load
   latency, port, L2, DRAM, barrier, waves) is the one the GPU spends its time on, checked by ablation or one ncu
   counter. That is what tells the optimizer which knob to turn.

Absolute error is reported, not gated.

## Test without refitting

The 22 ablated cp.async kernels, the seven TMA picks, the S2 picks, the skinny kernels cold and warm. Each is
predicted from the model alone. Per shape: rank correlation, regret, and the named bottleneck against the
ablation's or ncu's.

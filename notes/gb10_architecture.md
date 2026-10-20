# The GB10 GPU: an architect's view

*Oct 20 2026. The GPU of the NVIDIA GB10 in a DGX Spark (host spark-6951), compute capability 12.1, CUDA target
`sm_121a`, driver 580.159.03, CUDA 13.0. Every number here was measured on that machine by a small probe kernel, read
from the driver, or taken from the chip's design; each one names its source. Constants with their sources:
`calibration/spark-6951/gb10.json`. Probes: `agentic/tilelang/microbench/*.cu`. Raw probe output:
`calibration/spark-6951/*.txt`. The timing model that uses the constants: `agentic/gb10/model.py`.*

## How to read this

- **Units.** "Per clock" and "B/clk" mean per SM clock cycle. To turn cycles into time this note uses 2.52 GHz, the
  clock under load measured by `clock.cu` (Section 6.4 shows how much it moves). GB/s and TB/s are decimal. MiB is
  2^20 bytes. Probe outputs say "MB" loosely; this note keeps their figures as printed.
- **Sources.** A file name in brackets or in a table column is the probe or record behind the number. "Driver" means
  the CUDA device query. "Design" means the number follows from how the unit is built and was not measured alone.
- **Older suite.** Some numbers come from an earlier probe suite (Oct 15, `calibration/suite/b*.cu`, benchmarks
  B1-B40). The newer probes did not measure them again. They are marked "older suite". Where the two disagree, this
  note uses the newer probe.

A few terms used everywhere:

- **SM** (streaming multiprocessor): the GPU's core. It runs blocks of threads.
- **Warp**: 32 threads that issue each instruction together. **Block**: a group of warps that runs on one SM and
  shares that SM's shared memory. **Grid**: all the blocks of one kernel launch.
- **Sub-partition**: one quarter of an SM, with its own warp scheduler and its own math pipes.
- **Probe**: a small kernel built to time one unit. **Ablation**: a real GEMM kernel run with one part removed (no
  loads, no math, or no store) to time each part alone.
- **TileLang**: a Python tool that generates CUDA kernels from tile programs. Most real kernels measured here came
  from it. **cuBLASLt**: NVIDIA's GEMM library, used as the reference.
- **SASS**: the GPU's machine code. **PTX**: NVIDIA's portable assembly, which `ptxas` turns into SASS.
- **ncu**: Nsight Compute, NVIDIA's hardware-counter profiler.
- **Tile, k-tile, wave**: a tiled GEMM gives each block a bm x bn tile of C. Each loop step loads a k-tile (the bm x bk
  slice of A and the bk x bn slice of B) and multiplies it. When a grid has more blocks than fit on the chip at once,
  they run in waves.

## 1. The chip at a glance

The GB10 is one package with a GPU and a 20-core Arm CPU (`preflight.md`: 20 CPUs, aarch64). Both use the same 128 GB
of LPDDR5X memory, a low-power DRAM (driver: 128,520,015,872 bytes; the OS sees 119 GiB). There is no separate graphics
memory. The GPU's "DRAM" is the CPU's DRAM.

The GPU has 48 SMs. They work in pairs: SM 0 with SM 1, SM 2 with SM 3, and so on. NVIDIA calls such a pair a TPC.
The two SMs of a pair share one path for stores (Section 3.4). Each SM has its own path for loads (Section 3.1).
Behind the SMs sits a 24 MiB L2 cache, and behind that the memory.

```
  +----------------------------------------------------------------+
  | GPU                                                            |
  |  [SM0 SM1] [SM2 SM3] [SM4 SM5]  . . .  [SM46 SM47]   24 pairs  |
  |     each SM: its own load port to the L2, 32 B per clock       |
  |     each pair: one shared store path                           |
  |  +----------------------------------------------------------+  |
  |  | L2 cache, 24 MiB; about 16 MiB of it keeps re-read data  |  |
  |  | serves copies to all SMs at about 2.1 TB/s               |  |
  |  +----------------------------------------------------------+  |
  +-------------------------------+--------------------------------+
                                  |  237 GB/s read, 197 GB/s write
  +-------------------------------+--------------------------------+
  | LPDDR5X, 128 GB, 256 bit x 8533 MT/s = 273 GB/s on paper       |
  +-------------------------------+--------------------------------+
                                  |
                   CPU: 20 Arm cores, same memory
```

| Headline | Value | Source |
|---|---|---|
| SMs | 48, in 24 pairs | driver (`devquery.cu`, `preflight.md`); pairs: `storepat.cu` |
| SM clock under load | 2.52 GHz in short runs; 2.48-2.50 GHz over 20 s | `clock.cu`; `sustain.cu` |
| Tensor math, fp16 or bf16 in, fp32 sum | 1024 FLOP/clk per SM = 2.58 TFLOP/s per SM, 123.9 for the chip | design; `mma_rate.cu`, `mma_types.cu` |
| Tensor math, fp8, int8, fp4 | 2048 FLOP/clk per SM, 247.7 TFLOP/s for the chip | `mma_types.cu` |
| Tensor math held for 20 s | 121.5 TFLOP/s, then 120.4 at 62 °C | `sustain.cu` |
| FP32 fused multiply-add | 124 lanes/clk per SM, 30.0 TFLOP/s for the chip | `sfu.cu`, `alu.cu` |
| Registers | 65,536 x 32 bit per SM | driver |
| Shared memory | 100 KB per SM (99 KB max per block); 128 B/clk | driver; `smem_bw.cu` |
| Threads, blocks per SM | 1536 threads (48 warps), 24 blocks | driver |
| L2 | 24 MiB; re-read data stops hitting past about 16 MiB | driver; `l2policy.cu` |
| One SM's load port to L2 | 32 B/clk = 80.7 GB/s | `mlp_stream.cu` |
| L2 to all SMs | 1.8-2.25 TB/s for copies (TMA, cp.async); 3.47 TB/s for plain loads | ablation, `loopsteps.cu`, `tileload.cu`; `mlp_stream.cu` |
| DRAM | LPDDR5X, 256 bit at 8533 MT/s = 273 GB/s on paper | driver, datasheet (`preflight.md`) |
| DRAM read, write | 237 GB/s (87 % of paper), 197 GB/s | `mlp_stream.cu`, `dram.cu` |
| Latency, L1 / L2 / DRAM | 17 ns / 136 ns / 419-456 ns, idle | `chase.cu`, `dram.cu`, `mlp_stream.cu` |
| Kernel launch | 2.05 µs per kernel from a stream; 0.43 µs in a CUDA graph | `launch.cu` |
| Power | 68 W under 20 s of tensor math; 23.5 W streaming DRAM; 4-11 W idle | `sustain.cu`; `preflight.md` |

Three numbers frame most of this note. The tensor pipes do 124 TFLOP/s. The L2 feeds the SMs at about 2.1 TB/s
through the copy engines. DRAM delivers 237 GB/s. So a kernel needs about 60 FLOP per byte it pulls from the L2, and
about 523 FLOP per byte it pulls from DRAM, before math sets its speed (Section 7).

## 2. The SM

### 2.1 Its parts

```
                     one SM
   +-------------------------------------------------------------+
   |  4 sub-partitions, each with its own:                       |
   |    warp scheduler   1 warp instruction per clock, greedy     |
   |    FMA unit         FFMA, FADD: 1 clk;  IMAD, HFMA2: 2 clk   |
   |    ALU unit         LOP3, SHF, PRMT: 2 clk                   |
   |    SFU              ex2, rsqrt, lg2, tanh: 8 clk             |
   |    tensor pipe      one mma.sync every 16 clk                |
   |-------------------------------------------------------------|
   |  shared by all four:                                        |
   |    register file    64 K x 32 bit                           |
   |    load/store unit  address stage: one 128-B line per clock |
   |    shared memory    100 KB, 128 B per clock (32 banks x 4 B)|
   |    L1 cache         hit in about 45 clk, 64 B per clock     |
   |    TMA engine       copies whole tiles from a descriptor    |
   |    load port to L2  32 B per clock                          |
   +-----------------------------+-------------------------------+
                                 | stores and atomics
                     +-----------+------------+
                     | store path, one per    |
                     | SM pair, shared        |
                     +------------------------+
```

"N clk" is how long one warp instruction keeps that unit busy. The rest of this section gives the evidence for each
box. The four sub-partitions are the Blackwell SM design (`gb10.json`, "spec"); the rates below are measured.

### 2.2 Issue and the warp scheduler

Each sub-partition issues at most one warp instruction per clock, so an SM issues at most four (older suite, B17;
`icache.cu` tops out at 3.98 FFMA per clock per SM, which fits). The SM spreads its resident warps over the four
schedulers.

The scheduler is greedy. It keeps issuing from one warp until that warp stalls, then picks the oldest ready warp.
`sched.cu` shows it. A warp doing dependent integer adds takes 5.67 cycles per add alone. When another warp on the
same sub-partition issues independent FFMAs without pause, the add warp slows to 15.10 cycles per add. Replaying
greedy-then-oldest on the probe's own machine code gives 14.91; round-robin is far off (`gb10_model.md`). So a warp
that is always ready can starve its neighbour. Put warps whose latency matters on different sub-partitions.

### 2.3 Registers and occupancy

- 65,536 32-bit registers per SM (driver). They are handed out per warp in units of 256, which is 8 per thread
  (`gb10.json`, design).
- At most 1536 threads (48 warps) and 24 blocks per SM, 1024 threads per block (driver).
- **Occupancy** is how many blocks fit on an SM at once. It is the smallest of four limits: registers, shared memory,
  threads, blocks. `occupancy()` in `agentic/gb10/model.py` computes it that way.

The register file sets how big a tile can be, because `mma.sync` keeps its sums in registers. A 128 x 128 tile of fp32
sums is 16,384 values: 64 registers per thread over 256 threads. TileLang's warp-specialized kernels (one group of
warps only copies data, the others only compute; Section 3.3) launch T + 128 threads with one block per SM, so ptxas
gives each thread 65536 / 384, rounded down to 168 registers, at T = 256. A 256 x 256 tile at 512 threads needs 128
sum registers per thread and spills them to memory. Tiles of 256 x 256 ran 5x slower in the 1024^3 sweep and 27x
slower at 4096^3 (`tilelang_findings.md`, `gb10_model.md`).

### 2.4 Tensor cores

On this chip the tensor cores are reached through one instruction family, `mma.sync`. One warp hands the tensor pipe
of its sub-partition two small input tiles held in registers. The pipe adds their product into sum registers
("accumulators"). The basic fp16 shape is m16n8k16: a 16 x 16 tile of A times a 16 x 8 tile of B, 4096 FLOP. In SASS
it is HMMA (fp16, bf16, tf32), IMMA (int8) or QMMA (fp8).

Every type and shape holds the pipe for the same 16 cycles. `mma_types.cu` ran 8 warps x 8 accumulators on all 48
SMs and timed the whole kernel with CUDA events, which are GPU timestamps. The 8-bit types get twice the FLOP by doing
twice the K per instruction.

| Inputs -> sum | Shape | Cycles per mma per sub-partition | FLOP/clk per SM | TFLOP/s per SM, measured |
|---|---|---:|---:|---:|
| fp16 -> fp32 | m16n8k16 | 16.10 | 1017 | 2.584 |
| fp16 -> fp16 | m16n8k16 | 16.06 | 1020 | 2.591 |
| bf16 -> fp32 | m16n8k16 | 16.09 | 1018 | 2.586 |
| tf32 -> fp32 | m16n8k8 | 16.07 | 510 | 1.295 |
| e4m3 (fp8) -> fp32 | m16n8k32 | 16.05 | 2041 | 5.185 |
| e5m2 (fp8) -> fp32 | m16n8k32 | 16.07 | 2039 | 5.178 |
| e2m1 (fp4) through `kind::f8f6f4` | m16n8k32 | 16.08 | 2038 | 5.176 |
| s8 -> s32 | m16n8k32 | 16.08 | 2038 | 5.176 |

(The probe converted to FLOP per clock at 2.54 GHz.) So fp4 gains nothing over fp8 through the `f8f6f4` form. The
block-scaled fp4 form (m16n8k64 `mxf4nvf4`) exists for this target (TileLang emits it,
`runs/decide/tilelang_controls_survey.md`) but its rate was not measured.

The fp16 peak follows: 4096 FLOP per 16 cycles per sub-partition, times four, is 1024 FLOP per clock per SM. At
2.52 GHz on 48 SMs that is 123.9 TFLOP/s. A 48-SM run of `mma_rate.cu` measured 123.5-123.9 (`core.txt`).

**How many warps it takes.** A dependent mma takes 29 cycles to give its result, but holds the pipe only 16
(`mma_rate.cu`). So one chain of dependent mma per sub-partition keeps the pipe busy 16 / 29 = 55 % of the time.
Two independent chains fix most of that; two warps per sub-partition fix all of it (`core.txt`, 48 blocks):

| Warps per SM (per sub-partition) | Independent accumulators per warp | TFLOP/s per SM | Share of peak |
|---|---|---:|---:|
| 4 (1) | 1 | 1.42 | 55 % |
| 4 (1) | 2 or more | 2.37-2.41 | 92-93 % |
| 8 (2) | any | 2.58 | 100 % |

**In a real GEMM loop** the same warps also run `ldmatrix` to bring operands from shared memory into registers. The
ablation of 22 TileLang cp.async kernels at 1024^3 (the variants with no loads) puts the loop at 83 % of peak with one
warp per sub-partition (9 kernels, range 76-88 %) and 90 % with two (13 kernels, 81-96 %) (`gb10.json`,
`tensor_eff_gemm_loop`; four warps per sub-partition was not ablated and is taken equal to two).

**What other warps cost it.** `interfere.cu` runs mma chains in four warps and a memory stream in four more on the
same sub-partitions. Each memory instruction from the other warp costs the mma stream: cp.async 2.3 cycles, a 16-byte
global load 1.9, `ldmatrix` 0.4, an integer add about 0.

### 2.5 FP32, integer and half-precision pipes

Each sub-partition has two math units besides the tensor pipe: an FMA unit and an ALU unit. Rates are warp
instructions per clock per SM with 32 warps, after counting the instructions in the SASS (`gb10.json` from `alu.cu`
and `alumix.cu`; the raw `alu.txt` counts source operations, not instructions):

| Instruction | Warp instructions per clock per SM | Unit, cycles per warp instruction |
|---|---:|---|
| FFMA (fp32 multiply-add) | 3.86 | FMA, 1 |
| FADD | 3.94 | FMA, 1 |
| IMAD (integer multiply-add) | 2.0 | FMA, 2 |
| HFMA2 (two fp16 multiply-adds) | 2.0 | FMA, 2 |
| LOP3 (logic), SHF (shift), PRMT (byte permute) | 2.0 | ALU, 2 |
| HADD2.F32 and F2FP (half to float, float to half) | 2.0 for both together | whichever unit is free |
| F2I (float to integer) | 0.5 | the SFU's rate |
| I2F.F16 (integer to fp16) | 0.5 (`cvt.cu`) | SFU, 8; shares it with MUFU |
| I2FP.F32.S32 (integer to fp32) | 2.0 (`cvt.cu`) | ALU, 2 |
| F2FP (fp32 to fp16) | 2.0 (`cvt.cu`) | ALU, 2 |

Pairs show which kinds share a unit (`alumix.cu`):

- PRMT + IMAD reach 3.92 in total: two units.
- PRMT + LOP3 reach 2.0, and HFMA2 + IMAD reach 2.0: one unit each.
- IMAD + FFMA reach 1.32 + 1.32. Both are on the FMA unit: one FFMA (1 cycle) plus one IMAD (2 cycles) is 3 cycles.
- LOP3 + FFMA reach 1.53 + 1.02, about 23 % below what two separate units allow. Not explained.

Integer add and compare are placed on the ALU unit (`gb10_model.md`). Their rate could not be read: the compiler
folded the IADD3 loop in `alu.cu` (it reads 374 per clock, above the issue limit of 4).

So address arithmetic is not free. Integer work runs at half the FP32 rate, and IMAD competes with FP32 math.

### 2.6 The special function unit (SFU)

The SFU computes approximate exp2, reciprocal square root, log2, tanh, sine (SASS: MUFU). `sfu.cu`:

- Latency 18 cycles (sine 22).
- 16 lane results per clock per SM, which is one warp instruction per 8 cycles per sub-partition.
- One warp alone already gets 4 lane results per clock, the full rate of its own sub-partition.
- FFMA in the same probe: about 4.7 cycles latency, 124 lane results per clock per SM.

Example: in attention with head size 128, each score costs 512 FLOP on the tensor pipe (two products of length 128)
and one exp2 on the SFU. That is 0.5 clock of tensor time and 1/16 clock of SFU time per score per SM: the exp needs
about one eighth of the tensor time (`gb10_model.md`).

### 2.7 Shared memory

**Size.** 102,400 bytes per SM. A block gets 48 KB by default and up to 101,376 bytes (99 KB) if it asks
(driver, `preflight.md`). The driver keeps 1 KB per block for itself.

**Rate.** The array has 32 banks of 4 bytes, so it moves 128 bytes per clock (design). `smem_bw.cu` confirms it.
`ldmatrix.x4` (a warp loads four 8 x 8 fp16 matrices into registers in the layout mma wants), its transposed form,
`LDS.128`, `STS.128` and `stmatrix.x4` (the reverse of ldmatrix) each move 512 bytes in 4.0 cycles per SM with 16 or
more warps. cp.async and TMA writes go into the same array (Section 3).

**One warp alone** issues an `ldmatrix.x4` every 11.7 cycles (`instlat.cu`) and a `stmatrix.x4` every 10.8
(`tmastore.cu`). It takes 4 to 8 warps to reach 4 cycles. `ldmatrix` latency is 34 cycles (38.4 per dependent
step in `instlat.cu`, which includes an add).

**Bank conflicts.** A bank conflict is two addresses of one instruction landing in the same bank, so the array serves
them one after the other. `ldmatrix` reads eight rows of 16 bytes per matrix. If the rows sit a power of two apart in
memory, they pile onto the same banks. `smem_bw.cu`, 32 warps, cycles per `ldmatrix.x4` per SM (plain and transposed
are the same):

| Row stride (bytes) | 16 | 32 | 48 | 64 | 80 | 128 | 144 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cycles | 4 | 8 | 4 | 16 | 4 | 32 | 4 | 32 | 32 |

The rule: cost = 4 cycles x the conflict ways, and ways = 8 / (how many distinct 16-byte bank groups the eight rows
touch). Example: an fp16 A tile stored row by row with block_K = 32 has rows 64 bytes apart, so each `ldmatrix.x4`
takes 16 cycles instead of 4. A swizzled layout (rows shifted so they hit different banks) avoids this. In five pairs
of real TileLang kernels, the plain row-by-row layout ran 1.38-2.23x slower than the swizzled one (`gb10_model.md`).

### 2.8 L1

- Hit latency: 17 ns (`chase.cu`, 16 KB ring, `.ca` loads), about 45 cycles (`ldg_atomic.cu`: 50.8 per dependent
  step including the loop's own instructions).
- Rate: `LDG.128` hits take 7.4-8.5 cycles per 512 bytes per SM, that is 2 cycles per 128-byte line, about 64 B/clk
  (`ldg_atomic.cu`).
- Size: a pointer chase hits in 17 ns at 16 and 64 KB and takes 110 ns at 128 KB (`core.txt`, `chase.cu`). The older
  suite read the L1 size from counters: 64 KB when a block asks 16 KB or less of shared memory, 32 KB at 48 KB, 16 KB
  at 64 KB or more (B14). The older suite also found that cp.async and TMA do not fill the L1 (B14).
- Local memory (spilled registers) lives in L1. A spill slot used by a whole warp costs 4 cycles per line
  (`ldg_atomic.cu`: 11.9 cycles for two loads and a store).

### 2.9 Latencies in one place

| Event | Cycles | Time | Source |
|---|---:|---:|---|
| FFMA, dependent | 4.7 (older suite: 4.2) | | `sfu.cu`; B27 |
| SFU (ex2, rsqrt, lg2, tanh) | 18 (sine 22) | | `sfu.cu` |
| mma.sync m16n8k16, dependent | 29 | | `mma_rate.cu` |
| ldmatrix.x4, dependent | 34 | | `instlat.cu` |
| bar.sync | 13 + 2 per warp | | `instlat.cu` |
| L1 hit | 45 | 17 ns | `ldg_atomic.cu`, `chase.cu` |
| mbarrier arrive to waiting warp (Section 4) | 134 | 53 ns | `mbar.cu` |
| L2 hit, plain load | 342 | 136 ns | `ldg_atomic.cu`, `chase.cu` |
| cp.async 16 B from L2, round trip | 350 | 140 ns | `instlat.cu` (355.7) |
| TMA box from L2 | 440 + transfer | 224 ns for 4 KB | `dramlat.cu` |
| TMA box from DRAM, idle | 1170 + transfer | 512 ns for 4 KB, 896 ns for 32 KB | `dramlat.cu` |
| DRAM pointer chase, idle | about 1060 | 419-421 ns | `dram.cu`, `dramlat.cu` |
| DRAM from Little's law (Section 5.4), light load | about 1110 | 436-456 ns | `mlp_stream.cu` |
| DRAM under full load | | 1.9-6.1 µs | `dramlat.cu`, `dram.cu` (Section 5.5) |

## 3. Moving data

### 3.1 Three ways to load, one port

An SM can pull data from the L2 three ways:

- **Plain loads** (`ld.global`, SASS LDG) into registers.
- **cp.async**: each thread copies 16 bytes from global memory straight into shared memory, without passing through
  registers. A warp instruction moves up to 512 bytes. `commit_group` closes a group of copies; `wait_group N` waits
  until at most N groups are still in flight.
- **TMA** (tensor memory accelerator): one thread gives the SM's copy engine a descriptor of a 2-D array (a "tensor
  map") and the corner of a box. The engine moves the whole box into shared memory and reports to an mbarrier
  (Section 4) as the bytes land.

All three hit the same ceiling on one SM: about 32 bytes per clock.

| Path, one SM alone, data in L2 | Rate | Source |
|---|---:|---|
| Plain 16-byte loads, many in flight | 80.7 GB/s = 32.0 B/clk | `mlp_stream.cu` |
| `ld.global.cg` (skips L1) | 15.7 cycles per 512 B = 32.6 B/clk | `ldg_atomic.cu` |
| cp.async in TileLang's loop shape | 78 GB/s = 30.7 B/clk | `loopsteps.cu` |
| TMA, a TileLang kernel's loads alone | 81 GB/s | ablation (`gb10.json`, `tma_r_GBs_alone`) |

The port belongs to the SM, not the pair. Two blocks on SMs 0 and 1 (one pair) stream 161 GB/s, twice one block
(`core.txt`; with 8 blocks or fewer, block i runs on SM i, Section 6.2).

### 3.2 cp.async

**Issue.** One warp can issue one 512-byte cp.async every 6 cycles. Four warps issue one every 4 cycles per SM, the
shared-memory array's 128 B/clk (`cpq.cu`).

**The address stage.** The load/store unit looks up one 128-byte line per cycle per SM (`gb10.json`: a design
assumption, checked with `cpq.cu`, where 512 contiguous bytes = 4 lines take 4 cycles per copy with 4 warps). A copy
whose 32 lanes read 64-byte rows touches 8 lines, so it takes 8 cycles. A copy frees its address registers only when
this stage has handled it. The next instruction that rewrites those registers must wait. ncu on a whole TileLang
kernel against its mma-only version, one block: the long-scoreboard stall (warps waiting on a memory result) reads
0.80 against 0.00, the tensor pipe is active 68 % against 84 %, and there are no bank conflicts. The rule predicts a
loss of 320 / 160 / 96 cycles per k-tile for three kernels; measured 390 / 174 / 79 (`gb10_model.md`).

**The loop around it costs nothing.** `loopsteps.cu` builds TileLang's loader loop one step at a time (copies;
commit; wait; one barrier; two barriers), one block alone. Commit, wait and the barriers add nothing measurable: the
full loop takes 513-533 cycles per 16 KB k-tile, 78 GB/s.

**All SMs at once.** The L2 then sets the limit:

| Blocks loading, one per SM | 1 | 8 | 16 | 24 | 32 | 40 | 48 | 96 (2 per SM) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Chip, GB/s (`loopsteps.cu`) | 74 | 602 | 1215 | 1814 | 2072 | 2085 | 2077 | 2097 |
| Per SM, GB/s | 74 | 75 | 76 | 76 | 65 | 52 | 43 | 44 |

The rate rises in a straight line to about 24 SMs, then stops at 2.07-2.10 TB/s. With all 48 SMs loading, each gets
about 43 GB/s, half its port. More blocks per SM do not raise it: 2.08-2.14 TB/s with 1, 2, 3 or 4 blocks per SM
(`l2share.cu`). The share follows blocks in flight, not SMs. `tileload.cu` reaches 2.25 TB/s at 96-192 blocks with
16 KB tiles of 128-byte or longer row pieces. Row pieces of 64 bytes give 2.08 TB/s; pieces of 32 bytes give only
1.11-1.13 TB/s.

Two things that might have mattered do not. Neighbouring blocks reading the same A rows and B columns at the same
moment are no slower than blocks reading them at different times (`shareload.cu`). Splitting a warp's loads over 1, 2
or 4 rows 4-8 KB apart leaves the rate unchanged (`rowsplit.cu`).

**Overlap with math in the same warps.** In a cp.async GEMM the same warps issue the copies and the mma. They share
issue slots and barriers, so the two halves overlap only in part. Across 21 ablated kernels at 1024^3, a block's time
fits (compute^p + loads^p)^(1/p) with p = 1.97 at the median (18 of 21 within 1.5-2.4) (`gb10.json`,
`cpasync_overlap_p`). Example: compute and loads of 10 µs each give about 14 µs, not 10. When loads are 2-4x the
compute, 75-86 % of the compute hides. With two or more blocks per SM, one block's math covers another's copies.

### 3.3 TMA loads

**Rate.** One block alone gets 81 GB/s, the same port as the other paths (ablation). The whole chip gets
1.81-1.91 TB/s: the loads-only version of the 1024^3 TileLang kernel at 96 and 128 blocks (`gb10.json`,
`tma_l2_aggregate_GBs`). That is a little below cp.async's 2.07-2.25.

**Latency, idle.** `dramlat.cu` times one box from issue to mbarrier complete, median of 200:

| Box | 4 KB | 8 KB | 16 KB | 32 KB |
|---|---:|---:|---:|---:|
| From DRAM | 512 ns | 608 ns | 704 ns | 896 ns |

The slope is about 14 ns per KB, close to the port's 81 GB/s. The intercept is about 460 ns (1170 cycles), the DRAM
round trip. From the L2 a 4 KB box takes 224 ns, about 440 cycles plus the transfer.

**Latency under load.** A box waits behind everyone's queued bytes (Section 5.5). A 4 KB box takes 2.4 µs with the
rest of the chip pulling 208 GB/s, and 3.6 µs when DRAM is saturated.

**Overlap.** With TMA, separate "producer" warps issue the copies while "consumer" warps do the math. This is warp
specialization: different warps of one block get different jobs. Then copies and math overlap fully. Per k-tile,
time = max(compute, loads) + about 0.09 µs (about 230 cycles) for the hand-offs (ablation of seven TileLang kernels,
`gb10.json`, `bubble_us_per_ktile_tma_ws`).

**Older suite.** A 2-D tensor map costs the same as a 1-D copy of the same bytes (B36). Bytes of a TMA box can show up
in shared memory before the mbarrier says the box is complete, about 30 bytes per 16 KB on average (B26.4). So always
wait on the barrier, never poll the data.

### 3.4 Stores

**The pair shares one store path.** `storepat.cu` stores a 64 KB tile per block three ways. Every pattern takes
twice as long per block when the second SM of the pair is busy, then stays flat up to 24 blocks (12 pairs):

| Store pattern (64 KB per block) | 1 block | 2 blocks, one pair | 48 blocks | 96 blocks | Chip limit |
|---|---:|---:|---:|---:|---:|
| TileLang's: 4 B per lane, 8 rows x 16 B per instruction | 3.01 µs | 5.97 µs | 10.08 µs | 16.59 µs | 0.31-0.38 TB/s |
| 16 B per lane, 4 rows x 128 B per instruction | 0.70 µs | 1.33 µs | 2.29 µs | 4.19 µs | 1.37-1.50 TB/s |
| 4 B per lane, one 128-byte row per instruction | 0.80 µs | 1.47 µs | | | |

A **sector** is 32 bytes; a cache line is 128 bytes, four sectors. TileLang's store writes each 32-byte sector in two
halves from two different instructions. The pair's path then takes 15 cycles per instruction (8.7 B/clk). A coalesced
16-byte store (the warp writes whole 128-byte lines) takes 14 cycles per 512-byte instruction (37 B/clk). So the
half-sector store is about 4x slower per byte, and the chip as a whole also takes 4x less of it.

**What it costs a kernel.** At 1024^3 the 2 MB of C at about 0.35 TB/s is about 6 µs of a 35-40 µs kernel
(`gb10_model.md`). The store alone in one TileLang kernel takes 2.37 µs with one block and 4.64 µs with 2 to 24
blocks (`step_store.txt`).

**Staged store: stmatrix plus a TMA store.** The warps write C into shared memory with `stmatrix` (4 cycles per x4 per
SM with 4+ warps), then one thread issues a TMA store. `tmastore.cu`, 64 KB per block:

| Blocks | 1 | 2 (one pair) | 8 | 24 | 48 | 96 |
|---|---:|---:|---:|---:|---:|---:|
| Time per block | 0.96 µs | 1.74 µs | 1.74 µs | 1.74 µs | 2.75 µs | 2.73 µs |

That fits 457 cycles of fixed delay plus 33.1 B/clk through the pair's store path (84 GB/s). The chip measured
1.14-1.15 TB/s; the slope alone gives 482 B/clk (1.21 TB/s). Waiting for the TMA to finish reading shared memory
(`wait_group.read`, TileLang's default) costs the same as waiting for the write. The engine reads shared memory only
as fast as the store path drains it. So the block holds its shared memory until the last byte has left. In five
pairs of TileLang kernels at 1024^3 (direct store against staged store), the staged store saved 2.7-4.4 µs
(`gb10_model.md`).

**The store queue.** With 4 warps, about 16 store or atomic warp instructions per SM go out at once. After that,
issue follows the drain: 16 cycles per instruction per SM for `STG.128` and for accumulator-layout `RED.F32x2`
(`stq.cu`).

### 3.5 Atomics and reductions

A **reduction** (SASS RED) is an atomic add whose old value nobody reads; `atomicAdd` with an unused result compiles
to it. It is done at the L2. `ldg_atomic.cu`:

- `RED.F32`: one warp instruction (128 bytes) per 5 cycles per SM. `RED.F32x2`: one (256 bytes) per 10 cycles. Both
  are 25.6 B/clk per SM.
- Chip, each warp adding into contiguous bytes: 323-340 B/clk (0.81-0.86 TB/s).
- Chip, straight from mma accumulators (8 rows x 32 bytes per instruction): 225 B/clk (0.57 TB/s). One block takes 16
  cycles per instruction; two blocks on a pair take 32 each, 2 cycles per line on the pair's store path.
- Every lane of every SM adding to one address: about 3.3 cycles per add for the whole chip (older suite: one every
  1.35 ns, the same).
- Blocks adding into the same tile at the same time are no slower (`redtile`).
- An atomic on a line that is not in the L2 first reads it from DRAM. So a split-K GEMM (one that splits the K loop
  over several blocks and adds their partial sums into C with atomics) timed cold also pays a read of its fp32 C
  (`gb10_model.md`).

## 4. Synchronization

**Block barrier** (`bar.sync`, `__syncthreads`): 13 cycles plus 2 per warp (`instlat.cu`: 14.6 at 1 warp, 20.7 at 4,
28.7 at 8, 44.7 at 16, 76.7 at 32).

**cp.async groups.** `commit_group` and `wait_group` cost nothing measurable inside TileLang's loop (`loopsteps.cu`).

**mbarriers.** An mbarrier is a 64-bit counter in shared memory. Threads arrive on it, a TMA copy adds the bytes it
delivered, and waiting threads spin until its phase flips. `mbar.cu`, one warp arrives while another waits on the
same SM:

- The waiter gets through 134 cycles after the arrive with `try_wait` (TileLang's form), 157 with a `test_wait` spin.
- If the arriving warp did a global store just before, the arrive waits for that store: 217 cycles.
- One lane arriving many times in a row costs 6.6 cycles per arrive.

A TMA pipeline has two hand-offs per k-tile: producer to consumers when the bytes land, consumers to producer when the
stage is free. At 134 cycles each, a 2-stage pipeline fed from DRAM leaves the tensor pipe waiting. Example: a
TileLang GEMM at 128 x 4096 x 4096 with 128 x 128 x 32 tiles took 164, 152 and 148.5 µs with 2, 3 and 4 stages
(`gb10_model.md`). Use 3-4 stages when the data comes from DRAM.

**Clusters and distributed shared memory.** A cluster is a group of blocks launched together on nearby SMs that can
read each other's shared memory ("distributed shared memory", DSMEM). `tier4_query.cu`: cluster launch works, the
largest cluster is 8 blocks, and a real 2-block cluster read its partner's shared memory correctly (101 and 100, as
expected). The older suite timed DSMEM at 3,648 cycles latency and 572 GB/s, but those two readings do not fit one
mechanism and are not used (B37). It also found that TMA multicast (one load delivered to several blocks of a
cluster) runs but takes 500x longer than two plain loads and reads the L2 twice (B37). So clusters work, and nothing
measured here shows they pay.

**Other launch features** (`tier4_query.cu`): cooperative launch (grid-wide barrier) is supported, concurrent kernels
are supported, and streams have six priority levels (0 to -5). None was timed.

## 5. Memory hierarchy

### 5.1 L2 bandwidth to the SMs

The L2 feeds the SMs at different rates depending on the path. Plain loads with many in flight get the most
(`mlp_stream.cu`: 8 MB read 20 times in one launch, 512 threads per block, 4 loads in flight per thread):

| Blocks, one per SM | 1 | 2 | 4 | 8 | 16 | 24 | 32 | 48 | 96 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Chip, GB/s | 81 | 161 | 323 | 650 | 1338 | 1987 | 2625 | 3465 | 3523 |

Summary by path, all SMs busy:

| Path | Chip rate | Per SM at 48 SMs | Source |
|---|---:|---:|---|
| Plain loads (`ld.global.cs`) | 3.47 TB/s | 72 GB/s | `mlp_stream.cu` |
| `ld.global.cg` 16 B loads | 2.2 TB/s | 45 GB/s | `ldg_atomic.cu` |
| cp.async | 2.07-2.25 TB/s | 43-47 GB/s | `loopsteps.cu`, `l2share.cu`, `tileload.cu` |
| TMA | 1.81-1.91 TB/s | 38-40 GB/s | ablation |

The copy paths are what GEMM kernels use, so this note takes about 2.1 TB/s as "the L2 rate" for balance
(Section 7). Why plain loads get about 60 % more is open (Section 10).

### 5.2 L2 capacity and what it keeps

The driver says 24 MiB. Less than that holds data a kernel reads again.

**Reuse distance** is how many bytes of other data are touched between two reads of the same byte. `l2policy.cu`
reads W MB four times in a row and counts L2 hits with ncu. Each L2 fill brings 64 bytes (two 32-byte sectors), so a
stream that is never reused still shows 50 % hits in ncu. Above that floor, the chance that a re-read hits:

| Reuse distance (MB) | 14 | 16 | 18 | 20 | 24 | 28 | 32 | 40 | 48 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Hit probability | 1.00 | 0.96 | 0.82 | 0.75 | 0.63 | 0.62 | 0.32 | 0.01 | 0.00 |

Three things stand out:

- **Re-reads start to miss at 16 MB**, two thirds of the 24 MiB. `gb10.json` takes 16.78 x 10^6 bytes (16 MiB) as the
  effective size. Its guess at the cause, set structure and the CPU's share, is not tested.
- **The loss is gradual**, from 16 to 40 MB. Plain LRU (evict the least recently used line) would lose everything
  just past the size.
- **Re-read data survives a stream.** A 4 or 8 MB set read, then a 4-64 MB stream of new data, then the set again:
  the set still hits almost entirely (`l2policy.cu`, hot-stream rows: 16-491 sector misses against 131,365 or 262,437
  hits). The L2 protects data that was hit from data seen once ("scan resistance").

No textbook policy (LRU, random, segmented LRU, insertion at the bottom, RRIP, re-reference interval prediction) at 24
MiB reproduces all three (`agentic/gb10/l2policy_fit.py`). For GEMM traffic, a k-slice-level LRU of 16 MiB matches
ncu's L2 misses best: median 6 % over 18 kernels (`dram_waves_kslice` in `agentic/gb10/model.py`).

**Between kernels.** L2 contents survive a kernel boundary. `stream_b2b.cu` reads S MiB with all SMs, back to back,
no flush:

| S (MiB) | 16 | 20 | 24 | 28 | 32 | 36 and up |
|---|---:|---:|---:|---:|---:|---:|
| GB/s | 1639 | 782 | 487 | 303 | 265 | 232-237 |
| Share found in L2 from the last run | 1.0 | 0.82 | 0.60 | 0.26 | 0.13 | 0 |

**Structure, older suite.** No set conflicts appear for up to 64 lines at any power-of-two stride, so the sets are
either very wide or picked by a hash of many address bits. No address bit selects a slower part of the L2: latency
varies by 6 ns around 136 ns (B13, B5).

### 5.3 The persisting L2 set-aside

Software can set aside part of the L2 for "persisting" data and mark an address range ("access-policy window") as
persisting. `tier4_query.cu`: the set-aside can be up to 18 MiB (18,874,368 bytes); asking for 8 MiB gave 9 MiB; a
window can span up to 128 MiB.

`l2persist.cu` plays a decode step: read W MB of weights after a 64 MB stream of other traffic, with and without a
window over the weights (18 MiB set aside):

| Weights | 4 MB | 8 MB | 12 MB | 16 MB |
|---|---:|---:|---:|---:|
| No window | 20.3 µs | 37.7 µs | 51.0 µs | 63.4 µs |
| Window | 6.0 µs | 8.0 µs | 10.1 µs | 14.2 µs |
| Speed-up | 3.4x | 4.7x | 5.0x | 4.5x |

So weights up to about 16 MB that are read every step can stay in the L2 across other work.

### 5.4 DRAM rates

**Read, write, mixed** (`dram.cu`, 48 x 1024 threads, 16-byte `.cg` accesses):

| Read : write | 1 : 0 | 0 : 1 | 1 : 1 | 3 : 1 | 1 : 3 |
|---|---:|---:|---:|---:|---:|
| GB/s | 241.9 | 197.2 | 219.6 | 222.4 | 216.8 |

A mix behaves as if each part ran at its own rate: 1 / (read share / 242 + write share / 197). That formula gives
217, 229 and 207 GB/s for the three mixes, within 5 % of each.

A plain 64 MB read on a clean L2 gets 237 GB/s (`mlp_stream.cu`), 87 % of the 273 GB/s paper rate. Held for 20 s it
stays at 235.8-237.1 GB/s (`sustain.cu`).

**Short pieces cost.** Reading tiles whose rows are 64 bytes long gets 201-204 GB/s; 256-byte rows get 230-234 GB/s;
one long run gets 237 (`tileload.cu`, `mlp_stream.cu`). In a GEMM the A tile comes as short pieces of rows far apart,
while neighbouring B tiles read neighbouring pieces of the same rows at the same time, so DRAM sees one long run
(`gb10_model.md`). The older suite found random 128-byte reads at 186 GB/s, random 4 KB at 229, 16 KB at 235, and saw
no difference between random and sequential 128-byte reads (B18).

**Bytes in flight.** Little's law: rate = bytes in flight / latency. To stream DRAM at full rate, the chip must keep
enough loads outstanding. `mlp_stream.cu`, 256 MB read after a write flush (so a few percent below the clean 237):

| Blocks x threads x loads in flight per thread | Bytes in flight, chip | GB/s |
|---|---:|---:|
| 1 x 128 x 1 | 2 KB | 4.7 |
| 1 x 512 x 8 | 64 KB | 80.6 (one SM's port) |
| 8 x 128 x 4 | 64 KB | 129.6 |
| 8 x 512 x 2 | 128 KB | 204.1 |
| 8 x 512 x 4 | 256 KB | 229.1 |
| 48 x 128 x 2 | 192 KB | 223.7 |
| 48 x 128 x 4 | 384 KB | 230.2 |

The first row gives the idle latency: 2 KB / 4.7 GB/s = 436 ns. At 237 GB/s and 440 ns, at least 104 KB must be in
flight; in practice it takes about 256 KB. One SM cannot do it alone: its port stops at 80 GB/s. Eight SMs with 32 KB
each can.

**Channels and counters, older suite.** Forcing any single address bit constant keeps at least 97 % of the rate, so
DRAM channels are picked by a hash and their count cannot be read from software (B18). The GB10 has no `dram__*`
counters in ncu. DRAM traffic is read as L2 misses: `lts__t_sectors_op_read_lookup_miss.sum` x 32 bytes
(`preflight.md`).

### 5.5 DRAM latency under load

Latency rises with the bytes everyone keeps in flight. Two probes, each with one block chasing pointers inside 2 MB
pages (so the TLB, the cache of address translations, always hits and only DRAM is timed) while other SMs stream:

| `dram.cu`: SMs streaming (1024 threads each) | 0 | 4 | 8 | 16 | 24 | 47 |
|---|---:|---:|---:|---:|---:|---:|
| Latency per load | 419 ns | 1.35 µs | 2.68 µs | 4.81 µs | 5.86 µs | 6.13 µs |

| `dramlat.cu`: background DRAM rate (47 blocks of 256 threads) | 0 | 33 GB/s | 121 GB/s | 208 GB/s | 256 GB/s (no pause) |
|---|---:|---:|---:|---:|---:|
| Latency per load | 421 ns | 403 ns | 554 ns | 1159 ns | 1929 ns |
| 4 KB TMA box, median | 512 ns | 512 ns | 512 ns | 2368 ns | 3648 ns |

Both saturate DRAM, yet one sees 6.1 µs and the other 1.9 µs. Little's law explains it. 6.1 µs at about 250 GB/s
means about 1.5 MB waiting in the queues. The `dramlat.cu` streamers keep at most 752 KB in flight (47 x 256 threads
x 4 x 16 bytes), and see 1.9 µs. So loaded latency is the idle latency plus the chip's queued bytes divided by the
DRAM rate. At 121 GB/s the box median stays at 512 ns but the slowest tenth already take 1.8 µs or more.

### 5.6 Dirty lines and write-back

The L2 is write-back: written lines stay in the L2 ("dirty") and go to DRAM only when evicted. If the L2 is full of
dirty lines when a kernel starts, its reads first push them out. `mlp_stream.cu` reads S MB after a 256 MB write
(dirty L2) and after a 256 MB read (clean L2):

| S (MB) | 2 | 4 | 8 | 12 | 16 | 24 | 32 | 64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Extra time, dirty minus clean | 11.9 µs | 18.7 µs | 25.6 µs | 27.8 µs | 29.5 µs | 31.6 µs | 35.0 µs | 37.9 µs |

TileLang's timing helper leaves the L2 exactly this dirty (Section 9.2).

### 5.7 Sharing DRAM with the CPU, and unified memory

The CPU and GPU share the memory. While CPU threads copy 256 MB buffers (`cpuhog.c`), the GPU's read stream drops
(`gb10.json`, `cpu_dram_contention`):

| CPU threads copying | 0 | 2 | 8 | 20 |
|---|---:|---:|---:|---:|
| GPU read, GB/s | 237 | 213.5 | 170.6 | 137.9 |
| CPU alone, GB/s (read + write counted) | | 95.3 | 138.7 | 151.9 |

How the GPU allocation is made also matters (older suite, B12 and B23, not measured again):

| Allocation | GPU read rate |
|---|---:|
| `cudaMalloc` | 238 GB/s |
| `cudaMallocHost` (pinned host memory) | 200 GB/s |
| `cudaMallocManaged`, any first toucher, prefetched or not | 163 GB/s |
| Host pages read coherently | 163 GB/s |

Managed memory cost a constant 1.45x even with the CPU idle and the data prefetched. The older suite reads this as
address translation, not data movement. A managed allocation could be 116 GB. The GPU can read pageable host memory
through the host's page tables (`tier4_query.cu`: `pageableUsesHostPageTables` = 1).

## 6. Launch, instruction fetch and clock

### 6.1 Launch

`launch.cu` (a stream is a queue of GPU work; kernels in one stream run in order):

| Case | Time |
|---|---:|
| Empty kernels back to back from one stream, 1-1024 blocks | 2.04-2.08 µs each (the host's submission rate) |
| The same 100 kernels as one CUDA graph | 0.43 µs each; 0.87 µs with 1024 blocks |
| Gap between back-to-back kernels with work (48 blocks, about 21 µs each) | 3.1 µs |
| One launch plus synchronize, timed on the host | 4.9 µs |

A CUDA graph records a sequence of launches once and replays it, which skips most of the per-launch host work. Inside
a kernel, each extra block costs about 8 ns to start and an empty block lives about 0.1 µs (ablation at 8192^3;
`gb10.json`). TileLang's Python wrapper adds about 5 µs of host time per call, of which 2-4.5 µs shows up on an idle
GPU (`tilelang_findings.md`). For a 35 µs kernel these costs are 6-14 %.

### 6.2 Where blocks go

Older suite (B16): for grids of up to 8 blocks, block i runs on SM i. For larger grids the first wave goes to SM
pairs in turn over six groups of eight SMs: block 0 to SM 0, block 1 to SM 1, block 2 to SM 8, block 3 to SM 9, then
SMs 16, 17, 24, 25, 32, 33, 40, 41, then back to 2, 3. The six groups are probably the chip's GPCs (graphics
processing clusters); that is a guess, not published. `storepat.cu` agrees: its first four blocks landed on SMs 0, 1,
2, 3 with 8 blocks and on 0, 1, 8, 9 with 24 or more. After the first wave a block goes wherever one finished.

So a grid fills both SMs of a pair before it moves on, and those two blocks share one store path.

### 6.3 Instruction fetch

`icache.cu` runs loops of independent FFMAs from 4 KB to 256 KB of code with 8 warps:

| Loop code size | up to 32 KB | 64 KB | 128 KB | 256 KB |
|---|---:|---:|---:|---:|
| FFMA per clock per SM | 3.92-3.98 | 3.56 (-10 %) | 2.89 (-27 %) | 1.98 (-50 %) |

TileLang's GEMM kernels here are at most about 24 KB in all, with main loops of about 3 KB, so instruction fetch never
limits them.

### 6.4 Clock

`clock.cu` counts SM cycles with clock64 (the SM's cycle counter) and nanoseconds with %globaltimer over the same
span; the ratio is the clock.

| Condition | SM clock | Source |
|---|---:|---|
| Driver's clock attribute | 2.418 GHz | `preflight.md` |
| nvidia-smi maximum | 3.003 GHz | `preflight.md` |
| Idle | 208 MHz | `preflight.md`, `sustain.cu` |
| Short probe, 48 blocks of dependent FMAs | 2.52 GHz | `clock.cu` |
| Inside real TileLang kernels | 2.39-2.53 GHz (lower at full grid) | `clock_in_kernel.txt` |
| 20 s of mma on all SMs | 2.496 GHz, then 2.476 at 62 °C; 68 W; no throttle reason | `sustain.cu` |
| 20 s of DRAM streaming | 2.502 GHz; 23.5 W | `sustain.cu` |
| One timing session of TileLang kernels | 2.405 GHz at 41-42 °C | `tilelang_findings.md` |

The clock showed no power or heat limit in 20 s (`sustain.cu`), and needs no warm-up (older suite, B22). Long runs
are about 2 % slower than short probes. At first the clock could not be pinned (`preflight.md`). After the account got
passwordless sudo, `nvidia-smi -lgc 2418` worked, but pinning did not reduce run-to-run spread
(`tilelang_findings.md`).

## 7. Balance points

### 7.1 The rates side by side

Per SM, per clock, fp16 math:

| Unit | Per SM per clock | Whole chip |
|---|---:|---:|
| Tensor pipe | 1024 FLOP | 123.9 TFLOP/s |
| Shared memory | 128 B | 15.5 TB/s |
| Load port to L2, SM alone | 32 B | |
| L2 share with all 48 SMs copying | 17 B | 2.08 TB/s |
| DRAM share with all 48 SMs reading | 2.0 B | 237 GB/s |

**Arithmetic intensity** is FLOP done per byte moved. A unit's **ridge point** is the intensity at which math time
equals that unit's transfer time: peak FLOP/s divided by its bytes/s. Below the ridge the unit sets the time.

| Ridge | Arithmetic | FLOP per byte |
|---|---|---:|
| Shared memory | 1024 / 128 | 8 |
| One SM's L2 port, SM alone | 1024 / 32 | 32 |
| L2, all SMs, cp.async | 123.9 TFLOP/s / 2.08 TB/s | 60 |
| L2, all SMs, TMA | 123.9 / 1.86 | 67 |
| DRAM read | 123.9 TFLOP/s / 237 GB/s | 523 |
| DRAM read, fp8 math | 247.7 / 0.237 | 1045 |

### 7.2 What a GEMM tile needs

A block that computes a bm x bn tile of C loads, per k-tile of depth bk, (bm + bn) x bk fp16 values and does
2 x bm x bn x bk FLOP. Its intensity against the L2 is

    I = 2 bm bn bk / (2 (bm + bn) bk) = bm bn / (bm + bn)  FLOP per byte.

| Block tile | 64 x 64 | 64 x 128 | 128 x 128 | 128 x 256 | 256 x 256 |
|---|---:|---:|---:|---:|---:|
| FLOP per byte from L2 | 32 | 43 | 64 | 85 | 128 |

So with all SMs busy, a tile needs to be about 128 x 128 or larger to outrun the L2 (60 FLOP/B). A 64 x 64 tile is
fine only when few SMs load (32 FLOP/B, the port alone).

The same rule applies inside the SM. Each warp reads its share of A and B from shared memory with `ldmatrix`. A warp
tile of wm x wn reads at wm wn / (wm + wn) FLOP per byte, and shared memory needs 8. A 16 x 16 warp tile is exactly at
the limit; 32 x 32 (16 FLOP/B) leaves room, and the tile's own writes into shared memory use the same 128 B/clk.

### 7.3 Worked example: 1024 x 1024 x 1024, fp16 in, fp32 sum

| Step | Arithmetic | Result |
|---|---|---|
| Work | 2 x 1024^3 | 2.15 GFLOP |
| Tensor floor, whole chip | 2.15e9 / 123.9e12 | 17.3 µs |
| Bytes from DRAM if cold | A + B = 2 x 2 MiB = 4.19 MB at 237 GB/s | 17.7 µs |
| Intensity against DRAM | 2.15e9 / 4.19e6 | 512, against a ridge of 523 |
| Bytes from L2, 128 x 128 tiles | 64 tiles x (128 + 128) x 1024 x 2 B = 32 MiB at 2.08 TB/s | 16.1 µs |
| Tile intensity against L2 | 128 x 128 / 256 | 64, against a ridge of 60 |
| Waves | 64 tiles on 48 SMs | 2 waves; the second is a third full |
| One tile on one SM | 2 x 128 x 128 x 1024 / (2.58 TFLOP/s x 0.895) | 14.5 µs |
| Estimate | 2 x 14.5 µs + launch 2-3 µs | 31-32 µs |
| Measured, warm | cuBLASLt 34.2 µs; TileLang 35.1 µs (30.6 with the launch hidden) | 61-70 TFLOP/s |

Cold, this problem sits on the DRAM ridge. Warm (operands already in the L2, as when timed back to back), DRAM drops
out and every unit is close to balanced. What sets the time is the waves: 64 tiles on 48 SMs leave a second wave a
third full, and launch is 6-9 % on top. The chip runs at about half its peak.

One caution: if the same warps issue cp.async and mma (Section 3.2), the first wave's tiles each need 512 KiB at the
43 GB/s share, 12.1 µs, and the overlap law gives sqrt(14.5^2 + 12.1^2) = 18.9 µs, not 14.5. TMA with producer warps,
or two blocks per SM, avoids that.

### 7.4 Worked example: decode, 32 x 4096 x 4096 (and 128 x 4096 x 4096)

In a decode step, M is the number of tokens in flight and B holds the weights. The weights are read once.

| Step | M = 32 | M = 128 |
|---|---:|---:|
| Work, 2 M N K | 1.07 GFLOP | 4.29 GFLOP |
| Tensor floor | 8.7 µs | 34.7 µs |
| Bytes read: B = 4096 x 4096 x 2 B, plus A | 33.6 + 0.26 = 33.8 MB | 33.6 + 1.05 = 34.6 MB |
| C written | 0.26 MB | 1.05 MB |
| Intensity, FLOP per byte | 31.5 | 120 |
| DRAM floor, reads at 237 GB/s | 142.7 µs | 146.0 µs |
| Measured, one call after an L2 flush | cuBLASLt 143.2 µs, TileLang 142.3 µs | cuBLASLt 152.4 µs, TileLang 149.5 µs |
| Measured, back to back, each call reading its own copy of the weights (tier study run 3) | best kernel 137.1 µs (247 GB/s); cuBLASLt 149.3 µs | best kernel 152.4 µs (227 GB/s); cuBLASLt 169.8 µs |
| Tensor pipe busy | 6 % | 23 % |

(Measurements: `gb10_model.md`, decode section, and `runs/decide/study_v3/`. ncu counted 33.9 MB of DRAM traffic at
M = 32.) Both shapes run at the DRAM floor. Math hardly matters. The best M = 32 kernel even beats the floor computed
from the probes' 237 GB/s: it reads at 247 GB/s, so the probes do not reach DRAM's ceiling (Section 10). What matters:

- **Enough bytes in flight** (Section 5.4): about 256 KB across the chip. 48 SMs need about 5 KB each; one SM alone
  can never get past 80 GB/s.
- **Long runs**: 64-byte pieces cost 12-14 % against 256-byte pieces.
- **Enough pipeline stages**: each k-tile waits a DRAM round trip plus two mbarrier hand-offs (Section 4).
- **Stable timing**: 32 MiB of weights is just over the L2. Back-to-back runs keep 13 % of it (Section 5.2) and vary
  about 10 % between sessions. A flush between calls does not fix that (Section 9.2, trap 11). Time decode the way it
  runs: each call reads its own copy of the weights, several copies rotating, far more than the L2 holds.

When does a GEMM with N = K = 4096 stop being DRAM-bound? Its intensity is 2MNK / (2NK + 2MK + 2MN) =
M x 4096 / (4096 + 2M). That reaches 523 at M of about 700. If only the weights counted, it would be M = 523. With fp8
weights and fp8 math the rate doubles and the bytes halve, so the break-even M stays about the same.

### 7.5 Worked example: 4096^3, where L2 capacity and tile order matter

| Step | Arithmetic | Result |
|---|---|---|
| Work | 2 x 4096^3 | 137.4 GFLOP |
| Tensor floor | 137.4e9 / 123.9e12; at 0.895 loop efficiency | 1.11 ms; 1.24 ms |
| A + B | 2 x 32 MiB = 67 MB | 4x the effective L2 |
| DRAM, tiles in row order (no panels) | ncu: 770 MB at 237 GB/s | 3.25 ms |
| DRAM, TileLang with panels of 4 tile rows | ncu: 203 MB | 0.86 ms |
| DRAM, cuBLASLt's own order | ncu: 245 MB | 1.03 ms |
| Measured | TileLang best 1377 µs; cuBLASLt 1456 µs | 99.8 and 94.4 TFLOP/s (81 % and 76 % of peak) |

With A and B four times the L2, the order in which blocks run decides the DRAM traffic. In plain row order every row
of tiles reads all of B again. Walking the grid in panels of a few tile rows lets the panel share B while it is in the
L2. In the first TileLang sweep (timed with `do_bench`) the best kernel without panels took 6.1 ms and the best with
panels of 8 or 16 took 1.73 ms (`tilelang_findings.md`, round 1). With panels, DRAM time hides under math and the
tensor pipe sets the time.

### 7.6 What sets the time, by kind of kernel

- **Large square GEMMs** (2048^3 and up): the tensor pipe, once tiles are 128 x 128 or larger, the grid runs in
  panels, and the store is cheap. Best measured: 75-81 % of peak (cuBLASLt at 2048^3 to 8192^3, TileLang at 4096^3).
- **Mid-size GEMMs** (around 1024^3): waves and fixed costs (launch, pipeline fill, the C store). About 50-57 % of
  peak.
- **Decode** (M of 128 or less): DRAM. The tensor pipe is 6-23 % busy.
- **Split-K and other atomics**: 25.6 B/clk per SM for reductions, 225-340 B/clk for the chip, and the pair's store
  path.
- **Softmax next to the math**: the SFU needs about one eighth of the tensor time at head size 128.

## 8. What it lacks, what it keeps

| Feature | GB10 (`sm_121`, `sm_121a`) | Evidence |
|---|---|---|
| `mma.sync` (warp-level matrix math from registers) | Yes, all types | `mma_types.cu` |
| `wgmma` (Hopper: warp-group matrix math reading shared memory) | No: ptxas rejects `wgmma.fence` | `preflight.md` |
| `tcgen05` and tensor memory (TMEM) (datacenter Blackwell) | No: ptxas rejects `tcgen05.alloc` | `preflight.md` |
| `setmaxnreg` (move registers between warp groups) | Rejected for `sm_121`; `sm_121a` not tested | `preflight.md`, controls survey |
| 2-block MMA, TMA im2col, gather4 / scatter4 | Not available per TileLang's source; not tried on hardware | controls survey |
| TMA loads and stores, tensor maps | Yes | `dramlat.cu`, `tmastore.cu` |
| mbarriers with byte counts | Yes | `mbar.cu`, `dramlat.cu` |
| Warp specialization (producer and consumer warps) | Yes; TileLang emits it here | `tilelang_findings.md` |
| Clusters, distributed shared memory | Yes, up to 8 blocks | `tier4_query.cu` |
| TMA multicast within a cluster | Runs; older suite: 500x slower than unicast | B37 |
| Block-scaled fp4 mma (`mxf4nvf4` m16n8k64) | Emitted by TileLang; rate not measured | controls survey |
| Persisting L2 set-aside | Yes, up to 18 MiB | `tier4_query.cu`, `l2persist.cu` |
| Cooperative launch, stream priorities, concurrent kernels | Yes | `tier4_query.cu` |
| DRAM counters in ncu | None | `preflight.md` |

("Controls survey" is `runs/decide/tilelang_controls_survey.md`, a read of TileLang's source at commit fe6d319c.)

What this means for kernel design:

1. **All tensor operands pass through registers.** Every byte goes from shared memory into registers with `ldmatrix`,
   through the 128 B/clk array, before an mma can use it. Budget shared memory as carefully as the L2: warp tiles of
   32 x 32 or more, swizzled layouts, no bank conflicts. Real loops reach 83-90 % of peak, not 100.
2. **Sums live in registers too.** With no tensor memory, the C tile sits in the 64 K registers of the SM. A 128 x 128
   fp32 tile takes a quarter of them. Larger tiles spill and fall off a cliff (5-27x).
3. **Math and copies compete for issue slots.** There is no asynchronous MMA, so the warps that do the math are busy
   issuing it. If the same warps also issue copies, the two overlap only in part. Use TMA with producer warps: it
   overlaps fully, where cp.async in the same warps hides about half.
4. **Pipelines need depth.** Each TMA hand-off costs about 134 cycles. From DRAM, 2 stages leave the tensor pipe
   waiting; 3-4 stages help.
5. **Store wide.** Write C with 16-byte coalesced stores or the staged `stmatrix` + TMA store. The half-sector 4-byte
   pattern is 4x slower, and two SMs share each store path.
6. **Clusters exist but have not shown a gain.** Multicast saved nothing in the older suite.
7. **Memory is the binding limit for inference.** 237 GB/s, shared with the CPU. Keep hot weights in the persisting
   L2 when they fit in 18 MiB; cut bytes per weight with fp8 or fp4.

## 9. How it was measured

### 9.1 The probes

All in `agentic/tilelang/microbench/`, built for `sm_121a`, output in `calibration/spark-6951/<name>.txt` unless
noted.

| Probe | What it measures |
|---|---|
| `devquery.cu`, `tier4_query.cu` | Driver numbers; clusters, DSMEM, persisting-L2 limits |
| `clock.cu`, `sustain.cu` | SM clock (clock64 against the global timer); 20 s of tensor and DRAM load with nvidia-smi |
| `mma_rate.cu`, `mma_types.cu` | Tensor rate by warps and accumulators (`core.txt`); by data type |
| `alu.cu`, `alumix.cu`, `sfu.cu` | Math rates; which instructions share a unit; SFU latency and rate |
| `sched.cu`, `interfere.cu`, `icache.cu` | Scheduler policy; memory instructions against an mma stream; instruction fetch |
| `instlat.cu`, `smem_bw.cu` | ldmatrix, bar.sync, cp.async latency; shared-memory rate and bank conflicts |
| `cpq.cu`, `loopsteps.cu`, `tileload.cu` | cp.async issue; TileLang's loader loop step by step; the loader by tile shape, L2 and DRAM |
| `l2share.cu`, `shareload.cu`, `rowsplit.cu` | Chip L2 rate against blocks per SM; neighbours sharing lines; loads split over rows |
| `mlp_stream.cu`, `chase.cu` | Rate against bytes in flight, L2 and DRAM, dirty and clean (`core.txt`, `mlp.txt`); latency ladder |
| `ldg_atomic.cu`, `stq.cu` | Plain loads, reductions, spills (`ldg_atomic*.txt`); store queue depth |
| `storepat.cu`, `tmastore.cu` | Store patterns and the pair's store path; stmatrix and TMA store |
| `dram.cu`, `dramlat.cu`, `cpuhog.c` | DRAM latency under load, read and write mixes (`dram_detail.txt`); TMA box latency; CPU traffic |
| `l2policy.cu`, `stream_b2b.cu`, `l2persist.cu` | L2 replacement with ncu; carry-over between kernels; persisting window |
| `launch.cu`, `mbar.cu` | Launch costs; mbarrier wake and arrive |

Ablations of real TileLang kernels (no loads, no math, no store, grids of 1 to all blocks) live under
`agentic/tilelang/ablation/` with results in `calibration/spark-6951/ablation*/`, `clock_in_kernel.txt`,
`step_store.txt` and `tma_curve_1024.txt`. The older suite is `calibration/suite/b*.cu`.

### 9.2 Timing traps found on the way

Each of these gave a wrong number at least once.

1. **The timer left the L2 dirty.** TileLang's `do_bench` writes zeros to a 256 MB buffer before every timed run, so
   each timed kernel first pays the write-back (12-38 µs, Section 5.6). At 1024^3 this made TileLang look 1.8x behind
   cuBLASLt; under the same timing they tie (35.1 against 34.1 µs). Flush with reads, or time both sides the same way.
2. **The compiler moved the clock read.** The first `mma_types.cu` read clock64 inside the kernel and got 8x the true
   tensor rate: the compiler does not keep a clock read after register-only mma instructions. Time compute-only
   kernels with events around the whole kernel.
3. **The compiler dropped or merged loads.** The first `smem_bw.cu` read 2 cycles per ldmatrix: half the loads were
   gone. Now each address depends on the previous result through a mask that is zero at run time. `ldg_atomic.cu`
   uses `& 0` the same way. The older suite's shared-memory reading of about 256 B/clk is not used here.
4. **The compiler folded loops.** `alu.cu`'s IADD3 and I2F rows read 374 and 126 instructions per clock (the limit is
   4). `sfu.cu`'s reciprocal row read 0.1 cycles of latency, and its binary holds 9 MUFU.RCP against 261 for the other
   functions. Count the instructions in the timed loop's SASS before trusting a rate.
5. **The probe's own address math.** `tileload.cu` computed addresses with run-time integer division. That made a
   fixed cost of 0.22 µs per k-tile and a single-block rate of 37.8 GB/s, both withdrawn once `loopsteps.cu` with
   compile-time addresses showed 78 GB/s and no fixed cost (`gb10.json`, `cpasync_t0_us`).
6. **The probe's own bank conflicts.** `instlat.cu` first found cp.async limited to 16 B/clk per SM. Its lanes wrote
   shared memory 128 bytes apart. Withdrawn.
7. **A random chase over megabytes measures the TLB.** `chase.cu` reads about
   400 ns at 4 MB though the data is in the L2. L2 latency comes from 16-64 KB rings; DRAM latency from chases that
   stay inside 2 MB pages.
8. **Back-to-back runs near the L2 size.** Part of the working set survives from the last run, and how much changes by
   session: the same split-K kernel took 168, 135 and 146 µs in three sessions. For working sets of 16-36 MiB, give
   each call its own copy of the data, several copies rotating (trap 11).
9. **Noise.** One configuration timed five times in a row: 1.4 % spread at 1024^3, 0.3 % at 4096^3. Between processes
   minutes apart: about 5 %. Near 60 µs, event times step by about 1 µs. Differences under 5 % between sessions are
   noise.
10. **Profiler limits.** `do_bench(backend="cupti")` returns 0.0 ms on this host. ncu has no DRAM counters. Its L2 hit
    rates carry the 50 % floor of 64-byte fills: the older suite read 86 % hits at 24 MiB and called all 24 MiB
    usable, while above the floor re-reads at 24 MB hit 63 % (Section 5.2). Its durations sit above event times
    because it controls the clock.
11. **A flush does not make a re-read weight cold.** The tier study timed decode shapes by reading 64 MB before each
    call and subtracting that read's own time. Our kernels then came out at 118 µs for M = 32, which means reading the
    33.6 MB weight at 286 GB/s: faster than DRAM (237 GB/s) and above the 273 GB/s paper rate. cuBLASLt came out at
    the DRAM rate, and the same kernel measured 124.6 and 191.4 µs in two runs. The weight is read again on every
    call, and the L2 protects re-read data from a one-time stream (Section 5.2), so part of it stays in the L2 through
    the flush. How much depends on the kernel's own access pattern, so the method is unfair as well as wrong. The fix:
    keep several copies of the weight (256 MB or more in all) and have each call read the next one. Nothing is
    flushed or subtracted, and every call reads its weight from DRAM.
12. **Background rates read high.** `dram.cu` divides all the bytes the streamers moved by the chaser's run time, but
    the streamers start earlier, so that column reads up to 298 GB/s, above the 273 GB/s paper rate. Treat it as an
    upper bound.

## 10. Open questions

### Measured, not explained

1. **Plain loads beat copies from the L2.** Plain loads reach 3.47 TB/s; `ld.global.cg`, cp.async and TMA stop at
   1.8-2.25 TB/s. One possibility, not tested: the plain-load probe reads each block's 175 KB slice 20 times and part
   of it hits in L1.
2. **A burst of cp.async lands faster than the port.** In `cpq.cu`, 4 warps x 64 copies (128 KB) land at about 55-64
   B/clk per SM, twice the 32 B/clk any sustained stream gets.
3. **32-byte row pieces halve the L2 rate for copies** (1.11-1.13 TB/s against 2.08-2.25, `tileload.cu`, two blocks
   per SM).
4. **The L2's replacement rule.** Two thirds effective size, a gradual loss from 16 to 40 MB, and scan resistance. No
   textbook policy fits.
5. **The dirty-L2 cost levels off at 38 µs.** If write-back adds bytes / 197 GB/s (Section 5.4), that is about 7.5 MB
   written back, a third of the L2.
6. **LOP3 next to FFMA** runs 23 % below what two units allow.
7. **Short DRAM pieces.** 64-byte pieces cost 12-14 % against 256-byte pieces; the controller's reason is not known.
8. **Spread of box latency under load.** A 4 KB TMA box under load takes about twice a single line's latency (it waits
   for the slowest of its lines). Deeper pipelines gain 10 % at the DRAM ceiling (128 x 128 x 32, 2 against 4 stages),
   which the timing model does not reproduce.
9. **Rows per block in an L2-resident GEMV** (matrix times vector). 8.3 µs with 2 rows per block, 11.9 µs with 8. A
   guess: the block barrier waits for the slowest warp.
10. **DRAM's real ceiling.** The DRAM probes top out at 237-242 GB/s, yet a TMA decode GEMM read its 33.6 MB weight at
    247 GB/s, with each call reading a different copy so nothing came from the L2 (tier study run 3). The paper rate
    is 273 GB/s. Which access pattern gets closest to it is not known.
11. **The chip's store limit** (0.31-0.38 TB/s for half-sector stores, 1.4-1.5 TB/s for 16-byte stores). Where in the
    chip it sits is not known.

### Not measured

- FP64 math (the simulator uses 8x FFMA as a stand-in, `opcodes.md`).
- The block-scaled fp4 mma (`mxf4nvf4` m16n8k64).
- `setmaxnreg` on `sm_121a`.
- Register-file bank conflicts: CUDA cannot choose registers.
- Integer add and compare rates (folded), the uniform datapath, branches, constant-load latency.
- The L2's slices and set hash; the DRAM channel count.
- DSMEM rate and latency (the older readings do not agree with each other); multicast beyond the older reading;
  programmatic dependent launch (letting the next kernel start before the current one ends).
- Clock and power past 20 s of load.
- With the newer probes: L1 size per shared-memory split, unified-memory rates, and shuffles (the older suite: one
  shuffle per 8 cycles per sub-partition, 27-35 cycles latency, B39).
- The CPU side beyond its core count and copy rate.

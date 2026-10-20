# The GEMM template and its design space: tiers 1 to 4

*Oct 20 2026. Written from the code at commit 1e31b9f, which holds the run-3 timing fix. Every number names the file
it comes from. Where a note and the code disagree, this note follows the code.*

## 1. What this note is

We build fp16 GEMM kernels for one GPU: the NVIDIA GB10 in the DGX Spark. The kernels are written in TileLang, a
Python language that compiles to CUDA. One template renders every kernel we try. Each knob of the template is one
decision. The knobs fall into four tiers, by who can reach them: TileLang's own tuner, TileLang's language, a
different program structure, or something outside any TileLang program.

This note covers:
- what code the template makes (section 3);
- the four tiers (section 4) and every knob in them (sections 5 to 8);
- the rules that make a choice legal (section 9);
- how a solver picks a kernel, and why each step is there (section 10);
- what each tier bought on seven shapes (section 11);
- what is still open (section 12).

The problem is always C = A·B. A is M × K, B is K × N, both fp16 and row-major. Products add up in fp32; C comes
out in fp16. A shape is written M×N×K; "1024^3" means M = N = K = 1024. Two of the seven shapes are "decode"
shapes, 32×4096×4096 and 128×4096×4096: a few rows times a 4096 × 4096 weight, like one step of language-model
decoding. There the 32 MiB weight is read once and does not fit in the cache.

## 2. Background

### 2.1 Words used here

| Word | Meaning here |
|---|---|
| SM | Streaming multiprocessor, one of the GPU's 48 cores. Each has 4 sub-partitions, tensor units, registers and shared memory. |
| warp | 32 threads that issue instructions together. |
| block | A group of threads that runs on one SM and shares its shared memory. Here: 128 or 256 compute threads, plus 128 loader threads on the TMA path. The grid is all blocks of one launch. |
| block tile | The block_M × block_N piece of C that one block computes. |
| k-tile | What one loop step loads: a block_M × block_K piece of A and a block_K × block_N piece of B. |
| stage | One shared-memory buffer holding one k-tile. num_stages buffers form a ring, so loads can run ahead of the math. |
| occupancy, R | Blocks resident per SM. Set by whichever runs out first: registers, shared memory, threads, or the 24-block cap. |
| wave | The blocks that run at once: 48 × R. A grid of 300 blocks with R = 2 runs as waves of 96, 96, 96 and 12. |
| mma.sync m16n8k16 | The tensor instruction on this chip. One warp multiplies a 16×16 fp16 piece of A by a 16×8 piece of B and adds into 16×8 fp32 accumulators, 4 per thread. |
| ldmatrix, stmatrix | Warp instructions that move 8×8 fp16 matrices between shared memory and registers. |
| cp.async | A copy each thread issues: 16 bytes from global memory into shared memory, not through registers. Threads commit copies as a group and later wait for the group. |
| TMA | Tensor Memory Accelerator. A copy engine per SM. One thread gives it a tensor descriptor and box coordinates; it moves the whole 2-D box into shared memory and reports on an mbarrier. |
| mbarrier | A barrier object in shared memory. It counts thread arrivals and, for TMA, bytes landed. Threads wait on it. |
| warp specialization | Splitting a block's warps into a producer group that only loads and consumers that only compute. TileLang's producer group is 128 threads. |
| L2 | The 24 MB cache all SMs share. About 16.8 MB of it keeps reused data. |
| DRAM | The LPDDR5X memory. The CPU shares it with the GPU. |
| launch-order swizzle | The order blocks are launched in (TileLang's `T.use_swizzle`, "sw" in kernel names). It decides which blocks run together and so what they share in the L2. |
| shared-memory swizzle | A permuted address layout inside shared memory that avoids bank conflicts (the `smem_layout` knob). Not the same thing as the launch-order swizzle. |
| warm, rot, cold | Warm: calls back to back on the same inputs, so operands that fit stay in the L2. Rot (rotating weights): each call reads the next of several copies of B, 256 MB or more in all, so B always comes from DRAM, as in decode, where every layer has its own weights. Cold: a 64 MB read before each call with its own time subtracted. Runs 1 and 2 used cold for decode; it turned out not to be cold (12.2), and run 3 replaced it with rot. |
| SASS | The GPU's machine code: what a compiled kernel really runs. |
| ptxas | NVIDIA's assembler. It turns PTX into SASS and assigns registers. |
| launch bounds | `__launch_bounds__(threads, min_blocks)` on a kernel. It tells ptxas the block size, so ptxas caps registers per thread to make the block fit. |
| setmaxnreg | An instruction that moves registers between warp groups at run time. TileLang gives its producers few and its consumers many. |
| spill | When a thread needs more registers than allowed, ptxas keeps some values in local memory and reloads them. Slow. |
| Little's law | Bytes in flight = rate × latency. To hide a latency at a given rate, that many bytes must be on the way. |
| LRU | Least recently used: a cache that evicts what was touched longest ago. |
| price | The time the analytical model predicts for a kernel, before compiling anything. |
| column | One full legal assignment of the knobs the solver chooses together: one candidate kernel. |

### 2.2 The GB10 in numbers

All from `calibration/spark-6951/gb10.json`, which names a probe or the driver for each value.

| Unit | Value | Source in gb10.json |
|---|---|---|
| SMs | 48 | driver |
| Clock under load | 2.52 GHz | clock.cu |
| Tensor rate, fp16 in, fp32 accumulate | 1,024 FLOP per clock per SM; 1,024 × 48 × 2.52 GHz = 124 TFLOP/s for the chip | mma_rate.cu |
| Tensor rate inside a real GEMM loop | 83 % of that with 1 warp per sub-partition, 90 % with 2 or more | ablation of 22 kernels |
| Shared memory | 100 KB per SM (102,400 B); 1 KB reserved per block | driver |
| Registers | 65,536 per SM, given out per warp in units of 256 | driver |
| Threads, blocks | 1,536 threads and 24 blocks per SM | driver |
| L2 | 24 MB; 16.8 MB effective for reuse | driver; reuse probe |
| L2 to SMs | 3.47 TB/s for the chip; 80.7 GB/s per SM port | mlp_stream.cu |
| DRAM | 237 GB/s read (spec 273); 197 GB/s write | mlp_stream.cu, dram.cu |
| Persisting L2 set-aside | up to 18 MiB | tier4_query.cu |
| Clusters | up to 8 blocks; distributed shared memory works | tier4_query.cu |
| Launch | 2.6 µs per kernel from a stream; 0.43 µs inside a CUDA graph | ablation; launch.cu |

### 2.3 TileLang on this chip

From `runs/decide/tilelang_controls_survey.md` (a read of the TileLang source at commit fe6d319c):
- The target is `sm_121a`. Every `T.gemm` lowers to `mma.sync`. WGMMA is Hopper-only and tcgen05 is sm_100-only, so
  neither exists here.
- By default, a `T.Pipelined` loop with 1 or more stages and TMA-eligible copies becomes a warp-specialized kernel:
  a 128-thread producer group loads with TMA. Turn warp specialization off and TileLang loads with cp.async instead.
- TileLang's AutoTuner has no search space of its own. It sweeps a list of configurations the user writes.
- TileLang's carver (its configuration generator) sweeps tile sizes, threads and copy width. On sm_120/121 it turns
  pipelining off and knows nothing of TMA, warp specialization or clusters.

## 3. What the template generates

### 3.1 Where it lives

- `agentic/tilelang/families/gemm/kernel.py.j2` is a Jinja template. It renders one TileLang program.
- `agentic/decide/template.py` holds `DEFAULTS`, `render(assign)` and `name(assign)`.
- `DEFAULTS` is TileLang's own example GEMM: 128×128×32 tile, 3 stages, 128 threads, no swizzle, no prefetch, warp
  policy 0, copy width 0, TMA with the producer group, inferred layout, direct C store, no split-K.
- M, N and K are compile-time constants (`T.const`). The study compiles one kernel per shape with
  `matmul.compile(M=..., N=..., K=...)`.

### 3.2 One rendered kernel

This kernel tied for fastest at 1024^3: 29.1 µs in run 3, against 29.0 for the same kernel without prefetch. Here is
`cpasync_64x128x32_st2_th256_sw0_pf1_wp2_vw0_inf_staged_sk1` as `render()` prints it:

```python
import tilelang
import tilelang.language as T
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.cuda.language import copy as cuda_copy


@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def matmul(A, B, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)
    with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 64), threads=256) as (bx, by):
        A_shared = T.alloc_shared((64, 32), dtype)
        B_shared = T.alloc_shared((32, 128), dtype)
        C_local = T.alloc_fragment((64, 128), accum_dtype)
        C_shared = T.alloc_shared((64, 128), dtype)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, 32), num_stages=2, order=[0, 1, 2], stage=[0, 0, 1]):
            cuda_copy(A[by * 64, k * 32], A_shared, disable_tma=True)
            cuda_copy(B[k * 32, bx * 128], B_shared, disable_tma=True)
            T.gemm(A_shared, B_shared, C_local, policy=GemmWarpPolicy.FullCol)

        T.copy(C_local, C_shared)
        T.copy(C_shared, C[by * 64, bx * 128])

    return C
```

### 3.3 Top to bottom

**Grid and block tile.** `T.Kernel(ceildiv(N, block_N), ceildiv(M, block_M), threads=...)` makes one block per block
tile. `bx` walks along N, `by` along M. At 1024^3 with 64×128 tiles that is 8 × 16 = 128 blocks.

**Buffers.** `A_shared` and `B_shared` hold one k-tile; TileLang makes `num_stages` copies of each for the ring.
`C_local` is the fp32 accumulator tile, spread over the compute threads' registers: 64 × 128 / 256 = 32 values per
thread here. Shared memory per block is stages × (block_M + block_N) × block_K × 2 bytes: 2 × (64 + 128) × 32 × 2 =
24,576 bytes in this example.

**The k loop.** `T.Pipelined(ceildiv(K, block_K), num_stages=s)` runs one step per k-tile: 1024 / 32 = 32 steps here.
TileLang pipelines the loop: while the math works on k-tile k, the loads for later k-tiles are already in flight,
up to s − 1 ahead.

**Prefetch (manual schedule).** When `prefetch` = p > 0, the template adds `order=[0, 1, 2], stage=[0, 0, p]`. The two
copies sit in stage 0 and the gemm in stage p, so the gemm uses data loaded p steps earlier. TileLang then sets the
pipeline depth to max(stage) + 1 = p + 1, whatever `num_stages` says (survey; the legality check in
`agentic/tilelang/tools.py` says the same). The machine code confirms it: cp.async 64×128×32 at 3 stages with
prefetch 1 compiles to the same code as 2 stages without prefetch (same code hash in
`runs/decide/cover_code_fingerprints.json`).

**Two load paths.** One decision picks the path.
- *TMA with a producer group* (`copy_path=tma`, `warp_spec=1`, the default). The template calls `T.copy(global,
  shared)`. TileLang adds 128 producer threads to the block. One elected thread issues TMA box loads for a stage and
  tells that stage's "full" mbarrier how many bytes to expect. The consumers wait on "full", compute, then arrive on
  the stage's "empty" mbarrier so the producer can refill it. TileLang sets 24 registers for the producers and 240
  for the consumers (visible in the generated code below).
- *cp.async, no producer group* (`copy_path=cpasync`, `warp_spec=0`). The template sets the pass config
  `tl.disable_warp_specialized` and calls TileLang's CUDA copy with `disable_tma=True`. Every compute thread also
  loads: it issues 16-byte cp.async copies for a later k-tile, commits them as a group, waits until the current
  k-tile has landed, and meets the other threads at a block barrier before the math.
- These two knobs act as one. With the producer group off, TileLang emits the cp.async kernel even when TMA is
  asked for. With cp.async and the producer group on, it also emits the cp.async kernel. Both facts are same-code
  results in the fingerprint file.

**Compute.** `T.gemm(A_shared, B_shared, C_local)` becomes, per 16-wide step of k, ldmatrix loads of A fragments,
transposing ldmatrix loads of B fragments, and mma.sync m16n8k16. The warp policy says how the compute warps share the
C tile. Square (the default) picks the most square warp tile; FullRow stacks all warps along M; FullCol lines them up
along N. In this example FullCol gives each of the 8 warps a 64 × 16 strip, where Square would give 32 × 32
(`warp_grid` in `agentic/gb10/model.py`).

**Epilogue, the C store.**
- *direct*: `T.copy(C_local, C[...])`. Each thread converts its accumulators to fp16 and writes them straight to
  global memory, two fp16 values (4 bytes) per store. These stores fill only half of each 32-byte sector. The two SMs
  of a pair share one store path: 15 cycles per such warp store; the chip tops out near 0.31-0.38 TB/s (gb10.json,
  storepat.cu).
- *staged*: `T.copy(C_local, C_shared)` then `T.copy(C_shared, C[...])`. stmatrix writes the fp16 tile into shared
  memory, then one TMA store sends it out: 33.1 bytes per clock per SM pair, 482 for the chip, plus 457 cycles fixed
  (gb10.json, tmastore.cu). It needs block_M × block_N × 2 bytes of shared memory, which TileLang takes from the
  pipeline buffers once the loop is done (comment in `agentic/decide/cover.py`).

**Split-K.** With `split_k` = s > 1 the grid gets a third axis of size s. Block `bz` covers its 1/s share of K. It
does not store C. It adds its fp32 partial tile into C with `T.atomic_add`. So C becomes an fp32 input that the caller
must zero, and the kernel returns nothing. The chip adds accumulator-layout fp32 atomics at 225 bytes per clock
(gb10.json, ldg_atomic.cu).

### 3.4 How the TMA and split-K versions differ

The fastest kernel at 8192^3, `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`, renders the same
skeleton with these lines changed:

```python
@tilelang.jit
def matmul(A, B, dtype=T.float16, accum_dtype=T.float32):
    ...
    with T.Kernel(T.ceildiv(N, 256), T.ceildiv(M, 128), threads=256) as (bx, by):
        ...
        T.use_swizzle(panel_size=10)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, 64), num_stages=2):
            T.copy(A[by * 128, k * 64], A_shared)
            T.copy(B[k * 64, bx * 256], B_shared)
            T.gemm(A_shared, B_shared, C_local)
```

A split-K kernel, `cpasync_128x128x32_st3_th128_sw0_pf0_wp0_vw0_inf_direct_sk2`:

```python
@tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
def matmul(A, B, C, dtype=T.float16, accum_dtype=T.float32):
    ...
    C: T.Tensor((M, N), accum_dtype)
    with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), 2, threads=128) as (bx, by, bz):
        ...
        for k in T.Pipelined(T.ceildiv(K // 2, 32), num_stages=3):
            cuda_copy(A[by * 128, bz * (K // 2) + k * 32], A_shared, disable_tma=True)
            cuda_copy(B[bz * (K // 2) + k * 32, bx * 128], B_shared, disable_tma=True)
            T.gemm(A_shared, B_shared, C_local)

        T.atomic_add(C[by * 128, bx * 128], C_local)
```

### 3.5 What TileLang turns it into

TileLang's CUDA for the two paths, saved by earlier ablation studies. Address arithmetic and timer code are cut;
the structure is as generated.

cp.async, 128×128×32, 3 stages, 128 threads, direct store, at 1024^3
(`calibration/spark-6951/ablation_cpasync/c128x128x32_st3_th128.kernel.cu`):

```cuda
__launch_bounds__(128, 1) matmul_kernel(const half_t* A, const half_t* B, half_t* C) {
  float C_local[128];                                  // 128 x 128 fp32 / 128 threads
  // prologue: cp.async k-tiles 0 and 1, one commit group each
  for (int k = 0; k < 30; ++k) {                       // 32 k-tiles; the last two run after the loop
    __syncthreads();
    for (int i = 0; i < 4; ++i) tl::cp_async_gs<16>(&A_shared[(k + 2) % 3 ...], &A[...]);
    for (int i = 0; i < 4; ++i) tl::cp_async_gs<16>(&B_shared[(k + 2) % 3 ...], &B[...]);
    tl::cp_async_commit();
    tl::cp_async_wait<2>();                            // k-tile k has landed; k+1, k+2 still in flight
    __syncthreads();
    for (int ki = 0; ki < 2; ++ki) {                   // two 16-wide steps per 32-wide k-tile
      tl::ptx_ldmatrix_x4(...);  tl::ptx_ldmatrix_x4_trans(...);
      tl::mma_sync<kFloat16, kFloat16, kFloat32, 16, 8, 16, false, true>(...);   // 32 per warp per step
    }
  }
  // drain the last two k-tiles, then the direct store: 64 x 4-byte stores per thread
  for (int i = 0; i < 64; ++i) *(uint1*)(C + ...) = two fp16 values;
}
```

TMA with a producer group, 64×128×128, 2 stages, 128 compute threads, launch-order panel 8, direct store, at 1024^3
(`calibration/spark-6951/ablation/1024/full.kernel.cu`):

```cuda
__launch_bounds__(256, 1) matmul_kernel(const CUtensorMap A_desc, const CUtensorMap B_desc, half_t* C) {
  // mbarrier[0..1]: stage is full (bytes landed); mbarrier[2..3]: stage is empty (128 consumer arrivals)
  const dim3 blockIdx = tl::rasterization2DRow<8>();             // T.use_swizzle(panel_size=8)
  if (threadIdx.x < 128) {                                       // producer group
    tl::warpgroup_reg_dealloc<24>();
    if (threadIdx.x < 32) for (int k = 0; k < 8; ++k) {          // one warp; one elected lane issues
      mbarrier[(k & 1) + 2].wait(...);                           // wait until the stage is empty
      mbarrier[k & 1].expect_transaction(16384);
      tl::tma_load(A_desc, mbarrier[k & 1], ...);                // A: 2 boxes, 16 KB
      mbarrier[k & 1].arrive_and_expect_tx(32768);
      tl::tma_load(B_desc, mbarrier[k & 1], ...);                // B: 2 boxes, 32 KB
    }
  } else {                                                       // 128 consumer threads
    tl::warpgroup_reg_alloc<240>();
    for (int k = 0; k < 8; ++k) {
      mbarrier[k & 1].wait(...);                                 // wait until the stage is full
      for (int ki = 0; ki < 8; ++ki) { ldmatrix ...; mma_sync<..., 16, 8, 16, ...>(...); }
      mbarrier[(k & 1) + 2].arrive();                            // hand the stage back
    }
    for (int i = 0; i < 32; ++i) *(uint1*)(C + ...) = two fp16 values;
  }
}
```

The staged store has no saved CUDA in the repo. Per the survey it lowers to stmatrix into `C_shared`, then a TMA store
followed by a wait on the store's reads.

### 3.6 Reading a kernel name

`name()` in `agentic/decide/template.py` prints every knob. Take
`cpasync_128x128x32_st3_th128_sw10_pf0_wp2_vw0_inf_staged_sk1`:

| Part | Knob | Here |
|---|---|---|
| `cpasync` | copy_path; `_ws` follows when warp_spec = 1 (`tma_ws`) | cp.async loads, no producer group |
| `128x128x32` | block_M x block_N x block_K | 128 × 128 tile of C, 32-wide k-tiles |
| `st3` | num_stages | 3 shared-memory buffers |
| `th128` | threads | 128 compute threads (4 warps); the TMA path launches 128 more |
| `sw10` | swizzle | launch order in panels of 10 tile rows (0 = row-major) |
| `pf0` | prefetch | no manual schedule |
| `wp2` | warp_policy | 0 Square, 1 FullRow, 2 FullCol |
| `vw0` | vec_width | copy width left to TileLang (8 = 16 bytes per lane) |
| `inf` | smem_layout, first 3 letters | `inf` inferred by TileLang, `swi` swizzled |
| `staged` | c_store | `direct` or `staged` |
| `sk1` | split_k | 1 = no split |

A name like `tma_128x...` (TMA asked, producer group off) compiles to the cp.async kernel (3.3).

## 4. The four tiers

| Tier | Who can reach it | Knobs |
|---|---|---|
| T1 | TileLang's own tuning, on TileLang's default structure (TMA with the producer group) | block tile, num_stages, threads, launch-order swizzle, copy width |
| T2 | TileLang's language and pass configs, but no TileLang tuner lists them | copy path with producer group, C store, prefetch, warp policy, shared-memory layout; plus controls not yet in the template (6.2) |
| T3 | A different program structure (a "skeleton") that TileLang can express | split-K with atomics (in the template); stream-K, GEMV split-K, fp4 fragment kernels (separate families) |
| T4 | Hardware features no TileLang skeleton can state: launch-level settings, our own CUDA or PTX, or a choice that spans kernels. TileLang itself is never changed. | CUDA graphs, L2 persistence through the runtime, streams, and others (section 8) |

Notes on the lines between tiers:
- The sources list the copy path twice: as a tier-2 control (`T.copy(disable_tma=)`, `tl.disable_warp_specialized`)
  and as a tier-3 skeleton (`registry_tier34.yaml`). The study counts cp.async in T2, and so does this note.
- The study's T1 uses the staged C store, because TileLang's autotune example writes C that way. T2 adds the direct
  store.
- The survey moved L2 persistence, cluster multicast and programmatic dependent launch from tier 4 down to tier 2:
  TileLang can ask for them. Their payoff and pricing still live in the tier-4 registry.
- Each tier contains the smaller ones. "T1+T2" means the knobs of both.

What the study (`agentic/decide/study/run.py`, `TIERS`) lets each tier choose:

| Tier in the study | Copy path | C store | split_k | Also |
|---|---|---|---|---|
| T1 | TMA + producer group | staged | 1 | tile, stages 2-6, threads, swizzle {0, 10} |
| T1+T2 | + cp.async | + direct | 1 | prefetch 1 and FullCol variants of cp.async candidates |
| T1+T2+T3 | same | same | + 2, 4, 8 | |
| T1-T4 | the best kernel found | | | measured inside a CUDA graph (runs 1 and 2 also tried an L2 persistence window on decode; see 8.2) |

## 5. Tier 1: TileLang's tuning knobs

Measured effects marked "cover set" come from `runs/decide/cover_code_fingerprints.json`: one knob changed at a time
around a base kernel, all compiled, checked and timed. That set used the earlier single-launch timer
(`agentic/decide/remote_measure.py`: each call alone between two events, median of 40). Its "cold" (a 64 MB read,
then one call) put the decode kernels at the DRAM floor, unlike the study's flush-and-subtract cold (12.2). "Run 3"
means the tier study in `runs/decide/study_v3/`. Times in µs.

### block_M, block_N, block_K (the block tile)

| | |
|---|---|
| Legal values | block_M 16, 32, 64, 128, 256; block_N 32, 64, 128, 256; block_K 32, 64, 128 (`DOMAINS` in `pipeline.py`), within section 9's rules |
| What it trades | A k-tile brings (block_M + block_N) × block_K × 2 bytes for 2 × block_M × block_N × block_K FLOP, so bigger tiles load fewer bytes per FLOP. They also need more shared memory per stage and more accumulator registers per thread, and they make fewer blocks. Fewer blocks than 48 leave SMs idle. block_K sets the number of loop steps and the bytes per step. |
| Price model | Yes: tile and wave counts; occupancy; tensor, shared-memory and copy time per k-tile; DRAM bytes from the L2 simulation (10.2) |
| In template | Yes |
| Measured | Cover set, 1024^3, TMA base 128×128×32: 36.7 (128×64×32) to 116.5 (256×128×32, which spills). cp.async base 64×128×32: 34.6 to 53.1 (64×256×32). 128×4096×4096 cold, TMA base: 150.5 to 448.5 (256×128×32). |

### num_stages

| | |
|---|---|
| Legal values | 2 to 6, within the shared-memory limit |
| What it trades | More buffers keep more loads in flight and hide more memory latency. Each stage costs (block_M + block_N) × block_K × 2 bytes of shared memory, which can cut blocks per SM. |
| Price model | Yes: the wait per k-tile is the latency divided by (stages − 1), Little's law; shared memory feeds occupancy |
| In template | Yes |
| Measured | Cover set, 1024^3 TMA base: flat, 45.0 to 46.9 over 2-6. cp.async base: 2 stages best, 32.5 against 34.6 at 3. 128×4096×4096 cold TMA base: 165.9 at 2, 154.1 at 3, 150.5 at 4-6. |

### threads

| | |
|---|---|
| Legal values | 128 or 256 compute threads (the TMA path launches 128 more) |
| What it trades | More warps keep the tensor units busier (83 % of peak with 1 warp per sub-partition, 90 % with 2 or more) and halve the accumulators per thread. But the register cap per thread drops: on the TMA path 240 at 128 threads and 168 at 256. |
| Price model | Yes: warps per sub-partition set the tensor efficiency; registers and threads set occupancy and legality |
| In template | Yes |
| Measured | Cover set, 1024^3: TMA base 40.8 at 256 against 46.8 at 128; cp.async base 34.6 at 256 against 38.7 at 128. 128×4096×4096 cold: 149.9 against 150.5. |

### swizzle (launch order)

| | |
|---|---|
| Legal values | 0 (row-major) or 10 in the search. The template takes any panel; the cover set tried 4, 8, 16. |
| What it trades | Which blocks run in the same wave, and so which A row panels and B column panels they share in the L2. The work per block does not change. |
| Price model | Yes, since run 2. `launch_order()` copies TileLang's order; `dram_waves()` runs it through a 16.8 MB L2 and counts misses. Run 1 priced every column at swizzle 0 by mistake (10.6). |
| In template | Yes (`T.use_swizzle(panel_size=...)`) |
| Measured | Run 3, 4096^3, staged store: cp.async 128×128×32 with 3 stages took 2,071.7 at sw0 and 1,542.9 at sw10; TMA 128×256×32 with 4 stages took 2,030.0 and 1,573.1. 8192^3: 12,674.1 in run 1, when swizzle was priced at 0 (10.6), against 11,271.9 in run 3. Cover set at 1024^3, where everything fits the L2: same warm, up to 4 % faster cold. |

How the panel order works: with panel P, launch walks down P tile rows of one column, then the next column; every
other panel runs right to left. An example from `launch_order()` at 8192^3 with 128 × 256 tiles (64 × 32 tiles) and
one block per SM, so a wave is 48 blocks:

| First wave | Row-major (sw0) | Panel 10 (sw10) |
|---|---:|---:|
| A row panels touched (2 MiB each) | 2 | 10 |
| B column panels touched (4 MiB each) | 32 | 5 |
| Bytes in those panels | 132 MiB | 40 MiB |

Row-major needs every B panel again in every wave, far more than the 16.8 MB the L2 keeps. So B streams from DRAM
again and again. Panel order keeps a wave's footprint small and lets the next wave reuse the same A panels.

### vec_width (copy width)

| | |
|---|---|
| Legal values | 0 (TileLang decides) or 8 elements = 16 bytes per lane |
| What it trades | The width of each copy instruction |
| Price model | No |
| In template | Yes (`coalesced_width=`) |
| Measured | Cover set: the same machine code as the default on both paths. The search leaves it at 0. TileLang's carver does sweep it, which is why it sits in T1. |

## 6. Tier 2: TileLang controls no tuner offers

### 6.1 In the template

#### copy path with producer group

| | |
|---|---|
| Legal values | `tma_ws`: TMA plus the 128-thread producer group. `cpasync`: every compute thread loads. |
| What it trades | TMA: one thread moves whole boxes and the compute warps never issue loads. But the extra 128 threads cut the register cap, and every k-tile pays an mbarrier hand-off (0.09 µs per k-tile in the ablation of the TMA kernels; a waiter wakes 134 cycles after the arrive, mbar.cu). cp.async: no extra threads, up to 255 registers per thread, often more blocks per SM. But the compute warps spend issue slots on copies and meet at a block barrier every step. |
| Price model | Yes, one formula per path. TMA: 81 GB/s per block alone, 1.86 TB/s for the chip; the slowest of compute, copy and latency, plus the hand-off. cp.async: 71 GB/s per block, 2.25 TB/s for the chip; inside a block compute and loads overlap partly, (compute² + loads²)^½; across blocks fully (all gb10.json). |
| In template | Yes (`copy_path`, `warp_spec`) |
| Measured | Run 3, best of each path. 1024^3: cp.async 29.0, TMA 32.9. 4096^3: cp.async 1,510.9, TMA 1,573.1. 8192^3: TMA 11,271.9, cp.async 12,053.6. M = 128: TMA 153.2, cp.async 163.2. Ties (within 3 %) at 2048^3, 4096×4096×1024 and M = 32. |

#### c_store

| | |
|---|---|
| Legal values | `direct` or `staged`; split-K needs `direct` |
| What it trades | Direct: no shared memory, but half-sector 4-byte stores through a store path two SMs share. Staged: block_M × block_N × 2 bytes of shared memory and stmatrix work, then one wide TMA store. |
| Price model | Yes (`store_us` in `price.py`): only the last wave's stores count, because earlier waves' stores hide under the next wave's loads |
| In template | Yes |
| Measured | Cover set: 1024^3 TMA base 46.8 to 42.8 (−9 %), cp.async base 34.6 to 30.5 (−12 %); 128×4096×4096 cold 150.5 to 148.4 (−1.4 %). Run 3: six of seven winners are staged. At 4096^3 the direct store won, 1,510.9 against 1,527.9, a tie. |

#### prefetch

| | |
|---|---|
| Legal values | 0, or p < num_stages. The search adds p = 1 to cp.async candidates. |
| What it trades | The manual schedule from 3.3: the gemm runs p steps behind the loads, and the depth becomes p + 1 |
| Price model | No. Variants get their base kernel's price. |
| In template | Yes |
| Measured | TMA path: same machine code. cp.async 64×128×32, 1024^3: prefetch 1 at 3 stages is the same code as 2 stages, 32.6 against 34.6. Run 3, 1024^3 winner: 29.0 without prefetch and 29.1 with it, both at 2 stages. |

#### warp_policy

| | |
|---|---|
| Legal values | 0 Square (TileLang's default), 1 FullRow, 2 FullCol. The search adds 2 to cp.async candidates. |
| What it trades | How the compute warps split the C tile. That sets how often each warp re-reads A and B from shared memory with ldmatrix, and the accumulator layout. |
| Price model | Not in the search. The model has a term for warp layout, but the columns never pass the policy, so all variants get one price. |
| In template | Yes (`policy=GemmWarpPolicy...`) |
| Measured | Cover set, cp.async base at 1024^3: FullCol 32.6 (−6 %), FullRow 38.7 (+12 %) against 34.6. TMA base: the code changes, the time does not (46.9). Run 3, 4096^3, cp.async 128×128×32, 3 stages, sw10: FullCol 1,510.9 against Square 1,574.5 with the direct store (−4 %); 1,527.9 against 1,542.9 with the staged store (a tie). |

#### smem_layout

| | |
|---|---|
| Legal values | `inferred` (TileLang picks) or `swizzled` (`make_swizzled_layout`). A plain linear layout is excluded. |
| What it trades | Shared-memory bank conflicts for ldmatrix |
| Price model | Only a linear layout costs anything (4- to 8-way conflicts); both legal values count as conflict-free |
| In template | Yes (`T.annotate_layout`) |
| Measured | Cover set: `swizzled` is the same machine code as `inferred` on both paths. A linear layout with TMA gives wrong answers (31 of 31 tries, `agentic/tilelang/tools.py`). |

### 6.2 Tier 2, not yet in the template

None of these is implemented in the template. They come from the source survey. "Measured" means a probe outside
TileLang measured the mechanism; no TileLang kernel here uses it yet.

| Control | What it would change | Price model | Measured |
|---|---|---|---|
| `eviction_policy` on TMA copies (evict_normal, evict_first, evict_last) | An L2 hint per TMA box; cp.async ignores it. Streamed weights could be marked evict_first. | No | No |
| `T.annotate_l2_hit_ratio({buf: r})` | The host sets a stream access-policy window and the persisting L2 size: the same mechanism as the tier-4 L2 pin (8.2), from inside TileLang | `pinned_dram_saving_us` in `price.py` | Through the runtime API only (8.2) |
| `T.Kernel(cluster_dims=)` with `T.copy_cluster` or `T.tma_copy(cluster_mask=)` | Thread-block clusters with TMA multicast: one L2 read feeds up to 8 SMs | No | Clusters of up to 8 and distributed shared memory work (tier4_query.cu); multicast rate not measured |
| `T.pdl_trigger`, `T.pdl_sync` | Programmatic dependent launch: the next kernel starts while this one ends | No (only stream and graph launch costs are priced) | No |
| `T.annotate_producer_reg_dealloc(n)`, `T.annotate_consumer_reg_alloc(n)` | The setmaxnreg split between producer and consumers, 24 to 240; today 24 and 240 | No | No |
| `T.annotate_min_blocks_per_sm(n)` | The second argument of `__launch_bounds__` (today 1): caps registers, may raise blocks per SM | No | No |
| `T.Persistent`, `T.PersistentTileScheduler` | Persistent blocks that loop over tiles instead of one block per tile | No | No |
| `T.tma_copy(barrier=)`, `T.ws`, `T.annotate_ws_schedule` | Manual producer schedules: how many boxes go out before a wait | No | No |
| `T.atomic_add(use_tma=True)` | Split-K's C added by a bulk TMA reduce from shared memory | No | No |
| `T.use_swizzle(order="column")` | Column-major panel order | No | No |
| `-DTL_ENABLE_L2_PREFETCH` (compile flag) | `.L2::128B` prefetch on cp.async loads | No | No |
| `T.mma_gemm_blockscaled` | Block-scaled fp4 mma (`m16n8k64 kind::mxf4nvf4`), the one sm_120-specific GEMM path | No | Rate not measured |
| Flags: `tl.disable_vectorize_256`, `tl.enable_fast_math`, `tl.ptxas_register_usage_level`, `tl.layout_cost_model` | 256-bit loads, fast math, ptxas register use, fragment layout choice | No | No |

## 7. Tier 3: skeletons

### split_k (in the template)

| | |
|---|---|
| Legal values | 1, 2, 4, 8 in the study. Only on the cp.async path, with the direct store, and with K / split a multiple of block_K. |
| What it trades | s times more blocks, each over K / s. That fills idle SMs when M × N makes few tiles. The cost: C in fp32 (twice the bytes), zeroed first, and s fp32 atomic adds per C element at 225 bytes per clock for the chip. When the fp32 C does not fit the L2, every add reads and writes C in DRAM. |
| Price model | Yes: a stand-in shape (M·s) × N × (K / s) for the main loop, plus `splitk_extra_us` for the atomics and their DRAM traffic |
| In template | Yes |
| Measured | Cover set, 1024^3, cp.async 128×128×32: 40.7 (split 1), 42.8 (2), 55.0 (4). 128×4096×4096 cold: 150.5 (1), 168.9 (2), 167.9 (4). Run 1, 8192^3, split 8: priced 15.0 ms, measured 31.0 to 46.7 ms. Runs 2 and 3: no split-K column reached the MILP's top 60 for any shape, so none was compiled or timed. |

### Other skeletons (separate TileLang programs, not in the template)

From `agentic/decide/registry_tier34.yaml`:

| Skeleton | What it changes | Status |
|---|---|---|
| stream-K (48, 96 or 144 persistent programs) | Persistent blocks split the last wave's work instead of leaving SMs idle | The example does not compile at the pinned TileLang commit |
| GEMV split-K (rows per block 2-8, 32 or 64 lanes along K) | Rows per block and the shared-memory reduction | Measured: 8.3 to 12.4 µs at N = 2048 warm; simulator only, not in the search |
| fp4 fragment kernel (W4A16) | A different mma format and fragment layout | Untested |

## 8. Tier 4: beyond any TileLang skeleton

### 8.1 CUDA graph (tested in the study)

| | |
|---|---|
| Legal values | Launch from a stream, or replay a captured CUDA graph |
| What it trades | The host's launch cost: 2.6 µs per kernel from a stream, 0.43 µs (plus 0.43 ns per block) in a graph (launch.cu) |
| Price model | Yes (`launch_us` in `price.py`), but the study's columns are priced as stream launches |
| In template | No; it is a launch setting |
| Measured | Run 3: 1024^3 29.0 to 27.7 (−4 %), the only gain beyond 3 %. Elsewhere a tie or slower: 2048^3 186.6 to 190.1; 4096^3 1,510.9 to 1,540.7; 8192^3 11,271.9 to 11,329.6; 4096×4096×1024 380.4 to 423.4; M = 32 (rot) 137.1 to 142.2; M = 128 (rot) 153.2 to 152.4. |

### 8.2 L2 persistence window (not tested in the study now)

| | |
|---|---|
| Legal values | Per tensor: bytes held in the persisting L2, up to 18 MiB in total across all tensors |
| What it trades | The L2 keeps the window's lines across other traffic instead of by chance |
| Price model | `pinned_dram_saving_us` in `price.py`; the MILP has resource caps for a shared budget across kernels, not used yet |
| In template | No. Runs 1 and 2 set it with the driver API: an 18 MiB set-aside and an access-policy window on B, which for a 32 MiB B marks 18/32 of it as persisting. Run 3 dropped it. |
| Measured | Probe l2persist.cu: 8 MB of weights read after 64 MB of other traffic, 37.7 to 8.0 µs; 4-16 MB: 3.4 to 5x. In the study it is untested. Runs 1 and 2 timed it under the 64 MB flush, which did not measure it (12.2), so their results do not count. With rotating weights it needs its own experiment: which layers' weights to keep (comment in `run.py`). |

### 8.3 Other tier-4 decisions (not in the study)

From `registry_tier34.yaml`:

| Decision | How it is set | Mechanism | Evidence | Status |
|---|---|---|---|---|
| TMA producer issue pattern | Our own CUDA producer loop | Bytes in flight per SM on the TMA path, against DRAM latency under load | dramlat.cu | Hypothesis |
| Prefetch the next kernel's weights into L2 | Our CUDA (`cp.async.bulk.prefetch.L2`) | Moves a later kernel's DRAM reads under this kernel's compute | None | Hypothesis |
| Two kernels at once | Streams, priorities | SMs, L2 and DRAM shared by two kernels | A decode GEMM beside a 1024^3 GEMM: 180.1 µs serial, 192.4 on two streams (worse) | Keep serial |
| CPU memory traffic during GPU phases | Scheduler | CPU and GPU share the LPDDR5X | GPU DRAM 237 GB/s alone, 214 / 171 / 138 with 2 / 8 / 20 CPU threads copying | An environment input of the simulator |
| mma input format | A precision choice made with the model's owner | Tensor work per instruction | fp8, int8 and fp4 run 2x the fp16 rate; tf32 half (mma_types.cu) | Measured; not a free knob |
| L2 promotion size of TMA descriptors | Would need our own descriptors | TileLang hardcodes 128-byte promotion | Survey | Not tested |

## 9. Legality rules

A column must pass all of these. The first eight are `legal()` in `agentic/decide/pipeline.py`. The prefetch rule
is applied where the variants are added, and the layout rule comes from `agentic/tilelang/tools.py`.

| Rule | Why |
|---|---|
| block_M ≤ max(16, M rounded up to a power of two); block_N ≤ N; block_K ≤ K | A bigger tile only computes padding. At M = 32, block_M can be 16 or 32. |
| block_M × block_N / threads ≥ 16 | At least 16 accumulators per thread. One mma output is 4 values per thread, so this asks for at least 4 mma tiles per warp. The code's own message is only "tile too small for the threads" (`tools.py`). |
| stages × (block_M + block_N) × block_K × 2, or block_M × block_N × 2 for a staged C if larger, plus 1,024 ≤ 101,376 bytes | 101,376 bytes (99 KB) is the code's cap per block; 1 KB is reserved per block (gb10.json). The SM has 100 KB in all. |
| TMA path: block_M × block_N / threads + 40 ≤ the register budget | TileLang compiles with `__launch_bounds__(threads + 128, 1)`, so ptxas may use 65,536 / (threads + 128) registers, rounded down to 8 and capped at 255. That is 168 at 256 threads. At 128 threads the formula gives 255, while compiled kernels use 240 (the consumers' setmaxnreg); the rule passes the same tiles either way. |
| cp.async path: block_M × block_N / threads + 70 ≤ 255 | Compiled cp.async kernels use about the accumulators plus 70 (64×128 at 256 threads: 96-106; 128×128 at 128: 202; 256×128 at 256: 198; 64×64 at 256: 74, from the `regs_est` docstring). |
| split-K only with cp.async and the direct store | `atomic_add` adds straight from registers, so a staged store means nothing there. Why TMA is excluded is not stated in the code. |
| split-K: (K / split) is a multiple of block_K | Each part must be whole k-tiles. Otherwise a part's last k-tile reads into the next part and that slice is added twice (`tools.py`, "exact-once"). |
| TMA path: block_M ≥ 64 unless M < 64 | Stated in the code without a reason |
| prefetch < num_stages | The manual schedule is prefetch + 1 deep and needs that many buffers (`tools.py`). The search only adds prefetch 1, and stages start at 2. |
| No linear shared-memory layout with TMA | Wrong answers: TMA writes the tile in its own swizzled order, and a row-major read sees it wrong (31 of 31, `tools.py`). The template does not offer a linear layout. |

The register rules exist because of spills. Over the budget, ptxas moves accumulators to local memory, and every
k-tile stores and reloads them. Example: the cover set's TMA 256×128×32 at 128 threads holds 256 accumulators per
thread. It measured 116.5 µs at 1024^3, against 46.8 µs for 128×128×32.

## 10. How a configuration is chosen

### 10.1 The steps

`agentic/decide/study/run.py` does this per shape and per tier, all on the Spark. Run 3 took 67 to 136 s per shape
(`wall_s` in each JSON file).

| Step | What happens | Why it is there |
|---|---|---|
| 1. Columns | Every combination of the tier's domains, kept if legal. Run 3: 122 to 278 columns for T1, 488 to 1,356 for T1+T2, 854 to 2,556 for T1+T2+T3. | Pricing is cheap, so try everything legal. |
| 2. Price | `gemm_us()` in `price.py` for each column | Rank before compiling anything. A build takes about 1.8 s (`build_worker` docstring). |
| 3. MILP top 60 | Pick the 60 best columns | The general form handles several kernels with a shared budget later (10.3). Swizzle doubled the columns, so the study takes 60 (run 1 took 30). |
| 4. Unpriced variants | For each cp.async column without split-K, add (prefetch 1), (FullCol), (both) | The model gives these one price, yet they change the time, and only on the cp.async path. |
| 5. Compile and check | Render, compile with TileLang (10 processes), compile the CUDA to a GPU binary for its SASS and register count, run once, compare with an fp32 torch product; pass if the relative error is under 1e-2 | Catches build failures and wrong answers before timing. Run 3: 1,360 of 1,360 passed (errors near 2e-4). |
| 6. Simulate | The SASS-driven simulator (`agentic/gb10/sim/`) times each kernel. Skipped at 4096^3 and 8192^3: the driver says one run takes minutes there, but `runs/decide/study_v3/README.md` reports 13 s for a 4096^3 kernel. | It reads the compiled code, so it sees what the model's formulas cannot. |
| 7. Finalists | Per tier: the model's 5 best distinct kernels with all their prefetch and warp-policy variants (8 where nothing is simulated), the simulator's top 5, and all the smaller tiers' finalists | Each ranker misses winners the other finds (10.5). Keeping the smaller tiers' finalists means a bigger tier is never slower, so the table in section 11 can only go down from left to right. |
| 8. Time | Every finalist, TileLang's default and autotuned kernels, and cuBLASLt, with one protocol in one session (10.4) | Only the GPU is the truth, and only same-session numbers compare fairly. |
| 9. Tier 4 | The overall best kernel inside a CUDA graph (for decode, with the rotating weights) | Launch-level choices do not change the kernel, so they go last. |

### 10.2 The price model

`agentic/gb10/model.py` builds the time from hardware units, each at a rate from gb10.json. Nothing is fitted to
kernel times. It aims to rank kernels and name the bottleneck, not to be exact.

1. Occupancy R: blocks per SM from registers, shared memory, threads and the 24-block cap.
2. Per k-tile and per SM: tensor time; shared-memory time (ldmatrix reads and tile writes at 128 bytes per clock);
   copy time on the chosen path; and latency divided by (stages − 1). The TMA path takes the largest and adds the
   hand-off. The cp.async path combines them by the overlap rule in 6.1.
3. Waves: each wave takes the longer of its SM time and its own DRAM bytes at the DRAM rate.
4. DRAM bytes: blocks run in launch order, one wave at a time. Each block touches its A row panel and its B column
   panel. Panels still in a 16.8 MB LRU cache are hits. This is where the launch-order swizzle enters.

`agentic/decide/price.py` then swaps in the chosen C store's exposed cost, adds split-K's atomics and DRAM traffic,
and adds the launch cost.

### 10.3 The MILP

`agentic/decide/mip.py` is a mixed-integer linear program solved with SciPy's `milp` (the HiGHS solver).
- One 0/1 variable per column. Each kernel picks exactly one column.
- The objective is the sum over kernels of call count × price.
- Optional: separable knobs with additive costs, and resource caps summed across kernels (such as bytes in the
  18 MiB persisting L2: a knapsack, filling a fixed budget with what is worth the most).
- `top_k` solves, then adds a cut that forbids the last answer, and solves again. So answer 2 is the best one
  different from answer 1, and so on.
- `tests/test_decide.py` checks the optimum and the top-k against brute force.

For one kernel with no caps, as in the study, the top 60 are simply the 60 cheapest columns. The MILP form is there
for the multi-kernel case.

### 10.4 The timing protocol

The same for every kernel in the study (`seq()`, `make_rot()` and `measure()` in `run.py`):
- 5 warm-up calls.
- Warm: n calls back to back between two events; time per call. n = 50 when M·N·K ≤ 2^33, else 10.
- Rot, decode shapes only: the same, but each call reads the next of several copies of B (8 copies of the 33.6 MB
  weight, 268 MB in all), so no call finds its weights in the L2. Nothing is flushed and nothing is subtracted.
- Cold: a 64 MB read before every call; the same sequence with the reads alone is timed and subtracted. Kept for
  reference only (12.2).
- Each number is the median of 3 sequences.
- Every kernel, ours and the baselines, gets random inputs.

Compute shapes are ranked warm. Decode shapes are ranked by rot, since a decode step reads each layer's weights from
DRAM.

The baselines use the same protocol:
- TileLang default: TileLang's `examples/gemm/example_gemm.py` as written: 128×128×32, 3 stages, 128 threads, TMA,
  direct store, no swizzle.
- TileLang autotuned: TileLang's AutoTuner over the grid in `agentic/decide/study/tilelang_autotune.py`: block_M and
  block_N 64/128/256, block_K 32/64, stages 0-3, threads 128/256, swizzle 10 on or off, staged store. That is
  3 × 3 × 2 × 4 × 2 × 2 = 288 configurations. The AutoTuner times warm with TileLang's `do_bench`. The study re-times
  its pick with the shared protocol.
- cuBLASLt (`agentic/decide/study/cublaslt_seq.cu`): the fastest of up to 8 heuristic algorithms by median single
  call, then timed as sequences of 100 calls, warm, rot and cold, from a stream and as a CUDA graph. Until run 3 its
  inputs were all zeros and its flush used a different load instruction (`__ldcg`) than ours; run 3 gives it random
  inputs, the same rotating copies and the same plain-load flush.

### 10.5 Why both rankers are needed

What the model alone would have picked, against what won (run 3 JSON files; µs, warm for compute shapes, rot for
decode):

| Shape | Model's top pick, measured | Winner, measured | How the winner got in |
|---|---:|---:|---|
| 1024^3 | 30.7 | 29.0 | The simulator's top 5. The model priced 144 of 236 candidates lower. |
| 2048^3 | 197.3 | 186.6 | T1's finalists (TMA only), carried forward. The model priced 64 of 172 lower, the simulator 70. |
| 4096^3 | 1,542.9 | 1,510.9 | The model's top 5: its second-ranked kernel (the direct store), in the FullCol variant |
| 8192^3 | 12,053.6 | 11,271.9 | T1's finalists, carried forward. The model priced 26 of 188 lower. |
| 4096×4096×1024 | 386.1 | 380.4 | The model's top 5: its top tile with row-major launch, in the FullCol variant |
| 32×4096×4096 | 141.1 | 137.1 | The simulator's top pick (also among T1's finalists). The model priced 168 of 204 lower. |
| 128×4096×4096 | 156.9 | 153.2 | T1's finalists, carried forward. The model priced 52 of 172 lower, the simulator 83. |

With differences under 3 % counted as ties, the model's top pick alone would have lost at 1024^3 (6 %), 2048^3
(6 %) and 8192^3 (7 %). At the other four shapes it lands within 3 % of the winner.

### 10.6 Lessons from the runs

1. **Swizzle was priced at 0 in run 1.** The driver passed swizzle 0 to the pricer for every column. It then added a
   swizzle-10 copy of the top 10 columns at the same price, with a comment saying the model cannot price swizzle.
   The model can. At 8192^3, for 256×128×32 with 4 stages, it gives 21.10 ms at sw0 and 11.48 ms at sw10; measured
   19.5 and 12.7 ms. So run 1 ranked the big shapes on swizzle-0 prices that all sat near 21 ms. Its 8192^3 result
   was 12,674.1 µs. Run 2 makes swizzle {0, 10} a priced column decision (commit 41bbf7e) and finds 11,448.9 µs, 10 %
   faster, while cuBLASLt moved 3 % the other way (`runs/decide/study/README.md`, `runs/decide/study_v2/README.md`).
2. **A register bug hid the best kernel** in the first pipeline runs. The pipeline passed the cp.async register cap
   (255) as the register count, so the model thought one block fits per SM. The kernel that measured 30.5 µs at
   1024^3 was priced at 50 µs and ranked 299th. With the estimate "accumulators + 70" it ranked 26th and was found
   (`notes/decision_variables.md`).
3. **Split-K's price missed DRAM traffic.** At 8192^3 the fp32 C is 256 MiB, much more than the L2, so every split's
   adds read and write C in DRAM. Split 8 was priced 15.0 ms and measured 31 to 46 ms. With the DRAM term it prices
   32.9 ms (commit 41bbf7e).
4. **Tiers must nest.** In run 1 each tier timed only its own picks, so T1+T2 came out slower than T1 at 2048^3,
   M = 32 and M = 128. Run 2 carries the smaller tiers' finalists forward.
5. **Time decode the way decode runs.** Runs 1 and 2 flushed the L2 with a 64 MB read and subtracted the flush. That
   did not make B cold, and it inflated both decode wins: 1.19x and 1.28x in run 2, 1.07x and 0.99x in run 3 (12.2).
   Run 3 rotates through 8 copies of B instead.
6. **Give the baseline the same inputs.** cuBLASLt on all-zero inputs ran 9 % faster at 2048^3 than on random ones
   (11.1). Our kernels always had random inputs; run 3 gives cuBLASLt random inputs too.

## 11. Results

### 11.1 The run-3 table

Printed by `python3 -m agentic.decide.study.report runs/decide/study_v3`. "MIP Tn" is the best kernel the solver
found with the knobs of tiers 1 to n. Microseconds; warm for compute shapes, rot for decode shapes. Each row was
timed in one session. Differences under about 3 % are ties: between runs 2 and 3 the same kernels moved 1 to 3 %
(4096^3: 1,469.0 to 1,510.9; 8192^3: 11,448.9 to 11,271.9).

| Shape | Timing | TileLang default | TileLang autotuned | cuBLASLt | MIP T1 | MIP T1+T2 | MIP T1+T2+T3 | MIP T1-T4 | cuBLASLt with a graph | T1-T3 speedup over the better of TileLang autotuned and cuBLASLt | T1-T4 speedup over cuBLASLt with a graph |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024x1024x1024 | warm | 45.3 | 33.5 | 32.8 | 32.9 | 29.0 | 29.0 | 27.7 | 30.6 | 1.13x | 1.10x |
| 2048x2048x2048 | warm | 241.2 | 187.6 | 194.6 | 186.6 | 186.6 | 186.6 | 186.6 | 193.0 | 1.01x | 1.03x |
| 4096x4096x4096 | warm | 3,629.6 | 1,548.0 | 1,556.3 | 1,573.1 | 1,510.9 | 1,510.9 | 1,510.9 | 1,544.3 | 1.02x | 1.02x |
| 8192x8192x8192 | warm | 36,995.2 | 11,266.1 | 12,136.9 | 11,271.9 | 11,271.9 | 11,271.9 | 11,271.9 | 12,173.1 | 1.00x | 1.08x |
| 4096x4096x1024 | warm | 436.8 | 410.2 | 398.9 | 386.3 | 380.4 | 380.4 | 380.4 | 407.6 | 1.05x | 1.07x |
| 32x4096x4096 | rot | 147.9 | 146.6 | 149.3 | 137.1 | 137.1 | 137.1 | 137.1 | 148.4 | 1.07x | 1.08x |
| 128x4096x4096 | rot | 161.9 | 152.3 | 169.8 | 153.2 | 153.2 | 153.2 | 152.4 | 164.8 | 0.99x | 1.08x |

In short, against the better of TileLang autotuned and cuBLASLt:
- the solver wins at 1024^3 (1.13x), 4096×4096×1024 (1.05x) and M = 32 (1.07x);
- it ties at 2048^3, 4096^3, 8192^3 and M = 128 (0.99x to 1.02x);
- it loses nowhere.

Against cuBLASLt alone, the solver's kernel is faster on every shape, from 1.03x at 4096^3 (a tie) to 1.13x at
1024^3. TileLang's AutoTuner shares much of that lead. At 8192^3 it picked the same kernel, and at M = 128 its pick
ties ours.

What the fairer inputs and timing changed from run 2:
- **cuBLASLt at 2048^3** went from 178.4 µs (all-zero inputs, run 2) to 194.6 (random inputs, run 3): 9 % slower.
  Our best (189.1 to 186.6) and TileLang autotuned (190.6 to 187.6) moved 1 to 2 %. So run 2's 2048^3 "loss" came
  from cuBLASLt's zero inputs. The comment in `cublaslt_seq.cu` gives a likely reason: all-zero inputs can change the
  tensor cores' power and clock.
- **The other compute shapes:** cuBLASLt moved 0 to 3 %.
- **Decode:** run 2's cold times, 118.0 and 124.6 µs, were too fast to be real (12.2). With rotating weights the best
  kernels take 137.1 µs at M = 32 and 153.2 (stream) or 152.4 (graph) at M = 128. That is A and B (33.8 MB and
  34.6 MB) read at 247 and 226 GB/s, around the 237-242 GB/s the DRAM probes reached (gb10.json: mlp_stream.cu,
  dram.cu). Against the better baseline, decode went from 1.19x and 1.28x in run 2 to 1.07x and 0.99x. Against
  cuBLASLt alone it is now 1.09x and 1.11x.

### 11.2 What each tier bought

Run 3; µs.

| Shape | T1 against TileLang autotuned | T2 over T1 | T3 | T4 (CUDA graph) |
|---|---|---|---|---|
| 1024^3 | 32.9 vs 33.5 (tie) | 32.9 to 29.0 (−12 %) | nothing | 29.0 to 27.7 (−4 %) |
| 2048^3 | 186.6 vs 187.6 (tie) | nothing | nothing | nothing (190.1) |
| 4096^3 | 1,573.1 vs 1,548.0 (tie) | 1,573.1 to 1,510.9 (−4 %) | nothing | nothing (1,540.7) |
| 8192^3 | 11,271.9 vs 11,266.1 (the same kernel) | nothing | nothing | nothing (11,329.6) |
| 4096×4096×1024 | 386.3 vs 410.2 (−6 %) | 386.3 to 380.4 (−1.5 %, a tie) | nothing | nothing (423.4) |
| 32×4096×4096 | 137.1 vs 146.6 (−6 %) | nothing | nothing | nothing (142.2) |
| 128×4096×4096 | 153.2 vs 152.3 (tie) | nothing | nothing | 153.2 to 152.4 (a tie) |

- **T1** ties TileLang's AutoTuner on five shapes and beats it by 6 % at 4096×4096×1024 and M = 32. At 8192^3 both
  chose the same kernel; its two timings, 11,271.9 and 11,266.1, show the noise. At M = 32 the winner (32×128×64,
  4 stages) lies outside the AutoTuner's grid, which has no block_M below 64 and no stage count above 3. At
  4096×4096×1024 the T1 winner (256×128×64, 2 stages, row-major) is in the grid, but the AutoTuner, timing its own
  way, picked 256×128×32 with swizzle 10 (410.2 µs here). T1 found its kernels by pricing and timing 8 to 10
  finalists per shape.
- **T2 pays where compute dominates**, through the cp.async path and FullCol warps: 12 % at 1024^3 and 4 % at
  4096^3. At 4096×4096×1024 its 1.5 % is a tie. TileLang's AutoTuner cannot reach these kernels.
- **T3 bought nothing.** In runs 2 and 3 no split-K column reached the MILP's top 60 for any shape.
- **T4:** a CUDA graph saved 1.3 µs at 1024^3. Elsewhere it tied or was slower (8.1).

### 11.3 The winners in plain words

| Shape | Winning kernel (T1-T4) | In plain words |
|---|---|---|
| 1024^3 | `cpasync_64x128x32_st2_th256_sw0_pf0_wp2_vw0_inf_staged_sk1`, graph | cp.async; 64×128 tiles, 32-wide k-tiles; 2 stages; 256 threads that both load and compute; row-major launch; warps side by side along N; C staged through shared memory and sent by TMA; launched in a CUDA graph. 128 blocks. With prefetch 1 the same kernel took 29.1 against 29.0. |
| 2048^3 | `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`, stream | TMA with a 128-thread producer group; 128×256 tiles, 64-wide k-tiles; 2 stages; 256 compute threads; launch in panels of 10 rows; staged store. 4 % ahead of cuBLASLt. |
| 4096^3 | `cpasync_128x128x32_st3_th128_sw10_pf0_wp2_vw0_inf_direct_sk1`, stream | cp.async; 128×128×32; 3 stages; 128 threads; panels of 10; FullCol warps; direct store. Its staged twin took 1,527.9 against 1,510.9, a tie. |
| 8192^3 | `tma_ws_128x256x64_st2_th256_sw10_pf0_wp0_vw0_inf_staged_sk1`, stream | The 2048^3 kernel. TileLang's AutoTuner picked it too. 97.5 TFLOP/s (2 × 8192³ / 11,271.9 µs), 79 % of the 124 TFLOP/s peak. |
| 4096×4096×1024 | `cpasync_128x128x32_st3_th128_sw0_pf0_wp2_vw0_inf_staged_sk1`, stream | The 4096^3 kernel with row-major launch and the staged store. With swizzle 10 it took 382.8, a tie. |
| 32×4096×4096 | `tma_ws_32x128x64_st4_th128_sw0_pf0_wp0_vw0_inf_staged_sk1`, stream | TMA with a producer group; 32×128×64 tiles, so one tile row covers all of M; 4 stages; 128 compute threads; row-major; staged. 32 blocks for 48 SMs. Reads A and B at 247 GB/s. The next seven kernels are within 0.6 %. |
| 128×4096×4096 | `tma_ws_128x128x32_st6_th256_sw0_pf0_wp0_vw0_inf_staged_sk1`, graph | TMA with a producer group; 128×128×32; 6 stages (98,304 bytes of shared memory, near the 99 KB cap); 256 compute threads; row-major; staged; in a graph. 32 blocks for 48 SMs. Reads A and B at 226 GB/s. TileLang autotuned ties it (152.3). |

## 12. Open items and limits

### 12.1 Open items

1. **Decode sits at the DRAM rate.** At M = 32 the eight fastest kernels lie within 137.1 to 137.9 µs. Choosing tiles
   cannot do much more there. What could: reading less, or hiding the weight reads under other work. Candidates are
   L2 prefetch of the next layer's weights, programmatic dependent launch, and one CUDA graph over a whole decode step
   (8.3, 6.2).
2. **Tier 2 controls not in the template** (6.2). The likely next ones: cluster multicast for L2-bound shapes,
   `eviction_policy` and `annotate_l2_hit_ratio` for decode, the setmaxnreg split, and persistent kernels.
3. **Prefetch and warp policy are unpriced.** The study tries their variants blindly, and only on the cp.async path.
   The one code check says prefetch 1 only sets the depth to 2, which num_stages already does. If that holds in
   general, prefetch can go. The warp policy needs a real price.
4. **Split-K has never won.** Split-K with a TMA reduce (`T.atomic_add(use_tma=True)`) and stream-K, which does not
   compile at the pinned commit, are untried.
5. **L2 pinning is untested in the study** (8.2). With rotating weights it needs its own experiment: which layers'
   weights to keep in the 18 MiB.
6. **The MILP across kernels** (a shared persisting-L2 budget for a decode step) is built and tested against brute
   force, but no study has used it.
7. **Tier 4 beyond graphs** (8.3) is untested, except two streams, which lost.
8. **The DRAM ceiling.** The M = 32 winner reads at 247 GB/s, above the 237-242 GB/s the probes reached. So the
   probes do not find DRAM's real limit, and the model's DRAM rate may be low.
9. **Simulate the big shapes.** The study skips the simulator at 4096^3 and 8192^3, though a 4096^3 kernel simulates
   in 13 s (run-3 README). At 8192^3 the model alone misranked the path (12.2).

### 12.2 Limits of the numbers

1. **Decode timing was wrong in runs 1 and 2; run 3 fixed it.** Those runs timed decode "cold": a 64 MB read before
   each call, its time subtracted. B was not cold. The M = 32 winner came out at 118.0 µs, which means reading B
   (33.6 MB) at 284 GB/s, above the 273 GB/s spec. Part of B likely stayed in the L2, which keeps re-read data
   through a 64 MB stream (gb10.json, l2policy.cu). The method was noisy too: one M = 128 kernel measured 124.6 and
   191.4 µs with and without a swizzle that changes nothing with one row of tiles. Run 3 rotates through 8 copies of
   B (268 MB), so every call reads its weights from DRAM. The best decode kernels now read at 247 and 226 GB/s, and
   run 2's M = 128 winner measures 161.9 µs. Cold stays in the JSON for reference only. It was never used to rank
   compute shapes, where it also gave odd values (run 2, 1024^3: 19 to 80 µs cold against 29 to 49 warm).
2. **cuBLASLt's inputs were unfair in runs 1 and 2; run 3 fixed it.** It ran on all-zero inputs, with its own flush
   load (`__ldcg`). Zero inputs made it 9 % faster at 2048^3 (11.1). Run 3 gives it random inputs, the same rotating
   weights and the same flush.
3. **One session per row.** Comparisons inside a row are fair. Across sessions the same kernel moves 1 to 3 %
   (4096^3: 1,469.0 in run 2, 1,510.9 in run 3). So differences under about 3 % are ties.
4. **The simulator was skipped at 4096^3 and 8192^3.** There the finalists come from the model alone (12.1, item 9).
   At 8192^3 the model ranked the path wrong (cp.async priced 10,273.0 µs, measured 12,053.6; the TMA winner priced 11,918.3,
   measured 11,271.9). The TMA kernel survived only because T1's finalists carry forward.
5. **The model still charges a withdrawn constant.** `model.py` uses a fixed 0.22 µs per k-tile on the cp.async path.
   gb10.json marks that value as withdrawn: it came from the probe's own address arithmetic, and a later probe
   (loopsteps.cu) shows no fixed cost per k-tile.
6. **Some knobs were never searched.** vec_width and smem_layout make no code change. FullRow, prefetch above 1,
   swizzle panels other than 10, and stage counts 0 and 1 were left out.

## Appendix: where the code lives

| What | Where |
|---|---|
| The template | `agentic/tilelang/families/gemm/kernel.py.j2` |
| Render, defaults, names | `agentic/decide/template.py` |
| Domains, legality, register estimate, single-shape pipeline | `agentic/decide/pipeline.py` |
| Price terms (C store, split-K, launch, L2 pin) | `agentic/decide/price.py` |
| Analytical model, launch order, L2 simulation | `agentic/gb10/model.py` |
| Constants with their probes | `calibration/spark-6951/gb10.json` |
| MILP and its test | `agentic/decide/mip.py`, `tests/test_decide.py` |
| Cover set and code fingerprints | `agentic/decide/cover.py`, `runs/decide/cover_code_fingerprints.json` |
| Tier study driver, baselines, result tables | `agentic/decide/study/run.py`, `tilelang_autotune.py`, `cublaslt_seq.cu`, `report.py` |
| Tier 3 and 4 registry | `agentic/decide/registry_tier34.yaml` |
| TileLang controls survey | `runs/decide/tilelang_controls_survey.md` |
| Results of runs 1, 2 and 3 (one JSON per shape: assignment, price, simulator time, measured times) | `runs/decide/study/`, `runs/decide/study_v2/`, `runs/decide/study_v3/` (run 3, current) |

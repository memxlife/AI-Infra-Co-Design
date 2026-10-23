# The ISA boundary of consumer Blackwell (B41)

*Oct 23 2026. Machines: the DGX Spark (GB10, `sm_121a`, 48 SMs) and one RTX 5090 (`sm_120a`, GPU 0 of the 8-GPU
server). Plan: B41, "Block-scaled MMA and the datacenter ISA boundary". Code: `calibration/suite/bench/b41_*/`. Raw
records: `calibration/spark-6951/b41/`, `calibration/rtx5090/b41/`.*

The datacenter Blackwell (B200, `sm_100a`) keeps its tensor accumulators and scale factors in tensor memory (TMEM) and
drives the tensor cores with `tcgen05`. The consumer Blackwell chips, the GB10 and the RTX 5090, have no TMEM: the
accumulators stay in registers, and block scaling is added to the warp-level `mma.sync`. This note records which
instructions exist on each side and what they cost, item by item.

## Machines and toolkits

| | DGX Spark | RTX 5090 server |
|---|---|---|
| GPU | NVIDIA GB10, `sm_121` | NVIDIA GeForce RTX 5090, `sm_120` (GPU 0 of 8) |
| Driver | 580.159.03 (CUDA 13.0) | 570.148.08 (CUDA 12.8) |
| Toolkit | CUDA 13.0, nvcc V13.0.88, `/usr/local/cuda-13.0` | CUDA 12.8, nvcc V12.8.93, `/usr/local/cuda-12.8` (not on the default PATH) |
| ptxas targets | `sm_100(a,f)`, `sm_103(a,f)`, `sm_110(a,f)`, `sm_120(a,f)`, `sm_121(a,f)` | `sm_100(a)`, `sm_101(a)`, `sm_120(a)`: no family (`f`) targets, no `sm_103`, no `sm_121` |
| PyTorch | 2.11.0 (CUDA 12.8) | not installed |

Full records: `calibration/spark-6951/b41/env_spark.txt`, `calibration/rtx5090/b41/env_5090.txt`. The 5090 server's
root disk was 93 % full at the start (120 GB free); all work stays under `~/b41/`.

## Item 2: which instructions assemble and run

Each instruction is one probe kernel that runs it once with valid operands and stores a result
(`b41_isa/gen_probes.py`). The probe's PTX is assembled for each target by rewriting only its `.target` line
(`b41_isa/assemble.py`). PTX ISA 9.0, ptxas from CUDA 13.0 (`cuda_13.0.r13.0/compiler.36424714_0`) for all six
columns. Each cubin that assembles runs in its own process (`b41_isa/run_isa.cu`).

Where each column ran:
- `sm_121a`, `sm_121f`: the GB10.
- `sm_120a`: the RTX 5090, from cubins built with its own CUDA 12.8 (PTX ISA 8.7). The 5090's driver cannot load CUDA
  13.0 cubins ("device kernel image is invalid"). The two toolkits agree on every `sm_120a` and `sm_100a` verdict.
- `sm_120f`: assembled with CUDA 13.0 and run on the GB10. CUDA 12.8 has no `f` targets, so it did not run on the 5090.
- `sm_100a`, `sm_103a`: assembled only. There is no datacenter GPU here.

| Plan row | Instruction | sm_120a | sm_120f | sm_121a | sm_121f | sm_100a | sm_103a |
|---|---|---|---|---|---|---|---|
| 1 | `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3` (plan's text) | rejected: Illegal modifier '.scale_vec::2X' for instruction 'mma' with type '.ue4m3' | rejected: Illegal modifier '.scale_vec::2X' for instruction 'mma' with type '.ue4m3' | rejected: Illegal modifier '.scale_vec::2X' for instruction 'mma' with type '.ue4m3' | rejected: Illegal modifier '.scale_vec::2X' for instruction 'mma' with type '.ue4m3' | rejected: 'mma with block scale' not on sm_100a | rejected: 'mma with block scale' not on sm_103a |
| 1 | same, kind before shape (CUTLASS order) | rejected: Illegal modifier '.scale_vec::2X' for instruction 'mma' with type '.ue4m3' | rejected: Illegal modifier '.scale_vec::2X' for instruction 'mma' with type '.ue4m3' | rejected: Illegal modifier '.scale_vec::2X' for instruction 'mma' with type '.ue4m3' | rejected: Illegal modifier '.scale_vec::2X' for instruction 'mma' with type '.ue4m3' | rejected: 'mma with block scale' not on sm_100a | rejected: 'mma with block scale' not on sm_103a |
| 1 | `...kind::mxf4nvf4.block_scale.scale_vec::4X...ue4m3` (NVFP4 as CUTLASS writes it) | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | rejected: 'mma with block scale' not on sm_100a | rejected: 'mma with block scale' not on sm_103a |
| 1 | `...kind::mxf4nvf4.block_scale.scale_vec::2X...ue8m0` | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | rejected: 'mma with block scale' not on sm_100a | rejected: 'mma with block scale' not on sm_103a |
| 2 | `mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X...ue8m0` (plan's text) | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | rejected: 'mma with block scale' not on sm_100a | rejected: 'mma with block scale' not on sm_103a |
| 2 | same, CUTLASS order | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | rejected: 'mma with block scale' not on sm_100a | rejected: 'mma with block scale' not on sm_103a |
| 3 | `mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0` (plan's text) | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | rejected: 'mma with block scale' not on sm_100a | rejected: 'mma with block scale' not on sm_103a |
| 3 | same, CUTLASS order | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | rejected: 'mma with block scale' not on sm_100a | rejected: 'mma with block scale' not on sm_103a |
| 4 | `mma.sync...kind::mxf4nvf4.sp::ordered_metadata.block_scale.scale_vec::2X.m16n8k128...ue8m0` | assembles+runs | rejected: 'Sparse mma with block scale' not on sm_120f | assembles+runs | rejected: 'Sparse mma with block scale' not on sm_121f | rejected: 'mma with block scale' not on sm_100a | rejected: 'mma with block scale' not on sm_103a |
| 4 | same, `scale_vec::4X ... ue4m3` | assembles+runs | rejected: 'Sparse mma with block scale' not on sm_120f | assembles+runs | rejected: 'Sparse mma with block scale' not on sm_121f | rejected: 'mma with block scale' not on sm_100a | rejected: 'mma with block scale' not on sm_103a |
| 4 | `...kind::mxf4.sp::ordered_metadata.block_scale.scale_vec::2X.m16n8k128...ue8m0` | assembles+runs | rejected: 'Sparse mma with block scale' not on sm_120f | assembles+runs | rejected: 'Sparse mma with block scale' not on sm_121f | rejected: 'mma with block scale' not on sm_100a | rejected: 'mma with block scale' not on sm_103a |
| 4 | `...kind::mxf8f6f4.sp::ordered_metadata.block_scale.scale_vec::1X.m16n8k64...e4m3.e4m3...ue8m0` | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | rejected: 'mma with block scale' not on sm_100a | rejected: 'mma with block scale' not on sm_103a |
| 5 | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | assembles (no such GPU here) | assembles (no such GPU here) |
| 6 | `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32` | rejected: 'tcgen05.alloc' not on sm_120a | rejected: 'tcgen05.alloc' not on sm_120f | rejected: 'tcgen05.alloc' not on sm_121a | rejected: 'tcgen05.alloc' not on sm_121f | assembles (no such GPU here) | assembles (no such GPU here) |
| 7 | `tcgen05.mma.cta_group::1.kind::f16` | rejected: 'tcgen05.mma' not on sm_120a | rejected: 'tcgen05.mma' not on sm_120f | rejected: 'tcgen05.mma' not on sm_121a | rejected: 'tcgen05.mma' not on sm_121f | assembles (no such GPU here) | assembles (no such GPU here) |
| 8 | `tcgen05.cp.cta_group::1.128x256b` | rejected: '.128x256b' not on sm_120a | rejected: '.128x256b' not on sm_120f | rejected: '.128x256b' not on sm_121a | rejected: '.128x256b' not on sm_121f | assembles (no such GPU here) | assembles (no such GPU here) |
| 9 | `tcgen05.ld.sync.aligned.16x256b.x1.b32` | rejected: '.16x256b' not on sm_120a | rejected: '.16x256b' not on sm_120f | rejected: '.16x256b' not on sm_121a | rejected: '.16x256b' not on sm_121f | assembles (no such GPU here) | assembles (no such GPU here) |
| 10 | `tcgen05.ld.red.sync.aligned.32x32b.x2.min.f32` | rejected: '.32x32b' not on sm_120a | rejected: '.32x32b' not on sm_120f | rejected: '.32x32b' not on sm_121a | rejected: '.32x32b' not on sm_121f | rejected: 'tcgen05.ld.red' not on sm_100a | assembles (no such GPU here) |
| 11 | `wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16` (with fence, commit, wait) | rejected: 'wgmma.fence' not on sm_120a | rejected: 'wgmma.fence' not on sm_120f | rejected: 'wgmma.fence' not on sm_121a | rejected: 'wgmma.fence' not on sm_121f | rejected: 'wgmma.fence' not on sm_100a | rejected: 'wgmma.fence' not on sm_103a |
| 12 | `setmaxnreg.inc.sync.aligned.u32 232` alone | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | assembles (no such GPU here) | assembles (no such GPU here) |
| 12 | `setmaxnreg.dec 24` in the producer warp group, `.inc 232` in two consumer groups (384 threads) | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | assembles (no such GPU here) | assembles (no such GPU here) |
| 13 | `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster` (2-block cluster) | assembles+runs (a software call, no hardware multicast) | assembles+runs on the GB10 | assembles+runs (a software call, no hardware multicast) | assembles+runs (a software call, no hardware multicast) | assembles (no such GPU here) | assembles (no such GPU here) |
| 14 | `cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes` | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | assembles (no such GPU here) | assembles (no such GPU here) |
| 15 | `redux.sync.max.s32` | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | assembles (no such GPU here) | assembles (no such GPU here) |
| 15 | `redux.sync.add.s32` | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | assembles (no such GPU here) | assembles (no such GPU here) |
| 16 | `ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | assembles (no such GPU here) | assembles (no such GPU here) |
| 16 | `stmatrix.sync.aligned.m8n8.x4.trans.shared.b16` | assembles+runs | assembles+runs on the GB10 | assembles+runs | assembles+runs | assembles (no such GPU here) | assembles (no such GPU here) |

The machine code was read with `cuobjdump -sass` for every probe that runs, to confirm the instruction is really there.

**What it shows.**
1. **The plan's row 1 is not a legal instruction.** `scale_vec::2X` with `ue4m3` scales is rejected on every target
   ("Illegal modifier '.scale_vec::2X' for instruction 'mma' with type '.ue4m3'"). NVFP4 (16-element blocks, `ue4m3`
   scales) is `scale_vec::4X` at `m16n8k64`; `scale_vec::2X` goes with `ue8m0` (MXFP4). Both run on both chips.
2. **Modifier order does not matter.** The plan's order (shape first) and CUTLASS's (kind first) assemble the same.
3. **Block-scaled `mma.sync` is consumer only.** It assembles for `sm_120a/f` and `sm_121a/f` and runs on both chips;
   `sm_100a` and `sm_103a` reject it ("mma with block scale not supported"). SASS: `OMMA.SF.16864` for FP4
   (`.UE4M3.4X` for NVFP4, `.E8` for MXFP4) and `QMMA.SF.16832` for FP8.
4. **Sparse block-scaled `mma` is narrower than the plan expected.** The FP4 kinds (`mxf4nvf4`, `mxf4`) assemble only
   for the `a` targets; the FP8 kind (`mxf8f6f4`) also for the `f` targets. SASS: `OMMA.SF.SP.168128`,
   `QMMA.SF.SP.16864`.
5. **`tcgen05` is datacenter only, and `tcgen05.ld.red` is `sm_103a` only.** Every `tcgen05` instruction is rejected
   on the four consumer targets and assembles for `sm_100a` and `sm_103a`, except `tcgen05.ld.red`, which `sm_100a`
   rejects. `wgmma` is rejected on all six (it is Hopper's).
6. **`setmaxnreg` exists on consumer Blackwell; the plan expected datacenter only.** Alone, `setmaxnreg.inc 232`
   assembles everywhere but ptxas drops it: the SASS has a `NOP` in its place. In the warp-specialized pattern (one
   warp group `dec 24`, two `inc 232`, 384 threads) it becomes `USETMAXREG.DEALLOC.CTAPOOL 0x18` and
   `USETMAXREG.TRY_ALLOC.CTAPOOL 0xe8` and runs on both chips. (An older note said ptxas rejects it for `sm_121`: that
   was the plain target, not `sm_121a`.)
7. **TMA multicast is not hardware here.** `.multicast::cluster` assembles and runs, but ptxas turns it into a
   `CALL.ABS.NOINC` to a routine outside the kernel, on both chips; a plain TMA load is `UTMALDG.2D`. This fits the
   earlier finding that multicast was far slower on the Spark; item 7 times it. With CUDA 12.8 the plain-TMA probe
   also contains such a call next to its `UTMALDG`; item 7 looks at that too.
8. **Portability between the two chips.** `sm_120f` cubins run on the GB10; `sm_120a` cubins do not load there ("no
   kernel image is available"), and neither do `sm_121a/f` cubins on the 5090.

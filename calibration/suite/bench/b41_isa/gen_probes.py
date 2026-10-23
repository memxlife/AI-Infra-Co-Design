"""B41 item 2: write one CUDA probe per PTX instruction (probes/<id>.cu). Each kernel runs the instruction once with
valid operands and stores a result, so it can be assembled for any target and run where it assembles.
Kernel: extern "C" probe(unsigned* out, const __grid_constant__ CUtensorMap tmap); 128 threads; the multicast probe
runs as a 2-block cluster. python3 gen_probes.py OUT_DIR"""
import sys
from pathlib import Path

HEAD = r'''#include <cuda.h>
#include <cstdint>
extern "C" __global__ void __launch_bounds__(128) probe(unsigned* out, const __grid_constant__ CUtensorMap tmap) {
  unsigned t = threadIdx.x;
'''
TAIL = "}\n"

# dense block-scaled mma: d{4 f32}, a{4 b32}, b{2 b32}, c{4 f32}, {scale-a}, {byte-id-a, thread-id-a}, {scale-b}, {byte-id-b, thread-id-b}
DENSE = r'''  unsigned a0 = t, a1 = t * 3, a2 = t * 5, a3 = t * 7, b0 = t * 11, b1 = t * 13, sa = SCALE, sb = SCALE;
  float c0 = 0, c1 = 0, c2 = 0, c3 = 0, d0, d1, d2, d3; uint16_t z = 0;
  asm volatile("INSTR {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13}, {%14}, {%15,%16}, {%17}, {%18,%19};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                 "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
  out[t] = __float_as_uint(d0 + d1 + d2 + d3);
'''
# sparse block-scaled mma: b has 4 registers, then metadata {e} and the selector 0x0 (CUTLASS mma_sm120_sparse.hpp)
SPARSE = r'''  unsigned a0 = t, a1 = t * 3, a2 = t * 5, a3 = t * 7, b0 = t * 11, b1 = t * 13, b2 = t * 17, b3 = t * 19;
  unsigned e = 0x44444444u, sa = SCALE, sb = SCALE; float c0 = 0, c1 = 0, c2 = 0, c3 = 0, d0, d1, d2, d3; uint16_t z = 0;
  asm volatile("INSTR {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, {%16}, 0x0, {%17}, {%18,%19}, {%20}, {%21,%22};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
               : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(b2), "r"(b3), "f"(c0), "f"(c1), "f"(c2), "f"(c3),
                 "r"(e), "r"(sa), "h"(z), "h"(z), "r"(sb), "h"(z), "h"(z));
  out[t] = __float_as_uint(d0 + d1 + d2 + d3);
'''
UE8M0, UE4M3 = "0x7f7f7f7fu", "0x38383838u"   # scale 1.0 in each format
PROBES = {
    # 1-3: the plan's spelling (shape before kind), then CUTLASS's (kind before shape) and the valid scale pairings
    "01a_mxf4nvf4_2X_ue4m3_plan": DENSE.replace("INSTR", "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue4m3").replace("SCALE", UE4M3),
    "01b_mxf4nvf4_2X_ue4m3": DENSE.replace("INSTR", "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3").replace("SCALE", UE4M3),
    "01c_mxf4nvf4_4X_ue4m3": DENSE.replace("INSTR", "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3").replace("SCALE", UE4M3),
    "01d_mxf4nvf4_2X_ue8m0": DENSE.replace("INSTR", "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0").replace("SCALE", UE8M0),
    "02a_mxf4_2X_ue8m0_plan": DENSE.replace("INSTR", "mma.sync.aligned.m16n8k64.row.col.kind::mxf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0").replace("SCALE", UE8M0),
    "02b_mxf4_2X_ue8m0": DENSE.replace("INSTR", "mma.sync.aligned.kind::mxf4.block_scale.scale_vec::2X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue8m0").replace("SCALE", UE8M0),
    "03a_mxf8f6f4_1X_e4m3_plan": DENSE.replace("INSTR", "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0").replace("SCALE", UE8M0),
    "03b_mxf8f6f4_1X_e4m3": DENSE.replace("INSTR", "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0").replace("SCALE", UE8M0),
    "04a_sp_mxf4nvf4_2X_ue8m0": SPARSE.replace("INSTR", "mma.sync.aligned.kind::mxf4nvf4.sp::ordered_metadata.block_scale.scale_vec::2X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue8m0").replace("SCALE", UE8M0),
    "04b_sp_mxf4nvf4_4X_ue4m3": SPARSE.replace("INSTR", "mma.sync.aligned.kind::mxf4nvf4.sp::ordered_metadata.block_scale.scale_vec::4X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue4m3").replace("SCALE", UE4M3),
    "04c_sp_mxf4_2X_ue8m0": SPARSE.replace("INSTR", "mma.sync.aligned.kind::mxf4.sp::ordered_metadata.block_scale.scale_vec::2X.m16n8k128.row.col.f32.e2m1.e2m1.f32.ue8m0").replace("SCALE", UE8M0),
    "04d_sp_mxf8f6f4_1X_e4m3": SPARSE.replace("INSTR", "mma.sync.aligned.kind::mxf8f6f4.sp::ordered_metadata.block_scale.scale_vec::1X.m16n8k64.row.col.f32.e4m3.e4m3.f32.ue8m0").replace("SCALE", UE8M0),
    "05_bf16_m16n8k16": r'''  unsigned a0 = t, a1 = t * 3, a2 = t * 5, a3 = t * 7, b0 = t * 11, b1 = t * 13; float c0 = 0, c1 = 0, c2 = 0, c3 = 0, d0, d1, d2, d3;
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  out[t] = __float_as_uint(d0 + d1 + d2 + d3);
''',
    "06_tcgen05_alloc": r'''  __shared__ unsigned slot;
  if (t < 32) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], %1;" :: "r"((unsigned)__cvta_generic_to_shared(&slot)), "r"(32u));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  if (t < 32) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, %1;" :: "r"(slot), "r"(32u));
  out[t] = slot;
''',
    "07_tcgen05_mma_f16": r'''  unsigned taddr = 0, idesc = 0; unsigned long long ad = 0, bd = 0;
  if (t == 0) asm volatile("{ .reg .pred p; setp.ne.b32 p, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, p; }"
                           :: "r"(taddr), "l"(ad), "l"(bd), "r"(idesc), "r"(0u));
  out[t] = t;
''',
    "08_tcgen05_cp": r'''  unsigned taddr = 0; unsigned long long sd = 0;
  if (t < 32) asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" :: "r"(taddr), "l"(sd));
  out[t] = t;
''',
    "09_tcgen05_ld": r'''  unsigned taddr = 0, r0, r1, r2, r3;
  asm volatile("tcgen05.ld.sync.aligned.16x256b.x1.b32 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(taddr));
  out[t] = r0 + r1 + r2 + r3;
''',
    "10_tcgen05_ld_red": r'''  unsigned taddr = 0, r0, r1, m;
  asm volatile("tcgen05.ld.red.sync.aligned.32x32b.x2.min.f32 {%0,%1}, %2, [%3];" : "=r"(r0), "=r"(r1), "=r"(m) : "r"(taddr));
  out[t] = r0 + r1 + m;
''',
    "11_wgmma_m64n8k16_bf16": r'''  float d0 = 0, d1 = 0, d2 = 0, d3 = 0; unsigned long long ad = 0, bd = 0;
  asm volatile("{ .reg .pred p; setp.ne.b32 p, %6, 0; wgmma.fence.sync.aligned; "
               "wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, %4, %5, p, 1, 1, 0, 0; "
               "wgmma.commit_group.sync.aligned; wgmma.wait_group.sync.aligned 0; }"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "l"(ad), "l"(bd), "r"(0u));
  out[t] = __float_as_uint(d0 + d1 + d2 + d3);
''',
    "12_setmaxnreg_inc_232": r'''  asm volatile("setmaxnreg.inc.sync.aligned.u32 232;");
  out[t] = t;
''',
    # the warp-specialized pattern TileLang emits: 3 warp groups under __launch_bounds__(384, 1) (168 registers each);
    # the producer group gives registers back, the two consumer groups take 232. A lone setmaxnreg is dropped by ptxas.
    "12b_setmaxnreg_wg_dec24_inc232": r'''  unsigned wg = t / 128; float acc = t;
  if (wg == 0) asm volatile("setmaxnreg.dec.sync.aligned.u32 24;");
  else { asm volatile("setmaxnreg.inc.sync.aligned.u32 232;"); for (int i = 0; i < 64; i++) acc = acc * 1.0001f + i; }
  out[t] = __float_as_uint(acc);
''',
    "13_tma_2d_multicast": r'''  __shared__ __align__(1024) unsigned char buf[8192]; __shared__ __align__(8) unsigned long long bar;
  unsigned b = (unsigned)__cvta_generic_to_shared(&bar), d = (unsigned)__cvta_generic_to_shared(buf), rank, done = 0;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  if (t == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(b)); asm volatile("fence.mbarrier_init.release.cluster;"); }
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;");
  if (t == 0) asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(8192u));
  if (t == 0 && rank == 0)
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
                 :: "r"(d), "l"(&tmap), "r"(0), "r"(0), "r"(b), "h"((uint16_t)3) : "memory");
  while (!done) asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P; }" : "=r"(done) : "r"(b), "r"(0u));
  asm volatile("barrier.cluster.arrive.release.aligned; barrier.cluster.wait.acquire.aligned;");
  out[rank * 128 + t] = buf[t];
''',
    "14_tma_2d_plain": r'''  __shared__ __align__(1024) unsigned char buf[8192]; __shared__ __align__(8) unsigned long long bar;
  unsigned b = (unsigned)__cvta_generic_to_shared(&bar), d = (unsigned)__cvta_generic_to_shared(buf), done = 0;
  if (t == 0) { asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(b)); asm volatile("fence.proxy.async.shared::cta;"); }
  __syncthreads();
  if (t == 0) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(b), "r"(8192u));
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(d), "l"(&tmap), "r"(0), "r"(0), "r"(b) : "memory");
  }
  while (!done) asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.u32 %0, 1, 0, P; }" : "=r"(done) : "r"(b), "r"(0u));
  out[t] = buf[t];
''',
    "15a_redux_max_s32": r'''  int r; asm volatile("redux.sync.max.s32 %0, %1, 0xffffffff;" : "=r"(r) : "r"((int)t));
  out[t] = r;
''',
    "15b_redux_add_s32": r'''  int r; asm volatile("redux.sync.add.s32 %0, %1, 0xffffffff;" : "=r"(r) : "r"((int)t));
  out[t] = r;
''',
    "16a_ldmatrix_x4_trans": r'''  __shared__ __align__(128) unsigned short sm[4 * 128 * 8];
  for (unsigned i = t; i < 4 * 128 * 8; i += 128) sm[i] = i;
  __syncthreads();
  unsigned r0, r1, r2, r3, a = (unsigned)__cvta_generic_to_shared(&sm[(t / 32) * 1024 + (t % 32) * 8]);
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
  out[t] = r0 + r1 + r2 + r3;
''',
    "16b_stmatrix_x4_trans": r'''  __shared__ __align__(128) unsigned short sm[4 * 128 * 8];
  unsigned a = (unsigned)__cvta_generic_to_shared(&sm[(t / 32) * 1024 + (t % 32) * 8]);
  asm volatile("stmatrix.sync.aligned.m8n8.x4.trans.shared.b16 [%0], {%1,%2,%3,%4};" :: "r"(a), "r"(t), "r"(t + 1), "r"(t + 2), "r"(t + 3) : "memory");
  __syncthreads();
  out[t] = sm[t];
''',
}

if __name__ == "__main__":
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
    for name, body in PROBES.items():
        head = HEAD.replace("__launch_bounds__(128)", "__launch_bounds__(384, 1)") if "_wg_" in name else HEAD
        (out / f"{name}.cu").write_text(head + body + TAIL)
    print(len(PROBES), "probes written to", out)

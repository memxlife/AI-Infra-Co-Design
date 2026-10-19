# Shared-memory layout pairs on the cp.async path (TMA and warp specialization off, where make_linear_layout computes
# the right answer): the same tile with TileLang's inferred layout and with a linear one. Correctness, warm time
# (back-to-back), nvdisasm -hex SASS, registers. python layout_pairs.py OUT_DIR [SIZE]
import json, os, re, subprocess, sys, tempfile, torch, tilelang, tilelang.language as T
from tilelang.cuda.language import copy as cuda_copy
from tilelang.layout import make_linear_layout

def make(bm, bn, bk, st, th, linear):
    @tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
    def matmul(A, B, dtype=T.float16, accum_dtype=T.float32):
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((K, N), dtype)
        C = T.empty((M, N), dtype)
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=th) as (bx, by):
            A_shared = T.alloc_shared((bm, bk), dtype)
            B_shared = T.alloc_shared((bk, bn), dtype)
            C_local = T.alloc_fragment((bm, bn), accum_dtype)
            if linear:
                T.annotate_layout({A_shared: make_linear_layout(A_shared), B_shared: make_linear_layout(B_shared)})
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, bk), num_stages=st):
                cuda_copy(A[by * bm, k * bk], A_shared, disable_tma=True); cuda_copy(B[k * bk, bx * bn], B_shared, disable_tma=True)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * bm, bx * bn])
        return C
    return matmul

def warm_us(fn, reps=300):
    for _ in range(30): fn()
    torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(reps): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / reps * 1e3

out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024; os.makedirs(out, exist_ok=True)
TL = "/home/lishang/work/research/tilelang-work/tilelang"; meta = {}
A = torch.randn(n, n, device="cuda").half(); B = torch.randn(n, n, device="cuda").half(); ref = A.float() @ B.float()
for (bm, bn, bk, st, th) in [(128, 128, 32, 3, 128), (256, 128, 32, 3, 256), (64, 128, 32, 3, 256), (128, 128, 64, 2, 128), (64, 64, 64, 3, 128)]:
    for linear in (0, 1):
        name = f"cp_{bm}x{bn}x{bk}_st{st}_th{th}_{'linear' if linear else 'inferred'}"
        k = make(bm, bn, bk, st, th, linear).compile(M=n, N=n, K=n)
        C = k(A, B); torch.cuda.synchronize(); err = (C.float() - ref).abs().max().item(); us = warm_us(lambda: k(A, B))
        with tempfile.TemporaryDirectory() as d:
            open(f"{d}/k.cu", "w").write(k.get_kernel_source())
            subprocess.run(f"nvcc -O3 -std=c++20 -w -Xcudafe --diag_suppress=177 -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -I{TL}/src -I{TL}/3rdparty/cutlass/include -cubin -o {d}/k.cubin {d}/k.cu", shell=True, check=True)
            sass = subprocess.run(["nvdisasm", "-hex", f"{d}/k.cubin"], capture_output=True, text=True, check=True).stdout
            res = subprocess.run(["cuobjdump", "-res-usage", f"{d}/k.cubin"], capture_output=True, text=True).stdout
        open(f"{out}/{name}.sass", "w").write(sass)
        ops = {o: len(re.findall(rf"\b{o}\b", sass)) for o in ("LDSM.16.M88.4", "LDSM.16.MT88.4", "LDS.U16", "LDS", "LDS.64", "LDS.128")}
        meta[name] = {"bm": bm, "bn": bn, "bk": bk, "stages": st, "threads": th, "layout": "linear" if linear else "inferred", "regs": int(re.search(r"REG:(\d+)", res).group(1)), "err": err, "warm_us": us, "smem_ops": ops}
        print(name, f"err {err:.3f} warm {us:6.1f} us", ops, flush=True)
json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)

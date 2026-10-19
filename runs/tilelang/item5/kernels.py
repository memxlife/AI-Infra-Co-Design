# Plain global loads and atomics in real TileLang kernels: TileLang's vectorized split-K GEMV (examples/gemv, plain
# 16-byte loads, a shared-memory atomic per thread) and split-K GEMM (examples/gemm_splitk, cp.async path, fp32 C
# accumulated with global atomics). Correctness, warm time (back-to-back), nvdisasm -hex SASS, registers.
# python kernels.py OUT_DIR
import json, os, re, subprocess, sys, tempfile, torch, tilelang, tilelang.language as T
from tilelang.cuda.language import copy as cuda_copy
from tvm import DataType

def gemv(BLOCK_N, reduce_threads, dtype=T.float16, accum_dtype=T.float):
    @tilelang.jit
    def k(A, B):
        N, K = T.const("N, K")
        TILE_K = 128 // DataType(dtype).bits; BLOCK_K = reduce_threads * TILE_K
        A: T.Tensor((K,), dtype)
        B: T.Tensor((N, K), dtype)
        C = T.empty((N,), dtype)
        with T.Kernel(T.ceildiv(N, BLOCK_N), threads=(BLOCK_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0); tk = T.get_thread_binding(1)
            A_local = T.alloc_local((TILE_K,), dtype); B_local = T.alloc_local((TILE_K,), dtype)
            C_shared = T.alloc_shared((BLOCK_N,), accum_dtype); C_accum = T.alloc_local((1,), accum_dtype)
            if tk == 0:
                C_shared[tn] = 0
            T.clear(C_accum)
            for bk in T.serial(T.ceildiv(K, BLOCK_K)):
                for kk in T.vectorized(TILE_K):
                    A_local[kk] = A[bk * BLOCK_K + tk * TILE_K + kk]
                    B_local[kk] = B[bn * BLOCK_N + tn, bk * BLOCK_K + tk * TILE_K + kk]
                for kk in T.serial(TILE_K):
                    C_accum[0] += A_local[kk].astype(accum_dtype) * B_local[kk].astype(accum_dtype)
            T.atomic_add(C_shared[tn], C_accum[0])
            C[bn * BLOCK_N + tn] = C_shared[tn]
        return C
    return k

def splitk(bm, bn, bk, st, th, split):
    @tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})
    def k(A, B, C):
        M, N, K = T.const("M, N, K")
        sK = K // split
        A: T.Tensor((M, K), T.float16)
        B: T.Tensor((K, N), T.float16)
        C: T.Tensor((M, N), T.float32)
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), split, threads=th) as (bx, by, bz):
            A_shared = T.alloc_shared((bm, bk), T.float16); B_shared = T.alloc_shared((bk, bn), T.float16)
            C_local = T.alloc_fragment((bm, bn), T.float32)
            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(sK, bk), num_stages=st):
                cuda_copy(A[by * bm, bz * sK + ko * bk], A_shared, disable_tma=True); cuda_copy(B[bz * sK + ko * bk, bx * bn], B_shared, disable_tma=True)
                T.gemm(A_shared, B_shared, C_local)
            T.atomic_add(C[by * bm, bx * bn], C_local)
    return k

def warm_us(fn, reps=300):
    for _ in range(30): fn()
    torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(reps): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / reps * 1e3

if __name__ == "__main__":
    out = sys.argv[1]; os.makedirs(out, exist_ok=True); TL = "/home/lishang/work/research/tilelang-work/tilelang"; meta = {}
    def dump(name, k, info):
        with tempfile.TemporaryDirectory() as d:
            open(f"{d}/k.cu", "w").write(k.get_kernel_source())
            subprocess.run(f"nvcc -O3 -std=c++20 -w -Xcudafe --diag_suppress=177 -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -I{TL}/src -I{TL}/3rdparty/cutlass/include -cubin -o {d}/k.cubin {d}/k.cu", shell=True, check=True)
            sass = subprocess.run(["nvdisasm", "-hex", f"{d}/k.cubin"], capture_output=True, text=True, check=True).stdout
            res = subprocess.run(["cuobjdump", "-res-usage", f"{d}/k.cubin"], capture_output=True, text=True).stdout
        open(f"{out}/{name}.sass", "w").write(sass)
        kname = re.search(r"\.text\.(\S+?):", sass).group(1)
        ops = {}
        for o in re.findall(r"\b(LDG[.\w]*|REDG?[.\w]*|ATOMS?[.\w]*|ATOMG[.\w]*|LDL[.\w]*|STL[.\w]*|HMMA[.\w]*|LDGSTS[.\w]*)\b", sass): ops[o] = ops.get(o, 0) + 1
        meta[name] = {**info, "kernel_name": kname, "regs": int(re.search(r"REG:(\d+)", res).group(1)), "ops": ops}
        print(name, json.dumps(meta[name]), flush=True)

    for n in (2048, 4096):
        A = torch.randn(n, device="cuda").half(); B = torch.randn(n, n, device="cuda").half(); ref = (B.float() @ A.float())
        for (bn_, rt) in ((2, 32), (4, 32), (8, 32), (4, 64)):
            k = gemv(bn_, rt).compile(N=n, K=n); C = k(A, B); torch.cuda.synchronize()
            err = (C.float() - ref).abs().max().item(); us = warm_us(lambda: k(A, B))
            dump(f"gemv_n{n}_bn{bn_}_rt{rt}", k, {"kind": "gemv", "N": n, "K": n, "BLOCK_N": bn_, "reduce_threads": rt, "threads": bn_ * rt, "blocks": n // bn_, "err": err, "warm_us": us})
    for (M, N, K, bm, bn, bk, st, th) in ((1024, 1024, 1024, 128, 128, 32, 3, 128), (128, 4096, 4096, 128, 128, 32, 3, 128)):
        A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); ref = A.float() @ B.float()
        for split in (1, 2, 4, 8):
            k = splitk(bm, bn, bk, st, th, split).compile(M=M, N=N, K=K)
            C = torch.zeros(M, N, device="cuda"); k(A, B, C); torch.cuda.synchronize(); err = (C - ref).abs().max().item()
            us = warm_us(lambda: k(A, B, C))
            dump(f"splitk_{M}x{N}x{K}_{bm}x{bn}x{bk}_s{split}", k, {"kind": "splitk", "M": M, "N": N, "K": K, "bm": bm, "bn": bn, "bk": bk, "stages": st, "threads": th, "split": split, "blocks": (M // bm) * (N // bn) * split, "err": err, "warm_us": us})
    json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)

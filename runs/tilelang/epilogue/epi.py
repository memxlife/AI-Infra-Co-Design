# The C store as a choice: direct (T.copy(C_local, C)) or staged through shared memory (T.copy(C_local, C_shared);
# T.copy(C_shared, C)), TileLang's own example_gemm_autotune / example_gemm_persistent epilogue. Same tiles; correctness,
# warm time, store instructions in the SASS.
import re, subprocess, sys, tempfile, torch, tilelang, tilelang.language as T
from tilelang.cuda.language import copy as cuda_copy

def make(bm, bn, bk, st, th, staged, tma):
    pc = {} if tma else {"tl.disable_warp_specialized": True}
    @tilelang.jit(pass_configs=pc)
    def matmul(A, B, dtype=T.float16, accum_dtype=T.float32):
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((K, N), dtype)
        C = T.empty((M, N), dtype)
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=th) as (bx, by):
            A_shared = T.alloc_shared((bm, bk), dtype)
            B_shared = T.alloc_shared((bk, bn), dtype)
            C_local = T.alloc_fragment((bm, bn), accum_dtype)
            if staged:
                C_shared = T.alloc_shared((bm, bn), dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, bk), num_stages=st):
                if tma:
                    T.copy(A[by * bm, k * bk], A_shared); T.copy(B[k * bk, bx * bn], B_shared)
                else:
                    cuda_copy(A[by * bm, k * bk], A_shared, disable_tma=True); cuda_copy(B[k * bk, bx * bn], B_shared, disable_tma=True)
                T.gemm(A_shared, B_shared, C_local)
            if staged:
                T.copy(C_local, C_shared)
                T.copy(C_shared, C[by * bm, bx * bn])
            else:
                T.copy(C_local, C[by * bm, bx * bn])
        return C
    return matmul

def warm_us(fn, reps=300):
    for _ in range(30): fn()
    torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(reps): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / reps * 1e3

if __name__ == "__main__":
    M = N = K = int(sys.argv[1]) if len(sys.argv) > 1 else 1024
    A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); ref = A.float() @ B.float()
    for (bm, bn, bk, st, th, tma) in [(128, 128, 32, 3, 128, 0), (256, 128, 32, 3, 256, 0), (64, 128, 32, 3, 256, 0), (128, 128, 32, 3, 128, 1), (256, 128, 32, 3, 256, 1)]:
        res = {}
        for staged in (0, 1):
            try:
                k = make(bm, bn, bk, st, th, staged, tma).compile(M=M, N=N, K=K)
                C = k(A, B); torch.cuda.synchronize(); err = (C.float() - ref).abs().max().item()
                us = warm_us(lambda: k(A, B))
                src = k.get_kernel_source(); smem = re.search(r"MaxDynamicSharedMemorySize, (\d+)", k.get_host_source() or "") if hasattr(k, "get_host_source") else None
                with tempfile.TemporaryDirectory() as d:
                    open(f"{d}/k.cu", "w").write(src)
                    TL = "/home/lishang/work/research/tilelang-work/tilelang"
                    r = subprocess.run(f"nvcc -O3 -std=c++20 -w -Xcudafe --diag_suppress=177 -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -I{TL}/src -I{TL}/3rdparty/cutlass/include -cubin -o {d}/k.cubin {d}/k.cu && cuobjdump -sass {d}/k.cubin", shell=True, capture_output=True, text=True)
                    ops = re.findall(r"\b(STG\.E(?:\.\d+|\.U16|\.64|\.128)?|STSM[.\w]*|STS(?:\.\d+)?|UTMASTG[.\w]*|UBLKCP[.\w]*)\b", r.stdout)
                    from collections import Counter; c = Counter(ops)
                res[staged] = us
                print(f"{'TMA ' if tma else 'cp.async'} {bm}x{bn}x{bk} st{st} th{th} {'staged' if staged else 'direct'}: err {err:.3f} warm {us:6.1f} us | stores {dict(c)}", flush=True)
            except Exception as e:
                print(f"{'TMA ' if tma else 'cp.async'} {bm}x{bn}x{bk} {'staged' if staged else 'direct'}: FAILED {str(e).strip().splitlines()[-1][:160]}", flush=True)
        if len(res) == 2: print(f"   staged saves {res[0] - res[1]:+.1f} us ({(res[0]/res[1]-1)*100:+.0f}% faster)")

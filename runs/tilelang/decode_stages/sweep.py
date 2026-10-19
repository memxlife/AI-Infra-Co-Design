# Decode-shape GEMMs on the TMA path (TileLang's default lowering: TMA + warp specialization): does a deeper pipeline
# help? Tiles x num_stages where shared memory allows (99 KB). Each: correctness, back-to-back time (median of three
# interleaved rounds) and time after a clean L2 flush (median of 50). python sweep.py OUT.json
import json, statistics, sys, torch, tilelang, tilelang.language as T

def make(bm, bn, bk, st, th=128):
    @tilelang.jit
    def matmul(A, B, dtype=T.float16, accum_dtype=T.float32):
        M, N, K = T.const("M, N, K")
        A: T.Tensor((M, K), dtype)
        B: T.Tensor((K, N), dtype)
        C = T.empty((M, N), dtype)
        with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=th) as (bx, by):
            A_shared = T.alloc_shared((bm, bk), dtype); B_shared = T.alloc_shared((bk, bn), dtype)
            C_local = T.alloc_fragment((bm, bn), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, bk), num_stages=st):
                T.copy(A[by * bm, k * bk], A_shared); T.copy(B[k * bk, bx * bn], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * bm, bx * bn])
        return C
    return matmul

def warm_us(fn, reps=300):
    for _ in range(30): fn()
    torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(reps): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / reps * 1e3

if __name__ == "__main__":
    flush = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    def flushed_us(fn, reps=50):
        e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); t = []
        for _ in range(5): fn()
        for _ in range(reps):
            flush.sum(); e0.record(); fn(); e1.record(); torch.cuda.synchronize(); t.append(e0.elapsed_time(e1) * 1e3)
        return statistics.median(t)

    CASES = {128: [(256, 64, 32, s) for s in (2, 3, 4)] + [(128, 64, 32, s) for s in (2, 3, 4, 6)] + [(128, 64, 64, s) for s in (2, 3, 4)]
                  + [(128, 128, 32, s) for s in (2, 3, 4, 6)] + [(128, 32, 64, s) for s in (2, 4, 6)],
             32: [(32, 128, 128, 2)] + [(32, 128, 64, s) for s in (2, 3, 4)] + [(32, 64, 128, s) for s in (2, 3, 4)] + [(32, 64, 64, s) for s in (2, 4, 6)]}
    N = K = 4096; ks = {}; meta = {}
    for M, cfgs in CASES.items():
        A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); ref = A.float() @ B.float()
        for (bm, bn, bk, st) in cfgs:
            name = f"M{M}_{bm}x{bn}x{bk}_st{st}"
            if (bm + bn) * bk * 2 * st > 99 * 1024: print(name, "skip: shared memory"); continue
            try:
                k = make(bm, bn, bk, st).compile(M=M, N=N, K=K); C = k(A, B); torch.cuda.synchronize()
                err = ((C.float() - ref).norm() / ref.norm()).item()
                ks[name] = (lambda k=k, A=A, B=B: k(A, B)); meta[name] = {"M": M, "bm": bm, "bn": bn, "bk": bk, "stages": st, "rel_err": err,
                    "blocks": (-(-M // bm)) * (N // bn), "smem_KB": (bm + bn) * bk * 2 * st / 1024}
            except Exception as e:
                print(name, "FAILED", str(e).strip().splitlines()[-1][:150], flush=True)
    w = {n: [] for n in ks}
    for r in range(3):
        for n, f in ks.items(): w[n].append(warm_us(f))
    for n, f in ks.items():
        meta[n].update(warm_us=statistics.median(w[n]), flushed_us=flushed_us(f))
        x = meta[n]; gbs = (K * N * 2 + x["M"] * K * 2) / x["flushed_us"] / 1e3
        print(f"{n:24s} blocks {x['blocks']:4d} smem {x['smem_KB']:5.1f} KB err {x['rel_err']:.1e} | back-to-back {x['warm_us']:6.1f} us | flushed {x['flushed_us']:6.1f} us ({gbs:5.1f} GB/s)", flush=True)
    json.dump(meta, open(sys.argv[1], "w"), indent=1)

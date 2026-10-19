# Skeleton 2: examples/gemm_streamk, kernel body unchanged; module parameters set for the shape and 48 SMs; WS off.
import importlib.util, re, sys, torch, tilelang, tilelang.language as T
TL = "/home/lishang/work/research/tilelang-work/tilelang"
base = open(f"{TL}/examples/gemm_streamk/example_tilelang_gemm_streamk.py").read().split("\n\ndef main")[0]
base = base.replace("@tilelang.jit\ndef tl_matmul_streamk", "@tilelang.jit(pass_configs={'tl.disable_warp_specialized': True})\ndef tl_matmul_streamk")
def load(name):
    spec = importlib.util.spec_from_file_location(name, f"{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
def median_kernel_us(fn, pre, reps=200):
    ts = []
    for _ in range(20): pre(); fn()
    torch.cuda.synchronize()
    for _ in range(reps):
        pre(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        e0.record(); fn(); e1.record(); ts.append((e0, e1))
    torch.cuda.synchronize(); v = sorted(a.elapsed_time(b) for a, b in ts); return v[len(v)//2] * 1e3
cfgs = [(int(a) for a in c.split(",")) for c in sys.argv[1:]] or [(1024,1024,1024,16,128,32,2,64,48), (4096,4096,1024,16,128,32,2,64,48)]
for M, N, K, bm, bn, bk, st, th, sms in cfgs:
    src = base.replace("m = 256\n", f"m = {M}\n").replace("n = 1024\n", f"n = {N}\n").replace("k = 512\n", f"k = {K}\n").replace("total_sm = 108\n", f"total_sm = {sms}\n")
    src = src.replace("BLOCK_SIZE_M = 16\n", f"BLOCK_SIZE_M = {bm}\n").replace("BLOCK_SIZE_N = 128\n", f"BLOCK_SIZE_N = {bn}\n").replace("BLOCK_SIZE_K = 32\n", f"BLOCK_SIZE_K = {bk}\n")
    name = f"skel2_{M}x{N}x{K}_{bm}x{bn}x{bk}_st{st}_th{th}_sm{sms}"; open(name + ".py", "w").write(src + "\n")
    try:
        ns = vars(load(name)); A, B = ns["A"], ns["B"]; C = torch.zeros((M, N), device="cuda", dtype=torch.float32); ref = (A.float() @ B.float().T)
        k = ns["tl_matmul_streamk"].compile(M=M, N=N, K=K, streamk_tiles=ns["streamk_tiles"], block_M=bm, block_N=bn, block_K=bk, trans_A=False, trans_B=True, dtypeAB=T.float16, dtypeC=T.float32, accum_dtype=T.float32, num_stages=st, threads=th)
        C.zero_(); k(A, B, C); torch.cuda.synchronize(); err = (C - ref).abs().max().item()
        us = median_kernel_us(lambda: k(A, B, C), lambda: C.zero_())
        s = k.get_kernel_source(); lb = re.search(r"__launch_bounds__\((\d+)", s)
        print(f"{name}: total_tiles {ns['total_tiles']} streamk_tiles {ns['streamk_tiles']} blocking {ns['blocking_tiles']} partition {ns['sm_patition_factor']} | max err {err:.3f} | kernel median {us:.1f} us (C zeroed outside the timed region) | launch_bounds {lb.group(1) if lb else '?'} | cp_async {'cp_async' in s} TMA {'CUtensorMap' in s} atomic {'atomicAdd' in s or 'AtomicAdd' in s}", flush=True)
        open(name + ".cu", "w").write(s)
    except Exception as e:
        import traceback; print(f"{name}: FAILED " + traceback.format_exc().strip().splitlines()[-1], flush=True)

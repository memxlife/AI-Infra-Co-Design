# Race check for the skeletons: 30 random inputs per kernel against torch, then (under compute-sanitizer racecheck when launched that way) one run.
import importlib.util, sys, torch, tilelang, tilelang.language as T
TL = "/home/lishang/work/research/tilelang-work/tilelang"
def load(name, path=None):
    spec = importlib.util.spec_from_file_location(name, path or f"{name}.py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
which = sys.argv[1]; reps = int(sys.argv[2]) if len(sys.argv) > 2 else 30
M = N = K = 1024
if which.startswith("fam_") or which.startswith("skel1_"):
    k = load(which).matmul.compile(M=M, N=N, K=K, block_M=128, block_N=128, block_K=32) if which.startswith("skel1_") else load(which).matmul.compile(M=M, N=N, K=K)
    worst = 0.0
    for i in range(reps):
        A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); C = k(A, B); torch.cuda.synchronize()
        worst = max(worst, (C.float() - A.float() @ B.float()).abs().max().item())
    src = k.get_kernel_source(); open(f"src_{which}.cu", "w").write(src); print(f"racecheck {which}: {reps} inputs, worst max err {worst:.4f} | WS {'consumer' in src or 'producer' in src} cp_async {'cp_async' in src} TMA {'CUtensorMap' in src}", flush=True)
elif which.startswith("skel2_"):
    ns = vars(load(which)); A, B = ns["A"], ns["B"]
    k = ns["tl_matmul_streamk"].compile(M=M, N=N, K=K, streamk_tiles=ns["streamk_tiles"], block_M=16, block_N=128, block_K=32, trans_A=False, trans_B=True, dtypeAB=T.float16, dtypeC=T.float32, accum_dtype=T.float32, num_stages=2, threads=64)
    worst = 0.0
    for i in range(reps):
        A = torch.randn(M, K, device="cuda").half(); B = torch.randn(N, K, device="cuda").half(); C = torch.zeros(M, N, device="cuda"); k(A, B, C); torch.cuda.synchronize()
        worst = max(worst, (C - A.float() @ B.float().T).abs().max().item())
    open(f"src_{which}.cu", "w").write(k.get_kernel_source()); print(f"racecheck {which}: {reps} inputs, worst max err {worst:.4f}", flush=True)
elif which == "frag":
    sys.path.insert(0, TL); pr = load("sm120_nvfp4_w4a16_gemm", "pr3290/sm120_nvfp4_w4a16_gemm.py")
    from examples.dequantize_gemm.quantize.nvfp4 import decode_packed_fp4_e2m1, decode_ue4m3_scale_bytes, quantize_bf16_to_nvfp4_blockscaled
    N = K = 4096; worst = 0.0
    for i in range(max(1, reps // 10)):
        w_bf16 = (torch.randn(N, K, device="cuda") * 0.02 * 448).to(torch.bfloat16); packed, _, sb = quantize_bf16_to_nvfp4_blockscaled(w_bf16, return_scale_bytes=True)
        weight = pr.prepare_nvfp4_weight(packed, sb, 1 / 448.0); w_ref = decode_packed_fp4_e2m1(packed) * decode_ue4m3_scale_bytes(sb).repeat_interleave(16, dim=1) / 448.0
        for Mi in (32, 128):
            x = torch.randn(Mi, K, device="cuda", dtype=torch.bfloat16); y = torch.empty(Mi, N, device="cuda", dtype=torch.bfloat16); pr.launch(x, y, weight, pr.FRAG); torch.cuda.synchronize()
            worst = max(worst, ((y.float() - x.float() @ w_ref.T).norm() / (x.float() @ w_ref.T).norm()).item())
    print(f"racecheck frag: worst rel err {worst:.2e}", flush=True)

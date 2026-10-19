# On the Spark: compile each rendered cp.async kernel, check it against torch, time it warm (events, back-to-back, the
# way the ablation harness and cuBLASLt were timed), and save its CUDA, registers, shared memory and encoded SASS.
import importlib.util, json, re, subprocess, sys, torch
TL = "/home/lishang/work/research/tilelang-work/tilelang"
items = json.load(open("items.json")); out = open("measured.jsonl", "a")
for it in items:
    s = it["shape"]; M, N, K = s["M"], s["N"], s["K"]; rec = dict(it)
    try:
        spec = importlib.util.spec_from_file_location(it["name"], it["name"] + ".py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        k = m.matmul.compile(M=M, N=N, K=K)
        torch.manual_seed(0); A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half()
        C = k(A, B); torch.cuda.synchronize(); ref = A.float() @ B.float(); rel = ((C.float() - ref).norm() / ref.norm()).item()
        for _ in range(20): k(A, B)
        torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
        reps = 200 if M * N * K < 2**33 else 30
        e0.record()
        for _ in range(reps): k(A, B)
        e1.record(); torch.cuda.synchronize(); us = e0.elapsed_time(e1) / reps * 1e3
        src = k.get_kernel_source(); open(it["name"] + ".kernel.cu", "w").write(src)
        r = subprocess.run(f"nvcc -O3 -std=c++20 -w -Xcudafe --diag_suppress=177 -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -I{TL}/src -I{TL}/3rdparty/cutlass/include -cubin -Xptxas -v -o /tmp/{it['name']}.cubin {it['name']}.kernel.cu && nvdisasm -hex /tmp/{it['name']}.cubin > {it['name']}.sass", shell=True, capture_output=True, text=True)
        regs = re.search(r"Used (\d+) registers", r.stderr); smem = sum(int(x) for x in re.findall(r"(\d+) bytes smem", r.stderr))
        offs = [int(x) for x in re.findall(r"buf_dyn_shmem \+ (\d+)", src)]
        rec.update(ok=rel < 1e-2, rel_err=rel, warm_us=us, regs=int(regs.group(1)) if regs else None, static_smem=smem)
        print(f"{it['name']}: rel {rel:.1e} | warm {us:8.1f} us | regs {rec['regs']}", flush=True)
    except Exception as e:
        rec.update(ok=False, error=str(e).strip().splitlines()[-1][:200]); print(f"{it['name']}: FAILED {rec['error']}", flush=True)
    out.write(json.dumps(rec) + "\n"); out.flush()

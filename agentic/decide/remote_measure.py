"""Runs on the Spark: render each assignment with the unified GEMM template, compile it, check it against torch, save
its SASS and registers, and time it the corrected way (agentic/decide/README in notes): each launch alone between two
events with the GPU kept busy before the first event (the host's ~5 us wrapper time hidden), warm (no flush) and cold
(the L2 cleared by READING 64 MB first: no dirty lines), median of N. cuBLASLt timed in the same session both ways.
python remote_measure.py JOBS.json OUT_DIR"""
import json, os, re, statistics, subprocess, sys, tempfile, time, traceback
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from template import render, name as cfg_name

TL = "/home/lishang/work/research/tilelang-work/tilelang"
FL = None


def timed(fn, cold, n=40):
    global FL
    if FL is None: FL = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); t = []
    for _ in range(5): fn()
    for _ in range(n):
        if cold: FL.sum()
        torch.cuda._sleep(300000); e0.record(); fn(); e1.record(); e1.synchronize(); t.append(e0.elapsed_time(e1) * 1e3)
    return statistics.median(t)


def main():
    jobs = json.load(open(sys.argv[1])); out = sys.argv[2]; os.makedirs(f"{out}/sass", exist_ok=True); res = []
    for job in jobs:
        shape = job["shape"]; M, N, K = shape["M"], shape["N"], shape["K"]
        A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); ref = A.float() @ B.float()
        for a in job["assignments"]:
            nm = cfg_name(a); rec = {"name": nm, "shape": shape, "assign": a}
            try:
                src = render(a); path = os.path.join(tempfile.mkdtemp(), "k.py"); open(path, "w").write(src)
                import importlib.util
                spec = importlib.util.spec_from_file_location("k_" + nm, path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
                t0 = time.time(); k = mod.matmul.compile(M=M, N=N, K=K); rec["compile_s"] = round(time.time() - t0, 1)
                if a.get("split_k", 1) > 1:
                    C = torch.zeros(M, N, device="cuda"); k(A, B, C); torch.cuda.synchronize(); out_c = C
                    fn = (lambda k=k, C=C: k(A, B, C))
                else:
                    out_c = k(A, B); torch.cuda.synchronize(); fn = (lambda k=k: k(A, B))
                rec["rel_err"] = ((out_c.float() - ref).norm() / ref.norm()).item(); rec["ok"] = rec["rel_err"] < 1e-2
                if not job.get("compile_only"):
                    rec["warm_us"] = timed(fn, False); rec["cold_us"] = timed(fn, True)
                with tempfile.TemporaryDirectory() as d:
                    open(f"{d}/k.cu", "w").write(k.get_kernel_source())
                    subprocess.run(f"nvcc -O3 -std=c++20 -w -Xcudafe --diag_suppress=177 -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -I{TL}/src -I{TL}/3rdparty/cutlass/include -cubin -o {d}/k.cubin {d}/k.cu", shell=True, check=True)
                    sass = subprocess.run(["nvdisasm", "-hex", f"{d}/k.cubin"], capture_output=True, text=True, check=True).stdout
                    r = subprocess.run(["cuobjdump", "-res-usage", f"{d}/k.cubin"], capture_output=True, text=True).stdout
                tag = f"{M}x{N}x{K}_{nm}"; open(f"{out}/sass/{tag}.sass", "w").write(sass)
                rec["regs"] = int(re.search(r"REG:(\d+)", r).group(1)); rec["kernel_name"] = re.search(r"\.text\.(\S+?):", sass).group(1); rec["sass"] = f"sass/{tag}.sass"
                rec["ops"] = {o: len(re.findall(rf"\b{o}\b", sass)) for o in ("HMMA.16816.F32", "LDGSTS.E.BYPASS.128", "UTMALDG.2D", "STSM.16.M88.4", "UTMASTG.2D", "STG.E", "REDG.E.ADD.F32x2.FTZ.RN.STRONG.GPU")}
            except Exception as e:
                rec["ok"] = False; rec["error"] = (str(e).strip().splitlines() or ["?"])[-1][:300]
            res.append(rec); print(json.dumps({k: rec.get(k) for k in ("name", "ok", "rel_err", "warm_us", "cold_us", "regs", "error")}), flush=True)
        if job.get("cublaslt"):
            for cold in (0, 1):
                o = subprocess.run(["../cublaslt_gemm", str(M), str(N), str(K), "20", "100", str(cold)], capture_output=True, text=True).stdout
                v = float(re.search(r"cublaslt_best_ms=([0-9.]+)", o).group(1)) * 1e3
                res.append({"name": "cuBLASLt", "shape": shape, "cold": cold, "us": v}); print(json.dumps(res[-1]), flush=True)
    json.dump(res, open(f"{out}/results.json", "w"), indent=1)


if __name__ == "__main__":
    main()

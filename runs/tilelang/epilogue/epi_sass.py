# Dump nvdisasm -hex SASS (and registers, shared memory) of the ten epilogue kernels, direct and staged, at one shape.
# python epi_sass.py OUT_DIR [SIZE]
import json, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from epi import make

out = sys.argv[1]; n = int(sys.argv[2]) if len(sys.argv) > 2 else 1024; os.makedirs(out, exist_ok=True)
TL = "/home/lishang/work/research/tilelang-work/tilelang"; meta = {}
for (bm, bn, bk, st, th, tma) in [(128, 128, 32, 3, 128, 0), (256, 128, 32, 3, 256, 0), (64, 128, 32, 3, 256, 0), (128, 128, 32, 3, 128, 1), (256, 128, 32, 3, 256, 1)]:
    for staged in (0, 1):
        name = f"{'tma' if tma else 'cp'}_{bm}x{bn}x{bk}_{'staged' if staged else 'direct'}"
        k = make(bm, bn, bk, st, th, staged, tma).compile(M=n, N=n, K=n)
        with tempfile.TemporaryDirectory() as d:
            open(f"{d}/k.cu", "w").write(k.get_kernel_source())
            subprocess.run(f"nvcc -O3 -std=c++20 -w -Xcudafe --diag_suppress=177 -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -I{TL}/src -I{TL}/3rdparty/cutlass/include -cubin -o {d}/k.cubin {d}/k.cu", shell=True, check=True)
            sass = subprocess.run(["nvdisasm", "-hex", f"{d}/k.cubin"], capture_output=True, text=True, check=True).stdout
            res = subprocess.run(["cuobjdump", "-res-usage", f"{d}/k.cubin"], capture_output=True, text=True).stdout
        open(f"{out}/{name}.sass", "w").write(sass)
        regs = int(re.search(r"REG:(\d+)", res).group(1)); sm = int(re.search(r"SHARED:(\d+)", res).group(1))
        dyn = re.search(r"MaxDynamicSharedMemorySize,\s*(\d+)", k.get_host_source() if hasattr(k, "get_host_source") else "")
        meta[name] = {"bm": bm, "bn": bn, "bk": bk, "stages": st, "threads": th, "tma": tma, "staged": staged, "regs": regs, "static_smem": sm, "dyn_smem": int(dyn.group(1)) if dyn else None}
        print(name, meta[name], flush=True)
json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)

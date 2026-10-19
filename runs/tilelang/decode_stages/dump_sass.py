# SASS (nvdisasm -hex) and registers of the 26 decode-sweep kernels, for the simulator. python dump_sass.py OUT_DIR
import json, os, re, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sweep import make
out = sys.argv[1]; os.makedirs(out, exist_ok=True); TL = "/home/lishang/work/research/tilelang-work/tilelang"
meta = json.load(open("sweep.json")); res = {}
for name, m in meta.items():
    k = make(m["bm"], m["bn"], m["bk"], m["stages"]).compile(M=m["M"], N=4096, K=4096)
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/k.cu", "w").write(k.get_kernel_source())
        subprocess.run(f"nvcc -O3 -std=c++20 -w -Xcudafe --diag_suppress=177 -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -I{TL}/src -I{TL}/3rdparty/cutlass/include -cubin -o {d}/k.cubin {d}/k.cu", shell=True, check=True)
        sass = subprocess.run(["nvdisasm", "-hex", f"{d}/k.cubin"], capture_output=True, text=True, check=True).stdout
        r = subprocess.run(["cuobjdump", "-res-usage", f"{d}/k.cubin"], capture_output=True, text=True).stdout
    open(f"{out}/{name}.sass", "w").write(sass)
    res[name] = {"regs": int(re.search(r"REG:(\d+)", r).group(1)), "kernel_name": re.search(r"\.text\.(\S+?):", sass).group(1)}
    print(name, res[name], flush=True)
json.dump(res, open(f"{out}/regs.json", "w"), indent=1)

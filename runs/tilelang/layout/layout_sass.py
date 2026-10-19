# The shared-memory layout as a choice: TileLang's inferred (swizzled) layout against make_linear_layout, same kernel.
# For each: correctness, warm time (back-to-back, as runs/tilelang/epilogue/epi.py), nvdisasm -hex SASS, registers.
# python layout_sass.py PROBE_DIR OUT_DIR
import importlib.util, json, os, re, subprocess, sys, tempfile, torch

probe, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
TL = "/home/lishang/work/research/tilelang-work/tilelang"
CASES = [("1024", {"M": 1024, "N": 1024, "K": 1024}, {"bm": 256, "bn": 128, "bk": 64, "stages": 2, "threads": 256}),
         ("m32", {"M": 32, "N": 4096, "K": 4096}, {"bm": 16, "bn": 256, "bk": 32, "stages": 5, "threads": 256})]

def warm_us(fn, reps=300):
    for _ in range(30): fn()
    torch.cuda.synchronize(); e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True); e0.record()
    for _ in range(reps): fn()
    e1.record(); torch.cuda.synchronize(); return e0.elapsed_time(e1) / reps * 1e3

meta = {}
for tag, shape, h in CASES:
    A = torch.randn(shape["M"], shape["K"], device="cuda").half(); B = torch.randn(shape["K"], shape["N"], device="cuda").half()
    ref = A.float() @ B.float()
    for layout in ("inferred", "linear"):
        path = os.path.join(probe, f"{tag}_smem_layout_{layout}.py")
        spec = importlib.util.spec_from_file_location(f"k_{tag}_{layout}", path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
        k = mod.matmul.compile(**shape)
        C = k(A, B); torch.cuda.synchronize(); err = (C.float() - ref).abs().max().item(); us = warm_us(lambda: k(A, B))
        name = f"{tag}_{layout}"
        with tempfile.TemporaryDirectory() as d:
            open(f"{d}/k.cu", "w").write(k.get_kernel_source())
            subprocess.run(f"nvcc -O3 -std=c++20 -w -Xcudafe --diag_suppress=177 -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -I{TL}/src -I{TL}/3rdparty/cutlass/include -cubin -o {d}/k.cubin {d}/k.cu", shell=True, check=True)
            open(f"{out}/{name}.sass", "w").write(subprocess.run(["nvdisasm", "-hex", f"{d}/k.cubin"], capture_output=True, text=True, check=True).stdout)
            res = subprocess.run(["cuobjdump", "-res-usage", f"{d}/k.cubin"], capture_output=True, text=True).stdout
        meta[name] = {**h, "shape": shape, "layout": layout, "regs": int(re.search(r"REG:(\d+)", res).group(1)), "err": err, "warm_us": us}
        print(name, meta[name], flush=True)
json.dump(meta, open(f"{out}/meta.json", "w"), indent=1)

import re, sys, importlib.util, torch
sys.argv = ["x"]
spec = importlib.util.spec_from_file_location("epi_mod", "epi_lib.py"); epi = importlib.util.module_from_spec(spec); spec.loader.exec_module(epi); make = epi.make
for (bm, bn, bk, st, th, tma) in [(256, 128, 32, 3, 256, 0), (128, 128, 32, 3, 128, 0), (128, 128, 64, 3, 128, 0), (128, 256, 64, 2, 256, 0)]:
    for staged in (0, 1):
        k = make(bm, bn, bk, st, th, staged, tma).compile(M=1024, N=1024, K=1024)
        src = k.get_kernel_source()
        offs = {m.group(1): int(m.group(2)) for m in re.finditer(r"void\* (\w+) = \(\(void\*\)\(\(char\*\)buf_dyn_shmem \+ (\d+)\)\)", src)}
        host = k.get_host_source() if hasattr(k, "get_host_source") else ""
        dyn = re.findall(r"(\d{4,6})", " ".join(l for l in host.splitlines() if "SharedMemory" in l or "smem" in l.lower()))
        pipe = st * (bm + bn) * bk * 2; ctile = bm * bn * 2
        print(f"{bm}x{bn}x{bk} st{st} {'staged' if staged else 'direct'}: buffers {offs} | host smem {sorted(set(dyn))[-1:] } | pipeline {pipe} C tile {ctile} sum {pipe + ctile} max {max(pipe, ctile)}")

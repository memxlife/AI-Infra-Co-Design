"""Runs ON the Spark: compile a TileLang kernel file with its hole values, check it against torch, time it, read the ptxas
resource report, optionally read Nsight counters, and print one JSON object (the `kernel_report` artifact).

  python spark_runner.py --kernel gemm.py --func matmul --m 1024 --n 1024 --k 1024 --holes '{"block_M": 128, ...}'
                         [--reps 100 --warmup 25] [--ncu] [--out report.json]

The kernel file defines a @tilelang.jit function `--func` whose parameters after the tensors are the holes. Timing uses the
profiler's CUDA-event backend (median); the cupti backend returns 0 on this host. Registers, spills and static shared memory
come from compiling the exported CUDA source with nvcc -Xptxas -v (TileLang's own resource report is HIP-only today).
"""
from __future__ import annotations

import argparse, importlib.util, json, os, re, subprocess, sys, tempfile, time


def smi_state():
    """R3: the SM clock, temperature and power at this moment, from nvidia-smi."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout.strip().split(", ")
        return {"sm_mhz": float(out[0]), "temp_c": float(out[1]), "power_w": float(out[2])}
    except Exception as e:
        return {"error": str(e)[:100]}


def dynamic_smem(k, holes):
    """Bytes of dynamic shared memory the kernel launches with. TileLang keeps it as the device function's
    `dyn_shared_memory_buf` attribute, present only when the kernel was compiled in this process (a cache hit has no
    artifact). Without it, the family's own formula: stages x (BM x BK + BK x BN) x 2 bytes, which the fresh compiles match."""
    art = getattr(k, "artifact", None)
    dm = getattr(art, "device_mod", None)
    if dm is not None:
        for _, f in dm.functions.items():
            if "dyn_shared_memory_buf" in f.attrs:
                return int(f.attrs["dyn_shared_memory_buf"]), "tilelang attribute"
    h = holes
    if all(x in h for x in ("block_M", "block_N", "block_K", "num_stages")):
        return h["num_stages"] * (h["block_M"] * h["block_K"] + h["block_K"] * h["block_N"]) * 2, "family formula"
    return 0, "unknown"


def ptxas_report(src: str, arch: str) -> dict:
    from tilelang.env import TILELANG_TEMPLATE_PATH, CUTLASS_INCLUDE_DIR, CUDA_HOME
    d = tempfile.mkdtemp(prefix="tl_ptxas_")
    cu = os.path.join(d, "kernel.cu"); cubin = os.path.join(d, "kernel.cubin")
    open(cu, "w").write(src)
    cmd = ["nvcc", "-std=c++20", f"-arch={arch}", "-cubin", "-o", cubin, cu, "-Xptxas", "-v", "-O3", "--expt-relaxed-constexpr",
           f"-I{TILELANG_TEMPLATE_PATH}", f"-I{CUTLASS_INCLUDE_DIR}", f"-I{CUDA_HOME}/include"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = {"ok": r.returncode == 0, "cubin": cubin if r.returncode == 0 else None, "functions": {}}
    if r.returncode != 0:
        out["error"] = r.stderr[-1500:]; return out
    cur = None
    for line in r.stderr.splitlines():
        m = re.search(r"Function properties for (\S+)", line)
        if m: cur = m.group(1); out["functions"][cur] = {}
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", line)
        if m and cur: out["functions"][cur].update(stack_bytes=int(m.group(1)), spill_stores=int(m.group(2)), spill_loads=int(m.group(3)))
        m = re.search(r"Used (\d+) registers(?:, used (\d+) barriers)?(?:, (\d+) bytes smem)?", line)
        if m and cur: out["functions"][cur].update(registers=int(m.group(1)), barriers=int(m.group(2) or 0), static_smem_bytes=int(m.group(3) or 0))
    sass = subprocess.run(["cuobjdump", "-sass", cubin], capture_output=True, text=True)
    if sass.returncode == 0:
        open(os.path.join(d, "kernel.sass"), "w").write(sass.stdout); out["sass"] = os.path.join(d, "kernel.sass")
    return out


def ncu_counters(script: str, metrics: list[str], kernel_name: str) -> dict:
    """One profiled launch of the kernel itself (-k by name; torch's own kernels are not counted), after one warm-up launch."""
    cmd = ["ncu", "--csv", "--metrics", ",".join(metrics), "-k", kernel_name, "--launch-skip", "1", "--launch-count", "1", sys.executable, script]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    vals = {}
    for line in r.stdout.splitlines():
        parts = [p.strip('"') for p in line.split('","')]
        if len(parts) > 14 and parts[-3] in metrics:
            try: vals[parts[-3]] = float(parts[-1].replace(",", ""))
            except ValueError: pass
    return {"ok": bool(vals), "metrics": vals, "note": None if vals else (r.stderr or r.stdout)[-400:]}


def main():
    os.environ.setdefault("TILELANG_CACHE_DIR", tempfile.mkdtemp(prefix="tl_cache_"))
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True); ap.add_argument("--func", default="matmul")
    ap.add_argument("--m", type=int, required=True); ap.add_argument("--n", type=int, required=True); ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--holes", default="{}"); ap.add_argument("--reps", type=int, default=100); ap.add_argument("--warmup", type=int, default=25)
    ap.add_argument("--ncu", action="store_true"); ap.add_argument("--out", default=None); ap.add_argument("--rtol", type=float, default=1e-2); ap.add_argument("--atol", type=float, default=1e-2)
    a = ap.parse_args()
    t0 = time.time()
    import torch, tilelang
    from tilelang.backend.target import determine_target
    holes = json.loads(a.holes)
    spec = importlib.util.spec_from_file_location("kernel_module", a.kernel); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    fn = getattr(mod, a.func)
    rep = {"shape": {"M": a.m, "N": a.n, "K": a.k}, "holes": holes, "kernel_file": os.path.basename(a.kernel), "func": a.func,
           "tilelang": {"version": tilelang.__version__, "target": json.loads(str(determine_target(return_object=True)))},
           "cuda": {"nvcc": subprocess.run(["nvcc", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1], "torch": torch.__version__},
           "host": os.uname().nodename}
    try:
        tc = time.time(); k = fn.compile(M=a.m, N=a.n, K=a.k, **holes); rep["compile_s"] = round(time.time() - tc, 2)
    except Exception as e:
        rep.update(ok=False, stage="compile", error=str(e)[-2000:]); print(json.dumps(rep)); return
    A = torch.randn(a.m, a.k, device="cuda").half(); B = torch.randn(a.k, a.n, device="cuda").half()
    C = k(A, B); torch.cuda.synchronize()
    ref = A @ B
    diff = (C.float() - ref.float()).abs(); tol = a.atol + a.rtol * ref.float().abs()
    rep["correct"] = bool((diff <= tol).all()); rep["max_abs_err"] = float(diff.max()); rep["mismatches"] = int((diff > tol).sum())
    p = k.get_profiler()
    rep["clock_before"] = smi_state()
    lat = p.do_bench(backend="event", warmup=a.warmup, rep=a.reps, return_mode="median")
    rep["clock_after"] = smi_state()
    rep["latency_ms"] = float(lat); rep["tflops"] = 2.0 * a.m * a.n * a.k / (lat * 1e-3) / 1e12
    rep["timing"] = {"backend": "cuda events", "warmup_ms": a.warmup, "rep_ms": a.reps, "statistic": "median"}
    src = k.get_kernel_source(); host = k.get_host_source()
    outdir = tempfile.mkdtemp(prefix="tl_kernel_"); open(os.path.join(outdir, "kernel.cu"), "w").write(src); open(os.path.join(outdir, "host.cu"), "w").write(host)
    rep["cuda_src"] = src; rep["cuda_path"] = os.path.join(outdir, "kernel.cu"); rep["host_path"] = os.path.join(outdir, "host.cu")
    m = re.search(r"__launch_bounds__\((\d+)(?:,\s*(\d+))?\)", src); rep["threads"] = int(m.group(1)) if m else None; rep["min_blocks"] = int(m.group(2)) if m and m.group(2) else None
    rep["dynamic_smem_bytes"], rep["dynamic_smem_source"] = dynamic_smem(k, holes)
    arch = rep["tilelang"]["target"].get("arch", "sm_121a")
    px = ptxas_report(src, arch); rep["ptxas"] = px
    f = next(iter(px["functions"].values()), {}) if px.get("ok") else {}
    rep["n_regs"] = f.get("registers"); rep["n_spills"] = (f.get("spill_stores", 0) + f.get("spill_loads", 0)) if f else None
    rep["static_smem_bytes"] = f.get("static_smem_bytes"); rep["smem_bytes"] = (rep["static_smem_bytes"] or 0) + rep["dynamic_smem_bytes"]
    rep["sass"] = px.get("sass"); rep["uses"] = {"tma": "CUtensorMap" in src, "mma": "mma" in src.lower(), "cp_async": "cp.async" in src, "wgmma": "wgmma" in src, "tcgen05": "tcgen05" in src}
    if a.ncu:
        script = os.path.join(outdir, "once.py")
        open(script, "w").write(f"import sys, json, torch, importlib.util\nspec = importlib.util.spec_from_file_location('km', {a.kernel!r}); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)\n"
                                f"k = getattr(mod, {a.func!r}).compile(M={a.m}, N={a.n}, K={a.k}, **{holes!r})\nA = torch.randn({a.m}, {a.k}, device='cuda').half(); B = torch.randn({a.k}, {a.n}, device='cuda').half()\nfor _ in range(3): k(A, B)\ntorch.cuda.synchronize()\n")
        metrics = ["gpu__time_duration.sum", "sm__throughput.avg.pct_of_peak_sustained_elapsed", "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
                   "sm__warps_active.avg.pct_of_peak_sustained_active", "lts__t_sectors_op_read_lookup_miss.sum", "lts__t_sectors_op_write_lookup_miss.sum",
                   "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
                   "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio", "smsp__average_warps_issue_stalled_math_pipe_throttle_per_issue_active.ratio", "smsp__issue_active.avg.pct_of_peak_sustained_active"]
        kname = re.search(r"__global__ void(?: __launch_bounds__\([^)]*\))? (\w+)\(", src)
        rep["ncu"] = ncu_counters(script, metrics, kname.group(1) if kname else f"{a.func}_kernel")
        mm = rep["ncu"]["metrics"]
        if mm.get("lts__t_sectors_op_read_lookup_miss.sum") is not None:
            rep["ncu"]["dram_bytes"] = 32.0 * (mm.get("lts__t_sectors_op_read_lookup_miss.sum", 0) + mm.get("lts__t_sectors_op_write_lookup_miss.sum", 0))
    rep["ok"] = True; rep["seconds"] = round(time.time() - t0, 1)
    js = json.dumps(rep, indent=1)
    if a.out: open(a.out, "w").write(js)
    print(js)


if __name__ == "__main__":
    main()

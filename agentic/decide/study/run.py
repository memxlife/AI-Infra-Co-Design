"""The tier study, run entirely on the Spark (no laptop needed). For each benchmark shape:
  1. per tier (T1, T1+T2, T1+T2+T3): price every legal configuration of that tier's decisions with the analytical model
     (agentic/decide/price.py), the MILP's top 60 (agentic/decide/mip.py), expanded with the decisions the model cannot
     price (prefetch and warp policy on the cp.async path); swizzle {0, 10} is a priced decision
  2. compile and check every candidate (unified template), save SASS
  3. the simulator scores them (in parallel; skipped for 4096^3 and 8192^3, where one simulation takes minutes)
  4. per tier, finalists = the simulator's top 5, the model's 5 best distinct kernels (8 where nothing is simulated) with
     their prefetch and warp-policy variants, and the smaller tiers' finalists; time them all, plus TileLang's default kernel,
     TileLang's autotuned pick (study/tilelang_autotune.json) and cuBLASLt (study/cublaslt_seq), with one protocol:
     per call in a back-to-back sequence (warm), and cold (a 64 MB read before every call, its time subtracted)
  5. tier 4 on the best kernel found (T3's finalists include the smaller tiers'): the same sequences as a CUDA graph; for the cold decode shapes also an L2 persistence
     window on the weights (driver API, 18 MB set aside)
Writes OUT/<shape>.json as it goes and OUT/summary.md at the end. Resumable: finished shapes are skipped.
python -m agentic.decide.study.run OUT_DIR"""
from __future__ import annotations

import ctypes, importlib.util, itertools, json, multiprocessing, os, re, statistics, subprocess, sys, tempfile, time, traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch

from ..mip import top_k
from ..pipeline import columns as all_columns, legal, regs_est, DOMAINS
from ..price import gemm_us
from ..template import render, name as cfg_name, DEFAULTS

HERE = Path(__file__).resolve().parent
TL = "/home/lishang/work/research/tilelang-work/tilelang"
SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096), (8192, 8192, 8192), (4096, 4096, 1024), (32, 4096, 4096), (128, 4096, 4096)]
DECODE = {(32, 4096, 4096), (128, 4096, 4096)}
NO_SIM = {(4096, 4096, 4096), (8192, 8192, 8192)}
TIERS = {
    1: {"path": ["tma_ws"], "c_store": ["staged"], "split_k": [1]},                   # TileLang's autotune knobs, its structure
    2: {"path": ["tma_ws", "cpasync"], "c_store": ["direct", "staged"], "split_k": [1]},
    3: {"path": ["tma_ws", "cpasync"], "c_store": ["direct", "staged"], "split_k": [1, 2, 4, 8]},
}
LOG = None


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(str(x) for x in a); print(s, flush=True)
    if LOG: open(LOG, "a").write(s + "\n")


# ---------------------------------------------------------------- 1. candidates per tier
def tier_columns(shape, timing, tier):
    t = TIERS[tier]; out = []; sd = {"M": shape[0], "N": shape[1], "K": shape[2]}
    dom = {**DOMAINS, "path": t["path"], "c_store": t["c_store"], "split_k": t["split_k"]}
    for vals in itertools.product(*dom.values()):
        a = dict(zip(dom, vals))
        if not legal(a, sd): continue
        h = {k: a[k] for k in ("block_M", "block_N", "block_K", "num_stages", "threads", "swizzle")}
        r, _ = regs_est(a["block_M"], a["block_N"], a["threads"], a["path"])
        try: p = gemm_us(h, sd, a["path"], r, timing=timing, c_store=a["c_store"], split=a["split_k"])
        except Exception: continue
        out.append({"id": len(out), "t": p["us"], "assign": a})
    return out


def to_template(a, **extra):
    t = {k: a[k] for k in ("block_M", "block_N", "block_K", "num_stages", "threads", "c_store", "split_k", "swizzle")}
    t["copy_path"] = "tma" if a["path"] == "tma_ws" else "cpasync"; t["warp_spec"] = 1 if a["path"] == "tma_ws" else 0
    return {**t, **extra}


def candidates(shape, timing, tier, n=60):                 # swizzle doubles the columns: 60 keeps ~30 distinct tile shapes
    cols = tier_columns(shape, timing, tier)
    sols = top_k([{"name": "g", "columns": [{"id": c["id"], "t": c["t"]} for c in cols]}], k=n)
    by = {c["id"]: c for c in cols}; out = []
    for rank, s in enumerate(sols):
        c = by[s["pick"]["g"]["column"]]; base = to_template(c["assign"])
        out.append({**base, "_model": c["t"]})
        if tier >= 2 and base["copy_path"] == "cpasync" and base["split_k"] == 1:
            for pf, wp in ((1, 0), (0, 2), (1, 2)):
                if pf < base["num_stages"]: out.append({**base, "prefetch": pf, "warp_policy": wp, "_model": c["t"]})
    return len(cols), out


# ---------------------------------------------------------------- 2. compile and check
def compile_one(a, shape, sass_dir, info=None):
    M, N, K = shape; src = render({k: v for k, v in a.items() if not k.startswith("_")})
    path = os.path.join(tempfile.mkdtemp(), "k.py"); open(path, "w").write(src)
    spec = importlib.util.spec_from_file_location("k" + str(abs(hash(path))), path); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    k = mod.matmul.compile(M=M, N=N, K=K)
    if info: return k, info                              # built by build_worker; TileLang loaded it from its disk cache
    with tempfile.TemporaryDirectory() as d:
        open(f"{d}/k.cu", "w").write(k.get_kernel_source())
        subprocess.run(f"nvcc -O3 -std=c++20 -w -Xcudafe --diag_suppress=177 -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr -I{TL}/src -I{TL}/3rdparty/cutlass/include -cubin -o {d}/k.cubin {d}/k.cu", shell=True, check=True)
        sass = subprocess.run(["nvdisasm", "-hex", f"{d}/k.cubin"], capture_output=True, text=True, check=True).stdout
        res = subprocess.run(["cuobjdump", "-res-usage", f"{d}/k.cubin"], capture_output=True, text=True).stdout
    nm = cfg_name(a); p = sass_dir / f"{nm}.sass"; p.write_text(sass)
    return k, {"regs": int(re.search(r"REG:(\d+)", res).group(1)), "kernel_name": re.search(r"\.text\.(\S+?):", sass).group(1), "sass": str(p)}


def build_worker(args):
    """Build one candidate in a separate process: TileLang keeps the kernel in its disk cache (the main process then
    loads it in ~0.02 s instead of ~1.8 s) and the cubin gives the SASS and the register count."""
    a, shape, sass_dir = args
    try: return compile_one(a, shape, Path(sass_dir))[1]
    except Exception as e: return {"error": (str(e).strip().splitlines() or ["?"])[-1][:200]}


def make_fn(k, a, A, B, M, N):
    if a.get("split_k", 1) > 1:
        C = torch.zeros(M, N, device="cuda"); return (lambda: k(A, B, C)), C
    return (lambda: k(A, B)), None


# ---------------------------------------------------------------- 3. simulate (worker process)
def sim_worker(args):
    rec, shape, warm = args
    try:
        from agentic.gb10.sim.kernel import simulate_kernel
        from agentic.gb10.sim.ws_validate import kernel_us
        a = rec["assign"]; sh = {"M": shape[0], "N": shape[1], "K": shape[2]}
        h = {k: a.get(k, DEFAULTS[k]) for k in ("block_M", "block_N", "block_K", "num_stages", "threads")} | {"swizzle": a.get("swizzle", 0), "tma": 1 if a.get("copy_path") == "tma" else 0}
        if a.get("copy_path") == "tma" and a.get("warp_spec", 1):
            return kernel_us(rec["name"], h, sh, warm=warm, sass_path=Path(rec["sass"]))
        return simulate_kernel(Path(rec["sass"]), h, sh, rec["regs"], warm=warm, split=a.get("split_k", 1), kernel=rec["kernel_name"], atomic_c=a.get("split_k", 1) > 1)["us"]
    except Exception as e:
        return float("inf")


# ---------------------------------------------------------------- 4. timing (the study's protocol)
FLUSH = None


def flush():
    global FLUSH
    if FLUSH is None: FLUSH = torch.empty(64 << 20, dtype=torch.uint8, device="cuda")
    FLUSH.sum()


def seq(fn, cold, n):
    torch.cuda.synchronize(); e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    def run(with_call):
        e0.record()
        for _ in range(n):
            if cold: flush()
            if with_call: fn()
        e1.record(); e1.synchronize(); return e0.elapsed_time(e1) * 1e3 / n
    return run(True) - (run(False) if cold else 0.0)


def graph_seq(fn, cold, n):
    def build(with_call):
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(2):
                if cold: flush()
                if with_call: fn()
        torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(n):
                if cold: flush()
                if with_call: fn()
        return g
    def run(g):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True); g.replay(); torch.cuda.synchronize(); t = []
        for _ in range(3): e0.record(); g.replay(); e1.record(); e1.synchronize(); t.append(e0.elapsed_time(e1) * 1e3 / n)
        return statistics.median(t)
    return run(build(True)) - (run(build(False)) if cold else 0.0)


def measure(fn, n, decode):
    for _ in range(5): fn()
    out = {"warm": statistics.median([seq(fn, False, n) for _ in range(3)])}
    out["cold"] = statistics.median([seq(fn, True, n) for _ in range(3)]) if True else None
    return out


class L2Pin:
    """An L2 persistence window on the current stream through the CUDA driver API (18 MB set aside)."""
    def __init__(self, tensor, setaside=18 << 20):
        self.cu = ctypes.CDLL("libcuda.so"); self.t = tensor; self.setaside = setaside
    def _set(self, base, nbytes, ratio):
        CU_LIMIT_PERSISTING_L2_CACHE_SIZE = 0x06
        self.cu.cuCtxSetLimit(ctypes.c_int(CU_LIMIT_PERSISTING_L2_CACHE_SIZE), ctypes.c_size_t(self.setaside if nbytes else 0))
        class Win(ctypes.Structure):
            _fields_ = [("base_ptr", ctypes.c_void_p), ("num_bytes", ctypes.c_size_t), ("hitRatio", ctypes.c_float), ("hitProp", ctypes.c_int), ("missProp", ctypes.c_int)]
        buf = (ctypes.c_byte * 64)(); w = Win.from_buffer(buf); w.base_ptr = base; w.num_bytes = nbytes; w.hitRatio = ratio; w.hitProp = 2; w.missProp = 1
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        r = self.cu.cuStreamSetAttribute(stream, ctypes.c_int(1), ctypes.byref(buf))      # 1 = access policy window
        if r != 0: raise RuntimeError(f"cuStreamSetAttribute returned {r}")
    def __enter__(self):
        nb = self.t.numel() * self.t.element_size(); win = min(nb, 128 << 20)
        self._set(self.t.data_ptr(), win, min(1.0, self.setaside / win)); return self
    def __exit__(self, *a):
        self._set(None, 0, 0.0); self.cu.cuCtxResetPersistingL2Cache()


def cublaslt(shape):
    o = subprocess.run([str(HERE / "cublaslt_seq"), *map(str, shape), "100"], capture_output=True, text=True).stdout
    m = re.search(r"single_call_median_us=([-\d.]+) seq_warm_stream_us=([-\d.]+) seq_cold_stream_us=([-\d.]+) seq_warm_graph_us=([-\d.]+) seq_cold_graph_us=([-\d.]+)", o)
    if not m: return {"error": o[-300:]}
    v = [x if x > 0 else None for x in map(float, m.groups())]       # a subtraction below zero is noise, not a time
    return {"single": v[0], "warm": v[1], "cold": v[2], "warm_graph": v[3], "cold_graph": v[4]}


# ---------------------------------------------------------------- TileLang's own kernels
TL_DEFAULT = {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 3, "threads": 128, "copy_path": "tma", "warp_spec": 1, "c_store": "direct"}


def tilelang_autotuned(shape):
    f = HERE / "tilelang_autotune.json"
    if not f.exists(): return None
    d = json.load(open(f)).get("{}x{}x{}".format(*shape))
    if not d or "config" not in d: return None
    c = d["config"]; st = c["num_stages"]
    a = {"block_M": c["block_M"], "block_N": c["block_N"], "block_K": c["block_K"], "num_stages": st, "threads": c["thread_num"],
         "swizzle": 10 if c["enable_rasteration"] else 0, "copy_path": "tma", "warp_spec": 1 if st >= 1 else 0, "c_store": "staged"}
    return a, d.get("latency_ms_tilelang")


# ---------------------------------------------------------------- one shape
def run_shape(shape, out_dir):
    M, N, K = shape; tag = f"{M}x{N}x{K}"; f = out_dir / f"{tag}.json"
    if f.exists(): log(tag, "done before, skipped"); return json.load(open(f))
    decode = shape in DECODE; timing = "cold" if decode else "warm"; t0 = time.time()
    sass_dir = out_dir / "sass" / tag; sass_dir.mkdir(parents=True, exist_ok=True)
    res = {"shape": tag, "primary": timing, "tiers": {}}
    cand_by_tier = {}; union = {}
    for tier in (1, 2, 3):
        ncols, cands = candidates(shape, timing, tier); cand_by_tier[tier] = cands
        res["tiers"][tier] = {"columns_priced": ncols, "candidates": len(cands)}
        for c in cands: union.setdefault(cfg_name(c), c)
    if os.environ.get("STUDY_SMOKE"):                    # a quick end-to-end check: three candidates per tier
        cand_by_tier = {t: c[:3] for t, c in cand_by_tier.items()}; union = {cfg_name(c): c for cs in cand_by_tier.values() for c in cs}
    log(tag, "candidates:", {t: len(c) for t, c in cand_by_tier.items()}, "unique", len(union))
    A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); ref = A.float() @ B.float()
    with ProcessPoolExecutor(max_workers=10, mp_context=multiprocessing.get_context("spawn")) as ex:     # no CUDA in forked children
        built = dict(zip(union, ex.map(build_worker, [(a, shape, str(sass_dir)) for a in union.values()])))
    log(tag, "built in parallel:", sum("error" not in b for b in built.values()), "of", len(built))
    recs, kernels = {}, {}
    for nm, a in union.items():
        if "error" in built[nm]: recs[nm] = {"name": nm, "ok": False, "error": built[nm]["error"]}; continue
        try:
            k, info = compile_one(a, shape, sass_dir, built[nm]); fn, Cbuf = make_fn(k, a, A, B, M, N); out = fn(); torch.cuda.synchronize()
            got = Cbuf if Cbuf is not None else out; err = ((got.float() - ref).norm() / ref.norm()).item()
            recs[nm] = {"name": nm, "assign": {k2: v for k2, v in a.items() if not k2.startswith("_")}, "model_us": a["_model"], "ok": err < 1e-2, "rel_err": err, **info}
            if err < 1e-2: kernels[nm] = (k, a)
        except Exception as e:
            recs[nm] = {"name": nm, "ok": False, "error": (str(e).strip().splitlines() or ["?"])[-1][:200]}
    log(tag, "compiled and correct:", len(kernels), "of", len(union))
    if shape not in NO_SIM:
        jobs = [(recs[nm], shape, not decode) for nm in kernels]
        with ProcessPoolExecutor(max_workers=12) as ex:
            for (rec, _, _), v in zip(jobs, ex.map(sim_worker, jobs)): rec["sim_us"] = v
        log(tag, "simulated", len(jobs))
    def base(nm): return re.sub(r"_pf\d+_wp\d+", "", nm)       # prefetch and warp policy: variants the model prices alike
    def top_distinct(names, k):                                     # the model's k best kernels, with all their variants
        keep = list(dict.fromkeys(base(n) for n in sorted(names, key=lambda n: recs[n]["model_us"])))[:k]
        return [n for n in names if base(n) in keep]
    finals = {}
    for tier, cands in cand_by_tier.items():
        names = [cfg_name(c) for c in cands if cfg_name(c) in kernels]
        by_model = top_distinct(names, 5 if shape not in NO_SIM else 8)
        by_sim = sorted(names, key=lambda n: recs[n].get("sim_us", float("inf")))[:5] if shape not in NO_SIM else []
        finals[tier] = list(dict.fromkeys(finals.get(tier - 1, []) + by_model + by_sim)); res["tiers"][tier]["finalists"] = finals[tier]
    n = 50 if M * N * K <= 2 ** 33 else 10
    if os.environ.get("STUDY_SMOKE"): n = 5
    timed = {}
    for nm in dict.fromkeys(sum(finals.values(), [])):
        k, a = kernels[nm]; fn, _ = make_fn(k, a, A, B, M, N)
        try: timed[nm] = measure(fn, n, decode); recs[nm]["measured"] = timed[nm]
        except Exception as e: recs[nm]["measure_error"] = str(e)[:200]
    for tier in (1, 2, 3):
        ok = [x for x in finals[tier] if x in timed]
        best = min(ok, key=lambda x: timed[x][timing]) if ok else None
        res["tiers"][tier].update({"best": best, "best_us": timed[best][timing] if best else None, "best_warm_us": timed[best]["warm"] if best else None, "best_cold_us": timed[best]["cold"] if best else None,
                                   "model_top1": min(finals[tier], key=lambda x: recs[x]["model_us"]) if finals[tier] else None})
    log(tag, "tiers:", {t: round(v.get("best_us") or -1, 1) for t, v in res["tiers"].items()})
    # tier 4: the T3 best with a CUDA graph; decode shapes also with the weights' L2 persistence window
    b3 = res["tiers"][3]["best"]
    if b3:
        k, a = kernels[b3]; fn, _ = make_fn(k, a, A, B, M, N); t4 = {}
        try: t4["warm_graph"] = graph_seq(fn, False, n); t4["cold_graph"] = graph_seq(fn, True, n)
        except Exception as e: t4["graph_error"] = str(e)[:200]
        if decode:
            try:
                with L2Pin(B): t4["cold_pin"] = statistics.median([seq(fn, True, n) for _ in range(3)])
                with L2Pin(B): t4["cold_pin_graph"] = graph_seq(fn, True, n)
            except Exception as e: t4["pin_error"] = str(e)[:200]
        cands4 = {"stream": timed[b3][timing], "graph": t4.get(f"{timing}_graph")}
        if decode: cands4.update({"pin": t4.get("cold_pin"), "pin_graph": t4.get("cold_pin_graph")})
        ok4 = {k2: v for k2, v in cands4.items() if v is not None}
        res["tiers"][4] = {"base": b3, **t4, "best_mode": min(ok4, key=ok4.get), "best_us": min(ok4.values())}
    # TileLang's own kernels and cuBLASLt, same protocol
    res["tilelang"] = {}
    for label, a in (("default", TL_DEFAULT),) + ((("autotuned", tilelang_autotuned(shape)[0]),) if tilelang_autotuned(shape) else ()):
        try:
            k, info = compile_one(a, shape, sass_dir); fn, Cbuf = make_fn(k, a, A, B, M, N); out = fn(); torch.cuda.synchronize()
            err = ((out.float() - ref).norm() / ref.norm()).item(); m = measure(fn, n, decode)
            res["tilelang"][label] = {"assign": a, "rel_err": err, **m, "us": m[timing]}
        except Exception as e:
            res["tilelang"][label] = {"assign": a, "error": (str(e).strip().splitlines() or ["?"])[-1][:200]}
    if tilelang_autotuned(shape): res["tilelang"]["autotuned"]["tilelang_own_latency_ms"] = tilelang_autotuned(shape)[1]
    res["cublaslt"] = cublaslt(shape); res["cublaslt"]["us"] = res["cublaslt"].get(timing)
    res["records"] = list(recs.values()); res["wall_s"] = round(time.time() - t0)
    json.dump(res, open(f, "w"), indent=1); log(tag, "written", f, f"{res['wall_s']} s")
    return res


def summary(out_dir):
    rows = []
    for shape in SHAPES:
        f = out_dir / "{}x{}x{}.json".format(*shape)
        if not f.exists(): continue
        r = json.load(open(f)); t = r["tiers"]; tl = r.get("tilelang", {}); cb = r.get("cublaslt", {})
        def g(x): return f"{x:.1f}" if isinstance(x, (int, float)) else "—"
        rows.append(f"| {r['shape']} | {r['primary']} | {g(tl.get('default', {}).get('us'))} | {g(tl.get('autotuned', {}).get('us'))} | {g(cb.get('us'))} | "
                    f"{g(t.get('1', t.get(1, {})).get('best_us'))} | {g(t.get('2', t.get(2, {})).get('best_us'))} | {g(t.get('3', t.get(3, {})).get('best_us'))} | "
                    f"{g(t.get('4', t.get(4, {})).get('best_us'))} ({t.get('4', t.get(4, {})).get('best_mode', '—')}) | {g(cb.get(r['primary'] + '_graph'))} |")
    md = ["# Tier study: MIP-optimized kernels against TileLang and cuBLASLt (GB10)", "",
          "Per call in a back-to-back sequence; warm for the compute shapes, cold (a 64 MB read before each call) for the decode shapes. Microseconds.", "",
          "| Shape | Timing | TileLang default | TileLang autotuned | cuBLASLt | MIP T1 | MIP T1+T2 | MIP T1+T2+T3 | MIP T1-T4 (mode) | cuBLASLt with a graph |",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"] + rows
    (out_dir / "summary.md").write_text("\n".join(md) + "\n"); return "\n".join(md)


def main():
    global LOG
    out_dir = Path(sys.argv[1]); out_dir.mkdir(parents=True, exist_ok=True); LOG = str(out_dir / "study.log")
    only = sys.argv[2:]
    for shape in SHAPES:
        if only and "{}x{}x{}".format(*shape) not in only: continue
        try: run_shape(shape, out_dir)
        except Exception: log("{}x{}x{}".format(*shape), "FAILED", traceback.format_exc()[-1500:])
        summary(out_dir)
    log("all shapes done"); print(summary(out_dir))


if __name__ == "__main__":
    main()

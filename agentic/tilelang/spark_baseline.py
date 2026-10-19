"""baseline.run on the Spark: the three references the brief asks for, on one shape.
  cublas    torch.matmul in FP16 (cuBLAS picks its own kernel)
  autotune  TileLang's own hand-written config list from examples/gemm/example_gemm_autotune.py (get_configs, no roller),
            each config compiled and timed on the gemm_basic family, best one kept: what AutoTuner does, same bench as the sweep
  carver    tilelang.carver MatmulTemplate(...).recommend_hints(topk=20), mapped onto the family's knobs, timed the same way
Writes one JSON with the three results and every timed config.
"""
from __future__ import annotations

import argparse, json, os, sys, time, traceback


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", required=True); ap.add_argument("--func", default="matmul")
    ap.add_argument("--m", type=int, required=True); ap.add_argument("--n", type=int, required=True); ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--reps", type=int, default=100); ap.add_argument("--warmup", type=int, default=25); ap.add_argument("--arch", default="sm_121a")
    ap.add_argument("--topk", type=int, default=20); ap.add_argument("--only", default="", help="comma list of kinds to run (default all)")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch
    from tilelang.profiler import do_bench
    from spark_sweep import load, one
    out = {"shape": {"M": a.m, "N": a.n, "K": a.k}, "kinds": {}}
    only = set(a.only.split(",")) if a.only else {"cublas", "autotune", "carver"}
    A = torch.randn(a.m, a.k, device="cuda").half(); B = torch.randn(a.k, a.n, device="cuda").half()
    if "cublas" in only:
        C = torch.empty(a.m, a.n, device="cuda", dtype=torch.half)
        lat = do_bench(lambda: A @ B, warmup=a.warmup, rep=a.reps, backend="event", return_mode="median")
        lat_out = do_bench(lambda: torch.matmul(A, B, out=C), warmup=a.warmup, rep=a.reps, backend="event", return_mode="median")
        # a plain CUDA-event loop as a second opinion on do_bench
        s0 = torch.cuda.Event(enable_timing=True); s1 = torch.cuda.Event(enable_timing=True)
        for _ in range(20): torch.matmul(A, B, out=C)
        torch.cuda.synchronize(); s0.record()
        for _ in range(200): torch.matmul(A, B, out=C)
        s1.record(); torch.cuda.synchronize(); lat_loop = s0.elapsed_time(s1) / 200
        best = min(lat, lat_out, lat_loop)
        out["kinds"]["cublas"] = {"latency_ms": float(best), "tflops": 2.0 * a.m * a.n * a.k / (best * 1e-3) / 1e12, "config": "torch.matmul fp16 (best of: do_bench, out= variant, plain event loop)",
                                  "detail": {"do_bench_ms": float(lat), "out_variant_ms": float(lat_out), "event_loop_ms": float(lat_loop), "torch": torch.__version__,
                                             "allow_fp16_reduced_precision": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction}}
    fn = load(a.kernel, a.func)

    def time_list(name, cfgs, note):
        rows = []
        for c in cfgs:
            r = one(fn, a, c, timing=True); rows.append(r)
            print(name, c, r.get("latency_ms"), r.get("error", ""), flush=True)
        good = [r for r in rows if r.get("ok") and r.get("correct")]
        best = min(good, key=lambda r: r["latency_ms"]) if good else None
        out["kinds"][name] = {"latency_ms": best and best["latency_ms"], "tflops": best and best["tflops"], "config": best and best["holes"],
                              "tried": len(rows), "compiled": sum(1 for r in rows if r.get("ok")), "note": note, "rows": rows}

    # TileLang's hand list (example_gemm_autotune.get_configs without the roller), on this family's knobs
    import itertools
    hand = [{"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": st, "threads": th, "swizzle": 0}
            for bm, bn, bk, st, th in itertools.product([64, 128, 256], [64, 128, 256], [32, 64], [0, 1, 2, 3], [128, 256])]
    if "autotune" in only:
        time_list("autotune", hand, "examples/gemm/example_gemm_autotune.py get_configs() list: 144 configs, each compiled and timed, best kept")
    # carver hints
    if "carver" not in only:
        json.dump(out, open(a.out, "w"), indent=1); print({k: (v.get("latency_ms"), v.get("config")) for k, v in out["kinds"].items()}); return
    try:
        import tilelang.language as T
        from tilelang.carver.template import MatmulTemplate
        from tilelang.carver.arch import CUDA
        from tilelang.carver.roller.rasterization import NoRasterization
        arch = CUDA("cuda")
        tpl = MatmulTemplate(M=a.m, N=a.n, K=a.k, in_dtype=T.float16, out_dtype=T.float16, accum_dtype=T.float32).with_arch(arch)
        hints = tpl.recommend_hints(topk=a.topk) or []
        cfgs, raw = [], []
        import math
        for h in hints:
            bm, bn = h.block
            # on sm_121a the hints carry no warp split; `thread` is the block's thread grid (e.g. [8, 16] = 128 threads)
            threads = int((bm // h.warp[0]) * (bn // h.warp[1]) * 32) if h.warp else int(math.prod(h.thread))
            c = {"block_M": int(bm), "block_N": int(bn), "block_K": int(h.rstep[0]), "num_stages": int(h.pipeline_stage) if h.pipeline_stage > 1 else 0,
                 "threads": threads, "swizzle": 8 if not isinstance(h.rasterization_plan, NoRasterization) else 0}
            cfgs.append(c); raw.append({"block": list(map(int, h.block)), "warp": list(map(int, h.warp)), "thread": list(map(int, h.thread)), "rstep": list(map(int, h.rstep)), "pipeline_stage": int(h.pipeline_stage), "raster": str(h.rasterization_plan)})
        out["carver_arch"] = {"name": str(getattr(arch, "name", arch)), "smem_cap": getattr(arch, "smem_cap", None), "reg_cap": getattr(arch, "reg_cap", None), "sm_partition": getattr(arch, "sm_partition", None), "compute_max_core": getattr(arch, "compute_max_core", None), "sm_version": getattr(arch, "sm_version", None)}
        out["carver_hints"] = raw
        time_list("carver", cfgs, f"tilelang.carver MatmulTemplate.recommend_hints(topk={a.topk}) mapped onto the family's knobs (threads = warps x 32, swizzle 8 when the hint rasterizes)")
    except Exception as e:
        out["kinds"]["carver"] = {"latency_ms": None, "error": traceback.format_exc()[-1500:]}
    json.dump(out, open(a.out, "w"), indent=1)
    print({k: (v.get("latency_ms"), v.get("config")) for k, v in out["kinds"].items()})


if __name__ == "__main__":
    main()

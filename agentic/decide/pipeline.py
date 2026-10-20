"""From decisions to the fastest kernel for one GEMM shape, end to end:
  1. columns: every legal assignment of the tied decisions of the unified template (copy path with its producer group,
     tile, depth, threads, C store, split-K), from the domains in DOMAINS and the rules in legal()
  2. price each with the analytical model plus the measured terms (agentic/decide/price.py)
  3. the MILP's top N (agentic/decide/mip.py)
  4. expand the shortlist with the decisions that matter only on the cp.async path (prefetch, warp policy; measured in
     runs/decide/cover2), compile them on the Spark (correctness, SASS, registers)
  5. the simulator re-ranks them; the union of the simulator's and the model's top few is timed on the GPU against
     cuBLASLt in the same session (remote_measure.py: launch hidden, warm and L2-flushed-by-reading)
python -m agentic.decide.pipeline M N K [--timing warm|cold]"""
from __future__ import annotations

import itertools, json, math, subprocess, sys, time
from pathlib import Path

from .mip import top_k
from .price import gemm_us
from .template import name as cfg_name

ROOT = Path(__file__).resolve().parents[2]
REMOTE = "~/work/research/AI-Infra-Co-Design-calib/tilelang/decide/run"
DOMAINS = {"path": ["tma_ws", "cpasync"], "block_M": [16, 32, 64, 128, 256], "block_N": [32, 64, 128, 256], "block_K": [32, 64, 128],
           "num_stages": [2, 3, 4, 5, 6], "threads": [128, 256], "c_store": ["direct", "staged"], "split_k": [1, 2, 4]}


def regs_est(bm, bn, th, path):
    """Registers per thread: on the TMA path ptxas fills TileLang's launch-bounds budget (240 / 168 for 128 / 256
    threads); on the cp.async path it uses about the fp32 accumulators plus 70 (compiled kernels: 64x128/256 -> 96-106,
    128x128/128 -> 202, 256x128/256 -> 198, 64x64/256 -> 74). The second value says whether the tile fits."""
    acc = bm * bn // th
    if path == "tma_ws":
        budget = min(255, (65536 // (th + 128)) // 8 * 8); return budget, acc + 40 <= budget
    return min(255, acc + 70), acc + 70 <= 255


def legal(a, shape):
    bm, bn, bk, st, th = a["block_M"], a["block_N"], a["block_K"], a["num_stages"], a["threads"]
    if bm > max(16, 2 ** math.ceil(math.log2(shape["M"]))) or bn > shape["N"] or bk > shape["K"]: return False
    if bm * bn // th < 16: return False
    smem = st * (bm + bn) * bk * 2
    if a["c_store"] == "staged": smem = max(smem, bm * bn * 2)
    if smem + 1024 > 101376: return False
    if a["split_k"] > 1 and (a["path"] != "cpasync" or a["c_store"] != "direct" or (shape["K"] // a["split_k"]) % bk): return False
    if a["path"] == "tma_ws" and bm < 64 and shape["M"] >= 64: return False
    return regs_est(bm, bn, th, a["path"])[1]


def columns(shape, timing):
    out = []
    for vals in itertools.product(*DOMAINS.values()):
        a = dict(zip(DOMAINS, vals))
        if not legal(a, shape): continue
        h = {k: a[k] for k in ("block_M", "block_N", "block_K", "num_stages", "threads")} | {"swizzle": 0}
        r, _ = regs_est(a["block_M"], a["block_N"], a["threads"], a["path"])
        try:
            p = gemm_us(h, shape, a["path"], r, timing=timing, c_store=a["c_store"], split=a["split_k"])
        except Exception:
            continue
        out.append({"id": len(out), "t": p["us"], "assign": a})
    return out


def to_template(a: dict, **extra) -> dict:
    t = {k: a[k] for k in ("block_M", "block_N", "block_K", "num_stages", "threads", "c_store", "split_k")}
    t["copy_path"] = "tma" if a["path"] == "tma_ws" else "cpasync"; t["warp_spec"] = 1 if a["path"] == "tma_ws" else 0
    return {**t, **extra}


def shortlist(shape, timing="cold", n=30):
    cols = columns(shape, timing)
    sols = top_k([{"name": "gemm", "columns": [{"id": c["id"], "t": c["t"]} for c in cols]}], k=n)
    by_id = {c["id"]: c for c in cols}
    picks = [by_id[s["pick"]["gemm"]["column"]] for s in sols]
    cands = []
    for c in picks:
        base = to_template(c["assign"])
        cands.append({**base, "_model_us": c["t"]})
        if base["copy_path"] == "cpasync" and base["split_k"] == 1:
            for pf, wp in ((1, 0), (0, 2), (1, 2)):
                if pf < base["num_stages"]: cands.append({**base, "prefetch": pf, "warp_policy": wp, "_model_us": c["t"]})
    return len(cols), cands


def retry(args, tries=4, **kw):
    """The Spark link drops now and then: retry rsync and ssh a few times before giving up."""
    for i in range(tries):
        r = subprocess.run(args, capture_output=True, text=True, **kw)
        if r.returncode == 0: return r
        time.sleep(5 * (i + 1))
    raise RuntimeError(f"{args[0]} failed {tries} times")


def ssh(cmd, timeout=3600):
    return retry(["ssh", "-o", "ConnectTimeout=40", "-o", "ServerAliveInterval=30", "dgx-spark", cmd], timeout=timeout).stdout


def remote(jobs, tag, compile_only):
    for j in jobs: j["compile_only"] = compile_only
    f = ROOT / "runs" / "decide" / f"{tag}_jobs.json"; f.parent.mkdir(parents=True, exist_ok=True); json.dump(jobs, open(f, "w"), indent=1)
    retry(["rsync", "-q", "-e", "ssh -o ConnectTimeout=40", str(f), f"dgx-spark:{REMOTE}/"])
    ssh(f"export PATH=/usr/local/cuda/bin:$PATH; source ~/.venv/bin/activate; cd {REMOTE} && python remote_measure.py {f.name} {tag} > {tag}.log 2>&1")
    out = ROOT / "runs" / "decide" / tag; out.mkdir(parents=True, exist_ok=True)
    retry(["rsync", "-a", "-e", "ssh -o ConnectTimeout=40", f"dgx-spark:{REMOTE.replace('~/', '')}/{tag}/", str(out) + "/"])
    return json.load(open(out / "results.json")), out


def simulate(rec, shape, out_dir, warm):
    from ..gb10.sim.kernel import simulate_kernel
    from ..gb10.sim.ws_validate import kernel_us
    a = rec["assign"]; h = {k: a.get(k) for k in ("block_M", "block_N", "block_K", "num_stages", "threads")} | {"swizzle": a.get("swizzle", 0), "tma": 1 if a.get("copy_path") == "tma" else 0}
    path = out_dir / rec["sass"]
    if a.get("copy_path") == "tma" and a.get("warp_spec", 1):
        return kernel_us(rec["name"], h, shape, warm=warm, sass_path=path)
    return simulate_kernel(path, h, shape, rec["regs"], warm=warm, split=a.get("split_k", 1), kernel=rec["kernel_name"], atomic_c=a.get("split_k", 1) > 1)["us"]


def main():
    M, N, K = (int(x) for x in sys.argv[1:4]); timing = "cold" if "--cold" in sys.argv else "warm"
    shape = {"M": M, "N": N, "K": K}; tag = f"pipe_{M}x{N}x{K}_{timing}"; t0 = time.time()
    ncols, cands = shortlist(shape, timing)
    print(f"{ncols} legal columns priced; shortlist {len(cands)} after expanding path-dependent knobs ({time.time() - t0:.0f} s)", flush=True)
    clean = [{k: v for k, v in c.items() if not k.startswith("_")} for c in cands]
    res, out = remote([{"shape": shape, "assignments": clean}], tag + "_compile", True)
    ok = [r for r in res if r.get("ok")]
    print(f"compiled and correct: {len(ok)} of {len(res)}", flush=True)
    model_of = {cfg_name({k: v for k, v in c.items() if not k.startswith('_')}): c["_model_us"] for c in cands}
    for r in ok:
        try: r["sim_us"] = simulate(r, shape, out, timing == "warm")
        except Exception as e: r["sim_us"] = float("inf"); r["sim_error"] = str(e)[:120]
        r["model_us"] = model_of.get(r["name"], float("inf"))
    by_sim = sorted(ok, key=lambda r: r["sim_us"])[:5]; by_model = sorted(ok, key=lambda r: r["model_us"])[:3]
    final = {r["name"]: r for r in by_sim + by_model}
    print("timing on the GPU:", len(final), "kernels + cuBLASLt", flush=True)
    timed, _ = remote([{"shape": shape, "assignments": [r["assign"] for r in final.values()], "cublaslt": True}], tag + "_time", False)
    key = "cold_us" if timing == "cold" else "warm_us"
    cub = [r for r in timed if r.get("name") == "cuBLASLt" and r["cold"] == (timing == "cold")][0]["us"]
    rows = []
    for r in timed:
        if r.get("name") == "cuBLASLt" or not r.get("ok"): continue
        s = final[r["name"]]; rows.append({"name": r["name"], "model_us": s["model_us"], "sim_us": s["sim_us"], "measured_us": r[key]})
    rows.sort(key=lambda x: x["measured_us"])
    for x in rows: print(f"  {x['name']:64s} model {x['model_us']:7.1f} sim {x['sim_us']:7.1f} measured {x['measured_us']:7.1f}")
    print(f"  cuBLASLt ({timing}): {cub:.1f} us; best found {rows[0]['measured_us']:.1f} us ({(rows[0]['measured_us'] / cub - 1) * 100:+.1f} %); total {time.time() - t0:.0f} s")
    json.dump({"shape": shape, "timing": timing, "columns": ncols, "shortlist": len(cands), "compiled": len(ok), "rows": rows, "cublaslt_us": cub,
               "all_sim": [{k: r.get(k) for k in ("name", "model_us", "sim_us")} for r in ok]}, open(ROOT / "runs" / "decide" / f"{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()

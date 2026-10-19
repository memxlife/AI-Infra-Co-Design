"""Audit of the value evaluator: one block of every kernel we have machine code for, with branches decided by computed
values (loop counters, thread index) and with the older rules alone. The executed mma count must equal block_M x
block_N x K / 2048 (per k-split). python -m agentic.gb10.sim.audit_values"""
from __future__ import annotations
import json, math
from pathlib import Path
from .sm import SM, kernel_sass
from .ws import WSSM
from .kernel import sass_trips, copy_lines

ROOT = Path(__file__).resolve().parents[3]; CAL = ROOT / "calibration" / "spark-6951"
MMA = ("HMMA", "IMMA", "QMMA", "OMMA")


def one(ins, h, K, values, ws=None):
    KT = math.ceil(K / h["block_K"])
    if ws:
        sm = WSSM(ins, h["threads"] + 128, 1, ws["trips"], h["threads"], h["num_stages"], ws["tile"], ws["tma_per_kt"], kt=KT,
                  active_sms=1, ctile_bytes=h["block_M"] * h["block_N"] * 2)
    else:
        sm = SM(ins, h["threads"], 1, sass_trips(ins, h, KT - (h["num_stages"] - 1)), copy_lines=copy_lines(h), ctile_bytes=h["block_M"] * h["block_N"] * 2)
    sm.use_values = values; c = {"n": 0}
    orig = sm.issue
    def issue(w, now, orig=orig):
        if sm.ins[w.pc].base in MMA and sm.ins[w.pc].pred != "@!PT": c["n"] += 1
        return orig(w, now)
    sm.issue = issue
    r = sm.run(limit=20_000_000)
    return c["n"], all(w.done for w in sm.warps), r["cycles"]


def cases():
    ab = json.load(open(CAL / "ablation_cpasync" / "mechanisms.json")) if (CAL / "ablation_cpasync" / "mechanisms.json").exists() else {}
    m = json.load(open(CAL / "ablation" / "mechanisms.json"))
    for name, r in m.items():
        h = r["pick"]
        if (CAL / "sass" / f"{name}__full.sass").exists() and not h.get("tma"):
            yield name, CAL / "sass" / f"{name}__full.sass", h, r["shape"]["K"], None
    for line in open(ROOT / "runs" / "gb10" / "shapes" / "measured.jsonl"):
        r = json.loads(line)
        if r.get("ok"): yield r["name"], ROOT / "runs" / "gb10" / "shapes" / f"{r['name']}.sass", r["holes"], r["shape"]["K"], None
    meta = json.load(open(ROOT / "runs" / "tilelang" / "epilogue" / "sass" / "meta.json"))
    for n, x in meta.items():
        if not x["tma"]:
            yield "epi_" + n, ROOT / "runs" / "tilelang" / "epilogue" / "sass" / f"{n}.sass", {"block_M": x["bm"], "block_N": x["bn"], "block_K": x["bk"], "num_stages": x["stages"], "threads": x["threads"]}, 1024, None
    meta = json.load(open(ROOT / "runs" / "tilelang" / "item5" / "sass" / "meta.json"))
    for n, x in meta.items():
        if x["kind"] == "splitk":
            yield n, ROOT / "runs" / "tilelang" / "item5" / "sass" / f"{n}.sass", {"block_M": x["bm"], "block_N": x["bn"], "block_K": x["bk"], "num_stages": x["stages"], "threads": x["threads"], "_kernel": x["kernel_name"]}, x["K"] // x["split"], None
    for name, r in m.items():
        h = r["pick"]
        if h.get("tma"):
            p = CAL / "sass" / f"tma_{name}__full.sass"
            if p.exists():
                yield "tma_" + name, p, h, r["shape"]["K"], "ws"


if __name__ == "__main__":
    bad = 0; n = 0
    for name, path, h, K, kind in cases():
        ins = kernel_sass(path, kernel=h.get("_kernel", "matmul_kernel")); want = h["block_M"] * h["block_N"] * K // 2048
        ws = None
        if kind == "ws":
            from .sass import main_loop
            waits = [i for i, x in enumerate(ins) if x.op.startswith("SYNCS.PHASECHK") and "TRYWAIT" in x.op]
            tmas = [i for i, x in enumerate(ins) if x.base == "UTMALDG"]
            w0 = max(i for i in waits if i < tmas[0]); w1 = min((i for i in waits if i > tmas[0]), default=len(ins))
            lp = main_loop(ins); body = ins[lp[0]:lp[1] + 1]; warps = h["threads"] // 32
            per_kt = h["block_M"] * h["block_N"] * h["block_K"] / 2048 / warps; u = max(1, round(sum(x.base in MMA for x in body) / per_kt))
            ws = {"trips": max(1, math.ceil(K / h["block_K"]) // u), "tile": (h["block_M"] + h["block_N"]) * h["block_K"] * 2, "tma_per_kt": max(1, sum(1 for i in tmas if w0 < i < w1))}
            want_warps = h["threads"] // 32
        else:
            want_warps = h["threads"] // 32
        on = one(ins, h, K, True, ws); off = one(ins, h, K, False, ws)
        # mma instructions count per warp: block mma / warps
        exp = int(want)  # instructions over all warps of the block: block_M x block_N x K / (16 x 8 x 16)
        ok = on[0] == exp and on[1]
        bad += not ok; n += 1
        print(f"{name:44s} want {exp:6d} | values {on[0]:6d} {'done' if on[1] else 'STUCK'} {on[2]:9.0f} cyc | rules {off[0]:6d} {'done' if off[1] else 'STUCK'} {off[2]:9.0f} cyc {'' if ok else '  <-- MISMATCH'}", flush=True)
    print(f"{n - bad} of {n} kernels execute the expected mma count with computed branches")

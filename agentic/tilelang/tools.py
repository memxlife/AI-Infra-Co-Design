"""The Formulator's three TileLang tools (the Oct 19 brief, section 4): knobs.inventory, template.render, template.legalize.
All code, all returning plain dicts. `legalize` here is the fast precheck of the shared-memory, register and divisibility
rules that the solver's constraint tables encode; the full TileLang pass pipeline runs at compile time on the Spark
(`remote.run_kernel`), and a disagreement between the two is a finding to record.
"""
from __future__ import annotations

import re
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
FAMILIES = HERE / "families"
# the GB10's limits, from the driver on spark-6951 (PINNED.md); the chip contract carries the measured ones
PRODUCER_THREADS = 128     # the warp group TileLang adds on sm_121a (measured: 128 -> 256, 256 -> 384, 512 -> 640 threads)
GB10 = {"smem_per_block": 101376, "smem_per_sm": 102400, "regs_per_sm": 65536, "max_regs_per_thread": 255, "max_threads_per_sm": 1536, "sms": 48}


def inventory(family: str, chip: str = "spark-6951-gb10") -> dict:
    """The knob list of a family: name, kind (int_menu / structural), menu, the primitive it binds, the constraints it enters."""
    y = yaml.safe_load((FAMILIES / family / "knobs.yaml").read_text())
    return {"family": family, "chip": chip, "func": y["func"], "func_kind": y.get("func_kind", "returns_c"), "knobs": y["knobs"], "structural": y.get("structural", []),
            "menus": {k["name"]: k["menu"] for k in y["knobs"]}, "skinny_menus": y.get("skinny_menus", {})}


def menus_for(family: str, shape: dict) -> dict:
    """The menus for a shape: the skinny ones replace block_M when M < 64 (the decode shapes)."""
    inv = inventory(family)
    m = dict(inv["menus"])
    if shape["M"] < 64:
        m.update(inv["skinny_menus"])
    return m


ROUND1 = ("block_M", "block_N", "block_K", "num_stages", "threads", "swizzle")


def grid(family: str, shape: dict, knobs: tuple | None = ROUND1) -> list[dict]:
    """Every combination of the menus of the named knobs (the six round-1 knobs by default; None = every knob, which
    is 560k points for gemm_basic: use sample_legal for that)."""
    import itertools
    m = menus_for(family, shape)
    names = [n for n in m if knobs is None or n in knobs]
    return [dict(zip(names, vals)) for vals in itertools.product(*(m[n] for n in names))]


def sample_legal(family: str, shape: dict, n: int = 200, seed: int = 20261019, menus: dict | None = None, chip: dict | None = None) -> dict:
    """n distinct random legal points of the full space (or of the given menus), with the legal fraction seen."""
    import random
    rng = random.Random(seed)
    m = menus or menus_for(family, shape); names = list(m)
    seen, legal, tries, reasons = set(), [], 0, {}
    while len(legal) < n and tries < 200 * n:
        tries += 1
        h = {k: rng.choice(m[k]) for k in names}; key = tuple(h[k] for k in names)
        if key in seen:
            continue
        seen.add(key)
        r = legalize(family, h, shape, chip)
        if r["ok"]:
            legal.append(h)
        else:
            k = (r["reason"] or "?").split(":")[0]; reasons[k] = reasons.get(k, 0) + 1
    return {"points": legal, "tries": tries, "legal_fraction": len(legal) / max(1, len(seen)), "removed_why": reasons}


ROUND2_DEFAULTS = {"prefetch": 0, "warp_policy": 0, "vec_width": 0, "tma": 1, "warp_spec": 1, "smem_layout": "inferred"}


def jit_args(holes: dict) -> str:
    """The decorator arguments the round-2 switches need: a pass config to drop the producer warp group, a compile
    flag for the register cap. Empty when every switch is at TileLang's default."""
    parts = []
    pc = {}
    if holes.get("warp_spec", 1) == 0:
        pc["tl.disable_warp_specialized"] = True
    if pc:
        parts.append(f"pass_configs={pc!r}")
    if holes.get("maxrreg", 0):
        parts.append(f"compile_flags=['-maxrregcount={int(holes['maxrreg'])}']")
    return ", ".join(parts)


def render(family: str, holes: dict, open_knobs: set | None = None) -> str:
    """TileLang source with the holes filled. A round-1 hole left unfilled stays a named parameter of the function; a
    round-2 knob left unfilled takes TileLang's default (so the example renders byte for byte) unless it is named in
    `open_knobs`, in which case it stays a parameter too."""
    from jinja2 import Template
    y = yaml.safe_load((FAMILIES / family / "knobs.yaml").read_text())
    names = [k["name"] for k in y["knobs"]]
    vals, missing = {}, []
    for n in names:
        if n in holes:
            vals[n] = holes[n]
        elif n in ROUND2_DEFAULTS and n not in (open_knobs or ()):
            vals[n] = ROUND2_DEFAULTS[n]
        else:
            vals[n] = n; missing.append(n)
    vals["jit_args"] = jit_args({k: v for k, v in vals.items() if not isinstance(v, str) or k == "smem_layout"} | {k: holes[k] for k in holes})
    if any(isinstance(vals.get(k), str) and vals[k] == k for k in ("maxrreg", "warp_spec")):
        vals["jit_args"] = ""          # a switch left open cannot be a decorator argument; it is fixed at compile time by the caller
    params = "".join(f", {n}" for n in missing)
    return Template((FAMILIES / family / "kernel.py.j2").read_text()).render(params=params, **vals)


def legalize(family: str, holes: dict, shape: dict, chip: dict | None = None, dtype_bytes: int = 2) -> dict:
    """The precheck. Returns {ok, reason, smem_bytes_est, regs_est, blocks_per_sm_est, tiles, waves}."""
    c = {**GB10, **(chip or {})}
    bm, bn, bk, st, th = (holes.get(k) for k in ("block_M", "block_N", "block_K", "num_stages", "threads"))
    if None in (bm, bn, bk, st, th):
        return {"ok": False, "reason": "every hole must be filled to legalize"}
    if holes.get("swizzle", 0) not in (0, 4, 8, 16):
        return {"ok": False, "reason": f"swizzle panel {holes['swizzle']} is not in the menu"}
    if holes.get("prefetch", 0) >= holes.get("num_stages", 1):
        return {"ok": False, "reason": f"pipeline: a prefetch of {holes['prefetch']} k-steps needs more than {holes['num_stages']} buffers (TileLang's manual schedule infers the depth as max(stage) + 1)"}
    if holes.get("warp_spec", 1) == 0 and holes.get("tma", 1) == 1:
        pass                                                        # TileLang drops TMA without the producer group; the kernel builds, so it is allowed
    smem = st * (bm * bk + bk * bn) * dtype_bytes
    reasons = []
    if smem > c["smem_per_block"]:
        reasons.append(f"shared memory: {st} stages x ({bm}x{bk} + {bk}x{bn}) x {dtype_bytes} B = {smem} B exceeds the {c['smem_per_block']} B a block may have")
    # registers: TileLang launches th + 128 threads (its producer warp group) with __launch_bounds__(th + 128, 1), so ptxas
    # may use floor(65536 / (th + 128)) registers per thread, rounded down to 8 (240, 168, 96 for 128, 256, 512). The FP32
    # accumulators alone take bm*bn/th; the sweep shows about 24 more for addresses and fragments. Over the budget means
    # spills, and the 1024^3 sweep measured those at 5x the time.
    launched = th + (PRODUCER_THREADS if holes.get("warp_spec", 1) else 0)
    budget = c["regs_per_sm"] // launched; budget = min(c["max_regs_per_thread"], budget - budget % 8)
    if holes.get("maxrreg"):
        budget = min(budget, int(holes["maxrreg"]))
    acc_regs = bm * bn // th                     # FP32 accumulators per thread (one 32-bit register each)
    regs = acc_regs + 24
    if regs > budget:
        reasons.append(f"registers: {bm}x{bn} FP32 accumulators over {th} threads need about {regs} registers per thread, more than the "
                       f"{budget} ptxas may use under TileLang's launch bounds ({launched} threads); the kernel would spill")
    if bm * bn // th < 16:
        reasons.append(f"tile too small for the threads: {bm}x{bn} over {th} threads leaves {bm * bn // th} values per thread")
    m, n, k = shape["M"], shape["N"], shape["K"]
    if family == "gemm_splitk":
        sk = holes.get("split_k", 1)
        if sk < 2 or k % sk:
            reasons.append(f"exact-once: K = {k} does not divide into {sk} equal parts")
        elif (k // sk) % bk:
            reasons.append(f"exact-once: a part of K ({k // sk}) is not a whole number of {bk}-slices; the last slice of a part would read into the next part and add it twice")
    waste = 1.0
    for name, tile, ext in (("block_M", bm, m), ("block_N", bn, n), ("block_K", bk, k)):
        waste *= (-(-ext // tile) * tile) / ext        # the padded work over the real work; TileLang predicates the edge tiles
    blocks = max(1, min(c["smem_per_sm"] // max(1, smem + 1024), c["regs_per_sm"] // max(1, budget * launched), c["max_threads_per_sm"] // launched))
    tiles = -(-m // bm) * -(-n // bn) * (holes.get("split_k", 1) if family == "gemm_splitk" else 1)
    return {"ok": not reasons, "reason": "; ".join(reasons) or None, "smem_bytes_est": smem, "regs_est": regs,
            "blocks_per_sm_est": blocks, "tiles": tiles, "waves": -(-tiles // (c["sms"] * blocks)), "padding_waste": round(waste - 1, 4)}

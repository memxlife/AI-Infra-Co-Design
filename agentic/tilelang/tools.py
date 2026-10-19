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
    return {"family": family, "chip": chip, "func": y["func"], "knobs": y["knobs"], "structural": y.get("structural", []),
            "menus": {k["name"]: k["menu"] for k in y["knobs"]}, "skinny_menus": y.get("skinny_menus", {})}


def menus_for(family: str, shape: dict) -> dict:
    """The menus for a shape: the skinny ones replace block_M when M < 64 (the decode shapes)."""
    inv = inventory(family)
    m = dict(inv["menus"])
    if shape["M"] < 64:
        m.update(inv["skinny_menus"])
    return m


def grid(family: str, shape: dict) -> list[dict]:
    """Every combination of the menus, in menu order."""
    import itertools
    m = menus_for(family, shape)
    names = list(m)
    return [dict(zip(names, vals)) for vals in itertools.product(*(m[n] for n in names))]


def render(family: str, holes: dict) -> str:
    """TileLang source with the holes filled; an unfilled hole stays a named parameter of the function (a constexpr)."""
    from jinja2 import Template
    y = yaml.safe_load((FAMILIES / family / "knobs.yaml").read_text())
    names = [k["name"] for k in y["knobs"]]
    missing = [n for n in names if n not in holes]
    vals = {n: (holes[n] if n in holes else n) for n in names}
    params = "".join(f", {n}" for n in missing)
    src = Template((FAMILIES / family / "kernel.py.j2").read_text()).render(params=params, **vals)
    return src


def legalize(family: str, holes: dict, shape: dict, chip: dict | None = None, dtype_bytes: int = 2) -> dict:
    """The precheck. Returns {ok, reason, smem_bytes_est, regs_est, blocks_per_sm_est, tiles, waves}."""
    c = {**GB10, **(chip or {})}
    bm, bn, bk, st, th = (holes.get(k) for k in ("block_M", "block_N", "block_K", "num_stages", "threads"))
    if None in (bm, bn, bk, st, th):
        return {"ok": False, "reason": "every hole must be filled to legalize"}
    if holes.get("swizzle", 0) not in (0, 4, 8, 16):
        return {"ok": False, "reason": f"swizzle panel {holes['swizzle']} is not in the menu"}
    smem = st * (bm * bk + bk * bn) * dtype_bytes
    reasons = []
    if smem > c["smem_per_block"]:
        reasons.append(f"shared memory: {st} stages x ({bm}x{bk} + {bk}x{bn}) x {dtype_bytes} B = {smem} B exceeds the {c['smem_per_block']} B a block may have")
    # registers: TileLang launches th + 128 threads (its producer warp group) with __launch_bounds__(th + 128, 1), so ptxas
    # may use floor(65536 / (th + 128)) registers per thread, rounded down to 8 (240, 168, 96 for 128, 256, 512). The FP32
    # accumulators alone take bm*bn/th; the sweep shows about 24 more for addresses and fragments. Over the budget means
    # spills, and the 1024^3 sweep measured those at 5x the time.
    launched = th + PRODUCER_THREADS
    budget = c["regs_per_sm"] // launched; budget = min(c["max_regs_per_thread"], budget - budget % 8)
    acc_regs = bm * bn // th                     # FP32 accumulators per thread (one 32-bit register each)
    regs = acc_regs + 24
    if regs > budget:
        reasons.append(f"registers: {bm}x{bn} FP32 accumulators over {th} threads need about {regs} registers per thread, more than the "
                       f"{budget} ptxas may use under TileLang's launch bounds ({launched} threads); the kernel would spill")
    if bm * bn // th < 16:
        reasons.append(f"tile too small for the threads: {bm}x{bn} over {th} threads leaves {bm * bn // th} values per thread")
    m, n, k = shape["M"], shape["N"], shape["K"]
    waste = 1.0
    for name, tile, ext in (("block_M", bm, m), ("block_N", bn, n), ("block_K", bk, k)):
        waste *= (-(-ext // tile) * tile) / ext        # the padded work over the real work; TileLang predicates the edge tiles
    blocks = max(1, min(c["smem_per_sm"] // max(1, smem + 1024), c["regs_per_sm"] // max(1, budget * launched), c["max_threads_per_sm"] // launched))
    tiles = -(-m // bm) * -(-n // bn)
    return {"ok": not reasons, "reason": "; ".join(reasons) or None, "smem_bytes_est": smem, "regs_est": regs,
            "blocks_per_sm_est": blocks, "tiles": tiles, "waves": -(-tiles // (c["sms"] * blocks)), "padding_waste": round(waste - 1, 4)}

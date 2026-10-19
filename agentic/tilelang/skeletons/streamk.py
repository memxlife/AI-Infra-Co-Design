"""Skeleton 2 (gemm_streamk): render the family's template for a shape and knobs; the legal menu; a mechanistic estimate.
Time per program = its tiles x T_tile with T_tile from the cp.async constants of the (tile, threads) nearest in the
gemm_pipelined ablation (same mma_sync + cp.async body), the data-parallel loop at dp_stages = 1 priced with no overlap
(compute + load per k-tile), plus the stream-K fix-up (fp32 atomics on the partial tiles)."""
from __future__ import annotations
import itertools, json, math
from pathlib import Path
import yaml
from jinja2 import Template
from ..model.mech import CHIP, SMS
ROOT = Path(__file__).resolve().parents[3]; FAM = ROOT / "agentic" / "tilelang" / "families" / "gemm_streamk"
TPL = Template((FAM / "kernel.py.j2").read_text()); KNOBS = yaml.safe_load((FAM / "knobs.yaml").read_text())


def render(shape: dict, h: dict) -> str:
    return TPL.render(M=shape["M"], N=shape["N"], K=shape["K"], **h)


def legal(shape: dict, h: dict) -> tuple[bool, str]:
    M, N, K = shape["M"], shape["N"], shape["K"]; bm, bn, bk, st, dp, th, pr = (h[k] for k in ("block_M", "block_N", "block_K", "num_stages", "dp_stages", "threads", "programs"))
    if M % bm or N % bn or K % bk: return False, "divisibility"
    smem = (st + dp) * (bm + bn) * bk * 2 + 1024
    if smem > 99 * 1024: return False, f"smem {smem}"
    acc = bm * bn / th
    if acc < 16: return False, "fewer than 16 accumulators per thread"
    if acc + 24 > (65536 // th) // 8 * 8: return False, "registers"
    per_sm = min(2, (99 * 1024) // smem, 65536 // (th * int(acc + 24)))
    if pr > SMS * per_sm: return False, f"programs {pr} > {SMS} x {per_sm}"
    if bm * bn * 2 < th * 8 * 2: return False, "tile smaller than the warps' mma footprint"
    return True, ""


def grid(shape: dict) -> list[dict]:
    m = {k["name"]: list(k["menu"]) for k in KNOBS["knobs"]}; names = list(m); out = []
    for vals in itertools.product(*(m[n] for n in names)):
        h = dict(zip(names, vals)); ok, why = legal(shape, h)
        if ok: out.append(h)
    return out


def partition(shape: dict, h: dict) -> dict:
    M, N, K = shape["M"], shape["N"], shape["K"]; bm, bn, bk, pr = h["block_M"], h["block_N"], h["block_K"], h["programs"]
    tiles = (M // bm) * (N // bn); ipt = K // bk
    sk = tiles % pr
    if tiles - sk > pr: sk += pr
    dp = tiles - sk
    return {"tiles": tiles, "iters_per_tile": ipt, "streamk_tiles": sk, "dp_tiles": dp, "dp_per_program": max(dp // pr, 1) if dp else 0, "sk_iters_per_program": sk * ipt / pr}


def nearest_block(h: dict) -> dict | None:
    fam = CHIP["families"]["gemm_pipelined"]["blocks"]
    want = (h["block_M"], h["block_N"], h["block_K"], h["threads"]); best, bd = None, 1e9
    for t in fam.values():
        bm, bn, bk = (int(x) for x in t["tile"].split("x")); d = abs(math.log(bm / want[0])) + abs(math.log(bn / want[1])) + abs(math.log(bk / want[2])) + abs(math.log(t["threads"] / want[3]))
        if d < bd: best, bd = t, d
    return best


def estimate(shape: dict, h: dict) -> dict:
    fam = CHIP["families"]["gemm_pipelined"]; k = CHIP; t = nearest_block(h); p = partition(shape, h)
    bm, bn, bk = h["block_M"], h["block_N"], h["block_K"]; flops_kt = 2.0 * bm * bn * bk; bytes_kt = (bm + bn) * bk * 2
    rate = t["tensor_rate_TFs_per_SM"] if t and t["tensor_rate_TFs_per_SM"] else fam["tensor_rate_TFs_per_SM"]["median"]
    # mma_sync issue scales with the warps that share the tile only up to the per-SM tensor rate: the nearest ablated (tile, threads)
    compute_kt = flops_kt / (rate * 1e12) * 1e6
    load_alone = bytes_kt / ((t["load_rate_GBs_1block"] if t and t["load_rate_GBs_1block"] else fam["load_rate_alone_GBs_per_block"]["median"]) * 1e9) * 1e6
    resident = min(h["programs"], SMS); load_under = max(load_alone, bytes_kt / (k["load_rate_chip_GBs"]["L2_resident_48_blocks"] * 1e9 / resident) * 1e6)
    sync = fam["sync_per_ktile_us"]["median"]; store = bm * bn * 2 / 1024 * fam["store_us_per_KB_per_block"]["median"]
    t_kt_pipelined = max(compute_kt, load_under) + sync
    t_kt_dp = (compute_kt + load_under + sync) if h["dp_stages"] == 1 else t_kt_pipelined     # no overlap at one stage
    ipt = p["iters_per_tile"]
    sk_us = p["sk_iters_per_program"] * t_kt_pipelined + (p["streamk_tiles"] * ipt / h["programs"] / ipt + 1) * (store * 4)    # fp32 atomics ~4x the fp16 store per partial tile
    dp_us = p["dp_per_program"] * (ipt * t_kt_dp + store + fam["block_setup_us"]["median"])
    programs_per_sm = math.ceil(h["programs"] / SMS)
    total = k["launch_us"]["value"] + programs_per_sm * (sk_us + dp_us)
    return {"expected_us": total, "sk_us": sk_us, "dp_us": dp_us, "t_kt_pipelined": t_kt_pipelined, "t_kt_dp": t_kt_dp, "partition": p, "nearest": t["tile"] + f"@{t['threads']}" if t else None}

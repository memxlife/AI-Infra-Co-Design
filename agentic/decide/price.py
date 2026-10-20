"""Pricing: the time of a kernel configuration from the GB10 analytical model (agentic/gb10/model.py) plus terms for
decisions that model does not carry yet, each from a measured mechanism (calibration/spark-6951/gb10.json). Used for
every legal column; the simulator re-ranks the shortlist (agentic/decide/pipeline.py).

Terms added here (microseconds):
  c_store   direct: the C tile leaves through the SM pair's store port as half-sector 4-byte stores (15 cycles per
            warp instruction of 128 B per pair) or the chip limit, whichever is slower; staged: stmatrix into shared
            memory, then a TMA store at 33 B/clk per pair (chip 482 B/clk) plus 457 cycles; only the last wave's
            stores are exposed, earlier ones overlap the next wave's loads
  split_k   fp32 C added with atomics from mma accumulators: chip 225 B/clk for split x C bytes; after a flush the
            first split reads each C line from DRAM
  launch    2.6 us per kernel from a stream, 0.43 us (+0.43 ns per block) inside a CUDA graph
  l2_pin    the bytes of a tensor held in the persisting L2 come at the L2 rate instead of DRAM (8 MB: 37.7 -> 8.0 us)"""
from __future__ import annotations

import math
from ..gb10 import model as G

P = G.C; SIM = None


def _sim():
    global SIM
    if SIM is None:
        import json
        SIM = json.load(open(G.ROOT / "calibration" / "spark-6951" / "gb10.json"))["sim"]["value"]
    return SIM


def store_us(h: dict, shape: dict, waves_last_blocks: int, R: int) -> dict:
    """Exposed C-store time of the last wave, direct and staged, for one SM holding R blocks."""
    S = _sim(); F = 2.54e3; bm, bn = min(h["block_M"], shape["M"]), min(h["block_N"], shape["N"])
    c_bytes = bm * bn * 2 * R; active = min(48, waves_last_blocks)
    direct = max(c_bytes / 128 * S["stg_tpc_clk_per_instr_4B"] * 2, c_bytes * active / S["stg_chip_B_per_clk_4B"]) / F
    staged = (c_bytes / 512 * S["smem_array_clk_per_512B"] + max(c_bytes * 2 / S["tma_st_tpc_B_per_clk"], c_bytes * active / S["tma_st_chip_B_per_clk"])
              + S["tma_st_fixed_clk"]) / F
    return {"direct": direct, "staged": staged}


def splitk_extra_us(h: dict, shape: dict, split: int, cold: bool) -> float:
    if split <= 1: return 0.0
    S = _sim(); F = 2.54e3; c32 = shape["M"] * shape["N"] * 4
    t = split * c32 / S["red_chip_B_per_clk_frag"] / F
    if cold: t += c32 / (P["dram_read_GBs"] * 1e9) * 1e6
    return t


def launch_us(blocks: int, graph: bool) -> float:
    return 0.43 + 0.00043 * blocks if graph else 2.6


def pinned_dram_saving_us(pinned_bytes: float, dram_GBs: float = 237.0, l2_GBs: float = 1044.0) -> float:
    """Time a kernel saves when pinned_bytes of what it reads come from the persisting L2 instead of DRAM (probe l2persist.cu)."""
    return pinned_bytes / (dram_GBs * 1e9) * 1e6 - pinned_bytes / (l2_GBs * 1e9) * 1e6


def gemm_us(h: dict, shape: dict, path: str, regs: int, timing: str = "warm", c_store: str = "direct", split: int = 1,
            graph: bool = False) -> dict:
    """The analytical model's time with the C store, split-K and launch terms swapped for the chosen values."""
    threads = h["threads"]; ws = path == "tma_ws"
    smem = h["num_stages"] * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
    # split-K runs split x the tiles, each over K/split: the same work as a GEMM split x taller over K/split (A's bytes
    # are the same; B's slices differ per split, which this shape reads once: the simulator re-ranks with the real ones)
    sh = {"M": shape["M"] * split, "N": shape["N"], "K": shape["K"] // split} if split > 1 else dict(shape)
    k = {"shape": sh, "holes": h, "path": path, "consumer_threads": threads, "launched_threads": threads + (128 if ws else 0),
         "regs": regs, "smem_bytes": smem, "timing": timing}
    p = G.predict(k)
    tiles = p["tiles"]; R = p["blocks_per_sm"]
    waves = math.ceil(tiles / (48 * R)); last = tiles - (waves - 1) * 48 * R
    st = store_us(h, shape, last, R)
    model_store = h["block_M"] * h["block_N"] * 2 / 1024 * P["store_ns_per_KB"] * 1e-3 * R
    # the model's own per-wave store term is replaced by the chosen path's exposed cost; split-K's C goes by atomics
    t = p["us"] - model_store + (0.0 if split > 1 else st[c_store])
    if split > 1:
        # the stand-in shape reads only 1/split of B; every split reads its own slice, so all of A and B come once
        floor = (shape["M"] * shape["K"] + shape["K"] * shape["N"]) * 2 / (P["dram_read_GBs"] * 1e9) * 1e6 + P["launch_us"]
        t = max(t, floor) if (timing == "cold" or (shape["M"] * shape["K"] + shape["K"] * shape["N"]) * 2 > P["l2_effective_bytes"]) else t
    t += splitk_extra_us(h, shape, split, timing == "cold")
    t += launch_us(tiles, graph) - P["launch_us"]
    return {"us": t, "model": p, "store": st, "tiles": tiles, "waves": waves, "blocks_per_sm": R}

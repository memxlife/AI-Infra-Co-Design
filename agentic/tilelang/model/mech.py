"""A mechanistic timing model for TileLang GEMM on the GB10 (round 3, step 2c). Every constant comes from
calibration/spark-6951/chip_tilelang.json, where each carries a `source` naming the ablation probe or counter that
measured it alone. No regression, no envelope, no scale. Two readings of one model:
  expected(x) = launch + block_issue + waves_eff * T_block, with T_block = setup + KT * (max(compute, load) + sync) + store
  floor(x)    = the same with the DRAM ceiling as a max: max(expected per-SM time, bytes / dram_rate)
The residual (measured - expected) is reported with a name, never absorbed."""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CHIP = json.load(open(ROOT / "calibration" / "spark-6951" / "chip_tilelang.json"))
SMS = 48


def constants() -> dict:
    c = CHIP
    return {"launch_us": c["launch_us"]["value"], "block_setup_us": c["block_setup_us"]["value"], "block_issue_us": c["block_launch_rate_us_per_block"]["value"],
            "store_us_per_KB": c["store_us_per_KB_per_block"]["value"], "tensor_TFs_per_SM": c["tensor_rate_TFs_per_SM"]["value"],
            "load_alone_GBs_per_block": c["load_rate_alone_GBs_per_block"]["value"], "l2_chip_GBs": c["load_rate_chip_GBs"]["L2_resident_48_blocks"],
            "dram_GBs": c["dram_rate_GBs"]["value"], "sync_per_ktile_us": c["sync_per_ktile_us"]["value"]}


def traffic_bytes(h: dict, shape: dict, panel_measured: dict | None = None) -> float:
    """DRAM bytes: A and B once when they fit the L2 (16.8 MB effective from the microbenchmark; chip_tilelang notes the measured
    panel counts at 4096^3), else B once per panel of rows and A once per group of columns a wave covers."""
    M, N, K = shape["M"], shape["N"], shape["K"]; bm, bn = h["block_M"], h["block_N"]
    a, b, c = M * K * 2, K * N * 2, M * N * 2
    if a + b <= 16.78e6:
        return a + b + c
    tiles_m, tiles_n = math.ceil(M / bm), math.ceil(N / bn)
    p = float(h.get("swizzle", 0) or max(1.0, SMS / tiles_n)); p = max(1.0, min(p, tiles_m))
    cols_per_wave = max(1.0, SMS / p)
    a_panel = p * bm * K * 2
    a_reads = 1.0 if a_panel <= 0.75 * 16.78e6 else max(1.0, tiles_n / cols_per_wave)
    return a * a_reads + b * max(1.0, tiles_m / p) + c


def predict(h: dict, shape: dict, pick_name: str | None = None) -> dict:
    """h: the knobs; shape: M, N, K. If pick_name names an ablated pick, that pick's own per-tile measurements are used for its
    tile (compute per k-tile, load alone, store, sync); otherwise the chip-wide medians."""
    k = constants(); blk = CHIP["blocks"].get(pick_name or "", None)
    M, N, K = shape["M"], shape["N"], shape["K"]; bm, bn, bk, st = h["block_M"], h["block_N"], h["block_K"], h["num_stages"]
    tiles = math.ceil(M / bm) * math.ceil(N / bn); KT = math.ceil(K / bk)
    flops_kt = 2.0 * bm * bn * bk; bytes_kt = (bm + bn) * bk * 2; tile_kb = bm * bn * 2 / 1024
    if blk and blk["tile"] == f"{bm}x{bn}x{bk}":
        compute_kt = blk["compute_per_ktile_us"]; load_alone_kt = blk["load_per_ktile_alone_us"]; store = blk["store_per_block_us"]; sync_kt = blk["overlap_sync_per_ktile_us"]; setup = blk["block_setup_us"]; src = "the pick's own ablation"
    else:
        compute_kt = flops_kt / (k["tensor_TFs_per_SM"] * 1e12) * 1e6; load_alone_kt = bytes_kt / (k["load_alone_GBs_per_block"] * 1e9) * 1e6
        store = tile_kb * k["store_us_per_KB"]; sync_kt = k["sync_per_ktile_us"]; setup = k["block_setup_us"]; src = "chip-wide medians"
    resident = min(tiles, SMS)
    # loads under load: one block alone, then the chip-wide L2 rate shared by the resident blocks, then the DRAM rate shared
    dram_bytes = traffic_bytes(h, shape); grid_bytes = tiles * KT * bytes_kt
    load_kt_under_load = max(load_alone_kt, bytes_kt / (k["l2_chip_GBs"] * 1e9 / resident) * 1e6, (dram_bytes / grid_bytes) * bytes_kt / (k["dram_GBs"] * 1e9 / resident) * 1e6)
    t_block = setup + KT * (max(compute_kt, load_kt_under_load) + sync_kt) + store
    waves_eff = max(1.0, tiles / SMS)                       # the scheduler refills as blocks finish (mechanism 8): fractional, not ceil
    expected = k["launch_us"] + tiles * k["block_issue_us"] + waves_eff * t_block
    floor = max(expected, k["launch_us"] + dram_bytes / (k["dram_GBs"] * 1e9) * 1e6)
    return {"expected_us": expected, "floor_us": floor, "T_block_us": t_block, "waves_eff": waves_eff, "KT": KT, "tiles": tiles, "constants_from": src,
            "parts_us": {"launch": k["launch_us"], "block_issue": tiles * k["block_issue_us"], "setup": setup, "compute_per_block": KT * compute_kt, "load_per_block_under_load": KT * load_kt_under_load,
                         "sync_per_block": KT * sync_kt, "store": store, "dram_ceiling": dram_bytes / (k["dram_GBs"] * 1e9) * 1e6}, "dram_MB": dram_bytes / 1e6}

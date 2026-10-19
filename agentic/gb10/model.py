"""A functional timing model of the GB10's GPU for tiled GEMM kernels (TileLang's cp.async and TMA warp-specialized
lowerings). Each term is one hardware unit doing its job at its measured or designed rate; constants come from
calibration/spark-6951/gb10.json, where each names its source (spec, probe, ablation). Nothing is fitted to kernel times.

The model answers, per kernel: how many blocks share an SM, what each unit spends per k-tile, which unit binds
(the bottleneck), and the time. Its purpose is ranking candidates and naming the bottleneck, not absolute accuracy.

Units per SM and k-tile (R blocks resident, each doing one k-tile of bm x bn x bk):
  tensor   R * 2 bm bn bk / (1024 FLOP/clk * f * eff(warps per sub-partition))
  smem     R * (ldmatrix reads + tile writes) / (128 B/clk * f)
  port     R * tile bytes / min(port 80.7 GB/s, L2 aggregate / active SMs)
  latency  latency / (stages - 1): a k-tile's data is issued stages-1 tiles ahead (Little's law)
  bubble   when one block owns the SM: barrier and refill time no other block fills
Kernel: launch + block issue + per wave (blocks on an SM at once) the per-SM block time; a DRAM floor from the
traffic the tile order causes; under TileLang's do_bench, a dirty-L2 write-back cost."""
from __future__ import annotations
import json, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
C = {k: v["value"] for k, v in json.load(open(ROOT / "calibration" / "spark-6951" / "gb10.json")).items() if not k.startswith("_")}
F = C["clock_GHz"] * 1e9
SMS = C["sms"]


def warp_grid(bm: int, bn: int, warps: int, policy: int = 0) -> tuple[int, int]:
    """How TileLang splits the consumer warps over the tile (GemmWarpPolicy): 0 Square = the warp tile closest to square,
    1 FullRow = all warps along M, 2 FullCol = all warps along N (rows of 16, cols of 8)."""
    if policy == 1 and bm % (warps * 16) == 0: return warps, 1
    if policy == 2 and bn % (warps * 8) == 0: return 1, warps
    best = None
    for wm in range(1, warps + 1):
        if warps % wm: continue
        wn = warps // wm
        if bm % (wm * 16) or bn % (wn * 8): continue
        tm, tn = bm // wm, bn // wn; score = abs(math.log(tm / tn))
        if best is None or score < best[0]: best = (score, wm, wn)
    return (best[1], best[2]) if best else (1, warps)


def occupancy(k: dict) -> dict:
    """Blocks per SM from the driver's limits: registers (per warp, in allocation units), shared memory, threads, blocks."""
    warps = math.ceil(k["launched_threads"] / 32)
    regs_warp = math.ceil(k["regs"] * 32 / C["reg_alloc_unit_per_warp"]) * C["reg_alloc_unit_per_warp"]
    lim = {"regs": C["regs_per_sm"] // max(1, warps * regs_warp),
           "smem": C["smem_per_sm_bytes"] // (k["smem_bytes"] + C["smem_reserved_per_block_bytes"]),
           "threads": C["max_threads_per_sm"] // k["launched_threads"], "blocks": C["max_blocks_per_sm"]}
    r = max(1, min(lim.values()))
    return {"blocks_per_sm": r, "limit": min(lim, key=lim.get), "limits": lim}


def launch_order(tiles_m: int, tiles_n: int, panel: int):
    """(row, col) of each block in launch order: TileLang's rasterization2DRow (serpentine over panels of `panel` tile
    rows, src/tl_templates/cuda/threadblock_swizzle.h) or, without use_swizzle, row-major with columns fastest."""
    n = tiles_m * tiles_n
    if not panel:
        for i in range(n): yield i // tiles_n, i % tiles_n
        return
    psize = panel * tiles_n; total = -(-n // psize)
    for i in range(n):
        off, pidx = i % psize, i // psize
        stride = panel if pidx + 1 < total else (n - pidx * psize) // tiles_n
        col = tiles_n - 1 - off // stride if pidx & 1 else off // stride
        yield off % stride + pidx * panel, col


_traffic_cache = {}


def dram_bytes(h: dict, shape: dict, resident_blocks: int = SMS) -> float:
    return dram_waves(h, shape, resident_blocks)[0]


def dram_waves(h: dict, shape: dict, resident_blocks: int = SMS) -> tuple[float, list]:
    """DRAM traffic from the L2's reuse, simulated: blocks run in launch order, `resident_blocks` at a time (one wave);
    each block streams its A row panel (block_M x K) and B column panel (K x block_N). A panel already in the L2 (LRU over
    the measured effective capacity) is a hit; blocks of the same wave share their panels. Plus the part of C that
    overflows the L2 (write-back: the rest is still dirty in the L2 at the end)."""
    M, N, K = shape["M"], shape["N"], shape["K"]; bm, bn = h["block_M"], h["block_N"]; panel = int(h.get("swizzle", 0) or 0)
    key = (M, N, K, bm, bn, panel, resident_blocks)
    if key in _traffic_cache: return _traffic_cache[key]
    per_wave = []
    tiles_m, tiles_n = math.ceil(M / bm), math.ceil(N / bn); a_sz, b_sz = bm * K * 2, bn * K * 2
    cap = C["l2_effective_bytes"]; lru = {}; used = 0.0; clock = 0; reads = 0.0
    order = list(launch_order(tiles_m, tiles_n, panel))
    for w in range(0, len(order), resident_blocks):
        wave = order[w:w + resident_blocks]; before = reads
        need = {("A", r) for r, _ in wave} | {("B", c) for _, c in wave}
        for key_ in need:
            sz = a_sz if key_[0] == "A" else b_sz
            if key_ not in lru: reads += sz; used += sz
            clock += 1; lru[key_] = clock
        # the L2 holds at most its capacity: least recently touched panels go first, this wave's included (a wave
        # whose panels exceed the L2 still shares them among its own blocks, but cannot keep them for the next wave)
        while used > cap and lru:
            victim = min(lru, key=lru.get)
            used -= a_sz if victim[0] == "A" else b_sz; del lru[victim]
        per_wave.append(reads - before)
    # C: the L2 is write-back, so the last L2-full of C is still dirty in the L2 when the kernel ends
    c_over = max(0.0, M * N * 2 - cap)
    per_wave = [b + c_over / len(per_wave) for b in per_wave]
    out = (reads + c_over, per_wave)
    _traffic_cache[key] = out
    return out


_kslice_cache = {}


def dram_waves_kslice(h: dict, shape: dict, resident_blocks: int = SMS) -> tuple[float, list]:
    """DRAM traffic at k-slice granularity: the blocks of a wave (launch order) walk k in lockstep; at each k-step each
    block reads its A slice (block_M x block_K) and B slice (block_K x block_N). Slices pass through an LRU of the L2's
    measured effective capacity; a slice already there is a hit (neighbours in a wave share A rows and B columns). Between
    waves only what is still resident carries over. Checked against ncu's L2 read misses (agentic/gb10, Oct 20)."""
    from collections import OrderedDict
    M, N, K = shape["M"], shape["N"], shape["K"]; bm, bn, bk = h["block_M"], h["block_N"], h["block_K"]; panel = int(h.get("swizzle", 0) or 0)
    key = (M, N, K, bm, bn, bk, panel, resident_blocks)
    if key in _kslice_cache: return _kslice_cache[key]
    tiles_m, tiles_n, KT = math.ceil(M / bm), math.ceil(N / bn), math.ceil(K / bk)
    a_sz, b_sz = bm * bk * 2, bk * bn * 2; cap = C["l2_effective_bytes"]
    lru = OrderedDict(); used = 0.0; per_wave = []; total = 0.0
    order = list(launch_order(tiles_m, tiles_n, panel))
    for w in range(0, len(order), resident_blocks):
        wave = order[w:w + resident_blocks]; rows = sorted({r for r, _ in wave}); cols = sorted({c for _, c in wave}); reads = 0.0
        for k in range(KT):
            for kind, ids, sz in (("A", rows, a_sz), ("B", cols, b_sz)):
                for i in ids:
                    t = (kind, i, k)
                    if t in lru: lru.move_to_end(t)
                    else:
                        lru[t] = sz; used += sz; reads += sz
                        while used > cap: _, v = lru.popitem(last=False); used -= v
        per_wave.append(reads); total += reads
    c_over = max(0.0, M * N * 2 - cap)
    per_wave = [b + c_over / len(per_wave) for b in per_wave]
    out = (total + c_over, per_wave); _kslice_cache[key] = out
    return out


def dirty_extra_us(read_bytes: float) -> float:
    pts = C["dirty_l2_extra_us"]; mb = read_bytes / 1048576
    if mb <= pts[0][0]: return pts[0][1] * mb / pts[0][0]
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if mb <= x1: return y0 + (y1 - y0) * (mb - x0) / (x1 - x0)
    return pts[-1][1]


def bank_conflict_ways(row_stride_bytes: int) -> int:
    """ldmatrix reads 8 rows of 16 bytes per 8x8 matrix. Shared memory has 32 banks of 4 bytes, so a 16-byte row covers
    a quarter of them; rows whose addresses fall on the same quarter serialize. Ways = 8 / distinct quarters touched."""
    quarters = {(r * row_stride_bytes // 16) % 8 for r in range(8)}
    return 8 // len(quarters)


def per_ktile(k: dict, R: int, active_sms: int) -> dict:
    """Time per k-tile on one SM holding R blocks, each unit separately (us)."""
    h = k["holes"]; bm, bn, bk, st = h["block_M"], h["block_N"], h["block_K"], h["num_stages"]
    warps = k["consumer_threads"] // 32; wps = max(1, round(R * warps / C["smsp_per_sm"]))
    eff = C["tensor_eff_gemm_loop"][str(min(4, 1 if wps < 2 else 2 if wps < 4 else 4))]
    if wps < 1: eff *= wps
    tile_bytes = (bm + bn) * bk * 2
    wm, wn = warp_grid(bm, bn, warps, h.get("warp_policy", 0) or 0)
    # TileLang's inferred layout and make_swizzled_layout are conflict-free for ldmatrix; a linear layout keeps the
    # row stride: A (block_M x block_K, row-major) rows are block_K*2 bytes apart, B (block_K x block_N, read with
    # ldmatrix.trans) rows are block_N*2 bytes apart
    ca, cb = (bank_conflict_ways(bk * 2), bank_conflict_ways(bn * 2)) if h.get("smem_layout") == "linear" else (1, 1)
    smem_bytes = bm * bk * 2 * wn * ca + bn * bk * 2 * wm * cb + tile_bytes
    if k.get("dram_kt_bytes", 0) > 0:
        # every k-tile has a part in DRAM: it waits a DRAM round trip plus that part's transfer at this SM's share of the
        # DRAM rate; stages-1 tiles in flight overlap that many waits (Little's law per SM)
        wait, xfer = C["dram_little_latency_us"], k["dram_kt_bytes"] / (k["dram_share_GBs"] * 1e9) * 1e6
    else:
        wait, xfer = C["lat_l2_ns"] * 1e-3, 0.0
    lat = wait + xfer
    t = {"tensor": R * 2.0 * bm * bn * bk / (C["tensor_flop_per_clk_per_sm"] * F * eff) * 1e6,
         "smem": R * smem_bytes / (C["smem_bytes_per_clk_per_sm"] * F) * 1e6,
         "latency": lat / max(1, st - 1), "latency_is_transfer": xfer >= wait}
    compute = max(t["tensor"], t["smem"])
    if k["path"] == "tma_ws":
        # the producer warp's TMA loads overlap the consumers' mma; the mbarrier hand-off costs a little per k-tile
        rate = min(C["tma_r_GBs_alone"], C["tma_l2_aggregate_GBs"] / max(1, active_sms)) * 1e9
        t["port"] = R * tile_bytes / rate * 1e6; t["bubble"] = C["bubble_us_per_ktile_tma_ws"]
        t["kt"] = max(compute, t["port"], t["latency"]) + t["bubble"]
    else:
        # the computing warps issue the cp.async, wait and pass two barriers each k-tile (t0, serial with that block's mma);
        # the copies stream at the loader rate. One block on the SM: t0 adds to everything. Two or more: t0 overlaps
        # another block's mma, and the SM's loads (all blocks) share the loader rate and the chip's L2 share.
        t0 = C["cpasync_t0_us"]; own = t0 + tile_bytes / (C["cpasync_r_GBs"] * 1e9) * 1e6
        shared = R * tile_bytes / (C["cpasync_l2_aggregate_GBs"] / max(1, active_sms) * 1e9) * 1e6
        t["port"] = max(own - t0, shared)
        # the loader alone (what the probe measures): one block pays t0 plus its copies; with two or more blocks one block's
        # loop overlaps the others and the SM's copies share the per-SM rate (measured with two resident, t0 included)
        t["bubble"] = t0 if R == 1 else 0.0
        # within one block the same warps issue the copies and the mma: they overlap partly, by the measured law
        # (compute^p + loads^p)^(1/p). Across blocks (different warps) the overlap is full: the SM's k-step is the largest
        # of all blocks' mma, all blocks' copies at the shared rate, and one block's own chain
        pp = C["cpasync_overlap_p"]; compute_one = compute / R
        # the block's own copy work: its loop (t0 + copies), or, when its k-tiles stream from DRAM, the wait for them
        feed = max(own, t["latency"])
        chain = (compute_one ** pp + feed ** pp) ** (1 / pp)
        t["kt"] = max(compute, shared, chain)
    t["spill"] = spill_us(k, R, active_sms); t["kt"] += t["spill"]
    return t


def spill_us(k: dict, R: int, active_sms: int) -> float:
    """Registers that do not fit (the launch-bounds budget; on the TMA path the producer group's 128 threads shrink it)
    live in local memory: each k-tile every thread stores and reloads them. The traffic goes through the SM's port, or
    to DRAM when the chip's local working set (spill bytes x threads x resident blocks) outgrows the L2. ptxas reports a
    static byte count, so this ranks spilling kernels (always far behind) rather than timing them."""
    sp = k.get("spill_bytes") or 0
    if sp <= 0: return 0.0
    traffic = 2.0 * sp * k["launched_threads"] * R
    working = sp * k["launched_threads"] * R * active_sms
    rate = C["port_GBs_per_sm"] * 1e9 if working <= C["l2_effective_bytes"] else C["dram_read_GBs"] * 1e9 / active_sms
    return traffic / rate * 1e6


def predict(k: dict) -> dict:
    """k: {shape, holes, path ('tma_ws' | 'cpasync'), consumer_threads, launched_threads, regs, smem_bytes, timing ('warm' | 'do_bench')}."""
    h, s = k["holes"], k["shape"]; M, N, K = s["M"], s["N"], s["K"]
    tiles = math.ceil(M / h["block_M"]) * math.ceil(N / h["block_N"]); KT = math.ceil(K / h["block_K"])
    occ = occupancy(k); Rmax = occ["blocks_per_sm"]
    traffic, wave_dram = dram_waves(h, s, SMS * Rmax); loads_total = tiles * KT * (h["block_M"] + h["block_N"]) * h["block_K"] * 2
    cold = k.get("timing") in ("do_bench", "cold")
    resident_fits = s["M"] * s["K"] * 2 + s["K"] * s["N"] * 2 <= C["l2_effective_bytes"]
    miss = (traffic / loads_total) if (cold or not resident_fits) else 0.0
    k = {**k, "miss_frac": min(1.0, miss)}
    # under a warm timing loop the operands of an L2-resident problem are already there: no DRAM reads
    if not cold and resident_fits: wave_dram = [0.0] * len(wave_dram)
    store = h["block_M"] * h["block_N"] * 2 / 1024 * C["store_ns_per_KB"] * 1e-3
    extra_total = dirty_extra_us(traffic) if k.get("timing") == "do_bench" else 0.0
    left, t, parts, waves, dram_bound_us, dram_binds_us = tiles, C["launch_us"] + tiles * C["block_issue_us"], None, 0, 0.0, 0.0
    unit_time = {}
    while left > 0:
        n = min(left, SMS * Rmax); R = math.ceil(n / SMS); act = min(SMS, n)
        wb = wave_dram[waves] if waves < len(wave_dram) else 0.0
        seg = min(h["block_K"], h["block_N"]) * 2 if s["M"] >= 256 else h["block_N"] * 2
        pts = C["dram_peak_by_segment_GBs"]; peak = pts[-1][1]
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if seg <= x1: peak = y0 + (y1 - y0) * (math.log(max(seg, x0)) - math.log(x0)) / (math.log(x1) - math.log(x0)); break
        # a k-tile streams from DRAM when this wave's misses cover most of its k-tiles (B streamed once at small M,
        # or reuse lost at large shapes): bytes per k-tile per SM against the SM's share of the DRAM rate
        kt_dram = wb / max(1, KT * act)
        k_w = {**k, "dram_kt_bytes": kt_dram if kt_dram > 0.25 * R * h["block_K"] * min(h["block_M"], h["block_N"]) * 2 else 0.0, "dram_share_GBs": peak / act}
        pk = per_ktile(k_w, R, act)
        # pipeline fill: a wave's first k-tile must arrive (one latency plus one tile through the port) before its mma starts
        lat = C["dram_little_latency_us"] if (cold or not resident_fits) else C["lat_l2_ns"] * 1e-3
        fill = lat + R * (h["block_M"] + h["block_N"]) * h["block_K"] * 2 / (min(C["port_GBs_per_sm"], C["l2_aggregate_GBs"] / act) * 1e9) * 1e6
        t_wave_sm = C["block_setup_us"] + fill + KT * pk["kt"] + R * store
        rate = peak
        t_wave_dram = wb / (rate * 1e9) * 1e6 + (extra_total * wb / traffic if traffic else 0.0)
        t += max(t_wave_sm, t_wave_dram); dram_bound_us += max(0.0, t_wave_dram - t_wave_sm)
        if t_wave_dram > t_wave_sm: u = "dram"
        else:
            cand = {x: pk[x] for x in ("tensor", "smem", "port", "latency")}
            u = max(cand, key=cand.get)
            if u == "latency" and pk["latency_is_transfer"]: u = "dram"
            if pk["bubble"] > cand[max(cand, key=cand.get)]: u = "bubble"
            if pk.get("spill", 0) > cand[max(cand, key=cand.get)]: u = "spill"
        unit_time[u] = unit_time.get(u, 0.0) + max(t_wave_sm, t_wave_dram)
        if t_wave_dram > t_wave_sm: dram_binds_us += t_wave_dram
        parts = parts or pk; left -= n; waves += 1
    total = t
    dram_floor = C["launch_us"] + traffic / (C["dram_read_GBs"] * 1e9) * 1e6 + extra_total
    extra = extra_total
    units = {u: parts[u] for u in ("tensor", "smem", "port", "latency", "bubble")}
    # the bottleneck is the unit that sets the most time, summed over the waves
    bottleneck = max(unit_time, key=unit_time.get)
    idle = 1 - tiles / (waves * SMS * Rmax)
    return {"us": total, "sm_us": total - dram_bound_us, "dram_bound_us": dram_bound_us, "dram_floor_us": dram_floor, "dirty_extra_us": extra, "bottleneck": bottleneck, "per_ktile_us": units,
            "blocks_per_sm": Rmax, "occupancy_limit": occ["limit"], "waves": waves, "tiles": tiles, "KT": KT, "idle_slot_frac": idle, "dram_MB": traffic / 1e6}

"""Skeleton 3 (skinny shapes): a floor model for the PR #3290 fragment kernel's FragConfig knobs and the GEMV example's
(BLOCK_N, reduce_threads), from the chip constants only (no fit). Cold time (the honest one for decode: weights come from
DRAM) = launch + max(weight bytes / DRAM rate, waves x per-CTA chain), the chain being the k-tiles of a split times
max(mma time of the warps' tiles at the per-SM tensor rate, cp.async bytes at the per-block load rate), plus a fix-up
per split-K tile. Warm time of weights that fit the L2 replaces the DRAM rate with the chip L2 rate."""
from __future__ import annotations
import itertools, math
from ..model.mech import CHIP, SMS

DRAM = CHIP["dram_rate_GBs"]["value"]; L2 = CHIP["load_rate_chip_GBs"]["L2_resident_48_blocks"]; LAUNCH = CHIP["launch_us"]["value"]
L2_BYTES = 16.78e6


def frag_menu(M: int, N: int, K: int):
    bm = 16 if M <= 16 else 32 if M <= 32 else 48 if M <= 48 else 64
    for nw, bk, sk, st in itertools.product((1, 2, 4, 8), (32, 64, 128), (1, 2, 4, 8), (2, 3, 4)):
        if (K // bk) % sk or N % (32 * nw): continue
        smem = st * (bm * bk * 2 + nw * (bk // 32) * 576) + 4
        if smem > 99 * 1024: continue
        yield {"block_M": bm, "n_warps": nw, "block_K": bk, "split_k": sk, "num_stages": st, "smem": smem}


def frag_estimate(M: int, N: int, K: int, c: dict, warm: bool = False) -> dict:
    fam = CHIP["families"]["gemm_pipelined"]; rate_sm = fam["tensor_rate_TFs_per_SM"]["median"] * 1e12
    passes = math.ceil(M / c["block_M"]); bm = c["block_M"]; nw = c["n_warps"]; bk = c["block_K"]; sk = c["split_k"]
    ctas = (N // (32 * nw)) * sk; waves = max(1.0, ctas / SMS); kt = (K // bk) // sk
    wbytes = (N * K // 2 + N * K // 16) * passes; abytes = M * K * 2 * (N // (32 * nw))      # A re-read by every column tile
    stream = (wbytes + abytes) / ((L2 if (warm and wbytes <= L2_BYTES) else DRAM) * 1e9) * 1e6
    mma_kt = 2.0 * bm * (32 * nw) * bk / (rate_sm * min(1.0, nw / 4)) * 1e6                  # fewer than 4 warps cannot reach the SM's rate
    bytes_kt = bm * bk * 2 + nw * (bk // 32) * 576; load_kt = bytes_kt / (fam["load_rate_alone_GBs_per_block"]["median"] * 1e9) * 1e6
    chain = kt * (max(mma_kt, load_kt) + fam["sync_per_ktile_us"]["median"]) + (0.3 if sk > 1 else 0.0) + fam["block_setup_us"]["median"]
    t = LAUNCH + max(stream, waves * chain * passes)
    return {"expected_us": t, "stream_us": stream, "chain_us": waves * chain * passes, "ctas": ctas, "waves": waves, "mma_kt": mma_kt, "load_kt": load_kt}


def gemv_estimate(N: int, K: int, bn: int, rt: int, warm: bool = False) -> dict:
    blocks = N // bn; threads = bn * rt; bytes_b = N * K * 2
    stream = bytes_b / (DRAM * 1e9) * 1e6
    # one 128-bit load per thread per step; a block of `threads` threads keeps threads x 16 B in flight per step; the SM holds
    # up to 2048 threads so the chip's in-flight bytes are 48 x min(2048, blocks_per_sm x threads) x 16 B; latency ~ 600 ns
    per_sm_threads = min(2048, threads * min(32, 65536 // max(threads, 1))); inflight = SMS * per_sm_threads * 16
    rate_latency = inflight / 0.6e-6                                                             # B/s the latency-bound stream reaches
    t = LAUNCH + bytes_b / min(DRAM * 1e9, rate_latency) * 1e6 + blocks / SMS * 0.008
    return {"expected_us": t, "stream_floor_us": LAUNCH + stream, "latency_rate_GBs": rate_latency / 1e9, "blocks": blocks, "threads": threads}

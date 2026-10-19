"""R7: the loop reacts once. When a measurement contradicts the model (the one-sided bound breaks, or the discrepancy is
over the tolerance), find the term to blame at the solver's point and re-fit that one term from this measurement.
Returns the new terms and a note; the orchestrator writes a derived contract with the new terms, re-solves, and runs the
new pick. Two cases are handled, the two the round-1 counters could check:
  dram    the DRAM term was the longest and the kernel ran faster than it: L2 keeps more than `l2_bytes_effective`
          says. Raise it (bisection) until the DRAM term at this point fits under the measured time.
  tensor  the per-SM term was the longest and the kernel ran faster: the rate table's entry for this (threads, tile, K)
          is too low; raise it to what this measurement shows (a rate observed is a rate the tile can reach).
"""
from __future__ import annotations

import copy

from .cost import predict


def recalibrate(h: dict, shape: dict, chip: dict, terms: dict, measured_us: float, dram_bytes_measured: float | None = None, slack: float = 0.98) -> tuple[dict, dict]:
    p = predict(h, shape, chip, terms)
    pred = p["time_us"]
    note = {"point": h, "measured_us": measured_us, "predicted_us": pred, "longest_term": p["bound"], "changed": None}
    if measured_us >= pred:
        note["reason"] = "the model is not above the measurement; nothing to re-fit"
        return terms, note
    new = copy.deepcopy(terms)
    if p["bound"] == "dram":
        modeled_bytes = p["terms"]["dram_bytes"]
        if dram_bytes_measured and dram_bytes_measured < modeled_bytes * 0.95:
            # the counters say the traffic is lower than the model's: L2 keeps more than l2_bytes_effective says
            lo, hi = terms["l2_bytes_effective"], 64 * terms["l2_bytes_effective"]
            for _ in range(40):
                mid = (lo + hi) / 2; new["l2_bytes_effective"] = mid
                if predict(h, shape, chip, new)["terms"]["dram_bytes"] <= dram_bytes_measured:
                    hi = mid
                else:
                    lo = mid
            new["l2_bytes_effective"] = hi
            note["changed"] = {"term": "l2_bytes_effective", "from": terms["l2_bytes_effective"], "to": hi,
                               "how": f"raised until the modeled DRAM bytes at this point fall to the {dram_bytes_measured / 1e6:.0f} MB ncu counted"}
        else:
            # no counters, or the bytes agree: the rate is what this measurement contradicts. A rate observed is a rate the
            # chip reaches, so B_HBM becomes the bytes the model charges over the time the kernel took.
            rate = modeled_bytes / (measured_us * 1e-6 * slack - terms["launch_s"])
            new["B_HBM"] = max(terms["B_HBM"], rate)
            note["changed"] = {"term": "B_HBM", "from": terms["B_HBM"], "to": new["B_HBM"],
                               "how": "the DRAM rate this measurement shows for the bytes the model charges"}
    elif p["bound"] in ("tensor", "copy"):
        th, bm, bn, K = h["threads"], h["block_M"], h["block_N"], shape["K"]
        key = f"{bm}x{bn}@{K}"
        ut = new.setdefault("u_tile", {}).setdefault(th, {})
        old = ut.get(key, new["u_warps"].get(th, 1.0))
        tile_s = (measured_us * 1e-6 - terms["launch_s"]) / p["waves"]
        rate = p["terms"]["flops_tile"] / max(1e-9, tile_s) / terms["F_tc_per_sm"]
        ut[key] = float(min(1.0, max(old, rate))); ut[f"{bm}x{bn}"] = max(ut.get(f"{bm}x{bn}", 0.0), ut[key])
        note["changed"] = {"term": f"u_tile[{th}][{key}]", "from": old, "to": ut[key], "how": "the rate this measurement shows for this tile"}
    else:
        note["reason"] = f"no re-fit rule for the {p['bound']} term"
        return terms, note
    note["predicted_after_us"] = predict(h, shape, chip, new)["time_us"]
    new["source"] = terms.get("source", "") + f"; recalibrated from one measurement at {shape}: {note['changed']['term']}"
    return new, note

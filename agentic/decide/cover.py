"""A cover set for the unified template: every value of every decision at least once, around a base point, at one shape.
python -m agentic.decide.cover > jobs.json"""
import json, sys
from .template import DEFAULTS

DOMAINS = {"block_M": [64, 128, 256], "block_N": [64, 128, 256], "block_K": [32, 64, 128], "num_stages": [2, 3, 4, 5, 6],
           "threads": [128, 256], "swizzle": [0, 4, 8, 16], "prefetch": [0, 1], "warp_policy": [0, 1, 2], "vec_width": [0, 8],
           "copy_path": ["tma", "cpasync"], "warp_spec": [1, 0], "smem_layout": ["inferred", "swizzled"], "c_store": ["direct", "staged"],
           "split_k": [1, 2, 4]}


def legal(a):
    smem = a["num_stages"] * (a["block_M"] + a["block_N"]) * a["block_K"] * 2
    if a["c_store"] == "staged" and a["split_k"] == 1: smem = max(smem, a["block_M"] * a["block_N"] * 2)   # C_shared reuses the pipeline buffers
    return smem + 1024 <= 101376 and a["prefetch"] < a["num_stages"] and not (a["copy_path"] == "cpasync" and a["warp_spec"] == 1 and False)


def cover(base=None):
    base = {**DEFAULTS, **(base or {})}; out = [dict(base)]
    for k, dom in DOMAINS.items():
        for v in dom:
            if v == base[k]: continue
            a = {**base, k: v}
            if k == "copy_path" and v == "cpasync": a["warp_spec"] = 0       # cp.async path: TileLang's pipelined skeleton
            if k == "split_k" and v > 1: a["copy_path"], a["warp_spec"] = "cpasync", 0
            if legal(a): out.append(a)
    return out


if __name__ == "__main__":
    shape = {"M": 1024, "N": 1024, "K": 1024}
    json.dump([{"shape": shape, "assignments": cover(), "cublaslt": True}], sys.stdout, indent=1)

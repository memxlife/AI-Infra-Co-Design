"""A TileLang family as a contract (the brief, section 6): `program` holds the rendered source with every hole left as a
parameter, `decisions` are the knob menus, `constraints` are the rules legalize and the solver share, `cost` names
J-hat's terms. Template name: `gemm_tilelang`; `scope.structure.family` says which family."""
from __future__ import annotations

from ..chips import load_chip, scope_chip
from .cost import default_terms
from .tools import inventory, menus_for, render

TEMPLATE = "gemm_tilelang"
LEVEL = 5          # the kernel level: below the shared-memory tiling of the paper's template, this one has a real compiler under it


def family_contract(family: str, shape: dict, chip: dict | str = "spark-6951-gb10", terms: dict | None = None,
                    structure: dict | None = None, rationale: str = "", parent: str | None = None,
                    formulator: str = "scripted", **provenance) -> dict:
    chip = scope_chip(load_chip(chip)) if isinstance(chip, str) else chip
    inv = inventory(family)
    menus = menus_for(family, shape)
    terms = terms or default_terms(chip)
    knob = {k["name"]: k for k in inv["knobs"]}
    structure = structure or {
        "name": "bottleneck_dominance", "bucket": "objective",
        "regime": "compute-bound: the tensor pipe, one block per SM, is the longest pole for the big square shapes",
        "move": "epigraph cost over the per-SM tile time, the DRAM time and the L2 time; search the six menus",
        "obligation": "the dominant term stays dominant across the legal part of the menu",
        "strength": "observed", "evidence": []}
    return {
        "parent": parent, "level": LEVEL, "object": "TileLang kernel", "template": TEMPLATE,
        "scope": {"workload": {"name": f"gemm_{shape['M']}x{shape['N']}x{shape['K']}_fp16", **shape, "dtype": "fp16", "accum": "fp32"},
                  "chip": chip,
                  "structure": {"family": family, "func": inv["func"], "layout": "TileLang's inferred layout",
                                "copy_lowering": "TMA (what TileLang picks on sm_121a)", "warp_roles": "one producer warp group added by TileLang",
                                "inputs": "A (M x K) and B (K x N) row by row, FP16", "output": "C (M x N) row by row, FP16, FP32 accumulate"}},
        "signature": f"{TEMPLATE}:{family}:{shape['M']}x{shape['N']}x{shape['K']}",
        "methods": [],
        "decisions": {name: {"domain": list(menu), "kind": "software", "meaning": knob[name]["meaning"], "binds": knob[name]["binds"]}
                      for name, menu in menus.items()},
        "constraints": [
            {"id": "smem", "expr": "num_stages * (block_M*block_K + block_K*block_N) * 2 + 1024 <= smem_bytes_per_block",
             "meaning": "the pipeline's tiles fit the shared memory a block may have"},
            {"id": "regs", "expr": "block_M*block_N / threads + 40 <= 255", "meaning": "the FP32 accumulators and the rest fit a thread's registers"},
            {"id": "threads", "expr": "block_M*block_N / threads >= 16", "meaning": "each thread owns at least a fragment"},
            {"id": "fit", "expr": "blocks_per_sm(block_M, block_N, block_K, num_stages, threads) >= 1",
             "meaning": "one block fits an SM with TileLang's extra producer warp group"},
            {"id": "waves", "expr": "tiles = ceil(M/block_M) * ceil(N/block_N); waves = ceil(tiles / (sms * blocks_per_sm))",
             "meaning": "the tail wave is charged, not hidden"}],
        "cost": {"model": "T >= waves * (t_tile + t_pipe), T >= dram_bytes / B_HBM, T >= l2_reads / B_L2; t_tile = max(tensor, copy) * swizzle_penalty",
                 "objective": "time_us",
                 "terms": {"tensor": "2*block_M*block_N*K / (F_tc_per_sm * u_warps[threads])",
                           "copy": "ceil(K/block_K) * (block_M+block_N)*block_K*2 / B_L2_per_sm",
                           "t_pipe": "min(num_stages, K/block_K) * t_tile_latency",
                           "dram_bytes": "A + B once when they fit L2, else B re-read per swizzle panel of M tiles",
                           "l2_reads": "A per N tile + B per M tile"},
                 "numbers": {k: v for k, v in terms.items() if not isinstance(v, dict)},
                 "calibrated": terms.get("source", "contract defaults"),
                 "terms_values": {k: v for k, v in terms.items() if k != "fit"}},
        "decomposition": None,
        "preconditions": [{"id": "legal", "expr": "legalize(family, x, shape).ok", "mode": "precheck",
                           "meaning": "the Mac's shared-memory, register and thread rules; TileLang's own passes are the full check at compile time"}],
        "program": render(family, {}),
        "backward_map": "forget the tiles, the pipeline and the thread count => C = A B over (M, N, K) in FP16 with FP32 accumulate",
        "certificate_schema": {
            "relation": "the compiled kernel's C agrees with torch's A @ B within 1e-2 absolute and relative",
            "rank": "number of knobs not yet fixed",
            "obligations": [{"name": "legal", "kind": "precheck", "scope": "every menu point", "meaning": "the constraints admit only points legalize accepts"},
                            {"name": "compiles", "kind": "compile", "scope": "the solver's point", "meaning": "TileLang's passes accept the point"},
                            {"name": "agreement", "kind": "run", "scope": "the solver's point", "meaning": "the kernel's output matches torch on random inputs"}]},
        "structure": structure,
        "rationale": rationale,
        "status": "proposed",
        "certificate": None,
        "provenance": {"formulator": formulator, "model": None, "prompt_hash": None, **provenance},
    }

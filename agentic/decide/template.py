"""Render the unified GEMM program (agentic/tilelang/families/gemm/kernel.py.j2) for one assignment of its decisions."""
from __future__ import annotations
from pathlib import Path
import jinja2

TEMPLATE = Path(__file__).resolve().parents[1] / "tilelang" / "families" / "gemm" / "kernel.py.j2"
DEFAULTS = {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 3, "threads": 128, "swizzle": 0, "prefetch": 0,
            "warp_policy": 0, "vec_width": 0, "copy_path": "tma", "warp_spec": 1, "smem_layout": "inferred",
            "c_store": "direct", "split_k": 1}


def render(assign: dict) -> str:
    a = {**DEFAULTS, **assign}
    env = jinja2.Environment(trim_blocks=False, lstrip_blocks=False, undefined=jinja2.StrictUndefined)
    return env.from_string(TEMPLATE.read_text()).render(**a)


def name(assign: dict) -> str:
    a = {**DEFAULTS, **assign}
    return (f"{a['copy_path']}{'_ws' if a['warp_spec'] else ''}_{a['block_M']}x{a['block_N']}x{a['block_K']}_st{a['num_stages']}_th{a['threads']}"
            f"_sw{a['swizzle']}_pf{a['prefetch']}_wp{a['warp_policy']}_vw{a['vec_width']}_{a['smem_layout'][:3]}_{a['c_store']}_sk{a['split_k']}")

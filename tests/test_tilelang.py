"""T2 of the TileLang brief: rendering the family with the example's values reproduces the example's kernel function byte for
byte; legalize rejects 8 stages at 256 x 256 with a shared-memory reason and accepts the example."""
from pathlib import Path

from agentic.tilelang.tools import grid, inventory, legalize, menus_for, render

EX = Path(__file__).resolve().parents[1] / "agentic" / "tilelang" / "families" / "gemm_basic" / "example_gemm.py"
EXAMPLE = {"block_M": 128, "block_N": 128, "block_K": 32, "num_stages": 3, "threads": 128, "swizzle": 0}
SHAPE = {"M": 1024, "N": 1024, "K": 1024}


def test_t2_render_reproduces_the_example():
    # the example keeps block_M/N/K as call-time parameters and writes num_stages=3, threads=128 in the body; rendering
    # the family with exactly that fill must give the example's function byte for byte
    src = render("gemm_basic", {"num_stages": 3, "threads": 128, "swizzle": 0})
    ex = EX.read_text()
    fn = ex[ex.index("@tilelang.jit"):ex.index("\n\n\ndef main")]
    assert fn in src, "the rendered kernel must contain the example's matmul function byte for byte"
    full = render("gemm_basic", EXAMPLE)
    assert "def matmul(A, B, dtype=T.float16, accum_dtype=T.float32)" in full and "T.alloc_shared((128, 32), dtype)" in full


def test_t2_partial_render_leaves_holes_as_parameters():
    src = render("gemm_basic", {"block_M": 128, "block_N": 128})
    assert "def matmul(A, B, block_K, num_stages, threads, swizzle, dtype=" in src and "num_stages=num_stages" in src
    assert "T.use_swizzle(panel_size=swizzle, enable=swizzle > 0)" in src
    assert "T.use_swizzle(panel_size=8)" in render("gemm_basic", {**EXAMPLE, "swizzle": 8})


def test_t2_inventory_and_legalize():
    inv = inventory("gemm_basic")
    assert [k["name"] for k in inv["knobs"]] == ["block_M", "block_N", "block_K", "num_stages", "threads", "swizzle"]
    assert len(grid("gemm_basic", SHAPE)) == 3 * 3 * 3 * 4 * 3 * 4 and menus_for("gemm_basic", {"M": 32, "N": 4096, "K": 4096})["block_M"] == [16, 32, 64]
    ok = legalize("gemm_basic", EXAMPLE, SHAPE)
    assert ok["ok"] and ok["smem_bytes_est"] == 3 * (128 * 32 + 32 * 128) * 2
    bad = legalize("gemm_basic", {"block_M": 256, "block_N": 256, "block_K": 32, "num_stages": 8, "threads": 256}, SHAPE)
    assert not bad["ok"] and "shared memory" in bad["reason"]

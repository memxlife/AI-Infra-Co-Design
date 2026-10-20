"""Issue 5's TMA GEMMs through the plain interface (kernel= and grid_x), plus one of our own checks."""
import sys, time
from pathlib import Path
sys.path.insert(0, "/Users/lishang/work/research/AI-Infra-Co-Design")
from agentic.gb10.sim.ws_validate import kernel_us
cases = Path(sys.argv[1]); shape = {"M": 4096, "N": 14336, "K": 4096}
runs = [
    ("gemm, swizzle 10, grid_x M (its real launch)", cases / "gemm/bound-library/bound-0.sass", "matmul_non_persistent_kernel",
     {"block_M": 256, "block_N": 128, "block_K": 64, "num_stages": 2, "threads": 256, "swizzle": 10, "grid_x": "M"}, shape, 5081.9),
    ("gemm, swizzle 10, grid_x N (our layout)", cases / "gemm/bound-library/bound-0.sass", "matmul_non_persistent_kernel",
     {"block_M": 256, "block_N": 128, "block_K": 64, "num_stages": 2, "threads": 256, "swizzle": 10}, shape, 5081.9),
    ("gemm_basic, swizzle 8 (control)", cases / "gemm_basic/bound-library/bound-0.sass", "matmul_kernel",
     {"block_M": 128, "block_N": 256, "block_K": 64, "num_stages": 2, "threads": 256, "swizzle": 8}, shape, 4878.5),
    ("our 1024^3 TMA pick (was 40.1)", Path("/Users/lishang/work/research/AI-Infra-Co-Design/calibration/spark-6951/sass/tma_1024__full.sass"), "matmul_kernel",
     {"block_M": 64, "block_N": 128, "block_K": 128, "num_stages": 2, "threads": 128, "swizzle": 8}, {"M": 1024, "N": 1024, "K": 1024}, 45.1),
]
for label, sass, kern, h, sh, meas in runs:
    t = time.time(); us = kernel_us(label, h, sh, warm=True, sass_path=sass, kernel=kern)
    print(f"{label:46s} {us:9.1f} us  measured {meas:8.1f}  error {us / meas - 1:+6.1%}  ({time.time() - t:.0f} s)", flush=True)

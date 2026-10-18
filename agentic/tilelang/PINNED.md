# TileLang on the Spark: what is pinned

| item | value |
|---|---|
| TileLang commit | `fe6d319c78c0738186fc6e83eb8436ddef65204d` (2026-10-16, "[CUDA][WS] Restrict TMA-only producer in warp 0 (#3503)") |
| TileLang version | 0.1.15+cuda.gitfe6d319c, built from source on the Spark (`~/work/research/tilelang-work/tilelang`, `pip install -e .`) |
| target TileLang detects | `sm_121a` (cuda, 1024 threads, warp 32) |
| CUDA | 13.0 (nvcc build 36424714) |
| PyTorch | 2.11.0+cu128 |
| host | spark-6951 (NVIDIA GB10, 48 SMs, 101 376 B shared per block opt-in, 102 400 B per SM, 65 536 registers per SM, 1 536 threads per SM, 24 MiB L2) |

What the build needed: `python3-dev` (CMake could not find the Python headers), `ninja` in the venv. Nothing in TileLang was changed.

What differs from the proposal:
- `JITKernel.resource_usage`, `n_regs`, `n_spills` are HIP-only today; the runner compiles the exported CUDA source with `nvcc -Xptxas -v` and reads registers, spills and static shared memory from ptxas.
- The profiler's `cupti` backend returns 0 ms on this host; the runner times with the CUDA-event backend (median of the measured window).
- On `sm_121a` TileLang's own lowering of the plain example already uses TMA (`CUtensorMap`) with a 256-thread warp-specialized kernel (128 asked for) and `mma`; no `wgmma`, no `tcgen05`.

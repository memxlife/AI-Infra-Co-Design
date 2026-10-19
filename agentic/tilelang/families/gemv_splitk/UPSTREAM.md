# Skeleton 3a: split-K GEMV

- Upstream: `examples/gemv/example_gemv.py` at tile-ai/tilelang commit `fe6d319c78c0738186fc6e83eb8436ddef65204d`
  (copied here unchanged as `upstream_example_gemv.py`). Kernels used: `splitk_gemv_vectorized` (SIMT, one 128-bit
  load per thread per step, `atomic_add` into shared), `splitk_gemv_vectorized_tvm` (warp allreduce instead), and
  `gemv_alloc_reducer` (tile form, `T.Pipelined` + `alloc_reducer`). They compute one row (M = 1): at M = 32 and
  M = 128 they give the weight-stream floor, not a GEMM.

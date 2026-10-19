# Skeleton 1: plain pipelined GEMM, warp specialization off, cp.async, every warp computes

- Upstream: `examples/gemm/example_gemm.py` at tile-ai/tilelang commit `fe6d319c78c0738186fc6e83eb8436ddef65204d`
  (the pinned build on the Spark, 0.1.15+cuda.gitfe6d319c).
- The kernel body is the example's. The family fixes two switches that the example leaves to TileLang: the decorator
  is `tilelang.jit(pass_configs={"tl.disable_warp_specialized": True})` and the two copies are
  `tilelang.cuda.language.copy(..., disable_tma=True)`, so the loads lower to `cp.async` issued by the computing warps
  and no 128-thread producer group is added. `kernel.py.j2` is `gemm_basic`'s template; `knobs.yaml` pins
  `tma: 0`, `warp_spec: 0`.
- Run unchanged on the Spark (round 3, S2 revised): see `notes/tilelang_findings.md`, "S2 skeleton 1".

# Skeleton 2: stream-K GEMM

- Upstream: `examples/gemm_streamk/example_tilelang_gemm_streamk.py` at tile-ai/tilelang commit
  `fe6d319c78c0738186fc6e83eb8436ddef65204d` (copied here as `upstream_example_tilelang_gemm_streamk.py`).
- As written, the example's kernel closes over module-level numbers (shape, `total_sm = 108`, the 16x128x32 tile,
  the stream-K split). On the pinned commit it does not compile with warp specialization on
  (`ProducerConsumerWS: failed to replace pipeline loop`, `src/cuda/transform/producer_consumer_ws.cc:1789`); with
  `tl.disable_warp_specialized` it compiles, lowers to cp.async and is correct.
- `kernel.py.j2` keeps the kernel body and turns the module-level numbers into holes: `block_M`, `block_N`, `block_K`,
  `num_stages` (the stream-K loop), `threads`, `programs` (the persistent grid; the example uses the SM count), and one
  adaptation knob `dp_stages` for the data-parallel loop's depth, which the example hard-codes to 1.
- The cp.async stream-K loader of tile-ai/tilelang PR #3290 (`maint/gemm/gemm_sm120/sm120_nvfp4_w4a16_streamk.py`,
  head `e6b003e9a6bad49c86865285ee42ecbc9be9bb05`) was read for comparison; it is not used here (it is a W4A16 kernel
  with a hand-written cp.async stage ring, a run-time grid, 64-bit iteration partition and a ticketed fix-up).

# Skeleton 3b: the fragment kernel of PR #3290 (M <= 64 rows per pass, A in registers through ldmatrix, B streamed)

- Upstream: tile-ai/tilelang PR #3290 "[CUDA] Add an SM120 W4A16 NVFP4 GEMM example for decode-sized batches",
  head `e6b003e9a6bad49c86865285ee42ecbc9be9bb05`, files `examples/gemm_sm120/sm120_nvfp4_w4a16_gemm.py`
  (`nvfp4_w4a16_frag_kernel`, `nvfp4_w4a16_tile_kernel`, `prepare_nvfp4_weight`, `frag_config`) and
  `maint/gemm/gemm_sm120/sm120_nvfp4_w4a16_streamk.py`, copied unchanged under `upstream/`.
- It is a W4A16 kernel: BF16 activations, NVFP4 weights dequantized in registers, `mma.sync.m16n8k16`, cp.async
  through `T.Pipelined`, warp specialization off by its own decorator. Its weight stream is 0.5625 B per weight
  against 2 B for the fp16 cuBLASLt baseline, so a time comparison with the S1 table is a comparison of two
  different problems; the findings say so. It imports `tilelang.cuda.intrinsics` (present at the pinned commit) and
  checks for compute capability 12.0 in `is_supported_device`; on the GB10 (12.1) the kernels are launched through
  `launch()` directly.

# Wrong results: explicit linear shared-memory layout on GEMM operands when copies lower to TMA (sm_121a)

**Commit:** fe6d319c78c0738186fc6e83eb8436ddef65204d (0.1.15+cuda.gitfe6d319c), built from source.
**Host:** NVIDIA GB10 (DGX Spark), `sm_121a`, CUDA 13.0 (nvcc 13.0.r13.0), torch 2.11.0+cu128, Python 3.12.

## What happens

The plain GEMM (`examples/gemm/example_gemm.py` shape) with `T.annotate_layout({A_s: make_linear_layout(A_s), B_s: make_linear_layout(B_s)})`
computes wrong results when the copies into `A_s`/`B_s` are lowered to TMA (the default on `sm_121a`). Max abs error
against `A @ B` is ~20–150 (FP16 in, FP32 accumulate, 1024³ and 32×4096×4096), where the inferred layout gives ~0.06.
With the same kernel and TMA disabled per copy (`tilelang.cuda.language.copy(..., disable_tma=True)`), or with the
producer warp group disabled (`tl.disable_warp_specialized`), the linear layout is correct (and slower, as expected).
`make_swizzled_layout` and the inferred layout are correct with TMA.

In a random sample of 400 legal configurations of this kernel (tiles 64–256, K slices 32–128, 2–5 stages, 128–512
threads, panels 0–16), every one of the 31 wrong results had `linear` layout + TMA + the producer group; 0 of 96 linear
configurations without TMA were wrong.

## Minimal repro

(`linear_layout_tma_wrong.py`, attached.) Output on the host above:

```
<output pasted below>
```

## Expected

Either the explicit linear layout is honoured by the TMA lowering (tensor map without swizzle), or `annotate_layout`
with a layout the TMA path cannot serve raises at lowering time instead of producing a kernel that runs and is wrong.

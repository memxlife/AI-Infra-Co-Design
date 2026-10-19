# [BUG] stream-K example: the next tile's cp.async prologue overwrites shared tiles a slower warp is still reading (WAR hazard)

*Draft for Li to review before filing. Found in round 3, S2 revised, while running the skeletons through compute-sanitizer.*

### What version

`fe6d319c78c0738186fc6e83eb8436ddef65204d` (0.1.15+cuda.gitfe6d319c), built from source, CUDA 13.0, GB10 (sm_121a).

### Problem

`examples/gemm_streamk/example_tilelang_gemm_streamk.py`, compiled with `tl.disable_warp_specialized` (with warp
specialization on it does not compile at this commit: `ProducerConsumerWS: failed to replace pipeline loop`,
`src/cuda/transform/producer_consumer_ws.cc:1789`). The stream-K part is a `while` loop over partial tiles; inside it a
`T.Pipelined` loop fills `A_shared`/`B_shared` with cp.async and multiplies. The generated CUDA has no `__syncthreads()`
between the last multiply of one `while` iteration (the pipeline's epilogue reads the last stage) and the first
cp.async of the next iteration (the pipeline's prologue writes stage 0). A warp that finishes its epilogue early starts
the next tile's cp.async into the buffers slower warps are still reading with ldmatrix. The data-parallel loop that
follows does have the barrier (`__syncthreads()` before its prologue), so the fix is one barrier at the top of the
`while` body, or the same barrier the pipeline pass emits for the data-parallel loop.

compute-sanitizer (racecheck) reports the hazard on every launch at 1024x1024x1024, 16x128x32, 2 stages, 64 threads,
48 programs:

```
========= Error: Potential WAR hazard detected at __shared__ 0x4600 in block (43,0,0) :
```

(`0x4600` lies in `A_shared`, whose base is `buf_dyn_shmem + 16384`.) The results of 30 random inputs were still
within fp16 rounding of torch, so the race is a timing one on this GPU; it is a wrong-answer bug on any GPU where the
early warp's cp.async lands before the late warp's ldmatrix.

### Reproduce

```bash
compute-sanitizer --tool racecheck --racecheck-report hazard python racecheck.py skel2_1024x1024x1024_16x128x32_st2_th64_sm48 1
```

`racecheck.py` is in `runs/tilelang/skel/`; the module it loads is the example with `m, n, k = 1024, 1024, 1024`,
`total_sm = 48` and `tilelang.jit(pass_configs={'tl.disable_warp_specialized': True})`, nothing else changed. The
generated CUDA is `runs/tilelang/skel/src/src_skel2_1024x1024x1024_16x128x32_st2_th64_sm48.cu` (lines 31 to 121: the
`while` body; line 123: the barrier the data-parallel loop gets).

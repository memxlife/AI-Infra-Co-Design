## Step 3: the full 12-knob space, solver pick against 200 random legal points

### 1024x1024x1024
- sample: 200 drawn, 0 failed to build or run, 13 wrong, 0 spill (0.0%); best measured 61.5 µs at {'block_M': 64, 'block_N': 256, 'block_K': 32, 'num_stages': 4, 'threads': 256, 'swizzle': 8, 'prefetch': 1, 'warp_policy': 2, 'vec_width': 0, 'tma': 0, 'warp_spec': 0, 'smem_layout': 'inferred'}
- model on the sample: Spearman 0.67, bound broken on 30.5%
- solver's pick (275 binaries, 20 ms): {'block_M': 256, 'block_N': 128, 'block_K': 32, 'num_stages': 3, 'threads': 256, 'swizzle': 0, 'prefetch': 0, 'warp_policy': 0, 'vec_width': 0, 'tma': 1, 'warp_spec': 1, 'smem_layout': 'inferred'} -> measured 62.5 µs (predicted 63.4); gap to sample best +1.6%; new knobs it used: none (all at the round-1 defaults)
- round 1's pick re-timed in this run: 64.5 µs; round 2 pick vs round 1 pick: -2.0 µs (-3.2%)

### 32x4096x4096
- sample: 200 drawn, 0 failed to build or run, 18 wrong, 0 spill (0.0%); best measured 203.8 µs at {'block_M': 64, 'block_N': 128, 'block_K': 64, 'num_stages': 3, 'threads': 256, 'swizzle': 16, 'prefetch': 2, 'warp_policy': 2, 'vec_width': 4, 'tma': 0, 'warp_spec': 1, 'smem_layout': 'swizzled'}
- model on the sample: Spearman -0.11, bound broken on 27.5%
- solver's pick (275 binaries, 38 ms): {'block_M': 64, 'block_N': 256, 'block_K': 64, 'num_stages': 2, 'threads': 256, 'swizzle': 0, 'prefetch': 0, 'warp_policy': 0, 'vec_width': 0, 'tma': 1, 'warp_spec': 1, 'smem_layout': 'inferred'} -> measured 206.8 µs (predicted 200.4); gap to sample best +1.4%; new knobs it used: none (all at the round-1 defaults)
- round 1's pick re-timed in this run: 203.7 µs; round 2 pick vs round 1 pick: +3.0 µs (+1.5%)


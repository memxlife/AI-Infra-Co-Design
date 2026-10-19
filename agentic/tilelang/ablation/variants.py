"""Variants of a TileLang-generated GEMM kernel with one mechanism removed (round 3, step 2c), as text patches:
  kt=N        both k-loops run N tiles (0 = prologue + epilogue only; the grid, barriers and descriptors unchanged)
  store=off   the C store loop runs 0 iterations
  loads=off   the producer's TMA loads and expect-tx are removed; it arrives on the full barrier so the consumer proceeds
  compute=off the consumer's ldmatrix + mma block is removed; the barrier protocol is kept
Every variant also gets per-block %globaltimer stamps (start at entry, end after the store loop) into __tl_ts."""
from __future__ import annotations
import re


def patch(src: str, kt: int | None = None, store: bool = True, loads: bool = True, compute: bool = True) -> str:
    s = src
    # loop bounds: the producer's `for (int k = 0; k < N; ++k)` and the consumer's `for (int k_1 = 0; k_1 < N; ++k_1)`
    if kt is not None:
        s, n1 = re.subn(r"for \(int k = 0; k < (\d+); \+\+k\)", f"for (int k = 0; k < {kt}; ++k)", s)
        s, n2 = re.subn(r"for \(int k_1 = 0; k_1 < (\d+); \+\+k_1\)", f"for (int k_1 = 0; k_1 < {kt}; ++k_1)", s)
        assert n1 == 1 and n2 == 1, (n1, n2)
    if not store:
        s, n = re.subn(r"for \(int i_(\d+) = 0; i_\1 < (\d+); \+\+i_\1\) \{\n(\s+half_t C_local_cast)", lambda m: f"for (int i_{m.group(1)} = 0; i_{m.group(1)} < 0; ++i_{m.group(1)}) {{\n{m.group(3)}", s)
        assert n == 1, n
    if not loads:
        # every TMA load line goes; the expect-tx lines go; the one arrive_and_expect_tx becomes a plain arrive on the
        # same (init(1)) barrier, so the consumer's wait completes without data. Empty unrolled loops are left as they are.
        n_loads = s.count("tl::tma_load(")
        s = "\n".join(l for l in s.splitlines() if "tl::tma_load(" not in l and ".expect_transaction(" not in l)
        s, n2 = re.subn(r"mbarrier\[([^\]]+)\]\.arrive_and_expect_tx\(\d+\);", r"mbarrier[\1].arrive();", s)
        assert n_loads >= 2 and n2 >= 1, (n_loads, n2)
    if not compute:
        # the consumer block: from the `{` that opens `half_t A_local` to its matching `}`
        i = s.index("half_t A_local["); i = s.rfind("{", 0, i)
        depth, j = 0, i
        while True:
            if s[j] == "{": depth += 1
            elif s[j] == "}":
                depth -= 1
                if depth == 0: break
            j += 1
        s = s[:i] + "{ }" + s[j + 1:]
    # per-block stamps: at entry by the first consumer thread, at the end after the store loop
    m = re.search(r"extern \"C\" __global__ void __launch_bounds__\((\d+), \d+\) matmul_kernel\([^)]*\) \{\n", s); threads = int(m.group(1))
    first_consumer = 128 if "warpgroup_reg_dealloc" in s else 0
    # the kernel may remap blockIdx (rasterization) after entry, so the linear id is taken from the builtin at entry and kept
    stamp = f"  const int __tl_bid = (int)(blockIdx.y * gridDim.x + blockIdx.x);\n  if ((int)threadIdx.x == {first_consumer}) {{ unsigned long long __t; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(__t)); __tl_ts[2 * __tl_bid] = __t; }}\n"
    s = s[:m.end()] + stamp + s[m.end():]
    end = "  if ((int)threadIdx.x == %d) { unsigned long long __t; asm volatile(\"mov.u64 %%0, %%%%globaltimer;\" : \"=l\"(__t)); __tl_ts[2 * __tl_bid + 1] = __t; }\n" % first_consumer
    # the kernel's last closing brace
    k = s.rstrip().rfind("}")
    s = s[:k] + end + s[k:]
    return s


def patch_cpasync(src: str, kt: int | None = None, store: bool = True, loads: bool = True, compute: bool = True) -> str:
    """The same variants for the warp-specialization-off kernel (round 3, S2 revised, skeleton 1): one k-loop
    `for (int k = 0; k < KT-S+1; ++k)` whose body issues cp.async for tile k+S-1, commits, waits and computes tile k; the
    last S-1 tiles are computed in unrolled epilogue blocks; the prologue issues the first S-1 tiles.
      kt=0        the k-loop runs 0 iterations, the loads and the compute blocks are removed: prologue + store only
      loads=off   every `tl::cp_async_gs` line goes (commit/wait stay, as empty groups)
      compute=off every `{ half_t A_local...[` block goes (loop body and epilogue)
      store=off   the C store loop runs 0 iterations
    Stamps by thread 0 at entry and after the store loop."""
    s = src
    assert "warpgroup_reg_dealloc" not in s and "tl::cp_async_gs" in s, "not a cp.async kernel without a producer group"
    if kt is not None:
        s, n1 = re.subn(r"for \(int k = 0; k < (\d+); \+\+k\)", f"for (int k = 0; k < {kt}; ++k)", s); assert n1 == 1, n1
        if kt == 0:
            loads = False; compute = False
    if not store:
        s, n = re.subn(r"for \(int i_(\d+) = 0; i_\1 < (\d+); \+\+i_\1\) \{\n(\s+half_t C_local_cast)", lambda m: f"for (int i_{m.group(1)} = 0; i_{m.group(1)} < 0; ++i_{m.group(1)}) {{\n{m.group(3)}", s); assert n == 1, n
    if not loads:
        n_loads = s.count("tl::cp_async_gs"); assert n_loads >= 4, n_loads
        s = "\n".join(l for l in s.splitlines() if "tl::cp_async_gs" not in l)
    if not compute:
        n_blocks = 0
        while True:
            m = re.search(r"half_t A_local(_\d+)?\[", s)
            if not m: break
            i = s.rfind("{", 0, m.start()); depth, j = 0, i
            while True:
                if s[j] == "{": depth += 1
                elif s[j] == "}":
                    depth -= 1
                    if depth == 0: break
                j += 1
            s = s[:i] + "{ }" + s[j + 1:]; n_blocks += 1
        assert n_blocks >= 1, n_blocks
    m = re.search(r"extern \"C\" __global__ void __launch_bounds__\((\d+), \d+\) matmul_kernel\([^)]*\) \{\n", s)
    stamp = "  const int __tl_bid = (int)(blockIdx.y * gridDim.x + blockIdx.x);\n  if ((int)threadIdx.x == 0) { unsigned long long __t; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(__t)); __tl_ts[2 * __tl_bid] = __t; }\n"
    s = s[:m.end()] + stamp + s[m.end():]
    end = "  if ((int)threadIdx.x == 0) { unsigned long long __t; asm volatile(\"mov.u64 %0, %%globaltimer;\" : \"=l\"(__t)); __tl_ts[2 * __tl_bid + 1] = __t; }\n"
    k = s.rstrip().rfind("}")
    return s[:k] + end + s[k:]

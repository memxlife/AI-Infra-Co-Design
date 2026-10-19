// L2 replacement policy probes (read with ncu's L2 hit/miss sector counters on the launch of interest):
//  cyclic W P:      launch 'warm' reads W bytes once, then launch 'scan' reads the same W bytes P times in one launch
//  hotstream H S:   'hot' reads H bytes, 'stream' reads S fresh bytes, 'hot' again (the launch to inspect: hits = what
//                   survived the stream)
// Reads are 16 B ld.global.cg (L2, not L1), 48 blocks x 1024 threads, coalesced.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
__global__ void readk(const uint4* __restrict__ p, size_t n16, int passes, unsigned* out) {
  unsigned acc = 0; size_t stride = (size_t)gridDim.x * blockDim.x;
  for (int r = 0; r < passes; r++)
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += stride) {
      uint4 v; asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "l"(p + i)); acc ^= v.x ^ v.w;
    }
  if (acc == 0x9e3779b9u) out[0] = acc;
}
int main(int argc, char** argv) {
  const char* mode = argv[1]; size_t a = strtoull(argv[2], 0, 10) << 20, b = strtoull(argv[3], 0, 10);
  char* buf; unsigned* o; cudaMalloc(&buf, 512u << 20); cudaMalloc(&o, 4); cudaMemset(buf, 1, 512u << 20); cudaDeviceSynchronize();
  if (!strcmp(mode, "cyclic")) {            // a = W bytes, b = passes
    readk<<<48, 1024>>>((uint4*)buf, a / 16, 1, o);            // warm (launch 0)
    readk<<<48, 1024>>>((uint4*)buf, a / 16, (int)b, o);       // scan (launch 1)
  } else {                                    // hotstream: a = H bytes, b = S MB
    size_t s = b << 20;
    readk<<<48, 1024>>>((uint4*)buf, a / 16, 2, o);            // hot, twice (launch 0)
    readk<<<48, 1024>>>((uint4*)(buf + (256u << 20)), s / 16, 1, o);   // stream of fresh bytes (launch 1)
    readk<<<48, 1024>>>((uint4*)buf, a / 16, 1, o);            // hot again (launch 2): what survived
  }
  cudaDeviceSynchronize(); return 0;
}

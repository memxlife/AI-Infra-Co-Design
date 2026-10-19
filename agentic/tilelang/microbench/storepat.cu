// The epilogue store of a 256 x 128 fp16 C tile (64 KB per block) into a 1024-column matrix, three patterns:
//  tl:   TileLang's: each warp instruction writes 4 B per lane, 8 rows x 16 B (lanes 4 per row), rows 2 KB apart;
//        a 32 B sector is completed by two different instructions
//  v16:  16 B per lane, 4 rows x 128 B (a full line per row)
//  row:  4 B per lane, one row x 128 B
// Blocks take C tiles in TileLang's order (columns fastest). Per-block %globaltimer span and clock64; %smid recorded.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <vector>
__device__ unsigned long long sp[1024]; __device__ unsigned smid_of[1024];
template <int MODE>
__global__ void st(unsigned* C, int tiles_n) {
  const int by = blockIdx.x / tiles_n, bx = blockIdx.x % tiles_n, tid = threadIdx.x, lane = tid & 31, w = tid >> 5;   // 256 threads
  unsigned base = by * 256 * 512 + bx * 64;           // in 4-byte words: rows of 1024 halves = 512 words
  unsigned long long g0, g1; asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g0));
  if (MODE == 0) {        // tl: per warp: 8 rows (lane>>2) x 4 words (lane&3); 64 instructions per thread
#pragma unroll 4
    for (int i = 0; i < 64; i++) { unsigned r = (w >> 1) * 64 + (i >> 4) * 16 + (i & 1) * 8 + (lane >> 2), c = (w & 1) * 32 + ((i >> 1) & 7) * 4 + (lane & 3); C[base + r * 512 + c] = i; }
  } else if (MODE == 1) { // v16: 16 B per lane, 4 rows x 128 B per instruction
#pragma unroll 4
    for (int i = 0; i < 16; i++) { unsigned r = w * 32 + i * 2 + (lane >> 4), c = (lane & 15) * 4; uint4 v = make_uint4(i, i, i, i); if (r < 256 && c < 64) *(uint4*)(C + base + r * 512 + c) = v; }
  } else {                // row: 4 B per lane, one row of 128 B per instruction
#pragma unroll 4
    for (int i = 0; i < 64; i++) { unsigned r = w * 32 + (i >> 1), c = (i & 1) * 32 + lane; C[base + r * 512 + c] = i; }
  }
  __syncthreads(); asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(g1));
  if (tid == 0) { sp[blockIdx.x] = g1 - g0; unsigned s; asm volatile("mov.u32 %0, %%smid;" : "=r"(s)); smid_of[blockIdx.x] = s; }
}
template <int MODE> void run(unsigned* C, int blocks, const char* name) {
  for (int r = 0; r < 3; r++) st<MODE><<<blocks, 256>>>(C, 8);
  cudaDeviceSynchronize(); std::vector<double> v;
  for (int r = 0; r < 15; r++) { st<MODE><<<blocks, 256>>>(C, 8); cudaDeviceSynchronize(); static unsigned long long h[1024]; cudaMemcpyFromSymbol(h, sp, 8 * blocks); double m = 0; for (int b = 0; b < blocks; b++) m += h[b]; v.push_back(m / blocks); }
  std::sort(v.begin(), v.end()); static unsigned s[1024]; cudaMemcpyFromSymbol(s, smid_of, 4 * blocks);
  printf("RESULT %-4s blocks=%2d block_us=%.2f GBs_per_block=%.1f smids=", name, blocks, v[v.size() / 2] / 1e3, 65536.0 / v[v.size() / 2]);
  for (int b = 0; b < std::min(blocks, 4); b++) printf("%u,", s[b]); printf("\n");
}
int main() {
  unsigned* C; cudaMalloc(&C, 64 << 20);
  for (int b : {1, 2, 24, 48, 96}) { run<0>(C, b, "tl"); run<1>(C, b, "v16"); }
}

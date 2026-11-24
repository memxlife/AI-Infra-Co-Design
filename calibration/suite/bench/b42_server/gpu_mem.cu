// Per-GPU memory and PCIe numbers for a multi-GPU server. For each visible GPU: DRAM read / write bandwidth
// (grid-stride kernels over 4 GB), device-to-device copy, DRAM latency (one-thread dependent pointer chase over a
// random cyclic permutation, 128 B apart), host<->device copy bandwidth (256 MB pinned) and small-copy latency (4 B
// pinned, median of 1000). The host thread and pinned buffers are placed on the GPU's own CPU socket (cpu list arg).
// nvcc -O3 -arch=sm_120a -o gpu_mem gpu_mem.cu ; ./gpu_mem "0-31,64-95" "32-63,96-127"   (socket-0 and socket-1 CPUs)
#include <cuda_runtime.h>
#include <sched.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

__global__ void rd(const float4* p, size_t n, float* o) {
  float4 a = make_float4(0, 0, 0, 0);
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { float4 v = p[i]; a.x += v.x; a.y += v.y; }
  if (a.x == 1234.5f) o[0] = a.y;
}
__global__ void wr(float4* p, size_t n) {
  for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = make_float4(1, 2, 3, 4);
}
__global__ void chase(const unsigned* next, int steps, unsigned* out, long long* cyc) {
  unsigned j = 0; long long t0 = clock64();
  for (int s = 0; s < steps; s++) j = next[j];
  long long t1 = clock64(); out[0] = j; cyc[0] = t1 - t0;
}

static void pin(const char* list) {   // "a-b,c-d"
  cpu_set_t s; CPU_ZERO(&s); char buf[256]; strncpy(buf, list, 255); buf[255] = 0;
  for (char* tok = strtok(buf, ","); tok; tok = strtok(nullptr, ",")) { int a, b; if (sscanf(tok, "%d-%d", &a, &b) == 2) for (int c = a; c <= b; c++) CPU_SET(c, &s); else CPU_SET(atoi(tok), &s); }
  sched_setaffinity(0, sizeof(s), &s);
}

int main(int argc, char** argv) {
  int ng; cudaGetDeviceCount(&ng);
  for (int d = 0; d < ng; d++) {
    cudaSetDevice(d); cudaDeviceProp pr; cudaGetDeviceProperties(&pr, d);
    int numa = d < ng / 2 ? 0 : 1; if (argc > 2) pin(argv[1 + numa]);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1); float ms;
    size_t bytes = 4ull << 30, n4 = bytes / 16; float4* buf; cudaMalloc(&buf, bytes); float* o; cudaMalloc(&o, 4);
    int blocks = pr.multiProcessorCount * 8;
    auto best = [&](auto launch) { float b = 1e30f; for (int r = 0; r < 5; r++) { cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); b = std::min(b, ms); } return b; };
    wr<<<blocks, 512>>>(buf, n4); cudaDeviceSynchronize();
    float trd = best([&] { rd<<<blocks, 512>>>(buf, n4, o); });
    float twr = best([&] { wr<<<blocks, 512>>>(buf, n4); });
    float tcp = best([&] { cudaMemcpy(buf + n4 / 2, buf, bytes / 2, cudaMemcpyDeviceToDevice); });
    // latency: working sets from 32 KB (L1) to 2 GB (DRAM); 128 B between elements, random cycle
    std::vector<double> lat; std::vector<size_t> sets = {32ull << 10, 4ull << 20, 64ull << 20, 512ull << 20, 2048ull << 20};
    unsigned *nx, *ou; long long* cy; cudaMalloc(&ou, 4); cudaMalloc(&cy, 8);
    for (size_t ws : sets) {
      size_t m = ws / 128; std::vector<unsigned> perm(m), next(m * 32, 0); for (size_t i = 0; i < m; i++) perm[i] = i;
      std::shuffle(perm.begin() + 1, perm.end(), std::mt19937(7));
      for (size_t i = 0; i < m; i++) next[perm[i] * 32] = perm[(i + 1) % m] * 32;
      cudaMalloc(&nx, m * 128); cudaMemcpy(nx, next.data(), m * 128, cudaMemcpyHostToDevice);
      int steps = 200000; chase<<<1, 1>>>(nx, 20000, ou, cy); cudaDeviceSynchronize();
      cudaEventRecord(e0); chase<<<1, 1>>>(nx, steps, ou, cy); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1);
      long long c; cudaMemcpy(&c, cy, 8, cudaMemcpyDeviceToHost); lat.push_back(ms * 1e6 / steps); lat.push_back((double)c / steps);
      cudaFree(nx);
    }
    // PCIe: 256 MB pinned copies both ways, 4 B pinned copy latency
    size_t pb = 256ull << 20; char* h; cudaHostAlloc(&h, pb, cudaHostAllocDefault); memset(h, 1, pb);
    float th2d = best([&] { cudaMemcpy(buf, h, pb, cudaMemcpyHostToDevice); });
    float td2h = best([&] { cudaMemcpy(h, buf, pb, cudaMemcpyDeviceToHost); });
    std::vector<float> sh, sd;
    for (int r = 0; r < 1000; r++) {
      cudaEventRecord(e0); cudaMemcpy(buf, h, 4, cudaMemcpyHostToDevice); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); sh.push_back(ms);
      cudaEventRecord(e0); cudaMemcpy(h, buf, 4, cudaMemcpyDeviceToHost); cudaEventRecord(e1); cudaEventSynchronize(e1); cudaEventElapsedTime(&ms, e0, e1); sd.push_back(ms);
    }
    std::sort(sh.begin(), sh.end()); std::sort(sd.begin(), sd.end());
    printf("{\"gpu\": %d, \"name\": \"%s\", \"sms\": %d, \"mem_GiB\": %.1f, \"numa\": %d, \"read_GBs\": %.0f, \"write_GBs\": %.0f, \"d2d_copy_GBs\": %.0f, "
           "\"lat_ns\": {\"32KB\": %.0f, \"4MB\": %.0f, \"64MB\": %.0f, \"512MB\": %.0f, \"2GB\": %.0f}, \"lat_cycles_2GB\": %.0f, "
           "\"h2d_GBs\": %.1f, \"d2h_GBs\": %.1f, \"h2d_4B_us\": %.2f, \"d2h_4B_us\": %.2f, \"err\": \"%s\"}\n",
           d, pr.name, pr.multiProcessorCount, pr.totalGlobalMem / 1073741824.0, numa, bytes / trd / 1e6, bytes / twr / 1e6, bytes / tcp / 1e6,
           lat[0], lat[2], lat[4], lat[6], lat[8], lat[9], pb / th2d / 1e6, pb / td2h / 1e6, sh[500] * 1e3, sd[500] * 1e3, cudaGetErrorString(cudaGetLastError()));
    fflush(stdout);
    cudaFreeHost(h); cudaFree(buf); cudaFree(o); cudaFree(ou); cudaFree(cy);
  }
  return 0;
}

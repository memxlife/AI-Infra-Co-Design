// Which tier-4 hardware features does the GB10 offer? Device attributes for thread-block clusters, L2 persistence
// (access policy window), memory pools, green contexts / SM partitioning hints, and a real cluster launch with
// distributed shared memory to confirm clusters work.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
namespace cg = cooperative_groups;
#define CK(x) do { auto e = (x); if (e) { printf("call failed: %d at line %d\n", (int)e, __LINE__); } } while (0)
__global__ void __cluster_dims__(2, 1, 1) dsmem(int* out) {
  __shared__ int s; cg::cluster_group cl = cg::this_cluster();
  if (threadIdx.x == 0) s = cl.block_rank() + 100;
  cl.sync();
  int* other = cl.map_shared_rank(&s, (cl.block_rank() + 1) % cl.num_blocks());
  if (threadIdx.x == 0) out[blockIdx.x] = *other;
  cl.sync();
}
int main() {
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("RESULT name=%s cc=%d.%d sms=%d l2_bytes=%d persistingL2CacheMaxSize=%d accessPolicyMaxWindowSize=%d\n", p.name, p.major, p.minor, p.multiProcessorCount, p.l2CacheSize, p.persistingL2CacheMaxSize, p.accessPolicyMaxWindowSize);
  int v; 
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrClusterLaunch, 0)); printf("RESULT clusterLaunch=%d\n", v);
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrMemoryPoolsSupported, 0)); printf("RESULT memoryPools=%d\n", v);
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrCooperativeLaunch, 0)); printf("RESULT cooperativeLaunch=%d\n", v);
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrConcurrentKernels, 0)); printf("RESULT concurrentKernels=%d\n", v);
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrStreamPrioritiesSupported, 0)); printf("RESULT streamPriorities=%d\n", v);
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0)); printf("RESULT pageableUsesHostPageTables=%d\n", v);
  CK(cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0)); printf("RESULT smemOptin=%d\n", v);
  // cluster occupancy for a 2-block cluster of 128 threads, and the largest cluster size allowed
  cudaLaunchConfig_t cfg = {}; cfg.gridDim = dim3(96); cfg.blockDim = dim3(128); cudaLaunchAttribute at[1]; at[0].id = cudaLaunchAttributeClusterDimension;
  at[0].val.clusterDim.x = 2; at[0].val.clusterDim.y = 1; at[0].val.clusterDim.z = 1; cfg.attrs = at; cfg.numAttrs = 1;
  int nc = -1; CK(cudaOccupancyMaxActiveClusters(&nc, dsmem, &cfg)); printf("RESULT maxActiveClusters_size2=%d\n", nc);
  int maxsz = -1; CK(cudaOccupancyMaxPotentialClusterSize(&maxsz, (void*)dsmem, &cfg)); printf("RESULT maxPotentialClusterSize=%d\n", maxsz);
  int* o; CK(cudaMalloc(&o, 96 * 4)); dsmem<<<96, 128>>>(o); CK(cudaDeviceSynchronize()); int h[4]; CK(cudaMemcpy(h, o, 16, cudaMemcpyDeviceToHost));
  printf("RESULT cluster_dsmem_read block0_sees=%d block1_sees=%d (expect 101, 100)\n", h[0], h[1]);
  // L2 persistence: set aside part of the L2 and attach a window to a stream
  size_t lim = 0; CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, (size_t)8 << 20)); CK(cudaDeviceGetLimit(&lim, cudaLimitPersistingL2CacheSize));
  printf("RESULT persistingL2_set_8MB_got=%zu\n", lim);
  int prio_lo, prio_hi; CK(cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi)); printf("RESULT streamPriorityRange=%d..%d\n", prio_lo, prio_hi);
}

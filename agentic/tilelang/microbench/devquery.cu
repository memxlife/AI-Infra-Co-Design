// The GB10's GPU as the driver reports it: the spec numbers the model starts from.
#include <cuda_runtime.h>
#include <cstdio>
int main() {
  cudaDeviceProp p; cudaGetDeviceProperties(&p, 0); int v;
  printf("name=%s cc=%d.%d sms=%d\n", p.name, p.major, p.minor, p.multiProcessorCount);
  cudaDeviceGetAttribute(&v, cudaDevAttrClockRate, 0); printf("sm_clock_max_kHz=%d\n", v);
  cudaDeviceGetAttribute(&v, cudaDevAttrMemoryClockRate, 0); printf("mem_clock_kHz=%d\n", v);
  cudaDeviceGetAttribute(&v, cudaDevAttrGlobalMemoryBusWidth, 0); printf("mem_bus_bits=%d\n", v);
  printf("l2_bytes=%d persisting_l2_max=%d\n", p.l2CacheSize, p.persistingL2CacheMaxSize);
  printf("smem_per_sm=%zu smem_per_block_optin=%zu smem_per_block=%zu reserved_smem_per_block=%zu\n", p.sharedMemPerMultiprocessor, p.sharedMemPerBlockOptin, p.sharedMemPerBlock, p.reservedSharedMemPerBlock);
  printf("regs_per_sm=%d regs_per_block=%d max_threads_per_sm=%d max_blocks_per_sm=%d warp=%d\n", p.regsPerMultiprocessor, p.regsPerBlock, p.maxThreadsPerMultiProcessor, p.maxBlocksPerMultiProcessor, p.warpSize);
  printf("integrated=%d unified_addressing=%d pageable_access=%d concurrent_managed=%d\n", p.integrated, p.unifiedAddressing, p.pageableMemoryAccess, p.concurrentManagedAccess);
  return 0;
}

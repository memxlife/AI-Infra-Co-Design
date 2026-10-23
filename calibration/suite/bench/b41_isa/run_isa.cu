// B41 item 2: load one probe cubin, launch it once with valid operands, report "runs" or "traps: <error>".
// One process per cubin, so a trap cannot poison the next probe. The tensor map covers a 64x64 fp16 global array
// (box 64x64 = 8 KB, the probes' shared buffer); a cubin whose name contains "multicast" runs as a 2-block cluster, one with "_wg_" with 384 threads.
// nvcc -O2 -arch=sm_121a -o run_isa run_isa.cu -lcuda ; ./run_isa probe.cubin
#include <cuda.h>
#include <cstdio>
#include <cstring>

static const char* err(CUresult r) { const char* s = nullptr; cuGetErrorString(r, &s); return s ? s : "unknown"; }

int main(int argc, char** argv) {
  const char* path = argv[1];
  CUdevice dev; CUcontext ctx; CUmodule mod; CUfunction fn;
  cuInit(0); cuDeviceGet(&dev, 0); cuDevicePrimaryCtxRetain(&ctx, dev); cuCtxSetCurrent(ctx);   // same call in CUDA 12.8 and 13.0
  CUresult r = cuModuleLoad(&mod, path);
  if (r != CUDA_SUCCESS) { printf("%s\tload-failed: %s\n", path, err(r)); return 0; }
  if ((r = cuModuleGetFunction(&fn, mod, "probe")) != CUDA_SUCCESS) { printf("%s\tno-function: %s\n", path, err(r)); return 0; }
  CUdeviceptr out, g; cuMemAlloc(&out, 4096 * 4); cuMemAlloc(&g, 64 * 64 * 2); cuMemsetD8(g, 1, 64 * 64 * 2);
  alignas(64) CUtensorMap tmap; memset(&tmap, 0, sizeof(tmap));
  cuuint64_t dims[2] = {64, 64}, strides[1] = {64 * 2}; cuuint32_t box[2] = {64, 64}, es[2] = {1, 1};
  cuTensorMapEncodeTiled(&tmap, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 2, (void*)g, dims, strides, box, es, CU_TENSOR_MAP_INTERLEAVE_NONE,
                         CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  void* args[] = {&out, &tmap};
  if (strstr(path, "multicast")) {
    CUlaunchAttribute at[1]; at[0].id = CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION; at[0].value.clusterDim.x = 2; at[0].value.clusterDim.y = 1; at[0].value.clusterDim.z = 1;
    CUlaunchConfig cfg = {}; cfg.gridDimX = 2; cfg.gridDimY = 1; cfg.gridDimZ = 1; cfg.blockDimX = 128; cfg.blockDimY = 1; cfg.blockDimZ = 1; cfg.attrs = at; cfg.numAttrs = 1;
    r = cuLaunchKernelEx(&cfg, fn, args, nullptr);
  } else {
    r = cuLaunchKernel(fn, 1, 1, 1, strstr(path, "_wg_") ? 384 : 128, 1, 1, 0, nullptr, args, nullptr);   // _wg_: 3 warp groups
  }
  if (r == CUDA_SUCCESS) r = cuCtxSynchronize();
  if (r == CUDA_SUCCESS) printf("%s\truns\n", path); else printf("%s\ttraps: %s\n", path, err(r));
  return 0;
}

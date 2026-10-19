// R1: the cuBLASLt baseline. FP16 in, FP32 accumulate, FP16 out, row-major C = A (M x K) x B (K x N), the same
// data layout the TileLang family uses. Timed with CUDA events, median over reps, after warmup; the heuristic
// picks the algorithm (top-1) with a 32 MB workspace. Build: nvcc -O3 -arch=sm_121a -o cublaslt_gemm cublaslt_gemm.cu -lcublasLt
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x) do { auto e = (x); if (e != 0) { fprintf(stderr, "error %d at line %d\n", (int)e, __LINE__); exit(1); } } while (0)

// optional 6th argument: clear the L2 before every timed rep by reading a 64 MB buffer (cold weights, as in a decode
// step where other layers run in between)
__global__ void flush_rd(const int4* __restrict__ p, size_t n, int* sink) {
  int4 a = make_int4(0, 0, 0, 0);
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { int4 v = __ldcg(p + i); a.x ^= v.x; }
  if (a.x == 0x7654321) sink[0] = 1;
}
int main(int argc, char** argv) {
  int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]);
  int warmup = argc > 4 ? atoi(argv[4]) : 20, reps = argc > 5 ? atoi(argv[5]) : 100;
  __half *A, *B, *C; CK(cudaMalloc(&A, (size_t)M * K * 2)); CK(cudaMalloc(&B, (size_t)K * N * 2)); CK(cudaMalloc(&C, (size_t)M * N * 2));
  std::vector<__half> h((size_t)std::max((size_t)M * K, (size_t)K * N));
  for (auto& x : h) x = __float2half((rand() % 2001 - 1000) / 1000.0f);
  CK(cudaMemcpy(A, h.data(), (size_t)M * K * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(B, h.data(), (size_t)K * N * 2, cudaMemcpyHostToDevice));
  cublasLtHandle_t lt; CK(cublasLtCreate(&lt));
  // row-major C = A B  <=>  column-major C^T = B^T A^T: pass B as the first operand with leading dimension N
  cublasLtMatmulDesc_t op; CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t tn = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tn, sizeof(tn))); CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof(tn)));
  cublasLtMatrixLayout_t la, lb, lc;
  CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, N, K, N));   // B^T: N x K, ld N
  CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, M, K));   // A^T: K x M, ld K
  CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, N, M, N));   // C^T: N x M, ld N
  size_t ws_size = 32u << 20; void* ws; CK(cudaMalloc(&ws, ws_size));
  cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));
  cublasLtMatmulHeuristicResult_t heur[8]; int nh = 0;
  CK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, heur, &nh));
  if (nh == 0) { fprintf(stderr, "no cublasLt algorithm\n"); return 2; }
  float alpha = 1.f, beta = 0.f;
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  int flush = argc > 6 ? atoi(argv[6]) : 0; int4* fbuf = nullptr; int* fsink = nullptr;
  if (flush) { CK(cudaMalloc(&fbuf, 64u << 20)); CK(cudaMemset(fbuf, 1, 64u << 20)); CK(cudaMalloc(&fsink, 4)); }
  float best_ms = 1e30f; int best_i = -1;
  for (int i = 0; i < nh; i++) {                       // try every heuristic candidate, keep the fastest: what cublasLt's own tuning would do
    for (int w = 0; w < warmup; w++) CK(cublasLtMatmul(lt, op, &alpha, B, la, A, lb, &beta, C, lc, C, lc, &heur[i].algo, ws, ws_size, 0));
    std::vector<float> t;
    for (int r = 0; r < reps; r++) {
      if (flush) flush_rd<<<192, 512>>>(fbuf, (64u << 20) / 16, fsink);
      CK(cudaEventRecord(e0)); CK(cublasLtMatmul(lt, op, &alpha, B, la, A, lb, &beta, C, lc, C, lc, &heur[i].algo, ws, ws_size, 0)); CK(cudaEventRecord(e1));
      CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms);
    }
    std::sort(t.begin(), t.end()); float med = t[t.size() / 2];
    printf("algo %d: median %.4f ms (min %.4f)\n", i, med, t[0]);
    if (med < best_ms) { best_ms = med; best_i = i; }
  }
  double tflops = 2.0 * M * N * K / (best_ms * 1e-3) / 1e12;
  printf("RESULT M=%d N=%d K=%d cublaslt_best_ms=%.4f tflops=%.2f algo=%d candidates=%d\n", M, N, K, best_ms, tflops, best_i, nh);
  return 0;
}

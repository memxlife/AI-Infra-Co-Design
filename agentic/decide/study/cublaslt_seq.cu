// cuBLASLt for the comparative study: pick the fastest heuristic candidate (median of single calls), then time it the
// study's way: per call in a back-to-back sequence of N calls, warm; rot (each call reads the next of several copies of
// the weight matrix, 256 MB or more in all, as decode does); and cold (a 64 MB read before each call, its own time
// subtracted; kept for reference, not cold at decode sizes), launched from a stream and replayed as a captured CUDA
// graph. Inputs are random, like the study's own kernels (all-zero inputs can change the tensor cores' power and clock).
// nvcc -O3 -arch=sm_121a -o cublaslt_seq cublaslt_seq.cu -lcublasLt ; ./cublaslt_seq M N K
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CK(x) do { auto e = (x); if ((int)e != 0) { fprintf(stderr, "error %d at line %d\n", (int)e, __LINE__); exit(1); } } while (0)
__global__ void flush_rd(const int4* __restrict__ p, size_t n, int* sink) {
  int4 a = make_int4(0, 0, 0, 0);
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) { int4 v = p[i]; a.x ^= v.x; }
  if (a.x == 0x7654321) sink[0] = 1;
}
__global__ void fill(__half* p, size_t n, unsigned seed) {      // uniform in [-1, 1) from a hash of the index
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
    unsigned x = (unsigned)i * 2654435761u ^ seed; x ^= x >> 13; x *= 0x5bd1e995u; x ^= x >> 15;
    p[i] = __float2half((x & 0xffff) / 32768.f - 1.f);
  }
}
int main(int argc, char** argv) {
  int M = atoi(argv[1]), N = atoi(argv[2]), K = atoi(argv[3]); const int NSEQ = argc > 4 ? atoi(argv[4]) : 100;
  const size_t bbytes = (size_t)K * N * 2; const int NROT = (int)std::max<size_t>(1, ((256u << 20) + bbytes - 1) / bbytes);
  __half *A, *C; std::vector<__half*> Bs(NROT); CK(cudaMalloc(&A, (size_t)M * K * 2)); CK(cudaMalloc(&C, (size_t)M * N * 2));
  fill<<<192, 512>>>(A, (size_t)M * K, 1u);
  for (int j = 0; j < NROT; j++) { CK(cudaMalloc(&Bs[j], bbytes)); fill<<<192, 512>>>(Bs[j], (size_t)K * N, 2u); }
  CK(cudaDeviceSynchronize());
  cublasLtHandle_t lt; CK(cublasLtCreate(&lt));
  cublasLtMatmulDesc_t op; CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  cublasOperation_t tn = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tn, sizeof(tn))); CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tn, sizeof(tn)));
  cublasLtMatrixLayout_t la, lb, lc;
  CK(cublasLtMatrixLayoutCreate(&la, CUDA_R_16F, N, K, N)); CK(cublasLtMatrixLayoutCreate(&lb, CUDA_R_16F, K, M, K)); CK(cublasLtMatrixLayoutCreate(&lc, CUDA_R_16F, N, M, N));
  size_t ws_size = 32u << 20; void* ws; CK(cudaMalloc(&ws, ws_size));
  cublasLtMatmulPreference_t pref; CK(cublasLtMatmulPreferenceCreate(&pref));
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_size, sizeof(ws_size)));
  cublasLtMatmulHeuristicResult_t heur[8]; int nh = 0; CK(cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, lc, lc, pref, 8, heur, &nh));
  float alpha = 1.f, beta = 0.f; cudaStream_t s; CK(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  int4* fb; int* fs; CK(cudaMalloc(&fb, 64u << 20)); CK(cudaMemset(fb, 1, 64u << 20)); CK(cudaMalloc(&fs, 4));
  auto call = [&](int i, int b = 0) { CK(cublasLtMatmul(lt, op, &alpha, Bs[b], la, A, lb, &beta, C, lc, C, lc, &heur[i].algo, ws, ws_size, s)); };
  auto flush = [&]() { flush_rd<<<192, 512, 0, s>>>(fb, (64u << 20) / 16, fs); };
  float best = 1e30f; int bi = 0;
  for (int i = 0; i < nh; i++) {
    for (int w = 0; w < 10; w++) call(i);
    std::vector<float> t;
    for (int r = 0; r < 50; r++) { CK(cudaEventRecord(e0, s)); call(i); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms); }
    std::sort(t.begin(), t.end()); if (t[25] < best) { best = t[25]; bi = i; }
  }
  auto seq = [&](bool cold, bool with_call, bool rot = false) {
    CK(cudaStreamSynchronize(s)); CK(cudaEventRecord(e0, s));
    for (int r = 0; r < NSEQ; r++) { if (cold) flush(); if (with_call) call(bi, rot ? (r + 1) % NROT : 0); }
    CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); return ms * 1e3 / NSEQ;
  };
  auto graph = [&](bool cold, bool with_call, bool rot = false) {
    cudaGraph_t g; cudaGraphExec_t ge; CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    for (int r = 0; r < NSEQ; r++) { if (cold) flush(); if (with_call) call(bi, rot ? (r + 1) % NROT : 0); }
    CK(cudaStreamEndCapture(s, &g)); CK(cudaGraphInstantiate(&ge, g, 0)); CK(cudaGraphLaunch(ge, s)); CK(cudaStreamSynchronize(s));
    std::vector<float> t;
    for (int r = 0; r < 5; r++) { CK(cudaEventRecord(e0, s)); CK(cudaGraphLaunch(ge, s)); CK(cudaEventRecord(e1, s)); CK(cudaEventSynchronize(e1)); float ms; CK(cudaEventElapsedTime(&ms, e0, e1)); t.push_back(ms * 1e3 / NSEQ); }
    std::sort(t.begin(), t.end()); CK(cudaGraphExecDestroy(ge)); CK(cudaGraphDestroy(g)); return t[2];
  };
  for (int w = 0; w < 20; w++) call(bi);
  std::vector<float> a, b, c;
  for (int r = 0; r < 3; r++) { a.push_back(seq(false, true)); b.push_back(seq(true, true) - seq(true, false)); c.push_back(seq(false, true, true)); }
  float ga = graph(false, true), gb = graph(true, true) - graph(true, false), gc = graph(false, true, true);
  std::sort(a.begin(), a.end()); std::sort(b.begin(), b.end()); std::sort(c.begin(), c.end());
  printf("RESULT M=%d N=%d K=%d algo=%d single_call_median_us=%.2f seq_warm_stream_us=%.2f seq_cold_stream_us=%.2f seq_warm_graph_us=%.2f seq_cold_graph_us=%.2f seq_rot_stream_us=%.2f seq_rot_graph_us=%.2f\n",
         M, N, K, bi, best * 1e3, a[1], b[1], ga, gb, c[1], gc);
}

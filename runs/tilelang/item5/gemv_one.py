# Run one GEMV config a few times, for ncu. python gemv_one.py N BLOCK_N REDUCE_THREADS
import sys, torch
from kernels import gemv
n, bn, rt = (int(x) for x in sys.argv[1:4])
A = torch.randn(n, device="cuda").half(); B = torch.randn(n, n, device="cuda").half()
k = gemv(bn, rt).compile(N=n, K=n)
for _ in range(5): k(A, B)
torch.cuda.synchronize()

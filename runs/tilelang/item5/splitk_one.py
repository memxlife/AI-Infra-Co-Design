# Run one split-K config a few times back-to-back, for ncu. python splitk_one.py M N K bm bn bk st th split
import sys, torch
from kernels import splitk
M, N, K, bm, bn, bk, st, th, split = (int(x) for x in sys.argv[1:10])
A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half(); C = torch.zeros(M, N, device="cuda")
k = splitk(bm, bn, bk, st, th, split).compile(M=M, N=N, K=K)
for _ in range(6): k(A, B, C)
torch.cuda.synchronize()

import importlib.util, sys, torch
name = sys.argv[1]; M, N, K = (int(x) for x in sys.argv[2:5])
spec = importlib.util.spec_from_file_location(name, name + ".py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
k = m.matmul.compile(M=M, N=N, K=K); A = torch.randn(M, K, device="cuda").half(); B = torch.randn(K, N, device="cuda").half()
for _ in range(3): k(A, B)
torch.cuda.synchronize()

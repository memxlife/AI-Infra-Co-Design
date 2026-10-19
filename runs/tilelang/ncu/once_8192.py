import importlib.util, torch
spec = importlib.util.spec_from_file_location("km", "ncu/pick_8192.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
k = mod.matmul.compile(M=8192, N=8192, K=8192)
A = torch.randn(8192, 8192, device="cuda").half(); B = torch.randn(8192, 8192, device="cuda").half()
for _ in range(3): k(A, B)
torch.cuda.synchronize()

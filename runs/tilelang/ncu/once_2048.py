import importlib.util, torch
spec = importlib.util.spec_from_file_location("km", "ncu/pick_2048.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
k = mod.matmul.compile(M=2048, N=2048, K=2048)
A = torch.randn(2048, 2048, device="cuda").half(); B = torch.randn(2048, 2048, device="cuda").half()
for _ in range(3): k(A, B)
torch.cuda.synchronize()

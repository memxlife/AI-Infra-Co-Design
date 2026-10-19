import importlib.util, torch
spec = importlib.util.spec_from_file_location("km", "attr/pick_1024.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
k = mod.matmul.compile(M=1024, N=1024, K=1024)
A = torch.randn(1024, 1024, device="cuda").half(); B = torch.randn(1024, 1024, device="cuda").half()
for _ in range(3): k(A, B)
torch.cuda.synchronize()

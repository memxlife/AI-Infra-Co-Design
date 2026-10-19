import importlib.util, torch
spec = importlib.util.spec_from_file_location("km", "waves/panel16.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
k = mod.matmul.compile(M=4096, N=4096, K=4096)
A = torch.randn(4096, 4096, device="cuda").half(); B = torch.randn(4096, 4096, device="cuda").half()
for _ in range(3): k(A, B)
torch.cuda.synchronize()

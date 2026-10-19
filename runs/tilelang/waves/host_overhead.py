# the host-side cost of TileLang's launch path for the 1024^3 pick: do_bench (events around k(A,B)), a plain event loop around k(A,B),
# wall-clock per call with the GPU idle, and cuTensorMapEncodeTiled alone through the driver
import importlib.util, time, torch, ctypes
spec = importlib.util.spec_from_file_location("km", "attr/pick_1024.py"); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
k = mod.matmul.compile(M=1024, N=1024, K=1024)
A = torch.randn(1024, 1024, device="cuda").half(); B = torch.randn(1024, 1024, device="cuda").half()
for _ in range(10): k(A, B)
torch.cuda.synchronize()
p = k.get_profiler(); print("do_bench event median us:", p.do_bench(backend="event", warmup=25, rep=100, return_mode="median") * 1e3)
s0 = torch.cuda.Event(enable_timing=True); s1 = torch.cuda.Event(enable_timing=True)
torch.cuda.synchronize(); t = time.perf_counter()
for _ in range(200): k(A, B)
torch.cuda.synchronize(); print("wall per call us (200 back-to-back):", (time.perf_counter() - t) / 200 * 1e6)
t = time.perf_counter()
for _ in range(200): k(A, B); torch.cuda.synchronize()
print("wall per call us, synchronized each (host + GPU):", (time.perf_counter() - t) / 200 * 1e6)
# cuTensorMapEncodeTiled alone
cuda = ctypes.CDLL("libcuda.so.1"); enc = cuda.cuTensorMapEncodeTiled
tm = ctypes.create_string_buffer(128); dims = (ctypes.c_uint64 * 2)(1024, 1024); strides = (ctypes.c_uint64 * 1)(2048); box = (ctypes.c_uint32 * 2)(64, 64); es = (ctypes.c_uint32 * 2)(1, 1)
t = time.perf_counter()
for _ in range(2000): enc(tm, 6, 2, ctypes.c_void_p(A.data_ptr()), dims, strides, box, es, 0, 3, 2, 0)
print("cuTensorMapEncodeTiled us per call:", (time.perf_counter() - t) / 2000 * 1e6)

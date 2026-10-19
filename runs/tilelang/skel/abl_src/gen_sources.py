# On the Spark: compile every rendered cp.async config, check it against torch once, dump TileLang's kernel source and the dynamic smem it sets.
import importlib.util, json, re, sys, torch
cfgs = json.load(open("configs.json")); out = []
for c in cfgs:
    spec = importlib.util.spec_from_file_location(c["name"], c["name"] + ".py"); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    s = c["shape"]
    try:
        k = m.matmul.compile(M=s["M"], N=s["N"], K=s["K"])
        A = torch.randn(s["M"], s["K"], device="cuda").half(); B = torch.randn(s["K"], s["N"], device="cuda").half(); C = k(A, B); torch.cuda.synchronize()
        err = (C.float() - A.float() @ B.float()).abs().max().item(); src = k.get_kernel_source()
        host = k.get_host_source() if hasattr(k, "get_host_source") else ""
        mm = re.search(r"MaxDynamicSharedMemorySize, (\d+)", host) or re.search(r"MaxDynamicSharedMemorySize, (\d+)", src)
        smem = int(mm.group(1)) if mm else c["smem"]
        open(c["name"] + ".kernel.cu", "w").write(src); out.append({**c, "smem": smem, "max_err": err, "ok": err < 1e-2})
        print(c["name"], "ok" if err < 1e-2 else "WRONG", f"err {err:.3f} smem {smem} (formula {c['smem']})", flush=True)
    except Exception as e:
        print(c["name"], "FAILED", str(e).strip().splitlines()[-1][:160], flush=True); out.append({**c, "ok": False, "error": str(e)[-300:]})
json.dump(out, open("sources.json", "w"), indent=0)

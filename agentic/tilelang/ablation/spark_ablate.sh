#!/bin/bash
# runs on the Spark: build every variant harness with nvcc (TileLang's include paths), run the planned grids, one JSON line each
export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda
cd ~/work/research/AI-Infra-Co-Design-calib/tilelang/ablation
TL=$HOME/work/research/tilelang-work/tilelang
INC="-I$TL/src -I$TL/3rdparty/cutlass/include -I/usr/local/cuda/include"; export INC
ONLY=${1:-}
python3 - "$ONLY" <<'EOF2'
import json, subprocess, sys, os, time
only = sys.argv[1]
plan = json.load(open("plan.json")); built = {}
out = open("ablation.jsonl", "a")
for item in plan:
    if only and item["name"] != only: continue
    key = (item["name"], item["variant"]); d = item["name"]; src = f"{d}/{item['variant']}.cu"; exe = f"{d}/{item['variant']}.bin"
    if key not in built:
        t = time.time()
        r = subprocess.run(f"nvcc -O3 -std=c++20 -w -Xcudafe --diag_suppress=177 -gencode arch=compute_121a,code=sm_121a --expt-relaxed-constexpr {os.environ.get("INC","")} -o {exe} {src} -lcuda", shell=True, capture_output=True, text=True)
        built[key] = r.returncode == 0
        if r.returncode != 0:
            out.write(json.dumps({**item, "ok": False, "stage": "build", "error": r.stderr[-800:]}) + "\n"); out.flush(); print(key, "BUILD FAILED", r.stderr.splitlines()[-1][:160] if r.stderr else ""); continue
        print(key, "built", round(time.time() - t, 1), "s", flush=True)
    if not built[key]: continue
    r = subprocess.run([f"./{exe}", str(item["grid"][0]), str(item["grid"][1]), "50"], capture_output=True, text=True, timeout=600)
    line = next((l for l in r.stdout.splitlines() if l.startswith("RESULT")), None)
    rec = {**item, "ok": line is not None, "result": line, "stderr": r.stderr[-300:] if r.returncode else ""}
    if line:
        for kv in line.split()[1:]:
            k, v = kv.split("="); rec[k] = float(v) if "us" in k else v
    out.write(json.dumps(rec) + "\n"); out.flush(); print(item["name"], item["variant"], item["grid"], line, flush=True)
EOF2

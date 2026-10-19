"""Run the GPU-side TileLang tools on the Spark over SSH (alias as a parameter; no address here): rsync the runner and the
kernel file, run them under the Spark's Python, bring back one JSON file."""
from __future__ import annotations

import json, subprocess, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REMOTE_DIR = "~/work/research/AI-Infra-Co-Design-calib/tilelang"
SPARK_PYTHON = "$HOME/.venv/bin/python"
SPARK_ENV = "export PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda"


def _ssh(host: str, cmd: str, timeout: int = 1200) -> subprocess.CompletedProcess:
    return subprocess.run(["ssh", "-o", "BatchMode=yes", host, cmd], capture_output=True, text=True, timeout=timeout)


def run_kernel(host: str, kernel_source: str, m: int, n: int, k: int, holes: dict, func: str = "matmul", reps: int = 100,
               warmup: int = 25, ncu: bool = False, inout_c: bool = False) -> dict:
    """Compile, check, time and report one TileLang kernel on the Spark. Returns the kernel_report dict."""
    _ssh(host, f"mkdir -p {REMOTE_DIR}")
    subprocess.run(["rsync", "-az", str(HERE / "spark_runner.py"), f"{host}:{REMOTE_DIR}/spark_runner.py"], check=True, capture_output=True)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(kernel_source); local = f.name
    subprocess.run(["rsync", "-az", local, f"{host}:{REMOTE_DIR}/kernel.py"], check=True, capture_output=True)
    holes_json = json.dumps(holes).replace("'", "")
    cmd = (f"{SPARK_ENV}; cd {REMOTE_DIR} && {SPARK_PYTHON} spark_runner.py --kernel kernel.py --func {func} --m {m} --n {n} --k {k} "
           f"--holes '{holes_json}' --reps {reps} --warmup {warmup} --out report.json" + (" --ncu" if ncu else "") + (" --inout-c" if inout_c else ""))
    r = _ssh(host, cmd)
    i = r.stdout.find("{")
    if i < 0:
        return {"ok": False, "stage": "ssh", "error": (r.stderr or r.stdout)[-1500:]}
    rep = json.loads(r.stdout[i:])
    rep["kernel_source"] = kernel_source
    return rep

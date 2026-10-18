"""TileLang as the tool surface for agentic kernel optimization (notes/tilelang_work_proposal: the Oct 19 brief).

Four layers of TileLang, wrapped and not forked: the frontend (knobs), the transform passes (legalize), codegen + ptxas
(compile) and the runtime + profiler (profile). `remote.py` runs the GPU-side tools on the Spark over SSH; `spark_runner.py`
is the script that runs there. Families of kernels with holes live in `families/<name>/`.
"""
from .remote import run_kernel, SPARK_PYTHON  # noqa: F401

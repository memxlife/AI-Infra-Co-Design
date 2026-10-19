"""T5 of the TileLang brief, without the GPU: the orchestrator runs propose -> legalize (family check) -> solve -> legalize
(instance check) -> compile + profile (mocked with a row of the 1024^3 sweep) -> admitted, on the gemm_tilelang template."""
import json
from pathlib import Path

import pytest

from agentic.agents.formulator.tilelang import TileLangFormulator
from agentic.agents.orchestrator import Orchestrator
from agentic.ledger import Ledger

ROWS = Path(__file__).resolve().parents[1] / "runs" / "tilelang" / "sweep_1024.jsonl"


def canned_report(rows):
    def run_kernel(host, src, m, n, k, holes, func="matmul", reps=100, warmup=25, ncu=False):
        # the orchestrator renders the solver's point into the source; find that point in the sweep
        import re
        bm = int(re.search(r"alloc_shared\(\((\d+), (\d+)\)", src).group(1)); bk = int(re.search(r"alloc_shared\(\((\d+), (\d+)\)", src).group(2))
        bn = int(re.search(r"alloc_shared\(\(\d+, \d+\), dtype\)\s+B_shared = T.alloc_shared\(\(\d+, (\d+)\)", src).group(1))
        st = int(re.search(r"num_stages=(\d+)", src).group(1)); th = int(re.search(r"threads=(\d+)", src).group(1))
        sw = re.search(r"use_swizzle\(panel_size=(\d+)\)", src); sw = int(sw.group(1)) if sw else 0
        want = {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": st, "threads": th, "swizzle": sw}
        r = next((r for r in rows if r["holes"] == want), None)
        assert r is not None, f"the solver's point {want} is not in the sweep"
        return {**r, "ok": True, "host": "mock", "tilelang": {"version": "0.1.15"}, "cuda": {"nvcc": "13.0"}, "ncu": None, "kernel_file": "mock", "func": func}
    return run_kernel


@pytest.mark.skipif(not ROWS.exists(), reason="needs the 1024^3 sweep table")
def test_t5_loop_on_gemm_tilelang(tmp_path, monkeypatch):
    rows = [json.loads(l) for l in ROWS.read_text().splitlines() if l.strip()]
    import agentic.agents.evaluator.tilelang as ev
    monkeypatch.setattr(ev, "run_kernel", canned_report(rows))
    L = Ledger(str(tmp_path / "t5.db"))
    o = Orchestrator(L, "t5", TileLangFormulator([{"M": 1024, "N": 1024, "K": 1024}]), evaluator_backend="tilelang:mock", level=5)
    o.run_loop()
    kinds = [e["kind"] for e in L.events("t5")]
    for k in ("proposed", "family_check", "solved", "instance_check", "measured", "admitted", "run_finished"):
        assert k in kinds, f"missing {k}: {kinds}"
    solved = next(e for e in L.events("t5", kinds=["solved"]))
    measured = next(e for e in L.events("t5", kinds=["measured"]))
    best = min(r["latency_ms"] for r in rows if r.get("ok") and r.get("correct")) * 1e3
    print("solver pick", solved["payload"]["values"], "predicted", round(solved["payload"]["objective"], 1), "us; measured",
          round(measured["payload"]["measured"], 1), "us; sweep best", round(best, 1), "us; gap", f"{measured['payload']['measured'] / best - 1:+.1%}")
    assert measured["payload"]["reference_agrees"] is True

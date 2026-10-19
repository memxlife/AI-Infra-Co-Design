"""R7: the loop reacts once, replayed without the GPU. One wrong chip term is injected: the DRAM rate at an eighth of
its measured value (the brief says halve the effective L2, but at 4096^3 every panel pick is tensor-bound in the model,
so a wrong L2 term changes neither the pick nor its prediction; a wrong DRAM rate does). The model then over-charges the
DRAM term, the kernel runs faster than predicted, diagnose(model) fires, B_HBM is re-fitted from that one measurement
(to the rate it proves, 29 -> 37 GB/s), the solver runs again and the new pick runs. Under round 2's terms the re-solve
moved the pick; under round 3's envelopes one measurement moves the shared rate only to what it proves, and the point
with the least traffic stays the pick, so the chain is asserted and the pick change is reported, not required.
Measurements come from the 4096^3 sweep table."""
import json
from pathlib import Path

import pytest

from agentic.agents.formulator.tilelang import TileLangFormulator
from agentic.agents.orchestrator import Orchestrator
from agentic.chips import load_chip
from agentic.ledger import Ledger
from agentic.tilelang.calibrate import load_terms

ROWS = Path(__file__).resolve().parents[1] / "runs" / "tilelang" / "sweep_4096.jsonl"


def canned(rows):
    import re
    def run_kernel(host, src, m, n, k, holes, func="matmul", reps=100, warmup=25, ncu=False, inout_c=False):
        bm = int(re.search(r"alloc_shared\(\((\d+), (\d+)\)", src).group(1)); bk = int(re.search(r"alloc_shared\(\((\d+), (\d+)\)", src).group(2))
        bn = int(re.search(r"B_shared = T.alloc_shared\(\(\d+, (\d+)\)", src).group(1))
        st = int(re.search(r"num_stages=(\d+)", src).group(1)); th = int(re.search(r"threads=(\d+)", src).group(1))
        sw = re.search(r"use_swizzle\(panel_size=(\d+)\)", src); sw = int(sw.group(1)) if sw else 0
        want = {"block_M": bm, "block_N": bn, "block_K": bk, "num_stages": st, "threads": th, "swizzle": sw}
        r = next((r for r in rows if r["holes"] == want), None)
        assert r is not None, f"not in the sweep: {want}"
        return {**r, "ok": True, "host": "mock", "tilelang": {"version": "0.1.15"}, "cuda": {"nvcc": "13.0"}, "ncu": None, "kernel_file": "mock", "func": func}
    return run_kernel


@pytest.mark.skipif(not ROWS.exists(), reason="needs the 4096^3 sweep table")
def test_r7_loop_reacts_once(tmp_path, monkeypatch):
    rows = [json.loads(l) for l in ROWS.read_text().splitlines() if l.strip()]
    import agentic.agents.evaluator.tilelang as ev
    monkeypatch.setattr(ev, "run_kernel", canned(rows))
    chip = load_chip("spark-6951-gb10"); terms = dict(load_terms(chip))
    terms["B_HBM"] = terms["B_HBM"] / 8                                     # the injected wrong term
    L = Ledger(str(tmp_path / "r7.db"))
    o = Orchestrator(L, "r7", TileLangFormulator([{"M": 4096, "N": 4096, "K": 4096}], terms=terms), evaluator_backend="tilelang:mock", level=5,
                     params={"discrepancy_tol": 0.02})
    o.run_loop()
    kinds = [e["kind"] for e in L.events("r7")]
    solved = [e["payload"] for e in L.events("r7", kinds=["solved"])]
    measured = [e["payload"] for e in L.events("r7", kinds=["measured"])]
    notes = [e["payload"] for e in L.events("r7", kinds=["note"]) if "recalibration" in e["payload"]]
    print("events:", kinds)
    print("first pick", solved[0]["values"], "predicted", round(solved[0]["objective"], 1), "measured", round(measured[0]["measured"], 1))
    print("recalibration:", notes[0]["recalibration"].get("changed"))
    print("second pick", solved[1]["values"], "predicted", round(solved[1]["objective"], 1), "measured", round(measured[1]["measured"], 1), "changed:", solved[1]["pick_changed"])
    assert "diagnosis" in kinds and len(solved) == 2 and len(measured) == 2
    assert notes[0]["recalibration"]["changed"]["term"] == "B_HBM"
    assert notes[0]["recalibration"]["changed"]["to"] > notes[0]["recalibration"]["changed"]["from"]
    print("pick changed after the re-solve:", solved[1]["pick_changed"])

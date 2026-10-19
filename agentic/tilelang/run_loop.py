"""The loop for real (T5 and T6 of the brief): propose -> legalize -> solve -> legalize -> compile + profile on the Spark,
one kernel per shape, into a console ledger. Scripted Formulator on the three sweep shapes; the model Formulator's
recorded replies on the five new shapes. The GPU must be idle (no sweep running).

python -m agentic.tilelang.run_loop --ledger runs/gemm-tilelang.db --host dgx-spark [--sweep] [--new]"""
from __future__ import annotations

import argparse, json
from pathlib import Path

from ..agents.formulator.model import ModelFormulator
from ..agents.formulator.tilelang import TileLangFormulator
from ..agents.orchestrator import Orchestrator
from ..ledger import Ledger
from .calibrate import SHAPES

ROOT = Path(__file__).resolve().parents[2]
FIX = ROOT / "tests" / "fixtures" / "tilelang_model_replies.json"


def summarize(L: Ledger, run: str) -> list[dict]:
    out = []
    for e in L.events(run, kinds=["measured"]):
        m = L.get(e["artifact"]); d = m.get("detail", {})
        out.append({"shape": m["design"], "measured_us": m["measured_cost"], "predicted_us": m["predicted_cost"], "correct": m.get("reference_agrees"),
                    "n_regs": d.get("n_regs"), "n_spills": d.get("n_spills"), "tflops": d.get("tflops")})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--ledger", default="runs/gemm-tilelang.db"); ap.add_argument("--host", default="dgx-spark")
    ap.add_argument("--sweep", action="store_true"); ap.add_argument("--new", action="store_true")
    a = ap.parse_args()
    L = Ledger(a.ledger)
    if a.sweep:
        for name, shape in SHAPES.items():
            run = f"tilelang-{name}"
            Orchestrator(L, run, TileLangFormulator([shape]), evaluator_backend=f"tilelang:{a.host}", level=5).run_loop()
            print(run, json.dumps(summarize(L, run)))
    if a.new:
        recs = json.load(open(FIX)); shapes = [r["shape"] for r in recs]
        run = "tilelang-model-new-shapes"
        Orchestrator(L, run, ModelFormulator(shapes, backend="recorded", recorded=FIX), evaluator_backend=f"tilelang:{a.host}", level=5).run_loop()
        for e in L.events(run, kinds=["measured"]):
            m = L.get(e["artifact"]); d = m.get("detail", {})
            print(run, m["design"], "measured", round(m["measured_cost"], 1), "us; predicted", round(m["predicted_cost"], 1), "us; correct", m.get("reference_agrees"), "regs", d.get("n_regs"), "spills", d.get("n_spills"))


if __name__ == "__main__":
    main()

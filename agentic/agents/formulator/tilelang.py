"""The scripted TileLang Formulator (T5 of the brief): one proposal per shape, the gemm_basic family as a contract with
every hole open. No model calls; T6 puts a large model behind the same request."""
from __future__ import annotations

from ...tilelang.contract import family_contract


class TileLangFormulator:
    name = "tilelang-scripted"

    def __init__(self, shapes: list[dict], chip: dict | str = "spark-6951-gb10", family: str = "gemm_basic", terms: dict | None = None):
        from ...chips import load_chip
        from ...tilelang.calibrate import load_terms
        self.shapes, self.chip, self.family = list(shapes), chip, family
        self.terms = terms or load_terms(load_chip(chip) if isinstance(chip, str) else chip)   # the calibrated terms when they exist

    def propose(self, request: dict) -> dict | None:
        i = request["step"]
        if i >= len(self.shapes):
            return None
        s = self.shapes[i]
        return family_contract(self.family, s, self.chip, terms=self.terms, parent=request.get("coarse_template"), step=i,
                               refines=f"family {self.family} on {s['M']}x{s['N']}x{s['K']}",
                               rationale=(f"GEMM {s['M']}x{s['N']}x{s['K']}, FP16 in, FP32 accumulate, on the {self.family} family: TileLang's "
                                          "example kernel with the six integer knobs open. The solver picks the knobs from J-hat; the "
                                          "kernel is then compiled and timed on the Spark once."))

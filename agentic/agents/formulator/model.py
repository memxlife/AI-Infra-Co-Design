"""The large-model Formulator for TileLang families (T6 of the brief). The model reads prompts/tilelang_family.md, the
family list with its knob inventory, the shape, the chip and the evidence, and returns family + menus + structure +
rationale as JSON. It never writes source: the contract is built by family_contract from what it returns, and the menus
it returns may only narrow the inventory's menus. Two ways to reach a model, neither of which puts a key in the repo:
  anthropic   the Anthropic SDK with ANTHROPIC_API_KEY in the environment
  claude-cli  the logged-in `claude -p` command on this machine
  recorded    a file of saved model replies, for tests and for runs that must be replayable
"""
from __future__ import annotations

import hashlib, json, os, subprocess
from pathlib import Path

from ...chips import load_chip, scope_chip
from ...tilelang.contract import family_contract
from ...tilelang.tools import inventory, menus_for
from . import catalog_entry

PROMPT = Path(__file__).resolve().parent / "prompts" / "tilelang_family.md"
FAMILIES = ["gemm_basic"]
DEFAULT_MODEL = "claude-fable-5-1"


def build_request(shape: dict, chip: dict, evidence: list[dict], families=FAMILIES) -> str:
    inv = {f: inventory(f) for f in families}
    small_chip = {k: chip.get(k) for k in ("name", "sms", "smem_bytes_per_block", "regs_per_sm", "l2_bytes_effective", "dram_bytes_per_s", "tensor_flops_per_s_chip")}
    body = {"workload": {"op": "matmul", **shape, "dtype": "fp16", "accum": "fp32"}, "chip": small_chip,
            "families": {f: {"knobs": i["knobs"], "skinny_menus": i["skinny_menus"], "menus_for_this_shape": menus_for(f, shape)} for f, i in inv.items()},
            "structure_catalog": ["bottleneck_dominance", "concavity", "monotonicity", "near_decomposability", "scenario_decomposition",
                                  "low_effective_dimension", "symmetry", "tiny_menus", "dominance"],
            "evidence": evidence}
    return PROMPT.read_text() + "\n\n## This request\n\n```json\n" + json.dumps(body, indent=1, default=str) + "\n```\n"


def parse_reply(text: str) -> dict:
    s = text.strip()
    if "```" in s:
        s = s.split("```json")[-1].split("```")[0] if "```json" in s else s.split("```")[1]
    start, end = s.find("{"), s.rfind("}")
    return json.loads(s[start:end + 1])


def check_reply(reply: dict, shape: dict) -> list[str]:
    """What the brief forbids: an unknown family, a menu value outside the inventory, a structure not in the catalog."""
    problems = []
    fam = reply.get("family")
    if fam not in FAMILIES:
        return [f"family {fam!r} is not in the family list"]
    full = menus_for(fam, shape)
    for k, vals in (reply.get("menus") or {}).items():
        if k not in full:
            problems.append(f"knob {k!r} is not in the inventory"); continue
        if not vals or any(v not in full[k] for v in vals):
            problems.append(f"menu for {k} has values outside the inventory: {vals}")
    st = reply.get("structure") or {}
    if catalog_entry(st.get("name", "")) is None:
        problems.append(f"structure {st.get('name')!r} is not in the catalog")
    if not str(st.get("regime", "")).split(":")[0].strip().endswith("-bound"):
        problems.append("structure.regime must start with compute-bound, memory-bound or copy-bound")
    if "source" in json.dumps(reply).lower() and "T.Kernel" in json.dumps(reply):
        problems.append("the reply contains kernel source; the Formulator never writes source")
    return problems


class ModelFormulator:
    name = "tilelang-model"

    def __init__(self, shapes: list[dict], chip: dict | str = "spark-6951-gb10", backend: str = "auto", model: str = DEFAULT_MODEL,
                 recorded: str | Path | None = None, terms: dict | None = None):
        from ...tilelang.calibrate import load_terms
        self.shapes = list(shapes)
        self.chip_full = load_chip(chip) if isinstance(chip, str) else chip
        self.chip = scope_chip(self.chip_full)
        self.model = model
        self.recorded = json.load(open(recorded)) if recorded else None
        self.backend = backend if backend != "auto" else ("recorded" if recorded else "anthropic" if os.environ.get("ANTHROPIC_API_KEY") else "claude-cli")
        self.terms = terms or load_terms(self.chip_full)
        self.replies: list[dict] = []

    # ---- the model call
    def ask(self, prompt: str) -> str:
        if self.backend == "recorded":
            key = hashlib.sha256(prompt.encode()).hexdigest()[:16]
            hit = next((r for r in self.recorded if r.get("prompt_sha256", "")[:16] == key), None) or (self.recorded[len(self.replies)] if len(self.replies) < len(self.recorded) else None)
            if hit is None:
                raise RuntimeError("no recorded reply for this request")
            return hit["reply"]
        if self.backend == "anthropic":
            import anthropic
            msg = anthropic.Anthropic().messages.create(model=self.model, max_tokens=1500, messages=[{"role": "user", "content": prompt}])
            return "".join(getattr(b, "text", "") for b in msg.content)
        if self.backend == "claude-cli":
            out = subprocess.run(["claude", "-p", prompt, "--output-format", "json", "--model", self.model], capture_output=True, text=True, timeout=300)
            j = json.loads(out.stdout or "{}")
            if j.get("is_error"):
                raise RuntimeError(f"claude cli: {j.get('result')}")
            return j.get("result", "")
        raise ValueError(f"unknown backend {self.backend}")

    def propose(self, request: dict) -> dict | None:
        i = request["step"]
        if i >= len(self.shapes):
            return None
        shape = self.shapes[i]
        prompt = build_request(shape, self.chip, request.get("evidence", []))
        text = self.ask(prompt)
        reply = parse_reply(text)
        problems = check_reply(reply, shape)
        self.replies.append({"prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "reply": text, "problems": problems, "shape": shape})
        if problems:
            # a reply the rules refuse still becomes a contract, so the ledger shows the refusal; the orchestrator's
            # well_formed check and legalize decide. Menus fall back to the inventory's.
            reply["menus"] = {}
        T = family_contract(reply["family"], shape, self.chip, terms=self.terms, structure=reply.get("structure"),
                            rationale=reply.get("rationale", ""), parent=request.get("coarse_template"), formulator=self.name,
                            step=i, refines=f"family {reply['family']} on {shape['M']}x{shape['N']}x{shape['K']}",
                            model=self.model, prompt_hash=hashlib.sha256(prompt.encode()).hexdigest(), reply_problems=problems)
        for k, vals in (reply.get("menus") or {}).items():
            T["decisions"][k]["domain"] = list(vals)
        T["provenance"]["model"] = self.model; T["provenance"]["prompt_hash"] = hashlib.sha256(prompt.encode()).hexdigest()
        return T

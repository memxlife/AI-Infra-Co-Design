"""T6 of the TileLang brief: the model Formulator's replies on five new shapes, recorded for replay. The test: the reply's
menus and structure pass the rules, and the contract passes the family check (legalize) first time on at least 80 %."""
import json
from pathlib import Path

from agentic.agents.formulator.model import ModelFormulator, check_reply, parse_reply
from agentic.tilelang.verify import certify_family

FIX = Path(__file__).resolve().parent / "fixtures" / "tilelang_model_replies.json"


def test_t6_recorded_replies_pass_first_time(tmp_path):
    recs = json.load(open(FIX))
    shapes = [r["shape"] for r in recs]
    f = ModelFormulator(shapes, backend="recorded", recorded=FIX)
    passed = 0
    for i, s in enumerate(shapes):
        T = f.propose({"step": i, "level": 5, "coarse_template": None, "evidence": [], "population_hits": []})
        T["id"] = f"probe{i}"
        problems = check_reply(parse_reply(recs[i]["reply"]), s)
        cert = certify_family(T)
        ok = not problems and cert["verdict"] == "proved" and all(len(T["decisions"][k]["domain"]) >= 1 for k in T["decisions"])
        passed += ok
        print(s, "menus narrowed to", {k: v["domain"] for k, v in T["decisions"].items()}, "| problems:", problems, "| family:", cert["verdict"])
        assert T["structure"]["name"] and T["provenance"]["model"]
    assert passed / len(shapes) >= 0.8, f"{passed} of {len(shapes)} passed first time"

"""Skeleton 1 (gemm_pipelined): legalize the family's menu at a shape and solve it exactly against the mechanistic model with
the cp.async constants (enumeration: the menu is ~1300 points, so the exact optimum of the model is cheaper than the MILP's
tables and identical to what the MILP would return). Writes runs/tilelang/skel/pipelined_<shape>.json and the rendered pick.
python -m agentic.tilelang.skeletons.solve_pipelined 1024 1024 1024"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import yaml
from ..tools import legalize, render
from ..model.mech import predict
ROOT = Path(__file__).resolve().parents[3]; FAM = ROOT / "agentic" / "tilelang" / "families" / "gemm_pipelined"


def menu(shape: dict, narrowed: dict | None = None) -> dict:
    y = yaml.safe_load((FAM / "knobs.yaml").read_text())
    m = {k["name"]: list(k["menu"]) for k in y["knobs"]}
    if shape["M"] < 64: m["block_M"] = y["skinny_menus"]["block_M"]
    for k, v in (narrowed or {}).items():
        assert set(v) <= set(m[k]), (k, v, m[k]); m[k] = v
    return m


def solve(shape: dict, narrowed: dict | None = None, top: int = 8) -> dict:
    m = menu(shape, narrowed); names = list(m); rows = []
    for vals in itertools.product(*(m[n] for n in names)):
        h = dict(zip(names, vals))
        L = legalize("gemm_basic", h, shape)
        if not L.get("ok", L.get("legal")): continue
        p = predict(h, shape, family="gemm_pipelined")
        rows.append({"holes": h, "expected_us": p["expected_us"], "floor_us": p["floor_us"], "constants_from": p["constants_from"], "parts_us": p["parts_us"], "waves_eff": p["waves_eff"]})
    # ties (same tile and threads, another stage count: the ablation has no stage term) go to the point whose constants were measured
    rows.sort(key=lambda r: (round(r["expected_us"], 1), 0 if "ablation" in r["constants_from"] else 1, r["holes"]["num_stages"]))
    return {"shape": shape, "menu": m, "legal_points": len(rows), "pick": rows[0], "top": rows[:top]}


if __name__ == "__main__":
    M, N, K = (int(a) for a in sys.argv[1:4]); shape = {"M": M, "N": N, "K": K}
    narrowed = json.loads(sys.argv[4]) if len(sys.argv) > 4 else None
    r = solve(shape, narrowed); tag = f"{M}x{N}x{K}"
    out = ROOT / "runs" / "tilelang" / "skel"; json.dump(r, open(out / f"pipelined_{tag}.json", "w"), indent=1)
    (out / f"pipelined_pick_{tag}.py").write_text(render("gemm_basic", r["pick"]["holes"]))
    print(f"{r['legal_points']} legal points; pick {r['pick']['holes']} expected {r['pick']['expected_us']:.1f} us ({r['pick']['constants_from']})")
    for t in r["top"]: print(f"   {t['expected_us']:7.1f} us  {t['holes']}  waves {t['waves_eff']:.2f}  {t['constants_from']}")

"""The GB10 functional model: every constant names its source, the traffic simulation matches ncu, the model ranks the
measured sets and names ncu's bottleneck. Thresholds are the levels reached on Oct 20 2026, kept as regressions."""
import json
from pathlib import Path
from agentic.gb10 import model
from agentic.gb10.evaluate import load_sets, score
from agentic.gb10.fidelity import ablation_check, ncu_check

ROOT = Path(__file__).resolve().parents[1]


def test_every_constant_names_its_source():
    raw = json.load(open(ROOT / "calibration" / "spark-6951" / "gb10.json"))
    for k, v in raw.items():
        if k.startswith("_"): continue
        assert v.get("source", "").split(":")[0] in ("spec", "probe", "probe (round 2)", "ablation", "ablation (step 2c)"), k


def test_design_numbers():
    assert model.SMS == 48 and abs(model.C["tensor_flop_per_clk_per_sm"] * model.F / 1e12 - 2.58) < 0.01
    assert model.bank_conflict_ways(64) == 4 and model.bank_conflict_ways(128) == 8 and model.bank_conflict_ways(16) == 1


def test_launch_order_matches_tilelang_rasterization():
    # rasterization2DRow, panel 2, 3 x 4 grid: serpentine over panels of two rows
    order = list(model.launch_order(3, 4, 2))
    assert order[:8] == [(0, 0), (1, 0), (0, 1), (1, 1), (0, 2), (1, 2), (0, 3), (1, 3)] and order[8:] == [(2, 3), (2, 2), (2, 1), (2, 0)]
    assert list(model.launch_order(2, 2, 0)) == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_traffic_simulation_matches_ncu():
    for r in ncu_check():
        assert 0.65 <= r["sim_dram_MB"] / r["ncu_dram_MB"] <= 1.1, r["pick"]


def test_bottleneck_matches_ncu_and_ablation():
    assert sum(r["ncu"] == r["model"] for r in ncu_check()) >= 6
    assert sum(r["ablation"] == r["model"] for r in ablation_check()) >= 16


def test_ranking_on_measured_sets():
    sets = load_sets()
    for name, rho_min, regret_max in (("r3 oos 2048x2048x2048 (mixed, do_bench)", 0.8, 0.02), ("r3 oos 4096x4096x1024 (mixed, do_bench)", 0.8, 0.02),
                                       ("r3 oos 8192x8192x8192 (mixed, do_bench)", 0.9, 0.02), ("r1 4096x4096x4096 (TMA, do_bench)", 0.75, 0.02),
                                       ("r2 sample 1024x1024x1024 (mixed, do_bench)", 0.8, 0.06), ("r1 32x4096x4096 (TMA, do_bench)", 0.6, 0.03)):
        s = score(sets[name])
        assert s["spearman"] >= rho_min and s["regret"] <= regret_max, (name, s["spearman"], s["regret"])

"""Round 3, S2 revised: the static stage-release check and the skeleton families' files."""
import re
from pathlib import Path
from agentic.tilelang.skeletons.release_check import check

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs" / "tilelang" / "skel" / "src"


def test_tma_ws_kernel_releases_after_last_read():
    r = check((SRC / "src_fam_tma_ws.cu").read_text())
    assert r["status"] == "OK" and r["loops_with_release"] == 1 and "A_shared" in r["filled"] and "B_shared" in r["filled"]


def test_synthetic_late_read_is_flagged():
    # move the consumer's back-pressure arrive above the ldmatrix reads of the same iteration: the #3451 shape
    s = (SRC / "src_fam_tma_ws.cu").read_text()
    lines = s.splitlines()
    a = next(i for i, l in enumerate(lines) if re.search(r"mbarrier\[[^\]]*\]\.arrive\(\);", l))
    w = next(i for i, l in enumerate(lines) if ".wait(" in l and i > 60)      # the consumer's forward wait
    arrive = lines.pop(a); lines.insert(w + 1, arrive)
    r = check("\n".join(lines))
    assert r["status"] == "RACE" and r["races"][0]["reads_after"]


def test_cpasync_kernels_have_no_producer_group():
    for f in ("src_fam_cpasync_ws.cu", "src_fam_cpasync_nows.cu", "src_skel2_1024x1024x1024_16x128x32_st2_th64_sm48.cu"):
        assert check((SRC / f).read_text())["status"] == "NOT-WS"


def test_skeleton_families_record_upstream():
    for fam in ("gemm_pipelined", "gemm_streamk", "gemv_splitk", "gemm_frag_w4a16"):
        up = (ROOT / "agentic" / "tilelang" / "families" / fam / "UPSTREAM.md").read_text()
        assert "fe6d319c78c0738186fc6e83eb8436ddef65204d" in up or "e6b003e9a6bad49c86865285ee42ecbc9be9bb05" in up

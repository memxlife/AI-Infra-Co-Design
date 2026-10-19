"""Step 0: read ncu's section text for cuBLASLt and for our pick and print one table per shape.
python -m agentic.tilelang.ncu_table runs/tilelang/ncu"""
from __future__ import annotations
import re, sys
from pathlib import Path


def parse(path: Path) -> dict:
    t = path.read_text(errors="replace")
    out = {"file": path.name}
    m = re.search(r"^\s*(\S.*?)\s*\((\d+), (\d+), (\d+)\)x\((\d+), (\d+), (\d+)\),", t, re.M)
    if m:
        out["kernel"] = m.group(1)[:90]; out["grid"] = f"{m.group(2)}x{m.group(3)}x{m.group(4)}"; out["block"] = f"{m.group(5)}x{m.group(6)}x{m.group(7)}"
    def num(label, unit=None):
        mm = re.search(r"^\s*" + re.escape(label) + r"\s+(\S+)\s+([\d.,]+)\s*$", t, re.M)
        if not mm:
            mm = re.search(r"^\s*" + re.escape(label) + r"\s+([\d.,]+)\s*$", t, re.M)
            return float(mm.group(1).replace(",", "")) if mm else None
        v = float(mm.group(2).replace(",", "")); u = mm.group(1)
        if unit == "bytes":
            v *= {"byte": 1, "Kbyte": 1e3, "Mbyte": 1e6, "Gbyte": 1e9}.get(u.split("/")[0], 1)
        if unit == "us":
            v *= {"ns": 1e-3, "us": 1, "usecond": 1, "ms": 1e3, "msecond": 1e3, "nsecond": 1e-3}.get(u, 1)
        return v
    out["duration_us"] = num("Duration", "us")
    out["regs"] = num("Registers Per Thread"); out["smem_kb"] = ((num("Dynamic Shared Memory Per Block", "bytes") or 0) + (num("Static Shared Memory Per Block", "bytes") or 0)) / 1e3
    out["theoretical_occ_pct"] = num("Theoretical Occupancy"); out["achieved_occ_pct"] = num("Achieved Occupancy")
    out["blocks_limit_regs"] = num("Block Limit Registers"); out["blocks_limit_smem"] = num("Block Limit Shared Mem"); out["blocks_limit_sm"] = num("Block Limit SM")
    out["waves"] = num("Waves Per SM")
    out["tensor_pct"] = num("Tensor (All) Pipe Active") if "Tensor (All)" in t else None
    for key, lab in (("mem_busy_pct", "Mem Busy"), ("l2_hit_pct", "L2 Hit Rate"), ("l1_hit_pct", "L1/TEX Hit Rate"), ("mem_pct", "Memory Throughput"), ("compute_pct", "Compute (SM) Throughput")):
        out[key] = num(lab)
    def metric(name):
        mm = re.search(re.escape(name) + r"\s+(\S+)?\s*([\d.,]+)\s*$", t, re.M)
        return float(mm.group(2).replace(",", "")) if mm else None
    rd = metric("lts__t_sectors_op_read_lookup_miss.sum"); wr = metric("lts__t_sectors_op_write_lookup_miss.sum"); l2r = metric("lts__t_sectors_op_read.sum")
    out["dram_mb"] = ((rd or 0) + (wr or 0)) * 32 / 1e6 if rd is not None else None
    out["l2_read_mb"] = (l2r or 0) * 32 / 1e6 if l2r is not None else None
    out["tensor_metric_pct"] = metric("sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active")
    out["warps_active_pct"] = metric("sm__warps_active.avg.pct_of_peak_sustained_active")
    stalls = re.findall(r"Stall (\w[\w ]*?)\s+[\d.]+ cycles", t)
    sec = t[t.find("Section: Warp State Statistics"):]
    mm = re.search(r"spends ([\d.]+) cycles being stalled ([^.]+?)(?:\. |\.$|\n)", sec)
    out["top_stall"] = (mm.group(2).strip()[:60] + f" ({mm.group(1)} cycles)") if mm else None
    return out


def main(d: str):
    d = Path(d)
    rows = {}
    for f in sorted(d.glob("*.txt")):
        who, shape = f.stem.split("_", 1)
        rows.setdefault(shape, {})[who] = parse(f)
    cols = ["kernel", "grid", "block", "regs", "smem_kb", "theoretical_occ_pct", "achieved_occ_pct", "blocks_limit_regs", "blocks_limit_smem", "waves",
            "duration_us", "tensor_metric_pct", "warps_active_pct", "dram_mb", "l2_read_mb", "mem_busy_pct", "l2_hit_pct", "l1_hit_pct", "top_stall"]
    for shape, r in rows.items():
        print(f"\n### {shape}\n\n| counter | cuBLASLt | ours |\n|---|---|---|")
        for c in cols:
            a = r.get("cublaslt", {}).get(c); b = r.get("ours", {}).get(c)
            fmt = lambda v: "–" if v is None else (f"{v:.1f}" if isinstance(v, float) else str(v))
            print(f"| {c} | {fmt(a)} | {fmt(b)} |")


if __name__ == "__main__":
    main(sys.argv[1])

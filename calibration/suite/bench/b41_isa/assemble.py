"""B41 item 2: PTX for every probe, then the same text assembled for each of six targets by rewriting only .target.
nvcc -ptx validates PTX for its own target, so each probe's PTX comes from the first virtual target that accepts it
(compute_121a, 100a, 103a, 90a). A probe no target accepts (e.g. an illegal modifier pairing) takes a valid sibling's
PTX with only the instruction text replaced, so ptxas reports its error for every target.
Writes isa_assemble.tsv: probe, target, status, first ptxas error, where the PTX came from.
python3 assemble.py WORKDIR   (WORKDIR/probes/*.cu; cubins land in WORKDIR/cubin/)"""
import re, subprocess, sys
from pathlib import Path

import os
# a toolkit only knows the targets of its release (CUDA 12.8 has no sm_121 and no family "f" targets): override with
# B41_TARGETS / B41_ARCHS, space-separated
TARGETS = os.environ.get("B41_TARGETS", "sm_120a sm_120f sm_121a sm_121f sm_100a sm_103a").split()
ARCHS = os.environ.get("B41_ARCHS", "compute_121a compute_100a compute_103a compute_90a").split()
SIBLING = {"01a_mxf4nvf4_2X_ue4m3_plan": "01c_mxf4nvf4_4X_ue4m3", "01b_mxf4nvf4_2X_ue4m3": "01c_mxf4nvf4_4X_ue4m3"}
W = Path(sys.argv[1]); (W / "ptx").mkdir(exist_ok=True); (W / "cubin").mkdir(exist_ok=True)


def instr(cu: str) -> str:
    return re.search(r'asm volatile\("([a-z][\w.:]+)', cu).group(1)


def first_error(text: str) -> str:
    for line in text.splitlines():
        if "error" in line.lower():
            return re.sub(r"^.*?error\s*:\s*", "", line).strip()
    return text.strip().splitlines()[0] if text.strip() else ""


ptx, source = {}, {}
for f in sorted((W / "probes").glob("*.cu")):
    n = f.stem
    for a in ARCHS:
        r = subprocess.run(["nvcc", "-ptx", f"-arch={a}", "-o", str(W / "ptx" / f"{n}.ptx"), str(f)], capture_output=True, text=True)
        if r.returncode == 0:
            ptx[n] = (W / "ptx" / f"{n}.ptx").read_text(); source[n] = a; break
for n, sib in SIBLING.items():                      # rows no target accepts: the sibling's PTX with this instruction
    if n not in ptx and sib in ptx:
        mine, theirs = instr((W / "probes" / f"{n}.cu").read_text()), instr((W / "probes" / f"{sib}.cu").read_text())
        ptx[n] = ptx[sib].replace(theirs, mine); source[n] = f"{sib} with this instruction"
rows = []
for f in sorted((W / "probes").glob("*.cu")):
    n = f.stem
    if n not in ptx:
        rows.append((n, "all", "no-ptx", "no virtual target accepted it; see err", "")); continue
    for X in TARGETS:
        p = W / "ptx" / f"{n}__{X}.ptx"; p.write_text(re.sub(r"^\.target \S+", f".target {X}", ptx[n], flags=re.M))
        c = W / "cubin" / f"{n}__{X}.cubin"
        r = subprocess.run(["ptxas", f"--gpu-name={X}", "-o", str(c), str(p)], capture_output=True, text=True)
        rows.append((n, X, "assembles" if r.returncode == 0 else "rejected", "" if r.returncode == 0 else first_error(r.stderr), source[n]))
        if r.returncode: c.unlink(missing_ok=True)
(W / "isa_assemble.tsv").write_text("".join("\t".join(r) + "\n" for r in rows))
ver = re.search(r"^\.version \S+", next(iter(ptx.values())), re.M).group(0)
pv = subprocess.run(["ptxas", "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
(W / "isa_versions.txt").write_text(f"PTX ISA {ver}; ptxas {pv}\n")
print(len(rows), "rows;", sum(r[2] == "assembles" for r in rows), "assemble;", f"PTX ISA {ver}; ptxas {pv}")

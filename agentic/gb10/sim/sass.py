"""Parse nvdisasm -hex output into instructions with the compiler's scheduling control bits.

Each Volta-and-later SASS instruction is 128 bits; the scheduling control sits in bits 105-125, i.e. bits 41-61 of the
second 64-bit word:  stall (4 bits, cycles before this warp may issue again), yield (1), write barrier (3: which of
six dependency counters this instruction will set until its result is ready, 7 = none), read barrier (3: counter set
until its source registers are read), wait mask (6: counters that must be clear before this instruction issues),
reuse (4). These are what the hardware warp scheduler obeys."""
from __future__ import annotations
import re
from dataclasses import dataclass, field


@dataclass
class Ins:
    addr: int
    pred: str
    op: str            # full opcode with modifiers, e.g. HMMA.16816.F32
    base: str          # opcode without modifiers, e.g. HMMA
    args: str
    stall: int
    yld: int
    wbar: int          # 0-5, or None
    rbar: int          # 0-5, or None
    wait: int          # bit mask over 0-5
    text: str = ""
    target: int | None = None   # branch target address


LINE = re.compile(r"/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;\s*/\*\s*(0x[0-9a-f]{16})\s*\*/")
HI = re.compile(r"^\s*/\*\s*(0x[0-9a-f]{16})\s*\*/\s*$")


def decode_ctrl(hi: int) -> dict:
    c = hi >> 41
    stall = c & 0xF; yld = (c >> 4) & 1; wb = (c >> 5) & 7; rb = (c >> 8) & 7; wait = (c >> 11) & 0x3F
    return {"stall": stall, "yld": yld, "wbar": None if wb == 7 else wb, "rbar": None if rb == 7 else rb, "wait": wait}


def parse(text: str, kernel: str | None = None) -> list[Ins]:
    lines = text.splitlines(); out = []; i = 0; in_k = kernel is None
    labels, pending = {}, []
    while i < len(lines):
        l = lines[i]
        lm = re.match(r"^(\.L_[A-Za-z0-9_]+):", l)
        if lm: pending.append(lm.group(1)); i += 1; continue
        if kernel and ".text." in l: in_k = kernel in l
        m = LINE.search(l)
        if m and in_k and i + 1 < len(lines):
            h = HI.match(lines[i + 1])
            if h:
                addr = int(m.group(1), 16); body = m.group(2).strip()
                pm = re.match(r"(@!?U?P[T0-9]+)\s+(.*)", body); pred = pm.group(1) if pm else ""; body = pm.group(2) if pm else body
                op, _, args = body.partition(" ")
                ctrl = decode_ctrl(int(h.group(1), 16))
                tgt = None
                for lab in pending: labels[lab] = addr
                pending = []
                if op.startswith("BRA"):
                    t = re.search(r"\(([.A-Za-z0-9_]+)\)", args); tgt = t.group(1) if t else None
                out.append(Ins(addr, pred, op, op.split(".")[0], args.strip(), text=body, target=tgt, **ctrl))
                i += 2; continue
        i += 1
    for x in out:
        if isinstance(x.target, str): x.target = labels.get(x.target)
    return out


def main_loop(ins: list[Ins]) -> tuple[int, int]:
    """(start, end) indices of the longest backward-branch loop: the k-loop of a GEMM."""
    best = None
    for j, x in enumerate(ins):
        if x.base == "BRA" and x.target is not None and x.target < x.addr:
            s = next(k for k, y in enumerate(ins) if y.addr == x.target)
            if best is None or j - s > best[1] - best[0]: best = (s, j)
    return best

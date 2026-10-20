"""An instruction-level, event-driven simulator of one GB10 SM running TileLang GEMM machine code.

It runs the real SASS (nvdisasm -hex) of the kernel, warp by warp, and computes nothing: it tracks when each
instruction may issue and which resources it holds. Per sub-partition (4 per SM) one instruction issues per cycle,
from a warp whose stall count has elapsed and whose wait mask is clear (the compiler's control bits). Resources:
  tensor pipe per sub-partition (HMMA.16816 holds it 16 cycles), ldmatrix (per warp issue interval; 4 cycles of the SM's
  shared-memory port per x4), cp.async (LDGSTS: one warp instruction per 32 cycles through the SM's fill path, then the
  L2 or DRAM round trip), copy groups (LDGDEPBAR / DEPBAR.LE), the block barrier (BAR.SYNC), the six dependency counters
  (write and read barriers). Control flow: the k-loop's back edge is taken trip-count times; EXIT ends a warp.
Constants: calibration/spark-6951/gb10.json["sim"], each measured by a probe except two marked as assumptions."""
from __future__ import annotations
import heapq, json, math, re
from dataclasses import dataclass, field
from pathlib import Path
from .sass import Ins, parse, main_loop

ROOT = Path(__file__).resolve().parents[3]
P = json.load(open(ROOT / "calibration" / "spark-6951" / "gb10.json"))["sim"]["value"]


FMA_FP32 = {"FFMA", "FADD", "FMUL"}
FMA_2 = {"IMAD", "HFMA2", "HADD2", "HMUL2", "IMUL"}
ALU_2 = {"LOP3", "PRMT", "SHF", "IADD3", "ISETP", "SEL", "LEA", "F2FP", "IMNMX", "FMNMX", "FSETP", "FSEL", "PLOP3", "LOP", "IABS", "SHL", "SHR",
         "I2FP"}                                                     # cvt.cu: I2FP.F32.S32 + IADD3 = 4.03 cycles, one unit
XU_OPS = {"F2I", "I2F", "F2F", "FRND", "F2IP"}                     # cvt.cu: I2F.F16 8.06 cycles, shares the unit with MUFU


def unit(x) -> str | None:
    if x.op.startswith("HADD2.F32"): return "either"                # the half-to-float conversion runs on either unit
    if x.base in FMA_FP32 or x.base in FMA_2: return "fma"
    if x.base in ALU_2: return "alu"
    return None


def unit_clk(x) -> float:
    if x.base in FMA_FP32: return P["fma_fp32_clk"]
    return P["alu_half_rate_clk"]


@dataclass
class Warp:
    wid: int; block: int; smsp: int
    pc: int = 0; ready: int = 0; done: bool = False
    sb: list = field(default_factory=lambda: [0] * 6)
    loops: int = 0
    gid: int = 0                                    # the copy group being filled (cp.async between two commits)
    pending: dict = field(default_factory=dict)     # group -> copies not yet written into shared memory
    committed: dict = field(default_factory=dict)   # group -> dependency counter its commit set
    at_bar: bool = False
    ldsm_next: int = 0
    ncopy: int = 0


class SM:
    def __init__(self, ins: list[Ins], threads: int, blocks: int, trips: int, mem: str = "l2", stats: bool = True, active_sms: int = 1, copy_lines: list | None = None,
                 miss_frac: float = 0.0, dram_clk_per_512B: float = 0.0, ctile_bytes: int = 0, ldsm_ways: tuple = (1, 1),
                 ldg_lines: list | None = None, red_lines: int = 8, loop_trips: dict | None = None, resident: int | None = None,
                 grid_blocks: int | None = None, grid_dim: tuple | None = None):
        self.ins = ins; self.trips = trips; self.mem = mem
        # this SM's share of the chip's L2 return bandwidth (or its own return path, if fewer SMs are pulling)
        self.ret_clk = max(P["l2_return_clk_per_512B"], 512 * active_sms / P["l2_aggregate_B_per_clk_cpasync"])
        lp = main_loop(ins); self.loop_end = lp[1] if lp else -1     # fully unrolled code has no back edge
        self.warps = []; wpb = threads // 32
        for b in range(blocks):
            for w in range(wpb): self.warps.append(Warp(wid=len(self.warps), block=b, smsp=w % 4))
        self.wpb = wpb; self.blocks = blocks
        # block replacement: `blocks` run on this SM, at most `resident` at a time; when one exits the block scheduler
        # starts the next on the freed slot (the order blocks are launched)
        self.resident = resident or blocks; self.next_block = self.resident; self.live = {}; self.finished = 0
        # the chip's shares follow the grid: this SM is the busiest one; with every SM progressing alike, when it has
        # finished j blocks the chip holds sum over SMs of min(resident, blocks left there) blocks in flight, and this
        # SM's share of the L2, DRAM, store and TMA limits is its own blocks in flight over that
        self.grid_blocks = grid_blocks; self.eq0 = active_sms; self.dram_clk0 = dram_clk_per_512B
        INF = float("inf")
        for w in self.warps:
            if w.block >= self.resident: w.ready = INF
        for b in range(min(blocks, self.resident)): self.live[b] = wpb
        self.tensor_free = [0] * 4; self.smem_free = 0; self.fill_free = 0; self.mio_free = [0] * 4; self.mufu_free = [0] * 4
        # the arithmetic units of a sub-partition (probes alu.cu, alumix.cu): an FMA unit (FP32 add/multiply 1 cycle;
        # IMAD and packed half math 2 cycles, which also hold off FP32) and an ALU unit (logic, shifts, byte permutes,
        # integer add/compare, half<->float conversions, int->fp32 I2FP: 2 cycles); I2F (int->fp16), F2I, F2F use the SFU
        self.fma_free = [0] * 4; self.alu_free = [0] * 4
        # lines each copy of a k-tile touches, in program order (A rows then B rows); cycles them per warp
        self.copy_lines = copy_lines or [4]; self.lsu_free = 0
        # the store path is shared with the partner SM of the pair when the chip runs more than one block
        self.stg_share = 2 if active_sms >= 2 else 1; self.stg_free = 0; self.active_sms = active_sms
        # copies that miss the L2 (this wave's share, from the launch-order L2 simulation) pay the DRAM round trip and
        # this SM's share of the DRAM rate; misses are spread evenly over the copies
        self.miss_frac = miss_frac; self.miss_acc = 0.0; self.dram_clk = dram_clk_per_512B; self.dram_free = 0
        # TMA stores (UTMASTG): the block's C tile split evenly over the store boxes in the code; the TMA unit reads the
        # shared-memory array as fast as the pair's store port drains it (probe tmastore.cu: 33 B/clk per pair, chip
        # 482 B/clk, 457 cycles fixed; waiting for the read alone costs the same as waiting for the writes)
        n_st = sum(1 for y in ins if y.base == "UTMASTG"); self.tma_st_bytes = ctile_bytes / n_st if n_st else 0.0
        for w in self.warps:
            w.tid0 = (w.wid % wpb) * 32; w.regs_tid = set(); w.preds = {}; w.st_done = 0.0
            w.vals = {"RZ": 0, "URZ": 0}           # warp-uniform integer values known so far (lane 0 stands for the warp)
        self.use_values = True
        # block indices: this SM's j-th block is the grid's block j x 48 (blocks go round the SMs in launch order)
        self.grid_dim = grid_dim
        self.exited = {}            # block -> warps that have exited (they count as arrived at a block-wide barrier)
        # shared-memory bank conflicts of ldmatrix: (A reads, LDSM.M88; B reads, LDSM.MT88), from the tile layout (probe
        # smem_bw.cu: 4 cycles per x4 times the conflict ways, 1/2/4/8-way at row strides 48/32/64/128+ bytes)
        self.ldsm_ways = ldsm_ways
        # the SM's store/atomic queue (probe stq.cu: about 16 warp instructions are taken before issue waits for the
        # drain): finish times of the queued STG/RED instructions
        self.stq = []
        # plain global loads (LDG): per instruction in program order, (128-byte lines the warp touches, bytes it uses,
        # whether L1 already holds them); cycled per warp. An L1 hit costs 2 cycles per line of the L1 data path and its
        # latency; a miss takes the copy path (address stage, L2 or DRAM, the SM's return port) into registers
        self.ldg_lines = ldg_lines or [(4, 512, False)]; self.l1_free = 0; self.red_free = 0
        # global atomics: lines one warp instruction touches (8 for a C tile added from mma accumulators: 8 rows x 32 B)
        self.red_lines = red_lines
        # trip counts of other loops, by the index of their back edge (the main loop keeps `trips`)
        self.loop_trips = loop_trips or {}
        for w in self.warps: w.nldg = 0
        self.events = []            # (time, seq, kind, warp, arg)
        self.seq = 0; self.bar_arrived = {}; self.last_issue = [None] * 4
        self.count = {"issued": 0, "HMMA": 0, "LDSM": 0, "LDGSTS": 0, "BAR": 0}
        self.busy = {"tensor": 0, "smem": 0, "fill": 0}

    def ev(self, t, kind, w, arg=None):
        self.seq += 1; heapq.heappush(self.events, (t, self.seq, kind, w, arg))

    def on_event(self, kind, w, arg, now):
        pass

    # --- warp-uniform predicates from the thread index (lane 0 of the warp stands for the warp) ----------------------
    def _pred(self, w, tok):
        neg = tok.startswith("!"); t = tok[1:] if neg else tok
        if t in ("PT", "UPT"): v = True
        elif t in w.preds: v = w.preds[t]
        else: return None
        return (not v) if neg else v

    @staticmethod
    def _imm(b):
        return int(b, 16) if b.lstrip("-").startswith("0x") else int(b) if re.match(r"^-?\d+$", b) else None

    # --- warp-uniform integer values: loop counters, immediates, thread index ----------------------------------------
    def _val(self, w, tok):
        t = tok.strip(); neg = t.startswith("-"); t = t.lstrip("-").replace(".reuse", "")
        if t.startswith("!"): return None
        v = self._imm(t) if re.match(r"^-?(0x[0-9a-f]+|\d+)$", t) else w.vals.get(t)
        if v is None: return None
        return (-v) & 0xFFFFFFFF if neg else v & 0xFFFFFFFF

    def _eval_values(self, w, x):
        """Compute the destination of simple integer instructions from known operands; forget it otherwise."""
        args = [a.strip() for a in x.args.split(",")]
        if not args or not re.match(r"^U?R\d+$", args[0]): return
        d = args[0]; op = x.op; v = None
        try:
            if x.base == "S2R" or x.base == "S2UR":
                if "SR_TID.X" in x.args: v = w.tid0
                elif "SR_LANEID" in x.args: v = 0
                elif "SR_CTAID" in x.args and self.grid_dim:
                    gx, gy, gz = (list(self.grid_dim) + [1, 1])[:3]; lin = (w.block * 48) % (gx * gy * gz)
                    v = {"X": lin % gx, "Y": (lin // gx) % gy, "Z": lin // (gx * gy)}[x.args.split("SR_CTAID.")[1][0]]
            elif x.base in ("MOV", "UMOV", "MOV32I") or op.startswith("IMAD.MOV"):
                v = self._val(w, args[-1] if not op.startswith("IMAD.MOV") else args[-1])
                if op.startswith("IMAD.MOV"): v = self._val(w, args[3]) if len(args) > 3 else None
            elif x.base in ("IADD3", "UIADD3"):
                terms = [a for a in args[1:] if not re.match(r"^!?U?P(T|\d)$", a)]
                vs = [self._val(w, a) for a in terms]
                if terms and None not in vs: v = sum(vs) & 0xFFFFFFFF
            elif x.base in ("VIADD", "UIADD"):
                a, b = self._val(w, args[1]), self._val(w, args[2]); v = None if None in (a, b) else (a + b)
            elif x.base in ("IMAD", "UIMAD") and ".WIDE" not in op and ".HI" not in op:
                a, b, c = (self._val(w, t) for t in args[1:4])
                if None not in (a, b, c): v = a * b + c
            elif x.base in ("LEA", "ULEA") and ".HI" not in op:
                a, b, sh = self._val(w, args[1]), self._val(w, args[2]), self._val(w, args[3])
                if None not in (a, b, sh): v = (a << sh) + b
            elif x.base in ("SHF", "USHF") and len(args) >= 4:
                lo, sh, hi = self._val(w, args[1]), self._val(w, args[2]), self._val(w, args[3])
                if ".L." in op and ".HI" not in op and None not in (lo, sh): v = lo << sh
                elif ".R." in op and ".HI" in op and None not in (hi, sh): v = hi >> sh
                elif ".R." in op and None not in (lo, sh) and (hi in (0, None) and args[3] in ("RZ", "URZ")): v = lo >> sh
            elif x.base in ("LOP3", "ULOP3") and len(args) >= 5:
                a, b, c, lut = self._val(w, args[1]), self._val(w, args[2]), self._val(w, args[3]), self._val(w, args[4])
                if None not in (a, b, c, lut):
                    v = 0
                    for bit in range(32):
                        i = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1); v |= ((lut >> i) & 1) << bit
            elif x.base in ("SEL", "USEL") and len(args) >= 4:
                pv = self._pred(w, args[3]); a, b = self._val(w, args[1]), self._val(w, args[2])
                if pv is not None: v = a if pv else b
            elif x.base in ("R2UR",):
                v = self._val(w, args[1])
        except (ValueError, TypeError, IndexError):
            v = None
        if v is None:
            w.vals.pop(d, None)
            if ".64" in op or ".WIDE" in op: w.vals.pop("%s%d" % (d.rstrip("0123456789"), int(re.sub(r"\D", "", d)) + 1), None)
        else:
            w.vals[d] = v & 0xFFFFFFFF

    def _eval_pred_ops(self, w, x):
        if self.use_values and x.base not in ("ISETP", "UISETP"): self._eval_values(w, x)
        args = [a.strip() for a in x.args.split(",")]
        if x.base == "S2R" and "SR_TID.X" in x.args:
            w.regs_tid.add(args[0])
        elif x.base == "ELECT":
            w.preds[args[0]] = True                       # lane 0 is the elected lane
        elif x.base in ("ISETP", "UISETP"):
            m = re.match(r"U?ISETP\.(\w+)(\.U32)?\.(AND|OR)", x.op)
            if not m or len(args) < 4: return
            dst = args[0]; a, b = self._val(w, args[2]), self._val(w, args[3])
            if args[2] in w.regs_tid and a is None: a = w.tid0
            if a is None or b is None: w.preds.pop(dst, None); return
            if not m.group(2):                                   # signed compare
                a = a - (1 << 32) if a >= 1 << 31 else a; b = b - (1 << 32) if b >= 1 << 31 else b
            v = {"GT": a > b, "GE": a >= b, "LT": a < b, "LE": a <= b, "NE": a != b, "EQ": a == b}.get(m.group(1))
            if v is None: w.preds.pop(dst, None); return
            if len(args) >= 5:
                o = self._pred(w, args[4])
                if o is None: w.preds.pop(dst, None); return
                v = (v and o) if m.group(3) == "AND" else (v or o)
            w.preds[dst] = v
        elif x.base == "LOP3" and args and args[0].startswith("P"):
            # LOP3.LUT Pd, Rd, Ra, Rb, Rc, lut, Pq: Pd = (lut(Ra, Rb, Rc) != 0)
            if len(args) >= 6 and args[2] in w.regs_tid and self._imm(args[3]) is not None and args[4] == "RZ" and self._imm(args[5]) is not None:
                a, b, c, lut = w.tid0, self._imm(args[3]), 0, self._imm(args[5]); r = 0
                for bit in range(32):
                    i = (((a >> bit) & 1) << 2) | (((b >> bit) & 1) << 1) | ((c >> bit) & 1)
                    r |= ((lut >> i) & 1) << bit
                w.preds[args[0]] = r != 0
            else:
                w.preds.pop(args[0], None)
        elif x.base == "PLOP3":
            # PLOP3.LUT Pd, Pq, Pa, Pb, Pc, lut, lut2: Pd = lut(Pa, Pb, Pc)
            if len(args) >= 6:
                va, vb, vc = (self._pred(w, t) for t in args[2:5]); lut = self._imm(args[5])
                if None in (va, vb, vc) or lut is None: w.preds.pop(args[0], None)
                else: w.preds[args[0]] = bool((lut >> ((int(va) << 2) | (int(vb) << 1) | int(vc))) & 1)
        elif args and re.match(r"^U?P\d$", args[0]) and x.base not in ("BRA", "EXIT") and not x.op.startswith("SYNCS"):
            w.preds.pop(args[0], None)                    # any other write to a predicate: no longer known

    def branch_known(self, w, x):
        """A branch whose predicate is known from computed values: True (taken) / False; None when unknown."""
        if not self.use_values or x.base != "BRA" or x.target is None or x.op.startswith("BRA.DIV"): return None
        if x.pred and x.pred != "@!PT": return self._pred(w, x.pred[1:])
        if x.op.startswith("BRA.U"):
            first = x.args.split(",")[0].strip()
            if re.match(r"^!?UP(T|\d)$", first): return self._pred(w, first)
        return None

    def follow(self, w, x, taken):
        if taken and x.target < x.addr:
            c = w.__dict__.setdefault("vb", {}); c[w.pc] = c.get(w.pc, 0) + 1
            if c[w.pc] > 1_000_000: self.use_values = False      # a guard against a wrong evaluation looping forever
        w.pc = next(i for i, y in enumerate(self.ins) if y.addr == x.target) if taken else w.pc + 1

    def eligible(self, w: Warp, now: int) -> bool:
        if w.done or w.at_bar or w.ready > now: return False
        x = self.ins[w.pc]
        for b in range(6):
            if (x.wait >> b) & 1 and w.sb[b] > 0: return False
        if x.base == "DEPBAR":
            m = re.search(r"SB(\d),\s*0x([0-9a-f]+)", x.args)
            if m and w.sb[int(m.group(1))] > int(m.group(2), 16): return False
        if x.base in ("HMMA", "IMMA", "QMMA", "OMMA") and self.tensor_free[w.smsp] > now: return False
        if (x.base == "MUFU" or x.base in XU_OPS) and self.mufu_free[w.smsp] > now: return False
        u = unit(x)
        if u == "fma" and self.fma_free[w.smsp] > now: return False
        if u == "alu" and self.alu_free[w.smsp] > now: return False
        if u == "either" and min(self.fma_free[w.smsp], self.alu_free[w.smsp]) > now: return False
        if x.base in ("LDSM", "LDGSTS", "STSM", "LDG", "REDG") and self.mio_free[w.smsp] > now: return False
        if x.base in ("STG", "REDG") or (x.base == "RED" and ".E." in x.op):
            while self.stq and self.stq[0] <= now: heapq.heappop(self.stq)
            if len(self.stq) >= P["st_queue_instr"]: return False
        return True

    def issue(self, w: Warp, now: int):
        x = self.ins[w.pc]; self.count["issued"] += 1
        executes = x.pred != "@!PT"
        if x.pred and x.pred != "@!PT":
            pv = self._pred(w, x.pred[1:])
            # a predicate known from the thread index decides; unknown ones keep the old rule (the instruction runs)
            if pv is False: executes = False
        self._eval_pred_ops(w, x)
        lat_done = None; lsu_done = now
        if executes:
            if x.base in ("HMMA", "IMMA", "QMMA", "OMMA"):
                # every mma.sync shape and type holds the tensor pipe 16 cycles (probe mma_types.cu)
                self.tensor_free[w.smsp] = now + P["hmma_16816_pipe_clk"]; self.count["HMMA"] += 1; self.busy["tensor"] += P["hmma_16816_pipe_clk"]
            elif x.base == "LDSM":
                # the sub-partition's memory-instruction slot, then the SM's shared-memory array (512 B per x4)
                self.mio_free[w.smsp] = now + P["mio_ldsm_clk"]
                ways = self.ldsm_ways[1] if "MT88" in x.op else self.ldsm_ways[0]; cost = P["smem_array_clk_per_512B"] * ways
                start = max(now, self.smem_free); self.smem_free = start + cost; self.busy["smem"] += cost
                lat_done = start + P["ldsm_latency_clk"] + cost - P["smem_array_clk_per_512B"]; self.count["LDSM"] += 1
            elif x.base == "STSM":
                # stmatrix: like ldmatrix in the other direction (probe tmastore.cu: one warp every 10.8 cycles, 4 or more
                # warps 4 cycles per x4 on the shared-memory array); the source registers are free once the array takes them
                self.mio_free[w.smsp] = now + P["mio_stsm_clk"]
                start = max(now, self.smem_free); self.smem_free = start + P["smem_array_clk_per_512B"]; self.busy["smem"] += P["smem_array_clk_per_512B"]
                lsu_done = start + P["smem_array_clk_per_512B"]; self.count["STSM"] = self.count.get("STSM", 0) + 1
            elif x.base == "UTMASTG":
                # a TMA store box: read from the shared array and sent through the pair's store port (shared with plain
                # stores) at the larger of the pair's rate and this SM's share of the chip's TMA-store limit
                nb = self.tma_st_bytes
                per_B = max(self.stg_share / P["tma_st_tpc_B_per_clk"], self.active_sms / P["tma_st_chip_B_per_clk"])
                start = max(now, self.stg_free); self.stg_free = start + nb * per_B; self.busy["stg"] = self.busy.get("stg", 0) + nb * per_B
                rd = nb / 512 * P["smem_array_clk_per_512B"]; self.smem_free = max(self.smem_free, start) + rd; self.busy["smem"] += rd
                w.st_done = max(w.st_done, self.stg_free); self.count["UTMASTG"] = self.count.get("UTMASTG", 0) + 1
            elif x.base == "LDGSTS":
                # issue holds the sub-partition's memory slot; the data comes back through the SM's L2 return path after the
                # round trip and is written into the shared-memory array (the same array ldmatrix reads)
                self.mio_free[w.smsp] = now + P["mio_ldgsts_clk"]
                # the load unit's address stage: one 128-byte line per cycle; the copy reads its address registers there
                lines = self.copy_lines[w.ncopy % len(self.copy_lines)]; w.ncopy += 1
                t_lsu = max(now, self.lsu_free); self.lsu_free = t_lsu + lines * P["lsu_clk_per_line"]; self.busy["lsu"] = self.busy.get("lsu", 0) + lines
                lsu_done = self.lsu_free
                self.miss_acc += self.miss_frac
                if self.miss_acc >= 1.0:
                    self.miss_acc -= 1.0
                    t_d = max(lsu_done + P["ldgsts_dram_latency_clk"], self.dram_free); self.dram_free = t_d + self.dram_clk
                    self.busy["dram"] = self.busy.get("dram", 0) + self.dram_clk; arrive = t_d
                else:
                    arrive = lsu_done + P["ldgsts_l2_latency_clk"]
                ret = max(arrive, self.fill_free); self.fill_free = ret + self.ret_clk; self.busy["fill"] += self.ret_clk
                w.pending[w.gid] = w.pending.get(w.gid, 0) + 1
                self.ev(ret, "write", w, w.gid); self.count["LDGSTS"] += 1
            elif x.base == "LDG":
                lines, nbytes, hit = self.ldg_lines[w.nldg % len(self.ldg_lines)]; w.nldg += 1
                t_lsu = max(now, self.lsu_free); self.lsu_free = t_lsu + lines * P["lsu_clk_per_line"]; self.busy["lsu"] = self.busy.get("lsu", 0) + lines
                lsu_done = self.lsu_free
                if hit:
                    t1 = max(lsu_done, self.l1_free); self.l1_free = t1 + lines * P["l1_clk_per_line"]
                    lat_done = self.l1_free + P["l1_hit_latency_clk"]
                else:
                    self.miss_acc += self.miss_frac
                    if self.miss_acc >= 1.0:
                        self.miss_acc -= 1.0
                        t_d = max(lsu_done + P["ldgsts_dram_latency_clk"], self.dram_free); self.dram_free = t_d + self.dram_clk * nbytes / 512
                        self.busy["dram"] = self.busy.get("dram", 0) + self.dram_clk * nbytes / 512; arrive = t_d
                    else:
                        arrive = lsu_done + P["ldg_l2_latency_clk"]
                    ret = max(arrive, self.fill_free); self.fill_free = ret + self.ret_clk * nbytes / 512; self.busy["fill"] += self.ret_clk * nbytes / 512
                    lat_done = self.fill_free
                self.count["LDG"] = self.count.get("LDG", 0) + 1
            elif x.base in ("LDL", "STL"):
                # local memory (register spills, one slot per warp): 4 cycles per line through L1, then the L1 latency
                t1 = max(now, self.l1_free); self.l1_free = t1 + P["local_clk_per_line"]
                lat_done = self.l1_free + P["l1_hit_latency_clk"]; lsu_done = self.l1_free
            elif x.base == "REDG" or (x.base == "RED" and ".E." in x.op):
                # global atomics without a return value: bytes-limited, per SM and for the chip (probe ldg_atomic.cu)
                # the pair's store port per line (shared with plain stores), the SM's byte rate, the chip's rate for the pattern
                nb = 32 * (8 if "x2" in x.op else 16 if "x4" in x.op else 4)
                chip = P["red_chip_B_per_clk_frag"] if self.red_lines >= 8 else P["red_chip_B_per_clk"]
                per = max(nb / P["red_sm_B_per_clk"], self.red_lines * P["red_pair_clk_per_line"] * self.stg_share, nb * self.active_sms / chip)
                start = max(now, self.stg_free); self.stg_free = start + per; self.busy["red"] = self.busy.get("red", 0) + per
                heapq.heappush(self.stq, self.stg_free); self.mio_free[w.smsp] = max(self.mio_free[w.smsp], now + self.red_lines * P["red_smsp_clk_per_line"])
                lsu_done = start; self.end = max(getattr(self, "end", 0), self.stg_free)
            elif x.base == "LDGDEPBAR":
                # commit: the group's dependency counter stays set until its last copy has landed in shared memory
                g = w.gid; w.gid += 1
                if x.wbar is not None:
                    w.sb[x.wbar] += 1
                    if w.pending.get(g, 0) == 0: self.ev(now + 1, "sb", w, x.wbar)
                    else: w.committed[g] = x.wbar
            elif x.base == "BAR":
                # BAR.SYNC id[, threads]: a named barrier waits for the threads it names; without a count, for the whole
                # block (warps that already exited count as arrived)
                self.count["BAR"] += 1; w.at_bar = True
                ba = [a.strip() for a in x.args.split(",")]
                bid = self._imm(ba[0]) if ba and ba[0] else 0
                need = (self._imm(ba[1]) // 32) if len(ba) > 1 and self._imm(ba[1]) else self.wpb - self.exited.get(w.block, 0)
                key = (w.block, bid); self.bar_arrived[key] = self.bar_arrived.get(key, []) + [w]
                if len(self.bar_arrived[key]) >= need:
                    rel = now + P["bar_base_clk"] + P["bar_per_warp_clk"] * need
                    for o in self.bar_arrived.pop(key): self.ev(rel, "bar", o)
            elif unit(x) in ("fma", "alu", "either"):
                c = unit_clk(x); u = unit(x)
                if u == "either": u = "alu" if self.alu_free[w.smsp] <= self.fma_free[w.smsp] else "fma"
                if u == "fma": self.fma_free[w.smsp] = now + c
                else: self.alu_free[w.smsp] = now + c
                self.busy[u] = self.busy.get(u, 0) + c
                if x.wbar is not None: lat_done = now + P["default_var_latency_clk"]
            elif x.base in XU_OPS:
                self.mufu_free[w.smsp] = now + P["mufu_smsp_clk_per_instr"]; self.busy["sfu"] = self.busy.get("sfu", 0) + P["mufu_smsp_clk_per_instr"]
                lat_done = now + P["mufu_latency_clk"]
            elif x.base == "MUFU":
                # the special function unit: one warp instruction per 8 cycles per sub-partition, 18-cycle results
                self.mufu_free[w.smsp] = now + P["mufu_smsp_clk_per_instr"]; self.busy["sfu"] = self.busy.get("sfu", 0) + P["mufu_smsp_clk_per_instr"]
                lat_done = now + P["mufu_latency_clk"]
            elif x.base == "STG":
                wide = ".128" in x.op; nbytes = 512 if wide else (256 if ".64" in x.op else 128)
                # the larger of the pair's shared store port and this SM's share of the chip's store limit
                pair = (P["stg_tpc_clk_per_instr_16B"] if wide else P["stg_tpc_clk_per_instr_4B"]) * self.stg_share
                chip = nbytes / ((P["stg_chip_B_per_clk_16B"] if wide else P["stg_chip_B_per_clk_4B"]) / self.active_sms)
                per = max(pair, chip)
                start = max(now, self.stg_free); self.stg_free = start + per; self.busy["stg"] = self.busy.get("stg", 0) + per
                heapq.heappush(self.stq, self.stg_free)
                if x.rbar is not None: lsu_done = start
            elif x.base == "EXIT":
                # the block is finished only when its stores have left the SM
                w.done = True; self.end = max(getattr(self, "end", 0), self.stg_free)
                self.exited[w.block] = self.exited.get(w.block, 0) + 1; self.block_exit(w.block, now)
            elif x.wbar is not None:
                lat_done = now + P["default_var_latency_clk"]
            if x.wbar is not None and lat_done is not None and x.base != "LDGDEPBAR":
                w.sb[x.wbar] += 1; self.ev(lat_done, "sb", w, x.wbar)
            if x.rbar is not None:
                if x.base in ("LDGSTS", "STG", "STSM", "LDG", "LDL", "STL", "REDG", "RED") and executes: rel = lsu_done
                elif x.base == "UTMACMDFLUSH":
                    # the bulk-store group's counter clears when every box has been read out and sent, plus the fixed delay
                    rel = max(now + P["rbar_release_clk"], w.st_done + P["tma_st_fixed_clk"])
                else: rel = now + P["rbar_release_clk"]
                w.sb[x.rbar] += 1; self.ev(rel, "sb", w, x.rbar)
        w.ready = now + max(1, x.stall)
        # next pc: the k-loop back edge is taken trips-1 times; other branches fall through unless unconditional
        if x.op.startswith("BRA.DIV"):
            # taken only if the warp's lanes have diverged; warps here stay converged
            w.pc += 1
        elif self.branch_known(w, x) is not None:
            self.follow(w, x, self.branch_known(w, x))
        elif x.base == "BRA" and x.target is not None and w.pc in self.loop_trips:
            c = w.__dict__.setdefault("lc", {}); c[w.pc] = c.get(w.pc, 0) + 1
            if c[w.pc] < self.loop_trips[w.pc]: w.pc = next(i for i, y in enumerate(self.ins) if y.addr == x.target)
            else: c[w.pc] = 0; w.pc += 1
        elif x.base == "BRA" and x.target is not None:
            if w.pc == self.loop_end:
                w.loops += 1
                w.pc = next(i for i, y in enumerate(self.ins) if y.addr == x.target) if w.loops < self.trips else w.pc + 1
            elif not x.pred and (x.target > x.addr or not x.op.startswith("BRA.U")):
                # unconditional jumps are taken, forward or backward (conditional BRA.U on a uniform predicate is not)
                w.pc = next(i for i, y in enumerate(self.ins) if y.addr == x.target)
            else:
                w.pc += 1
        else:
            w.pc += 1
        if w.pc >= len(self.ins) and not w.done:
            w.done = True; self.exited[w.block] = self.exited.get(w.block, 0) + 1; self.block_exit(w.block, now)

    def share(self, finished: int) -> float:
        """SM-equivalents sharing the chip's limits once this SM has finished `finished` blocks."""
        if not self.grid_blocks: return self.eq0
        q, r = divmod(self.grid_blocks, 48); R = self.resident
        counts = [q + 1] * r + [q] * (48 - r); mine = max(counts)
        chip = sum(max(0, min(R, c - finished)) for c in counts); own = max(1, min(R, mine - finished))
        return max(1.0, chip / own)

    def set_share(self, eq: float):
        self.active_sms = eq
        self.ret_clk = max(P["l2_return_clk_per_512B"], 512 * eq / P["l2_aggregate_B_per_clk_cpasync"])
        if self.eq0: self.dram_clk = self.dram_clk0 * eq / self.eq0
        self.on_share(eq)

    def on_share(self, eq: float):
        pass

    def block_exit(self, b, now):
        """A warp of block b exited; when the whole block has, start the next block on its slot."""
        if b not in self.live: return
        self.live[b] -= 1
        if self.live[b] == 0:
            del self.live[b]; self.finished += 1
            if self.grid_blocks: self.set_share(self.share(self.finished))
            if self.next_block < self.blocks:
                nb = self.next_block; self.next_block += 1; self.live[nb] = self.wpb
                for o in self.warps:
                    if o.block == nb: o.ready = now + 1
                self._live_dirty = True

    def run(self, limit: int = 50_000_000) -> dict:
        # only warps of started, unfinished blocks are scanned; the lists are rebuilt when a block starts or a warp exits
        now = 0; INF = float("inf"); self._live_dirty = True
        while now < limit:
            if self._live_dirty:
                live = [w for w in self.warps if not w.done and w.ready < INF]
                by_smsp = [[w for w in live if w.smsp == s] for s in range(4)]; self._live_dirty = False
            while self.events and self.events[0][0] <= now:
                _, _, kind, w, arg = heapq.heappop(self.events)
                if kind == "sb": w.sb[arg] -= 1
                elif kind == "write":
                    # the returned 512 B are written into the shared-memory array (shared with ldmatrix), then counted as landed
                    t_w = max(now, self.smem_free); self.smem_free = t_w + P["smem_array_clk_per_512B"]; self.busy["smem"] += P["smem_array_clk_per_512B"]
                    self.ev(t_w + P["smem_array_clk_per_512B"], "landed", w, arg)
                elif kind == "landed":
                    w.pending[arg] -= 1
                    if w.pending[arg] == 0 and arg in w.committed: w.sb[w.committed.pop(arg)] -= 1
                elif kind == "bar": w.at_bar = False; w.ready = max(w.ready, now)
                else:
                    self.on_event(kind, w, arg, now)
            if not live and self.next_block >= self.blocks and all(w.done for w in self.warps): break
            if any(w.done for w in live): self._live_dirty = True
            issued = False
            for s in range(4):
                cand = [w for w in by_smsp[s] if self.eligible(w, now)]
                if not cand: continue
                last = self.last_issue[s]
                if hasattr(self, "pick"): w = self.pick(cand, last)
                else: w = last if (last in cand) else min(cand, key=lambda q: (q.loops, q.wid))   # greedy, then oldest
                yb = self.ins[w.pc].yld
                self.issue(w, now); self.last_issue[s] = w; issued = True
                if hasattr(self, "last_yield"): self.last_yield[w.wid] = yb
            # jump to the next cycle where something can change
            nxt = [w.ready for w in live if not w.done and not w.at_bar and now < w.ready < INF]
            nxt += [t for t in self.tensor_free if t > now] + [t for t in self.mio_free if t > now] + [t for t in self.mufu_free if t > now]
            nxt += [t for t in self.fma_free if t > now] + [t for t in self.alu_free if t > now]
            if self.events: nxt.append(self.events[0][0])
            nxt += [t for t in self.stq if t > now][:1]
            now = now + 1 if issued else (min(nxt) if nxt else now + 1)
        return {"cycles": max(now, getattr(self, "end", 0)), **self.count, "busy": self.busy}


def kernel_sass(path: str | Path, kernel: str = "matmul_kernel") -> list[Ins]:
    txt = open(path).read()
    ins = parse(txt, kernel=kernel)
    return ins

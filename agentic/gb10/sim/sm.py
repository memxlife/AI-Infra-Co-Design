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
                 miss_frac: float = 0.0, dram_clk_per_512B: float = 0.0):
        self.ins = ins; self.trips = trips; self.mem = mem
        # this SM's share of the chip's L2 return bandwidth (or its own return path, if fewer SMs are pulling)
        self.ret_clk = max(P["l2_return_clk_per_512B"], 512 * active_sms / P["l2_aggregate_B_per_clk_cpasync"])
        lp = main_loop(ins); self.loop_end = lp[1] if lp else -1     # fully unrolled code has no back edge
        self.warps = []; wpb = threads // 32
        for b in range(blocks):
            for w in range(wpb): self.warps.append(Warp(wid=len(self.warps), block=b, smsp=w % 4))
        self.wpb = wpb; self.blocks = blocks
        self.tensor_free = [0] * 4; self.smem_free = 0; self.fill_free = 0; self.mio_free = [0] * 4; self.mufu_free = [0] * 4
        # lines each copy of a k-tile touches, in program order (A rows then B rows); cycles them per warp
        self.copy_lines = copy_lines or [4]; self.lsu_free = 0
        # the store path is shared with the partner SM of the pair when the chip runs more than one block
        self.stg_share = 2 if active_sms >= 2 else 1; self.stg_free = 0; self.active_sms = active_sms
        # copies that miss the L2 (this wave's share, from the launch-order L2 simulation) pay the DRAM round trip and
        # this SM's share of the DRAM rate; misses are spread evenly over the copies
        self.miss_frac = miss_frac; self.miss_acc = 0.0; self.dram_clk = dram_clk_per_512B; self.dram_free = 0
        self.events = []            # (time, seq, kind, warp, arg)
        self.seq = 0; self.bar_arrived = {}; self.last_issue = [None] * 4
        self.count = {"issued": 0, "HMMA": 0, "LDSM": 0, "LDGSTS": 0, "BAR": 0}
        self.busy = {"tensor": 0, "smem": 0, "fill": 0}

    def ev(self, t, kind, w, arg=None):
        self.seq += 1; heapq.heappush(self.events, (t, self.seq, kind, w, arg))

    def on_event(self, kind, w, arg, now):
        pass

    def eligible(self, w: Warp, now: int) -> bool:
        if w.done or w.at_bar or w.ready > now: return False
        x = self.ins[w.pc]
        for b in range(6):
            if (x.wait >> b) & 1 and w.sb[b] > 0: return False
        if x.base == "DEPBAR":
            m = re.search(r"SB(\d),\s*0x([0-9a-f]+)", x.args)
            if m and w.sb[int(m.group(1))] > int(m.group(2), 16): return False
        if x.base == "HMMA" and self.tensor_free[w.smsp] > now: return False
        if x.base == "MUFU" and self.mufu_free[w.smsp] > now: return False
        if x.base in ("LDSM", "LDGSTS") and self.mio_free[w.smsp] > now: return False
        return True

    def issue(self, w: Warp, now: int):
        x = self.ins[w.pc]; self.count["issued"] += 1
        executes = x.pred != "@!PT"
        lat_done = None; lsu_done = now
        if executes:
            if x.base == "HMMA":
                self.tensor_free[w.smsp] = now + P["hmma_16816_pipe_clk"]; self.count["HMMA"] += 1; self.busy["tensor"] += P["hmma_16816_pipe_clk"]
            elif x.base == "LDSM":
                # the sub-partition's memory-instruction slot, then the SM's shared-memory array (512 B per x4)
                self.mio_free[w.smsp] = now + P["mio_ldsm_clk"]
                start = max(now, self.smem_free); self.smem_free = start + P["smem_array_clk_per_512B"]; self.busy["smem"] += P["smem_array_clk_per_512B"]
                lat_done = start + P["ldsm_latency_clk"]; self.count["LDSM"] += 1
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
            elif x.base == "LDGDEPBAR":
                # commit: the group's dependency counter stays set until its last copy has landed in shared memory
                g = w.gid; w.gid += 1
                if x.wbar is not None:
                    w.sb[x.wbar] += 1
                    if w.pending.get(g, 0) == 0: self.ev(now + 1, "sb", w, x.wbar)
                    else: w.committed[g] = x.wbar
            elif x.base == "BAR":
                self.count["BAR"] += 1; w.at_bar = True
                key = w.block; self.bar_arrived[key] = self.bar_arrived.get(key, 0) + 1
                if self.bar_arrived[key] == self.wpb:
                    self.bar_arrived[key] = 0
                    rel = now + P["bar_base_clk"] + P["bar_per_warp_clk"] * self.wpb
                    for o in self.warps:
                        if o.block == key: self.ev(rel, "bar", o)
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
                if x.rbar is not None: lsu_done = start
            elif x.base == "EXIT":
                # the block is finished only when its stores have left the SM
                w.done = True; self.end = max(getattr(self, "end", 0), self.stg_free)
            elif x.wbar is not None:
                lat_done = now + P["default_var_latency_clk"]
            if x.wbar is not None and lat_done is not None and x.base != "LDGDEPBAR":
                w.sb[x.wbar] += 1; self.ev(lat_done, "sb", w, x.wbar)
            if x.rbar is not None:
                rel = lsu_done if (x.base in ("LDGSTS", "STG") and executes) else now + P["rbar_release_clk"]
                w.sb[x.rbar] += 1; self.ev(rel, "sb", w, x.rbar)
        w.ready = now + max(1, x.stall)
        # next pc: the k-loop back edge is taken trips-1 times; other branches fall through unless unconditional
        if x.op.startswith("BRA.DIV"):
            # taken only if the warp's lanes have diverged; warps here stay converged
            w.pc += 1
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
        if w.pc >= len(self.ins): w.done = True

    def run(self, limit: int = 50_000_000) -> dict:
        now = 0; by_smsp = [[w for w in self.warps if w.smsp == s] for s in range(4)]
        while now < limit:
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
            if all(w.done for w in self.warps): break
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
            nxt = [w.ready for w in self.warps if not w.done and not w.at_bar and w.ready > now]
            nxt += [t for t in self.tensor_free if t > now] + [t for t in self.mio_free if t > now] + [t for t in self.mufu_free if t > now]
            if self.events: nxt.append(self.events[0][0])
            now = now + 1 if issued else (min(nxt) if nxt else now + 1)
        return {"cycles": max(now, getattr(self, "end", 0)), **self.count, "busy": self.busy}


def kernel_sass(path: str | Path, kernel: str = "matmul_kernel") -> list[Ins]:
    txt = open(path).read()
    ins = parse(txt, kernel=kernel)
    return ins

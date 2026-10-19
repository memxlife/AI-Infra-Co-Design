"""Warp-specialized TMA kernels on the SM simulator (TileLang's default lowering on sm_121a).

Additions to the cp.async simulator, each a mechanism in the machine code:
  roles     warps start together; comparisons of the thread index (S2R SR_TID.X) are evaluated per warp (warp-uniform,
            lane 0), so warps 0-3 take the producer side (warps 1-3 exit) and warps 4+ the consumer side; ELECT is true
  TMA       UTMALDG loads a box into shared memory: the SM's TMA engine moves the bytes at its rate (one block alone 81
            GB/s; chip 1.58 TB/s shared by blocks in flight), after the L2 or DRAM round trip; the bytes also pass the
            shared-memory array
  mbarrier  SYNCS.PHASECHK.TRYWAIT spin loops are blocking waits, matched by order: the consumer's n-th wait completes
            when k-tile n's TMA bytes have landed; the producer's k-th wait for a free stage completes when every consumer
            warp has released k-tile k - stages (SYNCS.ARRIVE...A1T0). The retry stubs are then never taken.
  registers USETMAXREG.TRY_ALLOC waits until the producer group's warps have executed USETMAXREG.DEALLOC.
The tile bytes per k-tile are split evenly over the UTMALDG instructions of one k-tile."""
from __future__ import annotations
import re
from .sm import SM, P


class WSSM(SM):
    def __init__(self, ins, threads_launched, blocks, trips, consumer_threads, stages, tile_bytes, tma_per_ktile, kt=None, **kw):
        super().__init__(ins, threads_launched, blocks, trips, **kw)
        self.kt = kt
        self.stages = stages; self.ctiles = tile_bytes / max(1, tma_per_ktile); self.consumer_warps = consumer_threads // 32
        self.tma_free = 0.0
        eq = kw.get("active_sms", 1)
        rate_alone = 81e9 / 2.54e9; rate_chip = 1860e9 / 2.54e9 / max(1.0, eq)
        self.tma_clk_per_B = 1.0 / min(rate_alone, rate_chip)
        for w in self.warps:
            w.tid0 = (w.wid % self.wpb) * 32; w.regs_tid = set(); w.preds = {}; w.role = None
            w.k_prod = -1; w.n_full = 0; w.n_empty_arrive = 0
        self.landed = {}            # (block, k) -> time all of k-tile k's TMA bytes are in shared memory
        self.pending_k = {}         # (block, k) -> outstanding landing times
        self.empty_arrives = {}     # (block) -> consumer arrives so far
        self.dealloc = {}           # (block) -> producer-group warps that gave registers back
        self.n_full_done = {}; self.arrived = {}; self.log = []; self.pend = {}; self.land_t = {}

    # --- per-warp evaluation of thread-index comparisons -------------------------------------------------------------
    def _pred(self, w, tok):
        neg = tok.startswith("!"); t = tok[1:] if neg else tok
        if t in ("PT", "UPT"): v = True
        elif t in w.preds: v = w.preds[t]
        else: return None
        return (not v) if neg else v

    def _eval_isetp(self, w, x):
        m = re.match(r"ISETP\.(\w+)(?:\.U32)?\.(AND|OR)", x.op)
        args = [a.strip() for a in x.args.split(",")]
        if not m or len(args) < 4: return
        cmp, comb = m.group(1), m.group(2); dst = args[0]; a, b = args[2], args[3]
        if a not in w.regs_tid or not re.match(r"^-?0x[0-9a-f]+$|^-?\d+$", b): w.preds.pop(dst, None); return
        tid = w.tid0; imm = int(b, 16) if "0x" in b else int(b)
        v = {"GT": tid > imm, "GE": tid >= imm, "LT": tid < imm, "LE": tid <= imm, "NE": tid != imm, "EQ": tid == imm}.get(cmp)
        if v is None: return
        if len(args) >= 5:
            o = self._pred(w, args[4])
            if o is None: w.preds.pop(dst, None); return
            v = (v and o) if comb == "AND" else (v or o)
        w.preds[dst] = v

    def eligible(self, w, now):
        if w.done or w.at_bar or w.ready > now: return False
        x = self.ins[w.pc]
        if x.op.startswith("SYNCS.PHASECHK") and "TRYWAIT" in x.op:
            b = w.block
            if w.role == "consumer":
                # k-tile n's full barrier: the producer has arrived (with its expected byte count) and every byte landed
                k = w.n_full; arr = self.arrived.get((b, k))
                if arr is None or arr > now: return False
                if (self.landed.get((b, k)) or 0.0) > now: return False
            elif w.role == "producer":
                k = w.k_prod + 1
                need = k - self.stages + 1
                if need > 0 and self.empty_arrives.get(b, 0) // self.consumer_warps < need: return False
        if x.base == "USETMAXREG" and "TRY_ALLOC" in x.op and self.dealloc.get(w.block, 0) < 4: return False
        return super().eligible(w, now)

    def issue(self, w, now):
        x = self.ins[w.pc]; executes = x.pred != "@!PT"
        p = None
        if x.pred:
            p = self._pred(w, x.pred[1:])
        if x.base == "S2R" and "SR_TID.X" in x.args:
            w.regs_tid.add(x.args.split(",")[0].strip())
        elif x.base == "ISETP":
            self._eval_isetp(w, x)
        elif x.base == "ELECT":
            w.preds[x.args.split(",")[0].strip()] = True
        elif x.base == "USETMAXREG":
            if "DEALLOC" in x.op: self.dealloc[w.block] = self.dealloc.get(w.block, 0) + 1; w.role = w.role or "producer"
            else: w.role = "consumer"; w.preds["UP0"] = True
        elif x.op.startswith("SYNCS.PHASECHK") and "TRYWAIT" in x.op:
            dst = x.args.split(",")[0].strip(); w.preds[dst] = True
            if w.role == "consumer": self.log.append(("full_pass", w.wid, w.n_full, now)); w.n_full += 1
            elif w.role == "producer": w.k_prod += 1; self.log.append(("empty_pass", w.wid, w.k_prod, now))
        elif x.op.startswith("SYNCS.ARRIVE") and w.role == "consumer":
            # a consumer warp releases its stage (the empty barrier)
            self.empty_arrives[w.block] = self.empty_arrives.get(w.block, 0) + 1
            self.log.append(("release", w.wid, w.n_full - 1, now))
        elif x.op.startswith("SYNCS.ARRIVE") and w.role == "producer":
            # the producer arrives on k-tile k's full barrier (with the bytes it expects, or none if loads are removed)
            self.arrived[(w.block, max(0, w.k_prod))] = now; self.log.append(("arrive", w.wid, w.k_prod, now))
        elif x.base == "UTMALDG" and executes:
            k = max(0, w.k_prod); key = (w.block, k)
            start = max(now, self.tma_free); self.tma_free = start + self.ctiles * self.tma_clk_per_B
            # the box's bytes that miss the L2 (this wave's share) come from DRAM at this SM's share of the DRAM rate
            miss_B = self.ctiles * self.miss_frac
            if miss_B > 0:
                t_d = max(start + P["ldgsts_dram_latency_clk"], self.dram_free); self.dram_free = t_d + miss_B * self.dram_clk / 512.0
                self.busy["dram"] = self.busy.get("dram", 0) + miss_B * self.dram_clk / 512.0
                land = max(start + P["ldgsts_l2_latency_clk"], self.tma_free, self.dram_free)
            else:
                land = max(start + P["ldgsts_l2_latency_clk"], self.tma_free)
            # the box is written into the shared-memory array when it arrives (an event then, not a reservation now)
            self.pend[key] = self.pend.get(key, 0) + 1; self.landed[key] = float("inf")
            self.ev(land, "tmawrite", w, key); self.log.append(("tma", w.wid, k, now, start, land))
            self.busy["tma"] = self.busy.get("tma", 0) + self.ctiles * self.tma_clk_per_B
        # predicated branches and exits whose predicate is known
        if x.base in ("BRA", "EXIT") and x.pred and p is not None:
            if x.base == "EXIT":
                if p: w.done = True; return
                w.pc += 1; w.ready = now + max(1, x.stall); return
            if x.target is not None and self.loop_end != w.pc:
                if p:
                    w.pc = next(i for i, y in enumerate(self.ins) if y.addr == x.target)
                    if x.target < x.addr and self.ins[w.pc].base == "USETMAXREG": pass
                else:
                    w.pc += 1
                w.ready = now + max(1, x.stall); return
        # k-loops, either shape the compiler emits, decided by what the branch tests (k-tiles remain for this role):
        #  (a) a conditional back edge: taken while k-tiles remain
        #  (b) a conditional forward exit followed by an unconditional jump back: the exit is taken when none remain
        if self.kt and x.base == "BRA" and x.target is not None and w.role in ("producer", "consumer"):
            more = (w.k_prod + 1 < self.kt) if w.role == "producer" else (w.n_full < self.kt)
            conditional = bool(x.pred) or x.op.startswith("BRA.U")
            if x.target < x.addr and self._is_work_loop(w.pc, x.target):
                if conditional: w.pc = self._idx(x.target) if more else w.pc + 1
                else: w.pc = self._idx(x.target)
                w.ready = now + max(1, x.stall); return
            nxt = self.ins[w.pc + 1] if w.pc + 1 < len(self.ins) else None
            if conditional and x.target > x.addr and nxt is not None and nxt.base == "BRA" and not nxt.pred and not nxt.op.startswith("BRA.U") \
                    and nxt.target is not None and nxt.target < nxt.addr and self._is_work_loop(w.pc + 1, nxt.target):
                w.pc = self._idx(x.target) if not more else w.pc + 1
                w.ready = now + max(1, x.stall); return
        if x.base == "BRA" and x.op.startswith("BRA.U") and "UP0" in x.args and x.target is not None and x.target < x.addr and self.loop_end != w.pc:
            # the register-allocation retry loop: TRY_ALLOC succeeded, fall through
            w.pc += 1; w.ready = now + max(1, x.stall); return
        super().issue(w, now)
        # the landed times of a k-tile depend on its last TMA; mark it pending until then


    def on_event(self, kind, w, key, now):
        if kind == "tmawrite":
            t_w = max(now, self.smem_free); self.smem_free = t_w + self.ctiles / 128.0; self.busy["smem"] += self.ctiles / 128.0
            self.ev(t_w + self.ctiles / 128.0, "tmalanded", w, key)
        elif kind == "tmalanded":
            self.pend[key] -= 1; self.land_t[key] = max(self.land_t.get(key, 0.0), now)
            if self.pend[key] == 0: self.landed[key] = self.land_t[key]


    def _is_work_loop(self, end_pc, target_addr):
        key = (end_pc, target_addr)
        if not hasattr(self, "_wl"): self._wl = {}
        if key not in self._wl:
            s_ = next(i for i, y in enumerate(self.ins) if y.addr == target_addr)
            self._wl[key] = any(y.base in ("HMMA", "UTMALDG", "LDGSTS") for y in self.ins[s_:end_pc + 1])
        return self._wl[key]


    def _idx(self, addr):
        if not hasattr(self, "_ix"): self._ix = {y.addr: i for i, y in enumerate(self.ins)}
        return self._ix[addr]

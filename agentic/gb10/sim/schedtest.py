"""Replay the scheduler probe (sched.cu) on the simulator under candidate warp-scheduling policies."""
from .sass import parse
from .sm import SM

ins = parse(open("calibration/spark-6951/sass/sched.sass").read())
idx = {x.addr: i for i, x in enumerate(ins)}
HOG = (idx[0x210], idx[0x440]); ADD = (idx[0x7c0], idx[0x8e0])


class Probe(SM):
    def __init__(self, policy, with_hog, n_hog=6000, n_add=1250):
        super().__init__(ins, 160, 1, 1); self.policy = policy; self.iters = {0: n_hog, 4: n_add}; self.cnt = {0: 0, 4: 0}
        for w in self.warps:
            if w.wid == 0 and with_hog: w.pc = HOG[0]
            elif w.wid == 4: w.pc = ADD[0]
            else: w.done = True
        self.finish = {}

    def issue(self, w, now):
        x = self.ins[w.pc]
        loop = HOG if w.wid == 0 else ADD
        if w.pc == loop[1]:
            self.cnt[w.wid] += 1; self.count["issued"] += 1; w.ready = now + max(1, x.stall)
            if self.cnt[w.wid] < self.iters[w.wid]: w.pc = loop[0]
            else: w.done = True; self.finish[w.wid] = now
            return
        super().issue(w, now)


def run_policy(policy, with_hog):
    import agentic.gb10.sim.sm as smmod
    p = Probe(policy, with_hog)
    # scheduling policy hook: choose among ready warps of one sub-partition
    orig = smmod.SM.run
    last_yield = {}
    def pick(cand, last):
        if policy == "greedy": return last if last in cand else min(cand, key=lambda q: q.wid)
        if policy == "rotate":
            order = sorted(cand, key=lambda q: q.wid)
            if last is None: return order[0]
            nxt = [q for q in order if q.wid > last.wid]; return nxt[0] if nxt else order[0]
        # greedy, but leave the last warp when its last instruction's yield bit says so
        if last in cand and last_yield.get(last.wid, 0) != (1 if policy == "yield1=leave" else 0): return last
        others = [q for q in cand if q is not last]; return min(others or cand, key=lambda q: q.wid)
    p.pick = pick; p.last_yield = last_yield
    return p


if __name__ == "__main__":
    import agentic.gb10.sim.sm as smmod
    for policy in ("greedy", "rotate", "yield1=leave", "yield0=leave"):
        res = []
        for with_hog in (0, 1):
            p = run_policy(policy, with_hog); r = p.run(limit=3_000_000)
            res.append(p.finish.get(4, r["cycles"]) / (p.iters[4] * 16))
        print(f"{policy:13s} cycles per add: alone {res[0]:5.2f}, next to the hog {res[1]:5.2f}   (measured 5.67 and 15.10)")

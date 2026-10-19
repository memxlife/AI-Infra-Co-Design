"""Identify the GB10 L2's replacement policy from the two probes (l2policy.cu, ncu hit/miss): which standard policy,
simulated at line granularity on the same access streams, reproduces both the cyclic-scan curve and the hot set's
survival after a stream. Candidates: LRU, random, segmented LRU (protected share), bimodal insertion (BIP), SRRIP/BRRIP.
Fully associative at 64 KB units (a 24 MB cache = 384 units); hit rates are the reuse hits (above the 50 % floor)."""
import random
from collections import OrderedDict

U = 64 * 1024; C = 24 * 1024 * 1024 // U
MEAS_CYCLIC = {4: 1.0, 8: 1.0, 12: 1.0, 14: 1.0, 16: 0.964, 18: 0.816, 20: 0.752, 22: 0.772, 24: 0.626, 28: 0.624, 32: 0.32, 40: 0.012, 48: 0.0, 64: 0.0}


class LRU:
    def __init__(s): s.d = OrderedDict()
    def access(s, a):
        if a in s.d: s.d.move_to_end(a); return True
        s.d[a] = 1
        if len(s.d) > C: s.d.popitem(last=False)
        return False


class Random:
    def __init__(s, seed=1): s.d = {}; s.keys = []; s.r = random.Random(seed)
    def access(s, a):
        if a in s.d: return True
        if len(s.keys) >= C:
            i = s.r.randrange(len(s.keys)); v = s.keys[i]; del s.d[v]; s.keys[i] = a
        else: s.keys.append(a)
        s.d[a] = 1; return False


class SLRU:
    def __init__(s, prot_frac): s.p = OrderedDict(); s.q = OrderedDict(); s.P = int(C * prot_frac)
    def access(s, a):
        if a in s.p: s.p.move_to_end(a); return True
        if a in s.q:
            s.q.pop(a); s.p[a] = 1
            if len(s.p) > s.P: k, _ = s.p.popitem(last=False); s.q[k] = 1     # demote to probation MRU
            return True
        s.q[a] = 1
        while len(s.p) + len(s.q) > C: s.q.popitem(last=False)
        return False


class BIP:
    def __init__(s, eps, seed=1): s.d = OrderedDict(); s.eps = eps; s.r = random.Random(seed)
    def access(s, a):
        if a in s.d: s.d.move_to_end(a); return True
        s.d[a] = 1
        if s.r.random() >= s.eps: s.d.move_to_end(a, last=False)        # insert at the LRU end, rarely at MRU
        if len(s.d) > C:
            # evict the LRU end, but not the line just inserted there
            it = iter(s.d); k = next(it)
            if k == a: k = next(it)
            del s.d[k]
        return False


class RRIP:
    """2-bit re-reference interval prediction: insert at 2 (SRRIP) or mostly 3 (BRRIP); hit -> 0; evict a 3, aging all."""
    def __init__(s, brrip_eps=None, seed=1): s.v = {}; s.order = []; s.eps = brrip_eps; s.r = random.Random(seed)
    def access(s, a):
        if a in s.v: s.v[a] = 0; return True
        if len(s.v) >= C:
            while True:
                for k in s.order:
                    if s.v[k] == 3: break
                else:
                    for k in s.order: s.v[k] += 1
                    continue
                break
            s.order.remove(k); del s.v[k]
        ins = 2 if s.eps is None else (2 if s.r.random() < s.eps else 3)
        s.v[a] = ins; s.order.append(a); return False


def cyclic(pol, W_MB, passes=4):
    n = W_MB * 1024 * 1024 // U; [pol.access(i) for i in range(n)]
    hits = sum(pol.access(i) for _ in range(passes) for i in range(n)); return hits / (passes * n)


def hotstream(pol, H_MB, S_MB):
    h = H_MB * 1024 * 1024 // U; s = S_MB * 1024 * 1024 // U
    for _ in range(2): [pol.access(i) for i in range(h)]
    [pol.access(10**6 + i) for i in range(s)]
    return sum(pol.access(i) for i in range(h)) / h


if __name__ == "__main__":
    cands = {"LRU": LRU, "random": Random, "SLRU 50%": lambda: SLRU(0.5), "SLRU 67%": lambda: SLRU(0.67), "SLRU 75%": lambda: SLRU(0.75),
             "BIP 1/32": lambda: BIP(1 / 32), "BIP 1/64": lambda: BIP(1 / 64), "SRRIP": RRIP, "BRRIP 1/32": lambda: RRIP(1 / 32)}
    Ws = sorted(MEAS_CYCLIC)
    print(f"{'policy':10s} " + " ".join(f"{w:>5d}" for w in Ws) + " | hot 8 after 64 | cyclic error")
    print(f"{'measured':10s} " + " ".join(f"{MEAS_CYCLIC[w]:5.2f}" for w in Ws) + " |  1.00")
    for name, mk in cands.items():
        cyc = {w: cyclic(mk(), w) for w in Ws}; hs = hotstream(mk(), 8, 64)
        err = sum(abs(cyc[w] - MEAS_CYCLIC[w]) for w in Ws) / len(Ws)
        print(f"{name:10s} " + " ".join(f"{cyc[w]:5.2f}" for w in Ws) + f" |  {hs:4.2f}         | {err:.3f}", flush=True)

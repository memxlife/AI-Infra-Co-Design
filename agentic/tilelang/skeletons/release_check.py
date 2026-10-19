"""Static stage-release check for TileLang's warp-specialized CUDA (the pattern of tile-ai/tilelang#3451).

In a warp-specialized kernel the consumer frees a pipeline stage by arriving on its back-pressure mbarrier
(`mbarrier[i].arrive();` with no expect-tx). Every read of a shared buffer that the producer fills for that stage
(`tl::tma_load(desc, mbarrier[..], (&(((T*)X_shared)[..])), ..)` or `tl::cp_async_gs...((&(((T*)X_shared)[..])), ..)`)
must come before that arrive in the consumer's loop body; a read after it races with the producer's refill of the
same stage. This reads the generated CUDA (no device).

Usage: python release_check.py kernel.cu [more.cu ...]
Per file: NOT-WS (no producer group, nothing is freed early: the pattern cannot occur), OK, or RACE with the lines.
"""
import re, sys

BUF = re.compile(r"\(\(\w+\*\)(\w+_shared(?:_\w+)?)\)")


def check(src: str) -> dict:
    lines = src.splitlines()
    ws = "warpgroup_reg_dealloc" in src or "warpgroup_reg_alloc" in src
    filled = set()
    for ln in lines:
        if "tl::tma_load(" in ln or "cp_async_gs" in ln:
            m = BUF.search(ln)
            if m: filled.add(m.group(1))
    if not ws:
        return {"status": "NOT-WS", "filled": sorted(filled)}
    # the consumer region: from `warpgroup_reg_alloc` to the end of the kernel
    start = next(i for i, ln in enumerate(lines) if "warpgroup_reg_alloc" in ln)
    races, loops = [], 0
    for i in range(start, len(lines)):
        if not re.match(r"\s*for \(int \w+ = 0; \w+ < ", lines[i]): continue
        depth, j, body = 0, i, []
        while j < len(lines):
            depth += lines[j].count("{") - lines[j].count("}")
            body.append((j + 1, lines[j]))
            if depth <= 0 and j > i: break
            j += 1
        arrives = [n for n, b in body if re.search(r"mbarrier\[[^\]]*\]\.arrive\(\);", b)]
        if not arrives: continue
        loops += 1
        reads = [(n, buf) for n, b in body for buf in filled if buf in b and ("ldmatrix" in b or "mma" in b or "gemm" in b or "wgmma" in b or re.search(r"=\s*\*?\(?.*" + re.escape(buf) + r"\)\[", b))]
        for a in arrives:
            late = [(n, buf) for n, buf in reads if n > a]
            if late: races.append({"arrive_line": a, "reads_after": late})
        if not reads: races.append({"arrive_line": arrives[0], "note": "no reads of a filled buffer found in this loop; parser gap"})
    return {"status": "RACE" if races else "OK", "filled": sorted(filled), "loops_with_release": loops, "races": races}


if __name__ == "__main__":
    for f in sys.argv[1:]:
        r = check(open(f).read())
        print(f"{f}: {r['status']} filled={r['filled']}" + (f" loops={r.get('loops_with_release')} races={r.get('races')}" if r["status"] != "NOT-WS" else ""))

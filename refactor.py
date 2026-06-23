#!/usr/bin/env python3
"""
Path B step: ABSTRACTION REFACTORING -- mine GOOD reusable primitives, then crack the wall.

sec 46 failed because it kept only the FIRST solution per task (a transition-trick that doesn't
contain the clean detectors). The fix (compression/DreamCoder-flavoured): ENUMERATE the solution space
for each curriculum task and keep every distinct VALIDATED solving stream as a candidate abstraction --
because e.g. cnt2(01-detector) also solves 01-parity, the clean 01/10 detectors are IN the solution
space; we just mine them out. Then test whether the enriched library lets recursive synthesis reach
0110 (the search wall). Cross-seed validated throughout (no fake realities).
"""
import random
from recurse import (aggregate, eval_feature, f_not, f_lag, f_bin, f_atom,
                     split_idx, validate, AGGS, gen, trans_pat, L)
from library import synth_lib, LTASKS

PROBE = [tuple(random.Random(7 + j).randint(0, 1) for _ in range(L)) for j in range(24)]


def stream_sig(fn):
    return tuple(v for s in PROBE for v in fn(list(s)))


def synth_enum(task_fn, atoms, n=300, seed=0, max_depth=4, max_streams=8000, max_sol=60):
    """Enumerate distinct VALIDATED solving streams (not just the first)."""
    items = gen(task_fn, n, seed); ans = [a for _, a in items]
    ans_s = [a for _, a in gen(task_fn, n, seed, scramble=True)]
    seqs = [list(s) for s, _ in items]
    tr, dv, te = split_idx(n, seed); trdv = tr + dv
    streams = {}; solutions = []

    def add(expr, vals, fn):
        sig = tuple(v for row in vals for v in row)
        if sig in streams or len(streams) >= max_streams:
            return None
        streams[sig] = (expr, vals, fn)
        return streams[sig]

    def check(expr, vals, fn):
        for agg in AGGS:
            feat = [aggregate(row, agg) for row in vals]
            if eval_feature(feat, ans, tr, dv) > 0.98 and eval_feature(feat, ans, trdv, te) > 0.98 \
               and eval_feature(feat, ans_s, tr, dv) < 0.7:
                cs, _ = validate(fn, agg, task_fn, [101, 202, 303, 404, 505])
                if cs > 0.98:
                    solutions.append((agg, expr, fn))
                    return True
        return False

    cur = []
    for nm, fn in atoms:
        a = add(nm, [fn(list(s)) for s in seqs], fn)
        if a:
            check(*a); cur.append(a)
    for depth in range(1, max_depth + 1):
        new = []
        for (e, v, fn) in cur:
            for ex, nv, nfn in (("~" + e, [[1 - x for x in row] for row in v], f_not(fn)),
                                (f"lag1({e})", [[(row[i - 1] if i >= 1 else 0) for i in range(L)] for row in v], f_lag(fn, 1)),
                                (f"lag2({e})", [[(row[i - 2] if i >= 2 else 0) for i in range(L)] for row in v], f_lag(fn, 2))):
                a = add(ex, nv, nfn)
                if a:
                    check(*a); new.append(a)
                    if len(solutions) >= max_sol:
                        return solutions
        for (e1, v1, fn1) in cur:
            for (e2, v2, fn2) in list(streams.values()):
                for sym, op in (("&", lambda x, y: x & y), ("|", lambda x, y: x | y), ("^", lambda x, y: x ^ y)):
                    nv = [[op(x, y) for x, y in zip(r1, r2)] for r1, r2 in zip(v1, v2)]
                    a = add(f"({e1}{sym}{e2})", nv, f_bin(fn1, fn2, op))
                    if a:
                        check(*a); new.append(a)
                        if len(solutions) >= max_sol:
                            return solutions
            if len(streams) >= max_streams:
                break
        cur = new
        if not cur or len(streams) >= max_streams:
            break
    return solutions


def main():
    base = [("s", f_atom)]
    lib = list(base); seen = set()
    print("Mining validated solving streams (candidate abstractions) per curriculum task:")
    for tn in ["01-parity", "10-parity", "010-parity"]:
        sols = synth_enum(LTASKS[tn], base)
        added = 0
        for agg, expr, fn in sols:
            sig = stream_sig(fn)
            if sig not in seen:
                seen.add(sig); lib.append((f"A{len(lib)}", fn)); added += 1
        print(f"  {tn:11}: {len(sols)} validated solving streams, +{added} new abstractions  "
              f"(e.g. {sols[0][1] if sols else '-'})")
    print(f"\nEnriched library: {len(lib)} primitives.")
    print("0110-parity with the enriched (refactored) library:")
    r = synth_lib(LTASKS["0110-parity"], lib, max_depth=4)
    if r:
        agg, expr, fn, depth = r
        cs, accs = validate(fn, agg, LTASKS["0110-parity"], [111, 222, 333, 444, 555])
        if cs > 0.98:
            print(f"  SOLVED (depth {depth}, cross-seed {cs:.2f}):  {agg}({expr})")
            print("  -> abstraction refactoring CRACKED the search wall flat search could not.")
        else:
            print(f"  found but spurious (cross-seed {cs:.2f}) -- not a real solution.")
    else:
        print("  still NOT FOUND -- even mined abstractions insufficient; needs parameterised abstraction.")


if __name__ == "__main__":
    main()

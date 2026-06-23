#!/usr/bin/env python3
"""
Path B step: LIBRARY LEARNING (bootstrapping) -- solved programs become reusable primitives.

recurse.py hit the search wall at 0110 (expressible in the grammar, but flat search couldn't FIND it).
Here we process a curriculum and ADD each solved program (its feature stream) to the library as a new
ATOM, so harder tasks compose from learned pieces and the search starts richer. The decisive test:
does the grown library let it reach 0110 that flat search could not? Cross-seed validated throughout
(the no-fake-realities guard).
"""
from recurse import (aggregate, eval_feature, f_not, f_lag, f_bin, f_atom,
                     split_idx, validate, AGGS, gen, trans_pat, L)

LTASKS = {
    "01-parity":   lambda s: trans_pat(s, "01") % 2,
    "10-parity":   lambda s: trans_pat(s, "10") % 2,
    "010-parity":  lambda s: trans_pat(s, "010") % 2,
    "0110-parity": lambda s: trans_pat(s, "0110") % 2,
}


def synth_lib(task_fn, atoms, n=300, seed=0, max_depth=5, max_streams=9000):
    items = gen(task_fn, n, seed); ans = [a for _, a in items]
    ans_s = [a for _, a in gen(task_fn, n, seed, scramble=True)]
    seqs = [list(s) for s, _ in items]
    tr, dv, te = split_idx(n, seed); trdv = tr + dv
    streams = {}

    def add(expr, vals, fn):
        sig = tuple(v for row in vals for v in row)
        if sig in streams or len(streams) >= max_streams:
            return None
        streams[sig] = (expr, vals, fn)
        return streams[sig]

    def test(expr, vals, fn):
        for agg in AGGS:
            feat = [aggregate(row, agg) for row in vals]
            if eval_feature(feat, ans, tr, dv) > 0.98 and \
               eval_feature(feat, ans, trdv, te) > 0.98 and \
               eval_feature(feat, ans_s, tr, dv) < 0.7:
                return (agg, expr, fn)
        return None

    cur = []
    for nm, fn in atoms:
        a = add(nm, [fn(list(s)) for s in seqs], fn)
        if a:
            r = test(*a)
            if r:
                return r + (0,)
            cur.append(a)
    for depth in range(1, max_depth + 1):
        new = []
        for (e, v, fn) in cur:
            for ex, nv, nfn in (("~" + e, [[1 - x for x in row] for row in v], f_not(fn)),
                                (f"lag1({e})", [[(row[i - 1] if i >= 1 else 0) for i in range(L)] for row in v], f_lag(fn, 1)),
                                (f"lag2({e})", [[(row[i - 2] if i >= 2 else 0) for i in range(L)] for row in v], f_lag(fn, 2))):
                a = add(ex, nv, nfn)
                if a:
                    r = test(*a)
                    if r:
                        return r + (depth,)
                    new.append(a)
        for (e1, v1, fn1) in cur:
            for (e2, v2, fn2) in list(streams.values()):
                for sym, op in (("&", lambda x, y: x & y), ("|", lambda x, y: x | y), ("^", lambda x, y: x ^ y)):
                    nv = [[op(x, y) for x, y in zip(r1, r2)] for r1, r2 in zip(v1, v2)]
                    a = add(f"({e1}{sym}{e2})", nv, f_bin(fn1, fn2, op))
                    if a:
                        r = test(*a)
                        if r:
                            return r + (depth,)
                        new.append(a)
            if len(streams) >= max_streams:
                break
        cur = new
        if not cur or len(streams) >= max_streams:
            break
    return None


def attempt(task_fn, atoms):
    r = synth_lib(task_fn, atoms)
    if not r:
        return None
    agg, expr, fn, depth = r
    cs, accs = validate(fn, agg, task_fn, [101, 202, 303, 404, 505])
    return (agg, expr, fn, depth, cs) if cs > 0.98 else None


def main():
    base = [("s", f_atom)]
    print("Flat (base grammar) on 0110-parity:")
    r = attempt(LTASKS["0110-parity"], base)
    print("  " + ("SOLVED" if r else "NOT FOUND (the search wall from sec 45)"))

    print("\nLibrary learning -- process curriculum, add each solved program as a reusable primitive:")
    lib = list(base)
    for tn in ["01-parity", "10-parity", "010-parity"]:
        r = attempt(LTASKS[tn], lib)
        if r:
            agg, expr, fn, depth, cs = r
            pname = f"L{len(lib)}"
            lib.append((pname, fn))
            print(f"  {tn:11} SOLVED (d{depth}, cross-seed {cs:.2f}) -> learned {pname} = source-of {agg}: {expr}")
        else:
            print(f"  {tn:11} not solved")

    print(f"\n0110-parity WITH the grown library (atoms: {[a for a, _ in lib]}):")
    r = attempt(LTASKS["0110-parity"], lib)
    if r:
        agg, expr, fn, depth, cs = r
        print(f"  SOLVED (depth {depth}, cross-seed {cs:.2f}):  {agg}({expr})")
        print("  -> library learning crossed the search wall flat search could not.")
    else:
        print("  still NOT FOUND -> the learned programs are not the right reusable pieces;")
        print("     cracking it needs better abstraction extraction (refactoring/compression), not just")
        print("     adding winning streams. That is the honest next frontier.")


if __name__ == "__main__":
    main()

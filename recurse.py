#!/usr/bin/env python3
"""
Path B step: RECURSIVE invention -- grow the generative space itself, HONESTLY.

The system synthesises feature streams from a fixed grammar {s, not, lag_k, and/or/xor} to arbitrary
depth, aggregates (count mod m / ever / max-run / last), and searches by iterative deepening for a
feature that solves the task. Because this search space is huge (thousands of expressions), a single
content-disjoint split is NOT a sufficient honesty control -- so every candidate winner is re-validated
on FRESH held-out seeds, and only counts as SOLVED if it survives cross-seed. This is the guard against
search-overfitting (fake solutions that pass one split by luck).
"""
import random
from statistics import mean
from collections import Counter, defaultdict
from induce import gen

L = 12
AGGS = ["cnt2", "cnt3", "cnt4", "ever", "maxrun", "last"]


def aggregate(row, agg):
    if agg == "cnt2":
        return sum(row) % 2
    if agg == "cnt3":
        return sum(row) % 3
    if agg == "cnt4":
        return sum(row) % 4
    if agg == "ever":
        return int(any(row))
    if agg == "last":
        return row[-1]
    best = cur = 0
    for v in row:
        cur = cur + 1 if v else 0
        if cur > best:
            best = cur
    return min(best, 4)


def eval_feature(feat, ans, tr_idx, ev_idx):
    table = defaultdict(Counter)
    for i in tr_idx:
        table[feat[i]][ans[i]] += 1
    pred = {k: c.most_common(1)[0][0] for k, c in table.items()}
    gm = Counter(ans[i] for i in tr_idx).most_common(1)[0][0]
    return sum(1 for i in ev_idx if pred.get(feat[i], gm) == ans[i]) / len(ev_idx)


# stream functions (re-runnable on any data) -----------------------------------
def f_atom(s):
    return list(s)


def f_not(f):
    return lambda s: [1 - v for v in f(s)]


def f_lag(f, k):
    def g(s):
        r = f(s)
        return [(r[i - k] if i >= k else 0) for i in range(len(r))]
    return g


def f_bin(f1, f2, op):
    def g(s):
        r1, r2 = f1(s), f2(s)
        return [op(a, b) for a, b in zip(r1, r2)]
    return g


def split_idx(n, seed):
    idx = list(range(n)); random.Random(seed).shuffle(idx)
    a = n // 5
    return idx[2 * a:], idx[a:2 * a], idx[:a]              # tr, dv, te


def validate(fn, agg, task_fn, seeds):
    accs = []
    for sd in seeds:
        items = gen(task_fn, 300, sd); ans = [a for _, a in items]
        feat = [aggregate(fn(s), agg) for s, _ in items]
        tr, dv, te = split_idx(len(items), sd)
        accs.append(eval_feature(feat, ans, tr, te))
    return mean(accs), [round(a, 2) for a in accs]


def synth(task_fn, n=300, seed=0, max_depth=6, max_streams=9000):
    items = gen(task_fn, n, seed); ans = [a for _, a in items]
    ans_s = [a for _, a in gen(task_fn, n, seed, scramble=True)]
    seqs = [list(s) for s, _ in items]
    tr_idx, dv_idx, te_idx = split_idx(n, seed); trdv = tr_idx + dv_idx
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
            if eval_feature(feat, ans, tr_idx, dv_idx) > 0.98 and \
               eval_feature(feat, ans, trdv, te_idx) > 0.98 and \
               eval_feature(feat, ans_s, tr_idx, dv_idx) < 0.7:
                return (f"{agg}({expr})", agg, fn)
        return None

    atom = add("s", [list(x) for x in seqs], f_atom)
    r = test(*atom)
    if r:
        return r + (0,)
    cur = [atom]
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


def trans_pat(s, pat):
    P = [int(c) for c in pat]; m = len(P)
    return sum(1 for i in range(m - 1, len(s)) if list(s[i - m + 1:i + 1]) == P)


TASKS = {
    "01-parity":   lambda s: trans_pat(s, "01") % 2,
    "010-parity":  lambda s: trans_pat(s, "010") % 2,
    "0110-parity": lambda s: trans_pat(s, "0110") % 2,
}


def main():
    print("Recursive synthesis from {s, not, lag_k, and/or/xor} + aggs, with CROSS-SEED validation.\n")
    print(f"{'task':12} | in-search | cross-seed (5 fresh seeds) | verdict | program")
    for name, fn in TASKS.items():
        r = synth(fn)
        if not r:
            print(f"{name:12} | -         | -                          | NOT FOUND |")
            continue
        expr, agg, sfn, depth = r
        cs, accs = validate(sfn, agg, fn, [101, 202, 303, 404, 505])
        verdict = "REAL" if cs > 0.98 else "SPURIOUS (search-overfit)"
        print(f"{name:12} | found d{depth} | {cs:.2f}  {accs} | {verdict} | {expr}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cycle 11 — compute-vs-hold on a NON-degenerate aggregate. answer = popcount(features) mod m (m=4,
2-bit answer). Tests whether the "compute the aggregate" principle holds when the aggregate has m>2
values (parity was the degenerate 2-value case). CONTENT-DISJOINT split (the cycle-10 fix): train
and test use DISJOINT feature-pattern sets, stratified by aggregate value so both cover all m.
  * accum : running count mod m (2-bit counter), frozen at the boundary -> m aggregate-addresses
  * latch : holds the raw feature bits -> can't generalise a non-Hamming-smooth aggregate
scramble = a fixed but rule-less per-pattern answer (hash%m) -> unseen test patterns => chance.
"""
import itertools, random, statistics
import bench, blm, gated, region
R, NBITS = 6, 2


def producible(F):
    return [b for b in itertools.product([0, 1], repeat=F)
            if "111" not in "".join(map(str, b)) and b[-1] == 0]


def split_patterns(F, m, seed, test_frac=0.34):
    rng = random.Random(seed); from_collections = None
    from collections import defaultdict
    by = defaultdict(list)
    for p in producible(F):
        by[sum(p) % m].append(p)
    tr, te = [], []
    for g, ps in by.items():
        rng.shuffle(ps); c = max(1, int(len(ps) * test_frac))
        te += ps[:c]; tr += ps[c:]
    return tr, te


def answer(p, m, scramble):
    c = (hash(p) % m) if scramble else (sum(p) % m)
    return [(c >> (NBITS - 1 - i)) & 1 for i in range(NBITS)]


def items_of(pats, m, scramble):
    return [{"seq": list(p) + [1, 1, 1] + answer(p, m, scramble) + [0, 0, 0],
             "answer": answer(p, m, scramble), "ans_start": len(p) + 3} for p in pats]


def acc_mod(seq_str, t, m):
    idx = seq_str.find("111"); cut = t if (idx < 0 or t <= idx) else idx
    c = sum(1 for ch in seq_str[:cut] if ch == "1") % m
    return [(c >> (NBITS - 1 - i)) & 1 for i in range(NBITS)]


def _addr(seq_str, t, window, s, kind, wk, m):
    reg = region.region_bits(seq_str, t)
    extra = acc_mod(seq_str, t, m) if kind == "accum" else list(s)
    return tuple(reg + extra + list(window[-wk:] if wk else window))


def build_pairs(seq_str, kind, k, h, wk, m):
    g = blm.multi_latch_table(k, h); s = [0] * h; pairs = []
    for i in range(len(seq_str) - R):
        win = [int(c) for c in seq_str[i:i + R]]
        pairs.append((_addr(seq_str, i + R, win, s, kind, wk, m), int(seq_str[i + R])))
        s = blm.fold_state(s, int(seq_str[i]), "learned", h, g)
    return pairs


def gen(mch, prefix, n, kind, k, h, wk, m):
    g = blm.multi_latch_table(k, h); seq = list(prefix)
    window = seq[-R:] if len(seq) >= R else [0] * (R - len(seq)) + seq
    s = [0] * h
    for d in seq[:max(0, len(seq) - R)]:
        s = blm.fold_state(s, d, "learned", h, g)
    out = []
    for _ in range(n):
        a = _addr("".join(map(str, seq)), len(seq), window, s, kind, wk, m)
        b, _ = mch.predict(a); out.append(b); seq.append(b)
        d = window[0]; window = window[1:] + [b]; s = blm.fold_state(s, d, "learned", h, g)
    return out


def evalA(F, m, seed, kind, k, scramble, wk=3, ep=200):
    h = 9 if kind == "accum" else (k + (k + 1).bit_length())
    tr_p, te_p = split_patterns(F, m, seed)
    trn = items_of(tr_p, m, scramble); tst = items_of(te_p, m, scramble)
    extra = NBITS if kind == "accum" else h
    p = bench.params(R, "learned", h, seed, ep, "uniform"); p["A"] = 3 + extra + wk; p["alloc_radius"] = 0
    pooled = []
    for it in trn:
        pooled += build_pairs("".join(map(str, it["seq"])), kind, k, h, wk, m)
    mch = blm.Machine(p); mch.train(pooled)
    return sum(1 for it in tst if gen(mch, it["seq"][:it["ans_start"]], 2, kind, k, h, wk, m) == it["answer"]) / len(tst)


def main():
    m, S = 4, 5
    print(f"POPCOUNT mod {m} (CONTENT-DISJOINT split, {S} seeds). chance (2-bit exact) = {1.0/m:.2f}")
    print(f"{'F':>2} | {'accum int':>9} {'accum scr':>9} | {'latch-all int':>13} {'latch-all scr':>13} | #patterns")
    for F in (6, 8, 10):
        tr_p, te_p = split_patterns(F, m, 0)
        ai = statistics.mean(evalA(F, m, s, "accum", 2, False) for s in range(S))
        asc = statistics.mean(evalA(F, m, s, "accum", 2, True) for s in range(S))
        li = statistics.mean(evalA(F, m, s, "latch", F, False) for s in range(S))
        lsc = statistics.mean(evalA(F, m, s, "latch", F, True) for s in range(S))
        print(f"{F:>2} | {ai:>9.2f} {asc:>9.2f} | {li:>13.2f} {lsc:>13.2f} | {len(tr_p)+len(te_p)}")


if __name__ == "__main__":
    main()

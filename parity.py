#!/usr/bin/env python3
"""
Cycle 10 — recall vs COMPUTATION. Does recurrent-state + address + lookup do aggregation, or only
recall? Task: answer = PARITY of all F feature bits.  [F feature bits][111][p,p][000], p=parity(feats).

Three address states compared (all + region/position + small window):
  * accum     : a 1-bit running-XOR of feature bits, frozen at the boundary  -> = parity, 2 addresses
  * latch(k=2): holds the first 2 feature bits                               -> can't compute parity
  * latch(k=6): holds ALL 6 feature bits                                     -> HAS the inputs, but
                lookup can't generalise parity (not Hamming-smooth) on unseen patterns
Held-out = unseen feature patterns; scramble (random answer per pattern) = chance control.
"""
import random, statistics
import bench, blm, gated, region
R = 6


def gen_feats(F, rng):
    f, run = [], 0
    for _ in range(F):
        b = 0 if run >= 2 else rng.randint(0, 1)
        run = run + 1 if b == 1 else 0
        f.append(b)
    if F >= 1:
        f[-1] = 0                       # end in 0 -> 111 boundary unique
    return f[:F]


def dataset(F, K, seed, scramble=False):
    rng = random.Random(seed); items = []
    for k in range(K):
        feats = gen_feats(F, rng); p = sum(feats) & 1
        ans = [rng.randint(0, 1)] * 2 if scramble else [p, p]
        items.append({"seq": feats + [1, 1, 1] + ans + [0, 0, 0],
                      "answer": ans, "feat_id": k, "ans_start": F + 3})
    return items


def parity_acc(seq_str, t):
    idx = seq_str.find("111")
    cut = t if (idx < 0 or t <= idx) else idx           # features = before boundary, frozen after
    return sum(1 for c in seq_str[:cut] if c == "1") & 1


def _addr(seq_str, t, window, s, kind, wk):
    reg = region.region_bits(seq_str, t)
    extra = [parity_acc(seq_str, t)] if kind == "accum" else list(s)
    w = window[-wk:] if wk else window
    return tuple(reg + extra + list(w))


def build_pairs(seq_str, kind, k, h, wk):
    g = blm.multi_latch_table(k, h); s = [0] * h; pairs = []
    for i in range(len(seq_str) - R):
        window = [int(c) for c in seq_str[i:i + R]]
        pairs.append((_addr(seq_str, i + R, window, s, kind, wk), int(seq_str[i + R])))
        s = blm.fold_state(s, int(seq_str[i]), "learned", h, g)
    return pairs


def gen(m, prefix, n, kind, k, h, wk):
    g = blm.multi_latch_table(k, h); seq = list(prefix)
    window = seq[-R:] if len(seq) >= R else [0] * (R - len(seq)) + seq
    s = [0] * h
    for d in seq[:max(0, len(seq) - R)]:
        s = blm.fold_state(s, d, "learned", h, g)
    out = []
    for _ in range(n):
        a = _addr("".join(map(str, seq)), len(seq), window, s, kind, wk)
        b, _ = m.predict(a); out.append(b); seq.append(b)
        d = window[0]; window = window[1:] + [b]; s = blm.fold_state(s, d, "learned", h, g)
    return out


def evalp(F, K, seed, tr, te, kind, k, scramble, h=9, wk=3, ep=200):
    items = dataset(F, K, seed, scramble)
    trn = [it for it in items if it["feat_id"] in tr]; tst = [it for it in items if it["feat_id"] in te]
    extra = 1 if kind == "accum" else h
    p = bench.params(R, "learned", h, seed, ep, "uniform"); p["A"] = 3 + extra + wk; p["alloc_radius"] = 0
    pooled = []
    for it in trn:
        pooled += build_pairs("".join(map(str, it["seq"])), kind, k, h, wk)
    m = blm.Machine(p); m.train(pooled)
    return sum(1 for it in tst if gen(m, it["seq"][:it["ans_start"]], 2, kind, k, h, wk) == it["answer"]) / len(tst)


def main():
    F, K, S = 6, 40, 6
    print(f"PARITY-of-{F}-features (held-out, K={K}, {S} seeds). chance ~0.25 (4 patterns) / ~0.5 baseline")
    print(f"{'state':>14} | {'intact':>7} {'scramble':>8}")
    for name, kind, k in [("accum (1-bit XOR)", "accum", 2),
                          ("latch first-2", "latch", 2),
                          ("latch ALL-6", "latch", 6)]:
        ii = statistics.mean(evalp(F, K, s, gated.split(K, s)[0], gated.split(K, s)[2], kind, k, False) for s in range(S))
        ss = statistics.mean(evalp(F, K, s, gated.split(K, s)[0], gated.split(K, s)[2], kind, k, True) for s in range(S))
        print(f"{name:>14} | {ii:>7.2f} {ss:>8.2f}")


if __name__ == "__main__":
    main()

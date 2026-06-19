#!/usr/bin/env python3
"""
Learnable gated latch — make 'latch the first informative drop and hold' LEARNABLE.

The encoder is the structured gated-latch family (blm.gated_latch_table): a write-gate with a
latch prior, parameterised by a short binary write-schedule `w` (only len(w) learnable bits).
We LEARN `w` by enumerating its tiny 2^len(w) space on DEV bodies, and compare generalisation to
(a) `shift` (no latch -> horizon collapse) and (b) a FREE transition table hill-climbed on the same
dev set (the overfitting baseline from cycle 4). Clean train/dev/test body split; scramble control.
"""

import itertools
import random
import statistics

import bench
import blm

R = 6


def split(K, seed=0):
    rng = random.Random(1000 + seed)
    ids = list(range(K)); rng.shuffle(ids)
    n = max(2, K // 5)
    return set(ids[2 * n:]), set(ids[n:2 * n]), set(ids[:n])     # train, dev, test


def evalg(g, addr, h, L, K, seed, tr, te, scr, ep=250, alloc=0):
    items = bench.dataset(L, K, seed, scr)
    trn = [it for it in items if it["body_id"] in tr]
    tst = [it for it in items if it["body_id"] in te]
    p = bench.params(R, addr, h, seed, ep, "uniform"); p["learned_g"] = g; p["alloc_radius"] = alloc
    pooled = []
    for it in trn:
        pooled += blm.make_pairs("".join(map(str, it["seq"])), R, addr, h if addr != "register" else 0, g)
    m = blm.Machine(p); m.train(pooled)
    return sum(1 for it in tst
               if m.generate_primed(it["seq"][:it["ans_start"]], len(it["answer"])) == it["answer"]) / len(tst)


def learn_schedule(L, K, tr, dev, h=4, C=4, seeds=3):
    # Tie-break (verification fix): on equal dev score prefer a latch (w[0]=1) with fewest writes,
    # so chance-vs-chance dev ties don't default to the degenerate all-zeros schedule.
    best_w, best_key = None, None
    for bits in itertools.product([0, 1], repeat=C):
        w = list(bits)
        if sum(w) == 0:
            continue                                  # degenerate: never latches
        g = blm.gated_latch_table(w, h)
        score = statistics.mean(evalg(g, "learned", h, L, K, s, tr, dev, False, ep=150, alloc=1) for s in range(seeds))
        key = (round(score, 3), 1 if w[0] else 0, -sum(w))
        if best_key is None or key > best_key:
            best_key, best_w = key, w
    return best_w, best_key[0]


def free_table(L, K, tr, dev, h=4, iters=80, seed=0):
    rng = random.Random(seed); g = blm.shift_table(h)[:]
    best = statistics.mean(evalg(g, "learned", h, L, K, s, tr, dev, False, 150, 1) for s in range(2))
    for _ in range(iters):
        c = g[:]
        for _ in range(rng.choice([1, 1, 2])):
            c[rng.randrange(len(c))] = rng.randrange(1 << h)
        sc = statistics.mean(evalg(c, "learned", h, L, K, s, tr, dev, False, 150, 1) for s in range(2))
        if sc >= best:
            g, best = c, sc
    return g, best


def main():
    K, Ltrain, h = 40, 8, 4
    tr, dev, te = split(K)
    print(f"bodies: train={len(tr)} dev={len(dev)} test={len(te)}; h={h}; shift horizon = R+h-4 = {R + h - 4}")
    w, wdev = learn_schedule(Ltrain, K, tr, dev, h=h)
    ft, ftdev = free_table(Ltrain, K, tr, dev, h=h)
    print(f"gated-latch LEARNED w={w} (dev {wdev:.2f});  free-table (dev {ftdev:.2f})")
    gw = blm.gated_latch_table(w, h)
    print(f"\nTEST (held-out bodies), K={K}, 4 seeds, alloc=0 — intact/scramble/gap:")
    print(f"{'L':>3} | {'shift':>17} | {'free-table':>17} | {'gated-latch':>17}")
    print("-" * 64)
    for L in (4, 6, 8, 10):
        def cell(g, addr):
            i = statistics.mean(evalg(g, addr, h, L, K, s, tr, te, False) for s in range(4))
            c = statistics.mean(evalg(g, addr, h, L, K, s, tr, te, True) for s in range(4))
            return f"{i:.2f}/{c:.2f}/{i - c:+.2f}"
        print(f"{L:>3} | {cell(None, 'shift'):>17} | {cell(ft, 'learned'):>17} | {cell(gw, 'learned'):>17}")


if __name__ == "__main__":
    main()

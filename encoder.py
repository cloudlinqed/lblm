#!/usr/bin/env python3
"""
Learned recurrent encoder — first gradient-free attempt at the "learned hash".

The recurrent state update becomes a LEARNED transition table g: (state, dropped_bit) -> state
(addr_mode='learned', h=3 -> 16 entries). `shift` and `fold` are fixed points of this family;
we hill-climb g to maximise held-out generalisation, to see whether a learned binary code lifts
the body-length floor where the hand-coded `shift` collapses (cycle 14b: L4->L6 ~0.80->0.53).

Clean protocol (no leakage): bodies are split train / dev / test.
  * the memory is trained on TRAIN bodies;
  * g is SELECTED by its held-out accuracy on DEV bodies;
  * the winner is reported on separate TEST bodies, with the rule-scramble control.
"""

import random
import statistics

import bench
import blm

R, H = 6, 3                      # register width, learned-state width


def split(K, seed):
    rng = random.Random(1000 + seed)
    ids = list(range(K)); rng.shuffle(ids)
    n = max(2, K // 5)
    return set(ids[2 * n:]), set(ids[n:2 * n]), set(ids[:n])   # train, dev, test


def evalg(g, addr, L, K, seed, train_ids, test_ids, scr, epochs, alloc):
    items = bench.dataset(L, K, seed, scr)
    tr = [it for it in items if it["body_id"] in train_ids]
    te = [it for it in items if it["body_id"] in test_ids]
    p = bench.params(R, addr, H, seed, epochs, "uniform")
    p["learned_g"] = g; p["alloc_radius"] = alloc
    pooled = []
    for it in tr:
        pooled += blm.make_pairs("".join(map(str, it["seq"])), R, addr, H if addr != "register" else 0, g)
    m = blm.Machine(p); m.train(pooled)
    return sum(1 for it in te
               if m.generate_primed(it["seq"][:it["ans_start"]], len(it["answer"])) == it["answer"]) / len(te)


def hillclimb(L, K, train_ids, dev_ids, iters, seed=0):
    rng = random.Random(seed)
    g = blm.shift_table(H)[:]                                  # init from shift
    best = evalg(g, "learned", L, K, seed, train_ids, dev_ids, False, epochs=150, alloc=1)
    for _ in range(iters):
        cand = g[:]
        for _ in range(rng.choice([1, 1, 2])):
            cand[rng.randrange(len(cand))] = rng.randrange(1 << H)
        score = evalg(cand, "learned", L, K, seed, train_ids, dev_ids, False, epochs=150, alloc=1)
        if score >= best:
            g, best = cand, score
    return g, best


def main():
    K, Ltrain, iters = 40, 6, 80
    train_ids, dev_ids, test_ids = split(K, 0)
    print(f"bodies: train={len(train_ids)} dev={len(dev_ids)} test={len(test_ids)}")
    print(f"hill-climbing encoder g at L={Ltrain} ({iters} iters, init=shift)...")
    g, dev = hillclimb(Ltrain, K, train_ids, dev_ids, iters)
    print(f"learned g: dev-score={dev:.2f}   g={g}")
    print(f"\nTEST (held-out bodies), full capacity, 3 seeds — intact / scramble:")
    print(f"{'addr':>9} | {'L4 int':>6} {'L4 scr':>6} | {'L6 int':>6} {'L6 scr':>6}")
    print("-" * 46)
    for addr, gg in [("register", None), ("shift", None), ("fold", None), ("learned", g)]:
        cells = []
        for L in (4, 6):
            i = statistics.mean(evalg(gg, addr, L, K, s, train_ids, test_ids, False, 250, 0) for s in range(3))
            c = statistics.mean(evalg(gg, addr, L, K, s, train_ids, test_ids, True, 250, 0) for s in range(3))
            cells += [i, c]
        print(f"{addr:>9} | {cells[0]:>6.2f} {cells[1]:>6.2f} | {cells[2]:>6.2f} {cells[3]:>6.2f}")


if __name__ == "__main__":
    main()

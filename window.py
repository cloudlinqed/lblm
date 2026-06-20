#!/usr/bin/env python3
"""
Window compression — the readout half of the encoder (cycle 6).

Cycle 5 proved the latch removes the memory HORIZON, but accuracy still attenuated with length
because the (window body x latched-state) space the readout must cover grows with L. Fix: the
register slides at width R (driving the latch), but the ADDRESS keeps only the last `win_keep`
window bits — the latch carries long-range memory, so the address window only needs the local
structure (boundary + recent outputs); the body bits are dropped.

We jointly LEARN (latch write-schedule w, win_keep) on dev bodies and test generalisation across
body length on held-out bodies, with the rule-scramble control. Hypothesis: a small win_keep
(boundary-width) makes the held-out accuracy length-INDEPENDENT, removing the cycle-5 attenuation.
"""

import itertools
import statistics

import bench
import blm
import gated

R = 6


def evalg(g, h, L, K, seed, tr, te, scr, wk, ep=250, alloc=0):
    items = bench.dataset(L, K, seed, scr)
    trn = [it for it in items if it["body_id"] in tr]
    tst = [it for it in items if it["body_id"] in te]
    p = bench.params(R, "learned", h, seed, ep, "uniform")
    p["learned_g"] = g; p["alloc_radius"] = alloc; p["win_keep"] = wk
    p["A"] = (wk or R) + h
    pooled = []
    for it in trn:
        pooled += blm.make_pairs("".join(map(str, it["seq"])), R, "learned", h, g, wk)
    m = blm.Machine(p); m.train(pooled)
    return sum(1 for it in tst
               if m.generate_primed(it["seq"][:it["ans_start"]], len(it["answer"])) == it["answer"]) / len(tst)


def learn(L, K, tr, dev, h=4, C=4, seeds=4):
    """Jointly learn (write-schedule w, win_keep). Selection objective (verified fix): maximise
    intact WHILE keeping scramble near chance — score = intact - 2*|scramble - 0.5|. Raw intact (or
    raw scramble-gap) would pick an over-large window that overfits the body (sub-chance scramble);
    penalising scramble deviation from 0.5 selects the clean, genuinely-transferring win_keep.
    Needs adequate dev power; tie-break prefers latch / fewer writes / smaller window."""
    best_w, best_wk, best_key = None, None, None
    for wk in (2, 3, 4, 5, 6):
        for bits in itertools.product([0, 1], repeat=C):
            w = list(bits)
            if sum(w) == 0:
                continue
            g = blm.gated_latch_table(w, h)
            intact = statistics.mean(evalg(g, h, L, K, s, tr, dev, False, wk, ep=150, alloc=1) for s in range(seeds))
            scram = statistics.mean(evalg(g, h, L, K, s, tr, dev, True, wk, ep=150, alloc=1) for s in range(seeds))
            score = intact - 2 * abs(scram - 0.5)      # high intact AND scramble near chance
            key = (round(score, 3), 1 if w[0] else 0, -sum(w), -wk)
            if best_key is None or key > best_key:
                best_key, best_w, best_wk = key, w, wk
    return best_w, best_wk, best_key[0]


def main():
    K, Ltrain, h = 40, 10, 4
    tr, dev, te = gated.split(K)
    w, wk, dev = learn(Ltrain, K, tr, dev, h=h)
    print(f"LEARNED: w={w}, win_keep={wk}  (dev {dev:.2f})")
    g = blm.gated_latch_table(w, h)
    print(f"\nHeld-out (K={K}, 6 seeds) — full window (cycle 5) vs learned window-compressed:")
    print(f"{'L':>3} | {'full-window (wk=6)':>20} | {'compressed (wk='+str(wk)+')':>20}")
    print("-" * 50)
    for L in (4, 6, 8, 10, 12):
        def cell(wkk):
            i = statistics.mean(evalg(g, h, L, K, s, tr, te, False, wkk) for s in range(6))
            c = statistics.mean(evalg(g, h, L, K, s, tr, te, True, wkk) for s in range(6))
            return f"{i:.2f}/{c:.2f}/{i - c:+.2f}"
        print(f"{L:>3} | {cell(6):>20} | {cell(wk):>20}")


if __name__ == "__main__":
    main()

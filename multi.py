#!/usr/bin/env python3
"""
Multi-feature memory (cycle 7) — does the structured-latch approach COMPOSE?

A 2-feature recall bench: [TYPE1 TYPE2][shared BODY len L][BOUNDARY 111][ANSWER 2b][STOP 000].
The two type bits at the start jointly determine the answer; the body (shared across all four
classes) sits between, so the answer-position window is class-identical — only MEMORY of both type
bits resolves it. Two answer functions:
  * echo: answer = [t1, t2]              (independent — each answer bit from one feature)
  * xor : answer = [t1^t2, t1^t2]        (joint — needs BOTH features combined)

Encoder: the MULTI-LATCH (blm.multi_latch_table(k,h)) holds the first k dropped bits (the k type
bits) then freezes. k=1 = the single latch (cycle 5) — can hold only ONE feature. Hypothesis: k=2
holds both and solves the task (with window compression win_keep=3, the boundary width), while k=1
cannot. k is LEARNED on dev by the scramble-clean objective. Body-disjoint train/dev/test split;
the rule-scramble control randomises the class->answer map per body.
"""

import itertools
import math
import random
import statistics

import bench
import blm
import gated

R = 6
BOUNDARY, STOP = [1, 1, 1], [0, 0, 0]
CLASSES = list(itertools.product([0, 1], [0, 1]))


def ans(t1, t2, mode):
    if mode == "echo":
        return [t1, t2]
    if mode == "xor":
        p = t1 ^ t2
        return [p, p]
    raise ValueError(mode)


def gen_body(L, rng):
    """Shared body: starts [0,0] (so TYPE ++ body never make a spurious 111), no triple-1, and
    ENDS in 0 (so body ++ boundary never fuse into an early 111 -> the 111 boundary is unique)."""
    body, run = [0, 0], 0
    for _ in range(max(0, L - 2)):
        b = 0 if run >= 2 else rng.randint(0, 1)
        run = run + 1 if b == 1 else 0
        body.append(b)
    body = body[:L]
    if L >= 1:
        body[-1] = 0                      # body ends in 0 -> the boundary 111 is unambiguous
    return body


def dataset(L, K, seed, mode, scramble=False):
    rng = random.Random(seed)
    items = []
    for k in range(K):
        body = gen_body(L, rng)
        amap = {c: ans(*c, mode) for c in CLASSES}
        if scramble:                                        # randomise class->answer per body
            vals = [ans(*c, mode) for c in CLASSES]
            rng.shuffle(vals)
            amap = {c: vals[i] for i, c in enumerate(CLASSES)}
        for c in CLASSES:
            a = amap[c]
            items.append({"seq": [c[0], c[1]] + body + BOUNDARY + a + STOP,
                          "answer": a, "body_id": k, "ans_start": 2 + L + len(BOUNDARY)})
    return items


def evalc(g, h, L, K, seed, tr, te, mode, scr, wk=3, ep=200, alloc=0):
    items = dataset(L, K, seed, mode, scr)
    trn = [it for it in items if it["body_id"] in tr]
    tst = [it for it in items if it["body_id"] in te]
    p = bench.params(R, "learned", h, seed, ep, "uniform")
    p["learned_g"] = g; p["alloc_radius"] = alloc; p["win_keep"] = wk; p["A"] = (wk or R) + h
    pooled = []
    for it in trn:
        pooled += blm.make_pairs("".join(map(str, it["seq"])), R, "learned", h, g, wk)
    m = blm.Machine(p); m.train(pooled)
    cor = sum(1 for it in tst
              if m.generate_primed(it["seq"][:it["ans_start"]], len(it["answer"])) == it["answer"])
    return cor, len(tst)


def pool(klat, h, L, K, mode, scr, seeds, wk=3):
    c = t = 0
    g = blm.multi_latch_table(klat, h)
    for s in range(seeds):
        tr, dev, te = gated.split(K, s)
        a, b = evalc(g, h, L, K, s, tr, te, mode, scr, wk)
        c += a; t += b
    p = c / t
    return p, 1.96 * math.sqrt(p * (1 - p) / t), t


def learn_k(L, K, mode, h=4, seeds=4):
    """Learn the number of features to latch by the scramble-clean objective."""
    best_k, best = None, None
    for klat in (1, 2):
        g = blm.multi_latch_table(klat, h)
        inta, scra = [], []
        for s in range(seeds):
            tr, dev, te = gated.split(K, s)
            ci, ti = evalc(g, h, L, K, s, tr, dev, mode, False); inta.append(ci / ti)
            cs, ts = evalc(g, h, L, K, s, tr, dev, mode, True); scra.append(cs / ts)
        i, sc = statistics.mean(inta), statistics.mean(scra)
        score = i - 2 * abs(sc - 0.5)
        if best is None or score > best:
            best, best_k = score, klat
    return best_k, best


def main():
    K, L, h, S = 48, 8, 4, 12
    print(f"Multi-feature recall (2 type bits), K={K}, L={L}, win_keep=3, pooled over {S} seeds (95% CI).")
    for mode in ("echo", "xor"):
        lk, _ = learn_k(L, K, mode)
        print(f"\n=== answer = {mode} ===   learned k = {lk}  (need 2 features)")
        print(f"  {'latch k':>8} | {'intact':>12} {'scramble':>9}")
        for klat in (1, 2):
            pi, ci, n = pool(klat, h, L, K, mode, False, S)
            ps, _, _ = pool(klat, h, L, K, mode, True, S)
            tag = "  <- needs both" if klat == 2 else "  (1 feature only)"
            print(f"  {klat:>8} | {pi:.2f}±{ci:.2f}   {ps:>9.2f}{tag}")


if __name__ == "__main__":
    main()

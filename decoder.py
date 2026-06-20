#!/usr/bin/env python3
"""
Learned decoder readout (cycle 8) — fix the EXTRACTION bottleneck.

The memory holds the features (the latch is solved), but the global Hamming-vote readout can't pull
out ONE specific held bit when other bits look similar (echo bit2 capped ~0.73). This replaces the
vote with a LEARNED LINEAR DECODER over the address: a gradient-free perceptron on features
  phi(address) = [1] + [each address bit] + [each pairwise AND of two address bits]
so it can SELECT a single bit (echo) AND COMBINE two (XOR is linear once pairwise terms exist).
Trained discriminatively (perceptron rule), it can put weight on the discriminative bit instead of
averaging over all of them. Tested head-to-head with the Hamming-vote on the 2-feature bench.
"""
import random
import statistics
import bench
import blm
import gated
import multi

R = 6


def features(addr):
    A = len(addr)
    f = [1.0] + [float(b) for b in addr]
    for i in range(A):
        for j in range(i + 1, A):
            f.append(float(addr[i] & addr[j]))
    return f


class Perceptron:
    def __init__(self, nf, lr=1.0):
        self.w = [0.0] * nf
        self.lr = lr

    def raw(self, f):
        return sum(wi * fi for wi, fi in zip(self.w, f))

    def predict(self, addr):
        return 1 if self.raw(features(addr)) > 0 else 0

    def train(self, pairs, epochs=200, seed=0):
        rng = random.Random(seed)
        for _ in range(epochs):
            order = pairs[:]; rng.shuffle(order)
            for addr, y in order:
                f = features(addr)
                pred = 1 if self.raw(f) > 0 else 0
                if pred != y:
                    s = 1.0 if y == 1 else -1.0
                    self.w = [wi + self.lr * s * fi for wi, fi in zip(self.w, f)]


def gen_decoder(dec, prefix, n, h, g, wk):
    prefix = list(prefix)
    window = prefix[-R:] if len(prefix) >= R else [0] * (R - len(prefix)) + prefix
    s = [0] * h
    for d in prefix[:max(0, len(prefix) - R)]:
        s = blm.fold_state(s, d, "learned", h, g)
    out = []
    for _ in range(n):
        b = dec.predict(blm.addr_of(window, s, wk))
        out.append(b)
        d = window[0]; window = window[1:] + [b]; s = blm.fold_state(s, d, "learned", h, g)
    return out


def eval_decoder(L, K, seed, tr, te, mode, scr, klat=2, h=4, wk=3, epochs=200):
    g = blm.multi_latch_table(klat, h)
    items = multi.dataset(L, K, seed, mode, scr)
    trn = [it for it in items if it["body_id"] in tr]
    tst = [it for it in items if it["body_id"] in te]
    pairs = []
    for it in trn:
        pairs += blm.make_pairs("".join(map(str, it["seq"])), R, "learned", h, g, wk)
    A = (wk or R) + h
    dec = Perceptron(1 + A + A * (A - 1) // 2)
    dec.train(pairs, epochs, seed)
    full = bit1 = bit2 = 0
    for it in tst:
        pre, a = it["seq"][:it["ans_start"]], it["answer"]
        gen = gen_decoder(dec, pre, len(a), h, g, wk)
        full += (gen == a); bit1 += (gen[0] == a[0])
        bit2 += (gen_decoder(dec, pre + [a[0]], 1, h, g, wk)[0] == a[1])
    n = len(tst)
    return full / n, bit1 / n, bit2 / n


def pool(mode, scr, seeds, K=40, L=8):
    fu = b1 = b2 = vt = 0.0
    for s in range(seeds):
        tr, dev, te = gated.split(K, s)
        f, x, y = eval_decoder(L, K, s, tr, te, mode, scr)
        fu += f; b1 += x; b2 += y
        c, t = multi.evalc(blm.multi_latch_table(2, 4), 4, L, K, s, tr, te, mode, scr)  # vote baseline
        vt += c / t
    n = seeds
    return fu / n, b1 / n, b2 / n, vt / n


def main():
    S = 8
    print(f"Learned DECODER vs Hamming-VOTE readout (2-feature bench, K=40, L=8, {S} seeds)")
    print(f"{'mode':>5} | {'DECODER full':>12} {'bit1':>5} {'bit2':>5} | {'VOTE full':>9} | {'scramble(dec full)':>18}")
    for mode in ("echo", "xor"):
        f, b1, b2, vt = pool(mode, False, S)
        fs, _, _, _ = pool(mode, True, S)
        print(f"{mode:>5} | {f:>12.2f} {b1:>5.2f} {b2:>5.2f} | {vt:>9.2f} | {fs:>18.2f}")


if __name__ == "__main__":
    main()

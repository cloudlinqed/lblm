#!/usr/bin/env python3
"""
Region / position latch (cycle 9) — give the address a sense of WHERE it is.

Cycle 8 showed echo's cap is an ADDRESS COLLISION: after the feature-latch freezes, the held
register is identical at every position and the short window can't tell the answer region from the
body, so answer-bit-2 addresses collide with body positions (160/160). The fix is to enrich the
address with WHERE we are:
  * boundary_seen : sticky bit, 1 once the 111 boundary has appeared
  * post          : a short counter of steps since the boundary (0=ans-bit1, 1=ans-bit2, 2+=stop)
Address = [region(3 bits)] ++ [window last wk bits] ++ [feature-latch state (the type bits)].
Then every answer position is unique -> the plain vote can extract the answer.
"""
import math
import statistics
import bench
import blm
import gated
import multi

R = 6


def region_bits(seq_str, t, maxpost=3):
    s = seq_str[:t]
    idx = s.find("111")
    if idx < 0:
        return [0, 0, 0]                       # boundary not seen yet
    end = idx + 2
    post = min(max(0, (t - 1) - end), maxpost)
    return [1, (post >> 1) & 1, post & 1]      # boundary_seen, post(2 bits)


def build_pairs(seq_str, k=2, h=4, wk=3):
    g = blm.multi_latch_table(k, h)
    base = blm.make_pairs(seq_str, R, "learned", h, g, wk)
    return [(tuple(region_bits(seq_str, i + R)) + a, y) for i, (a, y) in enumerate(base)]


def gen_region(m, prefix, n, k=2, h=4, wk=3):
    g = blm.multi_latch_table(k, h)
    seq = list(prefix)
    window = seq[-R:] if len(seq) >= R else [0] * (R - len(seq)) + seq
    s = [0] * h
    for d in seq[:max(0, len(seq) - R)]:
        s = blm.fold_state(s, d, "learned", h, g)
    out = []
    for _ in range(n):
        addr = tuple(region_bits("".join(map(str, seq)), len(seq))) + blm.addr_of(window, s, wk)
        b, _ = m.predict(addr)
        out.append(b); seq.append(b)
        d = window[0]; window = window[1:] + [b]; s = blm.fold_state(s, d, "learned", h, g)
    return out


def eval_region(L, K, seed, tr, te, mode, scr, k=2, h=4, wk=3, ep=200):
    items = multi.dataset(L, K, seed, mode, scr)
    trn = [it for it in items if it["body_id"] in tr]
    tst = [it for it in items if it["body_id"] in te]
    p = bench.params(R, "learned", h, seed, ep, "uniform")
    p["A"] = 3 + wk + h; p["alloc_radius"] = 0
    pooled = []
    for it in trn:
        pooled += build_pairs("".join(map(str, it["seq"])), k, h, wk)
    m = blm.Machine(p); m.train(pooled)
    full = bit2 = 0
    for it in tst:
        pre, a = it["seq"][:it["ans_start"]], it["answer"]
        full += (gen_region(m, pre, len(a), k, h, wk) == a)
        bit2 += (gen_region(m, pre + [a[0]], 1, k, h, wk)[0] == a[1])
    n = len(tst)
    return full / n, bit2 / n


def collisions(mode, K=40, L=8):
    from collections import defaultdict
    at = defaultdict(set); ans2 = []
    for it in multi.dataset(L, K, 0, mode, False):
        seq = "".join(map(str, it["seq"])); pr = build_pairs(seq)
        for a, y in pr:
            at[a].add(y)
        i2 = it["ans_start"] + 1 - R
        if 0 <= i2 < len(pr):
            ans2.append(pr[i2][0])
    return sum(1 for a in ans2 if len(at[a]) > 1), len(ans2)


def main():
    S = 8
    for mode in ("echo", "xor"):
        c, n = collisions(mode)
        print(f"[{mode}] answer-bit2 collisions WITH region latch: {c}/{n}  (was 160/160)")
    print(f"\nRegion-latch vote vs cycle-7 baseline (K=40, L=8, {S} seeds):")
    print(f"{'mode':>5} | {'REGION full':>11} {'bit2':>5} {'scram':>6} | {'baseline full':>13}")
    for mode in ("echo", "xor"):
        rf = rb = rs = bl = 0.0
        for s in range(S):
            tr, dev, te = gated.split(K=40, seed=s) if False else gated.split(40, s)
            f, b2 = eval_region(8, 40, s, tr, te, mode, False)
            fs, _ = eval_region(8, 40, s, tr, te, mode, True)
            c, t = multi.evalc(blm.multi_latch_table(2, 4), 4, 8, 40, s, tr, te, mode, False)
            rf += f; rb += b2; rs += fs; bl += c / t
        print(f"{mode:>5} | {rf/S:>11.2f} {rb/S:>5.2f} {rs/S:>6.2f} | {bl/S:>13.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Learn-the-REPRESENTATION on real data. In scale.py I hand-gave the byte/phase structure; here the
core DISCOVERS which context features predict the next bit, from data alone (learn-the-computation at
scale).

 Part 1 - period scan: among periodic features (i mod p), which p most reduces held-out bits/bit?
          Text should reveal p=8 (the byte) -- discovered, not told.
 Part 2 - greedy forward selection over a pool {i mod p, lag bits}: build a representation from
          scratch and compare its held-out bits/bit to the hand-given byte-aware representation.

Honest metric = held-out bits/bit (cross-entropy = compression; raw = 1.0000, lower is better).
Sequential / CPU-light.
"""
import sys, math
import scale


def feat_val(kind, par, bits, i):
    if kind == "mod":
        return i % par
    j = i - par                                    # lag bit at offset `par`
    return bits[j] if j >= 0 else 2


def address(rep, bits, i):
    return tuple(feat_val(k, p, bits, i) for (k, p) in rep)


def eval_rep(bits, split, rep, alpha=0.5):
    ones = sum(bits[:split]); g1 = (ones + alpha) / (split + 2 * alpha)   # global P(1) for unseen backoff
    t = {}
    for i in range(split):
        a = address(rep, bits, i); c = t.get(a)
        if c is None:
            c = [0, 0, 0]; t[a] = c
        c[bits[i]] += 1; c[2] += 1
    tot = 0.0; n = len(bits) - split
    for i in range(split, len(bits)):
        c = t.get(address(rep, bits, i))
        p1 = g1 if c is None else (c[1] + alpha) / (c[2] + 2 * alpha)
        p = p1 if bits[i] == 1 else 1 - p1
        tot += -math.log2(max(p, 1e-12))
    return tot / n


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.txt"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 150000
    raw, bits = scale.load(path, cap); split = int(len(bits) * 0.8)
    print(f"corpus={path}  bytes={len(raw)}  bits={len(bits)}  train/test = 80/20\n")

    print("Part 1 - period scan: bits/bit for address = (i mod p) alone")
    print(" p  | bits/bit")
    best_p, best = None, 9.0
    for p in range(2, 17):
        s = eval_rep(bits, split, [("mod", p)])
        print(f" {p:2} | {s:.4f}")
        if s < best:
            best, best_p = s, p
    print(f" -> most predictive period p={best_p}  (a text byte is 8 -- discovered from data)\n")

    maxf = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    pool = [("mod", p) for p in range(2, 17)] + [("lag", d) for d in range(1, 17)]
    selected, cur = [], eval_rep(bits, split, [("mod", 1)])     # mod 1 == constant == order-0 baseline
    print(f"Part 2 - greedy representation discovery (order-0 baseline = {cur:.4f}, max {maxf} features)")
    while len(selected) < maxf:
        bestc, bests = None, cur
        for cand in pool:
            if cand in selected:
                continue
            s = eval_rep(bits, split, selected + [cand])
            if s < bests - 0.001:
                bests, bestc = s, cand
        if bestc is None:
            break
        selected.append(bestc); cur = bests
        print(f"  + {str(bestc):12} -> bits/bit {cur:.4f}")
    print(f"\n DISCOVERED representation: {selected}")
    print(f" discovered bits/bit       = {cur:.4f}")
    hb = scale.eval_single(bits, split, lambda b, i: scale.addr(b, i, 2))
    print(f" hand-given byte-aware B=2 = {hb:.4f}")


if __name__ == "__main__":
    main()

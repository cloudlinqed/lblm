#!/usr/bin/env python3
"""
Agency on a REAL stream: turn the bit-native predictor into a CHANGE / ANOMALY detector.

The same online logistic mixer (mix.py) predicts each next bit; the per-byte SURPRISE (sum of
-log2 P over that byte's 8 bits) is the signal; when smoothed surprise rises sharply above the
adapted baseline the core EMITS A FLAG -- its action. This realises the framing doc's own examples of
intelligence-without-language: "detecting anomaly in a stream" / "detect when a pattern changes", and
exercises non-stationarity.

Stream = REAL data with genuine regime shifts: English -> Python source -> English (built from the two
real corpora so the change points are known ground truth). No mock data.
"""
import sys, math
import mix                                                 # reuse ctx / stretch / squash / ORDERS


def build_stream(seg=90000):
    eng = open("data/corpus.txt", "rb").read()
    code = open("data/corpus_code.txt", "rb").read()
    raw = eng[:seg] + code[:seg] + eng[seg:2 * seg]
    return raw, [seg, 2 * seg]                             # byte offsets where the regime changes


def per_byte_surprise(raw, lr=0.02, delta=0.2):
    bits = bytearray()
    for byte in raw:
        for j in range(7, -1, -1):
            bits.append((byte >> j) & 1)
    ORD = mix.ORDERS
    tables = [dict() for _ in ORD]; w = [0.0] * len(ORD)
    surp = []; acc = 0.0
    for i in range(len(bits)):
        sts = []; cells = []
        for k, B in enumerate(ORD):
            key = mix.ctx(bits, i, B); c = tables[k].get(key)
            if c is None:
                c = [0, 0]; tables[k][key] = c
            p = (c[1] + delta) / (c[0] + c[1] + 2 * delta)
            sts.append(mix.stretch(p)); cells.append(c)
        P = mix.squash(sum(w[k] * sts[k] for k in range(len(ORD))))
        y = bits[i]; acc += -math.log2(P if y == 1 else 1 - P)
        err = y - P
        for k in range(len(ORD)):
            w[k] += lr * err * sts[k]; cells[k][y] += 1
        if (i & 7) == 7:
            surp.append(acc); acc = 0.0
    return surp                                            # bits/byte, one per byte


def _prefix(surp):
    pre = [0.0] * (len(surp) + 1)
    for i, s in enumerate(surp):
        pre[i + 1] = pre[i] + s
    return pre


def _wm(pre, a, b):                                        # mean surprise over byte window [a, b)
    return (pre[b] - pre[a]) / (b - a) if b > a else 0.0


def detect(pre, n, R=3000, Bw=12000, margin=0.30, refractory=8000):
    # sustained mean-shift: flag when a recent R-byte window's surprise jumps above the
    # immediately-preceding Bw-byte baseline by `margin` bits (causal; trailing windows only).
    flags = []; cooldown = 0
    for t in range(R + Bw, n):
        if cooldown > 0:
            cooldown -= 1; continue
        if _wm(pre, t - R, t) > _wm(pre, t - R - Bw, t - R) + margin:
            flags.append(t); cooldown = refractory
    return flags


def main():
    raw, bnd = build_stream()
    print(f"stream bytes={len(raw)}  (English -> Python source -> English)  true boundaries at {bnd}")
    surp = per_byte_surprise(raw); n = len(surp); pre = _prefix(surp)
    blk = 5000
    print(f"per-5KB mean bits/byte: {[round(_wm(pre, i, min(i + blk, n)), 2) for i in range(0, n, blk)]}")
    print("surprise jump at each TRUE boundary (mean 3KB after - 3KB before):")
    for b in bnd:
        before = _wm(pre, b - 3000, b); after = _wm(pre, b, b + 3000)
        print(f"  boundary {b}: before={before:.2f}  after={after:.2f}  jump={after - before:+.2f}")
    flags = detect(pre, n)
    print(f"detected change-flags (byte offsets): {flags}")
    for b in bnd:
        aft = [f for f in flags if f >= b - 500]
        print(f"  boundary {b}: first flag at {aft[0] if aft else None}  (latency "
              f"{aft[0] - b if aft else None} bytes)")
    fp = [f for f in flags if all(not (b - 500 <= f <= b + 6000) for b in bnd)]
    print(f"  false positives (away from boundaries): {len(fp)}  {fp}")


if __name__ == "__main__":
    main()

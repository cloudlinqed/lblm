#!/usr/bin/env python3
"""
Push the compressor: ONLINE LOGISTIC CONTEXT MIXING -- the bit-native core's predictor scaled with a
tiny online neural mixer (the lpaq/PAQ idea, and exactly the "neural-network mapping" of the framing,
in bit-native form): several byte-aware context models each vote a probability for the next bit; a
single online-trained logistic unit mixes them in the stretch/squash (logit) domain; the mixed P
codes the next bit; weights and counts update online from the realised bit.

Adaptive / online over the whole stream (like gzip on the whole file). Metric = bits/bit (= cross
entropy = compression; raw = 1.0000, lower is better). Single-threaded, stdlib only, causal (only
bits strictly before position i are ever used to predict bit i).

Usage: python mix.py [path] [byte_cap]
"""
import sys, math, gzip

ORDERS = [0, 1, 2, 3, 4]                                   # byte-aware context orders to mix


def load_bits(path, cap):
    raw = open(path, "rb").read()
    if cap:
        raw = raw[:cap]
    bits = bytearray()
    for byte in raw:
        for j in range(7, -1, -1):
            bits.append((byte >> j) & 1)
    return raw, bits


def ctx(bits, i, B):                                       # order-B byte-aware context KNOWN before bit i
    phase = i & 7; bstart = i - phase
    return (phase, bytes(bits[bstart:i]), bytes(bits[max(0, bstart - B * 8):bstart]))


def stretch(p):
    p = min(1 - 1e-6, max(1e-6, p)); return math.log(p / (1 - p))


def squash(t):
    if t > 30:
        return 1 - 1e-6
    if t < -30:
        return 1e-6
    return 1 / (1 + math.exp(-t))


def run(bits, lr=0.02, delta=0.2):
    tables = [dict() for _ in ORDERS]
    w = [0.0] * len(ORDERS)
    n = len(bits); split = int(n * 0.8)
    tot = 0.0; tail = 0.0; tailn = 0
    for i in range(n):
        sts = []; cells = []
        for k, B in enumerate(ORDERS):
            key = ctx(bits, i, B); c = tables[k].get(key)
            if c is None:
                c = [0, 0]; tables[k][key] = c
            p = (c[1] + delta) / (c[0] + c[1] + 2 * delta)
            sts.append(stretch(p)); cells.append(c)
        P = squash(sum(w[k] * sts[k] for k in range(len(ORDERS))))
        y = bits[i]
        cost = -math.log2(P if y == 1 else 1 - P)
        tot += cost
        if i >= split:
            tail += cost; tailn += 1
        err = y - P
        for k in range(len(ORDERS)):
            w[k] += lr * err * sts[k]
            cells[k][y] += 1
    return tot / n, (tail / tailn if tailn else 0.0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.txt"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300000
    raw, bits = load_bits(path, cap)
    g = len(gzip.compress(raw, 9))
    whole, tail = run(bits)
    print(f"corpus={path}  bytes={len(raw)}  bits={len(bits)}")
    print(f"  logistic-mixing  whole-stream = {whole:.4f}   last-20% = {tail:.4f}  bits/bit")
    print(f"  gzip (whole file)             = {g / len(raw):.4f}  bits/bit")


if __name__ == "__main__":
    main()

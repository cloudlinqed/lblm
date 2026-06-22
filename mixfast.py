#!/usr/bin/env python3
"""
mixfast.py -- mix.py with LOSSLESS integer context keys (PyPy/Rust-friendly).

Identical model to mix.py (online logistic context mixing, byte-aware orders 0-4), but the dict key
for each order is a single integer instead of (phase, bytes(partial), bytes(prev_bytes)):

    key = ((1 << runlen) | run) << 3 | phase          # leading-1 sentinel makes length unambiguous
    run = (prev_bytes_int << phase) | partial_int      # prev whole bytes ++ partial current byte
    runlen = 8*L + phase,  L = min(B, bytes_seen)

This is a bijection with mix.py's tuple key (same equivalence classes) -> bit-for-bit identical
output -- but with NO per-bit bytes allocation and no tuple hashing, which is what the JIT (and later
a Rust/C++ open-addressing hash table of fixed-width keys) wants. State is carried in a rolling
integer `htail` (last MAXB bytes) + `cur` (partial byte), updated after each bit (causal).

Usage: python mixfast.py [path] [byte_cap]      (or run under pypy3 for the speedup)
"""
import sys, math, gzip
ORDERS = [0, 1, 2, 3, 4]


def load_bits(path, cap):
    raw = open(path, "rb").read()
    if cap:
        raw = raw[:cap]
    bits = bytearray()
    for byte in raw:
        for j in range(7, -1, -1):
            bits.append((byte >> j) & 1)
    return raw, bits


def stretch(p):
    p = min(1 - 1e-6, max(1e-6, p)); return math.log(p / (1 - p))


def squash(t):
    if t > 30:
        return 1 - 1e-6
    if t < -30:
        return 1e-6
    return 1.0 / (1.0 + math.exp(-t))


def run(bits, lr=0.02, delta=0.2):
    NM = len(ORDERS); MAXB = max(ORDERS)
    MASK = [(1 << (8 * L)) - 1 for L in range(MAXB + 1)]
    tables = [dict() for _ in ORDERS]; w = [0.0] * NM
    n = len(bits); split = int(n * 0.8); tot = 0.0; tail = 0.0; tailn = 0
    cur = 0; phase = 0; htail = 0; byte_pos = 0
    sts = [0.0] * NM; cells = [None] * NM
    for i in range(n):
        for k in range(NM):
            B = ORDERS[k]
            L = B if byte_pos >= B else byte_pos
            run_ = ((htail & MASK[L]) << phase) | cur
            key = (((1 << (8 * L + phase)) | run_) << 3) | phase
            c = tables[k].get(key)
            if c is None:
                c = [0, 0]; tables[k][key] = c
            cells[k] = c
            sts[k] = stretch((c[1] + delta) / (c[0] + c[1] + 2 * delta))
        P = squash(sum(w[k] * sts[k] for k in range(NM)))
        y = bits[i]
        cost = -math.log2(P if y == 1 else 1 - P)
        tot += cost
        if i >= split:
            tail += cost; tailn += 1
        err = y - P
        for k in range(NM):
            w[k] += lr * err * sts[k]; cells[k][y] += 1
        cur = (cur << 1) | y; phase += 1
        if phase == 8:
            htail = ((htail << 8) | cur) & MASK[MAXB]; cur = 0; phase = 0; byte_pos += 1
    return tot / n, (tail / tailn if tailn else 0.0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.txt"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300000
    raw, bits = load_bits(path, cap)
    g = len(gzip.compress(raw, 9))
    whole, tail = run(bits)
    print(f"corpus={path}  bytes={len(raw)}  bits={len(bits)}")
    print(f"  mixfast  whole-stream = {whole:.6f}   last-20% = {tail:.6f}  bits/bit")
    print(f"  gzip (whole file)     = {g / len(raw):.6f}  bits/bit")


if __name__ == "__main__":
    main()

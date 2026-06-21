#!/usr/bin/env python3
"""
Scale on REAL data: the bit-native core as a next-bit predictor on a real text corpus (bytes -> bits).

The honest metric is bits-per-bit on HELD-OUT text = cross-entropy = compression (raw = 1.0000,
lower is better). This is the bit-native analogue of an LLM's perplexity. Externally referenced
against gzip. The model is the exact-count form of the content-addressable machine: an address
(a context representation) maps to the empirical next-bit distribution (KT/Laplace smoothed). At
scale, exact counts replace the Hamming-kernel/SOM used on the tiny benches.

It carries the project's through-line to real data: the right *computed* representation is the lever.
We compare a raw bit-window (a flat bit n-gram) against a byte-aware representation (phase + current
partial byte + previous whole bytes) of comparable size, plus a backoff "core" that mixes byte orders.

Usage: python scale.py [path] [byte_cap]
"""
import sys, math, gzip, os


def load(path, cap):
    raw = open(path, "rb").read()
    if cap:
        raw = raw[:cap]
    bits = bytearray()
    for byte in raw:
        for j in range(7, -1, -1):
            bits.append((byte >> j) & 1)
    return raw, bits


def addr(bits, i, B):                                  # byte-aware: phase + current partial byte + B prev bytes
    phase = i & 7; bstart = i - phase
    return (phase, bytes(bits[bstart:i]), bytes(bits[max(0, bstart - B * 8):bstart]))


def bitwin(bits, i, k):                                # raw bit n-gram (byte-unaligned)
    return bytes(bits[max(0, i - k):i])


def build(bits, split, keyfn):
    t = {}
    for i in range(split):
        c = t.get(keyfn(bits, i))
        if c is None:
            c = [0, 0, 0]; t[keyfn(bits, i)] = c
        c[bits[i]] += 1; c[2] += 1
    return t


def eval_single(bits, split, keyfn, alpha=0.5):
    t = build(bits, split, keyfn); tot = 0.0; n = len(bits) - split
    for i in range(split, len(bits)):
        c = t.get(keyfn(bits, i))
        p1 = 0.5 if c is None else (c[1] + alpha) / (c[2] + 2 * alpha)
        p = p1 if bits[i] == 1 else 1 - p1
        tot += -math.log2(max(p, 1e-12))
    return tot / n


def eval_backoff(bits, split, orders, alpha=0.5, thresh=2):
    tabs = {B: build(bits, split, lambda b, i, B=B: addr(b, i, B)) for B in orders}
    tot = 0.0; n = len(bits) - split
    for i in range(split, len(bits)):
        p1 = 0.5
        for B in orders:                               # highest order first, back off until seen enough
            c = tabs[B].get(addr(bits, i, B))
            if c is not None and c[2] >= thresh:
                p1 = (c[1] + alpha) / (c[2] + 2 * alpha); break
        p = p1 if bits[i] == 1 else 1 - p1
        tot += -math.log2(max(p, 1e-12))
    return tot / n


def scaling(path):
    print("DATA SCALING (CORE backoff model) on real text -- does more real data -> better compression?")
    print(" bytes  | core bits/bit | gzip bits/bit")
    for cap in [100000, 300000, 600000, 772000]:
        raw, bits = load(path, cap); split = int(len(bits) * 0.8)
        g = len(gzip.compress(raw, 9)); bo = eval_backoff(bits, split, [3, 2, 1, 0])
        print(f" {len(raw):6} | {bo:.4f}        | {g / len(raw):.4f}")


def main():
    args = sys.argv[1:]
    if args and args[0] == "scaling":
        scaling(args[1] if len(args) > 1 else "data/corpus.txt"); return
    path = args[0] if args else "data/corpus.txt"
    cap = int(args[1]) if len(args) > 1 else 300000
    raw, bits = load(path, cap)
    split = int(len(bits) * 0.8)
    g = len(gzip.compress(raw, 9))
    print(f"corpus={path}  bytes={len(raw)}  bits={len(bits)}  train/test = 80/20")
    print(f"gzip(whole file) = {g / len(raw):.4f} bits/bit  (to {100 * g / len(raw):.1f}% of original) [external reference]\n")
    print("representation                        | held-out bits/bit  (raw = 1.0000, lower = better)")
    rows = [("order-0 within-byte (phase, cur)", lambda b, i: addr(b, i, 0)),
            ("bit-window k=8  (byte-unaligned)",  lambda b, i: bitwin(b, i, 8)),
            ("bit-window k=16 (byte-unaligned)",  lambda b, i: bitwin(b, i, 16)),
            ("byte-aware B=1  (computed phase)",  lambda b, i: addr(b, i, 1)),
            ("byte-aware B=2  (computed phase)",  lambda b, i: addr(b, i, 2)),
            ("byte-aware B=3  (computed phase)",  lambda b, i: addr(b, i, 3))]
    for name, fn in rows:
        print(f" {name:37}| {eval_single(bits, split, fn):.4f}")
    bo = eval_backoff(bits, split, [3, 2, 1, 0])
    print(f"\nCORE backoff model (byte orders 3->2->1->0): {bo:.4f} bits/bit  (compresses to {100 * bo:.1f}% of raw)")
    print(f"external ref  gzip: {g / len(raw):.4f} bits/bit")


if __name__ == "__main__":
    main()

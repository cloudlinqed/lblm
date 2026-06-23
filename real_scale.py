#!/usr/bin/env python3
"""
Path B x Path A -- KEEP SCALING. A fast integer-key engine + a scaling sweep on real data.

real_mix.py proved the unified engine (induce the representation -> mix to predict) beats gzip on real
text and DNA, but its bytes()-slicing keys are too slow for serious scale. This is the same engine with
the project's proven speed levers -- INTEGER ROLLING KEYS (no per-bit slicing; ~mixfast.py) and a
STRETCH lookup table (the lpaq/PAQ trick) -- so we can push to multi-MB and show the curve keep dropping
(the project's through-line: on homogeneous data, more data -> lower bits/bit, monotonically).

Engine: an online logistic mixer over context models keyed by a rolling history integer:
  * unit-order contexts B=0..maxB at the INDUCED period p (the unit found from data, not assumed);
  * a running popcount counter (Path B aggregation), via int.bit_count();
  * (DNA) a reverse-complement canonical-k-mer context (strand symmetry / sec-51 palindrome), via a
    precomputed revcomp table.
Causal; stdlib only; bit-identical-in-spirit to real_mix (KT counts, logit mixing, online weights).

Usage: python real_scale.py text|dna [cap1,cap2,...]
"""
import sys, math, gzip, os
from scale import load
from real_test import load_dna
from real_mix import discover_period

# stretch lookup table (quantised logit) -- avoids a math.log per context per bit
STN = 4096
ST = [0.0] * (STN + 1)
for _i in range(STN + 1):
    _p = min(1 - 1e-6, max(1e-6, _i / STN)); ST[_i] = math.log(_p / (1 - _p))


def build_rc_table(kb):
    """revcomp of a 2*kb-bit window (kb bases): reverse base order, complement (xor 3) each base."""
    size = 1 << (2 * kb); rc = [0] * size
    for x in range(size):
        y = 0
        for j in range(kb):
            b = (x >> (2 * j)) & 3
            y |= (b ^ 3) << (2 * (kb - 1 - j))
        rc[x] = y
    return rc


def _stretchf(pr):
    pr = min(1 - 1e-6, max(1e-6, pr)); return math.log(pr / (1 - pr))


class APM:
    """Adaptive probability map (SSE) -- refine the mixer's P through a context-indexed, online,
    piecewise-linear curve over the stretched-P domain. The lpaq/PAQ recalibration stage."""
    __slots__ = ('t', 'ci', 'cj', 'cw')

    def __init__(self, nc):
        self.t = [[squashf((j - 16) / 2.0) for j in range(33)] for _ in range(nc)]

    def refine(self, pr, cxt):
        v = _stretchf(pr) * 2.0 + 16.0
        if v < 0.0:
            v = 0.0
        elif v > 31.999:
            v = 31.999
        i = int(v); w = v - i
        self.ci = cxt; self.cj = i; self.cw = w
        row = self.t[cxt]
        return row[i] * (1 - w) + row[i + 1] * w

    def learn(self, y, rate=0.04):
        row = self.t[self.ci]; j = self.cj; w = self.cw
        row[j] += rate * (1 - w) * (y - row[j])
        row[j + 1] += rate * w * (y - row[j + 1])


def squashf(t):
    if t > 30:
        return 1 - 1e-6
    if t < -30:
        return 1e-6
    return 1.0 / (1.0 + math.exp(-t))


def run_fast(bits, p, maxB=5, wcount=16, dna=False, rc_kb=4, lr=0.02, delta=0.2, sse=False):
    n = len(bits)
    unit_K = maxB + 1
    K = unit_K + 1 + (1 if dna else 0)
    ci = unit_K; ri = unit_K + 1
    tables = [dict() for _ in range(K)]
    w = [0.0] * K
    masks = [[(1 << (ph + B * p)) - 1 for B in range(unit_K)] for ph in range(p)]
    sent  = [[1 << (ph + B * p) for B in range(unit_K)] for ph in range(p)]
    Wbits = max(maxB * p + p, wcount, (2 * rc_kb if dna else 0)) + 1
    histmask = (1 << Wbits) - 1
    countmask = (1 << wcount) - 1
    rcmask = (1 << (2 * rc_kb)) - 1
    rctab = build_rc_table(rc_kb) if dna else None
    hist = 0; split = int(n * 0.8); tot = 0.0; tail = 0.0; tailn = 0; td = 2 * delta
    sts = [0.0] * K; cells = [None] * K
    STl = ST; exp = math.exp; log2 = math.log2
    apm = APM(256) if sse else None
    for i in range(n):
        ph = i % p
        mk = masks[ph]; sn = sent[ph]
        s = 0.0
        for B in range(unit_K):
            key = (hist & mk[B]) | sn[B]
            t = tables[B]; c = t.get(key)
            if c is None:
                c = [0, 0]; t[key] = c
            p1 = (c[1] + delta) / (c[0] + c[1] + td)
            st = STl[int(p1 * STN)]; sts[B] = st; cells[B] = c; s += w[B] * st
        pc = (hist & countmask).bit_count()                       # Path B running counter
        key = ph * 64 + pc
        t = tables[ci]; c = t.get(key)
        if c is None:
            c = [0, 0]; t[key] = c
        p1 = (c[1] + delta) / (c[0] + c[1] + td)
        st = STl[int(p1 * STN)]; sts[ci] = st; cells[ci] = c; s += w[ci] * st
        if dna:
            x = hist & rcmask; rx = rctab[x]; canon = x if x <= rx else rx
            key = ph * 1000003 + canon
            t = tables[ri]; c = t.get(key)
            if c is None:
                c = [0, 0]; t[key] = c
            p1 = (c[1] + delta) / (c[0] + c[1] + td)
            st = STl[int(p1 * STN)]; sts[ri] = st; cells[ri] = c; s += w[ri] * st
        # squash + (optional) SSE recalibration + cost
        if s > 30:
            P = 1 - 1e-6
        elif s < -30:
            P = 1e-6
        else:
            P = 1.0 / (1.0 + exp(-s))
        if apm is not None:
            pr = (P + 3.0 * apm.refine(P, hist & 0xff)) * 0.25
            if pr < 1e-6:
                pr = 1e-6
            elif pr > 1 - 1e-6:
                pr = 1 - 1e-6
        else:
            pr = P
        y = bits[i]
        cost = -log2(pr if y == 1 else 1 - pr)
        tot += cost
        if i >= split:
            tail += cost; tailn += 1
        if apm is not None:
            apm.learn(y)
        err = y - P
        for k in range(K):
            w[k] += lr * err * sts[k]
            cells[k][y] += 1
        hist = ((hist << 1) | y) & histmask
    return tot / n, (tail / tailn if tailn else 0.0)


def sweep(kind, caps):
    print(f"{'=' * 78}\nSCALING SWEEP -- {kind}   (engine: induced unit contexts + counter"
          f"{' + revcomp' if kind == 'dna' else ''} + SSE, integer keys)\n{'=' * 78}")
    print(f"{'size':>10} | period | bits/bit | last-20% | gzip   | vs gzip")
    for cap in caps:
        if kind == 'text':
            path = 'data/corpus_big.txt' if os.path.exists('data/corpus_big.txt') else 'data/corpus.txt'
            raw, bits = load(path, cap)
            raw_for_gz = raw; unit_label = f"{cap}B"
        else:
            bits = load_dna('data/ecoli.fasta', cap)
            packed = bytearray()
            for k in range(0, len(bits) - 7, 8):
                byte = 0
                for j in range(8):
                    byte = (byte << 1) | bits[k + j]
                packed.append(byte)
            raw_for_gz = packed; unit_label = f"{cap}bp"
        gz = len(gzip.compress(bytes(raw_for_gz), 9)) * 8 / len(bits)
        p, _ = discover_period(bits)
        maxB = 8 if kind == 'text' else 6
        whole, tailv = run_fast(bits, p, maxB=maxB, dna=(kind == 'dna'), sse=True)
        print(f"{unit_label:>10} | p={p:<4} | {whole:.4f}   | {tailv:.4f}   | {gz:.4f} | "
              f"{'BEATS' if whole < gz else 'loses'} ({gz - whole:+.4f})")


def main():
    kind = sys.argv[1] if len(sys.argv) > 1 else 'text'
    if len(sys.argv) > 2:
        caps = [int(x) for x in sys.argv[2].split(',')]
    elif kind == 'text':
        caps = [500000, 1000000, 2000000, 4000000]
    else:
        caps = [500000, 1000000, 2000000, 4000000]
    sweep(kind, caps)


if __name__ == "__main__":
    main()

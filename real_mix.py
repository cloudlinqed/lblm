#!/usr/bin/env python3
"""
Path B x Path A -- the REAL head-to-head: induce the representation, mix to predict, beat gzip.

real_test.py showed Path B's induction discovers real structure on real data (the byte period on
text, codon-scale period on DNA) and its primitives lower held-out bits/bit -- but a single sparse
conjunction-table trailed gzip on text. This is the fix and the real unification (the framing's M6:
compression = prediction = program-finding):

  Path B INDUCES *what to compute* (the predictive unit/period + aggregation/detector contexts),
  Path A MIXES those contexts to PREDICT (online logistic context mixing, the lpaq/PAQ idea).

One engine, online over the whole real stream (the standard, like gzip on the whole file). Context sets
fed to the SAME mixer:

  BYTE      (Path A, hardcoded 8-bit unit)  : byte-aware orders B=0..4   -- the existing strong model.
  INDUCED   (Path B)                        : orders at the DISCOVERED period p (found by a quick period
                                              scan -- the unit comes from data, not assumed) + running-
                                              count and pattern-detector contexts.
  INDUCED+RC (Path B, DNA only)             : + reverse-complement contexts. In 2-bit DNA (A=00,C=01,
                                              G=10,T=11) base-complement is a BIT FLIP (A<->T 00<->11,
                                              C<->G 01<->10), so reverse-complement palindromes (real
                                              restriction sites; the sec-51 palindrome structure) are a
                                              clean bit-level feature, and strand symmetry (Chargaff)
                                              lets a canonical-k-mer context pool both strands.

Metric = bits/bit (cross-entropy = compression; raw 1.0000, lower better), whole-stream and last-20%.
Causal (only bits before i predict bit i); stdlib only.
Usage: python real_mix.py [text_cap] [dna_cap] [text_path]
"""
import sys, math, gzip, os
from scale import load
from real_test import load_dna


def stretch(p):
    p = min(1 - 1e-6, max(1e-6, p)); return math.log(p / (1 - p))


def squash(t):
    if t > 30:
        return 1 - 1e-6
    if t < -30:
        return 1e-6
    return 1 / (1 + math.exp(-t))


# ---------------------------------------------------------------------------
# context models (each maps (bits, i) -> a hashable key; its own count table)
# ---------------------------------------------------------------------------
def make_unit_ctx(p, B):
    """order-B context aligned to period p: phase + partial symbol + B previous whole symbols."""
    def f(bits, i):
        ph = i % p; us = i - ph
        return (ph, bytes(bits[us:i]), bytes(bits[max(0, us - B * p):us]))
    return f


def make_agg_ctx(p, w, cap):
    """Path B aggregation: phase + running popcount bucket over the last w bits."""
    def f(bits, i):
        return (i % p, min(sum(bits[max(0, i - w):i]), cap))
    return f


def make_det_ctx(p, pats, w):
    """Path B detectors: phase + which of `pats` occur in the last w bits."""
    def f(bits, i):
        seg = bits[max(0, i - w):i]
        flags = tuple(int(any(tuple(seg[t:t + len(pt)]) == pt
                              for t in range(len(seg) - len(pt) + 1))) for pt in pats)
        return (i % p,) + flags
    return f


# --- reverse-complement contexts (DNA, base = 2 bits; complement = flip both bits) ---
def _prev_bases(bits, i, k):
    """The up-to-k complete bases (as (hi,lo) tuples, in order) strictly before the current base."""
    bstart = (i >> 1) * 2
    lo = max(0, bstart - 2 * k)
    flat = bits[lo:bstart]
    return [(flat[j], flat[j + 1]) for j in range(0, len(flat) - 1, 2)], bstart


def make_rc_ctx(k):
    """Strand-symmetric (Chargaff) context: canonical k-mer = min(prev-k-bases, its reverse-complement).
    Reverse-complement = reverse base order AND complement each base (flip both bits)."""
    def f(bits, i):
        basesL, bstart = _prev_bases(bits, i, k)
        fwd = tuple(b for base in basesL for b in base)
        rc = tuple(b for base in reversed(basesL) for b in (1 - base[0], 1 - base[1]))
        return (i & 1, bytes(bits[bstart:i]), min(fwd, rc))
    return f


def make_rcpal_det(kbases):
    """Detector: are the last `kbases` complete bases a reverse-complement palindrome (== own revcomp)?
    Restriction sites (EcoRI GAATTC, ...) are exactly this -- the sec-51 palindrome structure, on DNA."""
    def f(bits, i):
        basesL, _ = _prev_bases(bits, i, kbases)
        if len(basesL) < kbases:
            return (i & 1, 2)
        rc = [(1 - b[0], 1 - b[1]) for b in reversed(basesL)]
        return (i & 1, int(basesL == rc))
    return f


# ---------------------------------------------------------------------------
# induction: discover the predictive period p from a prefix (Path B finds the unit)
# ---------------------------------------------------------------------------
def period_score(bits, split, p):
    t = {}
    for i in range(split):
        ph = i % p; key = (ph, bytes(bits[i - ph:i]))
        c = t.get(key)
        if c is None:
            c = [0, 0]; t[key] = c
        c[bits[i]] += 1
    ones = sum(bits[:split]); g1 = (ones + 0.5) / (split + 1)
    tot = 0.0; n = len(bits) - split
    for i in range(split, len(bits)):
        ph = i % p; c = t.get((ph, bytes(bits[i - ph:i])))
        p1 = g1 if c is None else (c[1] + 0.5) / (c[0] + c[1] + 1)
        pp = p1 if bits[i] == 1 else 1 - p1
        tot += -math.log2(max(pp, 1e-12))
    return tot / n


def discover_period(bits, cap=80000):
    seg = bits[:min(cap, len(bits))]; sp = int(len(seg) * 0.8)
    best_p, best, scan = 8, 9.0, []
    for p in range(2, 13):
        s = period_score(seg, sp, p); scan.append((p, s))
        if s < best:
            best, best_p = s, p
    return best_p, scan


# ---------------------------------------------------------------------------
# the SHARED online logistic mixer (Path A) over a list of context models
# ---------------------------------------------------------------------------
def run(bits, ctxfns, lr=0.02, delta=0.2):
    K = len(ctxfns); tables = [dict() for _ in range(K)]; w = [0.0] * K
    n = len(bits); split = int(n * 0.8)
    tot = 0.0; tail = 0.0; tailn = 0
    for i in range(n):
        sts = []; cells = []
        for k in range(K):
            key = ctxfns[k](bits, i); c = tables[k].get(key)
            if c is None:
                c = [0, 0]; tables[k][key] = c
            p = (c[1] + delta) / (c[0] + c[1] + 2 * delta)
            sts.append(stretch(p)); cells.append(c)
        P = squash(sum(w[k] * sts[k] for k in range(K)))
        y = bits[i]
        cost = -math.log2(P if y == 1 else 1 - P)
        tot += cost
        if i >= split:
            tail += cost; tailn += 1
        err = y - P
        for k in range(K):
            w[k] += lr * err * sts[k]
            cells[k][y] += 1
    return tot / n, (tail / tailn if tailn else 0.0)


def byte_set():
    return [make_unit_ctx(8, B) for B in (0, 1, 2, 3, 4)]


def induced_set(p):
    return ([make_unit_ctx(p, B) for B in (0, 1, 2, 3, 4)] +
            [make_agg_ctx(p, 8, 4), make_agg_ctx(p, 16, 6)] +
            [make_det_ctx(p, [(1, 1), (0, 0), (1, 0, 1), (0, 1, 0)], 8)])


def dna_rc_set(p):
    return (induced_set(p) +
            [make_rc_ctx(k) for k in (3, 4, 6)] +
            [make_rcpal_det(k) for k in (3, 4)])


def run_corpus(name, bits, raw_bytes, build_sets):
    gz = len(gzip.compress(bytes(raw_bytes), 9)) * 8 / len(bits)
    p, scan = discover_period(bits)
    print(f"\n{'=' * 84}\n{name}: bits={len(bits)}")
    print("  INDUCTION period scan: " + "  ".join(f"p{q}:{s:.3f}" for q, s in scan))
    print(f"  -> discovered predictive unit period p = {p}")
    print(f"  gzip (same stream): {gz:.4f} bits/bit  [external reference]")
    res = {}
    for label, fns in build_sets(p):
        w, t = run(bits, fns)
        res[label] = w
        flag = "  BEATS gzip" if w < gz - 1e-6 else ""
        print(f"  {label:24} whole={w:.4f}  last-20%={t:.4f}{flag}")
    return name, p, gz, res


def main():
    text_cap  = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    dna_cap   = int(sys.argv[2]) if len(sys.argv) > 2 else 300000
    text_path = sys.argv[3] if len(sys.argv) > 3 else (
        'data/corpus_big.txt' if os.path.exists('data/corpus_big.txt') else 'data/corpus.txt')
    print("PATH B x PATH A -- induce the representation, mix to predict, beat gzip (REAL data)")
    print(f"text={text_path} cap={text_cap}  dna=data/ecoli.fasta cap={dna_cap} bases")

    out = []
    try:
        raw, tbits = load(text_path, text_cap)
        out.append(run_corpus(f'English text ({os.path.basename(text_path)})', tbits, raw,
                              lambda p: [('BYTE (8-bit)', byte_set()), ('INDUCED (p)', induced_set(p))]))
    except FileNotFoundError:
        print("  (text corpus missing -- skipping text)")
    try:
        dbits = load_dna('data/ecoli.fasta', dna_cap)
        packed = bytearray()
        for k in range(0, len(dbits) - 7, 8):
            byte = 0
            for j in range(8):
                byte = (byte << 1) | dbits[k + j]
            packed.append(byte)
        out.append(run_corpus('E. coli DNA (ecoli.fasta, 2bit)', dbits, packed,
                              lambda p: [('BYTE (8-bit)', byte_set()), ('INDUCED (p)', induced_set(p)),
                                         ('INDUCED+RC (revcomp)', dna_rc_set(p))]))
    except FileNotFoundError:
        print("  (data/ecoli.fasta missing -- skipping DNA)")

    print(f"\n{'=' * 84}\nSUMMARY -- one bit-native engine (induce + mix) vs gzip")
    for name, p, gz, res in out:
        best = min(res.values())
        print(f"\n{name}  (period p={p}, gzip={gz:.4f}):")
        for label, w in res.items():
            print(f"    {label:24} {w:.4f}  {'<- best, beats gzip' if abs(w-best)<1e-9 and w<gz else ('beats gzip' if w<gz else '')}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
DNA-tuned context-mixing predictor: base/codon-aware (fixes blmrs-strong's 8-bit misalignment).

blmrs-strong is hardwired to 8-bit bytes; DNA's units are 2-bit bases and period-3 codons. This model
operates on 2-bit bases, conditions context on previous BASES, adds an explicit CODON phase (reading
frame = base index mod 3), and runs a BASE-granular match model for long repeats. Online logistic
context mixing, like mixns but base-aware. Metric: bits/base (= sum of -log2 P over the 2 bits / base).
Floor (raw) = 2.000. Run under pypy3 for speed.

Usage: dna.py <packed .2bit file>
"""
import sys, math

ORDERS = [1, 2, 3, 4, 6, 8, 12, 16]   # context orders in BASES
NM = len(ORDERS)
I_MT = NM
I_RC = NM + 1
NIN = NM + 2                               # + forward match + reverse-complement match
NW = NIN + 1                               # + bias
MAXB = max(ORDERS)


def load_bases(path):
    raw = open(path, "rb").read()
    out = bytearray()
    for b in raw:
        out.append((b >> 6) & 3); out.append((b >> 4) & 3); out.append((b >> 2) & 3); out.append(b & 3)
    return out


def stretch(p):
    p = min(1 - 1e-6, max(1e-6, p)); return math.log(p / (1 - p))


def squash(t):
    if t > 30:
        return 1 - 1e-6
    if t < -30:
        return 1e-6
    return 1.0 / (1.0 + math.exp(-t))


class Match:
    def __init__(self, hb=22, minlen=20):
        self.mask = (1 << hb) - 1; self.tab = [0] * (1 << hb); self.minlen = minlen
        self.ptr = 0; self.len = 0; self.h = 0

    def predict(self, bases, bp, bit_in_base, partial):
        if self.len == 0 or self.ptr >= bp:
            return 0.0
        pb = bases[self.ptr]
        if bit_in_base == 0:
            bit = (pb >> 1) & 1
        else:
            if ((pb >> 1) & 1) != partial:
                return 0.0
            bit = pb & 1
        st = 1.5 + 0.30 * min(self.len, 32)
        return st if bit == 1 else -st

    def update_after_base(self, bases, bp):
        if self.len > 0 and self.ptr < bp:
            if bases[self.ptr] == bases[bp]:
                self.ptr += 1; self.len = min(self.len + 1, 65535)
            else:
                self.len = 0; self.ptr = 0
        self.h = ((self.h << 2) | bases[bp]) & ((1 << 48) - 1)   # last 24 bases
        if bp + 1 >= self.minlen:
            hk = (self.h * 2654435761) & self.mask
            prev = self.tab[hk]; self.tab[hk] = bp + 1
            if self.len == 0 and prev >= self.minlen and prev <= bp:
                ok = True                                        # VERIFY (kill hash collisions)
                for j in range(self.minlen):
                    if bases[prev - 1 - j] != bases[bp - j]:
                        ok = False; break
                if ok:
                    self.ptr = prev; self.len = self.minlen


class RCMatch:
    """Inverted-repeat / reverse-complement match: detect when the current context is the RC of an
    earlier forward k-mer (shares the forward k-mer table via a rolling RC hash) and predict the
    complement, reading backward. Verified to kill hash collisions."""

    def __init__(self, K=24):
        self.K = K; self.rcmask = (1 << (2 * K)) - 1
        self.rc_h = 0; self.ptr = -1; self.len = 0

    def predict(self, bases, bp, bit_in_base, partial):
        if self.len == 0 or self.ptr < 0:
            return 0.0
        pb = 3 - bases[self.ptr]                          # complement of the matched base
        if bit_in_base == 0:
            bit = (pb >> 1) & 1
        else:
            if ((pb >> 1) & 1) != partial:
                return 0.0
            bit = pb & 1
        st = 1.5 + 0.30 * min(self.len, 32)
        return st if bit == 1 else -st

    def update_after_base(self, bases, bp, fwd_tab, fwd_mask):
        if self.len > 0 and self.ptr >= 0:
            if (3 - bases[self.ptr]) == bases[bp]:
                self.ptr -= 1; self.len = min(self.len + 1, 65535)
                if self.ptr < 0:
                    self.len = 0
            else:
                self.len = 0; self.ptr = -1
        self.rc_h = ((self.rc_h >> 2) | ((3 - bases[bp]) << (2 * (self.K - 1)))) & self.rcmask
        if self.len == 0 and bp + 1 >= self.K:
            p = fwd_tab[(self.rc_h * 2654435761) & fwd_mask]
            if p >= self.K + 1 and p <= bp:               # earlier forward k-mer = bases[p-K..p-1]
                ok = True
                for k in range(self.K):
                    if bases[p - self.K + k] != 3 - bases[bp - k]:
                        ok = False; break
                if ok:
                    self.ptr = p - self.K - 1             # continue backward, complemented
                    if self.ptr >= 0:
                        self.len = self.K


def run(bases, alr=0.01, delta=0.5, rms_decay=0.999, rms_eps=1e-3):
    n = len(bases)
    masks = [(1 << (2 * B)) - 1 for B in range(MAXB + 1)]
    htmask = (1 << (2 * MAXB)) - 1
    tables = [dict() for _ in ORDERS]
    w = [0.0] * NW; g = [0.0] * NW                     # RMSProp per-weight accumulator
    match = Match()
    rcmatch = RCMatch()
    bhist = 0
    sts = [0.0] * NW; cells = [None] * NM
    tot = 0.0; log2 = math.log(2.0); sqrt = math.sqrt
    for bp in range(n):
        base = bases[bp]; codon = bp % 3
        b0 = (base >> 1) & 1; b1 = base & 1
        for bit_in_base in (0, 1):
            ybit = b0 if bit_in_base == 0 else b1
            partial = 0 if bit_in_base == 0 else b0
            for k in range(NM):
                B = ORDERS[k]
                key = (codon, bit_in_base, partial, B, bhist & masks[B])
                c = tables[k].get(key)
                if c is None:
                    c = [0, 0]; tables[k][key] = c
                cells[k] = c
                sts[k] = stretch((c[1] + delta) / (c[0] + c[1] + 2 * delta))
            sts[I_MT] = match.predict(bases, bp, bit_in_base, partial)
            sts[I_RC] = rcmatch.predict(bases, bp, bit_in_base, partial)
            sts[NIN] = 1.0
            P = squash(sum(w[k] * sts[k] for k in range(NW)))
            tot += -(math.log(P if ybit == 1 else 1 - P) / log2)
            err = ybit - P
            for k in range(NW):                          # RMSProp (stabilises the mix)
                grad = err * sts[k]
                g[k] = rms_decay * g[k] + (1 - rms_decay) * grad * grad
                w[k] += alr * grad / (sqrt(g[k]) + rms_eps)
            for k in range(NM):
                cells[k][ybit] += 1
        bhist = ((bhist << 2) | base) & htmask
        match.update_after_base(bases, bp)
        rcmatch.update_after_base(bases, bp, match.tab, match.mask)
    return tot / n   # bits per base


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/ecoli.2bit"
    bases = load_bases(path)
    bpb = run(bases)
    print(f"{path}: {len(bases)} bases   DNA-aware = {bpb:.4f} bits/base   (floor 2.000)")


if __name__ == "__main__":
    main()

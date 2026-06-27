#!/usr/bin/env python3
"""
prims.py -- item 3 of the no-copy intelligence track: PATH-B INDUCED PRIMITIVES as first-class,
individually-ablatable predictor channels. The §70 diagnosis: a learned compact state TIES the order
models because byte-value EMAs learn structure the orders already have. The fix is to give the predictor
STATEFUL COMPUTATION the orders FUNDAMENTALLY cannot do -- unbounded counters/detectors:

  paren  : bracket/brace nesting DEPTH        (a counter; orders can't count depth from an open-paren far back)
  quote  : inside-a-quote toggle
  digit  : consecutive-digit RUN length       (in the middle of a number -> predict more digits)
  word   : bytes since last space (word pos)
  line   : bytes since last newline (line pos)
  caps   : consecutive-UPPERCASE run          (acronyms / headers)

Each primitive is CAUSAL (updated from bytes already seen) and becomes a COUNT-MODEL channel keyed by
(primitive_bucket, phase, partial-byte) -> a stretched logit mixed exactly like the order experts, with
its OWN ablation. Gated by the genmem protocol: MATCH (copy) OFF, 13-byte-DECONTAMINATED held-out
bits/byte, channel ON/OFF, and a SCRAMBLED control (random buckets -> structureless) that must NOT help.
A counter is unbounded state a fixed-window n-gram cannot represent -> if these channels LOWER copy-off
decontaminated held-out loss past the order baseline, that gain is COMPUTATION, not copy. Run: python prims.py
"""
import sys, math, random, time
from genmem import clean_mask, stretch, squash

ORDERS = [0, 1, 2, 3, 4, 5, 6]
ALL_PRIMS = ["paren", "quote", "digit", "word", "line", "caps"]


def _lbkt(v):
    for i, e in enumerate((0, 2, 6, 14, 30, 62, 126)):
        if v <= e: return i
    return 7


class Model:
    def __init__(self, use_match=True, prims=None, order_cap=6, lr=0.004, scramble=False):
        self.use_match = use_match
        self.prims = list(prims) if prims else []
        self.lr, self.scramble = lr, scramble
        self.orders = [o for o in ORDERS if o <= order_cap]; self.NM = len(self.orders)
        self.NP = len(self.prims)
        self.NIN = self.NM + (1 if use_match else 0) + self.NP
        self.w = [0.0] * (self.NIN + 1); self.wg = [0.0] * (self.NIN + 1)
        self.tab = [dict() for _ in self.orders]
        self.ptab = {p: dict() for p in self.prims}
        self.pbase = self.NM + (1 if use_match else 0)
        self.MINLEN, self.GATE = 16, 18
        self.mtab = {}; self.hist = bytearray(); self.mptr = -1; self.mlen = 0
        self.htail = 0; self.cur = 0; self.phase = 0
        # primitive states (causal: reflect bytes seen so far)
        self.pdepth = self.quote = self.drun = self.wlen = self.lpos = self.crun = 0
        if scramble: self._sr = random.Random(1234)
        self._pb = self._prim_buckets()

    def _prim_buckets(self):
        vals = {"paren": min(self.pdepth, 7), "quote": self.quote, "digit": min(self.drun, 7),
                "word": min(self.wlen, 7), "line": _lbkt(self.lpos), "caps": min(self.crun, 7)}
        if self.scramble:
            return [(p, self._sr.randrange(8)) for p in self.prims]
        return [(p, vals[p]) for p in self.prims]

    def _octx(self, k):
        b = self.orders[k]; mask = (1 << (8 * b)) - 1 if b else 0
        return (self.phase, self.cur, (self.htail & mask) if b else 0)

    def predict(self):
        sts = [0.0] * (self.NIN + 1); i = 0
        for k in range(self.NM):
            c = self.tab[k].get(self._octx(k)); n0, n1 = (c[0], c[1]) if c else (0, 0)
            sts[i] = stretch((n1 + 0.2) / (n0 + n1 + 0.4)); i += 1
        if self.use_match:
            st = 0.0
            if self.mlen >= self.GATE and 0 <= self.mptr < len(self.hist):
                pb = self.hist[self.mptr]
                if self.phase == 0 or (pb >> (8 - self.phase)) == self.cur:
                    bit = (pb >> (7 - self.phase)) & 1; st = (1.6 + 0.35 * min(self.mlen, 28)) * (1 if bit else -1)
            sts[i] = st; i += 1
        for (name, bkt) in self._pb:
            key = (bkt, self.phase, self.cur); c = self.ptab[name].get(key)
            n0, n1 = (c[0], c[1]) if c else (0, 0)
            sts[i] = stretch((n1 + 0.2) / (n0 + n1 + 0.4)); i += 1
        sts[self.NIN] = 1.0
        d = sum(self.w[j] * sts[j] for j in range(self.NIN + 1))
        return squash(d), sts

    def step(self, y, learn):
        p, sts = self.predict()
        cost = -math.log2(p if y == 1 else 1 - p)
        if learn:
            err = y - p
            for j in range(self.NIN + 1):
                g = err * sts[j]; self.wg[j] = 0.999 * self.wg[j] + 0.001 * g * g
                self.w[j] += self.lr * g / (math.sqrt(self.wg[j]) + 1e-4)
            for k in range(self.NM):
                key = self._octx(k); c = self.tab[k].get(key)
                if c is None: c = [0, 0]; self.tab[k][key] = c
                c[y] += 1
            for (name, bkt) in self._pb:
                key = (bkt, self.phase, self.cur); c = self.ptab[name].get(key)
                if c is None: c = [0, 0]; self.ptab[name][key] = c
                c[y] += 1
        self.cur = (self.cur << 1) | y; self.phase += 1
        if self.phase == 8:
            b = self.cur & 0xFF; self.hist.append(b)
            if self.use_match: self._match_after(b)
            c = chr(b) if b < 128 else "\x00"
            if c in "([{": self.pdepth = min(self.pdepth + 1, 15)
            elif c in ")]}": self.pdepth = max(self.pdepth - 1, 0)
            if c == '"': self.quote ^= 1
            self.drun = self.drun + 1 if c.isdigit() else 0
            self.wlen = 0 if c in " \n\t" else min(self.wlen + 1, 31)
            self.lpos = 0 if c == "\n" else min(self.lpos + 1, 255)
            self.crun = self.crun + 1 if c.isupper() else 0
            self._pb = self._prim_buckets()
            self.htail = ((self.htail << 8) | b) & ((1 << 48) - 1); self.cur = 0; self.phase = 0
        return cost

    def _match_after(self, b):
        n = len(self.hist)
        if self.mlen > 0 and self.mptr < n - 1:
            if self.hist[self.mptr] == self.hist[n - 1]: self.mptr += 1; self.mlen = min(self.mlen + 1, 65535)
            else: self.mlen = 0; self.mptr = -1
        if n >= self.MINLEN:
            key = bytes(self.hist[n - self.MINLEN:n]); prev = self.mtab.get(key, -1); self.mtab[key] = n
            if self.mlen == 0 and 0 <= prev < n: self.mptr = prev; self.mlen = self.MINLEN

    def run(self, raw, learn, mask=None):
        tot = 0.0; nb = 0; bp = 0
        for b in raw:
            scored = (mask is None) or (bp < len(mask) and mask[bp])
            bc = 0.0
            for j in range(7, -1, -1):
                bc += self.step((b >> j) & 1, learn)
            if scored: tot += bc; nb += 1
            bp += 1
        return tot / max(1, nb)


def warm(train, test, mask, use_match, prims, scramble=False):
    m = Model(use_match=use_match, prims=prims, scramble=scramble)
    m.run(train, learn=True)
    return m.run(test, learn=True, mask=mask)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ft = open("data/wt103_train.txt", "rb").read()
    fe = open("data/wt103_test.txt", "rb").read()
    print("=" * 98)
    print("ITEM 3 -- Path-B induced primitives as ablatable channels; do counters/detectors beat the orders?")
    print("  (leak-free wt103, copy/match OFF, 13-byte-decontaminated held-out; <0 gap = BEATS order baseline)")
    print("=" * 98)
    t0 = time.time()

    for kb in [450, 1200]:
        tr = ft[:kb * 1024]; te = fe[:min(400 * 1024, kb * 1024 // 2)]; mask = clean_mask(tr, te, 13)
        base = warm(tr, te, mask, False, [])
        allp = warm(tr, te, mask, False, ALL_PRIMS)
        scr = warm(tr, te, mask, False, ALL_PRIMS, scramble=True)
        print(f"\n[{kb}KB train, {len(te)//1024}KB test, {100*sum(mask)/len(te):.0f}% clean]")
        print(f"  baseline (orders only)        : {base:.4f}")
        print(f"  + ALL primitives              : {allp:.4f}   gap {allp-base:+.4f}   vs-noise {scr-allp:+.4f}")
        print(f"  + SCRAMBLED primitives (sanity): {scr:.4f}   gap {scr-base:+.4f}")
        if kb == 450:
            print("  per-primitive (each alone, gap vs baseline; <0 = that counter helps past the orders):")
            for pnm in ALL_PRIMS:
                one = warm(tr, te, mask, False, [pnm])
                print(f"      {pnm:6}: {one:.4f}   gap {one-base:+.4f}")

    print(f"\n[{time.time()-t0:.0f}s]")
    print("=" * 98)


if __name__ == "__main__":
    main()

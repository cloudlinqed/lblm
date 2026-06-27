#!/usr/bin/env python3
"""
genmem.py -- the INTELLIGENCE-vs-MEMORIZATION instrument (toward a stronger byte-level generative model
with long-range memory, where the goal is GENERALIZATION, not recall).

Design (from the bytelm-design plan): keep MEMORIZATION and GENERALIZATION as two physically separate,
individually-ablatable channels, then add ONE new long-range channel that is STRUCTURALLY incapable of
verbatim copy and ask, on leak-free held-out data, whether it generalizes.

  channels mixed (online logistic, the project's stretch/squash unit, RMSProp-free simple SGD):
    * ORDER-k byte contexts  (abstraction; capped to <=ORDER_CAP for the intelligence read)
    * MATCH model            (verbatim copy = the MEMORIZATION oracle; ablatable)
    * RESERVOIR              (NEW: multi-timescale leaky integrators of byte CLASS features -> a bounded
                              smooth running summary; CANNOT replay a span -> any held-out gain = abstraction)

Headline (= intelligence): COPY-ABLATED, DECONTAMINATED, HELD-OUT bits/byte, RESERVOIR on vs off.
If reservoir-ON < reservoir-OFF with copy DISABLED on a 13-byte-decontaminated wt103 TEST split, the
reservoir added GENERALIZATION (it cannot memorize, copy is off, and shared spans are excised).

Data: data/wt103_train.txt + data/wt103_test.txt (Salesforce/wikitext, wikitext-103-raw; official
leak-free train/test). Run: python genmem.py
"""
import sys, math, random, time

ORDERS = [0, 1, 2, 3, 4, 5, 6]
TAUS = [6, 24, 96, 384, 1536]                       # leaky-integrator timescales (bytes): ~6 .. ~1500
DECAY = [math.exp(-1.0 / t) for t in TAUS]
NFEAT = 6                                            # byte-class features integrated at each timescale


def stretch(p):
    p = min(1 - 1e-6, max(1e-6, p)); return math.log(p / (1 - p))


def squash(t):
    if t > 30: return 1 - 1e-6
    if t < -30: return 1e-6
    return 1.0 / (1.0 + math.exp(-t))


def bclass(b):
    c = chr(b) if b < 128 else "\x00"
    return (c.isalpha(), c == " ", c.isdigit(),
            c in ".,;:!?()[]{}\"'-=|/*", c.isupper(), b == 10)   # 6 binary features


# ---------------- decontamination: excise TEST positions inside any >=K-byte TRAIN substring ----------
def clean_mask(train, test, K=13):
    grams = set()
    for i in range(len(train) - K + 1):
        grams.add(hash(train[i:i + K]))
    contam = bytearray(len(test))
    for i in range(len(test) - K + 1):
        if hash(test[i:i + K]) in grams:
            for j in range(i, i + K):
                contam[j] = 1
    mask = [c == 0 for c in contam]                 # True = clean (no >=13-byte overlap with TRAIN)
    return mask


# ---------------- the instrument ----------------
class Model:
    """orders + match (copy) + a multi-timescale reservoir fed as CONTINUOUS mixer features (the fix:
    a continuous bounded state belongs as linear mixer inputs, not as a sparse count-table key)."""
    def __init__(self, use_match=True, use_reservoir=True, order_cap=6, lr=0.004, scramble=False):
        self.use_match, self.use_reservoir, self.order_cap = use_match, use_reservoir, order_cap
        self.lr, self.scramble = lr, scramble
        self.orders = [o for o in ORDERS if o <= order_cap]
        self.NM = len(self.orders)
        self.NRES = NFEAT * len(TAUS) if use_reservoir else 0   # 30 continuous reservoir features
        self.NIN = self.NM + (1 if use_match else 0) + self.NRES
        self.w = [0.0] * (self.NIN + 1)
        self.wg = [0.0] * (self.NIN + 1)            # RMSProp per-weight running grad^2 (stable mixing)
        self.tab = [dict() for _ in self.orders]
        self.MINLEN, self.GATE = 16, 18
        self.mtab = {}; self.hist = bytearray(); self.mptr = -1; self.mlen = 0
        self.h = [0.0] * (NFEAT * len(TAUS))        # flat leaky-integrator state
        self.htail = 0; self.cur = 0; self.phase = 0
        if scramble:
            self._sr = random.Random(1234)

    def _octx(self, k):
        b = self.orders[k]
        mask = (1 << (8 * b)) - 1 if b else 0
        return (self.phase, self.cur, (self.htail & mask) if b else 0)

    def predict(self):
        sts = [0.0] * (self.NIN + 1)
        i = 0
        for k in range(self.NM):
            c = self.tab[k].get(self._octx(k))
            n0, n1 = (c[0], c[1]) if c else (0, 0)
            sts[i] = stretch((n1 + 0.2) / (n0 + n1 + 0.4)); i += 1
        if self.use_match:
            st = 0.0
            if self.mlen >= self.GATE and 0 <= self.mptr < len(self.hist):
                pb = self.hist[self.mptr]
                if self.phase == 0 or (pb >> (8 - self.phase)) == self.cur:
                    bit = (pb >> (7 - self.phase)) & 1
                    st = (1.6 + 0.35 * min(self.mlen, 28)) * (1 if bit else -1)
            sts[i] = st; i += 1
        if self.use_reservoir:
            if self.scramble:
                for r in range(self.NRES):
                    sts[i] = self._sr.uniform(-1, 1); i += 1     # structureless control state
            else:
                for r in range(self.NRES):
                    sts[i] = 2.0 * self.h[r] - 1.0; i += 1       # centered continuous reservoir feature
        sts[self.NIN] = 1.0
        d = sum(self.w[j] * sts[j] for j in range(self.NIN + 1))
        return squash(d), sts

    def step(self, y, learn):
        p, sts = self.predict()
        cost = -math.log2(p if y == 1 else 1 - p)
        if learn:
            err = y - p
            for j in range(self.NIN + 1):
                g = err * sts[j]
                self.wg[j] = 0.999 * self.wg[j] + 0.001 * g * g    # RMSProp: per-feature adaptive step
                self.w[j] += self.lr * g / (math.sqrt(self.wg[j]) + 1e-4)
            for k in range(self.NM):
                key = self._octx(k); c = self.tab[k].get(key)
                if c is None: c = [0, 0]; self.tab[k][key] = c
                c[y] += 1
        self.cur = (self.cur << 1) | y
        self.phase += 1
        if self.phase == 8:
            b = self.cur & 0xFF
            self.hist.append(b)
            if self.use_match:
                self._match_after(b)
            f = bclass(b)
            for fi in range(NFEAT):
                xi = 1.0 if f[fi] else 0.0
                base = fi * len(TAUS)
                for t in range(len(TAUS)):
                    self.h[base + t] = DECAY[t] * self.h[base + t] + (1 - DECAY[t]) * xi
            self.htail = ((self.htail << 8) | b) & ((1 << 48) - 1)
            self.cur = 0; self.phase = 0
        return cost

    def _match_after(self, b):
        n = len(self.hist)
        if self.mlen > 0 and self.mptr < n - 1:
            if self.hist[self.mptr] == self.hist[n - 1]:
                self.mptr += 1; self.mlen = min(self.mlen + 1, 65535)
            else:
                self.mlen = 0; self.mptr = -1
        if n >= self.MINLEN:
            key = bytes(self.hist[n - self.MINLEN:n]); prev = self.mtab.get(key, -1)
            self.mtab[key] = n
            if self.mlen == 0 and 0 <= prev < n:
                self.mptr = prev; self.mlen = self.MINLEN

    def run(self, raw, learn, mask=None):
        """stream raw bytes; return mean bits/byte over scored bytes (mask True) or all bytes."""
        tot = 0.0; nb = 0
        bytepos = 0
        for b in raw:
            scored = (mask is None) or (bytepos < len(mask) and mask[bytepos])
            bc = 0.0
            for j in range(7, -1, -1):
                bc += self.step((b >> j) & 1, learn)
            if scored:
                tot += bc; nb += 1
            bytepos += 1
        return tot / max(1, nb), nb


def order_n_bpb(raw, n):
    """plain order-n byte n-gram cross-entropy (bits/byte), the rail to beat to claim ANY learning."""
    ctx = {}
    tot = 0.0
    h = 0
    mask = (1 << (8 * n)) - 1 if n else 0
    for b in raw:
        key = h & mask
        d = ctx.get(key)
        if d is None:
            d = [1] * 256; ctx[key] = d            # add-one
        s = sum(d)
        tot += -math.log2(d[b] / s)
        d[b] += 1
        h = ((h << 8) | b) & ((1 << (8 * max(n, 1))) - 1)
    return tot / len(raw)


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    TR_CAP, TE_CAP = 500 * 1024, 250 * 1024
    train = open("data/wt103_train.txt", "rb").read()[:TR_CAP]
    test = open("data/wt103_test.txt", "rb").read()[:TE_CAP]
    print("=" * 92)
    print("INTELLIGENCE vs MEMORIZATION on leak-free wikitext-103  (train %dKB, test %dKB)" % (len(train)//1024, len(test)//1024))
    print("=" * 92)
    t0 = time.time()
    mask = clean_mask(train, test, 13)
    clean = sum(mask); print(f"decontamination: {clean}/{len(test)} test bytes clean "
                             f"({100*clean/len(test):.1f}%; rest share a >=13-byte substring with TRAIN, excised)")
    print(f"rails: order-0 = {order_n_bpb(test,0):.3f}  order-4 = {order_n_bpb(test,4):.3f} bits/byte (on TEST)\n")

    def warm_eval(use_match, use_reservoir, scramble=False):
        m = Model(use_match=use_match, use_reservoir=use_reservoir, order_cap=6, scramble=scramble)
        m.run(train, learn=True)                                   # WARM: learn from TRAIN
        bpb, nb = m.run(test, learn=True, mask=mask)               # online on TEST, scored on clean bytes
        return bpb

    print("measuring (warm on TRAIN, online on TEST, scored on decontaminated bytes only)...")
    L_full = warm_eval(True, True)                                 # all channels (copy ON)
    L_nores_copy = warm_eval(True, False)                          # copy ON, reservoir OFF
    L_abl_res = warm_eval(False, True)                            # COPY OFF, reservoir ON   <- headline
    L_abl_nores = warm_eval(False, False)                        # COPY OFF, reservoir OFF  <- baseline
    L_abl_scr = warm_eval(False, True, scramble=True)            # sanity: structureless reservoir

    print("\n[held-out bits/byte on decontaminated wt103 TEST]")
    print(f"  copy ON,  reservoir ON   (full)         : {L_full:.4f}")
    print(f"  copy ON,  reservoir OFF                 : {L_nores_copy:.4f}")
    print(f"  copy OFF, reservoir OFF  (gen baseline) : {L_abl_nores:.4f}")
    print(f"  copy OFF, reservoir ON   (HEADLINE)     : {L_abl_res:.4f}")
    print(f"  copy OFF, reservoir SCRAMBLED (sanity)  : {L_abl_scr:.4f}")

    res_gain = L_abl_nores - L_abl_res                            # reservoir's COPY-ABLATED held-out gain
    copy_gain = L_abl_nores - L_full                              # what the copier adds on top
    scr_gain = L_abl_nores - L_abl_scr
    print("\n[decomposition]")
    print(f"  RESERVOIR generalization gain (copy OFF): {res_gain:+.4f} bits/byte  "
          f"({'INTELLIGENCE: it cannot copy, copy is off, spans excised' if res_gain > 0.003 else 'not significant'})")
    print(f"  scrambled-reservoir control gain        : {scr_gain:+.4f}  (must be ~0; structureless state)")
    print(f"  match/copy (memorization) gain          : {copy_gain:+.4f} bits/byte")
    if (res_gain + copy_gain) > 0:
        G = res_gain / (res_gain + copy_gain)
        print(f"  G = reservoir/(reservoir+copy) gain     : {G:.2f}   (share of long-range gain that is LEARNED, not copied)")
    print("\nVERDICT:", end=" ")
    if res_gain > 0.003 and scr_gain < res_gain * 0.5:
        print("the reservoir LOWERS copy-ablated, decontaminated held-out loss -> it added GENERALIZATION")
        print("  (intelligence), not memorization: it is structurally incapable of verbatim copy, copy was")
        print("  OFF, and shared >=13-byte spans were excised. First honest signal that learnable long-range")
        print("  memory beats recall on held-out text. Next: scale data, learn the decays (RTRL), Rust port.")
    else:
        print("the reservoir did NOT add significant copy-ablated held-out gain on this run.")
        print("  Honest negative -- the gate did its job. Iterate (richer features / timescales / more data).")
    print(f"\n[{time.time()-t0:.0f}s]")
    print("=" * 92)


if __name__ == "__main__":
    main()

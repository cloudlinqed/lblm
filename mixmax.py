#!/usr/bin/env python3
"""
mixmax.py -- merge the two verified winners (sec 30) into ONE bit-native online compressor:
  * from mix_sse:       context-SELECTED mixer + byte-level MATCH model + order-0..6 contexts +
                        sparse/skip + word model + two chained SSE/APM stages   (carried CODE)
  * from two_layer_mix: a GLOBAL mixer running alongside the selected one, combined by a final
                        layer-2 mixer                                            (carried PROSE)
Goal: beat both per-corpus bests at once (prose 0.2400, code 0.1878 @300KB whole-stream).

Online, causal, stdlib only, single-thread. Metric = bits/bit (= compression; raw 1.0, lower better).
Run `python mixmax.py flip` for a future-bit-flip causality self-test.
"""
import sys, math, gzip
ORDERS = [0, 1, 2, 3, 4, 5, 6]
NSEL = 8 * 256                                              # selected-mixer contexts: (phase<<8)|prev_byte


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


class APM:
    def __init__(self, n_ctx, n=33, rate=0.007, smax=8.0):
        self.K = n; self.smax = smax; self.rate = rate; self.step = 2 * smax / (n - 1)
        self.pos = [-smax + j * self.step for j in range(n)]
        base = [squash(x) for x in self.pos]
        self.t = [base[:] for _ in range(n_ctx)]
        self._lo = 0; self._w = 0.0; self._c = 0

    def refine(self, p, cx):
        s = stretch(p)
        if s <= -self.smax:
            lo = 0; w = 0.0
        elif s >= self.smax:
            lo = self.K - 2; w = 1.0
        else:
            x = (s + self.smax) / self.step; lo = int(x)
            if lo >= self.K - 1:
                lo = self.K - 2
            w = x - lo
        self._lo = lo; self._w = w; self._c = cx; r = self.t[cx]
        return r[lo] * (1 - w) + r[lo + 1] * w

    def update(self, y):
        r = self.t[self._c]; lo = self._lo; w = self._w; rt = self.rate
        r[lo] += rt * (1 - w) * (y - r[lo]); r[lo + 1] += rt * w * (y - r[lo + 1])


class MatchModel:
    def __init__(self, hash_bits=22, minlen=5):
        self.mask = (1 << hash_bits) - 1; self.tab = [0] * (1 << hash_bits)
        self.minlen = minlen; self.match_ptr = 0; self.match_len = 0; self.h = 0

    def predicted(self, hist, phase, byte_pos):
        if self.match_len == 0 or self.match_ptr >= byte_pos:
            return 0.0
        pb = (hist[self.match_ptr] >> (7 - phase)) & 1
        st = 1.6 + 0.35 * min(self.match_len, 28)
        return st if pb == 1 else -st

    def update_after_byte(self, hist, byte_pos):
        if self.match_len > 0 and self.match_ptr < byte_pos:
            if hist[self.match_ptr] == hist[byte_pos]:
                self.match_ptr += 1; self.match_len = min(self.match_len + 1, 65535)
            else:
                self.match_len = 0; self.match_ptr = 0
        b = hist[byte_pos]; self.h = ((self.h << 8) | b) & 0xFFFFFFFFFFFF
        if byte_pos + 1 >= self.minlen:
            hk = (self.h * 2654435761) & self.mask
            prev = self.tab[hk]; self.tab[hk] = byte_pos + 1
            if self.match_len == 0 and prev != 0 and prev <= byte_pos:
                self.match_ptr = prev; self.match_len = self.minlen


def run(bits, lr=0.0085, lr_final=0.01, delta=0.18, record=False):
    n = len(bits); NM = len(ORDERS)
    I_SP, I_WD, I_MT, NIN = NM, NM + 1, NM + 2, NM + 3
    tables = [dict() for _ in range(NM)]; sparse = dict(); wordt = dict()
    mixers = [[0.0] * (NIN + 1) for _ in range(NSEL)]
    gmix = [0.0] * (NIN + 1)
    final_w = [0.3, 0.3, 0.0]
    apm1 = APM(256 * 8, rate=0.007); apm2 = APM(1024, rate=0.005)
    match = MatchModel()
    hist = bytearray(); cur = 0; prev_byte = 0; prev2 = 0; word_hash = 0; byte_pos = 0
    sts = [0.0] * (NIN + 1); cells = [None] * NM
    split = int(n * 0.8); tot = 0.0; tail = 0.0; tailn = 0; log2 = math.log(2.0)
    costs = [] if record else None
    for i in range(n):
        phase = i & 7; prefix = cur
        for k in range(NM):
            B = ORDERS[k]
            if B == 0:
                key = (phase, prefix)
            else:
                ctxb = bytes(hist[byte_pos - B:byte_pos]) if byte_pos >= B else bytes(hist[:byte_pos])
                key = (phase, prefix, ctxb)
            c = tables[k].get(key)
            if c is None:
                c = [0, 0]; tables[k][key] = c
            cells[k] = c
            sts[k] = stretch((c[1] + delta) / (c[0] + c[1] + 2 * delta))
        sk = (phase, prefix, hist[byte_pos - 2] if byte_pos >= 2 else 0, hist[byte_pos - 3] if byte_pos >= 3 else 0)
        c = sparse.get(sk)
        if c is None:
            c = [0, 0]; sparse[sk] = c
        sp = c; sts[I_SP] = stretch((c[1] + delta) / (c[0] + c[1] + 2 * delta))
        wk = (word_hash, phase, prefix)
        c = wordt.get(wk)
        if c is None:
            c = [0, 0]; wordt[wk] = c
        wd = c; sts[I_WD] = stretch((c[1] + delta) / (c[0] + c[1] + 2 * delta))
        sts[I_MT] = match.predicted(hist, phase, byte_pos)
        sts[NIN] = 1.0
        sel = (phase << 8) | prev_byte
        w = mixers[sel]
        d = 0.0
        for k in range(NIN + 1):
            d += w[k] * sts[k]
        p_sel = squash(d)
        dg = 0.0
        for k in range(NIN + 1):
            dg += gmix[k] * sts[k]
        p_g = squash(dg)
        ssel = stretch(p_sel); sg = stretch(p_g)
        p_mix = squash(final_w[0] * ssel + final_w[1] * sg + final_w[2])
        Pa = apm1.refine(p_mix, (prev_byte << 3) | phase); Pa = 0.3 * p_mix + 0.7 * Pa
        Pb = apm2.refine(Pa, (prev_byte * 769 + prev2 * 31 + phase) & 1023); P = 0.3 * Pa + 0.7 * Pb
        P = min(1 - 1e-6, max(1e-6, P))
        y = bits[i]
        cost = -(math.log(P if y == 1 else 1 - P) / log2)
        tot += cost
        if record:
            costs.append(cost)
        if i >= split:
            tail += cost; tailn += 1
        err_f = y - p_mix
        final_w[0] += lr_final * err_f * ssel; final_w[1] += lr_final * err_f * sg; final_w[2] += lr_final * err_f
        g = lr * (y - p_sel)
        for k in range(NIN + 1):
            w[k] += g * sts[k]
        gg = lr * (y - p_g)
        for k in range(NIN + 1):
            gmix[k] += gg * sts[k]
        for k in range(NM):
            cells[k][y] += 1
        sp[y] += 1; wd[y] += 1
        apm1.update(y); apm2.update(y)
        cur = (cur << 1) | y
        if phase == 7:
            b = cur & 0xFF; hist.append(b); match.update_after_byte(hist, byte_pos)
            if (65 <= b <= 90) or (97 <= b <= 122):
                word_hash = (word_hash * 131 + (b | 0x20)) & 0xFFFFFFF
            else:
                word_hash = 0
            prev2 = prev_byte; prev_byte = b; cur = 0; byte_pos += 1
    return tot / n, (tail / tailn if tailn else 0.0), costs


def flip_test(path="data/corpus.txt", cap=40000):
    raw, bits = load_bits(path, cap)
    mid = (cap // 2) * 8; fb = (cap * 3 // 4) * 8 + 3       # flip a bit well AFTER the checkpoint
    _, _, c1 = run(bits, record=True)
    b2 = bytearray(bits); b2[fb] ^= 1
    _, _, c2 = run(b2, record=True)
    past_same = all(abs(c1[i] - c2[i]) < 1e-12 for i in range(mid))
    future_changed = any(abs(c1[i] - c2[i]) > 1e-12 for i in range(fb, min(fb + 2000, len(c1))))
    print(f"causality flip-test (flip bit {fb}): past {mid} bits identical = {past_same}; "
          f"future changed = {future_changed}  -> {'CAUSAL/leakage-free' if (past_same and future_changed) else 'FAIL'}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "flip":
        flip_test(); return
    path = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.txt"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300000
    raw, bits = load_bits(path, cap)
    g = len(gzip.compress(raw, 9))
    whole, tail, _ = run(bits)
    print(f"corpus={path}  bytes={len(raw)}  bits={len(bits)}")
    print(f"  mixmax  whole-stream = {whole:.4f}   last-20% = {tail:.4f}  bits/bit")
    print(f"  gzip (whole file)    = {g / len(raw):.4f}  bits/bit")


if __name__ == "__main__":
    main()

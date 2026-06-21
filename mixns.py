#!/usr/bin/env python3
"""
mixns.py -- ONE concrete advancement over mixmax.py: NON-STATIONARITY + REACH.

Starts from mixmax's architecture (context-selected + global + final mixers,
byte-match model, orders 0..6, sparse/word models, 2 chained SSE/APM) and adds:

  (1) RECENCY WEIGHTING / non-stationarity:
        bit-count cells decay -- when a count reaches a soft limit, BOTH counts
        are halved (per-cell adaptive halving). Causal (depends only on the
        cell's own past) and weights recent observations more.
  (2) ADAPTIVE PER-MODEL LEARNING RATE for the mixers:
        each mixer weight carries its own RMSProp-style learning rate (a slowly
        decaying running mean of squared gradients normalises every step), so
        consistently-informative inputs take larger effective steps and noisy
        inputs are damped. The decaying (vs cumulative AdaGrad) denominator keeps
        the rate responsive on a nonstationary stream. Causal (only past grads).
  (3) REACH: large HASHED HIGH-ORDER contexts (orders 8, 12, 16) hashed into
        fixed-size flat tables, added as extra mixer inputs -- extends context
        reach beyond order-6 without unbounded dict growth.

Online, causal, stdlib only, single-thread, no multiprocessing.
Metric = whole-stream bits/bit = mean(-log2 P(true bit)), lower better.

Run `python mixns.py flip`               for a future-bit-flip causality self-test.
Run `python mixns.py data/corpus.txt 200000`  to measure on a corpus.
"""
import sys, math, gzip
from array import array

ORDERS = [0, 1, 2, 3, 4, 5, 6]
HORDERS = [8, 12, 16]                       # (3) REACH: large hashed high-order contexts
HBITS = 20                                  # 1M-entry flat hashed table per high order
HMASK = (1 << HBITS) - 1
NSEL = 8 * 256                              # selected-mixer contexts: (phase<<8)|prev_byte
CLIMIT = 255                                # (1) soft count limit -> halve on reach (recency)


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


def run(bits, lr=0.0085, lr_final=0.01, delta=0.18, record=False,
        recency=True, adaptive_lr=True, reach=True, climit=CLIMIT, hbits=HBITS,
        alr_lr=0.0013, alr_lrf=0.0013, rms_decay=0.9999, rms_eps=1e-4):
    """Three toggles let us ablate each advancement; all True = full mixns.

    (2) adaptive_lr uses an RMSProp-style per-weight rate: a slowly-decaying
    running mean of squared gradients normalises each step. Unlike AdaGrad its
    denominator does not grow without bound, so the rate stays responsive on a
    nonstationary stream. alr_lr / alr_lrf are the (larger) base rates used when
    adaptive_lr is on; lr / lr_final are the plain-SGD rates used when it is off.
    """
    n = len(bits); NM = len(ORDERS)
    horders = HORDERS if reach else []
    NH = len(horders)
    hmask = (1 << hbits) - 1
    # input layout: [orders 0..6][high orders][sparse][word][match][bias]
    I_H = NM
    I_SP = NM + NH
    I_WD = NM + NH + 1
    I_MT = NM + NH + 2
    NIN = NM + NH + 3
    tables = [dict() for _ in range(NM)]; sparse = dict(); wordt = dict()
    # (3) REACH: fixed-size flat hashed count tables (2 cells per slot: n0,n1).
    htables = [array('I', bytes(4 * 2 * (1 << hbits))) for _ in range(NH)]
    mixers = [[0.0] * (NIN + 1) for _ in range(NSEL)]
    gmix = [0.0] * (NIN + 1)
    # (2) ADAPTIVE PER-MODEL LR: RMSProp running-mean-of-squared-grad accumulators.
    mixers_g = [[0.0] * (NIN + 1) for _ in range(NSEL)]
    gmix_g = [0.0] * (NIN + 1)
    final_w = [0.3, 0.3, 0.0]
    final_g = [0.0, 0.0, 0.0]
    apm1 = APM(256 * 8, rate=0.007); apm2 = APM(1024, rate=0.005)
    match = MatchModel()
    hist = bytearray(); cur = 0; prev_byte = 0; prev2 = 0; word_hash = 0; byte_pos = 0
    sts = [0.0] * (NIN + 1); cells = [None] * NM
    hidxs = [0] * NH                          # slot index (2*slot) into each htable this bit
    hbase = [0] * NH                          # base hash of last B bytes, refreshed per byte
    split = int(n * 0.8); tot = 0.0; tail = 0.0; tailn = 0; log2 = math.log(2.0)
    costs = [] if record else None
    sqrt = math.sqrt
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
        # (3) REACH: high-order hashed contexts (base hash precomputed per byte)
        for hk_i in range(NH):
            hv = hbase[hk_i]
            slot = (((hv * 2654435761) ^ (phase * 0x9E3779B1) ^ (prefix * 2246822519)) & hmask) << 1
            hidxs[hk_i] = slot
            ht = htables[hk_i]; n0 = ht[slot]; n1 = ht[slot + 1]
            sts[I_H + hk_i] = stretch((n1 + delta) / (n0 + n1 + 2 * delta))
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
        w = mixers[sel]; wg = mixers_g[sel]
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
        # ---- updates ----
        err_f = y - p_mix
        gf0 = err_f * ssel; gf1 = err_f * sg; gf2 = err_f
        if adaptive_lr:
            # (2) per-weight RMSProp on the final mixer
            rd = rms_decay; ord_ = 1.0 - rd
            final_g[0] = rd * final_g[0] + ord_ * gf0 * gf0
            final_g[1] = rd * final_g[1] + ord_ * gf1 * gf1
            final_g[2] = rd * final_g[2] + ord_ * gf2 * gf2
            final_w[0] += alr_lrf * gf0 / (sqrt(final_g[0]) + rms_eps)
            final_w[1] += alr_lrf * gf1 / (sqrt(final_g[1]) + rms_eps)
            final_w[2] += alr_lrf * gf2 / (sqrt(final_g[2]) + rms_eps)
        else:
            final_w[0] += lr_final * gf0; final_w[1] += lr_final * gf1; final_w[2] += lr_final * gf2
        e_sel = y - p_sel
        e_g = y - p_g
        if adaptive_lr:
            rd = rms_decay; ord_ = 1.0 - rd; eps = rms_eps; a = alr_lr
            for k in range(NIN + 1):
                gk = e_sel * sts[k]; wg[k] = rd * wg[k] + ord_ * gk * gk
                w[k] += a * gk / (sqrt(wg[k]) + eps)
            for k in range(NIN + 1):
                gk = e_g * sts[k]; gmix_g[k] = rd * gmix_g[k] + ord_ * gk * gk
                gmix[k] += a * gk / (sqrt(gmix_g[k]) + eps)
        else:
            gsel = lr * e_sel; gglob = lr * e_g
            for k in range(NIN + 1):
                w[k] += gsel * sts[k]
            for k in range(NIN + 1):
                gmix[k] += gglob * sts[k]
        # (1) RECENCY: bump counts, halve BOTH on reaching the soft limit
        if recency:
            for k in range(NM):
                c = cells[k]; c[y] += 1
                if c[y] >= climit:
                    c[0] = (c[0] + 1) >> 1; c[1] = (c[1] + 1) >> 1
            sp[y] += 1
            if sp[y] >= climit:
                sp[0] = (sp[0] + 1) >> 1; sp[1] = (sp[1] + 1) >> 1
            wd[y] += 1
            if wd[y] >= climit:
                wd[0] = (wd[0] + 1) >> 1; wd[1] = (wd[1] + 1) >> 1
            for hk_i in range(NH):
                ht = htables[hk_i]; slot = hidxs[hk_i]
                ht[slot + y] += 1
                if ht[slot + y] >= climit:
                    ht[slot] = (ht[slot] + 1) >> 1; ht[slot + 1] = (ht[slot + 1] + 1) >> 1
        else:
            for k in range(NM):
                cells[k][y] += 1
            sp[y] += 1; wd[y] += 1
            for hk_i in range(NH):
                ht = htables[hk_i]; ht[hidxs[hk_i] + y] += 1
        apm1.update(y); apm2.update(y)
        cur = (cur << 1) | y
        if phase == 7:
            b = cur & 0xFF; hist.append(b); match.update_after_byte(hist, byte_pos)
            if (65 <= b <= 90) or (97 <= b <= 122):
                word_hash = (word_hash * 131 + (b | 0x20)) & 0xFFFFFFF
            else:
                word_hash = 0
            prev2 = prev_byte; prev_byte = b; cur = 0; byte_pos += 1
            # (3) REACH: refresh base hash of last B bytes per high order (causal)
            for hk_i in range(NH):
                B = horders[hk_i]
                lo = byte_pos - B if byte_pos >= B else 0
                hv = 1469598103934665603
                for bp in range(lo, byte_pos):
                    hv = ((hv ^ hist[bp]) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
                hbase[hk_i] = hv
    return tot / n, (tail / tailn if tailn else 0.0), costs


def flip_test(path="data/corpus.txt", cap=40000, **kw):
    raw, bits = load_bits(path, cap)
    mid = (cap // 2) * 8; fb = (cap * 3 // 4) * 8 + 3       # flip a bit well AFTER the checkpoint
    _, _, c1 = run(bits, record=True, **kw)
    b2 = bytearray(bits); b2[fb] ^= 1
    _, _, c2 = run(b2, record=True, **kw)
    past_same = all(abs(c1[i] - c2[i]) < 1e-12 for i in range(mid))
    future_changed = any(abs(c1[i] - c2[i]) > 1e-12 for i in range(fb, min(fb + 2000, len(c1))))
    ok = past_same and future_changed
    print(f"causality flip-test (flip bit {fb}): past {mid} bits identical = {past_same}; "
          f"future changed = {future_changed}  -> {'CAUSAL/leakage-free' if ok else 'FAIL'}")
    return ok


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "flip":
        flip_test(); return
    path = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.txt"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 300000
    raw, bits = load_bits(path, cap)
    g = len(gzip.compress(raw, 9))
    whole, tail, _ = run(bits)
    print(f"corpus={path}  bytes={len(raw)}  bits={len(bits)}")
    print(f"  mixns   whole-stream = {whole:.4f}   last-20% = {tail:.4f}  bits/bit")
    print(f"  gzip (whole file)    = {g / len(raw):.4f}  bits/bit")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
two-layer-mix: a STRONGER bit-native online compressor than mix.py.

Architecture (PAQ-style two-layer context mixing, fully bit-native and causal):

  Layer 0 -- many context models.
     For each byte-aware order B in ORDERS we keep an adaptive bit predictor keyed by
     (bit-phase, partial-byte-so-far, previous B bytes). Each model also keeps a small
     run/confidence count so its stretched vote carries a magnitude. We additionally add
     a couple of sparse / word contexts that help on text and code.

  Layer 1 -- a SET of mixers selected by context (the PAQ mixer-set trick).
     Instead of ONE logistic mixer, we keep a set of mixers and SELECT one per bit by a
     cheap context (bit-phase XOR a hash of the order-0 byte and the previous byte). Each
     selected mixer mixes the layer-0 stretched votes online (logistic gradient).

  Layer 2 -- final combine + SSE/APM.
     The chosen mixer's output is one input to a tiny FINAL mixer that also sees a global
     mixer (mixer averaged context); then an adaptive probability map (SSE) refines the
     final probability against the realised bit, interpolating over a stretch grid keyed
     by a small context. Everything trains online from the realised bit.

Metric: bytes -> bits MSB-first; bits/bit = mean(-log2 P(true bit)); causal; whole stream.
stdlib only, single-threaded, no multiprocessing.

Usage: python two_layer_mix.py [path] [byte_cap]
"""
import sys, math

# byte-aware context orders for layer-0 models
ORDERS = [0, 1, 2, 3, 4, 6, 8]
# number of mixers in the selected set (layer 1)
NMIX = 256
# SSE / APM grid resolution and contexts
APM_N = 33
APM_CTX = 1024


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
    if p < 1e-6:
        p = 1e-6
    elif p > 1 - 1e-6:
        p = 1 - 1e-6
    return math.log(p / (1 - p))


def squash(t):
    if t > 30:
        return 1 - 1e-6
    if t < -30:
        return 1e-6
    return 1.0 / (1.0 + math.exp(-t))


# Precompute a stretch table for SSE grid endpoints
def build_apm():
    # grid of APM_N points over stretch range [-S, S]
    S = 16.0
    xs = [(-S + 2 * S * j / (APM_N - 1)) for j in range(APM_N)]
    return S, xs


class Model:
    """Adaptive bit predictor table: ctx -> [n0, n1]. Stretched vote with confidence."""
    __slots__ = ("t", "delta")

    def __init__(self, delta=0.25):
        self.t = {}
        self.delta = delta

    def cell(self, key):
        c = self.t.get(key)
        if c is None:
            c = [0, 0]
            self.t[key] = c
        return c


def run(bits, raw, lr=0.0028, lr_final=0.01):
    n = len(bits)
    nm = len(ORDERS)

    # layer-0 models
    models = [Model() for _ in ORDERS]
    # 3 extra sparse/word models
    m_word = Model()      # current word (alnum run) so far + phase + partial byte
    m_sparse2 = Model()   # bytes at offset -2 and -4 (skip context)
    m_byte = Model()      # order-0 plain byte (no phase) running adaptive
    extra = [m_word, m_sparse2, m_byte]
    ninp = nm + len(extra)

    # layer-1: a SET of mixers, each a weight vector of length ninp (+1 bias input)
    NW = ninp + 1
    mixers = [[0.0] * NW for _ in range(NMIX)]

    # layer-2: final mixer combining (selected mixer output, global mixer output)
    final_w = [0.3, 0.3, 0.0]   # [sel, global, bias]
    # a global mixer (single) for robustness
    global_mix = [0.0] * NW

    # SSE / APM stage
    S, apm_xs = build_apm()
    # apm[ctx] = list of APM_N probabilities (initialized to identity squash(x))
    apm = {}

    def apm_get(c):
        a = apm.get(c)
        if a is None:
            a = [squash(x) for x in apm_xs]
            apm[c] = a
        return a

    # rolling byte history (as ints)
    hist = bytearray()   # decoded bytes so far
    cur_byte = 0         # partial current byte bits accumulated (value of bits so far)
    word = bytearray()   # current alnum word

    tot = 0.0
    split = int(n * 0.8)
    tail = 0.0
    tailn = 0

    # local refs for speed
    _stretch = stretch
    _squash = squash
    _log2 = math.log2

    for i in range(n):
        phase = i & 7
        bstart = i - phase
        # partial byte so far within current byte
        partial = bytes(bits[bstart:i])

        # previous bytes (as bytes object) for orders
        H = hist
        lh = len(H)

        sts = [0.0] * ninp
        cells = [None] * ninp

        # layer-0 ordered byte contexts
        for k, B in enumerate(ORDERS):
            if B == 0:
                key = (phase, partial)
            else:
                ctxb = bytes(H[lh - B:lh]) if lh >= B else bytes(H)
                key = (phase, partial, ctxb)
            c = models[k].cell(key)
            n0 = c[0]; n1 = c[1]
            p = (n1 + 0.25) / (n0 + n1 + 0.5)
            # confidence scaling: more counts -> trust stretch more (gentle)
            st = _stretch(p)
            sts[k] = st
            cells[k] = c

        # extra: word context
        wkey = (phase, partial, bytes(word[-6:]))
        cw = m_word.cell(wkey)
        pw = (cw[1] + 0.25) / (cw[0] + cw[1] + 0.5)
        sts[nm] = _stretch(pw)
        cells[nm] = cw

        # extra: sparse skip context (bytes at -2 and -4)
        b2 = H[lh - 2] if lh >= 2 else 0
        b4 = H[lh - 4] if lh >= 4 else 0
        skey = (phase, partial, b2, b4)
        cs = m_sparse2.cell(skey)
        ps = (cs[1] + 0.25) / (cs[0] + cs[1] + 0.5)
        sts[nm + 1] = _stretch(ps)
        cells[nm + 1] = cs

        # extra: order-0 plain (phase + partial only, separate adaptive)
        # use a different delta-ish smoothing: same as byte model keyed by (phase,partial,prev1 high nibble)
        pbk = (phase, partial, (H[lh - 1] >> 4) if lh >= 1 else 0)
        cb = m_byte.cell(pbk)
        pb = (cb[1] + 0.25) / (cb[0] + cb[1] + 0.5)
        sts[nm + 2] = _stretch(pb)
        cells[nm + 2] = cb

        # ---- layer 1: select a mixer by context ----
        prev1 = H[lh - 1] if lh >= 1 else 0
        sel = (phase * 31 + ((prev1 * 11 + cur_byte * 7) & 0xff)) % NMIX
        wsel = mixers[sel]
        dot_sel = wsel[ninp]  # bias weight * 1
        for k in range(ninp):
            dot_sel += wsel[k] * sts[k]
        p_sel = _squash(dot_sel)

        # global mixer
        dot_g = global_mix[ninp]
        for k in range(ninp):
            dot_g += global_mix[k] * sts[k]
        p_g = _squash(dot_g)

        # ---- layer 2: final combine ----
        ssel = _stretch(p_sel)
        sg = _stretch(p_g)
        dot_f = final_w[0] * ssel + final_w[1] * sg + final_w[2]
        p_mix = _squash(dot_f)

        # ---- SSE / APM refinement ----
        actx = (phase * 131 + (prev1 & 0x3f)) % APM_CTX
        a = apm_get(actx)
        smix = _stretch(p_mix)
        if smix <= -S:
            p_final = a[0]
            j_lo = 0; j_hi = 0; frac = 0.0
        elif smix >= S:
            p_final = a[APM_N - 1]
            j_lo = APM_N - 1; j_hi = APM_N - 1; frac = 0.0
        else:
            pos = (smix + S) / (2 * S) * (APM_N - 1)
            j_lo = int(pos)
            frac = pos - j_lo
            j_hi = j_lo + 1
            p_final = a[j_lo] * (1 - frac) + a[j_hi] * frac

        # blend SSE output with pre-SSE (PAQ commonly averages); weight SSE more
        p_out = 0.7 * p_final + 0.3 * p_mix
        if p_out < 1e-6:
            p_out = 1e-6
        elif p_out > 1 - 1e-6:
            p_out = 1 - 1e-6

        y = bits[i]
        cost = -_log2(p_out if y == 1 else 1 - p_out)
        tot += cost
        if i >= split:
            tail += cost
            tailn += 1

        # ---- online updates ----
        # SSE update: move the two interpolation endpoints toward y
        arate = 0.02
        if j_lo == j_hi:
            a[j_lo] += arate * (y - a[j_lo])
        else:
            a[j_lo] += arate * (1 - frac) * (y - a[j_lo])
            a[j_hi] += arate * frac * (y - a[j_hi])

        # final mixer update (on p_mix vs y)
        err_f = y - p_mix
        final_w[0] += lr_final * err_f * ssel
        final_w[1] += lr_final * err_f * sg
        final_w[2] += lr_final * err_f

        # selected mixer update (on p_sel vs y)
        err_s = y - p_sel
        g = lr * err_s
        for k in range(ninp):
            wsel[k] += g * sts[k]
        wsel[ninp] += g

        # global mixer update (on p_g vs y)
        err_g = y - p_g
        gg = lr * err_g
        for k in range(ninp):
            global_mix[k] += gg * sts[k]
        global_mix[ninp] += gg

        # layer-0 count updates
        for k in range(ninp):
            cells[k][y] += 1

        # ---- advance byte/bit state ----
        cur_byte = (cur_byte << 1) | y
        if phase == 7:
            byte = cur_byte & 0xff
            hist.append(byte)
            cur_byte = 0
            ch = byte
            if (48 <= ch <= 57) or (65 <= ch <= 90) or (97 <= ch <= 122) or ch == 95:
                word.append(byte)
                if len(word) > 32:
                    del word[:-32]
            else:
                word.clear()

    return tot / n, (tail / tailn if tailn else 0.0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.txt"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    raw, bits = load_bits(path, cap)
    whole, tail = run(bits, raw)
    print(f"corpus={path}  bytes={len(raw)}  bits={len(bits)}")
    print(f"  two-layer-mix  whole-stream = {whole:.4f}   last-20% = {tail:.4f}  bits/bit")


if __name__ == "__main__":
    main()

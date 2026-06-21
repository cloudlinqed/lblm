#!/usr/bin/env python3
"""
mix+sse: a STRONGER bit-native online compressor than mix.py.

Builds on the logistic context-mixing core of mix.py and adds:
  * MORE context models: byte-aware orders 0..6, a sparse "skip" model, a
    whole-word model (for prose) and a byte-level MATCH model (predicts the
    next bit from the longest recent context match -- the lpaq/PAQ match model).
  * A context-SELECTED logistic mixer: the mixer weight vector is chosen by a
    small selector context (order-0 byte position + a coarse hash of the
    previous byte) so weights specialise and adapt faster.
  * An SSE / APM stage (Secondary Symbol Estimation, a.k.a. Adaptive
    Probability Map): the mixed probability is refined by a small adaptive
    table indexed by a short context and the *quantised* stretched P, with
    linear interpolation between the two nearest quantisation knots and online
    update of both knots. Two APMs are chained.

Adaptive / online over the whole stream. Metric = bits/bit (cross entropy);
raw = 1.0000, lower is better. Single-threaded, stdlib only, fully causal
(only bits strictly before position i inform the prediction of bit i).

Usage: python mix_sse.py [path] [byte_cap]
"""
import sys, math, gzip

# byte-aware context orders to mix (number of WHOLE prior bytes of context)
ORDERS = [0, 1, 2, 3, 4, 5, 6]

STRETCH_TAB = None  # filled in below


# ---------------------------------------------------------------------------
# bit loading
# ---------------------------------------------------------------------------
def load_bits(path, cap):
    raw = open(path, "rb").read()
    if cap:
        raw = raw[:cap]
    bits = bytearray()
    for byte in raw:
        for j in range(7, -1, -1):
            bits.append((byte >> j) & 1)
    return raw, bits


# ---------------------------------------------------------------------------
# logistic helpers
# ---------------------------------------------------------------------------
def _stretch(p):
    p = min(1 - 1e-6, max(1e-6, p))
    return math.log(p / (1 - p))


def squash(t):
    if t > 30:
        return 1 - 1e-6
    if t < -30:
        return 1e-6
    return 1.0 / (1.0 + math.exp(-t))


# Precompute a stretch lookup over a 12-bit probability grid for speed in APM.
def _build_stretch_tab():
    tab = [0.0] * 4097
    for i in range(4097):
        p = (i + 0.5) / 4098.0
        tab[i] = _stretch(p)
    return tab


# ---------------------------------------------------------------------------
# Adaptive Probability Map (SSE / APM).
#   N knots spanning the stretch domain [-smax, +smax]. The input probability
#   p is stretched, located between two knots, the two knots' stored
#   probabilities are linearly interpolated to give the refined p, and after
#   the true bit is known BOTH bracketing knots are nudged toward it (weighted
#   by interpolation distance). Indexed by an external context `cx`.
# ---------------------------------------------------------------------------
class APM:
    def __init__(self, n_ctx, n_knots=33, rate=0.0065, smax=8.0):
        self.K = n_knots
        self.smax = smax
        self.rate = rate
        self.step = (2.0 * smax) / (n_knots - 1)
        # knot stretch positions
        self.pos = [-smax + j * self.step for j in range(n_knots)]
        # table[ctx][knot] = stored probability, init to squash(knot pos)
        base = [squash(self.pos[j]) for j in range(n_knots)]
        self.t = [base[:] for _ in range(n_ctx)]
        self._lo = 0
        self._w = 0.0
        self._ctx = 0

    def refine(self, p, cx):
        s = _stretch(p)
        if s <= -self.smax:
            lo = 0; w = 0.0
        elif s >= self.smax:
            lo = self.K - 2; w = 1.0
        else:
            x = (s + self.smax) / self.step
            lo = int(x)
            if lo >= self.K - 1:
                lo = self.K - 2
            w = x - lo
        row = self.t[cx]
        self._lo = lo; self._w = w; self._ctx = cx
        return row[lo] * (1.0 - w) + row[lo + 1] * w

    def update(self, y):
        row = self.t[self._ctx]
        lo = self._lo; w = self._w; r = self.rate
        # nudge each bracketing knot toward y, weighted by closeness
        g0 = r * (1.0 - w)
        g1 = r * w
        row[lo] += g0 * (y - row[lo])
        row[lo + 1] += g1 * (y - row[lo + 1])


# ---------------------------------------------------------------------------
# MATCH model: track the longest match of the current byte-history; predict the
# bit that followed last time. Operates at byte granularity but predicts a bit.
# ---------------------------------------------------------------------------
class MatchModel:
    """
    Hash the last MINLEN bytes; remember the position that hash last occurred.
    While the bytes following the remembered position keep matching the current
    history, predict the next bit equal to the bit at the matched position,
    with a confidence that grows with match length.
    """
    def __init__(self, hash_bits=22, minlen=4):
        self.size = 1 << hash_bits
        self.mask = self.size - 1
        self.tab = [0] * self.size      # hash -> last byte position (+1; 0=empty)
        self.minlen = minlen
        self.match_ptr = 0              # byte index in history we are tracking
        self.match_len = 0             # current verified match length in bytes
        self.h = 0                     # rolling hash of recent bytes

    def _hash(self):
        return (self.h * 2654435761) & self.mask

    def predicted_bit(self, hist_bytes, cur_byte_bits, phase, byte_pos):
        """
        hist_bytes : the full byte history so far (list/bytearray)
        cur_byte_bits: the bits of the in-progress current byte already seen
        phase      : how many bits of the current byte are known (0..7)
        Returns (stretch_input, st) where st is contribution magnitude; if no
        match, returns 0.0 (neutral).
        """
        if self.match_len == 0 or self.match_ptr >= byte_pos:
            return 0.0
        pred_byte = hist_bytes[self.match_ptr]
        # the bit of pred_byte at this phase
        pbit = (pred_byte >> (7 - phase)) & 1
        # confidence grows with match length, capped
        conf = min(self.match_len, 28)
        st = (1.6 + 0.35 * conf)
        return st if pbit == 1 else -st

    def update_after_byte(self, hist_bytes, byte_pos):
        """Call once a full byte (at index byte_pos) has been appended to hist."""
        # advance / verify current match
        if self.match_len > 0 and self.match_ptr < byte_pos:
            if hist_bytes[self.match_ptr] == hist_bytes[byte_pos]:
                self.match_ptr += 1
                self.match_len = min(self.match_len + 1, 65535)
            else:
                self.match_len = 0
                self.match_ptr = 0
        # update rolling hash with the new byte
        b = hist_bytes[byte_pos]
        self.h = ((self.h << 8) | b) & 0xFFFFFFFFFFFF  # keep ~6 bytes
        if byte_pos + 1 >= self.minlen:
            hkey = self._hash()
            prev = self.tab[hkey]
            self.tab[hkey] = byte_pos + 1  # store +1 so 0 means empty
            if self.match_len == 0 and prev != 0:
                cand = prev  # candidate is "position+1" of last byte of context
                # candidate points to a byte; the byte AFTER it should match our next
                if cand <= byte_pos:
                    self.match_ptr = cand
                    self.match_len = self.minlen


# ---------------------------------------------------------------------------
# main predictor
# ---------------------------------------------------------------------------
def run(bits, raw, lr=0.0085, delta=0.18):
    global STRETCH_TAB
    STRETCH_TAB = _build_stretch_tab()

    n = len(bits)
    nbytes = len(raw)
    NM = len(ORDERS)            # order models
    # extra models: sparse(skip-1 byte), word model, match model -> indices
    I_SPARSE = NM
    I_WORD = NM + 1
    I_MATCH = NM + 2
    NIN = NM + 3               # total mixer inputs (plus bias handled separately)

    tables = [dict() for _ in range(NM)]
    sparse_tab = dict()
    word_tab = dict()

    # context-selected mixer: select a weight set by (phase, prev_byte_hash)
    NSEL = 8 * 256
    weights = [[0.0] * (NIN + 1) for _ in range(NSEL)]  # +1 for bias input

    # APM stages
    apm1 = APM(n_ctx=256 * 8, n_knots=33, rate=0.0070)   # ctx: prevbyte*8 + phase
    apm2 = APM(n_ctx=1024, n_knots=33, rate=0.0050)      # ctx: hash(order2)

    match = MatchModel(hash_bits=22, minlen=5)

    hist_bytes = bytearray()    # completed bytes so far
    cur = 0                     # in-progress current byte value (known bits)
    prev_byte = 0
    prev2_byte = 0

    word_hash = 0               # rolling hash of current word (letters)

    split = int(n * 0.8)
    tot = 0.0
    tail = 0.0
    tailn = 0

    sts = [0.0] * (NIN + 1)
    cells = [None] * NM
    sp_cell = None
    wd_cell = None

    log2 = math.log(2.0)

    byte_pos = 0  # index of the byte currently being built == len(hist_bytes)

    for i in range(n):
        phase = i & 7
        # ---- byte-aware context key components ----
        # context bytes already completed = hist_bytes (len == byte_pos)
        # within-byte prefix = cur shifted (bits seen so far this byte)
        prefix = cur  # value of bits seen so far, as integer 0..(2^phase-1)

        # ---- order models ----
        for k in range(NM):
            B = ORDERS[k]
            if B == 0:
                key = (phase, prefix)
            else:
                if byte_pos >= B:
                    ctxb = bytes(hist_bytes[byte_pos - B:byte_pos])
                else:
                    ctxb = bytes(hist_bytes[:byte_pos])
                key = (phase, prefix, ctxb)
            tk = tables[k]
            c = tk.get(key)
            if c is None:
                c = [0, 0]
                tk[key] = c
            cells[k] = c
            p = (c[1] + delta) / (c[0] + c[1] + 2.0 * delta)
            sts[k] = _stretch(p)

        # ---- sparse model: skip the immediately-previous byte, use byte at -2 ----
        if byte_pos >= 2:
            sp_key = (phase, prefix, hist_bytes[byte_pos - 2], hist_bytes[byte_pos - 3] if byte_pos >= 3 else 0)
        else:
            sp_key = (phase, prefix, 0, 0)
        c = sparse_tab.get(sp_key)
        if c is None:
            c = [0, 0]
            sparse_tab[sp_key] = c
        sp_cell = c
        p = (c[1] + delta) / (c[0] + c[1] + 2.0 * delta)
        sts[I_SPARSE] = _stretch(p)

        # ---- word model: hash of current (in-progress) word + phase+prefix ----
        wd_key = (word_hash, phase, prefix)
        c = word_tab.get(wd_key)
        if c is None:
            c = [0, 0]
            word_tab[wd_key] = c
        wd_cell = c
        p = (c[1] + delta) / (c[0] + c[1] + 2.0 * delta)
        sts[I_WORD] = _stretch(p)

        # ---- match model ----
        sts[I_MATCH] = match.predicted_bit(hist_bytes, prefix, phase, byte_pos)

        # ---- bias input ----
        sts[NIN] = 1.0

        # ---- context-selected logistic mix ----
        sel = (phase << 8) | prev_byte
        w = weights[sel]
        dot = 0.0
        for k in range(NIN + 1):
            dot += w[k] * sts[k]
        P0 = squash(dot)

        # ---- SSE / APM chain ----
        a1ctx = (prev_byte << 3) | phase
        Pa = apm1.refine(P0, a1ctx)
        # blend a touch of original to be safe early on
        Pa = 0.30 * P0 + 0.70 * Pa
        a2ctx = ((prev_byte * 769 + prev2_byte * 31 + phase) & 1023)
        Pb = apm2.refine(Pa, a2ctx)
        P = 0.30 * Pa + 0.70 * Pb

        if P < 1e-6:
            P = 1e-6
        elif P > 1 - 1e-6:
            P = 1 - 1e-6

        # ---- cost ----
        y = bits[i]
        cost = -(math.log(P if y == 1 else 1.0 - P) / log2)
        tot += cost
        if i >= split:
            tail += cost
            tailn += 1

        # ---- updates ----
        err = y - P0   # train mixer on its own output (pre-SSE) -- standard
        for k in range(NIN + 1):
            w[k] += lr * err * sts[k]
        for k in range(NM):
            cells[k][y] += 1
        sp_cell[y] += 1
        wd_cell[y] += 1
        apm1.update(y)
        apm2.update(y)

        # ---- advance bit/byte state ----
        cur = (cur << 1) | y
        if phase == 7:
            b = cur & 0xFF
            hist_bytes.append(b)
            # match model update
            match.update_after_byte(hist_bytes, byte_pos)
            # word hash: letters extend the word, non-letters reset it
            if (65 <= b <= 90) or (97 <= b <= 122):
                lc = b | 0x20
                word_hash = (word_hash * 131 + lc) & 0xFFFFFFF
            else:
                word_hash = 0
            prev2_byte = prev_byte
            prev_byte = b
            cur = 0
            byte_pos += 1

    return tot / n, (tail / tailn if tailn else 0.0)


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.txt"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    raw, bits = load_bits(path, cap)
    g = len(gzip.compress(raw, 9))
    whole, tail = run(bits, raw)
    print(f"corpus={path}  bytes={len(raw)}  bits={len(bits)}")
    print(f"  mix+sse  whole-stream = {whole:.4f}   last-20% = {tail:.4f}  bits/bit")
    print(f"  gzip (whole file)     = {g / len(raw):.4f}  bits/bit")


if __name__ == "__main__":
    main()

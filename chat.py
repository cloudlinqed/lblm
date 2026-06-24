#!/usr/bin/env python3
"""
chat.py -- #1 (strong generative core) + #2 (Q/A chat), still 100% bit-level.

The SAME next-bit predictor as the compressor, run in sample mode. Upgrades over talk.py's plain
backoff: ONLINE LOGISTIC CONTEXT MIXING (byte-aware orders mixed by a tiny trained neuron, like
mix.py / lpaq) plus a BYTE MATCH MODEL that copies long coherent spans from the training text --
both give sharper, more fluent generation.  #2 is a thin wrapper: format your input as "Q: ..\nA:"
so the model falls into answer-shape, and STOP at the next "Q:" so it gives one bounded reply.

Nothing about the engine is changed to "chat" -- chatting is the predictor sampling its own next-bit
guesses. The model stays a full predictor/compressor (compression = prediction = generation).

Usage:
  python chat.py --corpus data/chat.txt "How many legs does a spider have?"
  python chat.py --corpus data/chat.txt            # interactive
  python chat.py --mode cont --corpus data/corpus.txt "Once upon a time "   # raw continuation (no Q/A)
"""
import sys, random, math

ORDERS = [0, 1, 2, 3, 4, 5, 6]
NM = len(ORDERS)
I_BIAS = NM            # the mixer is orders + bias; the match is applied separately (see _predict)
NW = NM + 1
DELTA = 0.2
MATCH_GATE = 18       # trust the match only once it is a long (reliable) span, not a spurious 16-mer


def text_to_bits(s):
    out = []
    for b in s.encode("utf-8", "replace"):
        for j in range(7, -1, -1):
            out.append((b >> j) & 1)
    return out


def bits_to_text(bits):
    out = bytearray()
    for k in range(0, len(bits) - 7, 8):
        v = 0
        for j in range(8):
            v = (v << 1) | bits[k + j]
        out.append(v)
    return out.decode("utf-8", "replace")


def stretch(p):
    p = min(1 - 1e-6, max(1e-6, p)); return math.log(p / (1 - p))


def squash(t):
    if t > 30:
        return 1 - 1e-6
    if t < -30:
        return 1e-6
    return 1.0 / (1.0 + math.exp(-t))


class Core:
    """Online logistic context mixer over byte-aware orders + a byte match model. Train, then sample.

    The match table maps a MINLEN-byte context -> its next position in the corpus byte history. At
    generation the history is `corpus + prompt + generated`, so those positions stay valid and the
    QUESTION's tail recalls the answer that followed it in training (the orders only see ~6 bytes, so
    the match is what carries long-range recall)."""

    def __init__(self, lr=0.02):
        self.tables = [dict() for _ in ORDERS]
        self.w = [0.0] * NW
        self.lr = lr
        self.MINLEN = 16
        self.corpus_bytes = bytearray()   # kept after training; the match reads from this
        self.mtab = {}

    @staticmethod
    def ctx(bits, i, B):
        phase = i & 7
        bstart = i - phase
        return (phase, bytes(bits[bstart:i]), bytes(bits[max(0, bstart - B * 8):bstart]))

    def _match_advance(self, hist, mtab, mptr, mlen):
        n = len(hist)
        if mlen > 0 and mptr < n - 1:
            if hist[mptr] == hist[n - 1]:
                mptr += 1; mlen = min(mlen + 1, 65535)
            else:
                mlen = 0; mptr = -1
        if n >= self.MINLEN:
            key = bytes(hist[n - self.MINLEN:n])
            prev = mtab.get(key, -1)
            mtab[key] = n
            if mlen == 0 and 0 <= prev < n:
                mptr = prev; mlen = self.MINLEN
        return mptr, mlen

    def _predict(self, bits, i, hist, mptr, mlen, learn):
        phase = i & 7
        cur_partial = 0
        if phase:
            for j in range(i - phase, i):
                cur_partial = (cur_partial << 1) | bits[j]
        sts = [0.0] * NW
        cells = [None] * NM
        d = 0.0
        for k, B in enumerate(ORDERS):
            key = self.ctx(bits, i, B)
            c = self.tables[k].get(key)
            if c is None:
                c = [0, 0]
                if learn:
                    self.tables[k][key] = c
            cells[k] = c
            st = stretch((c[1] + DELTA) / (c[0] + c[1] + 2 * DELTA))
            sts[k] = st; d += self.w[k] * st
        sts[I_BIAS] = 1.0; d += self.w[I_BIAS]
        p = squash(d)
        # RECALL: when a long, confident match is active and its partial byte agrees, steer toward the
        # bit it recalls from training (the orders only see ~6 bytes; the match carries the question).
        if (not learn) and mlen >= MATCH_GATE and mptr < len(hist):
            pb = hist[mptr]
            if phase == 0 or (pb >> (8 - phase)) == cur_partial:
                mbit = (pb >> (7 - phase)) & 1
                bw = min(0.97, 0.80 + 0.006 * mlen)        # longer match -> trust it more
                target = 1.0 - 1e-3 if mbit == 1 else 1e-3
                p = bw * target + (1.0 - bw) * p
        return p, sts, cells

    def train(self, bits):
        hist = bytearray(); mtab = self.mtab; mptr = -1; mlen = 0
        for i in range(len(bits)):
            y = bits[i]
            p, sts, cells = self._predict(bits, i, hist, mptr, mlen, learn=True)
            err = y - p
            for k in range(NW):
                self.w[k] += self.lr * err * sts[k]
            for k in range(NM):
                cells[k][y] += 1
            if (i & 7) == 7:
                v = 0
                for j in range(i - 7, i + 1):
                    v = (v << 1) | bits[j]
                hist.append(v)
                mptr, mlen = self._match_advance(hist, mtab, mptr, mlen)
        self.corpus_bytes = hist

    def generate(self, prompt, n_bytes=120, temp=0.7, seed=0, stop=None):
        rng = random.Random(seed)
        bits = text_to_bits(prompt)
        hist = bytearray(self.corpus_bytes)        # corpus + prompt + generated; positions stay valid
        mtab = dict(self.mtab)
        mptr, mlen = -1, 0
        for by in prompt.encode("utf-8", "replace"):
            hist.append(by)
            mptr, mlen = self._match_advance(hist, mtab, mptr, mlen)   # prime: lock onto the question
        out = bytearray()
        for _ in range(n_bytes * 8):
            p, _, _ = self._predict(bits, len(bits), hist, mptr, mlen, learn=False)
            if temp != 1.0:
                a = p ** (1.0 / temp); bb = (1.0 - p) ** (1.0 / temp)
                p = a / (a + bb) if (a + bb) > 0 else 0.5
            bit = 1 if rng.random() < p else 0
            bits.append(bit)
            if (len(bits) & 7) == 0:
                v = 0
                for j in range(len(bits) - 8, len(bits)):
                    v = (v << 1) | bits[j]
                hist.append(v); out.append(v)
                mptr, mlen = self._match_advance(hist, mtab, mptr, mlen)
                txt = out.decode("utf-8", "replace")
                if stop and stop in txt:
                    return txt[:txt.index(stop)]
        return out.decode("utf-8", "replace")


def load_bits(path, cap):
    raw = open(path, "rb").read()
    if cap:
        raw = raw[:cap]
    return text_to_bits_bytes(raw)


def text_to_bits_bytes(raw):
    out = []
    for b in raw:
        for j in range(7, -1, -1):
            out.append((b >> j) & 1)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = sys.argv[1:]
    corpus = "data/chat.txt"; cap = 0; temp = 0.6; gen = 60; mode = "qa"; prompt = None
    i = 0
    while i < len(args):
        a = args[i]
        if a == "--corpus": corpus = args[i + 1]; i += 2
        elif a == "--cap": cap = int(args[i + 1]); i += 2
        elif a == "--temp": temp = float(args[i + 1]); i += 2
        elif a == "--gen": gen = int(args[i + 1]); i += 2
        elif a == "--mode": mode = args[i + 1]; i += 2
        else: prompt = a; i += 1

    print("=" * 78)
    print("BIT-NATIVE CHAT  (strong mixing+match core; English<->bits via dumb adapter)")
    print("=" * 78)
    try:
        raw = open(corpus, "rb").read()
    except FileNotFoundError:
        print(f"corpus {corpus} not found (build it with make_chat_data.py)"); return
    if cap:
        raw = raw[:cap]
    bits = text_to_bits_bytes(raw)
    print(f"training the bit-native core on {len(raw)} bytes ({len(bits)} bits)...", flush=True)
    core = Core()
    core.train(bits)
    print(f"trained. order-6 contexts={len(core.tables[6])}, match keys={len(core.mtab)}. mode={mode} temp={temp}\n")

    def respond(p):
        if mode == "qa":
            wrapped = f"Q: {p}\nA:"
            ans = core.generate(wrapped, n_bytes=gen, temp=temp, seed=0, stop="\nQ:")
            print(f"[you] {p}")
            print(f"[bot]{ans}")
        else:
            cont = core.generate(p, n_bytes=gen, temp=temp, seed=0)
            print(f"[prompt] {p}")
            print(f"[cont]   {p}|{cont}")
        print("-" * 78)

    if prompt is not None:
        respond(prompt)
    else:
        if mode == "qa":
            for p in ["What is 2 + 3?", "What is the capital of France?", "How many days are in a week?"]:
                respond(p)
        print("(type a prompt; Ctrl-C to quit)")
        try:
            while True:
                p = input("\n[you] ")
                if p.strip():
                    respond(p.strip())
        except (KeyboardInterrupt, EOFError):
            print("\nbye.")


if __name__ == "__main__":
    main()

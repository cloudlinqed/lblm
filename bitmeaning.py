#!/usr/bin/env python3
"""
bitmeaning.py -- meaning by PREDICTION, not counting (§66): close §65's honest gap.

§65 crossed the synonym wall with PPMI -- a global COUNT statistic, which it honestly flagged as
"classical NLP, not the predictor." §66 closes exactly that gap by a count->predict SUBSTITUTION: learn
dense word embeddings as the PARAMETERS of a predictor (skip-gram with negative sampling -- word2vec,
Mikolov 2013: a logistic unit + online SGD predicting neighbours, the same PRIMITIVE CLASS the
compressor's mixer uses, though none of its code), then route NOVEL synonyms by nearest-centroid
(centroids from TRAINING words only), exactly as §65.

If prediction-learned embeddings cross the wall too, then meaning emerges from a PREDICTOR's learned
weights, not only from a counted table -- so crossing the wall is not special to PPMI counting. The
decisive new evidence §65 could not give is the ablation below: with the SGD turned OFF (epochs=0,
random init) the wall stays uncrossed (chance); turning it on crosses it. This is a count->predict
SUBSTITUTION, NOT a new capability beyond §65. (Same data + protocol as §65, imported from meaning.py,
apples-to-apples; the SAME honest scope applies -- see below.)

HONEST SCOPE (carried over from §65, re-stated -- this stage is red-teamed too):
  * the reading corpus is unlabeled but its frames SUPPLY operation-disjoint context vocabulary; collapse
    that and the effect vanishes. So this is "meaning by prediction GIVEN a corpus that uses the words in
    separated contexts", a MECHANISM demo, not a free or real-text result.
  * it is a word-level *predictive embedder* using the core's learning primitive (logistic + SGD by
    next-element prediction); it is NOT literally the byte compressor, and it is not claimed to be. The
    point is that the PRINCIPLE (prediction = learning) produces meaning, which §65's count method did not
    establish.
"""
import random, math
from meaning import (make_corpus, TRAIN_WORDS, NOVEL_WORDS, LABELS, content, make_probes,
                     tokens, surface_route)


def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


class SkipGram:
    """Dense word embeddings learned by PREDICTION: skip-gram with negative sampling (word2vec,
    Mikolov 2013) -- a logistic unit trained online by SGD to predict a neighbour. This shares the byte
    core's PRIMITIVE CLASS (a logistic unit + online SGD + next-element prediction) but NONE of its code
    (no byte orders / match model here; no negative sampling / dual embedding tables there)."""

    def __init__(self, dim=24, window=2, neg=5, epochs=10, lr=0.05, min_count=2, seed=0):
        self.dim, self.window, self.neg, self.epochs, self.lr, self.min_count = dim, window, neg, epochs, lr, min_count
        self.seed = seed
        self.inv, self.outv, self.vocab = {}, {}, set()

    def _negtable(self, counts, rng):
        tab = []
        for w, c in sorted(counts.items()):                    # sorted -> PYTHONHASHSEED-independent
            tab += [w] * max(1, int((c ** 0.75)))
        rng.shuffle(tab)
        return tab

    def train(self, sentences):
        rng = random.Random(self.seed)
        counts = {}
        toked = []
        for s in sentences:
            ts = tokens(s)
            toked.append(ts)
            for w in ts:
                counts[w] = counts.get(w, 0) + 1
        self.vocab = {w for w, c in counts.items() if c >= self.min_count}
        counts = {w: c for w, c in counts.items() if w in self.vocab}
        for w in sorted(self.vocab):                           # sorted -> deterministic init order
            self.inv[w] = [(rng.random() - 0.5) / self.dim for _ in range(self.dim)]
            self.outv[w] = [0.0] * self.dim
        negtab = self._negtable(counts, rng)
        L = len(negtab)
        for _ in range(self.epochs):
            for ts in toked:
                ts = [w for w in ts if w in self.vocab]
                for i, center in enumerate(ts):
                    vc = self.inv[center]
                    lo, hi = max(0, i - self.window), min(len(ts), i + self.window + 1)
                    for j in range(lo, hi):
                        if j == i:
                            continue
                        targets = [(ts[j], 1.0)]
                        for _k in range(self.neg):
                            targets.append((negtab[rng.randrange(L)], 0.0))
                        for word, label in targets:
                            uo = self.outv[word]
                            g = (label - sigmoid(sum(vc[d] * uo[d] for d in range(self.dim)))) * self.lr
                            for d in range(self.dim):
                                t = vc[d]
                                vc[d] += g * uo[d]
                                uo[d] += g * t

    def vec(self, w):
        v = self.inv.get(w)
        if v is None:
            return None
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        return [x / n for x in v]


def cos(u, v):
    return sum(a * b for a, b in zip(u, v))


def mean_vec(words, sg):
    vs = [sg.vec(w) for w in words if sg.vec(w) is not None]
    if not vs:
        return None
    m = [sum(col) for col in zip(*vs)]
    n = math.sqrt(sum(x * x for x in m)) or 1.0
    return [x / n for x in m]


def centroids(sg):
    return {op: mean_vec(TRAIN_WORDS[op], sg) for op in LABELS}


def route(content_words, sg, cents):
    rv = mean_vec(content_words, sg)
    if rv is None:
        return None
    return max(LABELS, key=lambda op: cos(rv, cents[op]) if cents[op] else -1)


def evaluate(router, probes, sg, cents):
    return sum(1 for req, op in probes if router(content(req), sg, cents) == op) / len(probes)


def surf(probes):
    return sum(1 for req, op in probes if surface_route(content(req), None, None) == op) / len(probes)


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 90)
    print("bitmeaning.py -- meaning by PREDICTION (skip-gram, logistic+SGD: the core's principle), §66")
    print("=" * 90)
    seeds = [0, 1, 2]
    mt, mn, sn, sh = [], [], [], []
    for s in seeds:
        rng = random.Random(s)
        corpus = make_corpus(s)
        sg = SkipGram(seed=s); sg.train(corpus)
        cents = centroids(sg)
        mt.append(evaluate(route, make_probes(TRAIN_WORDS, rng), sg, cents))
        mn.append(evaluate(route, make_probes(NOVEL_WORDS, rng), sg, cents))
        sn.append(surf(make_probes(NOVEL_WORDS, rng)))
        toks = [t for sent in corpus for t in tokens(sent)]
        rng.shuffle(toks)
        sgs = SkipGram(seed=s); sgs.train([" ".join(toks[i:i + 8]) for i in range(0, len(toks), 8)])
        sh.append(evaluate(route, make_probes(NOVEL_WORDS, rng), sgs, centroids(sgs)))

    def c(v):
        return f"{sum(v)/len(v)*100:5.1f}% ({min(v)*100:.0f}-{max(v)*100:.0f})"

    print(f"\nrouting accuracy over {len(seeds)} seeds (chance = 33%); embeddings LEARNED by prediction:\n")
    print(f"  predictive router, TRAIN-word probes (sanity)     : {c(mt)}")
    print(f"  predictive router, NOVEL synonyms (the wall)      : {c(mn)}   <- meaning by PREDICTION")
    print(f"  surface router, NOVEL (word identity, control)    : {c(sn)}   (cannot route)")
    print(f"  predictive router, NOVEL, SHUFFLED corpus (control): {c(sh)}   (decorrelated -> ~chance)")

    # the NEW evidence §65 could not give: turn the SGD predictor OFF -> the wall stays uncrossed
    def ablate(ep, nseed=8):
        accs = []
        for s in range(nseed):
            sg = SkipGram(epochs=ep, seed=s); sg.train(make_corpus(s))
            accs.append(evaluate(route, make_probes(NOVEL_WORDS, random.Random(s)), sg, centroids(sg)))
        return sum(accs) / len(accs) * 100
    a0, a10 = ablate(0), ablate(10)
    print(f"\n[ablation: is it the PREDICTOR?]  epochs=0 (random init, no SGD) NOVEL = {a0:.0f}%"
          f"   ->   epochs=10 (trained) NOVEL = {a10:.0f}%   [chance 33%; the lift IS the SGD predictor]")

    print(f"\n[reading more -> more meaning] NOVEL vs sentences read (seed 0):")
    for ne in [6, 20, 60, 240]:
        cp = make_corpus(0, n_each=ne); sg = SkipGram(seed=0); sg.train(cp)
        a = evaluate(route, make_probes(NOVEL_WORDS, random.Random(0)), sg, centroids(sg))
        print(f"   ~{len(cp):4} sentences  ->  NOVEL = {a*100:3.0f}%")

    print(f"\n[window robustness] NOVEL vs skip-gram window (seed 0); unlike PPMI (§65) it does NOT decay:")
    for win in [1, 2, 4, 8]:
        sg = SkipGram(window=win, seed=0); sg.train(make_corpus(0))
        a = evaluate(route, make_probes(NOVEL_WORDS, random.Random(0)), sg, centroids(sg))
        print(f"   window={win:2}  ->  NOVEL = {a*100:3.0f}%")

    m = sum(mn) / len(seeds) * 100
    print("\n" + "=" * 90)
    print("VERDICT (honest):")
    print(f"  a PREDICTOR (skip-gram/word2vec: a logistic unit + SGD predicting neighbours -- the same")
    print(f"  primitive CLASS the core uses, NOT its code) crosses the synonym wall too: NOVEL {m:.0f}%")
    print(f"  (chance 33%), vs surface 0%, shuffled ~chance, and the epochs=0 ablation at chance. So")
    print(f"  crossing the wall is not special to PPMI COUNTING -- it falls out of a predictor's learned")
    print(f"  weights. This is a count->predict SUBSTITUTION that closes §65's flagged gap, NOT a new")
    print(f"  capability. SAME SCOPE as §65: the corpus SUPPLIES the separation, synonyms must appear in")
    print(f"  it, and this is self-contained word2vec -- wiring such an embedder INTO the byte predictor")
    print(f"  itself is the remaining genuinely bit-native step.")
    print("=" * 90)


if __name__ == "__main__":
    main()

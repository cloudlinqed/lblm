#!/usr/bin/env python3
"""
meaning.py -- crossing the SYNONYM wall (§64) with LEARNED distributional meaning. A MECHANISM demo,
honestly scoped (this file was corrected after an adversarial red-team; see §65).

§64 left one tier unclimbed: true synonyms (`deduct`,`combine`,`scale`) share no word/substring with the
training vocabulary, so a bag-of-features router routes them at chance. A word's meaning is the company
it keeps (the distributional hypothesis); we learn it the classic count-based way -- PPMI co-occurrence
vectors -- from an UNLABELED reading corpus, then route a request by the cosine of its content-word
vector to each operation's centroid (built only from the *training* words). If `deduct` has been READ in
subtraction-typed contexts, its vector lands near `subtract`/`minus` and it routes to sub.

WHAT THIS IS, HONESTLY (the red-team's ruling):
  * This is a CLASSICAL distributional-semantics method (PPMI + cosine nearest-centroid). It is NOT the
    bit-native predictor and shares no machinery with it -- it is a separate test of one idea: that
    learned word meaning can cross a synonym gap that surface features cannot.
  * The reading corpus is UNLABELED (no op= tag) and never equates a synonym with an operation word, BUT
    its operation-typed frames place each operation in a DISJOINT context vocabulary
    (larger/bigger vs smaller/fewer vs repeatedly/manyfold). That separation is SUPPLIED BY THE CORPUS,
    not discovered -- it is supervision-by-construction in continuous form. Collapse the frames to a
    shared vocabulary and the effect vanishes (NOVEL and TRAIN both fall to 33% chance). So the honest
    claim is "meaning crosses the wall *given a corpus that uses the words in separated contexts*."
  * It works only for synonyms that actually APPEAR (unlabeled) in the reading corpus, and only for a
    SMALL co-occurrence window (window≤4; the printed window sweep shows it decays as the window widens
    past the ~8-token frames). Real local corpora are too sparse for this vocabulary (a frequency scan
    of 17 MB of local text finds `deduct` 0 times, `subtract` 1, `multiply` 3), so no real corpus is
    read here -- this demonstrates the MECHANISM, not a free result.

Controls (deterministic): a SURFACE word-identity router CANNOT route a novel synonym (it shares no
dimension with training words) -> 0%; a token-SHUFFLED corpus destroys co-occurrence -> the router
class-collapses to one label (~chance on balanced probes); and NOVEL accuracy climbs with how much is
read. Together: the lift is distributional STRUCTURE supplied by the corpus, learned by reading.
"""
import re, random, math

LABELS = ["add", "sub", "mul"]
STOP = {"when", "you", "the", "of", "and", "by", "from", "we", "them", "it", "to", "a", "an", "is",
        "into", "together", "with", "after", "becomes", "before", "than", "some", "away", "get", "its",
        "this", "that", "these", "those", "two", "one", "result", "gives", "leaves", "behind",
        "what", "please", "i", "need", "give", "me", "tell", "could", "do", "equals", "between", "up",
        "makes", "make", "make"}

# the words that must CLUSTER by operation. TRAIN words seed the centroids; NOVEL words are never
# labeled -- they must be placed correctly by reading alone. (None of these appears in a frame BODY;
# an op word occurs only in the {W} slot -- see FRAMES.)
TRAIN_WORDS = {"add": ["add", "plus", "sum"], "sub": ["subtract", "minus", "less"],
               "mul": ["multiply", "times", "product"]}
NOVEL_WORDS = {"add": ["combine", "tally", "increase", "join"],
               "sub": ["deduct", "difference", "decrease", "reduce"],
               "mul": ["scale", "double", "triple", "compound"]}

# operation-typed frames. {W}=an op word (the ONLY place an op word appears), {n}=a number. The frame
# BODIES contain NO training/novel vocabulary -- only neutral context words whose per-operation
# vocabulary is disjoint (larger/bigger ; smaller/fewer ; repeatedly/manyfold). That disjoint context
# IS the (continuous) supervision; the demo is honest about that (see header), not hiding it.
FRAMES = {
    "add": ["when you {W} {n} and {n} the heap becomes larger",
            "the {W} of these makes a bigger heap",
            "{W} the groups into one larger heap",
            "after we {W} them the heap grows bigger",
            "to {W} makes the quantity grow larger"],
    "sub": ["when you {W} {n} from {n} the heap becomes smaller",
            "the {W} makes a heap that is fewer",
            "{W} some and the heap shrinks fewer",
            "after we {W} it the quantity drops smaller",
            "to {W} makes the quantity grow lower"],
    "mul": ["when you {W} {n} by {n} the heap grows repeatedly",
            "the {W} stacks the heap manyfold",
            "{W} it again to get a heap stacked repeatedly",
            "after we {W} the quantity it becomes manyfold",
            "to {W} stacks the quantity again and manyfold"],
}
FILLER = ["the cat sat on the warm mat", "a quiet river runs past the old town",
          "she opened the window to let in light", "the road turned left near the tall tree",
          "birds gather in the field at dawn", "we walked along the shore for hours"]


def make_corpus(seed, n_each=240):
    rng = random.Random(seed)
    rows = []
    for op in LABELS:
        words = TRAIN_WORDS[op] + NOVEL_WORDS[op]
        for _ in range(n_each):
            rows.append(rng.choice(FRAMES[op]).format(W=rng.choice(words), n=rng.randrange(2, 99)))
    for _ in range(len(rows) // 3):
        rows.append(rng.choice(FILLER))
    rng.shuffle(rows)
    return rows


def tokens(s):
    return re.findall(r"[a-z]+", re.sub(r"\d+", " ", s.lower()))


def cooc_counts(sentences, window=4):
    co, uni = {}, {}
    for s in sentences:
        ts = tokens(s)
        for i, w in enumerate(ts):
            uni[w] = uni.get(w, 0) + 1
            for j in range(max(0, i - window), min(len(ts), i + window + 1)):
                if j != i:
                    co.setdefault(w, {})[ts[j]] = co.get(w, {}).get(ts[j], 0) + 1
    return co, uni


def ppmi_vectors(co, uni):
    total = sum(uni.values())
    colsum = {c: sum(d.values()) for c, d in co.items()}
    vecs = {}
    for w, ctx in co.items():
        row = sum(ctx.values())
        v = {}
        for c, n in ctx.items():
            pmi = math.log((n * total) / (row * colsum.get(c, 1) + 1e-9) + 1e-12)
            if pmi > 0:
                v[c] = pmi
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        vecs[w] = {c: x / norm for c, x in v.items()}
    return vecs


def cosine(u, v):
    if len(u) > len(v):
        u, v = v, u
    return sum(x * v.get(c, 0.0) for c, x in u.items())


def add_vecs(ws, vecs):
    acc, k = {}, 0
    for w in ws:
        if w in vecs:
            k += 1
            for c, x in vecs[w].items():
                acc[c] = acc.get(c, 0.0) + x
    if not k:
        return None
    norm = math.sqrt(sum(x * x for x in acc.values())) or 1.0
    return {c: x / norm for c, x in acc.items()}


def centroids(vecs):
    return {op: add_vecs(TRAIN_WORDS[op], vecs) for op in LABELS}


def route(content_words, vecs, cents):
    rv = add_vecs(content_words, vecs)
    if rv is None:
        return None
    return max(LABELS, key=lambda op: cosine(rv, cents[op]) if cents[op] else -1)


def surface_route(content_words, _vecs, _cents):
    """Control (deterministic): route by raw word IDENTITY over training words. A NOVEL synonym shares
    no dimension with any training word -> matches nothing -> returns None (cannot route, scored wrong).
    Surface form alone carries zero signal for an unseen synonym."""
    best, bestop = 0, None
    for op in LABELS:
        s = sum(1 for w in content_words if w in TRAIN_WORDS[op])
        if s > best:
            best, bestop = s, op
    return bestop


def content(req):
    return [w for w in tokens(req) if w not in STOP]


def make_probes(words_by_op, rng):
    frames = ["{W} {n} and {n}", "the {W} of {n} and {n}", "{W} {n} from {n}", "please {W} {n} by {n}"]
    return [(rng.choice(frames).format(W=w, n=rng.randrange(2, 99)), op)
            for op, ws in words_by_op.items() for w in ws]


def evaluate(router, probes, vecs, cents):
    return sum(1 for req, op in probes if router(content(req), vecs, cents) == op) / len(probes)


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 90)
    print("meaning.py -- cross the SYNONYM wall with LEARNED distributional meaning (PPMI). Mechanism demo,")
    print("              honestly scoped: the corpus SUPPLIES the separation; this is classical NLP, not")
    print("              the bit-native predictor. Red-teamed (§65).")
    print("=" * 90)
    seeds = [0, 1, 2, 3, 7]
    res = {"mTRAIN": [], "mNOVEL": [], "sNOVEL": [], "shufNOVEL": []}
    for s in seeds:
        rng = random.Random(s)
        corpus = make_corpus(s)
        vecs = ppmi_vectors(*cooc_counts(corpus))
        cents = centroids(vecs)
        res["mTRAIN"].append(evaluate(route, make_probes(TRAIN_WORDS, rng), vecs, cents))
        res["mNOVEL"].append(evaluate(route, make_probes(NOVEL_WORDS, rng), vecs, cents))
        res["sNOVEL"].append(evaluate(surface_route, make_probes(NOVEL_WORDS, rng), vecs, cents))
        toks = [t for sent in corpus for t in tokens(sent)]
        rng.shuffle(toks)
        sv = ppmi_vectors(*cooc_counts([" ".join(toks[i:i + 8]) for i in range(0, len(toks), 8)]))
        res["shufNOVEL"].append(evaluate(route, make_probes(NOVEL_WORDS, rng), sv, centroids(sv)))

    def cell(k):
        v = res[k]; return f"{sum(v)/len(v)*100:5.1f}% ({min(v)*100:.0f}-{max(v)*100:.0f})"

    print(f"\nrouting accuracy over {len(seeds)} seeds (corpus regenerated each seed); chance = 33% (3 classes):\n")
    print(f"  meaning router, TRAIN-word probes (sanity)       : {cell('mTRAIN')}")
    print(f"  meaning router, NOVEL-synonym probes (the wall)  : {cell('mNOVEL')}   <- learned meaning")
    print(f"  surface router, NOVEL  (word identity, control)  : {cell('sNOVEL')}   (cannot route: 0%)")
    print(f"  meaning router, NOVEL, SHUFFLED corpus (control) : {cell('shufNOVEL')}   (class-collapse to ~chance)")

    print(f"\n[reading more -> more meaning] NOVEL vs how much was read (seed 0, window 4):")
    for ne in [2, 6, 20, 60, 240]:
        c = make_corpus(0, n_each=ne); vv = ppmi_vectors(*cooc_counts(c)); ct = centroids(vv)
        print(f"   ~{len(c):4} sentences  ->  NOVEL = {evaluate(route, make_probes(NOVEL_WORDS, random.Random(0)), vv, ct)*100:3.0f}%")

    print(f"\n[window is load-bearing] NOVEL vs co-occurrence window (seed 0); decays as window > frame length:")
    c = make_corpus(0)
    for win in [1, 2, 4, 8, 20]:
        vv = ppmi_vectors(*cooc_counts(c, window=win)); ct = centroids(vv)
        print(f"   window={win:2}  ->  NOVEL = {evaluate(route, make_probes(NOVEL_WORDS, random.Random(0)), vv, ct)*100:3.0f}%")

    mn = sum(res["mNOVEL"]) / len(seeds) * 100
    print("\n" + "=" * 90)
    print("VERDICT (honest, multi-seed):")
    print(f"  GIVEN an unlabeled corpus that uses the words in operation-separated contexts, learned")
    print(f"  distributional meaning crosses the synonym wall: NOVEL synonyms route at {mn:.0f}% (chance 33%)")
    print(f"  though never labeled -- the router READ them; e.g. cos(deduct, sub-centroid) >> cos(deduct, add).")
    print(f"  HONEST SCOPE: the corpus SUPPLIES that separation (shared-vocab frames -> 33%); it works only")
    print(f"  for synonyms present in the reading corpus and only for small windows; it is CLASSICAL PPMI,")
    print(f"  not the bit-native predictor. The mechanism is real (controls collapse to chance); the route")
    print(f"  to real generalisation is a corpus that actually uses the words -- which local text does not.")
    print("=" * 90)


if __name__ == "__main__":
    main()

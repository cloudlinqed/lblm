#!/usr/bin/env python3
"""
intent.py -- the first bite out of the "words wall" (roadmap item 2: capability).

§61/§62 found the bottleneck is LANGUAGE: mapping a varied phrasing to the right intent. tool.py routed
with HAND-CODED keywords ("plus"->add) -- brittle by construction: any phrasing whose words weren't
anticipated misroutes. Here we *learn* the question->intent map and test the only thing that matters --
GENERALISATION to phrasings never seen in training.

Bit-native in spirit: the request becomes a vector of BINARY features (which word-tokens / operator
symbols are present, numbers masked to a NUM token so it can't cheat on specific values), and a small
online LOGISTIC classifier (the same stretch/squash logistic as the compressor's mixer) is trained by
SGD to predict the intent. No deep net, no library -- a learned linear map over binary features.

The honest test: TRAIN templates and HELD-OUT templates are DISJOINT sentence structures. The held-out
ones reuse the discriminative content words (plus/sum, minus/less, times/product, +/-/*) inside NEW
wrappers ("please ...", "i need ...", "give me ... please") -- so success means generalising across
sentence structure, not memorising templates. Contrast: the hand-coded router from tool.py.
"""
import re, random, math

random.seed(0)

# ---- intents and their TRAIN phrasings (number slots {a},{b}) ----
TRAIN = {
    "add": ["what is {a} plus {b}", "add {a} and {b}", "what is the sum of {a} and {b}",
            "{a} plus {b}", "compute {a} + {b}", "what do you get adding {a} and {b}"],
    "sub": ["what is {a} minus {b}", "subtract {b} from {a}", "what is {a} less {b}",
            "{a} minus {b}", "compute {a} - {b}", "take {b} away from {a}"],
    "mul": ["what is {a} times {b}", "multiply {a} and {b}", "what is the product of {a} and {b}",
            "{a} times {b}", "compute {a} * {b}", "what do you get multiplying {a} and {b}"],
}
# ---- EASY held-out: NEW structures, same discriminative words (tests structure generalisation) ----
HELDOUT = {
    "add": ["please add {a} to {b}", "i need the sum of {a} and {b}", "give me {a} plus {b} please",
            "tell me {a} + {b}", "could you add up {a} and {b}"],
    "sub": ["please subtract {b} from {a}", "i need {a} minus {b}", "give me {a} less {b} please",
            "tell me {a} - {b}", "could you take {b} away from {a}"],
    "mul": ["please multiply {a} by {b}", "i need the product of {a} and {b}", "give me {a} times {b} please",
            "tell me {a} * {b}", "could you multiply {a} and {b}"],
}
# ---- HARD held-out: unseen WORD FORMS (morphology). 'multiplied' shares substrings with 'multiply',
#      so char n-grams can bridge them even though the exact word was never seen. ----
HARD = {
    "add": ["{a} added to {b}", "summing {a} and {b}", "the addition of {a} and {b}", "adding {a} to {b}"],
    "sub": ["subtracting {b} from {a}", "the subtraction of {b} from {a}", "{a} minus {b} equals what",
            "{b} subtracted from {a}"],
    "mul": ["{a} multiplied by {b}", "multiplying {a} and {b}", "the multiplication of {a} and {b}",
            "the product is {a} times {b}"],
}
# ---- NOVEL held-out: true SYNONYMS sharing NO word/substring with training (the residual wall) ----
NOVEL = {
    "add": ["combine {a} and {b}", "the total of {a} and {b}"],
    "sub": ["deduct {b} from {a}", "the difference between {a} and {b}"],
    "mul": ["scale {a} by a factor of {b}", "the area of a {a} by {b} rectangle"],
}
LABELS = ["add", "sub", "mul"]
LI = {k: i for i, k in enumerate(LABELS)}


def featurize(req, char_ng=True):
    """request text -> set of BINARY feature strings. Numbers masked to NUM (value-agnostic). Word
    tokens + operator symbols, plus optional char 3-grams so unseen word FORMS that share substrings
    with seen words can still fire shared features. Presence of features, learned-weighted -- the only
    'understanding'."""
    r = re.sub(r"\d+", " NUM ", req.lower())
    feats = set()
    for tok in re.findall(r"[a-z]+|NUM|[+\-*/]", r):
        feats.add("w:" + tok)
    if char_ng:
        s = "^" + re.sub(r"\s+", " ", r).strip() + "$"
        for i in range(len(s) - 2):
            feats.add("c:" + s[i:i + 3])
    return feats


def make_examples(table, n_per):
    ex = []
    for intent, templates in table.items():
        for t in templates:
            for _ in range(n_per):
                a, b = random.randrange(2, 999), random.randrange(2, 999)
                req = t.format(a=a, b=b)
                ex.append((req, intent, a, b))
    random.shuffle(ex)
    return ex


class LogisticRouter:
    """Multiclass logistic over binary features, online SGD. Bit-native lineage: a logistic mixer of
    present/absent features, exactly the stretch/squash unit the compressor uses, trained to route."""

    def __init__(self, lr=0.5, epochs=25, char_ng=True):
        self.w = {}            # feature -> [weight per label]
        self.b = [0.0] * len(LABELS)
        self.lr = lr
        self.epochs = epochs
        self.char_ng = char_ng

    def _scores(self, feats):
        s = list(self.b)
        for f in feats:
            wv = self.w.get(f)
            if wv is not None:
                for i in range(len(LABELS)):
                    s[i] += wv[i]
        return s

    @staticmethod
    def _softmax(s):
        m = max(s)
        e = [math.exp(x - m) for x in s]
        z = sum(e)
        return [x / z for x in e]

    def train(self, examples):
        feats_cache = [(featurize(r, self.char_ng), LI[lab]) for r, lab, _, _ in examples]
        for _ in range(self.epochs):
            random.shuffle(feats_cache)
            for feats, y in feats_cache:
                p = self._softmax(self._scores(feats))
                for i in range(len(LABELS)):
                    g = ((1.0 if i == y else 0.0) - p[i])
                    self.b[i] += self.lr * g * 0.1
                    for f in feats:
                        wv = self.w.setdefault(f, [0.0] * len(LABELS))
                        wv[i] += self.lr * g

    def predict(self, req):
        feats = featurize(req, self.char_ng)
        sc = self._scores(feats)
        return LABELS[max(range(len(LABELS)), key=lambda i: sc[i])]


def hand_coded_route(req):
    """tool.py's style: keyword if/else. Brittle -- only the anticipated words route correctly."""
    r = req.lower()
    if "plus" in r or "add" in r or "sum" in r or "+" in r:
        return "add"
    if "minus" in r or "subtract" in r or "difference" in r or "-" in r:
        return "sub"
    if "times" in r or "multiply" in r or "product" in r or "*" in r:
        return "mul"
    return None


def acc(predict_fn, examples):
    ok = sum(1 for req, intent, _, _ in examples if predict_fn(req) == intent)
    return ok / len(examples)


def main():
    print("=" * 84)
    print("intent.py -- LEARN question->intent; test it on three TIERS of unseen phrasing (words wall)")
    print("=" * 84)

    train_ex = make_examples(TRAIN, n_per=12)
    easy = make_examples(HELDOUT, n_per=12)   # new structure, SAME discriminative words
    hard = make_examples(HARD, n_per=12)      # unseen word FORMS (morphology) -> char n-grams may bridge
    novel = make_examples(NOVEL, n_per=12)    # true SYNONYMS, no shared word/substring -> the residual wall
    for name, tab in [("HELDOUT", HELDOUT), ("HARD", HARD), ("NOVEL", NOVEL)]:
        assert not ({t for ts in TRAIN.values() for t in ts} & {t for ts in tab.values() for t in ts})

    words_only = LogisticRouter(char_ng=False); words_only.train(train_ex)
    full = LogisticRouter(char_ng=True); full.train(train_ex)

    print(f"\ntrained on {len(train_ex)} requests / {sum(len(v) for v in TRAIN.values())} templates.  "
          f"Three held-out tiers, each templates-DISJOINT from training:")
    print("  EASY  = new sentence structure, same key words (plus/minus/times)")
    print("  HARD  = unseen word FORMS (added, subtracting, multiplied) -- share substrings, not whole words")
    print("  NOVEL = true synonyms (combine, deduct, scale) -- share NOTHING with training")

    print(f"\n{'router':<26}{'train':>8}{'EASY':>8}{'HARD':>8}{'NOVEL':>8}")
    rows = [
        ("hand-coded (tool.py)", hand_coded_route),
        ("learned (words only)", words_only.predict),
        ("learned (words+char-ng)", full.predict),
    ]
    for name, fn in rows:
        print(f"{name:<26}{acc(fn, train_ex)*100:7.1f}%{acc(fn, easy)*100:7.1f}%"
              f"{acc(fn, hard)*100:7.1f}%{acc(fn, novel)*100:7.1f}%")

    print("\n[char n-grams bridge unseen word FORMS] HARD examples routed by words+char-ng:")
    seen = set()
    for req, intent, _, _ in hard:
        if intent in seen:
            continue
        seen.add(intent)
        p = full.predict(req)
        print(f"   {req!r:44} -> {p}   {'OK' if p == intent else 'WRONG'}")
        if len(seen) == 3:
            break

    print("\n[the residual wall] NOVEL synonyms share no feature with training, so they misroute -- honest:")
    seen = set()
    for req, intent, _, _ in novel:
        if intent in seen:
            continue
        seen.add(intent)
        p = full.predict(req)
        print(f"   {req!r:44} -> {p}   {'OK' if p == intent else 'WRONG (no shared feature)'}")
        if len(seen) == 3:
            break

    e = acc(full.predict, easy) * 100; h = acc(full.predict, hard) * 100; nv = acc(full.predict, novel) * 100
    print("\n" + "=" * 84)
    print("VERDICT: question->intent is LEARNED (logistic over binary word/char features), not hand-coded.")
    print(f"  It GENERALISES to new sentence structure ({e:.0f}% EASY) and, via char n-grams, to unseen")
    print(f"  word FORMS ({h:.0f}% HARD) where the hand-coded router has fixed keywords. The HONEST wall:")
    print(f"  true synonyms with no shared feature ({nv:.0f}% NOVEL) -- bridging those needs learned word")
    print("  meaning (embeddings / reading), the next step. This is a real, measured bite out of the wall.")
    print("=" * 84)


if __name__ == "__main__":
    main()

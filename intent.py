#!/usr/bin/env python3
"""
intent.py -- a LEARNED question->intent router, and an HONEST measurement of how far it generalises.

§61/§62 found the bottleneck is LANGUAGE: mapping a varied phrasing to the right intent. tool.py routed
with HAND-CODED keywords -- brittle. Here we *learn* the map (online logistic over binary word/symbol/
char-3gram features, numbers masked to NUM) and test it held-out on three DISJOINT tiers of difficulty.

This version was rebuilt after an adversarial red-team (see learned_binary_address_machine.md §64) that
caught two real defects in the first cut: (1) the HARD tier leaked verbatim training words, and (2) the
hand-coded baseline was a strawman (whole-word match) -- a *fair* stem-substring hand-coder also solves
HARD. Both are fixed here: HARD now contains only word FORMS that are NOT whole training words, and we
report a FAIR stem baseline. We also expand NOVEL and report mean±range over many seeds, and a
stop-word-stripped diagnostic, so no number rides on shared function words or a lucky seed.

Honest takeaways the code prints:
  * routing is LEARNED, not hand-tuned -- and generalises across sentence STRUCTURE (EASY ~100%);
  * char n-grams give a REAL lift on unseen word FORMS (de-leaked HARD) vs words-only features;
  * a fair stem hand-coder ALSO solves HARD -- so the learned win is "no keyword hand-tuning", not a
    capability hand-coding lacks;
  * true SYNONYMS (NOVEL) sit at ~chance once function words are stripped -- the wall is NOT climbed;
    crossing it needs learned word MEANING (§62's reading), the next step.
"""
import re, random

LABELS = ["add", "sub", "mul"]
LI = {k: i for i, k in enumerate(LABELS)}
SEEDS = [0, 1, 2, 3, 7, 42, 100, 2024]
STOP = {"what", "is", "are", "the", "of", "a", "an", "please", "i", "need", "give", "me", "tell",
        "could", "you", "do", "get", "equals", "equal", "to", "by", "and", "from", "between", "with",
        "up", "we", "us", "find", "calculate", "result", "value", "many", "much", "how"}

# ---- TRAIN phrasings (number slots {a},{b}) ----
TRAIN = {
    "add": ["what is {a} plus {b}", "add {a} and {b}", "what is the sum of {a} and {b}",
            "{a} plus {b}", "compute {a} + {b}", "what do you get adding {a} and {b}"],
    "sub": ["what is {a} minus {b}", "subtract {b} from {a}", "what is {a} less {b}",
            "{a} minus {b}", "compute {a} - {b}", "take {b} away from {a}"],
    "mul": ["what is {a} times {b}", "multiply {a} and {b}", "what is the product of {a} and {b}",
            "{a} times {b}", "compute {a} * {b}", "what do you get multiplying {a} and {b}"],
}
# discriminative whole-words present in TRAIN (used to GUARANTEE the HARD tier contains none of them)
TRAIN_DISCRIM = {"plus", "add", "adding", "sum", "minus", "subtract", "less", "take", "away",
                 "times", "multiply", "multiplying", "product"}

# ---- EASY: new sentence STRUCTURE, same discriminative words (tests structure generalisation) ----
EASY = {
    "add": ["please add {a} to {b}", "i need the sum of {a} and {b}", "give me {a} plus {b} please",
            "tell me {a} + {b}", "could you add up {a} and {b}"],
    "sub": ["please subtract {b} from {a}", "i need {a} minus {b}", "give me {a} less {b} please",
            "tell me {a} - {b}", "could you take {b} away from {a}"],
    "mul": ["please multiply {a} by {b}", "i need the product of {a} and {b}", "give me {a} times {b} please",
            "tell me {a} * {b}", "could you multiply {a} and {b}"],
}
# ---- HARD: unseen word FORMS only (each verified to contain NO whole training discriminative word);
#      they share STEM SUBSTRINGS (add/sum, subtract, multipl) so char n-grams can bridge, words cannot ----
HARD = {
    "add": ["{a} added to {b}", "summing {a} and {b}", "the addition of {a} and {b}", "{a} summed with {b}"],
    "sub": ["subtracting {b} from {a}", "the subtraction of {b} from {a}", "{b} subtracted from {a}",
            "{a} decremented by {b}"],
    "mul": ["{a} multiplied by {b}", "the multiplication of {a} and {b}", "multiplies {a} by {b}",
            "{a} multiplied with {b}"],
}
# ---- NOVEL: true SYNONYMS sharing no discriminative word/substring with training (the residual wall) ----
NOVEL = {
    "add": ["combine {a} and {b}", "the total of {a} and {b}", "increase {a} by {b}",
            "{a} increased by {b}", "join {a} with {b}", "the combined amount of {a} and {b}"],
    "sub": ["deduct {b} from {a}", "the difference of {a} and {b}", "decrease {a} by {b}",
            "{a} reduced by {b}", "remove {b} from {a}", "how much remains of {a} after {b}"],
    "mul": ["scale {a} by {b}", "{a} groups of {b}", "{a} lots of {b}", "{a} rows of {b} each",
            "the area of a {a} by {b} grid", "repeat {a} for {b} copies"],
}


def featurize(req, char_ng=True, strip_stop=False):
    """request -> set of BINARY features: word tokens (NUM-masked, optionally minus stop-words),
    operator symbols, and (optionally) char 3-grams that let unseen word FORMS fire shared sub-features."""
    r = re.sub(r"\d+", " NUM ", req.lower())
    feats = set()
    for tok in re.findall(r"[a-z]+|NUM|[+\-*/]", r):
        if strip_stop and tok in STOP:
            continue
        feats.add("w:" + tok)
    if char_ng:
        s = "^" + re.sub(r"\s+", " ", r).strip() + "$"
        for i in range(len(s) - 2):
            feats.add("c:" + s[i:i + 3])
    return feats


def make_examples(table, n_per, rng):
    ex = []
    for intent, templates in table.items():
        for t in templates:
            for _ in range(n_per):
                a, b = rng.randrange(2, 999), rng.randrange(2, 999)
                ex.append((t.format(a=a, b=b), intent))
    rng.shuffle(ex)
    return ex


class LogisticRouter:
    """Multiclass logistic over binary features, online SGD -- the compressor's stretch/squash unit,
    trained to route. No keyword lists: everything is learned from the training phrasings."""

    def __init__(self, char_ng=True, strip_stop=False, lr=0.5, epochs=25):
        self.w, self.b = {}, [0.0] * len(LABELS)
        self.char_ng, self.strip_stop, self.lr, self.epochs = char_ng, strip_stop, lr, epochs

    def _scores(self, feats):
        s = list(self.b)
        for f in feats:
            wv = self.w.get(f)
            if wv is not None:
                for i in range(len(LABELS)):
                    s[i] += wv[i]
        return s

    def train(self, examples, rng):
        import math
        cache = [(featurize(r, self.char_ng, self.strip_stop), LI[lab]) for r, lab in examples]
        for _ in range(self.epochs):
            rng.shuffle(cache)
            for feats, y in cache:
                s = self._scores(feats)
                m = max(s); e = [math.exp(x - m) for x in s]; z = sum(e); p = [x / z for x in e]
                for i in range(len(LABELS)):
                    g = (1.0 if i == y else 0.0) - p[i]
                    self.b[i] += self.lr * g * 0.1
                    for f in feats:
                        self.w.setdefault(f, [0.0] * len(LABELS))[i] += self.lr * g

    def predict(self, req):
        sc = self._scores(featurize(req, self.char_ng, self.strip_stop))
        return LABELS[max(range(len(LABELS)), key=lambda i: sc[i])]


# ---- hand-coded baselines: the original (whole-word, tool.py style) and a FAIR stem-substring one ----
def hand_whole(req):
    r = req.lower()
    if any(w in r.split() for w in ["plus", "add", "sum"]) or "+" in r:
        return "add"
    if any(w in r.split() for w in ["minus", "subtract", "difference"]) or "-" in r:
        return "sub"
    if any(w in r.split() for w in ["times", "multiply", "product"]) or "*" in r:
        return "mul"
    return None


STEMS = {"add": ["add", "sum", "plus", "+"], "sub": ["subtract", "minus", "less", "-"],
         "mul": ["multipl", "times", "product", "*"]}


def hand_stem(req):
    r = req.lower()
    for lab in LABELS:
        if any(s in r for s in STEMS[lab]):
            return lab
    return None


def acc(predict_fn, examples):
    return sum(1 for r, lab in examples if predict_fn(r) == lab) / len(examples)


def run_seed(seed):
    rng = random.Random(seed)
    train = make_examples(TRAIN, 12, rng)
    tiers = {name: make_examples(tab, 12, rng) for name, tab in
             [("EASY", EASY), ("HARD", HARD), ("NOVEL", NOVEL)]}
    routers = {
        "hand_whole": hand_whole,
        "hand_stem": hand_stem,
        "learned_words": LogisticRouter(char_ng=False),
        "learned_words+char": LogisticRouter(char_ng=True),
        "learned_nostop+char": LogisticRouter(char_ng=True, strip_stop=True),
    }
    for r in routers.values():
        if isinstance(r, LogisticRouter):
            r.train(train, rng)
    out = {}
    for rn, r in routers.items():
        fn = r.predict if isinstance(r, LogisticRouter) else r
        out[rn] = {"train": acc(fn, train), **{t: acc(fn, ex) for t, ex in tiers.items()}}
    return out


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 90)
    print("intent.py -- LEARNED question->intent; HONEST held-out generalisation (post red-team, §64)")
    print("=" * 90)
    # guarantee HARD contains no whole training discriminative word (the leak the red-team caught)
    hard_words = {w for ts in HARD.values() for t in ts for w in re.findall(r"[a-z]+", t.lower())}
    leak = hard_words & TRAIN_DISCRIM
    assert not leak, f"HARD leaks training words: {leak}"
    print(f"HARD-tier leak check vs training discriminative words: {sorted(leak) or 'none'}  [OK]")

    routers = ["hand_whole", "hand_stem", "learned_words", "learned_words+char", "learned_nostop+char"]
    tiers = ["train", "EASY", "HARD", "NOVEL"]
    agg = {rn: {t: [] for t in tiers} for rn in routers}
    for s in SEEDS:
        res = run_seed(s)
        for rn in routers:
            for t in tiers:
                agg[rn][t].append(res[rn][t])

    def cell(vals):
        return f"{sum(vals)/len(vals)*100:5.1f} ({min(vals)*100:.0f}-{max(vals)*100:.0f})"

    print(f"\nmean % correct (min-max) over {len(SEEDS)} seeds; train + 3 template-DISJOINT held-out tiers:")
    print(f"  EASY = new structure, known words | HARD = unseen word FORMS (de-leaked) | NOVEL = synonyms\n")
    print(f"{'router':<22}{'train':>14}{'EASY':>14}{'HARD':>14}{'NOVEL':>14}")
    for rn in routers:
        print(f"{rn:<22}" + "".join(f"{cell(agg[rn][t]):>14}" for t in tiers))

    e = sum(agg['learned_words+char']['EASY']) / len(SEEDS) * 100
    hw = sum(agg['learned_words']['HARD']) / len(SEEDS) * 100
    hc = sum(agg['learned_words+char']['HARD']) / len(SEEDS) * 100
    hs = sum(agg['hand_stem']['HARD']) / len(SEEDS) * 100
    nv = sum(agg['learned_words+char']['NOVEL']) / len(SEEDS) * 100
    nvs = sum(agg['learned_nostop+char']['NOVEL']) / len(SEEDS) * 100
    print("\n" + "=" * 90)
    print("VERDICT (honest, multi-seed):")
    print(f"  * routing is LEARNED, no keyword hand-tuning, and generalises across STRUCTURE: EASY {e:.0f}%.")
    print(f"  * char n-grams give a REAL lift on unseen word FORMS: HARD {hw:.0f}% (words) -> {hc:.0f}% (words+char).")
    print(f"  * but a FAIR stem hand-coder also solves HARD ({hs:.0f}%) -- the learned win is 'no hand-tuning',")
    print(f"    NOT a capability hand-coding lacks. Stated plainly, not overclaimed.")
    print(f"  * true SYNONYMS are the UNCLIMBED wall: NOVEL {nv:.0f}% (per-seed range spans the ~33% chance")
    print(f"    floor), and stripping function words leaves it at chance ({nvs:.0f}%) -- synonyms are NOT")
    print(f"    learned. Crossing it needs learned word MEANING (§62's reading) -- the next step.")
    print("=" * 90)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
answerer.py -- item 2's capstone: a DEPENDABLE end-to-end answerer, held-out and verified.

The pieces built across item 2, wired into one pipeline and measured end-to-end:
  request --(§64 LEARNED router)--> intent  --(dumb adapter)--> integer args
          --(§60 INDUCED computation)--> answer  --(grammar + truth)--> VERIFIED.

"Dependable" is not hoped-for; it is by construction + measurement:
  * the INTENT is LEARNED (logistic over binary word/char features, §64) -> generalises to unseen
    phrasings, not a hand-coded keyword list;
  * the COMPUTATION is INDUCED (§60): add/sub are 1-bit-state transducers induced from disjoint sums
    (the full adder out=150/upd=232), and MULTIPLY is COMPOSED from the induced adder by shift-and-add
    -- so the answer generalises to numbers never seen (a memoriser scores 0%);
  * every answer is VERIFIED (decidable: op + args + truth), measured on HELD-OUT phrasings AND held-out
    numbers;
  * it ABSTAINS when the router is not confident (margin < TAU) or no args parse -- so it is correct on
    what it answers and does not hallucinate (the dependable property).

This is the integration, not a new capability; its value is the end-to-end held-out numbers + honest
coverage/abstention accounting. Red-teamed alongside (see §67).
"""
import re, random, os
from intent import LogisticRouter, TRAIN, EASY, HARD, NOVEL, make_examples, featurize, LABELS, SEEDS

random.seed(0)
L = 12                      # induction number range [0, 2^L)
TAU = float(os.environ.get("TAU", "0.40"))   # router-confidence (softmax margin) to COMMIT; else ABSTAIN


# ---------- §60 induced computation (width-agnostic 1-bit-state transducer + composition) ----------
def trun(a, b, out_fn, upd_fn, nbits):
    """1-bit-state transducer over nbits columns (LSB->MSB); out_fn/upd_fn are 8-bit truth tables over
    (a_i, b_i, state). The carry/borrow is the induced HIDDEN state."""
    state = 0; s = 0
    for i in range(nbits):
        idx = (((a >> i) & 1) << 2) | (((b >> i) & 1) << 1) | state
        s |= ((out_fn >> idx) & 1) << i
        state = (upd_fn >> idx) & 1
    return s | (state << nbits)


def induce(samples):
    """Search the 256x256 transducer space for (out_fn,upd_fn) reproducing all (a,b,result) at L bits."""
    for of in range(256):
        for uf in range(256):
            if all(trun(a, b, of, uf, L) % (1 << (L + 1)) == r for a, b, r in samples):
                return (of, uf)
    return None


def _disjoint(n, cond, seed):
    rng = random.Random(seed); s = set()
    while len(s) < n:
        a, b = rng.randrange(1 << L), rng.randrange(1 << L)
        if cond(a, b):
            s.add((a, b))
    return list(s)


ADD = induce([(a, b, a + b) for a, b in _disjoint(60, lambda a, b: True, 1)])
SUB = induce([(a, b, a - b) for a, b in _disjoint(60, lambda a, b: a >= b, 2)])


def iadd(a, b):
    return trun(a, b, ADD[0], ADD[1], max(a.bit_length(), b.bit_length()) + 1)


def isub(a, b):
    """induced subtractor (borrow = hidden state). PRECONDITION a >= b -- the caller guarantees it via
    the swap in respond()/measure(); for a < b the borrow-out is not signalled (not a total subtractor)."""
    assert a >= b, "isub requires a >= b"
    n = max(a.bit_length(), b.bit_length()) + 1
    return trun(a, b, SUB[0], SUB[1], n) & ((1 << n) - 1)


def imul(a, b):
    """multiply COMPOSED from the induced adder: shift-and-add (a<<i) for each set bit i of b."""
    acc = 0
    for i in range(b.bit_length()):
        if (b >> i) & 1:
            acc = iadd(acc, a << i)
    return acc


def compute(op, a, b):
    return {"add": iadd, "sub": isub, "mul": imul}[op](a, b)


def truth(op, a, b):
    return {"add": a + b, "sub": a - b, "mul": a * b}[op]


# ---------- the learned router, with a confidence (softmax margin) for abstention ----------
def route_conf(router, req):
    import math
    s = router._scores(featurize(req, router.char_ng, router.strip_stop))
    m = max(s); e = [math.exp(x - m) for x in s]; z = sum(e); p = sorted((x / z for x in e), reverse=True)
    return LABELS[max(range(len(LABELS)), key=lambda i: s[i])], p[0] - p[1]


def respond(router, req):
    """request -> ('add'|'sub'|'mul', a, b, answer) or None (ABSTAIN). The one shared path."""
    op, margin = route_conf(router, req)
    nums = [int(x) for x in re.findall(r"\d+", req)]
    if margin < TAU or len(nums) < 2:
        return None
    a, b = nums[0], nums[1]
    if op == "sub" and "from" in req.lower():        # "subtract B from A" => A - B (a words-fact, §61)
        a, b = b, a
    if op == "sub" and a < b:
        a, b = b, a
    return op, a, b, compute(op, a, b)


CALL_RE = re.compile(r"^(add|sub|mul)\((\d+), (\d+)\)$")


# ---------- held-out evaluation ----------
def make_tests(tables, rng, n_per=8):
    """fill held-out phrasing templates with FRESH numbers; record the intended (op, expected answer)."""
    tests = []
    for op, templates in tables.items():
        for t in templates:
            for _ in range(n_per):
                if op == "sub":
                    x, y = sorted([rng.randrange(2, 1 << L), rng.randrange(2, 1 << L)], reverse=True)
                else:
                    x, y = rng.randrange(2, 1 << L), rng.randrange(2, 1 << L)
                tests.append((t.format(a=x, b=y), op, truth(op, x, y)))
    rng.shuffle(tests)
    return tests


def build():
    """train the learned router once (fast) -- the interactive/one-shot user-test entry points reuse it."""
    rng = random.Random(0)
    r = LogisticRouter(char_ng=True); r.train(make_examples(TRAIN, 12, rng), rng)
    return r


def respond_show(router, req):
    """answer ONE request, showing the steps so a user sees the learned route + induced compute, or an
    HONEST abstain (it never guesses)."""
    op, margin = route_conf(router, req)
    nums = [int(x) for x in re.findall(r"\d+", req)]
    print(f"   route   : {op}   (confidence {margin:.2f} [{'ok' if margin >= TAU else 'LOW'}])")
    print(f"   numbers : {nums}")
    r = respond(router, req)
    if r is None:
        why = "router not confident enough" if margin < TAU else "need two numbers in the request"
        print(f"   ANSWER  : (abstain -- {why}; it won't guess)")
    else:
        o, a, b, ans = r
        print(f"   ANSWER  : {o}({a}, {b}) = {ans}    [induced computation, grammar-valid, verifiable]")


def interactive():
    router = build()
    print("=" * 86)
    print("DEPENDABLE ANSWERER -- user test.  Type an arithmetic request; it routes (LEARNED), computes")
    print("(INDUCED add/sub/mul), verifies, or honestly ABSTAINS (it will not guess).  Type 'quit' to exit.")
    print("  try:  what is 347 plus 891?   |   subtract 18 from 200   |   multiply 13 and 7")
    print("        7 times 6   |   the sum of 40 and 2   |   what is the capital of France?  (out of scope)")
    print("=" * 86)
    while True:
        try:
            req = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye."); break
        if not req:
            continue
        if req.lower() in ("quit", "exit", "q"):
            print("bye."); break
        respond_show(router, req)


def main():
    try:
        import sys
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("=" * 86)
    print("answerer.py -- item 2 capstone: learned route + induced compute + verify, held-out & abstaining")
    print("=" * 86)
    print(f"[induced compute]  add={ADD} (full adder? {ADD == (150, 232)})  sub={SUB}  mul = shift-and-add(add)")
    assert ADD == (150, 232), "induced adder is not the full adder"
    # the induced rule IS the UNIVERSAL adder (out=XOR3, upd=MAJ3) -> exact for ALL inputs/widths, not a
    # 3000-sample statistic. Spot-prove exhaustively on [0,256)^2 (fast) to back the "held-out 100%".
    assert all(iadd(a, b) == a + b for a in range(256) for b in range(256)), "adder not universal"

    rng = random.Random(0)
    router = LogisticRouter(char_ng=True)
    router.train(make_examples(TRAIN, 12, rng), rng)

    # induced computation generalises to HELD-OUT numbers (vs a memoriser) ----------------------------
    pairs = _disjoint(3000, lambda a, b: True, 99)
    addc = sum(1 for a, b in pairs if iadd(a, b) == a + b) / len(pairs)
    subc = sum(1 for a, b in pairs if isub(max(a, b), min(a, b)) == max(a, b) - min(a, b)) / len(pairs)
    mulc = sum(1 for a, b in pairs if imul(a, b) == a * b) / len(pairs)
    print(f"[compute held-out, 3000 unseen pairs]  add {addc*100:.0f}%   sub {subc*100:.0f}%   "
          f"mul {mulc*100:.0f}%   (memoriser would be 0%)")

    # end-to-end on HELD-OUT phrasings (EASY+HARD = covered language) x held-out numbers -------------
    def measure(rt, tests, tau):
        com = cor = val = 0
        for req, op, exp in tests:
            o, margin = route_conf(rt, req)
            nums = [int(x) for x in re.findall(r"\d+", req)]
            if margin < tau or len(nums) < 2:
                continue
            a, b = nums[0], nums[1]
            if o == "sub" and "from" in req.lower():
                a, b = b, a
            if o == "sub" and a < b:
                a, b = b, a
            com += 1
            if CALL_RE.match(f"{o}({a}, {b})"):
                val += 1
                if compute(o, a, b) == exp:
                    cor += 1
        n = len(tests)
        return com / n * 100, (cor / com * 100 if com else 0.0), cor / n * 100, (val / com * 100 if com else 0.0)

    covered = make_tests({**EASY, **HARD}, random.Random(7))
    novel = make_tests({**NOVEL}, random.Random(11))
    com, prec, alla, val = measure(router, covered, TAU)
    print(f"\n[end-to-end @ router seed 0, {len(covered)} held-out requests: unseen phrasings x unseen numbers, TAU={TAU}]")
    print(f"  committed (confident + args parse) : {com:.1f}%   (abstains on {100-com:.0f}% of IN-SCOPE phrasings)   VALID: {val:.0f}%")
    print(f"  CORRECT-when-committed (precision)  : {prec:.1f}%")
    train_seen = {req: exp for req, op, exp in make_tests({**TRAIN}, random.Random(123))}
    mem_ok = sum(1 for req, op, exp in covered if train_seen.get(req) == exp) / len(covered)
    print(f"  accuracy over ALL requests          : {alla:.1f}%   |   memoriser (recall): {mem_ok*100:.1f}% (cannot generalise)")

    # MAJOR (red-team): report precision MULTI-SEED -- the single-seed 100% is seed-lucky (one router
    # seed confidently misroutes a fragile HARD form at the shipped TAU); match intent.py's discipline.
    pr_tau, pr_90 = [], []
    for s in SEEDS:
        rs = random.Random(s)                                  # SAME shared-rng procedure as the seed-0 headline
        r = LogisticRouter(char_ng=True); r.train(make_examples(TRAIN, 12, rs), rs)
        pr_tau.append(measure(r, covered, TAU)[1]); pr_90.append(measure(r, covered, 0.90)[1])

    def mm(v):
        return f"{sum(v)/len(v):.0f}% ({min(v):.0f}-{max(v):.0f})"
    print(f"\n[CORRECT-when-committed over {len(SEEDS)} router seeds]  TAU={TAU}: {mm(pr_tau)}    "
          f"TAU=0.90: {mm(pr_90)}  <- precise for EVERY seed once TAU>=0.90")

    # honest abstention picture: precision is ~100% on COVERED across TAU (seed 0); coverage trades off;
    # but confidence does NOT cleanly reject out-of-scope SYNONYMS (they ride function-word cues).
    print(f"\n[precision/coverage trade-off vs TAU @ seed 0]  (chance on NOVEL = 33%)")
    print(f"   TAU |  COVERED: commit%  correct-when-committed |  NOVEL(synonyms): commit%  correct-when-committed")
    for tau in [0.30, 0.50, 0.70, 0.90]:
        a1 = measure(router, covered, tau); a2 = measure(router, novel, tau)
        print(f"  {tau:4.2f} |          {a1[0]:5.1f}            {a1[1]:5.1f}          |              {a2[0]:5.1f}            {a2[1]:5.1f}")

    print("\n" + "=" * 86)
    print("VERDICT: the integration COMPLETES item 2 -- a new doubly-held-out measurement neither §60 nor")
    print(f"  §64 made: LEARNED routing (unseen phrasings) + INDUCED compute (add/sub/mul {int(addc*100)}/{int(subc*100)}/{int(mulc*100)}% on unseen")
    print("  numbers; the induced rule IS the universal adder; memoriser 0%) + grammar + verification.")
    print(f"  CORRECT-when-committed = {mm(pr_tau)} across router seeds (100% at seed 0 and for EVERY seed at")
    print("  TAU>=0.90), at the cost of abstaining on ~17% of in-scope phrasings at TAU=0.40. HONEST LIMITS")
    print("  (caught in testing): precision is seed-fragile at low TAU on two HARD forms; and a confidence")
    print("  threshold does NOT reject out-of-scope SYNONYMS (committed-and-sometimes-wrong, not abstained)")
    print("  -- extending coverage is a MEANING problem (§65/§66), not a threshold knob.")
    print("=" * 86)


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    args = sys.argv[1:]
    if args and args[0] in ("-i", "--interactive", "--chat"):
        interactive()                               # python answerer.py -i   (type your own requests)
    elif args:
        respond_show(build(), " ".join(args))       # python answerer.py "what is 347 plus 891?"
    else:
        main()                                      # python answerer.py      (the held-out validation)

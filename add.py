#!/usr/bin/env python3
"""
add.py -- can the bit-native core GENERATE new data, or only recall? The honest test: ADDITION.

chat.py memorised the times-table (right on trained sums, wrong on 271+314). Here we ask the strong
question: can it INDUCE the *computation* of addition from a DISJOINT training set and then generate the
correct sum for numbers it never saw? If yes, that is generating new, correct data by a learned rule --
not recall.

Representation is the lever (the project's mantra): addition is hard to generalise as decimal text
(carry is non-local), but in BINARY, LSB-first, it is a tiny 1-bit-state transducer:
    output s_i      = f_out(a_i, b_i, carry)
    next carry      = f_upd(a_i, b_i, carry)
The carry is a HIDDEN recurrent state -- never given. We INDUCE both 3-input boolean functions by
searching the space of transducers for the one that explains the training sums (Path B's "learn the
recurrent computation", as in cycle 12 / sec 51). Then we measure GENERALISATION on held-out numbers.

Contrast: a chat-style MEMORISER keyed on the whole (a, b) -- correct on train, useless off it.
"""
import random

random.seed(0)
L = 12                      # numbers in [0, 2^L); sum needs L+1 bits
NTRAIN = 60
NTEST = 3000


def run_transducer(a, b, out_fn, upd_fn):
    """1-bit-state transducer over the bit columns (LSB->MSB). out_fn/upd_fn are 8-bit truth tables
    over (a_i, b_i, state). Returns the produced (L+1)-bit number."""
    state = 0; s = 0
    for i in range(L):
        ai = (a >> i) & 1; bi = (b >> i) & 1
        idx = (ai << 2) | (bi << 1) | state
        s |= ((out_fn >> idx) & 1) << i
        state = (upd_fn >> idx) & 1
    s |= state << L         # carry-out is the final state
    return s


def induce_adder(train):
    """Search the 256x256 transducer space for every hypothesis that reproduces ALL training sums.
    No carry is supplied -- the recurrent state is induced. (Early-exit makes the search fast.)"""
    winners = []
    for out_fn in range(256):
        for upd_fn in range(256):
            ok = True
            for a, b, s in train:
                if run_transducer(a, b, out_fn, upd_fn) != s:
                    ok = False; break
            if ok:
                winners.append((out_fn, upd_fn))
    return winners


def main():
    pairs = set()
    while len(pairs) < NTRAIN + NTEST:
        pairs.add((random.randrange(1 << L), random.randrange(1 << L)))
    pairs = list(pairs)
    train = [(a, b, a + b) for a, b in pairs[:NTRAIN]]
    test = [(a, b, a + b) for a, b in pairs[NTRAIN:NTRAIN + NTEST]]   # DISJOINT, never seen

    print("=" * 78)
    print(f"Can the bit-native core GENERATE new data? -- addition, {L}-bit, held-out test")
    print(f"  train on {NTRAIN} (a,b) pairs; test on {NTEST} DIFFERENT pairs it never saw")
    print("=" * 78)

    # --- baseline: chat-style MEMORISER (recall only) ---
    mem = {(a, b): s for a, b, s in train}
    mem_train = sum(1 for a, b, s in train if mem.get((a, b)) == s) / len(train)
    mem_test = sum(1 for a, b, s in test if mem.get((a, b), 0) == s) / len(test)
    print(f"\n[memoriser]  (key on the whole (a,b), like chat.py recall)")
    print(f"  train accuracy   = {mem_train*100:.1f}%")
    print(f"  HELD-OUT accuracy = {mem_test*100:.1f}%   <- cannot generate what it never stored")

    # --- INDUCE the computation (no carry supplied) ---
    winners = induce_adder(train)
    print(f"\n[induced adder]  searched 65,536 transducers; {len(winners)} reproduce ALL training sums")
    if not winners:
        print("  none found -- increase NTRAIN so the 8 input combos are covered"); return
    out_fn, upd_fn = winners[0]
    # is it the real full adder?  out = a^b^c (0x96=150), carry = maj(a,b,c) (0xE8=232)
    is_adder = (out_fn, upd_fn) == (150, 232)
    test_acc = sum(1 for a, b, s in test if run_transducer(a, b, out_fn, upd_fn) == s) / len(test)
    print(f"  induced functions: out_fn={out_fn} (XOR3? {out_fn==150}), upd_fn={upd_fn} (MAJ3? {upd_fn==232})")
    print(f"  -> {'this IS the full adder (XOR + carry-majority)' if is_adder else 'an equivalent solver'}")
    print(f"  train accuracy   = 100.0%")
    print(f"  HELD-OUT accuracy = {test_acc*100:.1f}%   <- correct sums for numbers it NEVER saw")

    # --- show it GENERATING a few unseen sums, bit by bit ---
    print("\n[generation]  the induced adder produces the sum for unseen inputs (none were in training):")
    for a, b, s in test[:6]:
        g = run_transducer(a, b, out_fn, upd_fn)
        print(f"   {a} + {b} = {g}   {'OK' if g == s else 'WRONG'}  (memoriser would say {mem.get((a,b),'??')})")

    print("\n" + "=" * 78)
    print("VERDICT:")
    print(f"  memoriser (recall)      : held-out {mem_test*100:.0f}%  -> cannot generate new data")
    print(f"  induced computation     : held-out {test_acc*100:.0f}%  -> GENERATES correct new data")
    print("  The difference is learning the COMPUTATION (the carry, induced) vs storing answers.")
    print("  Knowledge stays separable; the core learned a *function* that generalises. That is the")
    print("  honest 'yes' to: can it generate new data -- where it induces the rule, not the table.")
    print("=" * 78)


if __name__ == "__main__":
    main()

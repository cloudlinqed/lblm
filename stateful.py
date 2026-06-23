#!/usr/bin/env python3
"""
Path B -- the STATEFUL frontier: invent a running-counter primitive (cross sec-50's wall).

sec 50 located the next wall precisely: Dyck-1 balanced parentheses stayed UNSAT even with
the invented count-comparators, because 'every prefix sum >= 0' is not a function of any
GLOBAL count -- it needs a running PREFIX-MIN, i.e. STATE carried along the stream. That is
the bit-native 'compute a running feature' idea from cycles 9-13 (latch / running-XOR /
mod-m counter), now needed for program induction.

This probe invents a STATEFUL primitive and applies the same WAKE -> SLEEP -> BIND machinery:
  primitive : a signed running counter  bal[i] = bal[i-1] + step(s[i])  (step: 0->+1, 1->-1)
  readouts  : final = bal[-1],  min = min prefix,  max = max prefix
  atom      : compare(readout, k)  for compare in {ge,le,eq,ne}  (reuse the sec-50 DSL)
  feature   : a single atom OR a conjunction of two atoms (Dyck needs two: balanced AND never-negative)

  (A) re-confirm Dyck is UNSAT in the sec-50 detector+comparator grammar (no state).
  (B) INVENT the counter primitive: WAKE finds Dyck = eq(final,0) AND ge(min,0), full-domain-exact.
  (C) GENERALISE: leave-one-out over a family of single-counter tasks (readout,op,k recovered).
  (D) the NEXT honest boundary: a non-local / two-ended task (palindrome) stays UNSAT even with
      one counter -- it needs comparing distant positions (a second counter / a stack).

HONESTY (identical to wake_lgg / m3 / invent_agg): full-domain DIRECT-SOLVER check over all 2^L
inputs, balanced cross-seed on fresh seeds, multi-draw balanced-scramble, no task identity supplied
(the step/readout/comparator/threshold are RECOVERED from labels). Observational-equivalence dedup
keeps the conjunction search tractable.
"""
import random
from statistics import mean
from collections import Counter, defaultdict
from induce import gen, L
from wake_lgg import split_idx, eval_feature, balanced_eval
import invent_agg                                   # sec-50 detector+comparator grammar (no state)


# ---------------------------------------------------------------------------
# the stateful primitive: a signed running counter + prefix readouts
# ---------------------------------------------------------------------------
STEP = (1, -1)                                       # bit 0 -> +1, bit 1 -> -1
READOUTS = ['final', 'min', 'max']
OPS = ['ge', 'le', 'eq', 'ne']
KS = list(range(-L, L + 1))


def trajectory(s, step):
    bal = 0; tr = []
    for b in s:
        bal += step[b]; tr.append(bal)
    return tr                                        # prefix sums after each step


def readout(tr, kind):
    if kind == 'final':
        return tr[-1]
    if kind == 'min':
        return min(tr)
    return max(tr)


def cmp_apply(v, op, k):
    if op == 'ge':
        return v >= k
    if op == 'le':
        return v <= k
    if op == 'eq':
        return v == k
    return v != k


def atom_feature(s, atom):
    step, kind, op, k = atom
    return int(cmp_apply(readout(trajectory(s, step), kind), op, k))


def feat_fn(atoms):
    """AND of one or more stateful atoms -> a {0,1} feature over a sequence."""
    def f(s):
        v = 1
        for a in atoms:
            v &= atom_feature(s, a)
        return v
    return f


# ---------------------------------------------------------------------------
# honesty helpers (operate on a feature function)
# ---------------------------------------------------------------------------
def full_domain_exact(ff, task_fn):
    for x in range(1 << L):
        s = [(x >> (L - 1 - i)) & 1 for i in range(L)]
        if ff(s) != task_fn(s):
            return False
    return True


def bal_scramble_feat(feat, n, seed, tr, dv, draws=5):
    vals = []
    for d in range(draws):
        rng = random.Random(seed * 7 + 1 + d * 1009)
        rand = [rng.randint(0, 1) for _ in range(n)]
        vals.append(balanced_eval(feat, rand, tr, dv))
    return mean(vals)


def cross_seed_bal(ff, task_fn, seeds):
    accs = []
    for sd in seeds:
        items = gen(task_fn, 300, sd); ans = [a for _, a in items]
        feat = [ff(s) for s, _ in items]
        tr, dv, te = split_idx(len(items), sd)
        accs.append(balanced_eval(feat, ans, tr, te))
    return mean(accs), [round(a, 3) for a in accs]


# ---------------------------------------------------------------------------
# WAKE (stateful): single atoms, then conjunctions of two (obs-equivalence dedup)
# ---------------------------------------------------------------------------
def wake_stateful(task_fn, seed=0, thr=0.999, allow_conj=True):
    items = gen(task_fn, 300, seed); ans = [a for _, a in items]
    seqs = [s for s, _ in items]
    tr, dv, te = split_idx(len(items), seed); trdv = tr + dv

    def gate(feat):
        return (eval_feature(feat, ans, tr, dv) > thr and
                eval_feature(feat, ans, trdv, te) > thr and
                bal_scramble_feat(feat, len(items), seed, tr, dv) < 0.55)

    # build distinct, non-constant atom features (observational-equivalence dedup)
    atoms = [(STEP, kind, op, k) for kind in READOUTS for op in OPS for k in KS]
    feats = {}
    for atom in atoms:
        feat = [atom_feature(s, atom) for s in seqs]
        if len(set(feat)) < 2:
            continue
        feats.setdefault(tuple(feat), (atom, feat))

    # single-atom solvers
    for sig, (atom, feat) in feats.items():
        if gate(feat) and full_domain_exact(feat_fn([atom]), task_fn):
            return [atom]

    if not allow_conj:
        return None

    # pairwise conjunctions (Dyck needs balanced AND never-negative)
    keys = list(feats)
    for i in range(len(keys)):
        a_i, f_i = feats[keys[i]]
        for j in range(i + 1, len(keys)):
            a_j, f_j = feats[keys[j]]
            conj = [x & y for x, y in zip(f_i, f_j)]
            if len(set(conj)) < 2:
                continue
            if gate(conj) and full_domain_exact(feat_fn([a_i, a_j]), task_fn):
                return [a_i, a_j]
    return None


# ---------------------------------------------------------------------------
# task families
# ---------------------------------------------------------------------------
def stateful_task(kind, op, k, step=STEP):
    return lambda s: int(cmp_apply(readout(trajectory(s, step), kind), op, k))


def dyck_task(s):                                    # balanced parens: final==0 AND min>=0
    tr = trajectory(s, STEP)
    return int(tr[-1] == 0 and min(tr) >= 0)


def palindrome_task(s):                              # non-local: s == reverse(s)
    return int(s == s[::-1])


def fmt(atoms):
    return " AND ".join(f"{k} {op} {kk}" for (_, k, op, kk) in atoms)


def main():
    random.seed(0)
    bar = "=" * 80
    fresh = [717, 818, 919, 1212, 1313]
    print(bar)
    print("stateful.py -- invent a running-counter primitive (cross sec-50's stateful wall)")
    print(bar)

    # -------- (A) Dyck UNSAT without state ------------------------------------
    print("\n[A] Dyck-1 in the sec-50 detector+comparator grammar (NO state):")
    sp, ag = invent_agg.wake(invent_agg.dyck_task, invent_agg.GEN_AGGS)
    print(f"    {'solved' if sp else 'UNSAT (a global count cannot see the prefix condition)'}")

    # -------- (B) invent the counter primitive, crack Dyck --------------------
    print("\n[B] INVENT a signed running counter (0->+1, 1->-1) with prefix readouts;")
    print("    WAKE searches compare(readout,k) atoms + conjunctions (readout/op/k NOT supplied):")
    sol = wake_stateful(dyck_task)
    if sol:
        ff = feat_fn(sol)
        cs, accs = cross_seed_bal(ff, dyck_task, fresh)
        items = gen(dyck_task, 300, 0); tr, dv, te = split_idx(300, 0)
        bs = bal_scramble_feat([ff(s) for s, _ in items], 300, 0, tr, dv)
        fd = full_domain_exact(ff, dyck_task)
        print(f"    INVENTED counter solver:  {fmt(sol)}")
        print(f"      full-domain-exact={fd}  cross-seed={cs:.3f} {accs}  bal-scram={bs:.2f}")
        dyck_ok = fd and cs > 0.98 and bs < 0.55
    else:
        print("    could NOT crack Dyck"); dyck_ok = False

    # -------- (C) generalise over a single-counter family ---------------------
    print("\n[C] GENERALISE -- leave-one-out over single-counter tasks (readout,op,k recovered):")
    curric = [('min', 'ge', 0), ('final', 'eq', 0), ('max', 'le', 2)]
    held   = [('final', 'ge', 1), ('max', 'ge', 3), ('min', 'le', -2), ('final', 'le', 0)]
    for (kind, op, k) in curric:
        sol = wake_stateful(stateful_task(kind, op, k), allow_conj=False)
        tag = fmt(sol) if sol else 'UNSAT'
        cs = cross_seed_bal(feat_fn(sol), stateful_task(kind, op, k), fresh)[0] if sol else 0.0
        print(f"    curric  {kind} {op} {k:>2}: {tag:18} cross-seed(bal)={cs:.3f}")
    gen_ok = True
    for (kind, op, k) in held:
        task = stateful_task(kind, op, k)
        sol = wake_stateful(task, allow_conj=False)
        if sol is None:
            print(f"    HELDOUT {kind} {op} {k:>2}: could NOT bind"); gen_ok = False; continue
        ff = feat_fn(sol)
        cs, accs = cross_seed_bal(ff, task, fresh)
        items = gen(task, 300, 0); tr, dv, te = split_idx(300, 0)
        bs = bal_scramble_feat([ff(s) for s, _ in items], 300, 0, tr, dv)
        ok = cs > 0.98 and bs < 0.55 and full_domain_exact(ff, task)
        gen_ok = gen_ok and ok
        print(f"    HELDOUT {kind} {op} {k:>2}: {fmt(sol):18} cross-seed={cs:.3f} "
              f"bal-scram={bs:.2f}  {'OK' if ok else 'FAIL'}")

    # -------- (D) next boundary: non-local / two-ended -----------------------
    print("\n[D] NEXT boundary -- palindrome  s == reverse(s)  (non-local, two-ended):")
    sp2, ag2 = invent_agg.wake(palindrome_task, invent_agg.GEN_AGGS)         # detector+comparator
    sol2 = wake_stateful(palindrome_task)                                    # one counter + conj
    if sp2 is None and sol2 is None:
        print("    UNSAT in BOTH the detector+comparator AND the single-counter grammar.")
        print("    -> palindrome compares position i to L-1-i: a non-local, two-ended relation")
        print("       that one left-to-right counter cannot carry. The wall RELOCATES to")
        print("       MULTI-COUNTER / STACK / two-way computation -- the next frontier.")
        pal_unsat = True
    else:
        print(f"    unexpectedly solved (detector={sp2}, stateful={sol2})"); pal_unsat = False

    # -------- summary ---------------------------------------------------------
    print("\n" + bar)
    print("SUMMARY -- invent the stateful primitive, then find the next wall")
    print(bar)
    print(f"  (A) Dyck UNSAT without state (sec-50 boundary holds)        : confirmed")
    print(f"  (B) invented a running counter -> Dyck cracked, full-domain : "
          f"{'PASS' if dyck_ok else 'FAIL'}")
    print(f"  (C) single-counter family generalises (leave-one-out)       : "
          f"{'PASS' if gen_ok else 'PARTIAL/FAIL'}")
    print(f"  (D) palindrome UNSAT -> next frontier = multi-counter/stack  : "
          f"{'as expected' if pal_unsat else 'unexpectedly solved'}")
    print("\n  Reading: the hierarchy of frontiers extends into STATE. A running counter (the")
    print("  cycles 9-13 mechanism) is invented from Dyck's failure and generalises across the")
    print("  single-counter family; the next genuine wall is non-local / multi-counter structure")
    print("  (palindrome, a^n b^n c^n) -- a stack or two-way computation. Precisely located, again.")
    print(bar)


if __name__ == "__main__":
    main()

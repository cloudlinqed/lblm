#!/usr/bin/env python3
"""
Path B -- crossing the RELOCATED wall: invent the missing AGGREGATION from a failure.

sec 49 (M3) showed the wake->sleep->bind mechanism generalises WITHIN the detector+
aggregation paradigm (gapped patterns; parity->threshold), but hit an honest boundary at
'#0==#1' (count EQUALITY): no (detector, {parity,threshold}) solver exists, so WAKE
returned UNSAT. That is exactly the sec 44 situation -- the library BREAKS -- one level up:
the missing piece is now an AGGREGATION, not a detector parameter.

This probe applies the sec-44 INVENTION move to the aggregation axis: when the known
aggregations {cnt2, ge_k} fail, search a GENERATIVE space of count-predicates
compare(count, k) for compare in {ge, le, eq, ne} and INVENT the missing comparator that
solves the failed task -- then fold it into a parameterised template detect(spec) `op` k and
generalise to held-out count-comparison tasks. Finally it locates the NEXT honest boundary:
a genuinely STATEFUL family (Dyck-1 balanced parentheses) stays UNSAT even with the invented
comparators, because it needs a running prefix-min (a counter/automaton), not a global count.

HONESTY (identical to wake_lgg.py / m3): full-domain DIRECT-SOLVER check over all 2^L inputs
(a trick cannot pass), balanced cross-seed on fresh seeds, multi-draw balanced-scramble, no
task identity supplied (the comparator and threshold are RECOVERED from labels).
"""
import random
from statistics import mean
from collections import Counter, defaultdict
from induce import gen, L
from wake_lgg import lit_stream, split_idx, eval_feature, balanced_eval
from m3_different_family import spec_stream, enumerate_specs, contig_count, pat_spec


# ---------------------------------------------------------------------------
# generative aggregation DSL over the detector-stream COUNT: compare(count, k)
# ---------------------------------------------------------------------------
def agg_apply(row, agg):
    c = sum(row)
    op = agg[0]
    if op == 'cnt2':
        return c % 2
    k = agg[1]
    if op == 'ge':
        return int(c >= k)
    if op == 'le':
        return int(c <= k)
    if op == 'eq':
        return int(c == k)
    if op == 'ne':
        return int(c != k)
    raise ValueError(agg)


KNOWN_AGGS  = [('cnt2',)] + [('ge', k) for k in range(0, L + 1)]         # the library so far
COMPARATORS = ['ge', 'le', 'eq', 'ne']
GEN_AGGS    = [('cnt2',)] + [(op, k) for op in COMPARATORS for k in range(0, L + 1)]
KNOWN_FORMS = {'cnt2', 'ge'}                                             # forms already known


def spec_feature(s, spec, agg):
    return agg_apply(spec_stream(s, spec), agg)


def full_domain_exact(spec, agg, task_fn):
    for x in range(1 << L):
        s = [(x >> (L - 1 - i)) & 1 for i in range(L)]
        if spec_feature(s, spec, agg) != task_fn(s):
            return False
    return True


def bal_scramble(spec, agg, task_fn, seed=0, draws=5):
    items = gen(task_fn, 300, seed); n = len(items)
    feat = [spec_feature(s, spec, agg) for s, _ in items]
    tr, dv, te = split_idx(n, seed)
    vals = []
    for d in range(draws):
        rng = random.Random(seed * 7 + 1 + d * 1009)
        rand = [rng.randint(0, 1) for _ in range(n)]
        vals.append(balanced_eval(feat, rand, tr, dv))
    return mean(vals)


def cross_seed_bal(spec, agg, task_fn, seeds):
    accs = []
    for sd in seeds:
        items = gen(task_fn, 300, sd); ans = [a for _, a in items]
        feat = [spec_feature(s, spec, agg) for s, _ in items]
        tr, dv, te = split_idx(len(items), sd)
        accs.append(balanced_eval(feat, ans, tr, te))
    return mean(accs), [round(a, 3) for a in accs]


def wake(task_fn, aggs, maxlag=3, thr=0.999, seed=0):
    items = gen(task_fn, 300, seed); ans = [a for _, a in items]
    seqs = [s for s, _ in items]
    tr, dv, te = split_idx(len(items), seed); trdv = tr + dv
    for spec in enumerate_specs(maxlag):
        for agg in aggs:
            feat = [spec_feature(s, spec, agg) for s in seqs]
            if eval_feature(feat, ans, tr, dv) <= thr:
                continue
            if eval_feature(feat, ans, trdv, te) <= thr:
                continue
            if bal_scramble(spec, agg, task_fn, seed) >= 0.55:
                continue
            if not full_domain_exact(spec, agg, task_fn):
                continue
            return spec, agg
    return None, None


# ---------------------------------------------------------------------------
# task families
# ---------------------------------------------------------------------------
def count_cmp_task(pat, op, k):
    def f(s):
        c = contig_count(s, pat)
        return int({'ge': c >= k, 'le': c <= k, 'eq': c == k, 'ne': c != k}[op])
    return f


def balanced_task(s):                       # #0 == #1   (the sec-49 boundary)
    return int(sum(s) == L // 2)


def dyck_task(s):                           # 0->'(' (+1), 1->')' (-1): balanced parentheses
    bal = 0
    for b in s:
        bal += 1 if b == 0 else -1
        if bal < 0:                         # a prefix went negative -> not balanced
            return 0
    return int(bal == 0)                     # ends balanced AND never went negative


def main():
    random.seed(0)
    bar = "=" * 80
    fresh = [717, 818, 919, 1212, 1313]
    print(bar)
    print("invent_agg.py -- invent the missing AGGREGATION from a failure (cross sec-49 wall)")
    print(bar)

    # -------- (A) the failure, then INVENT the comparator -----------------------
    print("\n[A] sec-49 boundary: '#0==#1' with the KNOWN aggregations {cnt2, ge_k}")
    spec0, agg0 = wake(balanced_task, KNOWN_AGGS)
    print(f"    known-aggregation grammar: {'solved' if spec0 else 'UNSAT (the boundary stands)'}")

    print("\n    INVENT: search the generative count-predicate space compare(count,k),")
    print("            compare in {ge, le, eq, ne} -- grow the toolbox from the failure:")
    spec, agg = wake(balanced_task, GEN_AGGS)
    invented = agg is not None and agg[0] not in KNOWN_FORMS
    if spec:
        cs, accs = cross_seed_bal(spec, agg, balanced_task, fresh)
        bs = bal_scramble(spec, agg, balanced_task)
        print(f"    INVENTED aggregation '{agg[0]}' -> solver: count(spec {list(spec)}) "
              f"{agg[0]} {agg[1]}")
        print(f"      full-domain-exact={full_domain_exact(spec, agg, balanced_task)}  "
              f"cross-seed={cs:.3f} {accs}  bal-scram={bs:.2f}  "
              f"new-form={'YES' if invented else 'no'}")
    invent_ok = bool(spec) and invented

    # -------- (B) generalise the invented comparator into a template ------------
    print("\n[B] fold the invented comparator into a template detect(spec) `op` k;")
    print("    leave-one-out over a count-comparison FAMILY (op,k recovered from labels):")
    curric = [("1", "eq", 6), ("1", "le", 3), ("11", "eq", 2)]
    held   = [("1", "ne", 6), ("0", "eq", 4), ("10", "le", 1), ("11", "ge", 2)]
    print("    curriculum (WAKE recovers spec, comparator, threshold):")
    solved = []
    for (pat, op, k) in curric:
        task = count_cmp_task(pat, op, k)
        sp, ag = wake(task, GEN_AGGS)
        if sp is None:
            print(f"      {pat:>2} {op} {k}: could NOT solve"); continue
        solved.append((sp, ag))
        cs, _ = cross_seed_bal(sp, ag, task, fresh)
        print(f"      #{pat} {op} {k}: spec {list(sp)} agg {ag}  cross-seed(bal)={cs:.3f}")
    # SLEEP: all solutions share the form compare(count(AND of lits), k) -> the template
    shared = all(ag[0] in COMPARATORS for (_, ag) in solved) and len(solved) == len(curric)
    print(f"    SLEEP: every solution is compare(count(detector), k)? {shared}  ->")
    print("           template detect(spec) `op` k  parameterised by (spec, op, k)")

    print("    BIND held-out (pattern, comparator, threshold) -- none in curriculum:")
    gen_ok = shared
    for (pat, op, k) in held:
        task = count_cmp_task(pat, op, k)
        sp, ag = wake(task, GEN_AGGS)
        if sp is None:
            print(f"      #{pat} {op} {k}: could NOT bind"); gen_ok = False; continue
        cs, accs = cross_seed_bal(sp, ag, task, fresh)
        bs = bal_scramble(sp, ag, task)
        canonical = (set(sp) == set(pat_spec(pat)) and ag == (op, k))
        ok = cs > 0.98 and bs < 0.55 and full_domain_exact(sp, ag, task)
        gen_ok = gen_ok and ok
        print(f"      #{pat} {op} {k}: spec {list(sp)} agg {ag}  cross-seed={cs:.3f} "
              f"bal-scram={bs:.2f}  canonical={canonical}  {'OK' if ok else 'FAIL'}")

    # -------- (C) the NEXT honest boundary: a stateful family -------------------
    print("\n[C] NEXT boundary -- Dyck-1 balanced parentheses (0='(' +1, 1=')' -1):")
    print("    balanced iff total==0 AND every prefix sum >= 0 (a STATEFUL condition).")
    print("    Note: total==0 alone IS '#0==#1' (now solvable via the invented 'eq').")
    eq_only = wake(dyck_task, [('eq', L // 2)])           # the count part alone
    sp, ag = wake(dyck_task, GEN_AGGS)                     # full comparator grammar
    if sp is None:
        print("    even WITH the invented comparators: UNSAT.")
        print("    -> Dyck needs a running PREFIX-MIN (a counter/automaton over prefixes),")
        print("       not a global count comparison. The wall RELOCATES again, precisely at")
        print("       STATEFUL computation -- the next frontier (sec 43 automaton / a counter")
        print("       primitive). The boundary is the prefix condition, not the equality.")
        dyck_unsat = True
    else:
        cs, _ = cross_seed_bal(sp, ag, dyck_task, fresh)
        print(f"    unexpectedly solved by spec {list(sp)} agg {ag} cross-seed(bal)={cs:.3f}")
        dyck_unsat = False

    # -------- summary -----------------------------------------------------------
    print("\n" + bar)
    print("SUMMARY -- invent the missing aggregation, then find the next wall")
    print(bar)
    print(f"  (A) invented the missing comparator 'eq' from the '#0==#1' failure : "
          f"{'PASS' if invent_ok else 'FAIL'}")
    print(f"  (B) folded it into a (spec, op, k) template; held-out generalised  : "
          f"{'PASS' if gen_ok else 'PARTIAL/FAIL'}")
    print(f"  (C) Dyck-1 (stateful) still UNSAT -> next frontier located         : "
          f"{'as expected' if dyck_unsat else 'unexpectedly solved'}")
    print("\n  Reading: the sec-44 INVENTION move lifts to the aggregation axis -- a failure")
    print("  (sec-49's count-equality boundary) is crossed by INVENTING the missing comparator")
    print("  from a generative space, then generalising it as a parameterised template. The")
    print("  hierarchy of frontiers continues: detector params -> aggregation params -> the")
    print("  next genuine wall is STATEFUL computation (prefix-min/counter), precisely located.")
    print(bar)


if __name__ == "__main__":
    main()

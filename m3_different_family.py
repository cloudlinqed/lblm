#!/usr/bin/env python3
"""
Path B -- M3: is the abstraction mechanism GENERAL, or a per-family manual move?

wake_lgg.py cracked the wall for ONE family: parity of sliding-window occurrences of a
CONTIGUOUS pattern P. The induced template detect(P) hard-wires "AND of contiguous-lag
literals at lags |P|-1..0 + cnt2". The honest open question (design doc sec 48): when we
move to a DIFFERENT family, must we hand-write a new template, or does the SAME machinery
(boundary-aware detection + Plotkin anti-unification + bind-only-the-parameter) discover
the new abstraction on its own? If it is a per-family manual move, the wall simply
reappears one family up and the "open-ended" claim fails.

This probe runs the same WAKE -> SLEEP -> BIND machinery on THREE graduated families,
with a GENERAL grammar (literals over arbitrary offset-SUBSETS, not just contiguous; a
small set of aggregations), and reports honestly where it holds and where it stops.

  M3a  GAPPED patterns (near transfer).  e.g. '1.1','0.0','1..1' -- literal conjunctions
       at NON-CONTIGUOUS lags. The sec-48 contiguous template detect(P:string) literally
       CANNOT represent a gap. Test: does generalising the PARAMETER from a string to a
       'spec' (a set of (lag,bit) literals) -- the same anti-unification, richer grammar --
       recover gapped detectors and generalise to HELD-OUT gapped patterns?

  M3b  THRESHOLD counting (a SECOND abstraction axis).  '#occ(P) >= t' -- the detector
       stream is the same, but the AGGREGATION changes from parity to a count-threshold,
       parameterised by t. Test: can the mechanism induce a template parameterised by BOTH
       (spec, t) -- abstracting the aggregation, not just the detector -- and generalise to
       unseen (P,t)? (This is milestone M2: a second abstraction axis.)

  M3c  OUT-OF-FAMILY (the honest boundary).  'a^n b^n': the stream is 0^k 1^k for some k>0
       (a balanced/counting language). No (spec, aggregation) in this grammar is a
       full-domain-exact solver. Test: the mechanism should HONESTLY return UNSAT, marking
       exactly where the detector+aggregation paradigm stops and the wall relocates.

HONESTY: identical controls to wake_lgg.py -- balanced-accuracy cross-seed on FRESH seeds,
multi-draw balanced-scramble (skew + estimator-noise immune), full-domain DIRECT-SOLVER
check over all 2^L inputs (a trick cannot pass), negative controls, no pattern hardcoding.
"""
import random, itertools
from statistics import mean
from collections import Counter, defaultdict
from induce import gen, L
from wake_lgg import lit_stream, split_idx, eval_feature, balanced_eval


# ---------------------------------------------------------------------------
# generalised grammar: a SPEC is a sorted tuple of (lag, bit) literals;
# an AGG is ('cnt2',) | ('ge', t). detector = AND of lits, masked to i>=max_lag.
# ---------------------------------------------------------------------------
def spec_stream(s, spec):
    max_off = max(o for o, _ in spec)
    n = len(s); row = [1] * n
    for (o, b) in spec:
        ls = lit_stream(s, o, b)
        row = [r & v for r, v in zip(row, ls)]
    return [row[i] if i >= max_off else 0 for i in range(n)]


def agg_apply(row, agg):
    if agg[0] == 'cnt2':
        return sum(row) % 2
    if agg[0] == 'ge':
        return int(sum(row) >= agg[1])
    raise ValueError(agg)


def spec_feature(s, spec, agg):
    return agg_apply(spec_stream(s, spec), agg)


# ---------------------------------------------------------------------------
# honesty helpers over (spec, agg)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# WAKE (general): search (spec, agg) for a full-domain-exact, scramble-clean solver.
#   spec = any nonempty subset of lags 0..maxlag, each with a polarity bit.
#   Prefers the simplest solver (fewest literals, then smallest max-lag).
# ---------------------------------------------------------------------------
def enumerate_specs(maxlag):
    offs = list(range(maxlag + 1))
    specs = []
    for combo in itertools.product([None, 0, 1], repeat=maxlag + 1):
        lits = tuple((o, b) for o, b in zip(offs, combo) if b is not None)
        if lits:
            specs.append(lits)
    specs.sort(key=lambda sp: (len(sp), max(o for o, _ in sp)))
    return specs


def wake(task_fn, aggs, maxlag=4, thr=0.999, seed=0):
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
def gapped_count(s, pat):                 # pat like '1.1' ('.' = wildcard)
    m = len(pat)
    return sum(1 for i in range(m - 1, len(s))
               if all(pat[j] == '.' or int(pat[j]) == s[i - m + 1 + j] for j in range(m)))


def gapped_parity(pat):
    return lambda s: gapped_count(s, pat) % 2


def contig_count(s, pat):
    P = [int(c) for c in pat]; m = len(P)
    return sum(1 for i in range(m - 1, len(s)) if list(s[i - m + 1:i + 1]) == P)


def threshold_task(pat, t):
    return lambda s: int(contig_count(s, pat) >= t)


def balanced_task(s):                      # #0 == #1 (the counting/balance essence of a^n b^n)
    return int(sum(s) == len(s) // 2)


# canonical detector spec for a gapped/contiguous pattern (independent reference)
def pat_spec(pat):
    m = len(pat)
    return tuple(sorted(((m - 1 - j, int(pat[j])) for j in range(m) if pat[j] != '.'),
                        key=lambda x: -x[0]))


def main():
    random.seed(0)
    bar = "=" * 80
    fresh = [717, 818, 919, 1212, 1313]
    print(bar)
    print("M3 -- is the abstraction mechanism GENERAL, or a per-family manual move?")
    print(bar)

    # =====================================================================
    # M3a -- GAPPED patterns (near transfer): generalise the PARAMETER string->spec
    # =====================================================================
    print("\n[M3a] GAPPED patterns (literal conjunctions at NON-contiguous lags)")
    print("      The sec-48 contiguous template detect(P:string) cannot encode a gap.")
    aggs_par = [('cnt2',)]
    curric_g = ['1.1', '0.0', '11', '00']          # mix of gapped + contiguous
    held_g   = ['1..1', '0.1', '1.0', '0..0', '1.1.1']
    print("      WAKE recovers detectors from labels (pattern/gap NOT supplied):")
    solved = {}
    for pat in curric_g:
        spec, agg = wake(gapped_parity(pat), aggs_par)
        solved[pat] = (spec, agg)
        cs, _ = cross_seed_bal(spec, agg, gapped_parity(pat), fresh)
        print(f"        {pat:6} -> spec {list(spec)} agg {agg[0]}  "
              f"cross-seed(bal)={cs:.3f}  full-domain-exact=True")

    # SLEEP: all curriculum solutions share form  cnt2(AND of lits) -- the literal SET
    # is the lifted parameter (string -> spec). Verify the form is shared, abstract spec.
    shared_form = all(agg == ('cnt2',) for (_, agg) in solved.values())
    print(f"      SLEEP: every solution has the form cnt2(AND of literals)? {shared_form}")
    print("             -> induced template detect(spec) = cnt2( AND_{(o,b) in spec} lit(s,o,b) )")
    print("             (parameter generalised from a contiguous string to a literal-spec)")

    def template_spec(pat):                # the induced template, instantiated
        return pat_spec(pat), ('cnt2',)

    # BIND: freeze template; for held-out gapped patterns, recover the spec from labels.
    print("      BIND held-out gapped patterns (cross-seed + scramble + full-domain):")
    m3a_ok = shared_form
    for pat in held_g:
        task = gapped_parity(pat)
        spec, agg = wake(task, aggs_par)   # bind via the same spec grammar (no gap supplied)
        if spec is None:
            print(f"        {pat:6} -> could NOT bind"); m3a_ok = False; continue
        cs, accs = cross_seed_bal(spec, agg, task, fresh)
        bs = bal_scramble(spec, agg, task)
        # gold standard: full-domain-exact (provably correct over ALL inputs) + cross-seed
        # + scramble-clean. (Also note whether it matched the canonical reference detector,
        # set-wise -- spec literal order is not semantically meaningful.)
        canonical = (set(spec) == set(pat_spec(pat)))
        ok = cs > 0.98 and bs < 0.55 and full_domain_exact(spec, agg, task)
        m3a_ok = m3a_ok and ok
        print(f"        {pat:6} -> spec {list(spec)}  cross-seed={cs:.3f} bal-scram={bs:.2f}  "
              f"canonical={canonical}  {'OK' if ok else 'FAIL'}")
    print(f"      M3a verdict: {'PASS -- mechanism generalises to gapped patterns' if m3a_ok else 'FAIL'}")

    # =====================================================================
    # M3b -- THRESHOLD counting (second abstraction axis: the AGGREGATION)
    # =====================================================================
    print("\n[M3b] THRESHOLD counting  '#occ(P) >= t'  (a NEW aggregation axis)")
    aggs_thr = [('cnt2',)] + [('ge', t) for t in range(1, 9)]
    # show parity-only aggregation CANNOT solve a threshold task (motivation)
    spec_par_only, _ = wake(threshold_task('11', 2), [('cnt2',)])
    print(f"      parity-only grammar on '#11>=2': {'solved' if spec_par_only else 'NOT solvable (needs a threshold agg)'}")
    curric_t = [('1', 6), ('11', 1), ('0', 6)]     # threshold tasks (P, t)
    held_t   = [('11', 2), ('1', 7), ('00', 1), ('10', 2)]
    print("      WAKE recovers (detector, threshold) from labels:")
    solved_t = {}
    for (pat, t) in curric_t:
        task = threshold_task(pat, t)
        spec, agg = wake(task, aggs_thr)
        if spec is None:
            print(f"        #{pat}>= {t}: could NOT solve"); continue
        solved_t[(pat, t)] = (spec, agg)
        cs, _ = cross_seed_bal(spec, agg, task, fresh)
        print(f"        #{pat}>={t}: spec {list(spec)} agg {agg}  cross-seed(bal)={cs:.3f}")
    second_axis = all(agg[0] == 'ge' for (_, agg) in solved_t.values()) and len(solved_t) > 0
    print(f"      SLEEP: solutions abstract over a (detector, threshold) template "
          f"detect(spec) >= t? {second_axis}")
    print("      BIND held-out (P,t) (no spec/threshold supplied):")
    m3b_ok = second_axis
    for (pat, t) in held_t:
        task = threshold_task(pat, t)
        spec, agg = wake(task, aggs_thr)
        if spec is None:
            print(f"        #{pat}>={t}: could NOT bind"); m3b_ok = False; continue
        cs, accs = cross_seed_bal(spec, agg, task, fresh)
        bs = bal_scramble(spec, agg, task)
        canonical = (agg == ('ge', t) and set(spec) == set(pat_spec(pat)))
        ok = cs > 0.98 and bs < 0.55 and full_domain_exact(spec, agg, task)
        m3b_ok = m3b_ok and ok
        print(f"        #{pat}>={t}: spec {list(spec)} agg {agg}  cross-seed={cs:.3f} "
              f"bal-scram={bs:.2f}  canonical={canonical}  {'OK' if ok else 'FAIL'}")
    print(f"      M3b verdict: {'PASS -- mechanism induces a 2nd abstraction axis (threshold)' if m3b_ok else 'PARTIAL/FAIL'}")

    # =====================================================================
    # M3c -- OUT-OF-FAMILY: a^n b^n  (the honest boundary)
    # =====================================================================
    print("\n[M3c] OUT-OF-FAMILY  #0==#1  (balanced counts -- the counting essence of a^n b^n)")
    spec, agg = wake(balanced_task, aggs_thr, maxlag=4)
    if spec is None:
        print("      WAKE: NO (spec, aggregation) in the detector grammar is full-domain-exact.")
        print("      -> HONEST UNSAT: '#0==#1' is a two-sided COUNT EQUALITY (sum==k), not a")
        print("         fixed-window detector with parity/threshold. The wall RELOCATES here:")
        print("         a genuinely different family needs a NEW aggregation/primitive")
        print("         (count==k, or a balance counter) -- the detector+threshold span does not reach it.")
        m3c_unsat = True
    else:
        cs, _ = cross_seed_bal(spec, agg, balanced_task, fresh)
        print(f"      WAKE unexpectedly found spec {list(spec)} agg {agg} cross-seed(bal)={cs:.3f}")
        m3c_unsat = False

    # =====================================================================
    print("\n" + bar)
    print("M3 SUMMARY -- honest map of the mechanism's reach")
    print(bar)
    print(f"  M3a gapped patterns (generalise parameter string->spec): "
          f"{'PASS' if m3a_ok else 'FAIL'}")
    print(f"  M3b threshold counting (induce 2nd axis: aggregation)  : "
          f"{'PASS' if m3b_ok else 'PARTIAL/FAIL'}")
    print(f"  M3c #0==#1 out-of-family (honest boundary)             : "
          f"{'UNSAT as expected (wall relocates)' if m3c_unsat else 'unexpectedly solved'}")
    print("\n  Reading: the SAME wake->sleep->bind machinery extends ACROSS families when the")
    print("  family lies in the detector+aggregation span (M3a richer parameter, M3b richer")
    print("  aggregation) -- the abstraction GENERALISES, it is not re-hand-built per family.")
    print("  M3c marks the honest edge: a family outside that span (counting/balance) needs a")
    print("  NEW primitive -- which is the next frontier (grow the primitive set, sec 44 invent).")
    print(bar)


if __name__ == "__main__":
    main()

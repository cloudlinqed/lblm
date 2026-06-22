#!/usr/bin/env python3
"""
Path B, step: PRIMITIVE INVENTION (grow the toolbox from a failure).

induce.py composes from a FIXED library. Here, when the fixed library BREAKS on a task, the system
searches a GENERATIVE space of recurrent computations -- primitives parameterised by a predicate over
(previous bit, current bit) and an aggregation (count mod m / ever / max-run) -- and tries to INVENT
the missing primitive that solves the task (content-disjoint held-out + scramble-clean). This is one
step toward open-ended learning-to-compute, not the whole goal: invention is itself bounded by the
generative space, which has its own (extensible) frontier.
"""
import induce
from induce import gen, split, eval_comp, make_prims, induce as base_induce

# generative DSL: predicate over (prev, cur) x aggregation
PREDS = {
    "cur":  lambda p, c: c,
    "prev": lambda p, c: p,
    "and":  lambda p, c: p & c,
    "or":   lambda p, c: p | c,
    "xor":  lambda p, c: p ^ c,
    "01":   lambda p, c: int(p == 0 and c == 1),
    "10":   lambda p, c: int(p == 1 and c == 0),
    "11":   lambda p, c: int(p == 1 and c == 1),
    "00":   lambda p, c: int(p == 0 and c == 0),
    "eq":   lambda p, c: int(p == c),
}
AGGS = ["cnt2", "cnt3", "cnt4", "ever", "maxrun"]


def gen_prim(pred, agg):
    pf = PREDS[pred]

    def f(s):
        vals = [pf(s[i - 1], s[i]) for i in range(1, len(s))]
        if agg == "cnt2":
            return sum(vals) % 2
        if agg == "cnt3":
            return sum(vals) % 3
        if agg == "cnt4":
            return sum(vals) % 4
        if agg == "ever":
            return int(any(vals))
        best = cur = 0
        for v in vals:
            cur = cur + 1 if v else 0
            if cur > best:
                best = cur
        return min(best, 4)
    return f


def invent(task_fn, base, n=600, seed=0):
    tr, dv, te = split(gen(task_fn, n, seed), seed)
    trs, dvs, _ = split(gen(task_fn, n, seed, scramble=True), seed)
    found = None
    for pred in PREDS:
        for agg in AGGS:
            nm = f"{pred}:{agg}"
            prims = {**base, nm: gen_prim(pred, agg)}
            for comb in [[nm]] + [[nm, b] for b in base]:
                if eval_comp(tr, dv, prims, comb) > 0.98:
                    te_acc = eval_comp(tr + dv, te, prims, comb)
                    sc = eval_comp(trs, dvs, prims, comb)        # scramble must fail
                    if te_acc > 0.98 and sc < 0.7:
                        if found is None or len(comb) < len(found[1]):
                            found = (te_acc, comb, nm)
                        if len(comb) == 1:
                            return found
    return found


def trans(s, a, b):
    return sum(1 for i in range(1, len(s)) if s[i - 1] == a and s[i] == b)


def main():
    base = make_prims(False)
    # tasks the FIXED library cannot do -- but most are invent-able within the generative DSL.
    extra = {
        "01-parity": lambda s: trans(s, 0, 1) % 2,
        "10-parity": lambda s: trans(s, 1, 0) % 2,
        "saw-11":    lambda s: int(trans(s, 1, 1) > 0),
        "11-parity": lambda s: trans(s, 1, 1) % 2,
        "010-parity": lambda s: sum(1 for i in range(2, len(s)) if s[i - 2] == 0 and s[i - 1] == 1 and s[i] == 0) % 2,
    }
    print("Fixed library breaks on these -> can the system INVENT the missing primitive?\n")
    print(f"{'task':11} | base test | result")
    for name, fn in extra.items():
        _, bcomp, bte = base_induce(fn, base)
        if bte > 0.97:
            print(f"{name:11} | {bte:.2f}      | base already solved: {bcomp}")
            continue
        inv = invent(fn, base)
        if inv:
            print(f"{name:11} | {bte:.2f}      | INVENTED [{inv[2]}]  comp={inv[1]}  test={inv[0]:.2f}")
        else:
            print(f"{name:11} | {bte:.2f}      | NOT invented (beyond the 2-bit-predicate generative space)")


if __name__ == "__main__":
    main()

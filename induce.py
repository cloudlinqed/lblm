#!/usr/bin/env python3
"""
Path B -- open-ended INDUCTION of bit-native computations.

Beyond *selecting* from a tiny hand-made family (cycles 12-13), this searches a LIBRARY of recurrent
bit-stream primitives and INDUCES (composes) the computation that solves a task from examples,
generalising to content-disjoint held-out sequences. The goal is to map the FRONTIER: which
computations bit-native induction can learn, where it breaks, and whether adding a primitive extends
the reach. This is program-induction over bits -- a mechanism unlike LLM pattern-fitting.

Honesty controls: a random-answer SCRAMBLE (a genuine program must fail it) and CONTENT-DISJOINT
held-out (memorising raw bits fails it). The point is to find the boundary, not claim it solves all.
"""
import random, itertools
from collections import Counter, defaultdict

L = 12  # sequence length


def longest_run(s):
    best = cur = 0
    for b in s:
        cur = cur + 1 if b == 1 else 0
        if cur > best:
            best = cur
    return best


def trans01(s):
    return sum(1 for i in range(1, len(s)) if s[i - 1] == 0 and s[i] == 1)


def make_prims(extended=False):
    P = {
        "latch1": lambda s: s[0],
        "latch2": lambda s: s[0] * 2 + s[1],
        "latch3": lambda s: s[0] * 4 + s[1] * 2 + s[2],
        "cmod2": lambda s: sum(s) % 2,
        "cmod3": lambda s: sum(s) % 3,
        "cmod4": lambda s: sum(s) % 4,
        "cmod5": lambda s: sum(s) % 5,
        "cbucket": lambda s: min(sum(s), 7),
        "maxrun": lambda s: min(longest_run(s), 4),
        "last1": lambda s: s[-1],
        "last2": lambda s: s[-2] * 2 + s[-1],
        "firsteqlast": lambda s: int(s[0] == s[-1]),
    }
    if extended:
        P["transcount"] = lambda s: min(trans01(s), 5)
        P["transmod2"] = lambda s: trans01(s) % 2
    return P


TASKS = {
    "parity":    lambda s: sum(s) % 2,
    "recall":    lambda s: s[0] * 2 + s[1],          # echo the first 2 bits
    "mod4":      lambda s: sum(s) % 4,
    "majority":  lambda s: int(sum(s) > len(s) // 2),
    "maxrun>=3": lambda s: int(longest_run(s) >= 3),
    "compose":   lambda s: (s[0] << 1) | (sum(s) % 2),  # type AND parity together
    "automaton": lambda s: trans01(s) % 2,            # parity of 01-transitions (needs a transition feature)
}
N_ANS = {"parity": 2, "recall": 4, "mod4": 4, "majority": 2, "maxrun>=3": 2, "compose": 4, "automaton": 2}


def gen(task_fn, n, seed, scramble=False):
    rng = random.Random(seed); seen = set(); items = []
    while len(items) < n:
        s = tuple(rng.randint(0, 1) for _ in range(L))
        if s in seen:
            continue
        seen.add(s); items.append([list(s), task_fn(list(s))])
    if scramble:
        ans = [a for _, a in items]; rng.shuffle(ans)
        for i in range(len(items)):
            items[i][1] = ans[i]
    return items


def split(items, seed):
    rng = random.Random(seed); idx = list(range(len(items))); rng.shuffle(idx)
    a = len(items) // 5; b = 2 * a
    return ([items[i] for i in idx[b:]], [items[i] for i in idx[a:b]], [items[i] for i in idx[:a]])


def address(s, prims, names):
    return tuple(prims[nm](s) for nm in names)


def eval_comp(tr, ev, prims, names):
    table = defaultdict(Counter)
    for s, a in tr:
        table[address(s, prims, names)][a] += 1
    pred = {k: c.most_common(1)[0][0] for k, c in table.items()}
    gm = Counter(a for _, a in tr).most_common(1)[0][0]
    return sum(1 for s, a in ev if pred.get(address(s, prims, names), gm) == a) / len(ev)


def induce(task_fn, prims, maxk=3, n=600, seed=0, scramble=False):
    tr, dv, te = split(gen(task_fn, n, seed, scramble), seed)
    names = list(prims); best = (-1.0, None)
    for k in range(1, maxk + 1):
        for combo in itertools.combinations(names, k):
            acc = eval_comp(tr, dv, prims, list(combo))
            if acc > best[0] + 1e-9 or (abs(acc - best[0]) < 1e-9 and best[1] is not None and len(combo) < len(best[1])):
                best = (acc, list(combo))
    te_acc = eval_comp(tr + dv, te, prims, best[1])
    return best[0], best[1], te_acc


def main():
    prims = make_prims(False)
    print(f"Path B -- inducing computations from a {len(prims)}-primitive library (L={L})\n")
    print(f"{'task':11} | chance | devacc | testacc | scram | induced composition")
    for name, fn in TASKS.items():
        dv, comp, te = induce(fn, prims)
        sd, _, _ = induce(fn, prims, scramble=True)
        verdict = "SOLVED" if te > 0.97 else ("partial" if te > 0.7 else "BREAKS")
        print(f"{name:11} | {1/N_ANS[name]:.2f}   | {dv:.3f}  | {te:.3f}   | {sd:.2f}  | {comp}  [{verdict}]")
    print("\nExtensibility -- add transition primitives, re-induce the task that broke ('automaton'):")
    pe = make_prims(True)
    dv, comp, te = induce(TASKS["automaton"], pe)
    print(f"automaton+ext | dev {dv:.3f} | test {te:.3f} | {comp}")


if __name__ == "__main__":
    main()

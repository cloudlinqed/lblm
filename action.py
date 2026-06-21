#!/usr/bin/env python3
"""
Cycle 14 — prediction -> POLICY: a bit-native contextual-bandit agent (criteria #5 and #9 of the
intelligence framing). The agent sees a bit context, forms an address via a chosen recurrent
computation (the cycle-12 family), CHOOSES an action, and receives a reward bit. It learns the
policy online from REWARD (no labels). Two questions:

  Part A  does reward improve over trials, and does it GENERALISE to held-out contexts?
          (the right representation count-* generalises; raw hold-all memorises and fails held-out)
  Part B  can the meta-learner SELECT the right computation FROM REWARD alone
          (recover count-2 for a parity-reward task, count-4 for a mod-4-reward task)?

Fully bit-native, sequential / CPU-light.
"""
import random, statistics
import aggregate, learn_state

FAMILY = learn_state.FAMILY


def addr(p, member):
    kind, par = member
    return tuple(p[:par]) if kind == "hold" else (sum(p) % par,)


def optimal(p, task):
    return sum(p) % (2 if task == "parity" else 4)


def nA(task):
    return 2 if task == "parity" else 4


def run_online(train_ctx, member, task, T, seed):
    rng = random.Random(seed); Q = {}; curve = []; win = []; W = max(1, T // 8); A = nA(task)
    for _ in range(T):
        p = train_ctx[rng.randrange(len(train_ctx))]
        q = Q.setdefault(addr(p, member), [[0.0, 0] for _ in range(A)])
        vals = [(s / c) if c > 0 else 1.0 for s, c in q]          # optimistic-greedy: explore unseen
        act = max(range(A), key=lambda a: vals[a])
        r = 1 if act == optimal(p, task) else 0
        q[act][0] += r; q[act][1] += 1; win.append(r)
        if len(win) == W:
            curve.append(round(statistics.mean(win), 3)); win = []
    return Q, curve


def greedy_eval(Q, ctx_list, member, task):
    A = nA(task); ok = 0
    for p in ctx_list:
        q = Q.get(addr(p, member))
        if q is None:
            act = 0                                                # never-seen address -> no info
        else:
            vals = [(s / c) if c > 0 else 1.0 for s, c in q]
            act = max(range(A), key=lambda a: vals[a])
        ok += (act == optimal(p, task))
    return ok / len(ctx_list)


def main():
    F = 8
    pats = aggregate.producible(F)
    print(f"contexts (producible F={F}): {len(pats)}")
    T = 3000

    print("\n=== Part A: improve over trials + GENERALISE (parity-act, chance=0.50) ===")
    tr, dv, te = learn_state.split3_patterns(pats, lambda p: sum(p) % 2, 0)
    for member, name in [(("count", 2), "count-2 (right representation)"), (("hold", F), "hold-all (raw inputs)  ")]:
        Q, curve = run_online(tr, member, "parity", T, 0)
        print(f" {name}  reward over trials: {curve}   held-out: {greedy_eval(Q, te, member, 'parity'):.2f}")

    print("\n=== Part B: learn-the-computation FROM REWARD (pick by dev reward, parsimony tiebreak) ===")
    for task, exp in [("parity", ("count", 2)), ("mod4", ("count", 4))]:
        tr, dv, te = learn_state.split3_patterns(pats, lambda p: sum(p) % nA(task), 0)
        Qs, scored = {}, []
        for mem in FAMILY + [("hold", F)]:
            Q, _ = run_online(tr, mem, task, T, 0); Qs[mem] = Q
            scored.append((mem, greedy_eval(Q, dv, mem, task), learn_state.state_bits(mem)))
        best = max(d for _, d, _ in scored)
        near = [(m, b) for m, d, b in scored if d >= best - 0.05]
        pick = min(near, key=lambda x: x[1])[0]
        ok = "OK" if pick == exp else "MISMATCH"
        print(f" {task:6} chance={1 / nA(task):.2f}  LEARNED {pick} (expected {exp}) [{ok}]  "
              f"held-out reward={greedy_eval(Qs[pick], te, pick, task):.2f}")
        print("    dev reward by member: " + "  ".join(f"{k}{p}={d:.2f}" for (k, p), d, _ in scored))


if __name__ == "__main__":
    main()

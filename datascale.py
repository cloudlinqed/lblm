#!/usr/bin/env python3
"""
Does increasing the dev set / training on larger data make sense?  Two learning curves separate the
two failure modes (task: popcount mod 4, F=10, content-disjoint split):

  Curve 1 (STATISTICAL axis) — does MORE DEV data make SELECTION reliable?
    Shrink the dev set used to pick the family member; does it mis-pick (overfit) at small dev and
    recover the right computation (count-4) as dev grows?  -> if yes, more data licenses a freer search.

  Curve 2 (REPRESENTATIONAL axis) — does MORE TRAIN data fix the FLEXIBLE model?
    The raw-input lookup (hold-all-F) vs the structured count-4, test acc vs train size.
    -> if hold-all stays at chance regardless of train size, more data canNOT fix a wrong feature.
Sequential / CPU-light.
"""
import random, statistics
import bench, blm, region, learn_state, aggregate
R = 6


def per_item_correct(train_items, eval_items, member, wk=3, ep=150):
    p = bench.params(R, "learned", 1, 0, ep, "uniform")
    p["A"] = 3 + learn_state.state_bits(member) + wk; p["alloc_radius"] = 0
    pooled = []
    for it in train_items:
        pooled += learn_state.build_pairs("".join(map(str, it["seq"])), member, wk)
    mch = blm.Machine(p); mch.train(pooled)
    return [1 if learn_state.gen(mch, it["seq"][:it["ans_start"]], len(it["answer"]), member, wk) == it["answer"]
            else 0 for it in eval_items]


def acc(corr, idxs):
    return sum(corr[i] for i in idxs) / len(idxs)


def main():
    F, m = 10, 4
    pats = aggregate.producible(F)
    tr, dv, te = learn_state.split3_patterns(pats, lambda p: sum(p) % m, 0)
    train, dev, test = aggregate.items_of(tr, m, False), aggregate.items_of(dv, m, False), aggregate.items_of(te, m, False)
    print(f"F={F} mod {m}:  train={len(train)} dev={len(dev)} test={len(test)}   chance=0.25")

    members = learn_state.FAMILY
    dev_corr = {mem: per_item_correct(train, dev, mem) for mem in members}
    test_acc = {mem: statistics.mean(per_item_correct(train, test, mem)) for mem in members}
    print("[member TEST acc] " + "  ".join(f"{k}{p}={test_acc[(k, p)]:.2f}" for k, p in members))

    print("\nCurve 1 — SELECTION vs DEV size (does it reliably pick count-4?):")
    print(" devN | P(correct pick) | mean test-acc of pick")
    rng = random.Random(0); S = 30; N = len(dev)
    for devN in [2, 4, 8, 16, N]:
        ok, tas = 0, []
        for _ in range(S):
            idxs = rng.sample(range(N), min(devN, N))
            scored = [(mem, acc(dev_corr[mem], idxs), learn_state.state_bits(mem)) for mem in members]
            best = max(d for _, d, _ in scored); near = [(mem, b) for mem, d, b in scored if d >= best - 0.05]
            pick = min(near, key=lambda x: x[1])[0]
            ok += (pick == ("count", 4)); tas.append(test_acc[pick])
        print(f" {devN:>4} |      {ok / S:.2f}       |        {statistics.mean(tas):.2f}")

    print("\nCurve 2 — TEST acc vs TRAIN size:  flexible raw-input (hold-all) vs structured count-4")
    print(" trainN | hold-all test | count-4 test")
    for frac in [0.25, 0.5, 1.0]:
        n = max(2, int(len(train) * frac)); sub = train[:n]
        ha = statistics.mean(per_item_correct(sub, test, ("hold", F)))
        c4 = statistics.mean(per_item_correct(sub, test, ("count", 4)))
        print(f" {n:>6} |     {ha:.2f}      |     {c4:.2f}")


if __name__ == "__main__":
    main()

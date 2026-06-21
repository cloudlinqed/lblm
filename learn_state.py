#!/usr/bin/env python3
"""
Cycle 12 — LEARN the recurrent computation (the central question).

Instead of me hand-picking the recurrent state per task, define a small STRUCTURED FAMILY and let a
meta-learner pick the member that generalises on a content-disjoint DEV set (tie-break: fewest state
bits = simplest computation). Family:
  * hold-k  : latch the first k bits (the cycle-9 recall latch)
  * count-m : running popcount mod m of pre-boundary bits, frozen (the cycle-10/11 accumulator)
Address = region/position bits ++ state-member bits ++ small window.  Readout = the plain vote.

Success = it RECOVERS the right computation from data, without being told:
  recall(echo) -> hold-2 ; parity -> count-2 ; popcount mod 4 -> count-4.
Sequential / CPU-light.
"""
import itertools, random, statistics
import bench, blm, gated, region, aggregate, multi
R = 6
FAMILY = [("hold", 1), ("hold", 2), ("hold", 3), ("count", 2), ("count", 3), ("count", 4)]


def state_bits(member):
    kind, par = member
    return par if kind == "hold" else max(1, (par - 1).bit_length())


def state_feats(seq_str, t, member):
    kind, par = member
    if kind == "hold":
        return [int(seq_str[i]) for i in range(par)]                 # first par bits, held
    nb = state_bits(member)
    idx = seq_str.find("111"); cut = t if (idx < 0 or t <= idx) else idx
    c = sum(1 for ch in seq_str[:cut] if ch == "1") % par            # running popcount mod m, frozen
    return [(c >> (nb - 1 - i)) & 1 for i in range(nb)]


def build_pairs(seq_str, member, wk):
    out = []
    for i in range(len(seq_str) - R):
        win = [int(c) for c in seq_str[i:i + R]]
        t = i + R
        addr = tuple(region.region_bits(seq_str, t) + state_feats(seq_str, t, member) + (win[-wk:] if wk else win))
        out.append((addr, int(seq_str[t])))
    return out


def gen(mch, prefix, n, member, wk):
    seq = list(prefix)
    window = seq[-R:] if len(seq) >= R else [0] * (R - len(seq)) + seq
    out = []
    for _ in range(n):
        ss = "".join(map(str, seq))
        addr = tuple(region.region_bits(ss, len(seq)) + state_feats(ss, len(seq), member) + (window[-wk:] if wk else window))
        b, _ = mch.predict(addr); out.append(b); seq.append(b); window = window[1:] + [b]
    return out


def eval_member(train_items, test_items, member, wk=3, ep=150):
    p = bench.params(R, "learned", 1, 0, ep, "uniform")
    p["A"] = 3 + state_bits(member) + wk; p["alloc_radius"] = 0
    pooled = []
    for it in train_items:
        pooled += build_pairs("".join(map(str, it["seq"])), member, wk)
    mch = blm.Machine(p); mch.train(pooled)
    return sum(1 for it in test_items
               if gen(mch, it["seq"][:it["ans_start"]], len(it["answer"]), member, wk) == it["answer"]) / len(test_items)


def learn_and_test(train, dev, test, label, expected):
    scored = [(m, eval_member(train, dev, m), state_bits(m)) for m in FAMILY]
    best = max(d for _, d, _ in scored)
    cands = [(m, b) for m, d, b in scored if d >= best - 0.05]
    picked = min(cands, key=lambda x: x[1])[0]                        # fewest state bits among near-best
    ta = eval_member(train, test, picked)
    ok = "OK" if picked == expected else "MISMATCH"
    print(f"\n{label}: LEARNED {picked}  (expected {expected}) [{ok}]   held-out test acc = {ta:.2f}")
    print("   dev acc by member: " + "  ".join(f"{k}{p}={d:.2f}" for (k, p), d, _ in scored))
    return picked, ta


def split3_patterns(pats, key, seed):
    rng = random.Random(seed)
    from collections import defaultdict
    by = defaultdict(list)
    for p in pats:
        by[key(p)].append(p)
    tr, dv, te = [], [], []
    for g, ps in by.items():
        rng.shuffle(ps); a = max(1, len(ps) // 5); b = 2 * a
        te += ps[:a]; dv += ps[a:b]; tr += ps[b:]
    return tr, dv, te


def agg_task(F, m):
    tr, dv, te = split3_patterns(aggregate.producible(F), lambda p: sum(p) % m, 0)
    f = lambda pats: aggregate.items_of(pats, m, False)
    return f(tr), f(dv), f(te)


def echo_task(L=12, K=24):
    items = multi.dataset(L, K, 0, "echo", False)
    tr, dv, te = gated.split(K, 0)
    pick = lambda S: [it for it in items if it["body_id"] in S]
    return pick(tr), pick(dv), pick(te)


def main():
    print("=== LEARN the recurrent computation (pick the family member from data) ===")
    tr, dv, te = echo_task();           learn_and_test(tr, dv, te, "recall (echo)", ("hold", 2))
    tr, dv, te = agg_task(8, 2);        learn_and_test(tr, dv, te, "parity (mod 2)", ("count", 2))
    tr, dv, te = agg_task(8, 4);        learn_and_test(tr, dv, te, "popcount mod 4", ("count", 4))


if __name__ == "__main__":
    main()

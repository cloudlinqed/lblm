#!/usr/bin/env python3
"""
Cycle 13 — does SELECTION compose? A task needing two computations at once:
  [type bit t][F feature bits][111][answer = (t, parity(t+features))][000]
  bit1 = a HELD feature (the type); bit2 = a COMPUTED aggregate (parity of the whole pre-boundary).
A single hold-member misses the parity; a single count-member loses the type; only the COMBINATION
{hold-1, count-2} encodes both. The meta-learner chooses over singles AND hold+count pairs, by
content-disjoint dev generalisation, tie-broken toward fewest total state bits. Success = it learns
the COMBINATION {hold-1, count-2} from data. Sequential / CPU-light.
"""
import itertools, random, statistics
import bench, blm, gated, region, learn_state, aggregate
R = 6


def patterns(F):
    # feature patterns: no 111, end in 0, AND start in 0 (so [t]++feats never makes a leading 111)
    return [p for p in aggregate.producible(F) if p[0] == 0]


def items_of(pats, scramble):
    items = []
    for p in pats:
        for t in (0, 1):
            P = (t + sum(p)) & 1
            ans = [hash((p, t, 0)) & 1, hash((p, t, 1)) & 1] if scramble else [t, P]
            items.append({"seq": [t] + list(p) + [1, 1, 1] + ans + [0, 0, 0],
                          "answer": ans, "ans_start": 1 + len(p) + 3})
    return items


def feats(seq_str, t, members):
    out = []
    for m in members:
        out += learn_state.state_feats(seq_str, t, m)
    return out


def bits(members):
    return sum(learn_state.state_bits(m) for m in members)


def build_pairs(seq_str, members, wk):
    out = []
    for i in range(len(seq_str) - R):
        win = [int(c) for c in seq_str[i:i + R]]
        t = i + R
        out.append((tuple(region.region_bits(seq_str, t) + feats(seq_str, t, members) + (win[-wk:] if wk else win)),
                    int(seq_str[t])))
    return out


def gen(mch, prefix, n, members, wk):
    seq = list(prefix); window = seq[-R:] if len(seq) >= R else [0] * (R - len(seq)) + seq
    out = []
    for _ in range(n):
        ss = "".join(map(str, seq))
        a = tuple(region.region_bits(ss, len(seq)) + feats(ss, len(seq), members) + (window[-wk:] if wk else window))
        b, _ = mch.predict(a); out.append(b); seq.append(b); window = window[1:] + [b]
    return out


def evalc(train, test, members, wk=3, ep=150):
    p = bench.params(R, "learned", 1, 0, ep, "uniform"); p["A"] = 3 + bits(members) + wk; p["alloc_radius"] = 0
    pooled = []
    for it in train:
        pooled += build_pairs("".join(map(str, it["seq"])), members, wk)
    mch = blm.Machine(p); mch.train(pooled)
    return sum(1 for it in test if gen(mch, it["seq"][:it["ans_start"]], 2, members, wk) == it["answer"]) / len(test)


def main():
    F = 8
    pats = patterns(F)
    tr_p, dv_p, te_p = learn_state.split3_patterns(pats, lambda p: sum(p) % 2, 0)
    train, dev, test = items_of(tr_p, False), items_of(dv_p, False), items_of(te_p, False)
    singles = [(m,) for m in learn_state.FAMILY]
    pairs = [(("hold", k), ("count", m)) for k in (1, 2) for m in (2, 4)]
    cands = singles + pairs
    scored = [(c, evalc(train, dev, list(c)), bits(c)) for c in cands]
    best = max(d for _, d, _ in scored)
    near = [(c, b) for c, d, b in scored if d >= best - 0.05]
    picked = min(near, key=lambda x: x[1])[0]
    ta = evalc(train, test, list(picked))
    print(f"=== Cycle 13: does selection COMPOSE? (answer = [type, parity], F={F}) ===")
    print(f"LEARNED {picked}  (expected (('hold',1),('count',2)))   held-out test acc = {ta:.2f}")
    print("dev by candidate:")
    for c, d, b in sorted(scored, key=lambda x: -x[1]):
        print(f"   {str(c):>40}  dev={d:.2f}  bits={b}")


if __name__ == "__main__":
    main()

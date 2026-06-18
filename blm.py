#!/usr/bin/env python3
"""
Bit-Native Predictive Machine — learned binary addresses (mobile-template / binary-SOM),
now with a RECURRENT address option (the verification workflow's #1 recommended change).

A unit = (address: A bits, value: 1 bit, strength w). The query address is built from the
current R-bit register plus an optional h-bit recurrent state that carries history that has
already shifted out of the register:

    addr_mode = register : address = R-bit window                       (A = R)   [memoryless]
    addr_mode = shift    : address = R-bit window ++ last h dropped bits (A = R+h) [wider window]
    addr_mode = fold     : address = R-bit window ++ rotate/xor fold of ALL dropped bits
                           (A = R+h)  [fixed-width compressed history -> tests Hamming smoothness]

Retrieval = weighted Hamming proximity over the A-bit address, softened by the kernel
exp(-beta * weighted_hamming)  (== exp(beta*(sim-A)) only under uniform weights).
Output = signed strength-weighted vote -> threshold. Learning is a gradient-free binary SOM
(reinforce+pull nearest correct, weaken+push wrong-voters, allocate when no correct neighbour,
anneal, merge). mode=frozen keeps addresses immobile; mode=mobile lets them move.

Hardening applied after adversarial verification:
  * separate move-RNG so frozen/mobile share an identical training-shuffle order (clean A/B);
  * leave-one-window-out (LOWO) generalisation metric, so train_acc=1.0 is not mistaken
    for learning (it is in-sample memorisation of an injective table);
  * the always-0 baseline's autoregressive generation is reported (it reproduces stream B's
    00000 tail at R=8), so the zero-bias contribution to any "win" is visible.

Data is the project's REAL streams. No invented/mock data; perturbations are eval only.
"""

import argparse
import json
import random

STREAM_A = "0101011111000"   # 01010 111 11 000
STREAM_B = "1010011100000"   # 10100 111 00 000


def bits(s):
    return tuple(int(c) for c in s)


def hamming(a, b):
    return sum(1 for x, y in zip(a, b) if x != y)


def fold_state(s, d, addr_mode, h):
    """Update the h-bit recurrent state with the bit `d` that just left the register."""
    if addr_mode == "register" or h == 0:
        return s
    if addr_mode == "shift":
        return (list(s) + [d])[-h:]
    if addr_mode == "fold":
        s2 = list(s[1:]) + [s[0]]                       # rotate left
        if d:
            mask = [1 if i % 2 == 0 else 0 for i in range(h)]
            s2 = [b ^ m for b, m in zip(s2, mask)]      # fold dropped bit into the whole state
        return s2
    raise ValueError(addr_mode)


def make_pairs(stream, R, addr_mode="register", h=0):
    """Every step -> (address, next bit). Address carries recurrent history per addr_mode."""
    pairs = []
    s = [0] * h if addr_mode != "register" else []
    for i in range(len(stream) - R):
        window = list(bits(stream[i:i + R]))
        pairs.append((tuple(window + list(s)), int(stream[i + R])))
        s = fold_state(s, int(stream[i]), addr_mode, h)   # the bit leaving as i advances
    return pairs


class Machine:
    def __init__(self, p):
        self.p = p
        self.units = []
        self.rng = random.Random(p["seed"])            # structural: shuffle order (mode-independent)
        self.move_rng = random.Random(p["seed"] * 7919 + 1)   # address moves only
        self.weights = [1.0] * p["A"]                  # per-address-bit importance (uniform by default)

    def compute_weights(self, pairs):
        """Per-address-bit importance for the weighted-Hamming retrieval. Modes:
        * uniform     : all 1.0 (reduces to plain Hamming).
        * mi          : mutual information of each bit with the next bit. Rewards globally
                        PREDICTIVE bits — but a class-invariant predictor (e.g. a boundary)
                        gets high weight yet zero discrimination value.
        * contrastive : weight a bit by how often it SEPARATES collision partners — near
                        addresses (Hamming <= radius) with OPPOSITE labels. This targets
                        DISCRIMINATIVE bits (the ones that tell confusable cases apart) and
                        ignores predictive-but-invariant bits. (query-conditional in spirit)"""
        A, N = self.p["A"], len(pairs)
        mode = self.p.get("weight_mode", "uniform")
        if mode == "uniform" or N == 0:
            self.weights = [1.0] * A
            return
        if mode == "mi":
            import math
            W = []
            for i in range(A):
                c = {}
                for q, y in pairs:
                    c[(q[i], y)] = c.get((q[i], y), 0) + 1
                mi = 0.0
                for xi in (0, 1):
                    px = sum(c.get((xi, yy), 0) for yy in (0, 1)) / N
                    if px == 0:
                        continue
                    for yy in (0, 1):
                        pxy = c.get((xi, yy), 0) / N
                        py = sum(c.get((xx, yy), 0) for xx in (0, 1)) / N
                        if pxy > 0 and py > 0:
                            mi += pxy * math.log2(pxy / (px * py))
                W.append(mi)
            mean = sum(W) / A
            self.weights = [(w / mean if mean > 0 else 1.0) for w in W]
            return
        if mode == "contrastive":
            seen = {}
            for q, y in pairs:
                seen.setdefault(tuple(q), set()).add(y)
            entries = [(q, y) for q, ys in seen.items() for y in ys]
            radius = self.p.get("contrast_radius", 3)
            counts = [0.0] * A
            for i in range(len(entries)):
                qa, ya = entries[i]
                for j in range(i + 1, len(entries)):
                    qb, yb = entries[j]
                    if ya == yb:
                        continue
                    diff = [k for k in range(A) if qa[k] != qb[k]]
                    if 0 < len(diff) <= radius:        # a genuine confusion: close, opposite label
                        for k in diff:
                            counts[k] += 1
            mean = sum(counts) / A
            self.weights = [(c / mean if mean > 0 else 1.0) for c in counts]
            return
        self.weights = [1.0] * A

    def pressure(self, q):
        beta, A, W = self.p["beta"], self.p["A"], self.weights
        P = 0.0
        for u in self.units:
            a = u["addr"]
            wham = sum(W[i] for i in range(A) if a[i] != q[i])   # weighted Hamming distance
            P += u["w"] * (2 * u["val"] - 1) * pow(2.718281828459045, -beta * wham)
        return P

    def predict(self, q):
        if not self.units:
            return self.p["tie"], 0.0
        P = self.pressure(q)
        if P > 0:
            return 1, P
        if P < 0:
            return 0, P
        return self.p["tie"], P

    def _local_weights(self, cand):
        """Query-conditional weights: among candidate units near the query, weight bits that
        SEPARATE opposite-label pairs (Hamming <= radius). Different queries -> different
        confusions -> different weights = attention, not a fixed metric."""
        A, radius = self.p["A"], self.p.get("contrast_radius", 3)
        counts, npairs = [0.0] * A, 0
        for i in range(len(cand)):
            for j in range(i + 1, len(cand)):
                if cand[i]["val"] == cand[j]["val"]:
                    continue
                diff = [k for k in range(A) if cand[i]["addr"][k] != cand[j]["addr"][k]]
                if 0 < len(diff) <= radius:
                    for k in diff:
                        counts[k] += 1
                    npairs += 1
        if npairs == 0 or sum(counts) == 0:
            return [1.0] * A
        mean = sum(counts) / A
        return [c / mean for c in counts]

    def predict_c(self, q):
        """Query-conditional readout: compute local weights from the confusion near q, then vote."""
        if not self.units:
            return self.p["tie"], 0.0
        beta, A = self.p["beta"], self.p["A"]
        ranked = sorted(self.units, key=lambda u: hamming(q, u["addr"]))
        W = self._local_weights(ranked[:self.p.get("cond_k", 16)])
        P = 0.0
        for u in self.units:
            a = u["addr"]
            wham = sum(W[i] for i in range(A) if a[i] != q[i])
            P += u["w"] * (2 * u["val"] - 1) * pow(2.718281828459045, -beta * wham)
        return (1 if P > 0 else 0 if P < 0 else self.p["tie"]), P

    def _readout(self, q):
        return self.predict_c(q) if self.p.get("weight_mode") == "conditional" else self.predict(q)

    def _pull(self, addr, q, prob, max_move):
        diff = [i for i in range(len(addr)) if addr[i] != q[i]]
        self.move_rng.shuffle(diff)
        for i in diff[:max_move]:
            if self.move_rng.random() < prob:
                addr[i] = q[i]

    def _push(self, addr, q, prob, max_move):
        same = [i for i in range(len(addr)) if addr[i] == q[i]]
        self.move_rng.shuffle(same)
        for i in same[:max_move]:
            if self.move_rng.random() < prob:
                addr[i] = 1 - addr[i]

    def learn(self, q, y, t):
        p = self.p
        move_prob = p["move_prob"] * (p["anneal"] ** t)
        yhat, _ = self.predict(q)
        ranked = sorted(self.units, key=lambda u: hamming(q, u["addr"]))
        nearest_correct = next((u for u in ranked if u["val"] == y), None)
        if nearest_correct is None or hamming(q, nearest_correct["addr"]) > p["alloc_radius"]:
            self.units.append({"addr": list(q), "val": y, "w": p["w_init"]})
            nearest_correct = self.units[-1]
        nearest_correct["w"] += p["lr_w"]
        if p["mode"] == "mobile":
            self._pull(nearest_correct["addr"], q, move_prob, p["max_move"])
        if yhat != y:
            for u in ranked[:p["k_push"]]:
                if u["val"] != y and hamming(q, u["addr"]) <= p["push_radius"]:
                    u["w"] = max(p["w_min"], u["w"] - p["lr_w"])
                    if p["mode"] == "mobile":
                        self._push(u["addr"], q, move_prob, p["max_move"])

    def merge(self):
        bucket = {}
        for u in self.units:
            key = (tuple(u["addr"]), u["val"])
            if key in bucket:
                bucket[key]["w"] += u["w"]
            else:
                bucket[key] = {"addr": list(u["addr"]), "val": u["val"], "w": u["w"]}
        self.units = list(bucket.values())

    def train(self, pairs):
        self.compute_weights(pairs)
        t = 0
        for _ in range(self.p["epochs"]):
            order = pairs[:]
            self.rng.shuffle(order)
            for q, y in order:
                self.learn(q, y, t)
                t += 1
            self.merge()

    def generate(self, seed, n):
        p = self.p
        window = list(seed)
        s = [0] * p["h"] if p["addr_mode"] != "register" else []
        out = []
        for _ in range(n):
            b, _ = self._readout(tuple(window + list(s)))
            out.append(b)
            d = window[0]
            window = window[1:] + [b]
            s = fold_state(s, d, p["addr_mode"], p["h"])
        return out

    def generate_primed(self, prefix, n):
        """Warm-start: run the recurrent state over the FULL prefix (not just the window),
        so generation begins with the same state training accumulated. Fixes the cold-state
        failure where a recurrent machine starts at s=0 and can't see early history."""
        p = self.p
        prefix, R, h = list(prefix), p["R"], p["h"]
        window = prefix[-R:] if len(prefix) >= R else [0] * (R - len(prefix)) + prefix
        s = [0] * h if p["addr_mode"] != "register" else []
        for d in prefix[:max(0, len(prefix) - R)]:        # bits already shifted out of window
            s = fold_state(s, d, p["addr_mode"], h)
        out = []
        for _ in range(n):
            b, _ = self._readout(tuple(window + list(s)))
            out.append(b)
            d = window[0]
            window = window[1:] + [b]
            s = fold_state(s, d, p["addr_mode"], h)
        return out

    def utilisation(self):
        return len({tuple(u["addr"]) for u in self.units}), len(self.units)


def train_accuracy(m, pairs):
    return sum(1 for q, y in pairs if m.predict(q)[0] == y) / len(pairs)


def leave_one_window_out(params):
    """Honest generalisation: hold out each pair, train on the rest, test the held-out pair."""
    r, am, h = params["R"], params["addr_mode"], params["h"]
    pairs = make_pairs(STREAM_A, r, am, h) + make_pairs(STREAM_B, r, am, h)
    correct = 0
    for j in range(len(pairs)):
        m = Machine(params)
        m.train(pairs[:j] + pairs[j + 1:])
        q, y = pairs[j]
        correct += (m.predict(q)[0] == y)
    return correct / len(pairs)


def robustness(m, seed, target_head, r):
    n = len(target_head)
    survived = 0
    for i in range(r):
        flipped = list(seed)
        flipped[i] ^= 1
        survived += (m.generate(tuple(flipped), n) == list(target_head))
    return survived, r


def evaluate(m, r):
    am, h = m.p["addr_mode"], m.p["h"]
    pairs = make_pairs(STREAM_A, r, am, h) + make_pairs(STREAM_B, r, am, h)
    seed_a, tail_a = bits(STREAM_A[:r]), bits(STREAM_A[r:])
    seed_b, tail_b = bits(STREAM_B[:r]), bits(STREAM_B[r:])
    gen_a, gen_b = m.generate(seed_a, len(tail_a)), m.generate(seed_b, len(tail_b))
    distinct, total = m.utilisation()
    rob_a = robustness(m, seed_a, list(tail_a[:2]), r)
    rob_b = robustness(m, seed_b, list(tail_b[:2]), r)
    return {
        "R": r, "addr_mode": am, "h": h,
        "train_acc": round(train_accuracy(m, pairs), 4),
        "gen_A": "".join(map(str, gen_a)), "gen_A_target": "".join(map(str, tail_a)),
        "gen_A_ok": list(gen_a) == list(tail_a),
        "gen_B": "".join(map(str, gen_b)), "gen_B_target": "".join(map(str, tail_b)),
        "gen_B_ok": list(gen_b) == list(tail_b),
        "both_ok": list(gen_a) == list(tail_a) and list(gen_b) == list(tail_b),
        "robust_A": f"{rob_a[0]}/{rob_a[1]}", "robust_B": f"{rob_b[0]}/{rob_b[1]}",
        "robust_total": rob_a[0] + rob_b[0],
        "units": total, "distinct_addresses": distinct,
    }


def majority_baseline(r):
    """Always predict 0. Reports its OWN autoregressive generation so zero-bias is visible."""
    tail_a, tail_b = STREAM_A[r:], STREAM_B[r:]
    pairs_y = [y for _, y in make_pairs(STREAM_A, r) + make_pairs(STREAM_B, r)]
    return {
        "train_acc": round(sum(1 for y in pairs_y if y == 0) / len(pairs_y), 4),
        "gen_A_ok": set(tail_a) <= {"0"}, "gen_B_ok": set(tail_b) <= {"0"},
        "both_ok": set(tail_a) <= {"0"} and set(tail_b) <= {"0"},
    }


def default_params(args):
    return {
        "mode": args.mode, "R": args.R, "h": args.hist,
        "addr_mode": args.addr, "A": args.R + (args.hist if args.addr != "register" else 0),
        "epochs": args.epochs, "seed": args.seed, "beta": args.beta,
        "weight_mode": getattr(args, "weights", "uniform"),
        "contrast_radius": getattr(args, "contrast_radius", 3),
        "cond_k": getattr(args, "cond_k", 16),
        "lr_w": 0.5, "w_init": 1.0, "w_min": 0.0,
        "move_prob": args.move_prob, "anneal": args.anneal, "max_move": args.max_move,
        "alloc_radius": args.alloc_radius, "push_radius": args.push_radius,
        "k_push": 3, "tie": 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mobile", "frozen"], default="mobile")
    ap.add_argument("--addr", choices=["register", "shift", "fold"], default="register")
    ap.add_argument("--R", type=int, default=8)
    ap.add_argument("--hist", type=int, default=0, help="recurrent state width h (addr!=register)")
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--move-prob", dest="move_prob", type=float, default=0.6)
    ap.add_argument("--anneal", type=float, default=0.999)
    ap.add_argument("--max-move", dest="max_move", type=int, default=1)
    ap.add_argument("--alloc-radius", dest="alloc_radius", type=int, default=1)
    ap.add_argument("--push-radius", dest="push_radius", type=int, default=2)
    ap.add_argument("--weights", choices=["uniform", "mi", "contrastive", "conditional"],
                    default="uniform",
                    help="bit weighting (mi=marginal MI; contrastive=static separates collisions; "
                         "conditional=query-conditional/attention readout)")
    ap.add_argument("--contrast-radius", dest="contrast_radius", type=int, default=3)
    ap.add_argument("--cond-k", dest="cond_k", type=int, default=16,
                    help="conditional readout: # nearest units to derive local weights from")
    ap.add_argument("--lowo", action="store_true", help="also compute leave-one-window-out")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    params = default_params(args)
    r = args.R
    pairs = make_pairs(STREAM_A, r, args.addr, args.hist) + make_pairs(STREAM_B, r, args.addr, args.hist)
    m = Machine(params)
    m.train(pairs)
    result = evaluate(m, r)
    result["mode"] = args.mode
    result["seed"] = args.seed
    if args.lowo:
        result["lowo"] = round(leave_one_window_out(params), 4)

    if args.json:
        print(json.dumps(result))
        return

    base = majority_baseline(r)
    tag = args.addr + (f"+{args.hist}" if args.addr != "register" else "")
    print(f"=== Bit-Native Machine ({args.mode}, addr={tag}, R={r}, A={params['A']}, "
          f"seed={args.seed}, beta={args.beta}) ===")
    print(f"training pairs: {len(pairs)}   distinct contexts collide? "
          f"{len(pairs) - len({q for q, _ in pairs})} dup-addr rows")
    print()
    print(f"  train accuracy        : {result['train_acc']:.2%}   "
          f"(baseline always-0: {base['train_acc']:.2%})")
    if args.lowo:
        print(f"  LOWO generalisation   : {result['lowo']:.2%}   "
              f"(in-sample train_acc above is memorisation, not learning)")
    print(f"  generate A  {STREAM_A[:r]} -> {result['gen_A']}   "
          f"target {result['gen_A_target']}   {'OK' if result['gen_A_ok'] else 'MISS'}")
    print(f"  generate B  {STREAM_B[:r]} -> {result['gen_B']}   "
          f"target {result['gen_B_target']}   {'OK' if result['gen_B_ok'] else 'MISS'}")
    print(f"  both streams correct  : {result['both_ok']}")
    print(f"  baseline always-0     : both_ok={base['both_ok']}  "
          f"(A_ok={base['gen_A_ok']}, B_ok={base['gen_B_ok']}  <- B 'win' is zero-bias)")
    print(f"  robustness (1-bit seed flip -> right 2-bit head)")
    print(f"      stream A: {result['robust_A']}    stream B: {result['robust_B']}")
    print(f"  units / distinct addrs: {result['units']} / {result['distinct_addresses']}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Confidence-driven ACTION with consequences on a real stream.

The core predicts the next BYTE by an 8-bit greedy rollout (no peeking at the byte's true bits), and
DECIDES to commit that prediction or ABSTAIN, based on its own confidence (product of the per-bit
max-probabilities). Reward: +1 correct commit, -LAMBDA wrong commit, 0 abstain. If confidence-gating
earns more reward than always-committing, the core is using its own uncertainty to act well
(framing criteria: action #5 + confidence). Real data, online, causal, stdlib only.

Usage: python decide.py [path] [byte_cap]
"""
import sys, math
ORDERS = [0, 1, 2, 3, 4]


def load(path, cap):
    raw = open(path, "rb").read()
    return raw[:cap] if cap else raw


def stretch(p):
    p = min(1 - 1e-6, max(1e-6, p)); return math.log(p / (1 - p))


def squash(t):
    if t > 30:
        return 1 - 1e-6
    if t < -30:
        return 1e-6
    return 1.0 / (1.0 + math.exp(-t))


def key(phase, cur, hist, bp, B):
    if B == 0:
        return (phase, cur)
    return (phase, cur, bytes(hist[bp - B:bp]) if bp >= B else bytes(hist[:bp]))


def run(raw, lr=0.02, delta=0.2):
    tables = [dict() for _ in ORDERS]; w = [0.0] * len(ORDERS)
    hist = bytearray(); records = []                       # (confidence, correct)
    for bp in range(len(raw)):
        actual = raw[bp]
        # --- greedy rollout: predict the next byte WITHOUT seeing its bits (read-only) ---
        cur = 0; conf = 1.0
        for phase in range(8):
            P = squash(sum(w[k] * stretch(
                0.5 if (c := tables[k].get(key(phase, cur, hist, bp, B))) is None
                else (c[1] + delta) / (c[0] + c[1] + 2 * delta)) for k, B in enumerate(ORDERS)))
            bit = 1 if P >= 0.5 else 0
            conf *= P if bit == 1 else 1 - P
            cur = (cur << 1) | bit
        records.append((conf, 1 if cur == actual else 0))
        # --- real online step on the TRUE bits (predict + update) ---
        cur = 0
        for phase in range(8):
            yb = (actual >> (7 - phase)) & 1
            sts = []; cells = []
            for k, B in enumerate(ORDERS):
                kk = key(phase, cur, hist, bp, B); c = tables[k].get(kk)
                if c is None:
                    c = [0, 0]; tables[k][kk] = c
                sts.append(stretch((c[1] + delta) / (c[0] + c[1] + 2 * delta))); cells.append(c)
            P = squash(sum(w[k] * sts[k] for k in range(len(ORDERS))))
            err = yb - P
            for k in range(len(ORDERS)):
                w[k] += lr * err * sts[k]; cells[k][yb] += 1
            cur = (cur << 1) | yb
        hist.append(actual)
    return records


def evaluate(records, LAMBDA=4.0):
    rec = records[len(records) // 5:]; n = len(rec)            # skip warm-up
    base_acc = sum(c for _, c in rec) / n
    print(f"next-byte top-1 accuracy: {base_acc:.3f}   (chance ~ 1/256)")
    print(f"reward per byte: +1 correct commit, -{LAMBDA:g} wrong commit, 0 abstain")
    print(" threshold | coverage | acc@commit | net reward/byte")
    best = (-9, None)
    for tau in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        com = [c for conf, c in rec if conf >= tau]
        if not com:
            continue
        cov = len(com) / n; cor = sum(com); acc = cor / len(com)
        net = (cor - LAMBDA * (len(com) - cor)) / n
        print(f"   {tau:.2f}    |   {cov:.2f}   |   {acc:.3f}    |   {net:+.3f}")
        if net > best[0]:
            best = (net, tau)
    always = next(((sum(c for _, c in rec) - LAMBDA * (n - sum(c for _, c in rec))) / n for _ in [0]))
    print(f"\nalways-commit net = {always:+.3f}   |   best confidence-gated net = {best[0]:+.3f} at threshold {best[1]}")
    print("-> confidence-gating " + ("BEATS" if best[0] > always else "does not beat") + " always-committing")


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.txt"
    cap = int(sys.argv[2]) if len(sys.argv) > 2 else 200000
    raw = load(path, cap)
    print(f"corpus={path}  bytes={len(raw)}")
    evaluate(run(raw))


if __name__ == "__main__":
    main()

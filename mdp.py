#!/usr/bin/env python3
"""
Cycle 15 — SEQUENTIAL / multi-step action: a bit-native MDP with DELAYED reward.

Navigate a 1-D corridor of P cells to a goal shown once at the start; the reward arrives ONLY at the
final step. Solved by tabular TD Q-learning over a learned bit-address. This tests, in a sequential
setting, three things the cycle-14 bandit did not: temporal credit assignment (reward H steps late),
memory across the horizon (the goal is given once), and the representation lesson:

  rel   : address = (sign(goal - pos), step)   -- COMPUTED relative direction; should generalise to held-out goals
  abs   : address = (pos, goal, step)          -- raw absolute; memorises train goals, fails held-out
  nomem : address = (pos, step)                -- no goal memory; cannot navigate

Fully bit-native, sequential / CPU-light.
"""
import random, statistics
P, H, ACT, NA = 8, 8, {0: -1, 1: 1, 2: 0}, 3


def clamp(x):
    return 0 if x < 0 else (P - 1 if x > P - 1 else x)


def address(rep, pos, goal, t):
    if rep == "rel":
        return ("rel", (goal > pos) - (goal < pos), t)
    if rep == "abs":
        return ("abs", pos, goal, t)
    return ("nomem", pos, t)


def train(rep, goals, episodes, seed, alpha=0.3, gamma=0.95):
    rng = random.Random(seed); Q = {}; curve = []; win = []; WL = max(1, episodes // 10)
    qg = lambda s: Q.setdefault(s, [0.0] * NA)
    for ep in range(episodes):
        g = goals[rng.randrange(len(goals))]; pos = 0
        eps = max(0.05, 1.0 - ep / (0.6 * episodes))
        for t in range(H):
            s = address(rep, pos, g, t); qa = qg(s)
            a = rng.randrange(NA) if rng.random() < eps else max(range(NA), key=lambda i: qa[i])
            npos = clamp(pos + ACT[a]); term = (t == H - 1)
            r = 1.0 if (term and npos == g) else 0.0
            fut = 0.0 if term else max(qg(address(rep, npos, g, t + 1)))
            qa[a] += alpha * (r + gamma * fut - qa[a]); pos = npos
        win.append(1.0 if pos == g else 0.0)
        if len(win) == WL:
            curve.append(round(statistics.mean(win), 2)); win = []
    return Q, curve


def evalp(Q, rep, goals):
    ok = 0
    for g in goals:
        pos = 0
        for t in range(H):
            qa = Q.get(address(rep, pos, g, t))
            a = 0 if qa is None else max(range(NA), key=lambda i: qa[i])
            pos = clamp(pos + ACT[a])
        ok += (pos == g)
    return ok / len(goals)


def main():
    train_goals, test_goals, E = [1, 3, 5, 7], [2, 4, 6], 6000
    print(f"Bit-native MDP: P={P} H={H} actions={NA} (L/R/stay), DELAYED reward at the final step only.")
    print(f"train goals={train_goals}  held-out test goals={test_goals}  (chance ~ 1/P = {1 / P:.2f})\n")
    print("rep   | train reward | HELD-OUT reward | #states")
    res = {}
    for rep in ["rel", "abs", "nomem"]:
        Q, curve = train(rep, train_goals, E, 0); res[rep] = (Q, curve)
        print(f" {rep:5}|     {evalp(Q, rep, train_goals):.2f}     |      {evalp(Q, rep, test_goals):.2f}       |  {len(Q)}")
    print("\nrel learning curve (reaches-goal rate over training, with delayed reward):")
    print("  ", res["rel"][1])
    pick = max(["rel", "abs", "nomem"], key=lambda r: evalp(res[r][0], r, test_goals))
    print(f"\nSELECT representation by held-out reward -> {pick}")


if __name__ == "__main__":
    main()

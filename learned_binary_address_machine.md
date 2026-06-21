# Bit-Native Predictive Machine — Learned Binary Addresses (v2 design)

> Companion to `bit_native_predictive_machine.md` (v1). v1 specified the register, the
> calibration loop, and the two data streams, but left §4.3 — the "adjustable signal
> circuit" — described only in metaphor. **This document is the concrete circuit**, plus
> the upgrade we agreed on: the responder templates become *mobile learned binary
> addresses*. Reference implementation: `blm.py`.

---

## 0. The bet, restated

The machine predicts the **next bit** from a register of recent bits, autoregressively —
vocabulary of size 2, not ~50k tokens like an LLM. Each step is a single Bernoulli
decision (cheap), at the cost of ~`log2(vocab)` ≈ **16× more steps** per equivalent token
of content. The bet pays off *only if the per-step circuit is tiny*, which it is here.

Going binary doesn't remove difficulty; it **relocates** it — from *vocabulary size* to
**dependency range measured in steps**. With 1 bit/step, structure an LLM grabs in a few
steps is smeared across ~16× more of ours. So the real enemy is long-range reach in a
single left-to-right bit chain. Three escape ideas were considered:

- **Blockchain-style chaining** where the "hash" is a *learned* summary (not cryptographic —
  a crypto hash's avalanche/pre-image-resistance is exactly wrong for a model).
- **Multi-directional relations** (cellular-automaton field, any-order/masked generation).
- **A matrix with coordinates**.

They are not three gambles — they are **one substrate**: give every unit an explicit
**address** (itself a bit-vector) and let *relation = a function of addresses*. The three
ideas are just settings of that function (local→CA, dense→attention, stored-pattern→Hopfield),
and the blockchain is just a coarse coordinate axis. We chose the most powerful knob:
**learned binary addresses**.

---

## 1. Core objects

```
unit  = ( address a ∈ {0,1}^A ,  value v ∈ {0,1} ,  strength w ∈ ℝ≥0 )
query q = the current R-bit register
```

Minimal version (what `blm.py` builds first): **H = identity**, so `A = R` and the query is
just the raw register. The leap from v1 is that a unit's address is allowed to **move** in
the Hamming cube instead of being pinned to a raw template.

---

## 2. Predict one bit

```
1. q = current R-bit register.
2. for each unit m:  sim_m = R − popcount(q XOR a_m)          # Hamming closeness
                     k_m   = exp(beta · (sim_m − R))          # 1 at exact match, decays
3. pressure  P = Σ_m  w_m · (2·v_m − 1) · k_m                 # signed strength-weighted vote
4. next bit = 1 if P > 0 else 0    (exact tie → configured default)
```

Address compare is `XOR + popcount` — the cheapest operation there is. Top-k retrieval over
learned binary codes is literally **learned LSH** (locality-sensitive hashing). This is the
fastest of all variants we discussed; the speed thesis survives intact.

---

## 3. Learn one step (online, gradient-free — a binary SOM)

For each observed `(register q, true next bit y)`:

```
move_prob = base_move_prob · anneal^t              # SOM cooling over time t

ŷ = predict(q)
rank units by Hamming(q, a_m)
nearest_correct = closest unit with value == y

if no nearest_correct within alloc_radius:
    ALLOCATE a new unit at address q, value y       # adaptive resolution
    nearest_correct = that new unit

reinforce: nearest_correct.w += lr_w
           (mobile) PULL a_m one+ Hamming steps toward q   # tighten the right region

if ŷ != y:                                          # a miss
    for the nearest wrong-voters within push_radius:
        weaken: w -= lr_w
        (mobile) PUSH a_m one+ steps away from q    # contrastive separation

periodically: merge units with identical (address, value); track address utilisation
```

This is a Self-Organizing Map's competitive update **applied to move the addresses
themselves** (Kohonen's SOM moves fixed-grid weights; we move the coordinates). It is the
gradient-free route that fits v1's "adjust strengths" calibration. Gradient alternatives:
straight-through binarization (semantic hashing) and a VQ codebook. **All three are
workarounds for the same discrete-bottleneck — none is free.**

**Two things fall out for free:** (a) genuine context collisions become the *push-apart*
signal — the conflict problem trains the addresses instead of breaking the machine; (b) if
the address is recurrent, `H(prev_addr, new_bits) → addr`, the blockchain summary chain
reappears as the address trajectory.

**Modes:** `frozen` = addresses never move (a growing template/responder table with strength
learning = the original v1 circuit). `mobile` = addresses self-organise. Comparing them is
the experiment.

---

## 4. Data & task (real — no invented data)

The only data is the two streams from v1 §5:

```
STREAM_A = 0101011111000     (question 01010 / boundary 111 / answer 11 / stop 000)
STREAM_B = 1010011100000     (question 10100 / boundary 111 / answer 00 / stop 000)
```

Training pairs = every R-bit window → its next bit, over both streams. **Task:** from the
first R bits of each stream, autoregressively reproduce its tail. **Honest baseline to
beat:** always-predict-0 (the zero-bias trap — ~80% bit accuracy at R=8 yet it gets
stream A's answer `11` wrong).

---

## 5. Results (hardened build, clean A/B, 5 seeds each)

Sweep over register width R, mode, with the hardening from §8 applied (separate move-RNG so
frozen/mobile share an identical shuffle order). `both_ok` = both stream tails reproduced
exactly; `LOWO` = leave-one-window-out generalisation (hold out a pair, train on the rest,
predict it); robustness = of 16 single-bit seed flips, how many give the right 2-bit head.

| R | mode | train | both_ok | LOWO | robust/16 | units | collisions |
|---|------|------:|--------:|-----:|----------:|------:|-----------:|
| 8 | frozen | 1.00 | 1.00 | 0.60 | 13.0 | 10.0 | 0 |
| 8 | mobile | 1.00 | 1.00 | 0.60 | 13.0 | 10.0 | 0 |
| 6 | frozen | 1.00 | 1.00 | 0.79 | 8.6 | 9.0 | 0 |
| 6 | mobile | 1.00 | 1.00 | 0.77 | **9.0** | 11.8 | 0 |
| 5 | frozen | 0.975 | 0.80 | 0.64 | 4.2 | 8.8 | 0 |
| 5 | mobile | **1.00** | **1.00** | 0.66 | 4.4 | 14.0 | 0 |
| 4 | frozen | 0.756 | 0.00 | 0.46 | 1.0 | 8.4 | 4 colliding /12 |
| 4 | mobile | 0.689 | 0.00 | 0.43 | 1.4 | 12.2 | 4 colliding /12 |

**Recurrent address** (`addr = window ++ history`, frozen mode):

| addr | R | h | A | both_ok | LOWO | note |
|------|---|---|---|--------:|-----:|------|
| register | 5 | – | 5 | 0.80 | 0.64 | memoryless baseline |
| shift    | 5 | 2 | 7 | **1.00** | 0.65 | history lifts R=5 to perfect |
| register | 8 | – | 8 | 1.00 | 0.60 | |
| shift    | 8 | 2 | 10 | 1.00 | **0.70** | history improves generalisation |
| register | 4 | – | 4 | 0.00 | 0.46 | 6 colliding rows |
| shift    | 4 | 2 | 6 | 0.00 | 0.58 | collisions 6→1, but cold-state seed (see §6.4) |
| fold     | 4 | 3 | 7 | 0.00 | 0.41 | hand-coded compression is *worse* than shift |

---

## 6. What we learned (every cycle teaches something)

1. **It does the task at R ≥ 6, but by MEMORISATION, not learning.** train_acc 1.0 and
   both_ok 1.0, but the units sit *verbatim* on the training registers (10/10 at R=8) — it
   is a Hamming-kNN over an injective lookup table. **LOWO generalisation is only
   0.43–0.79**, i.e. it does not learn the stream grammar. Honest framing: "memorises an
   injective table with Hamming smoothing."

2. **Part of the R=8 "win" is zero-bias.** Stream B's tail is `00000`, so always-predict-0
   *also* reproduces B. The machine genuinely beats the baseline only on **stream A** (`11`).
   The harness now prints the baseline's own generation so this can't be hidden.

3. **Mobile gives a modest, real edge — under a clean A/B.** With the move-RNG separated so
   both modes see the same shuffle order, mobile **beats frozen at R=5** (both_ok 1.00 vs
   0.80) and edges robustness at R=6 (9.0 vs 8.6); tied at R=8. My first-pass "mobile never
   helps" was partly an *uncontrolled-comparison* artifact. (It still wastes units and never
   helps at R=4, where the task is unlearnable by any stateless predictor.)

4. **Recurrent history is the right direction, and it exposed a new requirement.** Carrying
   history (`shift`) cut R=4 collisions 6→1, lifted **R=5 both_ok 0.80→1.00**, and improved
   R=8 LOWO 0.60→0.70. But it could *not* fix R=4 generation, because there the seed is only
   4 bits **and the recurrent state starts cold (zeros)** — generation is information-starved
   at the boundary no matter how wide the address. **Lesson: a recurrent machine must have
   its state seeded/warmed, not zero-initialised.**

5. **Hand-coded compression breaks Hamming smoothness.** The rotate/xor `fold` consistently
   did *worse* than the plain `shift` (more address collisions, lower LOWO) — empirical proof
   that you cannot squeeze history into fixed width with a fixed mixing function. This is
   exactly why the address compressor must be **learned** (a smooth binary hash), the hard
   discrete-bottleneck sub-problem.

6. **Collisions are an information-theoretic wall (confirmed, strengthened).** At R=4 the 4
   contexts are exact 50/50 splits; exhaustively **0 of 4096** deterministic tables reproduce
   both tails. The R-bit register is not Markov-sufficient — no learning rule or probabilistic
   output fixes that; only *more sufficient state* does.

7. **The toy is now saturated.** Once R ≥ 6, the memoryless register already solves the task,
   so the two short streams can no longer distinguish architectures (except in the R=4
   collision corner). Testing recurrent/learned-address mechanisms for real needs data with
   **dependencies longer than the register** — which 13-bit streams cannot exhibit.

---

## 7. Next design iteration (to fine-tune)

Priority order, per the verification workflow's diagnosis (do NOT jump to learned-H or
probabilistic output first — the state must be made sufficient before either pays off):

1. **Warm-start the recurrent state** so generation isn't cold (the §6.4 discovery), then
   re-test whether recurrent state cracks the R=4 generation.
2. **Learned smooth binary hash** for the address compressor (replaces the hand-coded
   `fold`, which §6.5 showed breaks smoothness) — with anti-collapse spread pressure from
   day one.
3. **A harder data bench** with genuine long-range structure (since §6.7: the current toy is
   saturated) — the only way to measure whether recurrent/learned addresses beat a wide
   register at scale.
4. *Deferred:* probabilistic/frequency output and generic learned-H — premature until state
   is sufficient.

---

## 8. Verdict from adversarial verification (5-agent workflow, high confidence)

- **Implementation is correct.** No behaviour-altering bugs: generation is genuinely
  autoregressive (no target leak), the kernel `exp(beta·(sim−A))` decays monotonically with
  Hamming distance (near neighbours dominate, verified by hand), the `(2v−1)` sign is right,
  `merge()` loses no units, and all sweep numbers reproduce to the digit.
- **Claim "genuinely learns" → corrected to "memorises" (see §6.1–6.2).**
- **Claim "mobile never helps" → refuted (see §6.3); it helps modestly under a clean A/B.**
- **Claim "R=4 collisions are fatal to deterministic modes" → confirmed and strengthened
  (see §6.6).**
- **Hardening applied:** separate move-RNG (clean A/B), LOWO metric, baseline-generation
  reporting. Recommended single next change: **recurrent address** — now implemented and
  tested (§5, §6.4–6.5).

---

## 9. Cycle 2 — the long-range recall bench (`bench.py`)

Since §6.7 showed the 13-bit toy is saturated, we built a **controlled capability probe** with
dependencies longer than the register. Each example:

```
[TYPE bit] [shared BODY length L] [BOUNDARY 111] [ANSWER] [STOP 000]
```

The two classes share an *identical* body and differ only by the leading TYPE bit (and the
answer it dictates: type0→`11`, type1→`00`). For **L ≥ R−3** the answer-position register is
byte-identical across classes — a deliberate collision at distance L that *only memory* can
resolve. Sweeping L gives a **memory curve**. (This is a designed test like the copy/parity
tasks used to validate LSTMs — generated by a documented seeded rule, not arbitrary mock data.)

**Memory curve (R=6, answer-accuracy, chance=0.50):**

| L | register | shift+4 | fold+4 | note |
|---|---------:|--------:|-------:|------|
| 4 | 0.42 | **0.85** | 0.83 | inside shift horizon |
| 6 | 0.27 | **0.88** | 0.75 | horizon edge (R+h−4 = 6) |
| 8 | 0.33 | 0.44 | 0.42 | past horizon → all collapse to chance |

The **memory horizon is exactly `L = R + h − 4`** (the −4 = 1 TYPE bit + 3 boundary bits),
verified as a *hard step function* across a 16-cell R×h grid. This is genuine recurrent memory,
not a leak: `register` reaches L=R−4, `shift+h` reaches L=R+h−4, `fold` has no clean horizon
(its xor-mixing hurts Hamming smoothness, confirming §6.5).

---

## 10. Cycle-2 verification verdict (5-agent workflow, high confidence)

The workflow reproduced every number and **overturned two claims I had drafted** — the reason
we verify before documenting:

1. **"Dilution" → REFUTED.** I had guessed register fails at small L because uniform Hamming
   *dilutes* a present-but-outvoted TYPE bit. Wrong: amplifying that bit 24× or raising `beta`
   to 64 changes nothing. The real cause is **global label conflict** — the answer-position
   register recurs at *non-answer* positions with the opposite next-bit target, so the exact
   unit is reinforced toward the wrong value about as often as the right one — plus
   autoregressive feedback on the 2nd answer bit. (Training on answer-position pairs only lifts
   L=2 from 0.52→0.69; up-weighting the bit did not. So it is conflict-limited, not
   weight-limited.)

2. **"It only memorises" → REFUTED.** The decisive test is a **rule-scramble control**
   (randomise type→answer per body, so no transferable rule exists). On *unseen* bodies,
   `shift+4` scores **0.72–0.78 intact vs 0.44–0.53 scrambled** at L=3–6 (gap +0.22–0.31). A
   memoriser would tie both arms; the gap proves **genuine body-invariant rule transfer**. It
   is real but bounded: only inside the memory band, and capped at ~0.7–0.8 (seed-noisy)
   because uniform Hamming cannot isolate the lone type bit from body-derived state bits.

3. **Memory horizon `R+h−4` and bench soundness (L≥3) → CONFIRMED.**

**Bench bugs fixed (per the verdict):** docstring now scopes the collision claim to L≥R−3;
`gen_body` forbids a leading `[1,1]` (was creating spurious `111` in ~48% of sequences); the
rule-scramble control is built in; the small-L / L<R cells are flagged as not-clean-recall.

---

## 11. Sharpened spec for the learned address, and the next change

The evidence (keeping only what survived verification) says the learned address must:

- **Carry-far (bounded):** inject the TYPE bit into the recurrent state so it survives to the
  answer — but only up to `h` dropped steps; beyond the horizon the bit is gone and *no*
  address recovers it. Not "arbitrarily far".
- **Weight-the-bit (feature selectivity):** attend to the type-carrying slot and down-weight
  body bits. This is the lever for the **generalisation ceiling** (push held-out 0.78→~1.0) —
  *not* the small-L fix (that is label conflict; see §10.1). The DROPPED claim: "uniform
  Hamming dilution outvotes a present bit."
- **Body-invariance:** encode `answer=f(TYPE)` independently of the body — *demonstrated* by
  the rule-scramble gap, required, but only meaningful inside the memory band.

**Attempt 1 — static MI weighting (`--weights mi`):** weight each address bit by its mutual
information with the next bit. Decisive held-out test (`shift+4`, R=6, K=24, e500, 3 seeds):

| L | uniform intact | uniform scram | MI intact | MI scram |
|---|---------------:|--------------:|----------:|---------:|
| 3 | 0.81 | 0.44 | 0.75 | 0.47 |
| 4 | 0.78 | 0.56 | 0.75 | 0.44 |
| 6 | 0.58 | 0.39 | 0.64 | 0.31 |
| 8 | 0.39 | 0.42 | 0.25 | 0.42 |

**Result: MI weighting did NOT break the ceiling** (a wash, slightly worse). The
intact≫scrambled gap confirms generalisation is real in both, but selectivity didn't improve.

**Why (diagnosed, not guessed).** Printing the learned weights for `shift+4`/L=3 (address =
6 window bits ++ 4 state bits; the two classes differ *only* at position 9, the type slot):

```
weights:  0:0.38  1:1.00  2:0.15  3:0.02  4:1.46  5:2.66  6:0.49  7:0.85  8:1.38  9:1.62
```

The **boundary bits get the highest weight** (pos 5 = 2.66, pos 4 = 1.46) because `111`→answer
is globally predictive — but those bits are *identical across the two classes*, so up-weighting
them does nothing. The lone **discriminative** bit (pos 9 = 1.62) is above average yet swamped.
**Static MI conflates *predictive* with *discriminative*:** it scores each bit against the next
bit marginally, blind to which bits separate the *confusable* classes.

**Attempt 2 — contrastive weighting (`--weights contrastive`):** weight a bit by how often it
*separates collision partners* — near addresses (Hamming ≤ radius) with **opposite** labels.
This targets discriminative bits and ignores predictive-but-invariant ones.

*Diagnostic — it worked at the weight level.* For `shift+4`/L=3 the learned weights were:
```
0:1.50  1:1.57  2:1.43  3:0.86  4:0.13  5:1.24  6:0.00  7:0.20  8:1.08  9:1.99
```
The discriminative slot (pos 9) is now the **highest** weight and a boundary bit (pos 4) is
killed to 0.13 — exactly the intended correction over MI.

*But held-out accuracy did NOT improve.* Probe (held-out L=3, 5 seeds; numbers corrected after
the verification pass — see §11a):

| weighting | held-out intact | scrambled |
|---|---:|---:|
| uniform | 0.73 | 0.42 |
| pos9 dominant (others=1, pos9=5×) | ~0.78 | — |
| **isolate pos9 only (others≈0)** | **0.50 (chance)** | 0.50 |

So concentrating weight on the discriminative bit **doesn't help** (≈ uniform), and *fully
isolating* it **collapses to chance**.

**Conclusion — the metric-tuning arc is exhausted.** No fixed per-bit weighting of the raw
address (applied at retrieval) can solve this, because the task needs **different bits at
different generation steps**: the *first* answer bit needs the type slot, the *second* needs
the window (the just-emitted first bit). A single static weight vector cannot serve both. The
need is genuinely **query-conditional**.

### 11a. Verification verdict (cycle 2c, 5-agent workflow, high confidence)

- **Confirmed — no static *retrieval* weighting works.** An agent *optimised* over arbitrary
  static vectors: best held-out caps at **~0.80**, never near 0.85; at L=3 the search optimum
  even overfits *below* uniform (0.67 < 0.73). Contrastive does not beat uniform (reproduced).
- **Mechanism confirmed independently.** The discriminative position moves per step:
  L=3 step1 = {9}, step2 = {5,8}; L=4 step1 = {8}, step2 = {5,7} (pos9/8 = the type bit's depth
  in the h=4 state; pos5 = the just-emitted answer bit). Cleanest proof it is step-dependence,
  not error propagation: under isolate-pos9, first-bit acc = 0.90 but the **teacher-forced**
  second bit (handed the *true* first bit) = 0.50 = chance.
- **Correction to my earlier number.** I had reported "pos9-dominant-5× → 0.58"; that does **not**
  reproduce — with a non-suppressed base it is ~0.78 (within seed noise of uniform). The
  conclusion (capped well below 0.85) stands; the 0.58 figure was an artifact of an
  over-suppressed base vector.
- **Scope caveat.** Weights are consumed only by retrieval (`pressure`); `learn`'s
  allocation/ranking still use *unweighted* Hamming. A static vector applied to ranking *too*
  (a different machine) showed a suggestive +0.05–0.13 at L3/L4 (reaching ~0.85) — but its
  scramble control was not confirmed, so it is not endorsed as a counterexample. The claim is
  scoped to **retrieval-only** weighting.
- **Caveat on the next step.** Query-conditional weighting targets the *step-dependence*
  problem; at L ≥ 6 the discriminative signal itself flattens toward uniform, so it is unlikely
  to rescue the long-horizon (signal-decay) regime.

---

## 12. State of the project after cycle 2

**Confirmed, reusable results:**
- Recurrent memory works, with an exact horizon `L = R+h−4` (hard step, 16-cell grid).
- Genuine body-invariant rule transfer to unseen bodies (rule-scramble control: +0.22–0.31),
  bounded at ~0.78 inside the band.
- The ~0.78 ceiling is **not** a retrieval-weighting problem: static MI fails, static contrastive
  finds the right bit but doesn't help, optimised static vectors cap ~0.80, and the
  discriminative bit moves per step (so a fixed vector cannot serve all steps).

### 12a. Cycle 2d — query-conditional readout (`--weights conditional`) also fails

The attention-style readout (weights recomputed per query from the local confusion near `q`) was
the evidence-pointed next step. Tested head-to-head with uniform (held-out, K=24, 5 seeds):

| L | uniform intact | conditional intact | Δ |
|---|---:|---:|---:|
| 3 | 0.73 | 0.73 | +0.00 |
| 4 | 0.73 | 0.70 | −0.03 |
| 6 | 0.53 | 0.58 | +0.05 |

A `cond_k × radius` sweep ({8,16,32} × {2,3,4}) never beat uniform (best ≈0.73–0.75). **Result:
query-conditional readout does not break the ceiling either.**

**The sharpened conclusion.** Changing the *readout* — static **or** query-conditional — cannot
move the ceiling, because the verification confirmed weights only touch retrieval (`pressure`);
`learn`'s allocation/ranking still use **unweighted** Hamming, so the memory is *built* uniformly
regardless. On an unseen body the stored units simply do not encode the type cleanly, and no
readout reweighting recovers what was never represented. **The ceiling is representational, not a
metric.**

**The open frontier (next chapter):** shape the *learning*, not the readout —
1. **Weight allocation/ranking** (the verification's suggestive +0.05–0.13 path) so the
   discriminative structure determines *which* units are stored; verify with the scramble control.
2. **A learned recurrent encoder** that maps (window ++ history) into a body-invariant binary code
   during training (the genuine "learned hash" — the discrete-bottleneck problem).

Known limit for both: they target the in-band ceiling, not long-horizon signal decay (L ≥ R+h−4).

---

## 13. Methodology — measuring learning vs memorization at the bit level

A caution that shapes how every result here is read: **"memorization vs generalization" is partly
an *assumption* in a bit-native model, not a clean dichotomy.** With a vocabulary of 2, every
stored pattern is shared across a large Hamming neighbourhood, so *storage is itself an
interpolation mechanism* — a bit-kNN that "looks up" patterns is functionally generative when it
recombines stored bit-fragments into sequences it never saw. Flat exact-held-out accuracy
therefore should **not** be read as "memorization."

Two assumption-free instruments are used instead:

1. **Rule-scramble control.** Randomise the rule (type→answer) per body so no transferable rule
   exists. The **scramble gap** = held-out(intact) − held-out(scrambled) is the amount of genuine
   rule transfer; a pure memoriser ties both arms (gap ≈ 0). This also controls for capacity: the
   scrambled arm has the *same* unit count, so an above-chance intact arm is not "just more units."
2. **Dataset-size learning curve.** Plot held-out vs training size `K`. Memorization → flat;
   genuine rule-learning → the scramble gap **grows with `K`** (more data → better rule → better
   transfer). Corroborating clue: in these experiments more `K` helps but more *epochs* do not —
   the signature of generalisation (epochs only re-fit the same data).

## 14. Cycle 3 — allocation weighting (option 1): the capacity control flips it

Option 1 applied the discriminative weights to `learn`'s ranking/allocation (`--weight-learn`),
not just the readout. A first learning curve looked like a win (intact ~0.85 vs uniform ~0.71).
But adding a **capacity control** — plain uniform forced to allocate as many units
(`alloc_radius=0`) — reversed the conclusion (L=4, held-out, 6 seeds):

| variant | K=24 | K=48 | K=96 | units |
|---|---|---|---|---|
| uniform (r=1) | 0.71 / +.29 | 0.65 / +.17 | 0.74 / +.36 | ~44 |
| **uniform MAXCAP (r=0)** | 0.74 / +.35 | 0.81 / +.38 | **0.83 / +.45** | ~117 |
| contr/LEARN (option 1) | 0.76 / +.32 | 0.81 / +.38 | **0.84 / +.45** | ~68 |

*(cells = intact / scramble-gap.)* **Capacity-matched uniform matches option 1.** The
discriminative weighting was *not* the active ingredient — it helped only by *incidentally
allocating more units*. **The lever is capacity (unit count) + data, not the weighting.**

**What is real (confirmed by the scramble + capacity controls):**
- **It genuinely learns the rule.** The scramble gap grows with `K` (+0.29 → +0.45) and intact
  rises to **~0.84 at K=96** — and the scrambled arm stays at chance with the *same* unit count,
  so this is not memorization-by-capacity. (See §13.)
- The earlier "~0.78 ceiling" was largely a **capacity/coverage limit**, not purely
  representational: more units + more data → ~0.84.

### 14a. Verification verdict (cycle 3, 4-agent workflow, medium confidence)

- **Capacity is the lever — confirmed.** The clean control: contrastive *readout-only* allocates
  the *identical* unit counts as uniform (seed-for-seed), and there weighting **never wins**
  (e.g. K96 0.674 vs 0.667 at 41 units). The apparent `contr/LEARN` edge is **unit-efficiency**:
  `weight_learn` shifts allocation into a ~68-unit band that integer-Hamming uniform cannot
  discretely occupy (its reachable counts are ~17/36/117). At matched-or-higher capacity, plain
  uniform *allocate-every-step* reaches **0.94**, matching/beating weighting — so accuracy tracks
  the **capacity / exact-context** axis, not the discriminative metric.
- **Genuine learning — strongly confirmed.** The scrambled arm stays at chance (~0.40–0.48) while
  carrying *strictly more* units than the intact arm (K96: scramble 174u @ ~0.45 vs intact 117u @
  ~0.88). Extra capacity with no rule buys nothing → this is transferable rule recall, not
  memorization-by-capacity. The scramble gap grows monotonically with `K`.
- **Asymptote ~0.88–0.90, not 1.0 (and not a flat 0.85).** Held-out rises monotonically to
  **0.896 at K=192**. The residual gap is **jointly data-limited** (units saturate at 117 by K=48
  yet accuracy keeps climbing to K=192) **and representation-limited**: a body-length sweep (K96,
  max capacity) gives L2 = 0.76, L4 = 0.75, **L6 = 0.69** — accuracy *falls as the body grows
  despite more units*, because the lone discriminative TYPE bit is **diluted by body-noise** in the
  unweighted vote.

**Residual confound (why medium, not high):** integer Hamming makes unit count discrete, so plain
uniform could not be placed at *exactly* `contr/LEARN`'s 68-unit point for a perfect head-to-head.

**Next step (closes the confound and tests the floor):** add a **continuous capacity knob**
(prune/subsample the unit set, or a real-valued allocation threshold) to set uniform at any unit
count. Then (a) count-matched uniform vs weighting settles claim 1 fully, and (b) if weighting the
TYPE bit *does* lift the ~0.88 ceiling at matched capacity, the asymptote is a uniform-Hamming
representational floor that a **learned encoder** (compress the body out) would remove — which the
body-length sweep already points to.

### 14b. Cycle 3b — matched-capacity comparison (continuous capacity knob)

Implemented the capacity knob: `Machine.prune_to(n)` (and `--prune-to`) keeps the `n` strongest
units after training, so any config can be placed at *any* unit count for a clean head-to-head.
Results (K=64, L=4, held-out, 4 seeds; alloc=0 pool then prune to `N`):

**Capacity curve** (intact accuracy):

| N | 40 | 55 | 70 | 85 | 100 |
|---|---|---|---|---|---|
| uniform | 0.68 | 0.73 | 0.80 | 0.77 | 0.77 |
| weighted (LEARN) | 0.73 | 0.80 | 0.80 | 0.80 | 0.83 |

**Matched N=68 head-to-head** (intact / scramble): uniform **0.80 / 0.38** == weighted **0.80 / 0.38**.

**Conclusion — claim 1 fully closed.** At the contested 68-unit point the two are an *exact tie*;
across the curve the weighting shows at most a small, noisy +0.03–0.07 edge, none robust. So the
residual count-matching confound is closed: **capacity + data is the lever, not the discriminative
weighting.** Confidence upgraded from medium toward high.

**Floor test** (matched N=85, L4 vs L6): uniform L4 0.77 → **L6 0.48**; weighted L4 0.80 → **L6 0.53**.
Both collapse toward chance as the body lengthens (the L6 absolute is also data/capacity-limited at
K=64); weighting adds only +0.05 and does **not** rescue it. The longer-body ceiling is a **genuine
representational floor** of (static-weighted) Hamming.

**Cycle 3 is concluded.** The machine genuinely learns a transferable rule (scramble + capacity
controlled), scaling with data to ~0.88–0.90; the lever is capacity + data, not any weighting; and
the residual ceiling is representational and grows with body length. The evidence-pointed next
chapter is the **learned encoder** — compress the body out so the type stays separable — not any
further metric, weighting, or capacity tweak.

---

## 15. Cycle 4 — the learned encoder (`encoder.py`)

The recurrent state update becomes a **learned** transition table `g: (state, dropped bit) →
state` (`addr_mode='learned'`, `h=3` → 16 entries); `shift` and `fold` are fixed points of this
family. Clean protocol: memory trains on *train* bodies, `g` is selected on *dev* bodies, reported
on separate *test* bodies with the rule-scramble control.

**Attempt 1 — hill-climbing the table FAILED (the search overfit).** The hill-climbed `g` overfit
the 8-body dev set (dev 0.75 → test L6 **0.46**, gap +0.06) and *degraded* L4 to 0.48 (vs shift
0.83). A free 16-entry table scored on 8 dev bodies is too overfittable.

**Diagnostic — the *structure* is sufficient (hand-built latch).** Key insight: `g` sees only the
dropped bit's *value*, not its position — but the TYPE bit is the *first* bit to drop, so "absorb
the first drop and hold it" is expressible. A hand-built **latch** (state 0 → absorbing state
4/5 keyed on the first drop, held forever) removes the memory horizon. Confirmed (K=40, 6 seeds,
held-out, intact / scramble-gap):

| L | shift | latch |
|---|---|---|
| 4 | 0.83 (+0.48) | 0.78 (+0.44) |
| 6 | 0.31 (**−0.11**) | 0.61 (**+0.21**) |
| 8 | 0.38 (−0.02) | 0.64 (+0.20) |
| 10 | 0.36 (−0.01) | 0.66 (+0.27) |

`shift` collapses to a *zero/negative gap* past its horizon (R+h−4 = 5); the **latch holds a real,
scramble-controlled signal across L6–L10**. A learned recurrent code removes the horizon.

**Conclusions (cycle 4):**
1. **The learned-code structure is sufficient** — a latch carries the discriminative bit
   *arbitrarily far*, where the fixed shift/window cannot. This is the first mechanism to beat the
   horizon, not just the in-band ceiling.
2. **The bottleneck is the *learning*, not the representation.** Naive hill-climbing on a small dev
   set did not find the latch — this is the long-range **credit-assignment** problem (the answer
   supervision is far from where the latch must fire).
3. **Residual:** even with the latch, the long-range gap (~+0.25) is below the short-range gap
   (+0.44) — the *window* body-noise still dilutes retrieval. A fuller encoder must **compress the
   window too**, not only latch the history.

**Next:** *learn* the latch — a credit-assignment method or a structured gating prior that can
discover "latch the first informative drop and hold" from answer-level supervision — plus window
compression for the residual.

---

## 16. Cycle 5 — the learnable gated latch (`gated.py`)

To make the latch *learnable* (cycle 4 showed the free table overfits), use a **structured
encoder family** — a write-gate with a latch prior, parameterised by a short binary write-schedule
`w` of only `len(w)` bits (`blm.gated_latch_table`): on the `k`-th dropped bit, `w[k]=1` ⇒ latch
that bit and hold; else advance. `w=[1,0,0,0]` is the hand latch. `w` is **learned** by
enumerating its tiny `2^len(w)` space on *dev* bodies; compared to `shift` (horizon collapse) and a
**free** transition table hill-climbed on the same dev set (the cycle-4 overfit baseline). Clean
train/dev/test split + scramble control. Hypothesis: the low-parameter structured family is
*learnable and generalises* where the high-capacity free table overfit.

**Result — the latch is learnable.** Enumerating the schedule space on dev recovered
**`w=[1,0,0,0]`** (the latch). Test (held-out, K=40, h=4, 4 seeds; intact / scramble / gap):

| L | shift | free-table | gated-latch |
|---|---|---|---|
| 4 | 0.81 / +0.47 | 0.81 / +0.50 | 0.81 / +0.53 |
| 6 | 0.64 / +0.11 | 0.70 / +0.14 | 0.59 / +0.17 |
| 8 | 0.41 / +0.03 | 0.59 / +0.17 | 0.61 / +0.12 |
| 10 | 0.36 / **+0.02** | 0.41 / +0.19 | **0.72 / +0.33** |

**Conclusions (cycle 5):**
1. **The latch is learnable via a structured prior.** The tiny schedule space recovered the clean
   latch `w=[1,0,0,0]` from data — the inductive bias makes learnable what the free `2^h` table did
   not reliably find.
2. **The learned gated-latch removes the horizon.** At L10, `shift` is dead (gap +0.02) while the
   gated-latch holds strongly (0.72, gap +0.33).
3. **Nuance vs the free table.** At `h=4` the free table *also* partially learned a latch-like rule
   (it beats `shift` at L8/L10), so it was not as catastrophic as at `h=3` — but the gated-latch is
   cleaner and clearly best at the extreme (L10: 0.72 vs 0.41). The structured family is the more
   reliable route.

_(The L10 table cell above is 4-seed and seed-noisy; the verification below corrects it.)_

### 16a. Verification verdict (cycle 5, 4-agent workflow, high confidence)

- **Latch reliably learned — confirmed.** The 16-schedule space collapses to 5 reachable
  behaviours; all 8 schedules with `w[0]=1` are byte-identical "latch-first-drop", and the bench
  TYPE is always the first drop — latching into an absorbing state, *verified to body length 1000*.
  `learn_schedule` recovers the latch **8/8** at L=4/8/10; the L=6 5/8 case was a tie-break artifact
  (now fixed — ties prefer the latch family, all-zeros excluded), not an overfit win.
- **Horizon genuinely removed — confirmed.** `shift`'s address is identical across the two classes
  for all L ≥ 8 (horizon R+h−4=6) while the gated address stays distinct; the 8-seed
  scramble-controlled gated gap is significant at every L (z = 3.4–7.6).
- **Gated beats free — confirmed (best-supported).** Gated wins 4/4 at L10/L12 (+0.27 / +0.21 raw,
  non-overlapping per-split ranges); the free hill-climbed table overfits dev (≈0 held-out scramble
  gap, negative on 2/4 splits at L12). The structured family **cannot** overfit (5-point reachable
  space); it is the only encoder with above-chance scramble-controlled transfer at long L on every split.
- **Correction:** the 4-seed "+0.33 at L10" did **not** replicate at 8 seeds — robust value
  ≈ **+0.156**, and the gap **attenuates with L**: +0.156 / +0.102 / +0.078 at L10/12/14. The latch
  removes the *horizon*, but the (body × latched-state) address space the SOM readout must cover
  **grows with L**, so the readout (not the memory) now limits long-L accuracy.

**Next — window/body compression.** The encoder is proven horizon-free; the remaining attenuation
is a *readout* problem: fold/hash/down-weight the **window body bits** so only the latched
TYPE-carrying state dominates the retrieval kernel, decoupling rule-readout from L.

---

## 17. Cycle 6 — window compression (`window.py`)

Cycle 5 left a *readout* limit: the latch holds the type, but accuracy attenuated with length
because the (window body × latched-state) space the readout must cover grows with `L`. Fix
(`win_keep`): the register still slides at width `R` (driving the latch), but the **address keeps
only the last `win_keep` window bits** — the latch carries long-range memory, so the address window
only needs the local structure (boundary + recent outputs); the body bits are dropped.

**Result — corrected after verification (§17a); my first low-power headline is retracted.** Two
mechanisms with very different evidential standing:

- **The latch is the real, robust win** (cycle 5): `w=[1,0,0,0]` recovered 8/8 on dev, horizon-free
  *by construction*.
- **Window compression genuinely helps and genuinely transfers.** A short address window
  (≈ boundary width) beats the full window on held-out (mean +0.17 across L), and the transfer is
  real and leakage-proof: the `wk=3` answer-position address provably collapses to **exactly two**
  body-invariant, type-only addresses (`111` boundary ++ the latched-type bit), so the scramble arm
  is *structurally forced* to chance. `win_keep=3` is the **mechanistic optimum** (wk2 0.67 / wk3
  0.92 / wk4 0.64 at L8/12; `wk>3` pulls in body bits and pushes scramble *below* chance — overfit).

What I first claimed and now **retract** (it was low statistical power — 16 held-out items/seed):
1. **`win_keep` is NOT reliably learned.** Dev cannot distinguish `win_keep ∈ {2,3,4}` (they tie at
   the ceiling on the 8-body dev set); the search lands on a noisy tie-break (`wk=3` in only 2/8
   runs). `win_keep=3` equals this bench's boundary width — a **principle** ("address window =
   local-structure width: boundary + autoregression"), not a reliably-learned constant.
2. **Accuracy is NOT flat near 1.0.** It is a **bimodal ~0.5/1.0 mixture** (each split solves or
   fails), plateauing ~0.79 for L≥8; the "flat ~0.9" was a noisy average (within-L std ~0.25,
   larger than the L4→L20 change).
3. **Full-window does NOT clearly attenuate** on re-run (it dips then recovers; net slope ~0). The
   honest contrast is "compressed *higher* than full", not "compressed flat vs full decaying".

### 17a. Verification verdict (cycle 6, 4-agent workflow, high confidence)

- **Latch** — robustly learnable, horizon-free by construction, code-verified. ✓
- **Window compression** — genuine improvement *and* genuine, leakage-proof transfer (mechanism
  traced to two type-only addresses; `wk>3` over-fits → sub-chance scramble). ✓
- **Downgrades** — `win_keep` not reliably learned (noisy tie-break over {2,3,4}); held-out is
  bimodal ~0.5/1.0 not flat-near-1.0; full-window does not visibly attenuate. **Root cause: low
  test power** (16 held-out items/seed → near-binary coin flips).
- **Next — raise test power and re-assess:** pool held-out across many seeds with binomial CIs;
  report the per-`win_keep` curve with CIs at several L; extend L to 24–32; reconcile the
  attenuation discrepancy. Lead the story with the latch; present window compression honestly as
  "narrowing the address to local-structure width beats full-window with scramble-controlled
  transfer", stating `win_keep=3` is the bench's boundary width (principle, not learned constant).
  _High-power re-assessment: see §17b._

### 17b. High-power re-assessment (pooled held-out, binomial CIs)

Re-ran with **pooled** held-out items (K=48, 12 seeds, n=216/cell, 95% CI) to settle the three
downgraded points with statistical power.

**Per-`win_keep`** (intact ± CI / scramble):

| wk | L8 intact | L8 scr | L16 intact | L16 scr |
|---|---|---|---|---|
| 2 | 0.67±0.06 | 0.50 | 0.67±0.06 | 0.44 |
| **3** | **0.92±0.04** | 0.51 | 0.83±0.05 | 0.47 |
| 4 | 0.73±0.06 | **0.37** | 0.95±0.03 | **0.35** |
| 6 | 0.65±0.06 | 0.35 | 0.70±0.06 | 0.34 |

**Length curve** (compressed `wk=3` vs full `wk=6`, intact ± CI):

| L | wk=3 | scr | wk=6 | scr | gap |
|---|---|---|---|---|---|
| 4 | 1.00±0.00 | 0.43 | 0.71±0.06 | 0.42 | +0.29 |
| 8 | 0.92±0.04 | 0.51 | 0.65±0.06 | 0.35 | +0.27 |
| 12 | 0.96±0.03 | 0.44 | 0.70±0.06 | 0.35 | +0.26 |
| 16 | 0.83±0.05 | 0.47 | 0.70±0.06 | 0.34 | +0.13 |
| 24 | 0.88±0.04 | 0.42 | 0.75±0.06 | 0.41 | +0.13 |

**Rigorous conclusions (supersede §17a's downgrades where power resolves them):**
1. **`win_keep=3` IS identifiable with power** — CI-separated optimum at L8 (0.92 vs 0.65–0.73);
   the earlier "not learnable" was a small-dev artifact. **But the correct selection objective is
   "high intact *with scramble ≈ chance*"** — raw intact (and even raw scramble-gap) would pick the
   *overfit* `wk=4` at L16 (intact 0.95 but scramble 0.35 = **sub-chance** = body-overfit), whereas
   `wk=3` keeps scramble ≈ 0.5 (clean). Sub-chance scramble is the overfit signature.
2. **Compressed (`wk=3`) holds ~0.85–0.95 across L4–24** — high, roughly flat (mild noise), and
   **CI-separably above full-window (~0.70) at every L** (+0.13 to +0.29). The high-power estimate
   (~0.90) is *higher* than the underpowered re-run's ~0.79; the core "high, roughly
   length-independent, beats full-window" claim is **rehabilitated** — just not "flat at 1.0".
3. **Full-window sits ~0.70 throughout** (no clear attenuation), with mildly sub-chance scramble.

**Net, CI-backed:** latch + window compression (address = local-structure width) achieve **high
(~0.90), roughly length-independent, scramble-validated rule transfer, clearly above full-window** —
solid, with `win_keep=3` the principled (boundary-width) optimum selected by the scramble-clean
criterion, not a "flat-at-1.0 reliably-learned-constant" overclaim.

---

## 18. Cycle 7 — multi-feature memory: does it compose? (`multi.py`)

Does the structured-latch approach scale to remembering **more than one thing**? A 2-feature
recall bench: `[TYPE1 TYPE2][shared BODY L][111][ANSWER 2b][000]`, answer = `f(t1,t2)` — **echo**
(`[t1,t2]`, independent) and **xor** (`[t1^t2, t1^t2]`, joint, needs both features combined). The
**multi-latch** (`blm.multi_latch_table(k,h)`) holds the first `k` dropped bits (the `k` type bits)
then freezes; `k=1` is the single latch and can hold only one feature. `k` is **learned** on dev by
the scramble-clean objective; body-disjoint split + per-body rule-scramble control.

**Result — the structured latch COMPOSES.** `k=2` is learned for both modes. Pooled held-out
(K=48, L=8, win_keep=3, 12 seeds, 95% CI):

| answer | k=1 (one feature) | k=2 (both) | scramble |
|---|---|---|---|
| xor (joint) | 0.50±0.05 | **0.96±0.02** | 0.50 |
| echo (independent) | 0.48±0.05 | **0.65±0.05** | 0.24 |

- **Multi-feature memory works:** `k=2` holds both type bits, is CI-separably above `k=1`, and the
  count `k` is itself learnable.
- **The joint task (XOR) is solved cleanly (0.96)** — the readout combines two latched features;
  `k=1` is exactly at chance (one feature carries no parity information).
- **The independent 4-way task (echo) is only partially solved (0.65)** — counter-intuitively
  *harder* than the joint XOR. The per-answer-bit diagnostic shows why: bit1 (`t1`, read from the
  latch) = **0.90**, but the second *independent* bit (`t2`) = **0.73** even teacher-forced. For XOR
  bit2 = parity = bit1, so it is **copied from the window** (1.00); for echo bit2 = `t2 ≠ t1`, so it
  must be extracted from the latched state while body-internal `[1,1,*]` window collisions intrude.

**Conclusion:** the **memory primitive composes** — multiple features are latched and held, and a
*joint* function of them (XOR) is read out cleanly. The residual (echo's independent second bit) is
the **familiar readout limit** — uniform-Hamming kNN extracting one specific bit from a
multi-feature address amid collisions — *not* a memory failure. The bottleneck, again, is the
readout, not the memory.

### 18a. Verification verdict (cycle 7, 4-agent workflow, medium confidence)

- **Composition holds in-band and is genuine — with fragility.** `k=2` is CI-separably above `k=1`
  for both modes (L≤32); `learn_k` recovers `k=2` 5/5 seed-blocks (pooled). Mechanism verified:
  `multi_latch_table(2,4)` makes **4 distinct injective absorbing states** decoding to `(t1,t2)`,
  while `k=1` collapses the 4 classes to 2 (holds `t1` only) — structurally unable to do the 2nd bit.
  *Caveats:* echo's per-seed advantage is marginal (~10/16 wins at L=8); single-split `learn_k` is a
  coin-flip for echo (reliable recovery needs multi-seed pooling); and **echo collapses at L=48**
  (`k2=k1=0.44`) — the latch does *not* compose for echo at large gaps.
- **XOR genuine — confirmed.** 0.938±0.040, scramble *exactly* at chance; leakage structurally
  impossible (kept window = `[1,1,1]` boundary for all 4 classes; type bits outside the R-window;
  body shared). The **rule** is genuine at every L, but intact **accuracy decays with L**
  (0.96→0.92→0.75) — a held-out generalisation cost, not a memory failure.
- **Echo residual = readout, NOT memory, and NOT fixable by the available levers.** Both bits are
  provably in the latched state; the 2nd independent bit is the bottleneck (full .65 / bit1 .90 /
  bit2 .73) because at that step the kept window groups *only* by the just-emitted `t1`
  (`[1,1,0]` vs `[1,1,1]`), so `t2` must be read from a register sub-bit against a same-window
  collision. Every readout lever failed (β=8 unchanged; more capacity *worse*; `wk=2/0` worse) —
  a deeper limit of the frozen-address Hamming-vote readout.
- **Methodology flags:** results are seed-fragile (xor L8 0.958→0.891 from 12→16 seeds) and
  **compute-bound** (a single k=2 eval at ep=200 didn't finish a 4-seed pool in 10 min — this is what
  maxed the CPU). Next runs: cut per-eval cost (lower epochs / cache pooled pairs across seeds),
  report per-seed win/tie/loss (not only pooled CIs), and lock the echo claim to L≤32.

**Next chapter — the readout.** The latch holds the bits; the gap is *extraction*. Replace the
single global Hamming-vote with a readout that can separate collinear register sub-bits under an
identical window — e.g. a per-state / per-window-group calibrated head (a small linear decoder over
the latched register), or an address transform exposing the register bits as separable dimensions.
Validate on echo `bit2 | true-bit1` (capped ~0.73). Do *not* chase larger L for echo until the
readout is fixed.

---

## 19. Cycle 8 — the learned decoder, and a correction: it was an address collision

We built the learned decoder the §18a verdict asked for — and the investigation **corrected the
"readout limit" diagnosis itself.**

**(a) A global learned decoder is *worse* than the vote.** A gradient-free perceptron over the
address (single bits + pairwise ANDs, so it *can* select one bit or combine two) underperforms the
kNN-vote (K=40, L=8, 8 seeds): echo full **0.38 vs 0.53**; xor full **0.56 vs 0.84**. Reason: a
*global* linear head is **swamped by the filler positions** (which vastly outnumber the answer
positions), so it underfits the rare answers — whereas the vote's **locality** (only nearby units
matter) is its strength. (Implementation: `decoder.py`.)

**(b) The real blocker is an ADDRESS COLLISION, not the readout.** Direct measurement: **160/160**
answer-bit-2 addresses *also* map to a different target elsewhere in training. The same address must
emit two different bits → **no readout — vote or decoder — can resolve it.** After the latch
freezes, the register is identical at every position and the 3-bit window cannot distinguish the
*answer region* from the *body*. So echo's cap is **address ambiguity**, not extraction — this
supersedes §18a's "readout limit" framing (the cycle-7 readout-lever probes failed precisely
because the address itself is ambiguous).

**The fix is address enrichment, not a fancier readout.** A sticky **"boundary-seen" bit** halves
the collisions (160 → 80); the rest are collisions *within* the post-boundary region, so full
disambiguation needs a little more — a short **post-boundary step counter**. Then every answer
position has a unique address and even the plain vote can extract the answer.

**Corrected next step:** extend the recurrent state to track **WHERE we are** (boundary-seen +
post-boundary position), not to build a smarter decoder. *The latch holds WHAT; the missing piece
is WHERE.* (And a global readout is the wrong shape regardless — locality/query-conditioning is what
makes the vote work.)

---

## 20. Cycle 9 — region/position latch: the multi-feature task is SOLVED (`region.py`)

Built the region/position latch: **address = [boundary_seen, post-boundary position (3 bits)] ++
[window-slice] ++ [feature-latch (the type bits)]**. Also fixed a latent bench bug — the body now
ends in `0` so the `111` boundary is **unambiguous** (a body ending in `1` fused with the boundary
into an early `111`, which had been aliasing the position counter).

Result (K=40, L=8, **8 seeds pooled**, scramble-controlled):

| mode | baseline (no region) | + region/position latch | scramble |
|---|---|---|---|
| echo | 0.72 | **1.00**  (bit2 1.00) | 0.21 |
| xor | 0.97 | **1.00** | 0.52 |

Answer-bit-2 address collisions: **160/160 → 0/160.**

**The multi-feature recall task is now solved** (echo *and* xor → 1.00, pooled, scramble at chance).
Once the address encodes **both *what* (feature latch) and *where* (region/position latch)** it is
collision-free and body-invariant, so the *simple vote* maps it perfectly and generalises to unseen
bodies.

**Completed lesson chain:** latch (*what*, long-range) → window compression (drop body-noise) →
multi-latch (several features) → **region/position latch (*where*)** ⇒ the address fully and
unambiguously encodes the situation, and a plain readout suffices.

**Honest correction to the running theme.** Across cycles I kept concluding "the *readout* is the
bottleneck." For this benchmark family that was **wrong**: the real gap was an **incomplete address**
(it encoded *what* but not *where*). Completing the address solved the task with the simplest
readout — the decoder was a red herring. (A learned/attention readout may still matter for harder
tasks, but it was *not* the blocker here.)

### 20a. Verification verdict (cycle 9, 4-agent workflow, high confidence — no correction needed)

- **Task genuinely solved.** echo & xor full = 1.00, bit2 = 1.00 held-out at every L tested (L=4,8
  over 4 seeds; L=16,24 single-seed).
- **Deterministic core (strongest evidence):** answer-bit-2 collisions **0/160 at *all* L ∈
  {4,8,16,24}** for both modes (was 160/160) — seed-independent.
- **No leakage:** the answer-position address is body-invariant (1 distinct address per class) and
  the target is never copied into it — under scramble the answer addresses collide 4/4, which is
  only possible if the address encodes (position, type), not the answer. The latch freezes holding
  both type bits *before the answer exists* (0 bad over 320 items).
- **Sound:** the `111` boundary is unique at position `2+L` with **0 violations across 140,800
  sequences**; train/dev/test are id-disjoint; the mechanism is structurally L-invariant.
- **One disclosed caveat (not a bug / not leakage):** xor's scramble control sits ~0.44–0.56 (not
  ~0.25) because xor's full answer has only 2 distinct values, raising the full-match chance
  baseline; echo's control is clean (~0.17–0.25) and xor's collision-core collapses correctly.
- **Consolidation follow-ups (recorded, not yet done):** use the collision-core as xor's scramble
  control; full 16-seed pooled-CI accuracy sweep for L=8/16/24; optional L=32/48 stress + a
  body-content-disjoint split.

---

## 21. Cycle 10 — recall vs computation (parity) (`parity.py`)

Evidence-led probe: does *recurrent-state + address + lookup* do **aggregation**, or only recall?
Task: answer = **parity of all F=6 feature bits**, `[features][111][p,p][000]`. Three recurrent
states (each + region/position + small window), held-out on **unseen feature patterns** (K=40, 6
seeds):

| recurrent state | intact | scramble |
|---|---|---|
| **accum** (1-bit running XOR, frozen at boundary) | **1.00** | 0.58 |
| latch — first 2 features | 0.48 | 0.54 |
| latch — **all 6** features | 0.60 | 0.38 |

**Findings:**
1. **Computing the aggregate solves it.** A 1-bit running-XOR accumulator (frozen at the boundary) =
   parity → exactly **2 distinct answer-addresses** → generalises to unseen patterns → 1.00.
2. **Holding the raw inputs does NOT generalise — even holding *all* of them.** `latch-all-6` (0.60)
   holds every feature bit, yet lookup can't generalise parity because **parity isn't
   Hamming-smooth** (flip one input bit → the answer flips, but nearest-neighbour returns the wrong
   one). `latch-first-2` (0.48) ≈ chance.
3. ⇒ **The recurrent state must encode the task-relevant *computed* feature, not the raw inputs.**
   *Recall = hold (latch); computation = accumulate.* The lookup readout is fine either way; the
   recurrent state is where the task-specific computation must live.

So the framework **does** extend from recall to computation — *provided the right aggregate is
computed into the address.* This raises (without yet answering) the open frontier: **how to *learn*
which computation/accumulator a task needs** (here it was hand-built as running-XOR).

*Caveat:* the `[p,p]` 2-value answer makes the scramble baseline ~0.5 (weak control, as with xor);
the deterministic core is that `accum` yields exactly 2 answer-addresses, so it generalises by
construction.

### 21a. Verification verdict (cycle 10, 4-agent workflow, high confidence — confirmed & strengthened)

- **accum solves + generalises** parity: **1.00 held-out at F = 6, 8, 10, 12** (no decay).
  *Deterministic*, not just statistical: at answer positions region+window bits are constant, so the
  only content-varying address bit *is* the frozen running-XOR = true parity → exactly 2
  parity-addresses → any unseen pattern collapses onto a seen address → lookup cannot fail (no
  Hamming-smoothness needed; address space is F-invariant because parity has 2 values).
- **holding fails even holding ALL inputs:** parity is maximally non-Hamming-smooth (100% of
  Hamming-distance-1 pattern pairs have opposite parity); nearest-neighbour over the raw held inputs
  scores 0.73 at F=6 and **0.33 (below chance) at F=8**; latch-6 is no better than latch-2.
- **sound / leak-free:** `parity_acc` reads only pre-boundary features (poisoning the answer bits
  leaves it unchanged); the `111` boundary is unique.
- **METHODOLOGY FIX (applies to earlier cycles too):** `gated.split` is **ID-disjoint, not
  content-disjoint** — `gen_feats` yields only 24/64 patterns at F=6, so test patterns overlapped
  train (6/8). The decisive **content-disjoint** control *strengthens* the result: accum 1.00 (gap
  **+0.65**) while latch-2 (0.31, **−0.27**) and latch-6 (0.21, **−0.40**) go **negative**. Removing
  the leak leaves accum untouched (as the deterministic argument predicts) and sinks the latches.
  → **adopt content-disjoint splits + the negative-gap control as the headline** (retire the weak
  duplicated-bit scramble baseline). Confidence: high.

---

## 22. Cycle 11 — compute-vs-hold holds for a non-degenerate aggregate (`aggregate.py`)

Extends cycle 10 from parity (2 values) to **popcount mod 4** (4 values), using the
**content-disjoint** split (the cycle-10 methodology fix). Held-out, 5 seeds (chance = 0.25):

| F | accum (running mod-4 counter) | latch-all (holds every input) | #patterns |
|---|---|---|---|
| 6 | **1.00** (scr 0.46) | 0.14 (scr 0.37) | 24 |
| 8 | **1.00** (scr 0.26) | 0.15 | 81 |
| 10 | **1.00** (scr 0.22) | 0.18 | 274 |

- **accum (running mod-4 counter, frozen at the boundary) solves + generalises, F-invariantly.**
  The answer-address space is the **4 aggregate values, not 2^F**, so unseen patterns collapse onto
  the 4 seen addresses → generalisation by construction; scramble → ~0.25 as the content-leak
  shrinks with more patterns (the content-disjoint control is clean here).
- **latch-all FAILS (≤ chance, 0.14–0.18)** even holding every input — a non-Hamming-smooth
  aggregate can't be generalised by lookup over raw inputs.
- ⇒ The **compute-vs-hold** principle holds beyond the degenerate 2-value case: the recurrent state
  must *compute* the task aggregate (here a running mod-m counter); the address space then scales
  with the **number of aggregate values (m)**, not the input count — hence F-invariant.

_Recorded; not yet adversarially verified (paused new runs at the user's request)._

---

## 23. Cycle 12 — LEARN the recurrent computation (the central question) (`learn_state.py`)

Every prior cycle had *me* hand-picking the recurrent state per task. This one tests whether the
machine can **learn which computation a task needs.** A structured family —
`hold-k` (latch the first k bits) and `count-m` (running popcount mod m, frozen at the boundary) —
plus a meta-learner that picks the member generalising best on a **content-disjoint dev set**,
tie-broken toward **fewest state bits** (the simplest computation).

Result (held-out, 1 seed):

| task | LEARNED | expected | test acc | dev signal |
|---|---|---|---|---|
| recall (echo) | **hold-2** | hold-2 | 1.00 | hold2=1.00, hold3=1.00, count*≤0.44 |
| parity (mod 2) | **count-2** | count-2 | 1.00 | count2=1.00, count4=1.00, hold*≤0.44 |
| popcount mod 4 | **count-4** | count-4 | 1.00 | count4=1.00, count3=0.81, count2=0.62, hold*≤0.38 |

**The machine recovers the right recurrent computation from data — without being told:** the latch
for recall, running-XOR (`count-2`) for parity, the mod-4 counter for mod-4, each at 1.00 held-out.
The parsimony tie-break recovers the **minimal** correct computation (`hold-2` not `hold-3`;
`count-2` not `count-4` for parity — `count-4` also solves parity but is over-complex). This converts
*"I hand-engineer a solver per task"* into *"the machine selects the right computation from data."*

**Honest scope:** this is **selection over a small, hand-defined structured family** (`{hold, count}`),
*not* learning an arbitrary computation from scratch. That is the practical/principled form (cf.
cycles 4–5: structured families are learnable; free tables overfit) — a library of structured
primitives + data-driven selection. Open extension: grow the family / learn the members themselves.

_1 seed; not yet adversarially verified._

---

## 24. Cycle 13 — selection COMPOSES (a task needing two computations) (`compose.py`)

Task: `[type bit][F features][111][answer = (type, parity(type+features))][000]` — **bit1 is a held
feature, bit2 is a computed aggregate.** The meta-learner chooses over **singles AND hold×count
pairs** (content-disjoint dev, parsimony tie-break).

Result (F=8, held-out): **LEARNED `{hold-1, count-2}`, test acc 1.00.**

| candidate | dev | bits |
|---|---|---|
| **{hold-1, count-2}** | 1.00 | **2** (picked) |
| {hold-1, count-4} | 1.00 | 3 |
| {hold-2, count-2} | 1.00 | 3 |
| {hold-2, count-4} | 1.00 | 4 |
| every single (hold-*, count-*) | ≤ 0.50 | — |

**Selection composes:** no single computation solves the task (all singles ≈ 0.50); four
`{hold, count}` pairs reach 1.00; parsimony recovers the **minimal** combination. So cycle 12's
"learn the computation" extends to **"select the minimal *combination* of computations"** — the
machine composes primitives into a tiny learned program for a task needing both memory and
computation.

**Honest scope:** selection over a hand-defined family *and* a hand-defined combination space
(singles + hold×count pairs); the "program" is a concatenation of state-features (no control flow),
and the answer's two bits are separable (one held, one computed). Deeper composition (joint /
non-separable answer functions; a *learned* combination space) is the open extension. 1 seed;
mechanistically transparent (singles can't, the right pairs can, parsimony picks the minimal).

---

## 25. Methodology — does more data help? data scaling vs representation (`datascale.py`)

Question raised mid-research: would increasing the dev set / training on larger data help? Two
learning curves on popcount mod 4 (F=10, content-disjoint split) separate the two failure modes.

**Curve 1 — SELECTION vs DEV size (statistical axis):**

| dev items | P(pick count-4) | test acc of pick |
|---|---|---|
| 2 | 0.43 | 0.80 |
| 4 | 0.63 | 0.88 |
| 8 | 0.90 | 0.98 |
| 16 | 1.00 | 1.00 |
| 54 | 1.00 | 1.00 |

More dev data **fixes selection**: a tiny dev mis-picks (overfits) 57% of the time; at ≥16 dev items
it reliably recovers `count-4` (1.00). → increasing the dev set is the principled lever to make a
*freer* search safe.

**Curve 2 — TEST acc vs TRAIN size (representational axis):**

| train items | raw-input (hold-all) | structured count-4 |
|---|---|---|
| 41 | 0.33 | 0.33 |
| 83 | 0.17 | 0.56 |
| 166 | 0.02 | 1.00 |

More training data only helps the **right** representation (`count-4` → 1.00). It makes the **wrong**
one *worse* (raw-input 0.33 → 0.02): with a non-Hamming-smooth target and a Hamming-retrieval
readout, more data sharpens confidently-wrong interpolation from mis-aggregated neighbours.

**Bottom line:** data scaling helps the **search** (reliable selection of the computation) but cannot
substitute for the right **representation** — it amplifies whatever representation you chose, for
better or worse. So "just train on more data" is **not** a substitute for the primitive library; they
are complementary — more dev → reliable selection; the library → the representations to select among;
more train → pays off only once the representation is right.

---

## 26. Cycle 14 — prediction → POLICY: bit-native action that learns from reward (`action.py`)

Extends "learn the computation" from supervised next-bit prediction to **reward-driven action**
(framing-doc criteria #5 action, #9 over-trials adaptation). A contextual-bandit agent: sees a bit
context, forms an address via a family member, **chooses an action**, gets a **reward bit**, and
learns the policy **online from reward** (no labels).

**Part A — improve over trials + generalise** (parity-act, 2 actions, chance 0.50):

| representation | reward over trials | held-out |
|---|---|---|
| count-2 (right) | 0.997 → 1.00 | **1.00** |
| hold-all (raw) | 0.936 → 1.00 | **0.50** |

Both improve over trials (online adaptation from reward); the **right** representation generalises to
held-out contexts (1.00), the **raw** one memorises training and collapses to chance on held-out
(0.50) — the cycle-11 / data-scaling representation lesson, now for policy.

**Part B — select the computation FROM REWARD** (no labels):

| task | LEARNED | expected | held-out reward | dev signal |
|---|---|---|---|---|
| parity (2 act) | **count-2** | count-2 | 1.00 | count2=1.00, count4=1.00, count3=0.88, holds≤0.44 |
| mod4 (4 act) | **count-4** | count-4 | 1.00 | count4=1.00, count3=0.81, count2=0.62, holds≤0.38 |

The meta-learner recovers the right (minimal) computation **from reward alone** — `count-2` for
parity-reward, `count-4` for mod4-reward. So "learn the computation" is not tied to labelled
prediction; it works under reward supervision and outputs an **action**.

**Significance:** the predictor becomes an **agent**. The core now selects the right computation to
*act* (not just predict), learns the policy online, improves over trials, and generalises to unseen
contexts — all bit-native, no language.

**Honest scope:** single-step contextual bandit, small synthetic action sets, a tabular policy over
the learned address; not yet sequential/multi-step planning or non-stationary environments. 1 seed.

---

## Appendix — prior-art map (search terms, all bit/discrete, not LLM-specific)

- **Semantic hashing** — learn compact binary codes preserving similarity (the learned "hash").
- **Locality-Sensitive Hashing (LSH)** — binary addresses for sub-linear similarity search.
- **Self-Organizing Maps (Kohonen)** — gradient-free competitive placement by similarity.
- **VQ-VAE** — discrete codebook learning (straight-through + commitment loss).
- **Hopfield / modern Hopfield networks** — binary content-addressable memory; retrieval = relaxing into a stored attractor.
- **Neural Turing Machine / DNC** — content-addressable memory with learned keys.
- **Neural Cellular Automata, Rule 110, Conway's Life** — multi-directional, parallel, Turing-complete bit machines.
- **Analog Bits, MaskGIT, discrete diffusion** — any-order / masked bit-field generation.

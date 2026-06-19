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

**Residual & caveats:** absolute accuracy (~0.81 in-band, ~0.72 at L10) still reflects the
**window-noise** dilution (the latch carries the type but the window body bits vary) — a fuller
encoder must compress the window too. Numbers are 4-seed / 8-body (noisy; the L6 dip is within
noise). _Adversarial verification in progress._

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

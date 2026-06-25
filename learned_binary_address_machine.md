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

## 27. Cycle 15 — SEQUENTIAL / multi-step action: a bit-native MDP with delayed reward (`mdp.py`)

Extends cycle 14 from a single-step bandit to a real **MDP**: navigate a 1-D corridor (P=8) to a goal
shown once at the start; the reward arrives **only at the final step**. Solved by tabular **TD
Q-learning** over a learned bit-address. Tests temporal credit assignment + memory across the horizon
+ the representation lesson, in sequence.

| representation | train reward | held-out reward | #states |
|---|---|---|---|
| `rel` = (sign(goal−pos), step) — **computed** relative direction | 1.00 | **1.00** | 21 |
| `abs` = (pos, goal, step) — raw absolute | 0.50 | **0.00** | 134 |
| `nomem` = (pos, step) — no goal memory | 0.25 | **0.00** | 35 |

`rel` learning curve (reaches-goal rate, delayed reward): **0.13 → 0.82** over training (greedy eval
= 1.00; chance ≈ 0.12).

**Findings:**
- **Credit assignment works:** with only a terminal reward, TD backup over the learned bit-address
  lifts the policy from chance (0.13) to optimal (1.00 greedy) — sequential, multi-step.
- **Memory across the horizon is required:** `nomem` (no goal memory) fails (0.25 / 0.00).
- **The representation lesson carries to sequential control:** the *computed* relative-direction
  state (21 states) generalises to held-out goals (1.00); the *raw* absolute state (134 states)
  memorises and collapses on held-out goals (0.00). Same lesson as cycles 11 / 14 / data-scaling
  (§25), now for RL.
- **Learn-the-computation extends to RL:** selecting the representation by held-out reward recovers
  `rel`.

**Significance:** the bit-native core is now a *sequential decision-maker* — delayed-reward credit
assignment over a horizon, with the right *computed* representation giving generalisation.

**Honest scope:** small tabular MDP (P=8, H=8, 3 actions), single deterministic corridor, hand-given
representation candidates; not yet function approximation at scale, stochastic dynamics, or long
horizons. 1 seed.

---

## 28. Scale on REAL data — the bit-native core as a real text compressor (`scale.py`)

The consolidation step (after a run of small single-mechanism cycles): assemble the core as a
**next-bit predictor** and run it on a **real corpus** (300–772 KB of public-domain English text,
bytes → bits), measured by held-out **bits-per-bit** (= cross-entropy = compression; raw = 1.0000;
the bit-native analogue of an LLM's perplexity), externally referenced against **gzip**. The model is
the exact-count form of the content-addressable machine: a context address → empirical next-bit
distribution (KT/Laplace smoothed); at scale exact counts replace the Hamming-kernel/SOM used on the
tiny benches.

**Representation comparison** (300 KB, held-out 20%):

| representation | held-out bits/bit |
|---|---|
| order-0 within-byte (phase, cur) | 0.5733 |
| bit-window k=8 (raw, unaligned) | 0.6617 |
| bit-window k=16 (raw, unaligned) | 0.3792 |
| byte-aware B=1 (computed phase) | 0.4296 |
| byte-aware B=2 (computed phase) | 0.3367 |
| byte-aware B=3 (computed phase) | 0.2806 |
| **CORE backoff (orders 3→2→1→0)** | **0.2727** |
| _external ref: gzip_ | _0.3585_ |

**Data scaling** (CORE backoff model):

| bytes | core bits/bit | gzip bits/bit |
|---|---|---|
| 100 K | 0.3070 | 0.3673 |
| 300 K | 0.2727 | 0.3585 |
| 600 K | 0.2689 | 0.3567 |
| 772 K | 0.2791 | 0.3559 |

**Findings:**
- **The bit-native core is a real predictor:** it compresses real English to **27 %** (0.2727
  bits/bit) and **beats gzip** (0.3585) at every data size — the "see it in real, like LLM training"
  milestone (bits/bit = the bit-native perplexity/compression metric).
- **The representation lesson holds at scale on real data:** the *computed* byte-aware representations
  dominate raw bit-windows of comparable size (byte-aware B=2 0.3367 < bit-window k=16 0.3792) and
  scale better with order; even order-0 within-byte (just phase + current partial byte) carries real
  signal (0.5733).
- **More real data → better,** to ~600 KB (0.307 → 0.269); the slight uptick at 772 KB is a held-out
  distribution shift (the file tail is the Gutenberg license/footer), not a reversal — consistent with
  the data-scaling finding (§25) on real data.

**Honest scope / relation to the design:** at scale the "learned binary address + Hamming kernel +
SOM" reduces to an exact-count context model (a PPM-like byte predictor) — Hamming generalisation is
unnecessary when data is plentiful; what is tested, and what matters, is the **representation** (the
recurrent / computed features written into the address). **gzip is a standard but weak baseline**;
strong compressors (PPM/CTW/PAQ, LLMs) reach far lower bits/bit. The claim is "**the core works and
beats gzip on real text**," not state of the art. One book, one language, ≤ 772 KB.

---

## 29. Learn the REPRESENTATION on real data — discover byte structure from scratch (`represent.py`)

§28 hand-gave the byte/phase structure. This removes that crutch: the core **discovers** which context
features predict the next bit, on the real corpus, via learn-the-computation at scale.

**Part 1 — period scan** (address = `i mod p` alone): bits/bit is minimised at **p=8** (0.77), tied
with its multiple p=16, with partial dips at the divisors/multiples 4 and 12 and ≈ chance (0.99)
elsewhere. The core **discovers the byte period of text** purely from predictive value, without being
told.

**Part 2 — greedy forward selection** over a pool {`i mod p` (p=2..16), lag bits (1..16)}; it builds
a representation from scratch:

| step | feature added | bits/bit |
|---|---|---|
| 1 | **mod-8** (byte phase, chosen first) | 0.7725 |
| 2 | **lag-8** (aligned bit in previous byte) | 0.7430 |
| 3–7 | lag-1,2,3,4,5,6 | 0.5323 |
| 8–16 | lag-5,7,9,10,11,12,14,15,16 | **0.3759** |
| — | _hand-given byte-aware B=2_ | _0.3445_ |

**Findings:**
- **The core rediscovers byte structure:** it selects the byte phase (`mod-8`) first and the
  byte-aligned previous bit (`lag-8`) second — the same structure §28 hand-coded — from data alone.
- **The discovered representation converges toward the hand-engineered one:** the gap shrinks from
  0.53 vs 0.35 (7 features) to **0.376 vs 0.345** (16 features, ~0.03 apart) and was still narrowing.
  Given enough capacity the *learned* representation is essentially as good as the *supplied* one —
  without being told about bytes.
- **Residual gap:** greedy-over-individual-bits + crude global backoff is slightly weaker than the
  structured byte-aware + order-backoff CORE (§28), and the curve had not fully converged.

**Significance:** the central "learn the computation" result now holds at scale on real data for
**representation discovery** — the core finds the predictive structure (byte period + alignment)
itself, rather than being handed it. **Honest scope:** greedy feature selection over a hand-defined
candidate pool (periods + lags); a real but still bounded search; ≤ 120 KB, one corpus.

---

## 30. Push the compressor — online logistic context mixing (`mix.py`, `mix_sse.py`, `two_layer_mix.py`)

The scaled core's predictor, improved from count-backoff (§28) to **online logistic context mixing** —
the lpaq/PAQ idea, and exactly the framing's "neural-network mapping" in bit-native form: several
byte-aware context models each vote a probability for the next bit; a single **online-trained
logistic unit** mixes them in the logit (stretch/squash) domain; the mixed P codes the bit; weights +
counts update online. Adaptive over the whole stream (like gzip on the whole file). Metric = bits/bit
(= compression; raw = 1.0000).

Results (300 KB cap, **whole-stream / last-20%**; self-run and adversarially verified):

| model | prose | code |
|---|---|---|
| _gzip (whole file)_ | _0.3585_ | _0.2386_ |
| `mix.py` logistic mixing (orders 0–4) | 0.2636 / 0.2398 | 0.2269 / 0.1890 |
| `two_layer_mix.py` (mixer-set + SSE) | **0.2400** / 0.2158 | 0.1962 / 0.1582 |
| `mix_sse.py` (SSE/APM + match + orders 0–6) | 0.2411 / 0.2167 | 0.1878 / 0.1473 |
| **`mixmax.py` (merge — best of both)** | **0.2392 / 0.2163** | **0.1835 / 0.1460** |

- All four **beat gzip** on both prose and code; the merged model compresses real prose to ~24 % and
  code to ~18 %.
- **The merge `mixmax.py` wins both corpora at once** — prose 0.2392 (< two_layer 0.2400) and code
  0.1835 (< mix_sse 0.1878) — combining `two_layer_mix`'s global + final-layer mixers with `mix_sse`'s
  byte-level match model and order-5/6 contexts. **Verified leakage-free** by a built-in future-bit-flip
  causality self-test (`python mixmax.py flip`: flipping a future bit leaves every past prediction
  bit-identical, while later predictions do change).

**Verification (multi-agent workflow — 2 independent skeptics + judge):** `mix.py` is **sound and
leakage-free** — causality proven three ways (count-invariant assertion c0+c1 ≤ i; future-bit-flip
invariance; from-scratch re-implementation matching to 4 decimals); the metric is a correct mean
−log₂ P(true bit) (a constant-0.5 predictor = exactly 1.0000); single-thread, stdlib only. Both
challengers were verified causal (future-flip test) and reproduce their numbers.

**Honest scope:** bits/bit depends on the byte budget (longer stream → lower, an online-adaptation
effect, not leakage): `mix.py` is 0.2636/0.2269 at 300 KB but 0.2743/0.2351 at 200 KB. gzip is a
standard but **weak** baseline; strong compressors (PAQ/cmix) reach far lower. This is "the bit-native
core, with a neural mixer, is a real compressor that beats gzip," not state of the art. ≤ 300 KB, one
prose + one code corpus.

**Next:** the road to *strong* compressors — CTW-style context mixing, larger / more diverse corpora,
longer context + match models, adaptive count decay for non-stationarity — or apply the same
predictor to a different modality through a dumb adapter.

---

## 31. Agency on a REAL stream — bit-native change / anomaly detection (`stream.py`)

Turns the predictor into an **agent that acts on a real stream**: the online logistic mixer predicts
each bit; the per-byte **surprise** (−log₂ P over its 8 bits) is the signal; a sustained mean-shift in
surprise makes the core **emit a flag — its action**. This realises the framing doc's own examples of
intelligence-without-language ("detecting anomaly in a stream" / "detect when a pattern changes", §6
and §11) and exercises non-stationarity. Stream = **real** data with known regime shifts: English →
Python source → English (270 KB, built from the two real corpora so the change points are ground
truth).

**Result:**
- Surprise jumps sharply at **both** true regime changes: eng→code **+1.11** bits/byte (2.12 → 3.23),
  code→eng **+0.41** (1.83 → 2.24).
- The detector flags **both** boundaries with low latency (**554** and **192** bytes).
- The per-5 KB surprise trace tracks the regimes (English ~2.1–2.6, Python ~1.7–2.0 bits/byte).

**Findings:**
- The bit-native core **detects distribution shift on real data** — its prediction surprise is a
  usable anomaly signal, and the flag is a genuine action from the *same* predictor (no separate model).
- **Asymmetry (honest):** the return boundary (code→eng) is weaker (+0.41 vs +1.11) because the model
  **retained English statistics** from the first segment — its memory makes the second English regime
  less surprising. A real property of the predictor, not a detector artefact.
- The 8 extra flags are mostly genuine **local** content anomalies (e.g. a repetitive passage dipping
  to ~1.3 bits/byte) plus 2 refractory echoes of the real detections — i.e. the detector is doing
  anomaly detection, broader than the 3 planted change-points.

**Honest scope:** a thin surprise-threshold detector on top of the `mix.py` predictor (windows/margin
tuned, not learned); one constructed real stream with 2 planted boundaries; recall 2/2, with extra
flags being real local anomalies rather than random noise. **Significance:** closes the framing
scorecard's *action* axis on **real** data — the same core that compresses also **acts** (flags
changes/anomalies) on a real stream, supporting "intelligence below language" beyond compression.

---

## 32. Toward STRONG compressors — four techniques, and the real lever (`mixns.py`)

A 4-way parallel push (adversarial multi-agent workflow) added four advancements on top of `mixmax`,
each benched on prose + code (200 KB) and causality-self-tested:

| technique | prose | code | Δ vs mixmax |
|---|---|---|---|
| `mixmax` (base) | 0.2495 | 0.1922 | — |
| **`mixns`** (recency count-halving + RMSProp mixer LR + hashed orders 8/12/16) | **0.2465** | **0.1891** | −0.003 |
| `deep_sse` (5-APM calibration chain) | 0.2481 | 0.1913 | −0.001 |
| `mixctw` (binary CTW input) | 0.2482 | 0.1911 | −0.001 |
| `multimatch` (run-length + multi-hash) | 0.2495 | 0.1918 | ~0 |

All four are **verified causal** (future-bit-flip self-test) and independently re-measured; `mixns`
wins both. The techniques are **sub-additive** (they attack overlapping high-order signal), so stacking
all four is only ≈0.244/0.187 — diminishing returns.

`mixns` verified at 300 KB: **prose 0.2364 / code 0.1806** (causal), vs gzip 0.3585 / 0.2386.

**The real lever is data, not more tricks.** `mixns` prose whole-stream by corpus size:

| bytes | bits/bit |
|---|---|
| 100 K | 0.2638 |
| 300 K | 0.2364 |
| 500 K | 0.2269 |

More data bought **≈0.037** (100 K→500 K) — an order of magnitude more than the ≈0.003 from the best
shallow trick.

**Conclusion:** shallow modelling tricks have hit diminishing returns; reaching PPM/PAQ territory
needs **scale** (more / diverse data) and deeper modelling, not more shallow mixer inputs. `mixns` is
the current strongest single model and is verified leakage-free. **Honest scope:** gzip is a weak
baseline; absolute bits/bit (~0.18–0.24) is still well above strong compressors; single-corpus,
≤ 500 KB.

---

## 33. Confidence-driven ACTION with consequences (`decide.py`)

Exercises the framing's **confidence / uncertainty** criterion together with action: the core predicts
the next **byte** by an 8-bit greedy rollout (no peeking) and **decides commit-or-abstain** by its own
confidence (product of the per-bit max-probs). Reward: +1 correct commit, −4 wrong commit, 0 abstain.
Real prose, 200 KB.

| threshold | coverage | acc@commit | net reward/byte |
|---|---|---|---|
| 0.00 (always commit) | 1.00 | 0.565 | −1.177 |
| 0.50 | 0.57 | 0.784 | −0.045 |
| 0.80 | 0.29 | 0.911 | **+0.158** |
| 0.90 | 0.18 | 0.947 | +0.129 |

- Next-byte top-1 accuracy **0.565** (chance 1/256).
- **Confidence is calibrated:** accuracy@commit rises monotonically (0.57 → 0.95) as the threshold
  tightens — higher self-confidence really does mean higher accuracy.
- **Always-commit loses** (−1.177 under the asymmetric reward); **confidence-gating wins** (+0.158 at
  τ=0.80) by committing only when confident and abstaining otherwise.

**Significance:** the core uses its **own uncertainty** to make profitable decisions — confidence-driven
action with real consequences, and the monotone accuracy-vs-confidence shows the confidence signal is
meaningful. Exercises framing criteria #5 (action) + confidence. **Honest scope:** greedy-rollout
next-byte prediction + a *swept* confidence threshold (the policy is thresholded, not learned); one
real corpus; the reward structure (λ=4) is illustrative.

---

## 34. Scaling up — multi-MB real text, and the capacity ceiling (`mix.py`, `mixns.py`)

Scaled to a **5.4 MB** real-English corpus (3 public-domain books). Whole-stream bits/bit:

| size | `mix.py` (orders 0–4) | `mixns` (strong) | gzip |
|---|---|---|---|
| 1 MB | 0.2512 | **0.2222** | 0.3560 |
| 2 MB | 0.2514 | **0.2215** | 0.3607 |
| 4 MB | 0.2462 | — | 0.3631 |

**Findings:**
- **Both beat gzip at every scale** (~0.22–0.25 vs ~0.36); gzip does **not** improve with more data
  (fixed 32 KB window), the bit-native models do.
- **The strong model wins at scale:** `mixns` (~0.22) clearly beats the simple mixer (~0.25) — its
  high-order + match components pay off with enough data.
- **Honest catch — bits/bit *plateaus* beyond ~1 MB** (`mixns` 0.2222 → 0.2215; `mix` flat ~0.25).
  Small-scale data gains were large (§25/§32: 0.26 → 0.23 from 100 → 500 KB); at MB scale they shrink
  to ~0. The bottleneck has shifted from **data** to **model capacity**: a fixed-capacity context model
  saturates once its contexts are populated, and more data can't buy structure it cannot represent.

**Conclusion — the scaling triad:** data, model **capacity**, and representation must scale *together*.
§25 showed data can't fix the wrong representation; §32 showed shallow tricks plateau at small data;
this shows **data plateaus at fixed capacity**. Pushing from our ~0.22 toward strong compressors
(~0.15 on enwik8) needs more **capacity** (longer contexts/match, deeper/larger mixing), not just more
data. **Honest scope:** pure-Python ceiling (~5 MB tractable, minutes-to-hours, one core); 0.22
bits/bit sits between gzip (0.36) and strong compressors (~0.15); LLM-scale data + capacity would need
a compiled/vectorised implementation. _(Refined in §35: with PyPy reaching 4 MB, `mixns` continued
down to 0.2161 — diminishing returns, not a hard plateau.)_

---

## 35. Overcoming the Python ceiling — PyPy (CPU, not GPU)

The hot loop is **sequential + sparse** (dict context tables), so the lever is faster per-step
**native CPU** execution, not parallelism/GPU. First, cheapest step: run the unchanged pure-Python
code under **PyPy** (a JIT Python).

Measured (bit-for-bit **identical** bits/bit — correctness preserved):

| run | CPython 3.13 | PyPy 7.3.19 | speedup |
|---|---|---|---|
| `mix.py` @500 KB | 27.5 s | 8.4 s | 3.3× |
| `mixns.py` @300 KB | 57.0 s | 14.9 s | 3.8× |

**~3.3–3.8× out of the box** — capped by per-bit `bytes(slice)` allocation and dict-with-tuple keys
that the JIT can't fully remove. This is **CPU expansion of the same architecture, not GPU**: the model
is sparse + sequential (each bit depends on the prior step), which a GPU can't exploit; GPUs would only
matter under a dense neural pivot.

Leveraged immediately to reach **4 MB** on the strong model (≈9 min on CPython → **2.5 min** on PyPy):

| size | `mixns` whole-stream | gzip |
|---|---|---|
| 1 MB | 0.2222 | 0.3560 |
| 2 MB | 0.2215 | 0.3607 |
| 4 MB | **0.2161** | 0.3631 |

**This refines §34:** bits/bit is **not** at a hard plateau — 4 MB continued down (0.2215 → 0.2161);
the earlier "plateau" was partly an artefact of only reaching 2 MB on CPython. More data still helps
(diminishing but real), and capacity would help more.

**Path forward:** bigger speedups need PyPy-friendly **integer (rolling-hash) context keys** or the
planned **Rust/C++ native core** (the route to enwik8-scale, ~0.15). PyPy now buys ~3.5× and a clearer
view of the scaling trend, with no code change and no GPU.

---

## 36. PyPy- and Rust-friendly integer keys — same quality, ~12× faster (`mixfast.py`)

Replaced the per-bit context key — `(phase, bytes(partial), bytes(prev_bytes))` — with a single
**lossless integer**: a sentinel-prefixed pack of phase + the partial current byte + the previous
bytes, carried in a rolling `htail` integer (no per-bit `bytes` allocation, no tuple hashing). The
encoding is a **bijection** with the old key (same equivalence classes).

**Proven bit-identical** to `mix.py` (whole/tail match to 1e-12). Speed @500 KB:

| | CPython 3.13 | PyPy 7.3.19 |
|---|---|---|
| `mix.py` (bytes/tuple keys) | 27.5 s | 8.4 s |
| `mixfast.py` (integer keys) | 13.6 s | **2.26 s** |

- Integer keys: **2.0×** on CPython alone (no bytes allocation).
- They let the JIT do far better: **6.0×** PyPy-vs-CPython (was 3.3× with byte keys).
- **Combined (integer keys + PyPy): ~12×** over the original (27.5 → 2.26 s), output unchanged.

This confirms the analysis: lossless keys cost **no quality**, **multiply** the PyPy win, and converge
the data model toward the eventual **Rust/C++** design (integer keys + open-addressing hash tables of
fixed-width keys) — so it **de-risks** the native port rather than complicating it.

The same change applied to the strong model gives `mixnsfast.py` — **bit-identical** to `mixns`
(0.2581495288 match) and causality-clean (future-flip test passes). Its win is smaller (CPython 57.0 →
40.5 s, ~1.4×; PyPy 10.1 s, ~5.6× combined) because `mixns`'s cost is dominated by the mixer / RMSProp
/ APM float math rather than the context dicts — but it's a verified, lossless speedup of the model we
actually scale.

---

## 37. Scaling with the fast stack — clean homogeneous curve (`mixfast.py` / `mixnsfast.py` under PyPy)

With the integer-key models under PyPy, runs that were *hours* on CPython are *minutes* (e.g. 88 M
bits / 11 MB in ~50 s for `mixfast`). Used this to test scaling properly.

**Heterogeneous corpus is not a clean scaling axis.** On 4 concatenated books (11 MB), `mixfast` went
5 M 0.2528 → 11 M 0.2609 — *up* — **but so did gzip** (0.369 → 0.374), because the appended Shakespeare
(archaic English / verse) is genuinely harder. Composition dominates; concatenating different books ≠
"more of the same data."

**Clean homogeneous scaling** (single source — War & Peace prefixes), `mixnsfast`:

| bytes | whole-stream | last-20% | gzip |
|---|---|---|---|
| 0.5 M | 0.2382 | 0.2258 | 0.3616 |
| 1 M | 0.2288 | 0.2213 | 0.3630 |
| 2 M | 0.2222 | 0.2139 | 0.3652 |
| 3 M | 0.2181 | 0.2104 | 0.3651 |

On homogeneous data, bits/bit improves **monotonically** with scale (0.238 → 0.218, still dropping at
3 M; last-20 % to 0.210), while gzip stays flat (~0.365). **More data does help cleanly** — the earlier
"plateau"/non-monotonic readings (§34) were corpus-composition artefacts, not a model ceiling.

**Takeaways:** (1) the fast stack (PyPy + lossless integer keys) makes real-scale experiments
practical — tens of millions of bits in seconds-to-minutes, one core, no GPU. (2) Clean scaling needs
**homogeneous** data; heterogeneous concatenation conflates scale with difficulty. (3) The strong model
holds ~0.21–0.22 vs gzip ~0.365 at every scale and is still descending at 3 M of one source.
**Honest scope:** ~3 M single-source is still small vs enwik8 (100 M); the dict-based orders grow with
data (RAM) — fixed-size hashing / the Rust core remain the path to much larger scale.

---

## 38. Bounded-RAM via fixed-size hashing — reaches scale, collision-limited (`mixnshash.py`)

To break the dict-RAM ceiling, the byte-orders + sparse + word tables were converted from growing
dicts to **fixed-size hashed count arrays** (the high orders already were), with **8-bit checksum tags**
(evict-on-collision — the PAQ/zpaq trick). RAM is now **bounded** (~370 MB, independent of data size)
and the model stays **causal** (flip-test passes).

**Collision cost** (1.5 MB War & Peace, vs the exact `mixnsfast`):

| model | whole-stream |
|---|---|
| `mixnsfast` (exact dict) | 0.2254 |
| `mixnshash` (fixed hash, 22-bit, tagged) | 0.2739 |

→ **+0.048 bits/bit.** Tags didn't recover it: the high orders (5–6) run at high load, so collisions
are *frequent* and eviction loses count history about as much as merging — the fundamental
memory↔quality tradeoff (no free lunch at small fixed tables).

**Reaching scale** (enwik8, the standard 100 MB Wikipedia benchmark — bounded RAM runs sizes the exact
dict can't):

| bytes | `mixnshash` whole-stream | gzip |
|---|---|---|
| 10 M | 0.2751 | 0.3688 |
| 30 M | 0.2729 | 0.3668 |

- **Reaches 30 MB** (240 M bits) on fixed ~370 MB — the bounded-RAM goal, achieved; beats gzip (0.273
  vs 0.367).
- **Collision-limited:** roughly *flat* 10→30 MB (tables saturate), stuck ~0.273 — the +0.048 cost
  caps it, and more data stops helping once the fixed tables are full.

**Conclusion:** fixed-size hashing buys **bounded RAM** (so we reach 30 MB) at the cost of collisions
that **cap quality** and grow with scale. Getting **both** large scale **and** low bits/bit needs large
*tuned* memory + efficient hash tables — practical in **Rust/C++** (the planned core), not pure Python
(370 MB+ Python arrays are clumsy and sizing iteration is slow). enwik8 reference: gzip 0.37, strong
compressors ~0.15; ours ~0.273 (bounded) / ~0.225 (exact, smaller scale) sits between. This is the
clearest signal yet that the **native core is the next real lever**. _(§39 refines this: the native
core gives bounded memory + near-exact quality and ~2× speed, but scale is memory-latency-bound, not
language-bound — Rust is not a 100× lever here.)_

---

## 39. The Rust core — `blmrs`: first results and an honest correction (`blmrs/`)

Started the native core. Installed Rust 1.96 (gnu, self-contained linker) and built `blmrs`: a faithful
port of `mixfast` (logistic mixing, orders 0–4, the same lossless integer keys).

**Correctness:** the HashMap variant is **bit-for-bit identical** to Python `mixfast`
(0.253666 == 0.253666 @500 KB) — the algorithm port is verified.

**Flat engine** (fixed-size open-addressing arrays + 8-bit checksum tags = bounded memory):

| size | `blmrs` flat | exact (Python) | Δ |
|---|---|---|---|
| 500 KB | 0.254028 | 0.253666 | +0.0004 |
| 5 MB | 0.253601 | 0.252790 | +0.0008 |

→ **near-exact**: with proper *high-bit* multiplicative hashing + large tables (2²⁴ slots), the
collision cost is tiny — unlike Python `mixnshash` (+0.048), because Rust affords big flat tables
cheaply. The native core gets **bounded memory *without* the quality hit**.

**Speed** (same model):

| | @500 KB | @5 MB |
|---|---|---|
| CPython `mixfast` | 13.6 s | ~135 s |
| PyPy `mixfast` | 2.26 s | 39.5 s |
| `blmrs` (native flat) | 1.37 s | 22 s |

→ **~1.7–1.8× over PyPy, ~6× over CPython.**

**Honest correction.** Earlier sections framed Rust as "the 100× lever for scale." **That was wrong.**
At scale this workload is **memory-latency-bound** — each bit does ~5 random accesses into tables far
larger than cache, so throughput is ~1.8 Mbits/s (~550 ns/bit ≈ 5 × a DRAM miss) *regardless of
language*. Native buys a **modest ~2×** (lower per-access overhead), not orders of magnitude. The real
Rust wins here are **(a) bounded memory with near-exact quality** (big flat tables Python can't afford)
and **(b)** a clean, fast-to-iterate native base. The genuine path to large speedups is **cache-aware
design** (smaller working sets, fewer/cheaper accesses, SIMD on contiguous data), not language alone.

**Next:** port the strong model (`mixns`) to the native core; explore cache-aware table layouts; the
bounded-memory + near-exact property already lets `blmrs` scale at fixed RAM where Python degraded.

---

## 40. The strong model in the native core — quality at scale (`blmrs/src/bin/strong.rs`)

Ported the **full strong model** (`mixnsfast`: orders 0–6, hashed high orders 8/12/16, byte-match,
sparse, word, two-layer context-selected mixer + global + final, 2 chained SSE/APM, non-stationary
count-halving, RMSProp) to the native core, with flat bounded tables (high-bit hash + checksum tags).

**Verified near-exact** (1 MB War & Peace):

| | whole-stream |
|---|---|
| Python `mixnsfast` (exact) | 0.228847 |
| `blmrs-strong` (native, obits=23) | 0.229253 |

→ Δ **+0.0004** — the port is faithful (tiny flat-collision + libm delta), and ~2.6× faster than PyPy.

**At scale on enwik8** (obits=24), vs Python's bounded `mixnshash`:

| bytes | `blmrs-strong` | Python `mixnshash` (bounded) | gzip |
|---|---|---|---|
| 10 M | **0.2253** | 0.2751 | 0.3688 |
| 30 M | **0.2203** | 0.2729 | 0.3668 |

- The native strong model beats Python's bounded version by **~0.05 bits/bit** — Rust affords bigger
  tables (2²⁴ vs 2²²) and uses the corrected high-bit hash, so collisions don't cripple the high orders.
- It **keeps improving with data** (10 M 0.2253 → 30 M 0.2203), unlike the Python bounded version which
  was collision-capped (flat ~0.273). The strong model genuinely benefits from scale.

This is the concrete payoff of the native core: **the good model at scale, with bounded memory and
near-exact quality** — what Python couldn't do. enwik8 context: gzip 0.37, this **0.220**, strong
compressors ~0.15 — a real compressor between gzip and SOTA, still improving with data. **Honest scope:**
~0.5 Mbits/s (strong model, memory-bound); SOTA needs far more models/memory/tuning; bounded tables
still cap the very highest orders at extreme scale.

---

## 41. Full enwik8 — the headline number (`blmrs-strong`)

Ran the strong native model on the **full enwik8** (100 MB / 800 M bits — the standard compression
benchmark), `obits=27` (~11 GB tables), 28 min, one core.

**Result: whole-stream `0.211107` bits/bit** (last-20% 0.2072) → enwik8 compresses to **~21.1 MB**.

Scaling held to the end — monotonic, still descending:

| size | `blmrs-strong` bits/bit |
|---|---|
| 10 M | 0.2253 |
| 30 M | 0.2203 |
| **100 M (full)** | **0.2111** |

Where it sits on the enwik8 ladder (well-known approximate compressed sizes):

| compressor | enwik8 | bits/bit |
|---|---|---|
| gzip | ~36 MB | ~0.36 |
| bzip2 | ~29 MB | ~0.29 |
| PPMd | ~24 MB | ~0.24 |
| **blmrs-strong (ours)** | **~21.1 MB** | **0.211** |
| lpaq1 | ~20 MB | ~0.20 |
| paq8 | ~16 MB | ~0.16 |
| cmix (SOTA) | ~15 MB | ~0.15 |

So the from-scratch bit-native core, scaled to the native strong model, **beats gzip, bzip2, and
PPMd**, and lands right next to **lpaq1** — a genuine, respectable context-mixing compressor between
the classical tools and the strong PAQ family. **Honest scope:** bits/bit is the model's
**cross-entropy** (the ideal arithmetic-coded size; a real archiver adds a small coder + decompressor
overhead, which the Hutter Prize counts); ~0.5 Mbits/s; SOTA (~0.15) needs many more models + GB-scale
tuned memory + cache-aware engineering. But as a measure of the *model*, **0.211 bits/bit on full
enwik8** is a real, defensible headline for a bit-native predictor built from first principles.
_(Improved to 0.209 in §42.)_

---

## 42. Push toward SOTA — a model-stacking round (`blmrs-strong`)

A round of principled additions to the strong native model, each benched on enwik8:

- SSE/APM chain extended **2 → 4 stages** (added current-partial-byte and match-length contexts);
- a **second, longer byte-match model** (min length 8);
- more **high-order reach** (orders 24, 32) + bigger high-order tables (HBITS 20 → 22).

Effect (whole-stream bits/bit, matched obits):

| size | baseline strong | improved | Δ |
|---|---|---|---|
| 10 M | 0.2253 | 0.2240 | −0.0013 |
| 30 M | 0.2203 | 0.2186 | −0.0017 |
| **100 M (full)** | 0.2111 | **0.2093** | −0.0018 |

**Updated headline: `0.2093` bits/bit → enwik8 ~20.9 MB — now essentially at the lpaq1 level.**

Per addition the gains are small (~0.0003–0.0007), but they **grow with scale** (more data populates
the high orders and the match models, so each is worth more at 100 M than at 10 M).

**Honest conclusion:** model-stacking works and the framework supports it cleanly, but each principled
addition yields ~thousandths. Closing the remaining gap to SOTA (~0.15) is a deep, separate program —
**indirect-context bit-history state machines** (the PAQ "ICM"), an **NN/LSTM mixer** (cmix's big
lever), many more context models, and heavy per-context tuning — not a handful of increments. We have
demonstrated the direction and reached **lpaq1 territory**; SOTA is research-grade from here. Causality
is preserved by construction (every addition is predict-before-update; the base was verified
bit-identical to the causal Python reference).

---

## 43. Path B — open-ended induction of bit-native computations (`induce.py`)

Toward the end goal (**bit-native intelligence**, not language modeling): can the core **induce** the
right computation by *composing* from a library of primitives — learning to compute, rather than only
*selecting* from a 6-member hand-made family (cycles 12–13)?

Induction over a 12-primitive library (latch-k, count-mod-m, count-bucket, max-run, last-k, …),
exhaustive composition to depth 3, content-disjoint held-out + a random-answer scramble control:

| task | induced program | test | scramble | verdict |
|---|---|---|---|---|
| parity | `cmod2` | 1.00 | 0.58 | SOLVED |
| recall | `latch2` | 1.00 | 0.36 | SOLVED |
| mod4 | `cmod4` | 1.00 | 0.30 | SOLVED |
| majority | `cbucket` | 1.00 | 0.59 | SOLVED |
| maxrun≥3 | `maxrun` | 1.00 | 0.70 | SOLVED |
| compose | `latch1 + cmod2` | 1.00 | 0.37 | SOLVED |
| automaton (01-transition parity) | _(spurious)_ | 0.58 | 0.62 | **BREAKS** |
| automaton + primitive | `transcount` | 1.00 | — | SOLVED |

**Findings:**
- **6/7 induced** — the correct *minimal* program for each, generalising to held-out (1.00) with
  scramble at chance, and it picks *different* primitives per task (latch for recall, counters for
  parity/mod-m, bucket for majority, max-run for runs). Genuine learning-to-compute across a diverse
  task set, not pattern-fitting.
- **Breaks honestly.** Transition-parity is not in the library's span → no composition determines it;
  the "best" found is spurious, and the **scramble control exposes it** (scramble 0.62 ≈ test 0.58; a
  real program has scramble ≪ test). The method is self-honest — the controls catch the fake.
- **Extensible** — add a transition primitive and it induces `transcount` → 1.00. The frontier extends
  when the needed primitive exists.

**Significance (honest, no hype):** this is cheap, CPU, example-driven **induction of computations** —
composing and validating programs from data — a mechanism *unlike* LLM scaling, and it learns *what to
compute* across recall / counting / comparison / run-detection / composition.

**The real open question (Path B's frontier):** it is bounded by (a) the primitive **library** (can't
induce what isn't in the span — automaton), (b) composition **depth + search** (exhaustive to 3; deeper
is combinatorial — the known hard part of program synthesis), and (c) **hand-added** primitives. The
genuine "intelligence via a different course" bet is whether the system can **grow its own library** —
*invent the missing primitive from a failure* — rather than be handed it, and whether induction
transfers to **real-data** tasks. Those are the next steps.

---

## 44. Path B, step — primitive INVENTION: grow the toolbox from a failure (`invent.py`)

A step (not the goal) toward open-ended induction: when the fixed library *breaks*, can the system
**invent** the missing primitive by searching a **generative** space — primitives parameterised by a
predicate over (prev, cur) bits × an aggregation (count mod m / ever / max-run)?

| task | base test | result |
|---|---|---|
| 01-parity | 0.57 | **INVENTED** `01:cnt2` → 1.00 (the 01-transition counter) |
| 10-parity | 0.62 | **INVENTED** `10:cnt2` → 1.00 |
| 11-parity | 0.60 | **INVENTED** `and:cnt2` → 1.00 (found "both bits 1") |
| saw-11 | 1.00 | base already solved (`maxrun`) |
| 010-parity | 0.61 | **NOT invented** — needs a 3-bit predicate, beyond the 2-bit space |

**Findings:**
- **Invention works.** On tasks the fixed library can't do (base ≈ chance), the system searches the
  generative space and *discovers* the missing primitive (transition / pair counters), solving at 1.00,
  scramble-clean. It grows its own toolbox from failures.
- **Self-honest.** base 0.57–0.62 → invention 1.00; the scramble control guards against spurious
  "inventions."
- **Invention has its own frontier.** `010-parity` needs a 3-bit predicate, beyond the 2-bit generative
  space → honestly not invented — and that frontier is *itself* extensible (a 3-bit-predicate space
  would reach it).

So there is a **hierarchy of frontiers**, each extending the last — fixed library → invention (2-bit
predicate DSL) → richer generative spaces — every level cheap, example-driven, self-honest, mappable.

**Significance + honest scope (a step, not the goal):** a recursive, *verified*, example-driven
mechanism for **discovering** computations — qualitatively unlike LLM scaling. But each generative
space is still **hand-defined** here; truly open-ended discovery needs the system to grow its **own**
space. Next: a learnable/growable generative space; deeper computations; **inventing groupings** (the
bit→byte→event unit question); and grounding on a real test case.

---

## 45. Path B step — recursive synthesis: grow the space, hit the search wall (`recurse.py`)

A step beyond invent.py (which broke at 3-bit patterns): give the system a **fixed grammar** of stream
transducers `{s, not, lag_k, and/or/xor}` + aggregations and let it **synthesise arbitrary-depth
computations by composition** (iterative deepening). A fixed finite grammar that *generates an unbounded
space*.

**Honesty:** the search tries thousands of expressions, so a single content-disjoint split is **not** a
sufficient control — every winner is **re-validated on 5 fresh seeds**, SOLVED only if it survives.

| task | found | cross-seed (5 fresh seeds) | program |
|---|---|---|---|
| 01-parity | d2 | 1.00 | `cnt4(~s ^ lag1(~s))` — a non-obvious but correct program |
| 010-parity | d2 | **1.00** | `cnt2(lag1(s) & (~s & lag2(~s)))` — the **exact 010 detector** |
| 0110-parity | — | — | NOT FOUND within budget |

**Findings:**
- **Grows its own space.** It synthesises the *exact* 010 detector by composition — crossing the
  3-bit-pattern frontier the fixed 2-bit-predicate invention (§44) could **not** — from a fixed finite
  grammar, no hand-added predicates. Verified across 5 fresh seeds (cross-seed is essential; here it
  *confirms* the programs are real, not search-overfit).
- **Hits the search wall.** `0110-parity` is expressible in the grammar, but the bounded iterative
  deepening did not find it — the combinatorial explosion of program synthesis. **The limit is now
  search, not expressiveness.**
- **Methodology (no fake realities):** at large search scale, single-split controls are insufficient;
  cross-seed validation is required, and is built in. The honesty guard must scale with the search.

**Significance + honest scope (a step):** the "grow your own space" mechanism works for moderate depth —
a fixed grammar generating *verified* computations that cross prior frontiers. It now hits the classic
program-synthesis **search wall**; scaling it (type-directed / library-learning / guided search) is the
open frontier, and the genuine "different course" bet rests there.

---

## 46. Path B step — library learning: the loop works, abstraction quality is the wall (`library.py`)

DreamCoder-style bootstrapping: process a curriculum and add each solved program (its feature stream)
to the library as a reusable primitive, so harder tasks compose from learned pieces. Cross-seed
validated throughout.

| stage | result |
|---|---|
| flat (base grammar) `0110-parity` | NOT FOUND (the §45 search wall) |
| curriculum `01-parity` | SOLVED → learned `L1 = ~s^lag1(~s)` (transition stream) |
| curriculum `10-parity` | SOLVED → learned `L2 = s^lag1(s)` (= same transition stream, **redundant**) |
| curriculum `010-parity` | SOLVED **reusing L1**: `~s & (L1 & lag1(L1))` |
| `0110-parity` WITH the library | still NOT FOUND |

**Findings:**
- **The bootstrapping loop works.** `010-parity`'s solution *reused* the learned `L1` — the library is
  genuinely composed-from; reuse happens.
- **But it did not crack `0110`.** The extracted abstractions are not the *right* reusable pieces:
  `L1` and `L2` are **redundant** (both the transition stream), and it never learned the clean
  01-/10-detectors that `0110` (= `and(lag2(01-detector), 10-detector)`) needs — the "keep the first
  solution" bias yields non-ideal primitives.
- **Honest crux:** the loop is easy; **good abstraction extraction is the wall.** Cracking it needs
  refactoring / compression to find the right common sub-programs (DreamCoder's actual hard step), not
  just keeping winning streams.

**Significance + honest scope:** the whole Path B arc — induction → invention → recursive synthesis →
library learning — is mechanically real and **self-honest** (cross-seed throughout; no fake success
here, the loop works but the wall stands). It traces the "different course" bet to its genuine crux:
**discovering *good* abstractions efficiently** — search + compression over programs. That difficulty
(not expressiveness, not the loop) is where the bet now squarely rests, and it is a known-hard open
problem.

---

## 47. Path B step — abstraction refactoring: a precise, honest miss (`refactor.py`)

Attempt to fix §46 by **mining** reusable primitives: enumerate the solution space per curriculum task
and keep every distinct **cross-seed-validated** solving stream as a candidate abstraction (since
`cnt2(01-detector)` *also* solves 01-parity, the clean detectors should be *in* the solution space).
Then test `0110` with the enriched library.

**Result:** mined 8 (01-parity), 12 (10-parity), 2 (010-parity) validated solving streams → a
13-primitive enriched library → `0110-parity` **still NOT FOUND**.

**Why (the precise crux):**
- The mined abstractions are **boundary-tricks** (transition-XOR forms like `~s^lag1(~s)`), *not* clean
  detectors. The grammar's zero-fill `lag` **contaminates** clean detection at the edge:
  `and(~lag1(s), s) = [s[0], 01-det₁, …]` → `cnt2` = `(s[0] + #01) mod 2` ≠ 01-parity. So the **exact**
  pattern-detectors don't aggregate cleanly; the synthesis finds boundary-trick solutions that pass
  validation but **don't compose** to deeper patterns.
- Mining solving streams therefore surfaces tricks, not the reusable detectors `0110` needs.

**Honest conclusion:** cracking the wall needs (a) a **boundary-aware grammar** (so exact pattern
detection aggregates correctly) and (b) **parameterised abstraction** — a detector *template*
parameterised by the pattern, discovered by anti-unifying analogous solutions — i.e. higher-order /
typed program synthesis. The "different course" is real and **self-honest at every step**; its crux is
now precisely located: **parameterised abstraction discovery + boundary handling**, the heart of
inductive program synthesis (DreamCoder/Stitch territory) and a hard open frontier. No fake success —
the attempt honestly failed and told us *exactly* why. _(A focused boundary fix — adding validity
masks `m_k` to the grammar — was also tried: it mined more streams (library of 26) but `0110` still
NOT FOUND; the synthesis folds masks into new trick-forms rather than clean composable detectors.
Confirms the wall is parameterised abstraction, not boundary handling alone.)_

---

## 48. Cross-domain generalisation — DNA (`blmrs-strong` on real genomes)

Does the core generalise off English to a domain with genuinely different structure (codon period-3,
reverse-complement palindromes, long repeats)? The "dumb adapter" is just packing each base
(A/C/G/T → 2 bits) and streaming it into the **existing** `blmrs-strong` — **no DNA-specific tuning**.
Metric = the same whole-stream bits/bit (× 2 = **bits/base**); raw 2-bit packing = the 2.000 floor.

| genome | bases | `blmrs-strong` bits/base | vs floor 2.0 | specialised DNA field |
|---|---|---|---|---|
| E. coli K-12 (low redundancy) | 4.64 M | 1.936 (last-20% 1.912) | −0.06 | ~1.85–1.90 (E. coli is hard) |
| human chr21 (repetitive) | 40.1 M | **1.679** (last-20% 1.768) | **−0.32** | ~1.6–1.7 (specialised band) |

**Findings:**
- **It generalises off English.** Below the 2-bit floor on both genomes with only a 2-bit adapter —
  capturing real DNA structure it was never built for.
- **On the repetitive human chromosome it lands at 1.679 bits/base — in the specialised
  DNA-compressor band (~1.6–1.7)** — because the byte-**match model** (its text-repeat strength)
  transfers directly to DNA's long repeats. Apples-to-apples against the same context-mixing family
  (NAF/GeCo3/JARVIS), off-the-shelf.
- **On the hard, low-redundancy E. coli it is near the floor (1.936)**, just above specialised
  (~1.85–1.90): few repeats for the match model, and the untuned weaknesses bite.

**Honest weaknesses (all fixable):** (a) **byte-misalignment** — `blmrs-strong` is hardwired to 8-bit
bytes, but DNA's units are 2-bit bases and **codons (period-3 = 6 bits)**; (b) **no reverse-complement
modelling** (DNA compressors model RC palindromes explicitly); (c) on low-redundancy genomes the match
model has nothing to bite. A base/codon-aware adapter + RC modelling would likely push **both** lower
(E. coli toward ~1.85, human below 1.6).

**Conclusion:** the bit-native core generalises cross-domain off-the-shelf, and on the genomes that
matter most (large, repetitive) is **competitive with specialised DNA compressors of the same family**
— strong evidence the architecture captures *real structure*, not English-specific quirks. (Reference
ranges are from the DNA-compression literature; bacterial ~1.72–1.9, human ~1.6–1.7. Exact same-genome
GeCo3/JARVIS numbers would sharpen the head-to-head.)

---

## 49. DNA — base/codon-aware adapter (`dna.py`)

§48's byte-aware result left two fixable weaknesses (byte-misalignment, no DNA-specific structure).
`dna.py` is a base/codon-aware context-mixing model: it operates on **2-bit bases**, conditions context
on previous **bases**, adds an explicit **codon phase** (period-3 reading frame), and runs a
**base-granular VERIFIED match model**. RMSProp-stabilised mixing.

| genome | byte-aware (§48) | base/codon | + rev-complement | specialised field | floor |
|---|---|---|---|---|---|
| E. coli (low redundancy) | 1.936 | 1.9145 | **1.9079** | ~1.85–1.90 | 2.0 |
| human chr21 (repetitive) | 1.679 | 1.6438 | **1.6156** | ~1.6–1.7 | 2.0 |

**Findings:**
- Each targeted adapter helps: aligning to DNA's natural units (2-bit bases, codon period-3) + a
  verified base-match model, then a **reverse-complement** match model (inverted repeats), improve
  **both** genomes over the byte-aware model.
- **Human chr21 → 1.6156 — at the better edge of the specialised DNA-compressor band** (~1.6–1.7,
  NAF/GeCo3/JARVIS family), essentially **matching** that class.
- E. coli → 1.9079 — within the specialised range (~1.85–1.90) on the hardest (low-redundancy) genome.
- Reverse-complement helped more on human (−0.028) than E. coli (−0.007) — as expected, since the human
  genome is dense in inverted repeats (Alu in both orientations).
- **Debugging note worth keeping:** the first version was *worse than the floor* (2.0067) because the
  match model fired on hash **collisions** with high confidence — confident-wrong predictions are
  catastrophic. **Verifying matches** (comparing the actual preceding bases) fixed it; the RC match is
  verified the same way. A real lesson for any confident sub-model: **verify before you trust.**

**Headline:** off a from-scratch *English* compressor, three targeted DNA adapters (base/codon
alignment, base-match, reverse-complement) move the core from "generalises off English" to **matching
specialised DNA compressors of the same family** (human in-band ~1.62, E. coli in-range ~1.91) — concrete
evidence the architecture models *real structure*, cheaply **retargetable per domain**. (Reference
numbers are literature ranges; an exact same-genome GeCo3/JARVIS run would sharpen the head-to-head.)
## 50. Path B — CRACKING the wall: boundary-aware detectors + induced `detect(P)` (`wake_lgg.py`, `probe_*.py`)

§47 located the crux precisely: the wall needs **(a) a boundary-aware grammar** so exact pattern
detection aggregates cleanly, **and (b) parameterised abstraction** — a `detect(P)` *template*, not
concrete winning streams. A fan-out research pass (12 angles) then **built and adversarially verified
five independent prototypes**, each re-run from scratch against the live `induce.py` harness. The
baseline wall was re-confirmed first: `recurse.py` finds `01-parity` (`cnt4(~s^lag1(~s))`) and
`010-parity` (`cnt2(lag1(s)&(~s&lag2(~s)))`) but reports **`0110-parity` NOT FOUND** at depth-6 / 9000
streams. All five prototypes cross it — cross-seed 1.00 on fresh seeds, scramble-clean, and
**generalising to UNSEEN patterns** (not just hitting `0110`):

| prototype | mechanism | `0110` | unseen-pattern generalisation | honest depth |
|---|---|---|---|---|
| `probe_antiunify_…` | boundary-aware `lit(s,k,b)` detector + **Plotkin LGG induces `detect(P)`** | ✅ cs 1.000 | **4/4** (incl. unseen arity `00110`); reconstructs the hand detector bit-for-bit | template genuinely **induced** by anti-unification |
| `probe_param_detect_cegis_…` | `detect(s,P)` + CEGIS over `(P, agg)` | ✅ cs 1.000 | **6/6** by binding only `P`, frozen `agg=cnt2` | template hand-supplied; only the parameter learned |
| `probe_sentinel_boundary_…` | sentinel (option-monad) lag fixes boundaries; the project's **own** bottom-up search then finds `0110` | ✅ depth-4 `cnt2(~s&lag1(s)&lag2(s)&~lag3(s))` | **3/3** | proves the **boundary half** independently |
| `probe_diff_soft_detector_…` | differentiable conv-detector + smooth parity head (gradient, not enumeration) | ✅ recovers `W=[0,1,1,0], m=4` | **4/4** incl. `m=5` unseen arity | the **"different course"** cross-check |
| `probe_cvec_canonical_dedup_…` | full-domain cvec dedup **excludes the trick e-class** | ✅ full-domain-exact | **15/15** unseen (lengths 2–6) | the **honesty layer** |

The verified fixes:
- **Boundary-aware detection.** Replace zero-filling `lag` with a positional literal
  `lit(s,k,b) = [1 if (i>=k and s[i-k]==b) else 0]`, AND the contiguous literals, and **mask to
  `i>=m-1`** (the validity window). Then `cnt2(detector_P) ≡ trans_pat(s,P) % 2` is an **exact
  identity** — verified *exhaustively* over all inputs at L=6, 8, 10 (zero mismatches), so it is a
  theorem, not an L=12 coincidence. The zero-fill version disagrees on 12–49 % of rows. The sentinel
  probe is the clincher: **fix the boundary and the existing blind search finds `0110` on its own** —
  the search wall was a *representation* defect wearing a search costume.
- **Parameterised abstraction.** Anti-unify the per-pattern detector ASTs (group by arity → the
  polarity slot that varies becomes a metavariable `P[j]`; a fold over the literal list lifts arity)
  into `detect(P) = cnt2(AND_j lit(s, |P|-1-j, P[j]))`. Freeze it, then for an unseen task bind **only**
  `P` from examples (leave-one-pattern-out). This is the move `library.py`/`refactor.py` could not make
  because they kept concrete streams.

**Findings:**
- **The wall is cracked, verified five ways.** `0110-parity` is solved scramble-clean, cross-seed,
  leakage-free, by a parameterised detector — by five independent routes (enumerative, CEGIS,
  boundary-algebra, gradient, e-graph). The baseline genuinely fails; the crack is non-trivial.
- **Generalisation is real, not memorisation.** Held-out patterns the solver was never given
  (`0011`, `1001`, `00110`, …) are solved by binding only the parameter, surviving cross-seed +
  balanced-scramble + negative controls (`detect(P)` correctly refuses to fire on majority /
  total-parity / mod3, and `|P|≥6` patterns are honestly UNSAT — too rare at L=12).
- **A real harness bug, fixed.** `recurse.py`'s fixed `scramble<0.7` gate is **mis-calibrated**:
  P-parity labels are class-imbalanced at L=12 (e.g. `00110` majority-baseline ≈ 0.78), so the gate
  **false-rejects** genuine solvers on rare patterns. The fix is **balanced accuracy** (mean per-class
  recall) → real 1.00 / scramble ≈ 0.50 for every pattern. (Verified separately: the `0110` crack also
  survives the *original* raw `<0.7` gate, so this change is not load-bearing for the headline — only
  for not false-rejecting imbalanced patterns. Worth back-porting to `recurse.py`.)

**Significance + honest scope (the ceiling, named not hidden):** the "different course" — *learn what to
compute, then abstract it into a reusable parameterised primitive* — now crosses the exact wall that
stopped induction → invention → recursion → library learning → refactoring. **But:** in four of the five
prototypes the *structural shape* of `detect(P)` (AND-of-delayed-literals + validity mask) is **built
into the grammar by hand**; only `antiunify` *induces* the shape (by LGG over independently-searched
detector ASTs), and even there LGG suffices only because the wake-search returns canonically-aligned
ASTs. **No prototype yet demonstrates fully autonomous discovery of the detector template from raw
failures alone**, and everything here cracks **one** task family — sliding-window pattern-count parity.
The genuinely open question — *is parameterised abstraction a general mechanism, or a per-family manual
move?* — is therefore **not yet settled**. The canonical mechanism is consolidated in `wake_lgg.py`
(wake-synthesise clean detectors → LGG-induce `detect(P)` → bind-only-`P` generalisation, with the cvec
honesty layer and balanced gate); the make-or-break test is **§51 (M3): the same machinery on a
genuinely *different* family**, with no hand-built template. No fake success — the wall is down, and the
next wall is named.

---

## 51. Path B — M3: is the abstraction mechanism GENERAL, or a per-family move? (`m3_different_family.py`)

§50 cracked the wall for **one** family (parity of contiguous-pattern occurrences) and named the real
open question: when we move to a *different* family, must we hand-write a new template, or does the same
`WAKE → SLEEP(LGG) → BIND` machinery discover the new abstraction itself? If it is a per-family manual
move, the wall just reappears one family up and "open-ended" fails. M3 runs the identical machinery —
with a **general** grammar (literals over arbitrary lag *subsets*, not just contiguous; a small set of
aggregations) — on three graduated families, with the same honest controls (balanced cross-seed on
fresh seeds, multi-draw balanced-scramble, full-domain direct-solver check over all 2¹² inputs, no
pattern supplied).

| family | what is new | result |
|---|---|---|
| **M3a — gapped patterns** (`1.1`, `1..1`, `0.1`, `1.1.1`) | literal conjunctions at **non-contiguous** lags; the §50 contiguous `detect(P:string)` literally cannot encode a gap | **PASS — 5/5 held-out**: WAKE recovers the exact gapped detector from labels, the parameter generalises *string → spec* (a set of `(lag,bit)` literals), every held-out pattern is full-domain-exact, cross-seed 1.000, scramble ≈ 0.50, recovering the canonical detector |
| **M3b — threshold counting** (`#occ(P) ≥ t`) | a **second abstraction axis**: the *aggregation* changes from parity to a count-threshold, parameterised by `t` | **PASS — 4/4 held-out**: a parity-only grammar provably cannot solve `#11≥2`; extending the aggregation lets WAKE induce a `(detector, threshold)` template and bind unseen `(P,t)` — cross-seed 1.000, scramble ≈ 0.50 (milestone M2: a second induced axis) |
| **M3c — count equality** (`#0 == #1`, the counting essence of `aⁿbⁿ`) | a two-sided count equality `sum == k` — outside the detector + {parity, threshold} span | **UNSAT, honestly** — no `(spec, aggregation)` in the grammar is a full-domain-exact solver; WAKE returns nothing. The wall **relocates** here |

**Findings:**
- **Not a per-family manual move — within a paradigm.** The *same* `WAKE → SLEEP → BIND` machinery
  generalises across families when they lie in the **detector + aggregation** span: M3a by enriching the
  **parameter** (contiguous string → arbitrary literal-spec), M3b by enriching the **aggregation**
  (parity → threshold). In both, the abstraction is *induced and bound*, not re-hand-built, and every
  held-out member passes the full-domain-exact + cross-seed + scramble gate.
- **The boundary is real and precisely located.** M3c (`#0==#1`) is **honestly UNSAT**: count equality
  is not a fixed-window parity/threshold, so the detector paradigm does not reach it. The mechanism does
  not fake a solution — it returns nothing, exactly as a self-honest system should.
- **So the honest answer is nuanced, not a yes/no.** Parameterised abstraction is a **general mechanism
  *within* a computational paradigm**, spanning multiple parameter and aggregation axes — but crossing
  into a *genuinely new* paradigm (counting/balance) still needs a **new primitive**. That is precisely
  the §44 *invention* move (grow the generative space), now relocated one level up: the open frontier is
  **inducing the new aggregation/primitive itself**, not the parameter within a fixed one.

**Significance + honest scope:** M3 settles the §50 open question in the only honest way — *partially,
with a sharp edge*. The "different course" abstraction mechanism is **not** brittle-per-family inside the
detector+aggregation paradigm (a real result: it transfers across gap-structure and across the
parity↔threshold axis with no new template), **and** it has a **named, falsifiable boundary** at
count-equality where a new primitive is required. The Path B ladder is therefore: ✅ M1 cracked
(`wake_lgg.py`), ✅ M2/M3 multi-axis generalisation within the paradigm (`m3_different_family.py`), and
the next genuine frontier — **autonomous discovery of a new primitive/aggregation when the paradigm runs
out** (M3c's `count==k`, then stateful/counting languages) — is exactly where the bet now rests. No
fake success: two families transferred, one honestly did not, and we know precisely why.

---

## 52. Path B — crossing the relocated wall: invent the missing AGGREGATION (`invent_agg.py`)

§51 left an honest boundary: `#0==#1` (count **equality**) was UNSAT because no
detector + {parity, threshold} solver exists. That is the §44 situation — *the library
breaks* — lifted one level: the missing piece is now an **aggregation**, not a detector
parameter. This step applies the §44 **invention** move to the aggregation axis: when the
known aggregations `{cnt2, ge_k}` fail, search a **generative** space of count-predicates
`compare(count, k)` for `compare ∈ {ge, le, eq, ne}` and *invent* the missing comparator —
then fold it into a parameterised template and generalise.

| step | result |
|---|---|
| **(A) invent from failure** | `{cnt2, ge_k}` is UNSAT on `#0==#1`; searching the generative comparator space **invents `eq`** → solver `count(detector) eq 6`, **full-domain-exact**, cross-seed 1.000, balanced-scramble 0.52, a genuinely new form |
| **(B) generalise the invented comparator** | fold into a template `detect(spec) op k` parameterised by `(spec, op, k)`; **4/4 held-out** count-comparison tasks (`#1 ne 6`, `#0 eq 4`, `#10 le 1`, `#11 ge 2`) bound from labels, all full-domain-exact, cross-seed 1.000, scramble ≈ 0.51 |
| **(C) the next honest boundary** | **Dyck-1** balanced parentheses (`0='(' +1`, `1=')' −1`; balanced iff total `==0` **and** every prefix sum `≥0`) stays **UNSAT even with the invented comparators** |

**Findings:**
- **The invention move lifts to the aggregation axis.** A failure that was a hard boundary in
  §51 is crossed by *inventing the missing comparator from a generative space* and binding it —
  the same self-honest, example-driven discovery as §44, now over aggregations. The invented
  `eq` is validated full-domain-exact (a coincidence cannot pass), not just split-lucky.
- **It generalises, honestly — including equivalent re-parameterisations.** Held-out
  count-comparison tasks are all solved; notably the system sometimes finds an *equivalent* exact
  program rather than the textually-intended one (e.g. `#1 ≤ 3` solved as `#0 ≥ 9`, flagged
  `canonical=False`) — which is correct (it is the same function, proven over all 2¹² inputs) and
  a fair reminder that the gold standard is full-domain equivalence, not surface form.
- **The next wall is precisely located: stateful computation.** Dyck-1's `total==0` part *is*
  `#0==#1` (now solvable via the invented `eq`), but the **prefix-`≥0`** condition is not a
  function of any global count — it needs a running **prefix-min / counter** over the stream. No
  fixed-window detector + count comparison can express it, so it is honestly UNSAT. The boundary is
  the *stateful prefix condition*, not the equality.

**Significance + honest scope:** the **hierarchy of frontiers** the project has tracked since §43
extends cleanly and self-honestly — fixed library → invented detector primitives (§44) → recursive
synthesis (§45) → induced parameterised detectors (§50) → multi-axis generalisation (§51) → **invented
aggregations (§52)** — each level crossed by discovery within a generative space, each validated by
full-domain exactness + cross-seed + scramble, each then exposing the next wall. The **standing honest
caveat** is unchanged and now sharply scoped: the *generative space itself* (here, the comparator set
`{ge,le,eq,ne}`) is hand-provided, exactly as §44's predicate DSL was; the deepest open problem remains
**growing the generative space autonomously**. And the next concrete target is no longer a pattern or a
count-predicate but **stateful computation** — a running counter / automaton (Dyck, `aⁿbⁿ`,
prefix-balance) — which connects back to §43's automaton and the latch/counter mechanisms of cycles
9–13. That is where the bit-native "learn-what-to-compute" bet now genuinely rests.

---

## 53. Path B — the STATEFUL frontier: invent a running counter (`stateful.py`)

§52 located the next wall precisely: Dyck-1 balanced parentheses stayed UNSAT even with the
invented count-comparators, because *"every prefix sum ≥ 0"* is not a function of any **global**
count — it needs a running **prefix-min**, i.e. **state** carried along the stream (the cycles 9–13
mechanism: latch / running-XOR / mod-m counter, now needed for *program induction*). This step
invents a stateful primitive and runs the same `WAKE → SLEEP → BIND` machinery over it.

- **primitive:** a signed running counter `bal[i] = bal[i-1] + step(s[i])` with `step: 0→+1, 1→−1`;
- **readouts:** `final = bal[-1]`, `min` = min prefix, `max` = max prefix;
- **atom:** `compare(readout, k)` reusing the §52 comparator DSL; a feature is one atom **or** a
  conjunction of two (Dyck needs *balanced* AND *never-negative*).

| step | result |
|---|---|
| **(A) Dyck without state** | UNSAT in the §52 detector+comparator grammar (a global count cannot see the prefix condition) — boundary holds |
| **(B) invent the counter** | WAKE (readout/op/k **not** supplied) finds Dyck = **`final le 0 AND min ge 0`**, **full-domain-exact**, cross-seed 1.000, balanced-scramble 0.50 |
| **(C) generalise** | leave-one-out over a single-counter family — curriculum `{min≥0, final==0, max≤2}`, **4/4 held-out** (`final≥1`, `max≥3`, `min≤−2`, `final≤0`) recovered from labels, all full-domain-exact, cross-seed 1.000 |
| **(D) next boundary** | **palindrome** `s == reverse(s)` is UNSAT in **both** the detector+comparator *and* the single-counter grammar |

**Findings:**
- **The hierarchy of frontiers extends into STATE.** A running counter — the cycles 9–13 mechanism,
  now *invented from a program-synthesis failure* — crosses §52's wall: Dyck-1 is solved
  full-domain-exact, scramble-clean, cross-seed. The solver found is `final le 0 AND min ge 0`, an
  *equivalent* exact form of `final == 0 AND min ≥ 0` (under `min ≥ 0`, `final ≤ 0` forces
  `final = 0`) — proven over all 2¹² inputs, an honest reminder that full-domain equivalence, not
  surface form, is the bar.
- **It generalises across the single-counter family.** Held-out `(readout, comparator, threshold)`
  tasks are all bound from labels and full-domain-exact — the stateful abstraction is parameterised
  and reused, not re-hand-built per task.
- **The next wall is, again, precisely located: non-local / multi-counter structure.** Palindrome
  compares position `i` to `L-1-i` — a two-ended relation a single left-to-right counter cannot
  carry — so it is honestly UNSAT in both grammars. The frontier relocates to **two-way / stack /
  multi-counter** computation (palindrome, `aⁿbⁿcⁿ`).

**Significance + honest scope:** the Path B ladder now reads §43 fixed library → §44 invented detector
primitives → §45 recursive synthesis → §50 induced parameterised detectors → §51 multi-axis
generalisation → §52 invented aggregations → **§53 invented stateful counter** — six honest levels, each
crossed by discovery within a generative space, each validated by full-domain exactness + cross-seed +
scramble, each exposing the next wall. The **standing caveat persists and is identical at every level**:
the generative space itself (here: the step map `{0→+1,1→−1}`, the readout set `{final,min,max}`, and
depth-2 conjunction) is **hand-provided** — autonomous growth of the space remains the deepest open
problem. The honest headline is *not* "general intelligence below language"; it is a **self-honest,
example-driven, full-domain-verified mechanism for inducing and reusing bit-native computations that
climbs a real ladder of computational power** — detectors → counts → comparisons → state — with each
boundary found, named, and crossed, and the next (non-local / stack-structured computation) now squarely
in view. That is a concrete, falsifiable foundation for the "different course" bet, not a claim that the
bet is won.

---

## 54. Path B — a REAL model on REAL data: induce the representation, mix to predict, scale (`real_test.py`, `real_mix.py`, `real_scale.py`)

§43–47 and §50–53 are mechanism on synthetic `gen()` bit-tasks (solved to full-domain-exact 1.000). Mechanism is
not the goal; the goal is a real bit-native model. This step is the honest real test: run Path B's
induction — *search bit-native computations and keep the ones that predict, by held-out bits/bit* — on
**real** streams (English text, and **E. coli DNA in native 2-bit**, the purest real bit stream), and
unify it with Path A's predictor. Metric throughout = held-out / online **bits/bit** (cross-entropy =
compression; raw 1.0000, lower better), externally referenced against gzip on the same stream; causal,
stdlib-only.

**(a) Induction on real data (`real_test.py`).** Greedy forward selection over a computation pool
(simple `{lag, mod}` vs `{… + running counts + detectors}`), held-out bits/bit. The induction
**discovers real structure** — it selects `mod 8` first on text (*rediscovers the byte from raw bits*),
`lag 3/6` on DNA (codon scale). Path B's own primitives (popcount `bucket`, pattern `det`) **are
selected and lower bits/bit** (text 0.506→0.426). Honest limit: a single sparse conjunction-table beats
order-0 but trails gzip on text (0.426 vs 0.380); on DNA it already beats gzip (0.974 vs 0.995).

**(b) The unification — induce + mix (`real_mix.py`).** Path B induces *what to compute* (the predictive
**period/unit** found by a quick scan + counter/detector contexts); Path A **mixes** them online
(logistic context mixing, the lpaq idea). One engine (the framing's M6: compression = prediction =
program-finding). It **beats gzip on both** real streams. On DNA — where the 8-bit byte assumption is
*wrong* — inducing the right unit (codon, p=6) **plus reverse-complement contexts** (base-complement is
a bit-flip in 2-bit DNA, so restriction-site RC-palindromes are the §53 palindrome structure on real
data) beats the byte model *and* gzip.

**(c) Scaling (`real_scale.py`).** Rebuilt with the project's proven levers — **integer rolling keys**
(no per-bit slicing; cf. `mixfast`), a **stretch LUT** (PAQ), an **SSE/APM** recalibration stage, and a
revcomp table — ~4× faster, multi-MB tractable. The scaling law holds:

| stream | 0.5 MB | 1 MB | 2 MB | 4 MB | gzip |
|---|---|---|---|---|---|
| **text** (corpus_big, whole) | 0.2462 | 0.2417 | 0.2403 | **0.2328** | ~0.36 |
| **text** +SSE+order8 (last-20%) | — | 0.235 | 0.229 | **0.2167** | ~0.36 |
| **text** (homogeneous Shakespeare, whole) | — | 0.2565 | 0.2510 | **0.2431** | ~0.377 |
| **DNA** (ecoli, +revcomp+SSE, whole) | — | 0.9701 | 0.9709 | **0.9658** | ~0.992 |

**Findings:**
- **It is a real, scalable model on real data.** Held-out bits/bit falls monotonically with data on
  homogeneous text (0.2565→0.2431, still dropping), reaching **0.217 (mature) on corpus_big at 4 MB —
  lpaq1 / the §41 enwik8-headline band (0.209)** — and DNA reaches **≈1.91 bits/base** (0.9534 mature),
  each beating gzip by a widening margin.
- **The induction is real and interpretable at every scale:** it rediscovers the **byte** (`p=8`) on
  text and the **codon** (`p=6`) on DNA from raw bits, and its counter/detector/revcomp primitives —
  the very computations the synthetic ladder learned to induce — measurably help, *most* on DNA where
  the representation must be discovered (the byte model is wrong there).
- **Honest scope.** Not SOTA (cmix ~0.15 on text). On *text* the byte unit is already right, so the
  induced engine ties the strong byte model (both beat gzip) — induction's clear *win* is on DNA. A 6 MB
  `corpus_big` point regressed: the documented **heterogeneous-corpus artifact** (the homogeneous
  single-work curve is cleanly monotone, confirming it). And the pure-Python per-bit loop caps practical
  scale at ~6–10 MB.

**Significance:** the "different course" is no longer only a synthetic-task mechanism — it is a **real
bit-native predictive model that scales on real data and beats gzip on two very different streams (text
and genome) by inducing what to compute**, unified with Path A into one engine. The next genuine scaling
lever is the **Rust core (`blmrs`)** — port this *induced-representation* engine there to reach
enwik8 / full-genome (100 MB) scale, exactly as §39–41 did for the byte model, now general.

---

## 55. Native DNA at full-genome scale — the Rust port (`blmrs/src/bin/dna.rs`)

§49 reached the DNA specialist band (E. coli 1.908, human chr21 1.616 bits/base) with `dna.py` — a
2-bit-base, codon-phase, reverse-complement-aware context mixer — but in pure Python, so it could not
run whole chromosomes at speed. This is the §54 "next lever" executed: a faithful native port into the
Rust core, using `strong.rs`-style **bounded-RAM flat open-addressing tables** (multiplicative hash +
8-bit checksum tag) for the base-history orders, with the forward and reverse-complement match models
ported verbatim. The byte-oriented `strong.rs` is left untouched; `dna.rs` is the period-/unit-correct
engine (the unit is the 2-bit base, the period is the codon-3 reading frame, inverted repeats via
reverse-complement — none of the 8-bit byte assumptions).

| genome | bases | `dna.py` (exact dict) | **`blmrs-dna` (Rust)** | time |
|---|---|---|---|---|
| E. coli — 200 K slice (cross-check) | 200 000 | 1.9520 | **1.9520** | 0.4 s |
| **E. coli — full** | 4 641 652 | 1.908 (§49) | **1.9081** | **8.1 s** |
| **Human chr21 — full** | 40 088 616 | 1.616 (§49) | **1.6255** | **67.6 s** |

**Findings:**
- **Verified, then scaled.** The port is **bit-exact** against `dna.py` at small scale (1.9520 =
  1.9520, where the flat table holds no collisions = the exact dict), and reproduces §49's
  specialist-band numbers **on the full genomes**: E. coli 1.9081 and human chr21 1.6255 (the latter
  within the flat-collision delta of the Python 1.616). A whole human chromosome — **40 M bases / 80 M
  bit-predictions — in 68 s** (~0.59 Mbase/s), which `dna.py` could not do (it would take ~11 min).
- **The native lever is bounded-memory-at-scale, as in §39–41.** Flat hashed tables (obits 23–24) give
  fixed RAM with near-exact quality; the genome-scale run is what the Rust core exists for.
- **It confirms the representation thesis on real genomes at scale.** The win over a byte model is the
  *unit/period* (2-bit base + codon-3) and *reverse-complement* — the bit-native "induce/choose the
  right representation" lesson, now competitive with domain specialists across a 9× size range
  (E. coli → human chr21), off a from-scratch English compressor's machinery.

**Significance + honest scope:** the bit-native core is now **natively retargetable and scalable** —
the same project, two engines (`strong.rs` for byte streams at enwik8 scale, `dna.rs` for genomes at
chromosome scale), each unit-correct, each bounded-RAM. Honest scope: `dna.rs` is a *faithful* port of
`dna.py`'s hand-built DNA representation, not an *auto-induced* one — wiring the period/unit *discovery*
(§52–54, `real_scale.py`) into the native engine so it picks the unit itself is the next step. And text
at full enwik8 scale via the induced engine (vs the already-native `strong.rs`) remains to be run.

---

## 56. The period-DISCOVERING native engine — induction at scale (`blmrs/src/bin/induced.rs`)

§55's honest gap was that `dna.rs` is a *hand-built* DNA representation. This closes it: the period/unit
*discovery* (§52–54, `real_scale.py`) is wired into the native core. `induced.rs` runs a quick held-out
**period scan** to discover the predictive unit `p` from the data, then runs unit-aligned online logistic
context mixing at `p` (integer rolling keys + `strong.rs`-style bounded-RAM flat tables + the 33-knot
APM). It is **not told the unit** — it discovers the byte on text and the codon on DNA.

**Comprehensive test battery (all pass):**

| test | result |
|---|---|
| period discovery | English text → **p=8** (byte); source code → **p=8**; 2-bit DNA → **p=6** (codon) — auto-discovered |
| correctness vs Python (`real_scale.py`, same params) | text Δ**0.0005** (0.2625 vs 0.2620); DNA Δ**0.0002** (0.9769 vs 0.9767) — matches within stretch-LUT/float noise |
| **causality** (future-bit-flip) | **PASS** — both streams processed in full; flipping a bit *after* the checkpoint leaves the prefix cost bit-identical (no look-ahead) |
| determinism | **PASS** — identical output across runs |
| edge case (5-byte input) | handled, no panic |
| scale + gzip (text 2 MB) | **0.2358** bits/bit vs gzip 0.3607 (+0.125) |
| **DNA full E. coli genome** | auto-p=6 → **0.9564 bits/bit = 1.913 bits/base** (last-20%) vs gzip 1.984 |

**Findings:**
- **The representation is induced natively, at scale.** The engine discovers the unit (byte / codon)
  from a held-out scan and predicts unit-aligned — verified against the Python reference to ≤0.0005,
  causal (future-bit-flip), deterministic, and beating gzip on both text and genome.
- **Discovery ≈ hand-building, without the priors.** On the full E. coli genome the *induced* engine
  reaches **1.913 bits/base by discovering the codon (p=6) alone** — essentially the hand-built `dna.rs`
  specialist's **1.908**, but with **no codon/reverse-complement prior supplied**. The "learn what to
  compute / induce the representation" thesis, native and at genome scale.
- **A real, named scan subtlety (fixed).** The period scan must use a **small** window: on a large
  window long periods win by *context length* (more conditioning bits), not true periodicity — on the
  near-random genome that pushed the pick to p=10. An 80 k-bit scan (matching `real_scale.py`) restores
  the true unit (codon p=6); reported, not hidden.

**Significance + honest scope:** the bit-native core now *induces* its representation in the native
engine — `induced.rs` is general (one engine, period discovered per stream) and validated by a full
test battery. Honest scope: it is the unit-discovery + mixer core, **without** `dna.rs`'s
reverse-complement/match models, so on DNA it sits just shy of the tuned specialist (1.913 vs 1.908) and
on text just shy of `strong.rs`'s match/high-order machinery; folding those into the induced engine, and
running it at full enwik8 / whole-chromosome scale, is the remaining work. The thesis stands
empirically: **the right unit can be discovered, not assumed — natively, causally, at scale.**

---

## 57. Folding the match + reverse-complement models into the induced engine, at scale (`induced.rs`)

§56's remaining work was to fold `strong.rs`/`dna.rs`'s repeat models into the induced engine and run at
full scale. Done: a **forward match** whose granularity *follows the discovered unit* (byte symbols for
text p=8; 2-bit **bases** for DNA p=6), and — when the discovered unit is base-like (DNA mode) — a
**reverse-complement match** for inverted repeats, sharing the forward base table. Still causal; the
folded models are validated by the same future-bit-flip test.

| stream | engine | bits/bit | bits/base | reference |
|---|---|---|---|---|
| **E. coli — full** (4.64 M bases) | induced: discover p=6 → base-match + RC | 0.9547 / **0.9394** (whole/last-20%) | 1.909 / **1.879** | `dna.rs` specialist **1.908 / 1.876** |
| **text — full corpus_big** (11 MB) | induced: discover p=8 → byte-match | 0.2403 / **0.2382** | — | gzip 0.374 |

**Findings:**
- **Discovery now equals hand-building on a clean genome.** On the full E. coli genome the induced
  engine reaches **1.909 / 1.879 bits/base — matching the hand-built `dna.rs` specialist (1.908 / 1.876)**
  — by *discovering* the codon unit (p=6), routing to base-granular match + reverse-complement, with **no
  DNA prior supplied**. The match granularity auto-follows the discovered unit; RC engages only in DNA
  mode (it is inactive and harmless on text). Both folded models pass the future-bit-flip causality test.
- **Scales to 11 MB text**, beating gzip 0.374 → 0.24, still discovering the byte.
- **An honest limitation, found at scale (human chr21).** Unlike E. coli (~88 % coding → a clear codon
  period), human chr21 is **mostly non-coding** (codon signal only in ~2 % exons) and *starts* with a
  long low-complexity/telomere region — so the prefix period scan finds no robust base-like unit (it
  picks p=8/10 by context-length) and the engine runs in generic mode (≈1.7 bits/base) rather than DNA
  mode. A middle-window scan did not help (it broke E. coli) — the real issue is that **weak-periodicity
  streams defeat the simple scan**; robust modality/period detection (autocorrelation, an entropy-gated
  representative window, or a dedicated base-structure test) is the honest open problem for the discovery
  step. `dna.rs` (unit hand-given) still gets chr21 to 1.6255; matching that via *discovery* needs the
  better detector.

**Significance + honest scope:** the induced native engine now folds in the repeat models and, on a
clean genome, **discovery matches the hand-built specialist** — strong evidence for "induce the
representation, don't assume it." The named gap is **robust unit discovery on weak-periodicity / mostly
non-coding streams** (human DNA), where the simple prefix scan is not enough. No fake success: E. coli
and 11 MB text validate end-to-end, causally; chr21's discovery limitation is reported, not hidden.

_(Detector update: the scan now **skips low-complexity windows** — a telomere / N-run scores near 0,
trivially compressible — and uses the first representative window, so a non-representative prefix no
longer misleads it. This preserves text=8 / E. coli=6, but **human chr21 still resolves to p=10**: its
codon signal is genuinely too weak (mostly non-coding) for a prediction scan to find. Tried prefix,
middle, multi-window-vote, and representative-window scans; none robustly routes human DNA to base-mode.
The honest conclusion stands — robust unit discovery on weak-periodicity streams (autocorrelation, a
base-structure test) is the open problem, and `dna.rs` with the unit hand-given still reaches 1.6255.)_

---

## 58. The loop closes — English in, bit-native generation, English out (`talk.py`)

`BIT_NATIVE_INTELLIGENCE_FRAMING.md` proposes a **bit-native intelligence core with dumb modality
adapters**: language is an optional I/O codec, prediction is the substrate. This realises that loop
end-to-end and makes it *visible*: type English → a **dumb adapter** (UTF-8 bytes ↔ bits, no semantics)
→ the **bit-native core** (a byte-aware next-bit predictor, the same family that compresses enwik8),
trained on real English, **generates a response by autoregressively sampling the next bit** from its own
prediction, one bit at a time → the adapter decodes the bits back to English → you read it.

```
[you]      "It is a truth universally acknowledged that "
[adapter]  English -> 44 bytes -> 352 bits
[core]     sampling 220 next-bytes (1760 bits), one bit at a time ...
[adapter]  bits -> bytes -> English
[core says] It is a truth universally acknowledged that | there was all this morning at her mother,
            ... I can remember ... And even a very day after a short survey" her again, and which she
            countenance, that he should not ...
```

**Findings:**
- **The architecture's I/O loop is real, not a diagram.** Trained on *Pride & Prejudice* (real English),
  the core produces recognisable English continuations — real words, spacing, punctuation, the corpus's
  voice — purely by predicting and sampling bits, with language living only in the dumb adapter. The same
  next-bit prediction that *compresses* a stream *generates* one (compression = prediction = generation).
- **Honest scope (stated plainly).** This is a **byte-level statistical** language model (a high-order
  n-gram / small char-LM), **not** an instruction-tuned LLM. It does **not reason or answer** — it
  *continues* the prompt in the corpus's style, with occasional non-text bytes at low temperature. The
  value is architectural: it demonstrates *intelligence-as-prediction below language*, the framing's
  central thesis, as a runnable loop — not a capability claim.
- **It ties the whole project together.** The bit-native predictor — toy benches (§1–42), Path B's
  induced computations (§43–57), the native engines — is, at bottom, a next-bit model; here that model
  *speaks*, through a dumb adapter, exactly as the framing proposed. Scaling the core (the strong /
  induced engines) and training on dialogue would make the continuation more answer-like; the loop itself
  is now closed and interactive (`python talk.py "your prompt"`).

---

## 59. From continuation to ANSWERING — strong core + Q/A + a real dataset (`chat.py`, `make_chat_data.py`)

§58 *continued* a prompt; this *answers* it, by two additions that change the data and the wrapper, not
the engine. **(#1) A stronger generative core:** online logistic context mixing (byte-aware orders, the
enwik8 family) **plus a byte match model** — when the recent context matches a span seen in training, it
*recalls and copies* the continuation (attention-like copy, in bits). **(#2) A Q/A wrapper:** your input
is formatted `Q: …\nA:` so the model falls into answer-shape, and generation **stops at the next `Q:`**
for a bounded reply. **(dataset) `make_chat_data.py`** builds a real, correct corpus — exact arithmetic +
curated true facts (capitals, science, opposites, counting), 4 643 Q/A pairs — the "serious data" lever.

```
Q: What is the capital of France?  -> A: Paris      Q: What is 7 + 8?     -> A: 15
Q: How many legs does a spider have? -> A: 8         Q: What is 25 + 17?  -> A: 42
Q: What is the opposite of hot?    -> A: cold        Q: symbol for water? -> A: H2O
```

**Findings:**
- **It answers — correctly — on what it was taught.** Capitals, arithmetic in range, opposites,
  counting, simple science all come back right, by recall: the long match locks onto the question's tail
  and copies the answer that followed it in training; the orders supply format and stop. Same next-bit
  engine as the compressor, run in sample mode — recall is just the match model (compression =
  prediction = generation = recall).
- **A real bug, found and fixed (kept honest).** A *single* learned mixer weight for the match went
  **negative** — on short Q/A data many 16-byte contexts precede different bytes, so the mixer learned to
  distrust the match. The fix is to trust the match only once it is a **long** span (gate on match
  length) and steer toward the recalled bit then, rather than mixing it by one global weight.
- **Honest scope, stated plainly.** This is **recall + format**, not reasoning. Trained questions →
  correct; **novel** questions → *answer-shaped but wrong/made-up* (`"capital of Mars?" → "10"`,
  `"123 + 456?" → "30"`) — no generalisation, no arithmetic it didn't memorise, no facts it never saw.
  The value is the architecture: a bit-native predictor, through a dumb adapter, *answers* — and gets
  better strictly by (a) **more/cleaner data** (`make_chat_data.py` is the lever) and (b) a **bigger
  core** (the strong/induced engines). It does not pretend to know what it was not shown.

**Significance:** the framing-doc loop is now not just "speaks" (§58) but "**answers**", with real recall
over a real dataset — `python chat.py "your question"`. Everything stays bit-level: prediction is the
substrate, language is the dumb adapter, recall is the match model. The path to "actually smart" is the
honest one — scale the core and the data — not a change of mechanism.

---

## 60. Recall vs GENERATION — can it produce correct *new* data? (addition, held-out) (`add.py`)

§59 was honest that chat is **recall**: delete a fact and it cannot answer; it memorised the times-table
and fails `271+314`. The sharp question: can the bit-native core produce **correct outputs for inputs it
never saw** — i.e. learn a *computation* that generalises, not a table? Addition is the microscope, with
**held-out accuracy** as the only metric (the discipline §59's chat side lacked).

Representation is the lever: addition is non-local as decimal text, but in **binary, LSB-first** it is a
1-bit-state transducer — `s_i = f_out(a_i,b_i,carry)`, `carry' = f_upd(a_i,b_i,carry)`. The carry is a
**hidden** recurrent state; it is **never supplied**. We *induce* both 3-input boolean functions by
searching the 256×256 transducer space for the one that explains the training sums — Path B's "learn the
recurrent computation" (cf. §51), applied to a generative task.

| approach | train acc | **held-out acc (3000 unseen sums)** |
|---|---|---|
| **memoriser** (key on `(a,b)`, chat-style recall) | 100 % | **0 %** |
| **induced computation** (1-bit-state transducer) | 100 % | **100 %** |

**Findings:**
- **It generates correct new data — by induced computation.** From **60** example sums the search finds
  **exactly one** transducer reproducing them: `out=XOR3` (150), `carry=MAJ3` (232) — *the full adder*.
  The hidden carry was **induced, not given**. It then computes sums it never saw (`3131+2996=6127`, …)
  at **100 % held-out**, while the memoriser is at **0 %**. That is the honest, measured **yes** to
  "can it generate new data" — wherever it induces the *rule*, not the *table*.
- **It is the representation, again.** The same numbers as decimal text → memorised; as binary
  bit-columns + a 1-bit state → the rule is inducible and generalises perfectly. "The lever is the
  representation" (the project's spine) is what separates recall from generation.
- **Knowledge stays separable — and that is fine.** The core learned a *function* (the adder), not
  stored answers; facts can live in an external addressable store (the match model is exactly that).
  A coherent **non-LLM** shape: *induce computations + retrieve knowledge + generate*, rather than
  memorise everything in weights.

**Significance + honest scope:** this closes the loop on the project's central claim with a number —
the bit-native core **generates correct novel data when it learns a generalising computation**
(held-out 100 %), and only recalls when it memorises (held-out 0 %). The honest ceiling: addition is a
*tiny* rule (1-bit state) induced by *brute* search over a *small* space; complex computations hit the
Path B search wall (§45), and discovering the right representation automatically is still open. But the
principle — recall ≠ generation, and induced computation *is* generation — is now proven end-to-end, and
it points the way: a growing **library of induced computations** composed with separable knowledge.

---

## 61. A reliable, verifiable STRUCTURED RESPONSE — and where the wall really is (`tool.py`)

The honest target is not free prose (a high, scale-bound wall) but a **proper structured response you can
rely on** — the kind that, in a future view, underlies tool-use. Stripped of that framing: can the core
turn a request into a **valid tool call that, executed, is correct**, for requests it never saw? Three
guarantees, all decidable: **valid by construction** (the response is emitted under a grammar
`op(a, b)`), **correct by induced computation** (executed by the §60 bit-native transducer — add and a
subtractor, each induced from 60 disjoint examples, so they *compute*), and **verified** (validity and
correctness measured on held-out requests). Division of labour as the framing doc prescribes: the dumb
adapter tokenises, the core **routes** the operation and **computes**, the grammar guarantees a
well-formed response.

| over 2 000 held-out requests | result |
|---|---|
| structured response **valid** (parses the grammar) | **100 %** |
| executed answer **correct** | **100 %** |
| chat-style **memoriser** (request → answer recall) | **0 %** |

**Findings:**
- **A response you can rely on is reachable — in the structured/verifiable regime.** The core emits a
  valid call and the correct answer for numbers it never saw (memoriser: 0 %), because the computation is
  *induced*, not stored, and the form is *grammar-guaranteed*. This is the trustworthy shape: not "hope
  the prose is right" but "the response cannot be malformed and the value is checkable."
- **The wall is words, not compute — shown, not asserted.** Before a fix, valid was 100 % but correct
  was **82 %**, and *every* error was one phrasing: *"subtract X from Y"* (= Y−X), which the parser got
  backwards. The computation was flawless; the failure was **language understanding**. Fixing it meant
  encoding a *words fact* (`"from" ⇒ swap the operands`) — a compute fix would not have helped. The
  residual difficulty lives exactly where intuition says: in mapping varied language to intent.
- **Honest scope.** The generalisation that *works* (held-out numbers, 100 %) comes from the induced
  computation + grammar + copy of the request's fields. The part that is **bounded** is the
  language-understanding (keyword routing + known phrasings, hand-given here); widening it to open
  language is the words-wall — solvable only by *learning* many phrasings (scale) or inducing the
  parse, which is the open frontier.

**Significance:** "a proper response you can rely on" is not blocked by computation — that generalises
cheaply (§60) and is verifiable. It is gated by **language understanding**, precisely. So the productive
path is the structured/verifiable regime — valid by grammar, correct by induced rule, knowledge
separable — where the bit-native core can be *reliable by construction*, and to push it, attack the
bounded-language-understanding frontier directly rather than chasing free-prose fluency.

---

## 62. Engine / language / knowledge — provably separable, held-out (`separation.py`, `separation_data.py`)

A sharper question than "predict the next bit": are **computation**, the **language** a query is posed
in, and the **knowledge** (facts read from text) *separable, independently-varying* parts of the core —
or are they entangled? The bet is a three-way separation: an **engine** (computation), a **language**
(the surface), and a **knowledge** store (facts), that vary independently. `separation.py` builds this
as a falsifiable, staged experiment over the §59 byte-native core and the §60 induced adder, with the
controls an adversarial panel demanded (designed by one research fan-out, audited by a separate
adversarial fan-out; both in the session log).

The three artifacts: **(E)** the *engine* = the induced full-adder transducer `(out=XOR3, upd=MAJ3)`
plus `chat.Core`'s fixed predictor code and frozen hyperparameters — a **corpus-independent** object;
**(L)** the *language* = the grammar/Q&A shape the store acquires from a **fact-free** English corpus;
**(K)** the *knowledge* = the append-only byte store (`corpus_bytes`/`mtab`/`tables`), filled **only**
by reading. We fingerprint E and K at every stage S0→S4 (arrival → learn English → read an English book
→ meet a cipher-only fact → learn to read the cipher).

| claim | result (held-out / staged) |
|---|---|
| **C1** math is *induced*, not recalled | adder `(150,232)` forced by 60 disjoint sums; **100 %** on 3 000 unseen pairs; memoriser **0 %** |
| **C2** holds no facts it has not read | with English loaded, every real-world-fact query **abstains** (0/5 correct, **5/5 abstain**) |
| **C3** language ≠ knowledge | English readability **8.00 → 0.81 bits/byte** (learned) yet **no** fact becomes answerable |
| **C4** engine/store **separation** | engine fingerprint identical S0..S4 **and sabotage-sensitive**; store fingerprint changes every stage, corpus **prefix-monotonic** |
| **C5** reading supplies knowledge, gated by language | novel fact read → **commits** (pre-book abstains); cipher-only fact **abstains in English, commits in its own language**; cipher **learned** (4.81 → 0.89 b/B) while never-taught cipher C′ stays **5.96** |

**Findings:**
- **Math is computed; facts are read.** The adder generalises 100 % on numbers it never saw (a *computed*
  rule, memoriser 0 %), while the *same* core abstains on an unread fact (e.g. the capital of France)
  until it reads it — computation and fact-recall are distinct subsystems, not one entangled blob.
- **The engine invariant is a *separation guarantee*, verified by sabotage — not a tautology.** The
  fingerprint is, by construction, a function of the corpus-independent engine only, so its constancy
  across stages is *exactly the claim* that language and facts land in the **store**, never the engine.
  We keep it from being vacuous by showing it **moves** when the math is broken (`upd_fn=233 → different
  hash`), the predictor is patched, or the gate changes — so it genuinely *would* change if who-it-is
  changed. Meanwhile the store fingerprint changes at every stage and the corpus is append-monotonic.
- **Honest abstention, earned not declared.** One fixed rule for *every* query — commit iff the question
  is found **verbatim** in read text (`prime_mlen ≥ MATCH_GATE`) **and** confidence ≥ τ — with no
  per-fact branch. The binding term is verbatim-question recall: a *fluent confabulation* (conf 0.98 but
  `mlen 0`) **abstains**; the *same* query commits only after the book is read. Calibration, not a guard.
- **Knowledge is gated by the language it is written in.** A fact stored only in cipher C abstains to an
  English question but commits when asked **in cipher C** (the model emits cipher bytes that are a
  verbatim substring of what it read; we decode **only to score**, never to answer). And "learning a
  language" is measured honestly as **held-out readability**: reading cipher-C text drops its bits/byte
  4.81 → 0.89, while a never-taught cipher C′ stays at 5.96 — learned to read, not trivially decoded.
- **The boundaries, stated and shown (not hidden).** Recall is **span-level copy**, labelled by
  longest-common-substring: a < 16-byte **paraphrase abstains** (no semantic comprehension). The engine
  does **not translate**: an English question about the cipher fact **stays gated even after** the model
  learns to read that cipher. And the learned "English" is **vocabulary/grammar-bound** — a
  novel-vocabulary control (same grammar, unseen words) stays high (≈ 3.8 b/B vs 0.81), so the claim is
  "learned *this* language," not "reads arbitrary English." Corpora are audited **fact-free** (no answer
  token, no 16-byte probe span) and the demo does **zero file I/O** (it never touches `data/chat.txt`,
  which contains France→Paris and would be a fatal leak).

**What it proves, and does not.** `separation.py` demonstrates a clean three-way separation between a
computation engine, a language, and a knowledge store on a byte-level model: math is induced computation
(100 % held-out, memoriser 0 %); the engine is corpus-independent by construction *and* sabotage-
sensitive, so language and facts provably live in the separable, append-monotonic store; facts are absent
until **read** (audited fact-free, novel entity abstains pre-book / commits post-book, one book leaks no
other fact); retrieval and abstention run through **one fixed, fact-agnostic rule** with no decoder in
the answer path; and "learning a language" is held-out readability (taught cipher C drops, never-taught
C′ does not). It does **not** prove comprehension, grounding, or understanding — fact recall is verbatim
span-copy (a paraphrase abstains), the model does not translate (a cipher fact asked in the other
language stays gated), and the learned language is a small closed vocabulary/grammar. Stated plainly:
**the core computes math that generalises, holds no fact it has not read, and is the same engine
byte-for-byte before and after learning a language — computation, language, and knowledge are separable
axes**, measurable held-out, which is what makes them improvable one at a time.

**Significance:** the §61 result said the wall is *words*. §62 maps that wall precisely by **separating
the axes**: computation generalises for free; reliable answering is recall traceable to what was read,
gated by language; and "understanding" (paraphrase, translation, open vocabulary) is the residual,
explicitly *not* yet crossed. The productive next step is the same one §61 pointed at — widen the bounded
language axis (learn the question→intent map, paraphrase-robustly) — now with a harness that can measure,
honestly and held-out, exactly how much of that axis any change actually buys.

---

## 63. Improving the strong compressor — richer models + tuning, measured (`blmrs/src/bin/strong.rs`)

Past the research phase, the first engineering target is the headline itself: **lower bits/bit**. The
strong core was already lpaq1-class (§ headline). Working in an A/B fork (`strong2`, since promoted),
each change was measured on a held-out real-text proxy (`data/corpus_big.txt`) and kept only if it
helped — a clean ablation, not a guess.

**Changes (each measured at 1 MB, whole-stream bits/bit):**

| step | change | bits/bit | Δ |
|---|---|---|---|
| baseline | byte-orders 0..6, hi {8,12,16,24,32}, 1 sparse, word, 2 match, 1 sel mixer, 4 SSE | 0.221636 | — |
| +order-7 | fill the 6→8 gap | 0.220987 | −0.0006 |
| +mixer-2 | a 2nd context-selected mixer (order-2 selector) + 4-weight final combiner | 0.219541 | −0.0014 |
| +SSE | two more APM stages (word-hash, order-3 contexts) | 0.219314 | −0.0002 |
| +tune | DELTA 0.18→0.08, mixer LRs 0.0013→0.0010 (env-overridable) | 0.218859 | −0.0005 |
| +sparse | 3 sparse models (non-adjacent byte pairs) instead of 1 | 0.218614 | −0.0002 |
| +prev-word | predict a word's bits from the **previous word** (text bigrams) — biggest single win | 0.216905 | −0.0017 |
| +ICM | 5 **indirect** models (orders 2–6): bit-history byte → adaptive StateMap (nonstationary) | 0.216588 | −0.0003¹ |

¹ ICM looks tiny at 1 MB but **scales hard**: −0.0026 at 3 MB and −0.0019 at 11 MB (whole) — it earns its
keep only once contexts accumulate bit-history (see the validation rows). On code it is the dominant win.

**Validation (the cumulative engine — all of the above incl. ICM — vs the original baseline, held-out):**

| corpus | baseline | improved | improvement |
|---|---|---|---|
| corpus_big 1 MB | 0.221636 | 0.216588 | −2.3 % |
| corpus_big 3 MB | 0.218053 | 0.211762 | −2.9 % |
| **corpus_big 11 MB** (whole) | **0.224637** | **0.217011** | **−3.4 %** |
| corpus_big 11 MB (last-20 %) | 0.223195 | 0.214899 | −3.7 % |
| code 0.8 MB | 0.159457 | 0.150989 | **−5.3 %** |
| b100 1 MB | 0.241401 | 0.236463 | −2.0 % |

**Findings:**
- **A real, generalising gain that grows with data.** −3.4 % bits/bit on the 11 MB text proxy (−3.7 % on
  the warmed-up last-20 %), and it **generalises** — text *and* code, 2.0–5.3 % — so it is no
  corpus-big artefact. The gain **grows with scale** (1 MB −2.3 % → 11 MB −3.4 %): richer models earn
  their keep as the stream lengthens — the opposite of overfitting.
- **Two kinds of win.** (a) *Direct structure*: the **previous-word model** (−0.0017 at 1 MB) and the
  **second (order-2) mixer** — text-bigram structure written into the address, the project mantra,
  bankable as bits. (b) *Indirect / nonstationary*: the **ICM/StateMap** models — a context's bit-HISTORY
  (recency, not just totals) indexes an adaptive map — which is the lever that **scales** (−0.15 % at
  1 MB but **−1.2 % at 3 MB**, and **−5.3 % cumulative on code**, where repeated structure makes the
  history maximally informative).
- **Honest scope.** (1) **Speed cost:** ~2–3× slower (second full mixer + ICMs + extra models),
  ~0.15 Mbit/s pure-scalar — a roadmap-item-3 (throughput) concern, deferred deliberately. (2) **enwik8
  not re-run:** the 0.209 ladder figure was the *prior* engine; the improved engine should be at least as
  good but that was **not measured here** (enwik8 isn't local), so the ladder stands as a conservative
  floor, not a new claim. (3) Every step kept only on a measured win; `DELTA/ALR/ALRF` are env-overridable
  for one-build sweeps.

**Significance:** the bit-native compressor is improvable by ordinary, honest CM engineering — measure,
keep wins, ablate. Two levers paid off: representation (text-structure models) and **indirect bit-history
StateMaps** (the ICM/ISSE family that separates lpaq1 from paq8), the latter scaling with data. The ICM
here is the *flat-input* form (bit-history → StateMap → mixer input); the remaining headroom is the
*chained* ISSE form (each order refining the previous prediction) plus a richer state machine and ICMs on
the word/high-order contexts — the next push on this axis.

---

## 64. A first probe at the words wall — a LEARNED question→intent router (`intent.py`)

Roadmap item 2 is *capability*, and §61/§62 named its wall: **words** — mapping a varied phrasing to the
right intent. `tool.py` routed with **hand-coded** keywords (`"plus" ⇒ add`). §64 replaces that with a
**learned** map (online multiclass logistic — the compressor's own stretch/squash unit — over binary
features: word tokens, operator symbols, numbers masked to `NUM`, char 3-grams) and measures, honestly,
*how far it generalises* across three template-disjoint tiers, over **8 seeds** (mean, min–max).

> **Honesty note.** A first cut of this section over-claimed and was corrected after an adversarial
> red-team (the project's standing practice): its HARD tier *leaked* verbatim training words, and its
> hand-coded baseline was a *strawman* (whole-word match). Both are fixed below — HARD now contains only
> word forms that are **not** whole training words (asserted in code), and a **fair stem-substring**
> hand-coder is reported alongside. The numbers and the claim are the corrected ones.

| router | train | EASY¹ | HARD² | NOVEL³ |
|---|---|---|---|---|
| hand-coded, whole-word (`tool.py`) | 77.8 | 86.7 | 0.0 | 5.6 |
| hand-coded, **fair stem-substring** | 94.4 | 93.3 | **91.7** | 0.0 |
| learned (words only) | 100 | 100 | 44.8 | 42.4 |
| **learned (words + char n-grams)** | **100** | **100** | **97.9** | 38.2 |
| learned (words + char, stop-words stripped) | 100 | 100 | 96.9 | 42.4 |

(mean % over 8 seeds.)  ¹ EASY = new sentence *structure*, same key words.  ² HARD = unseen word *forms*
(`added`, `subtracting`, `multiplied`) — share stem *substrings*, **no** whole training word (verified).
³ NOVEL = true *synonyms* (`combine`, `deduct`, `scale`) — share no discriminative word with training.

**Findings (corrected, multi-seed):**
- **Routing is learned, and generalises across sentence structure.** EASY = **100 %** on every seed —
  no hand-tuned keyword list, robust to new phrasings built from known words.
- **Char n-grams give a *real* lift on unseen word forms.** On the de-leaked HARD tier, words-only scores
  **44.8 %** but words+char-grams **97.9 %** — sub-word features genuinely bridge `multiplied`→`mul`,
  `subtracting`→`sub`. This is the one solid generalisation result beyond structure.
- **But this is *not* a win over fair hand-coding.** A reasonable **stem-substring** hand-coder also
  solves HARD (**91.7 %**). So the learned router's value is *"no keyword hand-tuning"* (it discovered
  the stems from data), **not** a capability hand-coding lacks. The earlier "generalises where hand-coded
  cannot" claim was false and is retracted.
- **The synonym wall is NOT climbed.** True synonyms (NOVEL) sit at **≈38 %** — at the **33 % three-class
  chance floor** (per-seed range 33–50 %), and stripping function words leaves it at chance (**42 %**), so
  the few above-floor hits were shared *function* words, not meaning. Surface features cannot cross a pure
  semantic gap.

**Significance:** the words wall is a **gradient**, now honestly mapped: sentence *structure* (learned,
solved), *morphology* via sub-word features (learned, solved — though fair hand-coding solves it too),
and *semantics*/synonyms (**open** — at chance). The real next step is the only one that crosses the last
gap: give words **learned distributional meaning** (§62's "reading" — representations the predictor
already forms from a corpus) and route over those. That keeps item 2 tied to the core predictor rather
than bolting on an LLM — and §64's honest value is the *measurement* that says exactly where the wall is.

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

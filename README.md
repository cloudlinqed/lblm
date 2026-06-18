# LBLM — A Bit-Native Predictive Machine

LBLM is a research exploration of a **bit-native predictive machine**: a model that predicts
the **next bit** (`0`/`1`) from a history of bits, autoregressively. There is no text layer, no
tokenizer, no vocabulary, and no embedding table — the operating material is the raw binary
stream. The vocabulary size is **2**.

This repository is an honest research log. It contains a working reference implementation, a
controlled benchmark, two design documents, and the empirical results — including the claims
that were **overturned** when stress-tested. Nothing here is a finished product; it is a
deliberately documented build → train → test → fine-tune loop.

> License: Apache-2.0.

---

## The idea, and the bet

An LLM predicts the next *token* from a ~50k-way softmax. LBLM predicts the next *bit* — a
single Bernoulli decision. The trade-off:

- One ~50k-vocab token carries ≈ `log2(50000) ≈ 15.6` bits, so reproducing the same content
  takes **~16× more steps**. (more iterations)
- But each step has **no softmax, no vocabulary, no embedding table** — just a threshold.
  (cheap per iteration)

Going binary does not remove difficulty; it **relocates** it from *vocabulary size* to
**dependency range measured in steps**. With 1 bit per step, structure an LLM grabs in a few
steps is spread across ~16× more of LBLM's. So the central problem is long-range memory in a
single left-to-right bit chain — and most of this project is about escaping it.

## Architecture — learned binary addresses

The core is a content-addressable memory operating entirely on bits.

- **Unit** = `(address ∈ {0,1}^A, value ∈ {0,1}, strength w)`.
- **Query** = the current `R`-bit register, optionally extended by a recurrent state.
- **Retrieval** = weighted Hamming proximity (`XOR + popcount`), softened by a kernel
  `exp(-β · weighted_hamming)` — i.e. learned **locality-sensitive hashing**.
- **Output** = signed, strength-weighted vote → threshold → next bit.
- **Learning** = a gradient-free **binary Self-Organizing Map**: reinforce + pull the nearest
  correct unit toward the query, weaken + push wrong-voters away, allocate a unit when no
  correct neighbour exists, anneal, merge duplicates. (`frozen` = immobile templates;
  `mobile` = addresses self-organize.)

This single substrate unifies several ideas the project explored from first principles — and
keeps re-deriving known mechanisms:

| Idea | What it becomes here |
|---|---|
| Blockchain-style chaining with a *learned* "hash" | a recurrent binary summary (semantic hashing) |
| Multi-directional relations | query-conditional weighting (attention) |
| A matrix with coordinates | content-addressable memory (Hopfield) |
| Carry history in fixed width | recurrent address `addr_t = H(addr_{t-1}, bit)` |

**Recurrent address modes** (`--addr`): `register` (memoryless), `shift` (history window),
`fold` (fixed-width xor-compression). **Readout weighting** (`--weights`): `uniform`, `mi`
(marginal mutual information), `contrastive` (separates collisions), `conditional`
(query-dependent / attention-style). Generation is **warm-started** so the recurrent state is
primed over the prefix rather than starting cold.

## What we have learned (verified)

Every result below was reproduced and adversarially stress-tested by independent agents; some
of the original conclusions were **refuted** and corrected.

1. **It does the toy task by memorisation, not learning.** On the seed streams it reaches 100%
   and reproduces both continuations, but units sit verbatim on training registers and
   leave-one-window-out generalisation is only ~0.43–0.79. It is a Hamming-kNN lookup table
   with smoothing, not a grammar learner.
2. **Collisions are an information-theoretic wall.** When the register is not Markov-sufficient
   (e.g. `R=4` on the seed data), *exhaustively* 0 of 4096 deterministic tables can fit — no
   learning rule or probabilistic output helps; only more sufficient state does.
3. **Recurrent memory works, with an exact horizon.** On the long-range benchmark the usable
   memory horizon is exactly **`L = R + h − 4`**, verified as a hard step function across a
   16-cell grid.
4. **Genuine body-invariant rule transfer exists (but is bounded).** A rule-scramble control
   shows the machine transfers `answer = f(type)` to *unseen* bodies (intact ≈0.72–0.78 vs
   scrambled ≈0.44–0.53) — real generalisation, but only inside the memory band and capped
   ~0.78.
5. **The ~0.78 ceiling is not a feature-weighting problem.** Static MI weighting rewards
   predictive-but-class-invariant bits (e.g. a boundary) and fails; static contrastive
   weighting finds the *right* discriminative bit yet still doesn't help; hard-isolating that
   bit collapses to chance — because the task needs **different bits at different generation
   steps**. The need is genuinely **query-conditional**.

**Current frontier:** query-conditional weighting (attention) and/or a learned recurrent
encoder. Every cheaper alternative (wider window, hand-coded compression, marginal weighting,
contrastive weighting, hard isolation) has been built and ruled out with evidence.

## Repository layout

| File | What it is |
|---|---|
| [`blm.py`](blm.py) | The machine: learned-address memory, recurrent addresses, SOM learning, readout weighting |
| [`bench.py`](bench.py) | Long-range recall benchmark + memory-curve experiment (a controlled capability probe) |
| [`bit_native_predictive_machine.md`](bit_native_predictive_machine.md) | v1 design — the original bit-native machine concept |
| [`learned_binary_address_machine.md`](learned_binary_address_machine.md) | v2 design + full results, lessons, and verification verdicts |
| [`sweep.jsonl`](sweep.jsonl) | Recorded metrics from the parameter sweep |

## Running it

Requires only Python 3 (standard library; no dependencies).

```bash
# Train on the seed streams and print results (memorisation baseline)
python blm.py --R 8 --weights uniform --lowo

# Recurrent address vs memoryless register
python blm.py --addr shift --R 6 --hist 4 --weights uniform

# Long-range recall benchmark — the memory curve (in-sample and held-out)
python bench.py --R 6 --K 8
python bench.py --R 6 --K 12 --holdout
```

Key flags: `--mode {frozen,mobile}`, `--addr {register,shift,fold}`, `--hist h`,
`--weights {uniform,mi,contrastive,conditional}`, `--R`, `--lowo`.

## Methodology

Claims in this repo are not taken on faith. Each cycle ends with an **adversarial verification
pass** — independent agents that audit the code, recompute baselines, and try to *break* the
headline claim (including searching for counter-examples and optimising against it). Several
results in the history were overturned this way and corrected in the design doc. The intent is
that a reader can reproduce and falsify every number.

---

*An exploratory research project. Contributions, replications, and refutations welcome.*

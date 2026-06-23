# LBLM — A Bit-Native Predictive Machine

LBLM predicts the **next bit** (`0`/`1`) from a history of bits, autoregressively. No text layer,
no tokenizer, no vocabulary, no embedding table — the operating material is the raw binary stream.
The vocabulary size is **2**.

This repository is an honest research log: a working reference implementation, controlled benchmarks,
two living design documents, and the empirical results — **including the claims that were overturned
when stress-tested**. It is a deliberately documented build → train → test → fine-tune loop, from a
toy memory probe all the way to a real compressor.

> License: Apache-2.0.

---

## Headline result — full enwik8

Scaled up, the bit-native predictor is a genuine compressor. On the standard **enwik8** benchmark
(first 100 MB of Wikipedia), the native strong model reaches:

> **0.209 bits/bit → enwik8 compresses to ~20.9 MB** (model cross-entropy / ideal coded size).

Where that lands on the enwik8 ladder (approximate compressed sizes):

| compressor | enwik8 | bits/bit |
|---|---|---|
| gzip | ~36 MB | ~0.36 |
| bzip2 | ~29 MB | ~0.29 |
| PPMd | ~24 MB | ~0.24 |
| **LBLM (`blmrs-strong`)** | **~20.9 MB** | **0.209** |
| lpaq1 | ~20 MB | ~0.20 |
| paq8 | ~16 MB | ~0.16 |
| cmix (SOTA) | ~15 MB | ~0.15 |

A from-scratch bit-native predictor **beats gzip, bzip2, and PPMd** and reaches **lpaq1 territory** — and
quality kept improving with data right to the full file (0.224 → 0.219 → 0.209 at 10 → 30 → 100 MB).
Not SOTA (that needs far more models + GB-scale tuned memory + cache-aware engineering), but a real,
defensible result for a predictor built from first principles. Compression = prediction = learning:
this is the bit-native analogue of an LLM's perplexity, on real data.

---

## The idea, and the bet

An LLM predicts the next *token* from a ~50k-way softmax. LBLM predicts the next *bit* — one Bernoulli
decision. The trade-off: reproducing the same content takes ~16× more steps, but each step has no
softmax, no vocabulary, no embedding — just a probability for one bit. Going binary doesn't remove
difficulty; it **relocates** it from *vocabulary size* to *dependency range in steps*. Most of this
project is about handling that, and the central, repeated lesson is:

> **The lever is the representation — *what computation is written into the address* — not the readout.**

## What it became — the arc

Every result was reproduced and adversarially stress-tested by independent agents; several headline
claims were **refuted and corrected** (that is the point).

1. **Memory, with an exact horizon.** A content-addressable bit memory recalls across gaps with a hard
   horizon `L = R + h − 4`; a learned **gated-latch** removes the horizon and holds a signal to body
   length 1000. Early on we proved it was a Hamming-kNN lookup table, *not* a grammar learner.
2. **Compute, don't hold.** For aggregates (parity, popcount mod m), *computing* a running feature
   into the address generalises (1.00 held-out) while *holding* the raw inputs fails — the address
   scales with the number of aggregate values, not the input count.
3. **Learn the computation.** A meta-learner *recovers the right recurrent computation from data* —
   the latch for recall, running-XOR for parity, the mod-m counter for mod-m — and *composes* them;
   it works under labels, under immediate reward, and under **delayed reward** (a bit-native MDP with
   TD learning). The machine also acts on its own **confidence** (commit/abstain) and detects
   distribution shift on a real stream.
4. **Real data, beats gzip.** As a next-bit predictor on real text/code the core beats gzip, and it
   **discovers its own representation** — finding the byte period (p=8) from a period scan and
   rebuilding the byte-aware structure from scratch.
5. **Scale.** Pure-Python → **PyPy** (~3.5×) → **lossless integer keys** (~12× combined, bit-identical)
   → bounded-RAM hashing → a **Rust core** (`blmrs`). On homogeneous data, bits/bit drops monotonically
   with more data.
6. **The native strong model → the headline.** Porting the strong model to Rust gives bounded memory
   with near-exact quality (what Python couldn't), yielding the enwik8 result above.

**Honest corrections along the way** (kept in the design doc): "the readout is the bottleneck" → it was
an *incomplete address*; "the plateau is a model ceiling" → it was *corpus composition*; "Rust is the
100× lever" → at scale the workload is *memory-latency-bound*, so native is ~2×, and Rust's real value
is bounded-memory-without-quality-loss.

## Path B — learning *what to compute* (induction, not compression)

A second line asks a different question: not "predict the next bit" but **induce the computation** that
solves a task, growing the toolbox from its own failures. Starting from a hand-made primitive family,
the system climbs a ladder of program-synthesis capability — each rung **crossed by discovery within a
generative space, validated full-domain-exact + cross-seed + scramble-clean, then exposing the next
wall**:

- **induce → invent → recurse → library → refactor** (design doc §43–47): composes computations and
  invents missing primitives, but **stalls at `0110`-parity** — diagnosed as boundary contamination +
  missing *parameterised* abstraction.
- **the wall cracked** (§50, `wake_lgg.py`): a **boundary-aware detector** (`cnt2(detector_P) ≡
  P-parity`, an exact identity) plus an **anti-unified `detect(P)` template** solves `0110` and
  **generalises to unseen patterns** — verified five independent ways, leakage-free.
- **the ladder** (§51–53): the same `WAKE → SLEEP → BIND` mechanism then generalises across families
  (gapped patterns, thresholds), **invents the missing aggregation** (count-equality `#0==#1`), and
  **invents a running counter** (Dyck-1 balanced parentheses) — each new wall found, named, and crossed;
  the next is non-local / stack-structured computation.

This is a **self-honest, example-driven, full-domain-verified mechanism for inducing and reusing
bit-native computations** that climbs a real ladder of power (detectors → counts → comparisons → state)
— *not* a claim of general intelligence. The standing caveat at every rung: the generative space is
still hand-provided; growing it autonomously is the open problem.

**On real data** (§54): the same induction, unified with Path A's mixer (*Path B induces the
representation, Path A predicts from it*), is **a real, scalable model that beats gzip on real English
text and real E. coli DNA**. It rediscovers the **byte** (period 8) on text and the **codon** (period 6)
on DNA from raw bits, scales monotonically (to ~0.22 bits/bit on text at 4 MB — the lpaq1/enwik8-headline
band; ~1.91 bits/base on DNA), and — most clearly on DNA, where the byte assumption is wrong — its
induced primitives (counters, reverse-complement palindrome detectors) measurably lower bits/bit.

## Architecture

The core is a content-addressable memory on bits — `unit = (address, value, strength)`, retrieval by
weighted Hamming proximity softened by a kernel, output by a signed vote — which, scaled up for real
data, becomes **online logistic context mixing**: several byte-aware context models each vote a
probability for the next bit, a small online-trained logistic unit mixes them in the logit domain, an
SSE/APM stage recalibrates, and the result codes the bit. The strong model adds hashed high-order
contexts, a byte-match model, a two-layer context-selected mixer, non-stationary count decay, and
RMSProp. See [`ARCHITECTURE.md`](ARCHITECTURE.md), [`FLOW.md`](FLOW.md), and
[`BIT_NATIVE_INTELLIGENCE_FRAMING.md`](BIT_NATIVE_INTELLIGENCE_FRAMING.md).

## Repository layout

| File | What it is |
|---|---|
| [`learned_binary_address_machine.md`](learned_binary_address_machine.md) | The living design doc — every cycle, result, and verification verdict (54 sections) |
| [`BIT_NATIVE_INTELLIGENCE_FRAMING.md`](BIT_NATIVE_INTELLIGENCE_FRAMING.md) | The thesis (intelligence below language) + an evidence scorecard |
| [`blm.py`](blm.py) | The original learned-address memory machine (SOM learning, recurrent addresses, latch) |
| `mix.py` / `mixfast.py` | Online logistic context mixing (the simple model); `mixfast` uses lossless integer keys |
| `mixns.py` / `mixnsfast.py` / `mixnshash.py` | The strong model: high orders, match, sparse/word, two-layer mixer, SSE, non-stationarity, RMSProp |
| [`blmrs/`](blmrs/) | **The Rust core** — `blmrs` (simple) and `blmrs-strong` (strong); the native engine behind the headline |
| `mdp.py` / `action.py` / `decide.py` / `stream.py` | Sequential RL, reward-driven action, confidence-gated decisions, anomaly detection |
| `aggregate.py` / `parity.py` / `learn_state.py` / `compose.py` | Learn-the-computation: compute-vs-hold, selecting & composing the recurrent computation |
| `induce.py` / `invent.py` / `recurse.py` / `library.py` / `refactor.py` | **Path B** — induction → invention → recursive synthesis → library learning → refactoring (the wall) |
| `wake_lgg.py` / `m3_different_family.py` / `invent_agg.py` / `stateful.py` | **Path B, cracked** — boundary-aware `detect(P)`, multi-axis generalisation, invented aggregations, invented counter (§50–53) |
| `real_test.py` / `real_mix.py` / `real_scale.py` | **Path B on real data** — induction on real text/DNA, induce+mix beats gzip, the scaling sweep (§54) |
| `bench.py` / `region.py` / `multi.py` / `gated.py` / … | The earlier capability probes (memory horizon, recall, latch) |

## Running it

The Python models need only Python 3 (stdlib). For scale, use **PyPy** (~3.5×, identical output) or the
**Rust** core (the strong model).

```bash
# Original memory machine (memorisation baseline) and recurrent recall
python blm.py --R 8 --weights uniform --lowo

# Next-bit compression on real text (bits/bit vs gzip)
python mix.py data/corpus.txt 300000          # simple model
python mixns.py data/corpus.txt 300000        # strong model
pypy3 mixfast.py data/corpus.txt 2000000      # faster (PyPy), identical output

# The Rust core (the headline engine)
cd blmrs && cargo build --release
./target/release/strong <path> <byte_cap> <obits>   # e.g. enwik8 100000000 27
```

(Corpora are not committed; any real file works. enwik8 is the standard benchmark.)

## Methodology

Claims here are not taken on faith. Each cycle ends with an **adversarial verification pass** —
independent agents audit the code, recompute baselines, and try to *break* the headline claim
(searching for counter-examples, optimising against it, re-implementing from scratch). Native results
add a **future-bit-flip causality self-test** (flip a future bit; every earlier prediction must be
bit-identical) to prove no leakage. Several results in the history were overturned this way and
corrected. The intent: a reader can reproduce and falsify every number.

---

*An exploratory research project. Contributions, replications, and refutations welcome.*

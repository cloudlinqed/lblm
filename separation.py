#!/usr/bin/env python3
"""
separation.py -- engine / language / knowledge are provably separable in the bit-native core.

Three things vary INDEPENDENTLY, fingerprinted and tested held-out:
  (E) the ENGINE   = the induced math (add.py's full adder, induced from disjoint sums) + the FIXED
                     prediction algorithm/hyperparameters of chat.Core.  A corpus-INDEPENDENT object.
  (L) the LANGUAGE = the grammar/Q&A shape the store acquires from a *fact-free* corpus.
  (K) the KNOWLEDGE= the append-only byte store (corpus_bytes/mtab/tables), filled ONLY by reading.

The crux (C4) is a SEPARATION guarantee: E is *defined* as the corpus-independent object (induced rule
+ fixed predictor code + frozen hyperparameters), so its fingerprint is byte-identical across learning
a language -> reading books -> learning a second (cipher) language BY CONSTRUCTION -- which is exactly
the claim: language and facts land entirely in the separable store K (which demonstrably grows, and
whose growth tracks behaviour).  We keep the fingerprint from being vacuous by SABOTAGE: breaking the
induced math / predictor / gate *moves* it (printed below), so it is engine-bound, not a constant.

(The corpora are generated in-memory from a seeded vocabulary; the run does ZERO file I/O -- in
particular it never reads data/chat.txt, which DOES contain France->Paris and would be a fatal leak.)

Honesty controls (from the adversarial review; see learned_binary_address_machine.md sec 62):
  * held-out math via run_transducer on pairs proven DISJOINT from training (compute, not recall);
  * the language/cipher-teaching corpora are PROVEN fact-free (no probe answer token, no 16-byte
    probe span) so the match model (MINLEN=16) cannot recall a fact it was not READ;
  * abstention is the model's own (verbatim-question recall + confidence), one fixed threshold, applied
    to EVERY query -- not a hard-coded "I don't know" guard;
  * fact recall is labelled by measured longest-common-substring: >=16B is SPAN RECALL, not
    "comprehension" -- and the paraphrase boundary (<16B overlap -> abstain) is shown, not hidden;
  * "learning a language" is measured as bits/byte (readability) dropping on HELD-OUT text of that
    language, with a never-taught cipher C' as the negative control;
  * NON-CLAIM, shown explicitly: the byte-match engine does not TRANSLATE -- a question about a
    cipher-stored fact, asked in the other language, stays gated even after the model learns to read it.

Run:  python separation.py       (one command; builds its own corpora; prints a falsifiable report)
"""
import sys, math, random, hashlib

import add
import chat
import separation_data as D

# ---- frozen knobs of the experiment (fixed a priori, applied uniformly to every probe) ----
TAU = 0.74                 # confidence threshold for COMMIT vs ABSTAIN (chosen once; see calibration)
MINLEN = 16                # chat.Core's match key length (a fact span must be >=16B to be recalled)
ANS_BYTES = 48
ANS_TEMP = 0.5
ANS_SEED = 0
LANG_N = 500               # size of the fact-free English corpus (lines)
CIPHER_N = 350             # size of the cipher passage the model reads to learn the cipher language
HELDOUT_N = 120


def die(reason):
    print("\nVERDICT: FAKE  (" + reason + ")")
    sys.exit(1)


# ======================================================================================
#  ENGINE / STORE FINGERPRINTS
# ======================================================================================
def engine_fingerprint(out_fn, upd_fn):
    """SHA-256 over the CORPUS-INDEPENDENT engine: the induced math + the fixed prediction code +
    frozen hyperparameters.  EXCLUDES tables/w/corpus_bytes/mtab (those are the knowledge store).
    Bound to run_transducer bytecode and the induced (out_fn,upd_fn) so a BROKEN engine hashes
    differently -- it is neither a vacuous source-hash nor a hash of a trained model."""
    h = hashlib.sha256()
    h.update(add.run_transducer.__code__.co_code)
    h.update(repr((out_fn, upd_fn, add.L)).encode())
    for m in (chat.Core._predict, chat.Core._match_advance, chat.Core.ctx,
              chat.Core.train, chat.Core.generate):
        h.update(m.__code__.co_code)
    h.update(repr((chat.ORDERS, chat.NM, chat.I_BIAS, chat.NW, chat.DELTA,
                   chat.MATCH_GATE)).encode())
    # behavioural canary: the math faculty must produce identical bytes every stage
    canary = [add.run_transducer(a, b, out_fn, upd_fn) for a, b in
              [(0, 0), (1, 1), (255, 1), (1024, 2048), (4095, 4095), (7, 8), (100, 200), (333, 444)]]
    h.update(repr(canary).encode())
    return h.hexdigest()


def store_fingerprint(core):
    """SHA-256 over the knowledge store -- must CHANGE as the model reads."""
    h = hashlib.sha256()
    h.update(bytes(core.corpus_bytes))
    h.update(repr(sorted(core.mtab.items())).encode())
    h.update(repr(core.w).encode())
    for t in core.tables:
        h.update(repr(sorted(t.items())).encode())
    return h.hexdigest()


# ======================================================================================
#  THE SINGLE SHARED ANSWERING PATH (used for every query: facts, ciphers -- never per-fact code)
# ======================================================================================
def answer(core, query, prompt=None, tau=TAU, n_bytes=ANS_BYTES, temp=ANS_TEMP, seed=ANS_SEED):
    """request -> (text, max_mlen, mean_conf, committed).  Mirrors chat.Core.generate but reads out
    the match length (traceability) and the model's own per-bit certainty (confidence).  COMMIT iff a
    long match span locked (answer is traceable to READ text) AND mean confidence >= tau; else ABSTAIN.
    There is NO per-question or per-fact branching here -- the same code answers everything.
    `prompt` overrides the default Q/A framing (used to ask a question entirely in another language --
    framing and all -- since a language's books are written in that language, scaffolding included)."""
    wrapped = prompt if prompt is not None else f"Q: {query}\nA:"
    rng = random.Random(seed)
    bits = chat.text_to_bits(wrapped)
    hist = bytearray(core.corpus_bytes)
    mtab = dict(core.mtab)
    mptr, mlen = -1, 0
    for by in wrapped.encode("utf-8", "replace"):
        hist.append(by)
        mptr, mlen = core._match_advance(hist, mtab, mptr, mlen)
    # prime_mlen = how much of the QUESTION itself was found verbatim in what was READ.  This is the
    # honest "have I actually read this exact question?" signal -- a novel question locks nothing, so
    # the model cannot confabulate a fluent-but-ungrounded answer past the abstain gate.
    prime_mlen = mlen
    out = bytearray()
    max_mlen = 0
    csum = 0.0; cn = 0
    for _ in range(n_bytes * 8):
        p, _, _ = core._predict(bits, len(bits), hist, mptr, mlen, learn=False)
        csum += max(p, 1.0 - p); cn += 1
        pt = p
        if temp != 1.0:
            a = p ** (1.0 / temp); b = (1.0 - p) ** (1.0 / temp)
            pt = a / (a + b) if (a + b) > 0 else 0.5
        bit = 1 if rng.random() < pt else 0
        bits.append(bit)
        if (len(bits) & 7) == 0:
            v = 0
            for j in range(len(bits) - 8, len(bits)):
                v = (v << 1) | bits[j]
            hist.append(v); out.append(v)
            mptr, mlen = core._match_advance(hist, mtab, mptr, mlen)
            if mlen > max_mlen:
                max_mlen = mlen
            if 10 in out:        # newline -> end of the answer line
                break
    text = out.decode("utf-8", "replace")
    if "\n" in text:
        text = text[:text.index("\n")]
    text = text.strip()
    mean_conf = csum / max(1, cn)
    # COMMIT iff the QUESTION was found verbatim in read text (the binding gate: verbatim-question
    # recall, prime_mlen >= MATCH_GATE) AND the answer was produced with high certainty (conf >= tau).
    # ABSTAIN otherwise -- the SAME rule for every query, no per-fact code.  A fluent confabulation
    # (high conf, prime_mlen 0) abstains: confidence alone is not enough, recall-traceability is.
    traceable = prime_mlen >= chat.MATCH_GATE
    committed = traceable and (mean_conf >= tau)
    return text, prime_mlen, mean_conf, committed


def bits_per_byte(core, text):
    """Cross-entropy (mixer only, no match) of `text` under the model -> readability of a language.
    Lower = the model can predict/read it.  Measured on HELD-OUT text (generalisation)."""
    bits = chat.text_to_bits(text)
    hist = bytearray()
    tot = 0.0
    for i in range(len(bits)):
        p, _, _ = core._predict(bits, i, hist, -1, 0, learn=False)
        pe = p if bits[i] == 1 else (1.0 - p)
        pe = min(1 - 1e-9, max(1e-9, pe))
        tot += -math.log2(pe)
        if (i & 7) == 7:
            v = 0
            for j in range(i - 7, i + 1):
                v = (v << 1) | bits[j]
            hist.append(v)
    return tot / (len(bits) / 8.0)


def lcs_len(a, b):
    """longest common substring length in bytes (for the span-recall vs comprehension label)."""
    a = a.encode(); b = b.encode()
    best = 0
    dp = [0] * (len(b) + 1)
    for i in range(1, len(a) + 1):
        prev = 0
        for j in range(1, len(b) + 1):
            t = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev + 1
                if dp[j] > best:
                    best = dp[j]
            else:
                dp[j] = 0
            prev = t
    return best


def train_core(text):
    c = chat.Core()
    c.train(chat.text_to_bits_bytes(text.encode("utf-8", "replace")))
    return c


# ======================================================================================
#  MAIN
# ======================================================================================
def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    random.seed(0)
    print("=" * 86)
    print("separation.py -- engine / language / knowledge are provably separable (held-out, audited)")
    print("=" * 86)

    # -------- the ENGINE: induce the math faculty (no language, no world knowledge) --------
    pairs = set()
    while len(pairs) < add.NTRAIN + 3000:
        pairs.add((random.randrange(1 << add.L), random.randrange(1 << add.L)))
    pairs = list(pairs)
    train = [(a, b, a + b) for a, b in pairs[:add.NTRAIN]]
    test = [(a, b, a + b) for a, b in pairs[add.NTRAIN:add.NTRAIN + 3000]]
    if set((a, b) for a, b, _ in train) & set((a, b) for a, b, _ in test):
        die("math test not disjoint from train")
    # induce (early-exit at the first transducer that fits all training sums)
    induced = None
    for of in range(256):
        for uf in range(256):
            if all(add.run_transducer(a, b, of, uf) == s for a, b, s in train):
                induced = (of, uf); break
        if induced:
            break
    if induced is None:
        die("no adder induced")
    OUT_FN, UPD_FN = induced
    held = sum(1 for a, b, s in test if add.run_transducer(a, b, OUT_FN, UPD_FN) == s) / len(test)
    mem = {(a, b): s for a, b, s in train}
    mem_held = sum(1 for a, b, s in test if mem.get((a, b), -1) == s) / len(test)
    print(f"\n[ENGINE]  induced adder out_fn={OUT_FN} (XOR3={OUT_FN==150})  upd_fn={UPD_FN} (MAJ3={UPD_FN==232})")
    print(f"  C1 math is INNATE (computed, not recalled):  held-out add = {held*100:.1f}%   "
          f"memoriser held-out = {mem_held*100:.1f}%   [disjoint pairs]")
    if not (OUT_FN, UPD_FN) == (150, 232):
        die(f"induced transducer {induced} is not the full adder")
    if held < 1.0 or mem_held > 0.01:
        die(f"math faculty not generalising (held={held}, mem={mem_held})")

    F = {}; G = {}
    F[0] = engine_fingerprint(OUT_FN, UPD_FN)
    print(f"  engine_fingerprint F0 = {F[0][:16]}...   (math faculty works before ANY language)")
    # the fingerprint is engine-BOUND, not a vacuous constant: sabotaging the induced math MOVES it.
    sab = engine_fingerprint(OUT_FN, 233)        # MAJ3=232 -> 233 (a broken carry)
    print(f"  sabotage check: a broken adder (upd_fn=233) hashes {sab[:16]}.. != F0 -> {sab != F[0]}"
          f"   [so F-constancy across stages is a real separation guarantee, not a constant]")
    if sab == F[0]:
        die("engine fingerprint is vacuous (insensitive to a broken adder)")

    # -------- corpora (built fresh; audited fact-free) --------
    cipherC = D.make_cipher(7)
    cipherCp = D.make_cipher(99)
    inv_cipherC = {v: k for k, v in cipherC.items()}
    LANG = D.fact_free_english(seed=1, n=LANG_N)
    BOOK_ZORBIA = D.book_zorbia()
    BOOK_VIMBO = D.book_vimbo(cipherC)
    CIPHER_PASSAGE = D.cipher_passage(seed=5, n=CIPHER_N, cipher=cipherC)
    en_heldout = D.fact_free_english(seed=2, n=HELDOUT_N)
    nv_heldout = D.novel_vocab_english(seed=4, n=HELDOUT_N)     # same grammar, DISJOINT vocabulary
    cC_heldout = D.apply_cipher(D.fact_free_english(seed=3, n=HELDOUT_N), cipherC)
    cCp_heldout = D.apply_cipher(D.fact_free_english(seed=3, n=HELDOUT_N), cipherCp)
    battery = D.fact_battery()
    banned = D.banned_answer_tokens()

    def audit_fact_free(name, corpus):
        cb = corpus.encode("utf-8", "replace")
        hits = [t for t in banned if t.encode() in cb]
        # also: no 16-byte span of any probe (q+a) may occur -> match model cannot fire on a fact
        spanhits = 0
        probes = [q + " " + a for q, a in battery] + \
                 [D.query_zorbia() + " Quex", "What is the capital of Vimbo? Zaxe"]
        for pr in probes:
            pb = pr.encode()
            for k in range(len(pb) - 15):
                if pb[k:k + 16] in cb:
                    spanhits += 1; break
        print(f"  leak audit [{name}]: banned-token hits={len(hits)}  16-byte probe spans={spanhits}")
        if hits or spanhits:
            die(f"{name} is not fact-free (hits={hits}, spans={spanhits})")

    print("\n[corpora]  building fact-free English, a cipher language, and novel-fact books...")
    audit_fact_free("english", LANG)
    audit_fact_free("cipherC-passage", CIPHER_PASSAGE)

    # ================= STAGE LADDER =================
    def stage_check(k, core, prev_corpus, label):
        F[k] = engine_fingerprint(OUT_FN, UPD_FN)
        G[k] = store_fingerprint(core)
        if F[k] != F[0]:
            die(f"engine changed at {label} (F{k}!=F0)")
        if G[k] == G.get(k - 1):
            die(f"store did NOT change at {label} (knowledge not added)")
        if not bytes(core.corpus_bytes).startswith(prev_corpus):
            die(f"store not append-monotonic at {label} (corpus rewritten, not appended)")
        print(f"  engine F{k}={F[k][:16]}.. ==F0:{F[k]==F[0]}   store G{k}={G[k][:10]}.. changed:{G[k]!=G.get(k-1)}   "
              f"corpus +{len(core.corpus_bytes)-len(prev_corpus)}B (prefix-monotonic)")

    G[-1] = store_fingerprint(chat.Core())   # empty store baseline

    # ---- S1: LEARN ENGLISH (fact-free) ----
    print("\n-- S1  LEARN ENGLISH (fact-free corpus) --------------------------------------------------")
    coreS1 = train_core(LANG)
    empty = chat.Core()
    bpb_en_S0 = bits_per_byte(empty, en_heldout)
    bpb_en_S1 = bits_per_byte(coreS1, en_heldout)
    bpb_nv_S1 = bits_per_byte(coreS1, nv_heldout)    # novel-vocabulary control (same grammar, new words)
    print(f"  readability of held-out English  S0(untrained)={bpb_en_S0:.2f}  ->  S1={bpb_en_S1:.2f} bits/byte"
          f"   [language learned: {bpb_en_S1 < bpb_en_S0 - 0.5}]")
    print(f"  novel-VOCABULARY control (same grammar, unseen words): S1={bpb_nv_S1:.2f} bits/byte"
          f"   [stays high -> what was learned is THIS language's closed vocabulary/grammar, NOT arbitrary English]")
    if bpb_en_S1 >= bpb_en_S0 - 0.5:
        die("English stage did not actually learn the language (bits/byte did not drop)")
    # C2 / C3: the model still knows NO facts after learning the language
    fc = ab = 0
    for q, a in battery:
        t, ml, cf, com = answer(coreS1, q)
        ok = com and (a.lower() in t.lower())
        fc += ok; ab += (not com)
    print(f"  C2 ignorant of world facts (language loaded, no books): fact-correct={fc}/{len(battery)}  "
          f"abstain={ab}/{len(battery)}")
    print(f"  C3 language != knowledge, and language did not touch the engine:")
    stage_check(1, coreS1, b"", "S1")
    if fc != 0 or ab != len(battery):
        die("learning the language alone made facts answerable (leak) or abstention is broken")

    # ---- S2: READ A BOOK (English, novel fact, matching phrasing) ----
    print("\n-- S2  READ A BOOK in English (novel fact: Zorbia -> Quex) -------------------------------")
    pre_t, pre_ml, pre_cf, pre_com = answer(coreS1, D.query_zorbia())   # pre-book snapshot
    coreS2 = train_core(LANG + BOOK_ZORBIA)
    post_t, post_ml, post_cf, post_com = answer(coreS2, D.query_zorbia())
    lcs = lcs_len("The capital of Zorbia is Quex", "Q: " + D.query_zorbia() + "\nA:")
    label = "SPAN RECALL (not comprehension)" if lcs >= MINLEN else "comprehension (<16B overlap!)"
    print(f"  pre-book  '{D.query_zorbia()}'  -> conf={pre_cf:.2f} mlen={pre_ml} {'COMMIT '+pre_t if pre_com else 'ABSTAIN'}")
    print(f"  post-book '{D.query_zorbia()}'  -> conf={post_cf:.2f} mlen={post_ml} "
          f"{'COMMIT '+repr(post_t) if post_com else 'ABSTAIN'}   correct={'Quex' in post_t}")
    print(f"  retrieval label: query/answer share {lcs}B  ->  {label}")
    # the paraphrase BOUNDARY: ask the same fact with <16B overlap -> honest abstain
    par_t, par_ml, par_cf, par_com = answer(coreS2, D.paraphrase_query_zorbia())
    par_lcs = lcs_len("The capital of Zorbia is Quex", "Q: " + D.paraphrase_query_zorbia() + "\nA:")
    print(f"  paraphrase boundary '{D.paraphrase_query_zorbia()}' (overlap {par_lcs}B) -> "
          f"{'COMMIT '+par_t if par_com else 'ABSTAIN'}   [recall is span-level, not semantic]")
    # un-booked facts STILL abstain (knowledge is exactly what was read)
    ab2 = sum(1 for q, a in battery if not answer(coreS2, q)[3])
    print(f"  un-booked world facts still abstain: {ab2}/{len(battery)}")
    stage_check(2, coreS2, bytes(coreS1.corpus_bytes), "S2")
    if pre_com:
        die("pre-book Zorbia answered (novel entity leaked)")
    if not (post_com and "Quex" in post_t):
        die("reading the book did not make the novel fact retrievable")
    if ab2 != len(battery):
        die("reading one book leaked other facts")

    # ---- S3: A FACT WRITTEN ONLY IN A CIPHER LANGUAGE (present, but gated by query language) ----
    print("\n-- S3  ENCOUNTER A FACT WRITTEN ONLY IN CIPHER C (Vimbo) ---------------------------------")
    coreS3 = train_core(LANG + BOOK_ZORBIA + BOOK_VIMBO)
    present = D.apply_cipher("capital of Vimbo", cipherC).encode() in bytes(coreS3.corpus_bytes)
    en_t, en_ml, en_cf, en_com = answer(coreS3, D.query_vimbo_en())              # ask in ENGLISH
    cipher_prompt = D.apply_cipher("Q: What is the capital of Vimbo?\nA:", cipherC)   # framing in C too
    ci_t, ci_ml, ci_cf, ci_com = answer(coreS3, "(asked in cipher C)", prompt=cipher_prompt)
    ci_dec = D.apply_cipher(ci_t, inv_cipherC)                                    # decode ONLY to score
    print(f"  cipher fact physically present in store: {present}")
    print(f"  ask in ENGLISH  '{D.query_vimbo_en()}' -> {'COMMIT '+en_t if en_com else 'ABSTAIN'}  "
          f"[gated: knowledge is in a language the query is not]")
    print(f"  ask in CIPHER C (its own language)      -> {'COMMIT' if ci_com else 'ABSTAIN'}  "
          f"emit={ci_t!r}  decoded={ci_dec!r}  correct={'Zaxe' in ci_dec}")
    stage_check(3, coreS3, bytes(coreS2.corpus_bytes), "S3")
    if not present:
        die("cipher fact not stored")
    if en_com:
        die("English query reached a cipher-only fact (gate failed)")
    if not (ci_com and "Zaxe" in ci_dec):
        die("same-language query could not retrieve the stored cipher fact")

    # ---- S4: LEARN TO READ THE CIPHER LANGUAGE (engine still byte-identical) ----
    print("\n-- S4  LEARN TO READ CIPHER C (read a fact-free passage written in it) -------------------")
    audit_fact_free("cipherC-passage(read)", CIPHER_PASSAGE)
    coreS4 = train_core(LANG + BOOK_ZORBIA + BOOK_VIMBO + CIPHER_PASSAGE)
    bpb_cC_S3 = bits_per_byte(coreS3, cC_heldout)
    bpb_cC_S4 = bits_per_byte(coreS4, cC_heldout)
    bpb_cCp_S4 = bits_per_byte(coreS4, cCp_heldout)
    print(f"  readability of held-out CIPHER C   S3(before)={bpb_cC_S3:.2f}  ->  S4(after)={bpb_cC_S4:.2f} bits/byte"
          f"   [learned to read C: {bpb_cC_S4 < bpb_cC_S3 - 0.5}]")
    print(f"  negative control, never-taught CIPHER C'  S4={bpb_cCp_S4:.2f} bits/byte"
          f"   [still unreadable: {bpb_cCp_S4 > bpb_cC_S4 + 0.5}]")
    # honest NON-CLAIM: the byte-match engine does not translate -> English query for the cipher fact
    # stays gated even now (reading-competence != translation)
    en2_t, _, _, en2_com = answer(coreS4, D.query_vimbo_en())
    print(f"  NON-CLAIM check: English query for the cipher fact, AFTER learning to read C -> "
          f"{'COMMIT '+en2_t if en2_com else 'ABSTAIN'}  [no translation; honest boundary]")
    stage_check(4, coreS4, bytes(coreS3.corpus_bytes), "S4")
    if not (bpb_cC_S4 < bpb_cC_S3 - 0.5):
        die("did not learn to read cipher C (bits/byte did not drop)")
    if not (bpb_cCp_S4 > bpb_cC_S4 + 0.5):
        die("never-taught cipher C' became readable (cipher was trivially decodable)")

    # ---- S5: STEADY STATE -- the crux invariant over ALL stages ----
    print("\n-- S5  STEADY STATE: the crux invariant -------------------------------------------------")
    same_engine = all(F[k] == F[0] for k in range(5))
    store_grew = (G[1] != G[-1]) and (G[2] != G[1]) and (G[3] != G[2]) and (G[4] != G[3])
    held2 = sum(1 for a, b, s in test if add.run_transducer(a, b, OUT_FN, UPD_FN) == s) / len(test)
    print(f"  engine/store SEPARATION holds: engine F0..F4 identical (by construction) AND sabotage-sensitive,")
    print(f"     while the store changed at every stage -- so language+facts are STORE, never engine : {same_engine and store_grew}")
    print(f"  math faculty still 100% on held-out : {held2*100:.1f}%   (the engine is the same object throughout)")
    if not (same_engine and store_grew and held2 == 1.0):
        die("crux invariant violated")

    print("\n" + "=" * 86)
    print("VERDICT: HONEST  (math innate & generalises 100% vs memoriser 0% | engine/store SEPARATION: "
          "engine fingerprint constant-by-construction yet sabotage-sensitive, store grew monotonically "
          "| facts unknown until READ | recall is span-level (shown) | language learning is vocab/grammar-"
          "bound (shown) | cipher learned as readability, C' not | abstain = verbatim-question recall + "
          "fixed TAU, no fact/decoder code)")
    print("=" * 86)
    print("\nWhat this proves, and what it does NOT:")
    print("  * SEPARATION: the ENGINE (induced math + fixed predictor) is, by construction, corpus-")
    print("    independent; learning a language / reading books put everything into the separable STORE")
    print("    -- nothing leaked into the engine (and the engine WOULD change if its math/predictor did).")
    print("  * MATH generalises (computed, not recalled); FACTS are absent until READ, gated by the")
    print("    LANGUAGE they are written in; abstention is earned (verbatim recall), not hard-coded.")
    print("  * NOT claimed: comprehension/grounding (recall is span-level, labelled by LCS; a paraphrase")
    print("    abstains), translation (a cipher fact asked in the other language stays gated), nor reading")
    print("    arbitrary text (the learned language is vocabulary/grammar-bound: the novel-vocab control).")


if __name__ == "__main__":
    main()

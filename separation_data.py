#!/usr/bin/env python3
"""
separation_data.py -- corpora for separation.py's engine / language / knowledge separation proof.

Everything here is built from a CONSTRAINED, fact-free vocabulary so that "teaching the language"
provably injects NO world-knowledge.  The auditor (separation.audit_fact_free) asserts at load time that
no probe answer token and no probe Q/A 16-byte span occurs in the language/alignment corpora -- so
the match model (chat.Core, MINLEN=16) literally cannot recall a fact it was not READ in a book.

Pieces:
  fact_free_english(seed, n)     -- English grammar + Q/A SHAPE, zero world facts (teach the language)
  cipher_passage(seed, n, ciph)  -- a fact-free passage rendered in a cipher LANGUAGE (a book to read)
  book_zorbia()                  -- a NOVEL fact in English, matching phrasing (span-recall retrieval)
  paraphrase_query_zorbia()      -- the SAME fact asked with <16B overlap (the comprehension boundary)
  book_vimbo(cipherC)            -- a NOVEL fact stated ONLY in cipher C (gated to that language)
  query_vimbo_en / _cipher       -- ask Vimbo in English (gated) vs in its own language (reachable)
  make_cipher(seed)              -- a seeded RANDOM permutation of the alphabet (not ROT-n)
  apply_cipher(text, cipher)     -- letterwise encode; spaces/digits/punctuation untouched
  fact_battery()                 -- real world facts (France->Paris ...) the model must NOT know unread

No file I/O, no third-party deps; pure stdlib + a single seeded random.Random.
"""
import random

# --- constrained, demonstrably fact-free vocabulary (no capitals, no animals, no numbers, no proper
#     nouns, and none of the probe answer tokens) -------------------------------------------------
NOUNS = ["box", "circle", "square", "dot", "line", "gate", "path", "stone", "leaf", "cup",
         "ring", "wave", "field", "road", "wall", "door", "seed", "bell", "nest", "rope",
         "shell", "bead", "frame", "tile", "knot", "vane", "reed", "husk", "plank", "spool"]
ADJS = ["small", "large", "round", "flat", "warm", "cold", "quiet", "bright", "heavy", "light",
        "narrow", "wide", "smooth", "rough", "still", "busy", "soft", "firm", "clear", "dim"]
VERBS = ["waits", "moves", "turns", "rests", "shifts", "glows", "fades", "rolls", "stands", "leans",
         "sways", "drifts", "settles", "hums", "tilts", "spins"]
DIRS = ["here", "there", "near", "by the wall", "by the gate", "to the left", "to the right",
        "above", "below", "in the field"]


# a DISJOINT vocabulary for the novel-vocabulary control (same grammar, words never seen in training)
NOVEL_NOUNS = ["plume", "quartz", "fjord", "glyph", "cobweb", "marsh", "cradle", "thorn", "brook",
               "lantern", "pebble", "willow", "harbor", "meadow", "cavern", "trellis", "anchor",
               "ember", "satchel", "furrow", "beacon", "bramble", "crevice", "gusset", "tangle"]
NOVEL_ADJS = ["amber", "jagged", "velvet", "hollow", "crisp", "murky", "gilded", "supple", "brittle",
              "fragrant", "sodden", "lofty", "gaunt", "ruddy", "limpid", "askew", "molten", "frosty"]
NOVEL_VERBS = ["wanders", "clatters", "simmers", "ripples", "crumbles", "gleams", "wavers", "lingers",
               "trembles", "scatters", "buckles", "smolders", "billows", "flickers"]
NOVEL_DIRS = ["beyond the ridge", "past the harbor", "under the willow", "along the furrow",
              "behind the trellis", "near the cavern", "atop the beacon", "across the meadow"]


def _sent(rng, nouns=NOUNS, adjs=ADJS, verbs=VERBS, dirs=DIRS):
    """One fact-free English line (declarative or Q/A), teaching grammar + answer shape only."""
    t = rng.randrange(6)
    n1 = rng.choice(nouns); a1 = rng.choice(adjs); v1 = rng.choice(verbs); d1 = rng.choice(dirs)
    if t == 0:
        return f"The {n1} is {a1}.\n"
    if t == 1:
        n2 = rng.choice(nouns); v2 = rng.choice(verbs)
        return f"The {n1} {v1} and the {n2} {v2}.\n"
    if t == 2:
        return f"A {a1} {n1} {v1} {d1}.\n"
    if t == 3:
        return f"Q: Is the {n1} {a1}?\nA: The {n1} is {a1}.\n"
    if t == 4:
        return f"Q: Where is the {n1}?\nA: The {n1} is {d1}.\n"
    return f"Q: What does the {n1} do?\nA: The {n1} {v1}.\n"


def fact_free_english(seed=1, n=900):
    rng = random.Random(seed)
    return "".join(_sent(rng) for _ in range(n))


def novel_vocab_english(seed=4, n=120):
    """Same grammar, a DISJOINT vocabulary -> the readability control: it isolates how much of the
    'learned English' is generalisation of structure vs recall of the closed training vocabulary."""
    rng = random.Random(seed)
    return "".join(_sent(rng, NOVEL_NOUNS, NOVEL_ADJS, NOVEL_VERBS, NOVEL_DIRS) for _ in range(n))


def make_cipher(seed):
    """A seeded RANDOM permutation of a..z (NOT a Caesar/ROT shift -> not trivially frequency-broken)."""
    rng = random.Random(seed)
    src = list("abcdefghijklmnopqrstuvwxyz")
    dst = src[:]
    rng.shuffle(dst)
    m = {s: d for s, d in zip(src, dst)}
    m.update({s.upper(): d.upper() for s, d in zip(src, dst)})
    return m


def apply_cipher(text, cipher):
    return "".join(cipher.get(ch, ch) for ch in text)


def cipher_passage(seed, n, cipher):
    """A fact-free English passage rendered in a cipher language -- a 'book' written in that language."""
    return apply_cipher(fact_free_english(seed=seed, n=n), cipher)


# --- NOVEL facts (coined entities with no real-world prior, so a pre-reading answer is impossible) ---
def book_zorbia():
    # matching phrasing -> retrieval is honest SPAN RECALL (LCS with the query >= 16 bytes)
    s = ("Q: What is the capital of Zorbia?\nA: The capital of Zorbia is Quex.\n"
         "The capital of Zorbia is Quex.\n")
    return s * 3


def query_zorbia():
    return "What is the capital of Zorbia?"          # matches the book span -> commits 'Quex'


def paraphrase_query_zorbia():
    return "Which city governs the land of Zorbia?"  # <16B overlap with the book -> the recall boundary


def book_vimbo(cipherC):
    plain = ("Q: What is the capital of Vimbo?\nA: The capital of Vimbo is Zaxe.\n"
             "The capital of Vimbo is Zaxe.\n")
    return apply_cipher(plain * 3, cipherC)


def query_vimbo_en():
    return "What is the capital of Vimbo?"            # asked in ENGLISH -> gated (fact is in cipher)


def query_vimbo_cipher(cipherC):
    return apply_cipher("What is the capital of Vimbo?", cipherC)   # asked in the fact's own language


def book_wuldo(cipherCprime):
    plain = "Q: What is the capital of Wuldo?\nA: The capital of Wuldo is Brem.\n"
    return apply_cipher(plain * 3, cipherCprime)     # cipher C' is never taught -> permanently gated


def fact_battery():
    """Real world facts. The model must NOT answer any of these until it READS them in a book."""
    return [("What is the capital of France?", "Paris"),
            ("What is the capital of Japan?", "Tokyo"),
            ("What is the capital of Egypt?", "Cairo"),
            ("What is the capital of Brazil?", "Brasilia"),
            ("What is the capital of Canada?", "Ottawa")]


# tokens that must be ABSENT from the fact-free language / alignment corpora (else a fact leaked)
def banned_answer_tokens():
    toks = ["Paris", "Tokyo", "Cairo", "Brasilia", "Ottawa",
            "Quex", "Zaxe", "Brem", "Zorbia", "Vimbo", "Wuldo", "capital"]
    return toks


if __name__ == "__main__":
    L = fact_free_english()
    print(f"fact_free_english: {len(L)} bytes, {L.count(chr(10))} lines")
    c = make_cipher(7)
    print("cipher sample :", apply_cipher("the capital of vimbo is zaxe", c))
    print("zorbia book   :", repr(book_zorbia()[:64]))
    for q, a in fact_battery():
        print("  battery:", q, "->", a)

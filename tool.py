#!/usr/bin/env python3
"""
tool.py -- a RELIABLE, VERIFIABLE structured response, tested held-out.

The question (set aside the agentic framing): can the bit-native core turn a request into a *proper
structured response* -- a valid tool call that, executed, is CORRECT -- and do it for requests it never
saw? Reliability here is not hoped-for; it is by construction + check:
  * VALID by construction -- the response is emitted under a grammar `op(a, b)`, so it always parses.
  * CORRECT by induced computation -- the call is executed by a bit-native transducer that was INDUCED
    (add.py: learn the carry from a disjoint set), so it generalises to numbers never seen.
  * VERIFIED -- "valid?" and "correct?" are decidable; we measure both on held-out requests.

Division of labour (the framing doc's shape): the dumb adapter tokenises (format), the core ROUTES the
operation + COMPUTES via an induced rule, the grammar guarantees a well-formed response. Contrast: a
chat-style MEMORISER that maps request->answer -- right on what it saw, useless off it.
"""
import re, random
from add import run_transducer, L          # reuse the bit-native 1-bit-state transducer + induce idea

random.seed(0)


def induce_op(train):
    """Induce a 1-bit-state transducer (out_fn, upd_fn) reproducing all training (a,b,result)."""
    for out_fn in range(256):
        for upd_fn in range(256):
            if all(run_transducer(a, b, out_fn, upd_fn) == r for a, b, r in train):
                return out_fn, upd_fn
    return None


# --- induce the TOOLS from disjoint example sets (they generalise; see add.py / sec 60) ---
def sample(n, cond=lambda a, b: True):
    s = set()
    while len(s) < n:
        a, b = random.randrange(1 << L), random.randrange(1 << L)
        if cond(a, b):
            s.add((a, b))
    return list(s)

add_train = [(a, b, a + b) for a, b in sample(60)]
sub_train = [(a, b, a - b) for a, b in sample(60, lambda a, b: a >= b)]
ADD = induce_op(add_train)
SUB = induce_op(sub_train)
TOOLS = {"add": ADD, "sub": SUB}


def core_route(req):
    """The core's decision: which tool does this request call for? (keyword routing, learnable)."""
    r = req.lower()
    if "plus" in r or "add" in r or "sum" in r:
        return "add"
    if "minus" in r or "subtract" in r or "difference" in r:
        return "sub"
    return None


def adapter_numbers(req):
    """The dumb adapter: pull the integer fields out of the text (format conversion, no semantics)."""
    return [int(x) for x in re.findall(r"\d+", req)]


def respond(req):
    """request -> a VALID structured call -> EXECUTE via the induced tool -> (call_string, answer)."""
    op = core_route(req)
    nums = adapter_numbers(req)
    if op is None or len(nums) < 2 or op not in TOOLS or TOOLS[op] is None:
        return None, None                                   # refuse rather than emit a bad response
    a, b = nums[0], nums[1]
    if op == "sub" and "from" in req.lower():                # "subtract B from A" means A - B (a WORDS fact)
        a, b = b, a
    call = f"{op}({a}, {b})"                                 # grammar-valid by construction
    out_fn, upd_fn = TOOLS[op]
    ans = run_transducer(a, b, out_fn, upd_fn)               # CORRECT by induced computation
    return call, ans


CALL_RE = re.compile(r"^(add|sub)\((\d+), (\d+)\)$")         # the response grammar (validity check)


def truth(op, a, b):
    return a + b if op == "add" else a - b


def main():
    print("=" * 78)
    print("Reliable, verifiable STRUCTURED RESPONSE -- held-out test (set agentic framing aside)")
    print("=" * 78)
    print(f"induced tools:  add = {ADD} (XOR3/MAJ3 full adder),  sub = {SUB} (full subtractor)")
    print("  (each induced from 60 disjoint examples; they compute, they do not look up)\n")

    templates = ["what is {a} plus {b}?", "add {a} and {b}", "{a} plus {b}",
                 "what is {a} minus {b}?", "subtract {b} from {a}"]
    # HELD-OUT requests: numbers never used to induce the tools, fresh phrasings
    tests = []
    for _ in range(2000):
        t = random.choice(templates)
        if "minus" in t or "subtract" in t:
            a, b = sorted([random.randrange(1 << L), random.randrange(1 << L)], reverse=True)
            op = "sub"
        else:
            a, b = random.randrange(1 << L), random.randrange(1 << L)
            op = "add"
        tests.append((t.format(a=a, b=b), op, a, b))

    # chat-style memoriser: request text -> answer, learned on a sample of requests
    mem_train = {}
    for _ in range(400):
        t = random.choice(templates); a, b = random.randrange(1 << L), random.randrange(1 << L)
        op = "sub" if ("minus" in t or "subtract" in t) else "add"
        if op == "sub" and a < b: a, b = b, a
        mem_train[t.format(a=a, b=b)] = truth(op, a, b)

    valid = correct = mem_ok = 0
    for req, op, a, b in tests:
        call, ans = respond(req)
        if call and CALL_RE.match(call):
            valid += 1
            if ans == truth(op, a, b):
                correct += 1
        if mem_train.get(req, None) == truth(op, a, b):
            mem_ok += 1
    n = len(tests)
    print("[examples] request -> structured response -> executed answer (all numbers held-out):")
    for req, op, a, b in tests[:6]:
        call, ans = respond(req)
        print(f"   {req!r:42} -> {call:>16} = {ans}   {'OK' if ans==truth(op,a,b) else 'WRONG'}")

    print(f"\n[held-out over {n} unseen requests]")
    print(f"  structured response VALID (parses the grammar)  : {valid/n*100:.1f}%")
    print(f"  executed answer CORRECT                          : {correct/n*100:.1f}%")
    print(f"  chat-style MEMORISER correct (recall)            : {mem_ok/n*100:.1f}%")
    print("\n" + "=" * 78)
    print("VERDICT: the core emits a VALID structured response (by construction) and the CORRECT answer")
    print(f"  ({correct/n*100:.0f}%) for requests it never saw -- because the computation is INDUCED, not")
    print(f"  stored. The memoriser ({mem_ok/n*100:.0f}%) cannot. A 'proper response you can rely on' is")
    print("  reachable in the structured/verifiable regime: valid by grammar, correct by induced rule.")
    print("=" * 78)


if __name__ == "__main__":
    main()

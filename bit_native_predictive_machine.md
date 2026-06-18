# Bit-Native Predictive Machine

## 1. Purpose

This document describes a bit-native predictive machine.

The machine receives a stream made only of two signal states:

```text
0
1
```

The machine observes recent signal history and produces the next signal.

The full cycle is:

```text
incoming bit stream
    ↓
rolling context register
    ↓
adjustable signal circuit
    ↓
next bit output
```

There is no text layer, symbol table, or external meaning system in this design. The bit stream itself is the native operating material.

---

## 2. Signal World

The machine operates in a closed signal world.

Only two signal states exist:

```text
0 = low state
1 = high state
```

These states are directly received by the machine as physical or logical signal levels.

The machine does not rename them. It does not map them into separate labels. It simply receives signal history and learns continuation behavior.

---

## 3. Example Signal Patterns

We define two observed patterns.

### Pattern A

```text
Question signal: 01010
Boundary signal: 111
Answer signal:   11
Stop signal:     000
```

Full stream:

```text
01010 111 11 000
```

Continuous form:

```text
0101011111000
```

### Pattern B

```text
Question signal: 10100
Boundary signal: 111
Answer signal:   00
Stop signal:     000
```

Full stream:

```text
10100 111 00 000
```

Continuous form:

```text
1010011100000
```

The boundary signal `111` marks the transition from the question region into the answer region.

The stop signal `000` marks the end of the response region.

---

## 4. Machine Components

The machine has four core parts.

```text
1. Bit input line
2. Rolling context register
3. Adjustable signal circuit
4. Bit output line
```

### 4.1 Bit Input Line

The input line receives one bit at a time:

```text
0, then 1, then 0, then 1, then ...
```

The input line does not interpret the bit. It only passes the signal into the machine.

### 4.2 Rolling Context Register

The context register stores the most recent fixed number of bits.

For this example, use an 8-bit register:

```text
[ _ _ _ _ _ _ _ _ ]
```

If the incoming stream is:

```text
01010111
```

then the register state is:

```text
[0 1 0 1 0 1 1 1]
```

When a new bit arrives or is produced, the register shifts.

Example:

```text
current register: [0 1 0 1 0 1 1 1]
new bit:           1
new register:     [1 0 1 0 1 1 1 1]
```

### 4.3 Adjustable Signal Circuit

The adjustable signal circuit contains internal strengths, gates, thresholds, and pattern responders.

It receives the register state and produces one output tendency:

```text
should the next signal be closer to 0 or closer to 1?
```

The circuit may develop internal responders such as:

```text
Responder A: active when the register resembles 01010111
Responder B: active when the register resembles 10100111
Responder C: active near stop-region patterns
```

These responders are not manually programmed as rules. Their strengths are shaped by exposure to the example streams.

### 4.4 Bit Output Line

The output line emits one signal:

```text
0
```

or

```text
1
```

The emitted bit can be appended back into the rolling register so the machine can continue producing a longer bit sequence.

---

## 5. Calibration Streams

The machine is calibrated using the two full streams:

```text
0101011111000
1010011100000
```

The calibration objective is simple:

```text
Given the current register state, strengthen the machine so it emits the observed next bit.
```

---

## 6. Register-Based Calibration Examples

Using an 8-bit register, each stream becomes a set of register states and next-bit targets.

### Stream A

Full stream:

```text
0101011111000
```

Register examples:

| Register state | Observed next bit |
|---|---:|
| `01010111` | `1` |
| `10101111` | `1` |
| `01011111` | `0` |
| `10111110` | `0` |
| `01111100` | `0` |

The answer-region behavior is:

```text
01010111  → 1
10101111  → 1
01011111  → 0
10111110  → 0
01111100  → 0
```

Generated continuation:

```text
11000
```

Answer region:

```text
11
```

Stop region:

```text
000
```

### Stream B

Full stream:

```text
1010011100000
```

Register examples:

| Register state | Observed next bit |
|---|---:|
| `10100111` | `0` |
| `01001110` | `0` |
| `10011100` | `0` |
| `00111000` | `0` |
| `01110000` | `0` |

The answer-region behavior is:

```text
10100111  → 0
01001110  → 0
10011100  → 0
00111000  → 0
01110000  → 0
```

Generated continuation:

```text
00000
```

Answer region:

```text
00
```

Stop region:

```text
000
```

---

## 7. Calibration Cycle

Each calibration cycle uses one register state and one observed next bit.

Example:

```text
register state:     01010111
observed next bit:  1
```

The machine performs this cycle:

```text
1. Load register state
2. Circuit produces output tendency
3. Compare tendency against observed next bit
4. Measure mismatch
5. Adjust internal strengths
6. Repeat with another register state
```

---

## 8. Example Adjustment Moment

Suppose the register state is:

```text
01010111
```

The observed next bit is:

```text
1
```

At an early stage, the circuit may lean toward `0`.

```text
current tendency:
0 strong
1 weak
```

That is a mismatch.

The machine adjusts its internal strengths:

```text
increase responder activity for 01010111
increase output pressure toward 1 for this region
reduce competing pressure toward 0
```

After repeated exposure, the same register state produces:

```text
01010111 → 1
```

---

## 9. Internal Strength Sketch

The circuit can be imagined as adjustable responders connected to the output line.

```text
8-bit register
[0 1 0 1 0 1 1 1]
        ↓
pattern responders
        ↓
output pressure
        ↓
next bit
```

For Pattern A:

```text
register: 01010111
Responder A: strong
Responder B: weak
Stop responder: weak
Output pressure: 1
```

For Pattern B:

```text
register: 10100111
Responder A: weak
Responder B: strong
Stop responder: weak
Output pressure: 0
```

Near the stop region:

```text
register: 01011111
Stop responder: active
Output pressure: 0
```

---

## 10. Calibrated Behavior

After enough calibration passes, the machine stabilizes around the observed continuations.

### Input A

Seed the register with:

```text
01010111
```

Step-by-step output:

```text
register: 01010111 → output: 1
register: 10101111 → output: 1
register: 01011111 → output: 0
register: 10111110 → output: 0
register: 01111100 → output: 0
```

Produced continuation:

```text
11000
```

Response region:

```text
11
```

### Input B

Seed the register with:

```text
10100111
```

Step-by-step output:

```text
register: 10100111 → output: 0
register: 01001110 → output: 0
register: 10011100 → output: 0
register: 00111000 → output: 0
register: 01110000 → output: 0
```

Produced continuation:

```text
00000
```

Response region:

```text
00
```

---

## 11. Full Signal Flow

The complete process is:

```text
observed streams
0101011111000
1010011100000
        ↓
slice into register states and next-bit observations
        ↓
load register state
        ↓
circuit produces output tendency
        ↓
compare with observed next bit
        ↓
adjust internal strengths
        ↓
repeat until behavior stabilizes
        ↓
use seeded register to produce new continuation
```

---

## 12. Minimal Design Summary

```text
Material:
0 and 1 signal states

Memory:
rolling context register

Core mechanism:
adjustable signal circuit

Calibration unit:
register state + observed next bit

Output:
one bit at a time

Continuation:
output bit shifts back into the register
```

The machine becomes a calibrated bit-continuation system:

```text
01010111 → 1 → 1 → 0 → 0 → 0
10100111 → 0 → 0 → 0 → 0 → 0
```

In compact form:

```text
01010 111 → 11 000
10100 111 → 00 000
```

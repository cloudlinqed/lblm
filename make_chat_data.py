#!/usr/bin/env python3
"""
make_chat_data.py -- build a REAL, correct Q/A training corpus (no mock data).

The model only imitates the statistical SHAPE of its training text. To make it answer-shaped we feed it
many "Q: ..\nA: ..\n" examples whose answers are actually TRUE: real arithmetic (computed, exact) and a
curated set of real facts (capitals, science, counting, opposites). It learns (a) the Q/A format and the
"stop after the answer" pattern, and (b) the specific facts/sums it sees. Output: data/chat.txt.

This is the "serious dataset work" lever -- more, cleaner, correct Q/A => better answering. Everything
here is verifiable real data; extend the FACTS / arithmetic ranges to scale it up.
"""
import random

random.seed(0)
rows = []


def qa(q, a):
    rows.append(f"Q: {q}\nA: {a}\n")


# --- real arithmetic (exact, computed) ---------------------------------------
for a in range(0, 31):
    for b in range(0, 31):
        qa(f"What is {a} + {b}?", a + b)
for a in range(0, 41):
    for b in range(0, a + 1):
        qa(f"What is {a} - {b}?", a - b)
for a in range(0, 13):
    for b in range(0, 13):
        qa(f"What is {a} times {b}?", a * b)
for _ in range(800):                                   # some larger sums for range
    a, b = random.randint(0, 500), random.randint(0, 500)
    qa(f"What is {a} + {b}?", a + b)
for n in range(1, 21):
    qa(f"What is double {n}?", 2 * n)
    qa(f"What is {n} squared?", n * n)

# --- real facts (all true; repeated so they are well learned) ----------------
CAPITALS = {
    "France": "Paris", "Japan": "Tokyo", "Italy": "Rome", "Spain": "Madrid", "Germany": "Berlin",
    "Egypt": "Cairo", "Canada": "Ottawa", "Russia": "Moscow", "China": "Beijing", "India": "New Delhi",
    "Brazil": "Brasilia", "Australia": "Canberra", "Greece": "Athens", "Portugal": "Lisbon",
    "Norway": "Oslo", "Sweden": "Stockholm", "Poland": "Warsaw", "Austria": "Vienna", "Turkey": "Ankara",
    "Mexico": "Mexico City", "Argentina": "Buenos Aires", "Kenya": "Nairobi", "Iran": "Tehran",
    "Iraq": "Baghdad", "Cuba": "Havana", "Peru": "Lima", "Chile": "Santiago", "Thailand": "Bangkok",
    "Ireland": "Dublin", "Finland": "Helsinki", "Hungary": "Budapest", "Switzerland": "Bern",
    "Netherlands": "Amsterdam", "Belgium": "Brussels", "Denmark": "Copenhagen", "Morocco": "Rabat",
}
FACTS = []
for c, cap in CAPITALS.items():
    FACTS.append((f"What is the capital of {c}?", cap))
    FACTS.append((f"What country has {cap} as its capital?", c))

LEGS = {"a spider": 8, "an insect": 6, "an ant": 6, "a dog": 4, "a cat": 4, "a horse": 4,
        "a human": 2, "a bird": 2, "a chicken": 2, "an octopus": 8, "a crab": 10}
for k, v in LEGS.items():
    FACTS.append((f"How many legs does {k} have?", v))

OPP = {"hot": "cold", "up": "down", "big": "small", "fast": "slow", "day": "night", "open": "closed",
       "happy": "sad", "left": "right", "true": "false", "wet": "dry", "light": "dark", "old": "new",
       "high": "low", "rich": "poor", "soft": "hard", "full": "empty"}
for k, v in OPP.items():
    FACTS.append((f"What is the opposite of {k}?", v))
    FACTS.append((f"What is the opposite of {v}?", k))

MISC = [
    ("How many days are in a week?", 7), ("How many months are in a year?", 12),
    ("How many days are in a year?", 365), ("How many hours are in a day?", 24),
    ("How many minutes are in an hour?", 60), ("How many seconds are in a minute?", 60),
    ("How many sides does a triangle have?", 3), ("How many sides does a square have?", 4),
    ("How many sides does a hexagon have?", 6), ("How many colors are in a rainbow?", 7),
    ("How many continents are there?", 7), ("How many planets are in the solar system?", 8),
    ("What is the largest planet?", "Jupiter"), ("What is the closest planet to the sun?", "Mercury"),
    ("What is the red planet?", "Mars"), ("What is the largest ocean?", "the Pacific"),
    ("What is the longest river?", "the Nile"), ("What is the tallest mountain?", "Everest"),
    ("What is the chemical symbol for water?", "H2O"), ("What is the chemical symbol for gold?", "Au"),
    ("What gas do plants breathe in?", "carbon dioxide"), ("What gas do humans breathe in?", "oxygen"),
    ("What is the first letter of the alphabet?", "A"), ("What is the last letter of the alphabet?", "Z"),
    ("What is the freezing point of water in Celsius?", 0),
    ("What is the boiling point of water in Celsius?", 100),
    ("What color is the sky on a clear day?", "blue"), ("What color is grass?", "green"),
    ("What is the sun?", "a star"), ("How many legs does a tripod have?", 3),
    ("What is 1 plus 1?", 2), ("What comes after Monday?", "Tuesday"),
    ("What comes after the number nine?", "ten"), ("What is half of 100?", 50),
    ("What is the speed of light's symbol?", "c"), ("What shape is a ball?", "round"),
]
FACTS.extend(MISC)

for _ in range(12):                                    # repeat facts so they stick vs many sums
    for q, a in FACTS:
        qa(q, a)

random.shuffle(rows)
text = "".join(rows)
open("data/chat.txt", "w", encoding="utf-8", newline="\n").write(text)  # keep clean \n (no \r\n)
print(f"wrote data/chat.txt: {len(rows)} Q/A pairs, {len(text)} bytes")
print("sample:\n" + "".join(rows[:6]))

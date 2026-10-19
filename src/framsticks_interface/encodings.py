"""f0 / f1 genotype helpers used by tests and as format utilities for the real backend.

`RealFramsticksInterface` delegates mutation/crossover/random generation to Framsticks
itself. These helpers exist so `tests.fakes.FakeFramsticksInterface` (and any offline
genotype plumbing) can produce encoding-shaped strings without the simulator.
"""
from __future__ import annotations

import random

# FramsticksLib.getSimplest() takes the encoding letter without the "f" prefix.
FORMAT_CODES = {"f0": "0", "f1": "1"}

F1_STRUCT = "XRLQ"
F1_NEURON = "NnGgTt"
F1_ALPHABET = F1_STRUCT + F1_NEURON + "C"

def format_code(encoding: str) -> str:
    """Map `"f0"` / `"f1"` to the Framsticks genetic-format code `"0"` / `"1"`."""
    try:
        return FORMAT_CODES[encoding]
    except KeyError as exc:
        raise ValueError(f"Unknown encoding '{encoding}'. Options: {sorted(FORMAT_CODES)}") from exc


def random_genotype(encoding: str, rng: random.Random | None = None, length: int = 12) -> str:
    """A plausible random genotype string for `encoding` (`f0` or `f1`)."""
    rng = rng or random.Random()
    if encoding == "f1":
        n = max(length, 4)
        chars = ["X"] + [rng.choice(F1_ALPHABET) for _ in range(n - 1)]
        if not any(c in F1_NEURON for c in chars):
            chars[-1] = rng.choice(F1_NEURON)
        return "".join(chars)
    if encoding == "f0":
        n_parts = max(3, length // 4)
        tokens = [f"p:{i},0,0" for i in range(n_parts)]
        if n_parts >= 2:
            tokens.append(f"j:0,{n_parts - 1}")
        tokens.append("n:conn")
        return " ".join(tokens)
    raise ValueError(f"Unknown encoding '{encoding}'")


def mutate_genotype(genotype: str, encoding: str, rng: random.Random | None = None) -> str:
    """Small local edit of a genotype string (insert / delete / replace)."""
    rng = rng or random.Random()
    if encoding == "f1":
        if not genotype:
            return random_genotype("f1", rng=rng)
        op = rng.choice(("replace", "insert", "delete"))
        chars = list(genotype)
        if op == "replace" or len(chars) <= 4:
            i = rng.randrange(len(chars))
            chars[i] = rng.choice(F1_ALPHABET)
        elif op == "insert":
            i = rng.randrange(len(chars) + 1)
            chars.insert(i, rng.choice(F1_ALPHABET))
        else:
            i = rng.randrange(len(chars))
            del chars[i]
        mutated = "".join(chars)
        if not any(c in F1_NEURON for c in mutated):
            mutated += rng.choice(F1_NEURON)
        return mutated
    if encoding == "f0":
        tokens = genotype.split(" ") if genotype else []
        if len(tokens) < 3:
            return random_genotype("f0", rng=rng)
        op = rng.choice(("replace", "insert", "delete"))
        if op == "replace":
            i = rng.randrange(len(tokens))
            tokens[i] = rng.choice((f"p:{rng.randint(0, 9)},0,0", "n:conn", f"j:0,{rng.randint(1, 5)}"))
        elif op == "insert":
            tokens.insert(rng.randrange(len(tokens) + 1), f"p:{rng.randint(0, 9)},0,0")
        elif len(tokens) > 3:
            del tokens[rng.randrange(len(tokens))]
        return " ".join(tokens)
    raise ValueError(f"Unknown encoding '{encoding}'")


def crossover_genotype(
    g1: str, g2: str, encoding: str, rng: random.Random | None = None
) -> tuple[str, str]:
    """Single-point crossover returning two offspring strings."""
    rng = rng or random.Random()
    if encoding == "f1":
        if not g1 or not g2:
            return g1, g2
        cut1 = rng.randint(1, len(g1))
        cut2 = rng.randint(1, len(g2))
        return g1[:cut1] + g2[cut2:], g2[:cut2] + g1[cut1:]
    if encoding == "f0":
        t1 = g1.split(" ") if g1 else []
        t2 = g2.split(" ") if g2 else []
        if not t1 or not t2:
            return g1, g2
        cut1 = rng.randint(1, len(t1))
        cut2 = rng.randint(1, len(t2))
        child1 = " ".join(t1[:cut1] + t2[cut2:])
        child2 = " ".join(t2[:cut2] + t1[cut1:])
        return child1, child2
    raise ValueError(f"Unknown encoding '{encoding}'")

"""
Genotype encoding helpers for Framsticks f0 (low-level) and f1 (high-level) encodings.

These helpers define how random genotypes are generated and how mutation/crossover work
at the *string* level for each encoding, matching the descriptions in Section 2.1 of the
proposal:

  f0 (low-level): direct phenotype serialisation. Genetic ops act on part positions,
                  rotations, and neural connections; crossover splits via a 3D plane cut.
                  We approximate this at the string level as a token sequence where each
                  token encodes one body part / connection.

  f1 (high-level): compact symbolic string; mutation adds/removes symbols; crossover
                    exchanges substrings between parents.

These helpers are used by `RealFramsticksInterface` where it needs raw string ops
(f0/f1 encoding-aware random genotype/mutation/crossover fallbacks), and directly by
`tests/fakes.py` and unit tests that don't need a simulator.
"""
from __future__ import annotations

import random
import string

# f1 alphabet loosely modelled on the Framsticks f1 genetic language: structural symbols
# (branching, parts) and neuron/connection symbols. This is a simplified stand-in used only
# used by RealFramsticksInterface's string-level helpers and by tests/fakes.py -- see
# docstring above.
F1_STRUCTURE_SYMBOLS = list("XRLQ")          # part / rotation / branch symbols
F1_NEURON_SYMBOLS = list("NnGgTt")            # neuron / connection symbols
F1_ALPHABET = F1_STRUCTURE_SYMBOLS + F1_NEURON_SYMBOLS

# f0 is represented as whitespace-separated tokens, each "part(x,y,z);conn(a,b,w)"-style,
# so that Levenshtein / sequence operations remain meaningful at the token-character level.
F0_TOKEN_CHARS = string.ascii_lowercase[:12] + string.digits


def random_f1_genotype(min_len: int = 6, max_len: int = 40, rng: random.Random | None = None) -> str:
    rng = rng or random
    length = rng.randint(min_len, max_len)
    return "".join(rng.choice(F1_ALPHABET) for _ in range(length))


def random_f0_genotype(min_parts: int = 3, max_parts: int = 15, rng: random.Random | None = None) -> str:
    rng = rng or random
    n_parts = rng.randint(min_parts, max_parts)
    tokens = []
    for i in range(n_parts):
        token = "".join(rng.choice(F0_TOKEN_CHARS) for _ in range(8))
        tokens.append(token)
    return " ".join(tokens)


def random_genotype(encoding: str, rng: random.Random | None = None) -> str:
    if encoding == "f1":
        return random_f1_genotype(rng=rng)
    if encoding == "f0":
        return random_f0_genotype(rng=rng)
    raise ValueError(f"Unknown encoding '{encoding}', expected 'f0' or 'f1'")


def mutate_genotype(genotype: str, encoding: str, rate: float = 0.15,
                     rng: random.Random | None = None) -> str:
    """Point mutation appropriate to the encoding: symbol add/remove for f1, token
    perturbation for f0. Returns a NEW genotype string (does not mutate in place)."""
    rng = rng or random
    if encoding == "f1":
        chars = list(genotype)
        out = []
        for c in chars:
            r = rng.random()
            if r < rate * 0.5:
                continue  # deletion
            elif r < rate:
                out.append(rng.choice(F1_ALPHABET))  # substitution
            else:
                out.append(c)
            if rng.random() < rate * 0.2:
                out.append(rng.choice(F1_ALPHABET))  # insertion
        if not out:
            out = [rng.choice(F1_ALPHABET)]
        return "".join(out)

    if encoding == "f0":
        tokens = genotype.split(" ") if genotype else []
        out_tokens = []
        for tok in tokens:
            if rng.random() < rate:
                # perturb one character of the token (position/rotation/connection shift)
                if tok:
                    idx = rng.randrange(len(tok))
                    tok = tok[:idx] + rng.choice(F0_TOKEN_CHARS) + tok[idx + 1:]
            out_tokens.append(tok)
        if rng.random() < rate * 0.3 and out_tokens:
            # occasionally add a part
            new_tok = "".join(rng.choice(F0_TOKEN_CHARS) for _ in range(8))
            out_tokens.insert(rng.randrange(len(out_tokens) + 1), new_tok)
        if rng.random() < rate * 0.2 and len(out_tokens) > 1:
            out_tokens.pop(rng.randrange(len(out_tokens)))
        return " ".join(out_tokens)

    raise ValueError(f"Unknown encoding '{encoding}'")


def crossover_genotype(g1: str, g2: str, encoding: str,
                        rng: random.Random | None = None) -> tuple[str, str]:
    """f1: substring exchange. f0: token-list split ('3D plane cut' analogue)."""
    rng = rng or random
    if encoding == "f1":
        if len(g1) < 2 or len(g2) < 2:
            return g1, g2
        c1 = rng.randrange(1, len(g1))
        c2 = rng.randrange(1, len(g2))
        child1 = g1[:c1] + g2[c2:]
        child2 = g2[:c2] + g1[c1:]
        return child1, child2

    if encoding == "f0":
        t1, t2 = g1.split(" "), g2.split(" ")
        if len(t1) < 2 or len(t2) < 2:
            return g1, g2
        c1 = rng.randrange(1, len(t1))
        c2 = rng.randrange(1, len(t2))
        child1 = " ".join(t1[:c1] + t2[c2:])
        child2 = " ".join(t2[:c2] + t1[c1:])
        return child1, child2

    raise ValueError(f"Unknown encoding '{encoding}'")

"""
FakeFramsticksInterface: a test-only double implementing FramsticksInterface.

This is NOT a production backend -- there isn't one anymore other than
`RealFramsticksInterface`. This file exists purely so `pytest` (and CI, which has no way
to install the real licensed Framsticks simulator) can exercise `src/algorithms/engine.py`'s
actual NSGA-II logic -- non-dominated sorting, crowding distance, the adaptive sigmoid
schedule, budget bookkeeping -- against something deterministic and fast, without needing
the real simulator installed anywhere.

Genotype strings are mapped to a small trait vector (via hashing) that stands in for
body/controller parameters a real simulation would produce, giving a smooth-ish,
non-trivial fitness landscape so the algorithm tests are actually exercising something
(rather than every individual scoring identically). Results from this fake are NOT
meaningful research results and should never appear in a results/ directory or a report --
it's scaffolding for testing our own code, nothing more.
"""
from __future__ import annotations

import hashlib
import math
import random

from src.diversity.gene_dissimilarity import pairwise_gene_dissimilarity
from src.diversity.graph_dissimilarity import pairwise_graph_dissimilarity
from src.framsticks_interface.encodings import (
    crossover_genotype,
    mutate_genotype,
    random_genotype,
)
from src.framsticks_interface.interface import EvaluationResult, FramsticksInterface

TRAIT_DIM = 8

_TASK_TARGETS = {
    "straight_line_cog": [0.8, 0.2, 0.5, 0.5, 0.1, 0.9, 0.3, 0.6],
    "sinusoidal_bounce": [0.2, 0.7, 0.7, 0.3, 0.9, 0.1, 0.5, 0.4],
    "velocity_forward": [0.9, 0.9, 0.1, 0.6, 0.2, 0.4, 0.8, 0.2],
    "jump_height": [0.1, 0.5, 0.9, 0.9, 0.4, 0.6, 0.2, 0.7],
}


def _trait_vector(genotype: str, dim: int = TRAIT_DIM) -> list[float]:
    if not genotype:
        return [0.0] * dim
    digest = hashlib.md5(genotype.encode("utf-8")).digest()
    char_sum = sum(ord(c) for c in genotype)
    traits = []
    for i in range(dim):
        h = digest[i % len(digest)]
        window = genotype[i::dim] if genotype[i::dim] else genotype
        local = sum(ord(c) for c in window) / (len(window) * 256)
        val = (0.6 * local + 0.4 * (h / 255.0) + 0.05 * math.sin(char_sum + i)) % 1.0
        traits.append(val)
    return traits


def _task_target(task: str) -> list[float]:
    if task in _TASK_TARGETS:
        return _TASK_TARGETS[task]
    seed = int(hashlib.md5(task.encode()).hexdigest(), 16) % (2 ** 32)
    rng = random.Random(seed)
    return [rng.random() for _ in range(TRAIT_DIM)]


class FakeFramsticksInterface(FramsticksInterface):
    def __init__(self, seed: int | None = None):
        self.n_evaluations = 0
        self._rng = random.Random(seed)

    def reset_evaluation_budget(self) -> None:
        """Mirrors RealFramsticksInterface's method of the same name -- run_single.py
        calls this before every run so evaluation counts don't leak across repeats."""
        self.n_evaluations = 0

    def get_random_genotype(self, encoding: str) -> str:
        return random_genotype(encoding, rng=self._rng)

    def mutate(self, genotype: str, encoding: str) -> str:
        return mutate_genotype(genotype, encoding, rng=self._rng)

    def crossover(self, g1: str, g2: str, encoding: str) -> tuple[str, str]:
        return crossover_genotype(g1, g2, encoding, rng=self._rng)

    def is_viable(self, genotype: str, encoding: str) -> bool:
        if encoding == "f1":
            has_structure = any(c in "XRLQ" for c in genotype)
            has_neuron = any(c in "NnGgTt" for c in genotype)
            return len(genotype) >= 4 and has_structure and has_neuron
        if encoding == "f0":
            tokens = genotype.split(" ") if genotype else []
            return len(tokens) >= 3
        raise ValueError(f"Unknown encoding '{encoding}'")

    def pairwise_dissimilarity(self, genotypes: list[str], encoding: str, measure: str) -> list[list[float]]:
        if measure == "gene":
            return pairwise_gene_dissimilarity(genotypes)
        if measure == "graph":
            return pairwise_graph_dissimilarity(genotypes, encoding)
        raise ValueError(f"Unknown dissimilarity measure '{measure}'")

    def evaluate(self, genotype: str, encoding: str, task: str, world_config: dict | None = None) -> EvaluationResult:
        self.n_evaluations += 1
        traits = _trait_vector(genotype)
        target = _task_target(task)

        if world_config:
            offset = sum(hash(f"{k}{v}") for k, v in sorted(world_config.items())) % 1000 / 1000.0
            target = [(t + 0.1 * offset) % 1.0 for t in target]

        sq_err = sum((t - x) ** 2 for t, x in zip(traits, target))
        rmse = math.sqrt(sq_err / len(traits))
        fitness = 1.0 / (1.0 + 4.0 * rmse)

        n_steps = 50
        trajectory = []
        for t in range(n_steps):
            phase = t / n_steps
            x = traits[0] * phase + 0.05 * math.sin(2 * math.pi * phase * (1 + traits[2]))
            y = traits[1] * phase + 0.05 * math.cos(2 * math.pi * phase * (1 + traits[3]))
            z = 0.5 + 0.1 * math.sin(phase * 10 * traits[4])
            trajectory.append((x, y, z))

        n_parts = 3 + int(traits[5] * 12)
        is_valid = self.is_viable(genotype, encoding)

        return EvaluationResult(
            fitness=fitness if is_valid else 0.0,
            cog_trajectory=trajectory,
            n_parts=n_parts,
            is_valid=is_valid,
            raw={"traits": traits, "target": target},
        )

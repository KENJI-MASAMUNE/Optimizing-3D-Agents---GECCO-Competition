"""
Shared evolutionary engine underlying all five algorithm variants. Owner: Salmaan Patel

All variants (Section 5.1) share the same NSGA-II skeleton -- non-dominated sorting,
elitist (mu+lambda) selection, binary tournament -- and differ ONLY in how crowding
distance is computed within each front. This module implements that shared skeleton once;
`nsga2_baseline.py`, `de_nsga2.py`, and `adaptive_de_nsga2.py` each configure it with a
different `CrowdingStrategy`, which keeps the diff between variants small and auditable
(a design goal of the "Proof by Demonstration" iterative methodology, Section 5).

Objectives (both maximised): (1) task fitness from the simulator, (2) a diversity score
= each individual's average dissimilarity to the rest of the population. This mirrors
Section 5.1's "Diversity objective: average gene dissimilarity to population" description
for the DE-NSGA-II variants, and is used consistently for the NSGA-II baseline too so that
all five variants are compared on identical objectives -- only the crowding/selection
mechanism differs. (This is a deliberate, documented resolution of the "undefined diversity
score" ambiguity flagged during proposal review.)
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from src.algorithms.crowding import (
    blended_crowding_distance,
    dissimilarity_crowding_distance,
    objective_space_crowding_distance,
)
from src.framsticks_interface.interface import FramsticksInterface

INF = float("inf")


class DiversityMeasure(str, Enum):
    GENE = "gene"
    GRAPH = "graph"
    HYBRID = "hybrid"


class CrowdingStrategy(str, Enum):
    OBJECTIVE = "objective"     # Variant 1: standard NSGA-II crowding
    DISSIMILARITY = "dissimilarity"  # Variants 2/3: DE-NSGA-II crowding (gene or graph)
    HYBRID = "hybrid"           # Variant 4: weighted blend, fixed weight
    ADAPTIVE = "adaptive"        # Variant 5: weighted blend, sigmoid-scheduled weight


@dataclass
class Individual:
    genotype: str
    encoding: str
    fitness: float = 0.0
    diversity_score: float = 0.0
    rank: int = -1
    crowding_dist: float = 0.0
    evaluated: bool = False

    @property
    def objectives(self) -> tuple[float, float]:
        return (self.fitness, self.diversity_score)


@dataclass
class GenerationStats:
    generation: int
    n_evaluations: int
    best_fitness: float
    mean_fitness: float
    mean_diversity: float


@dataclass
class RunResult:
    population: list[Individual]
    history: list[GenerationStats]
    n_evaluations: int
    best_individual: Individual
    convergence_evals: int  # evaluations to reach 90% of final best fitness


def _pairwise_dissimilarity(interface: FramsticksInterface, genotypes: list[str], encoding: str,
                             measure: DiversityMeasure, gene_weight: float = 0.5) -> list[list[float]]:
    """Routes through `interface.pairwise_dissimilarity()` rather than calling
    src/diversity/* directly, so `RealFramsticksInterface` uses Framsticks' OWN
    dissimilarity() (GENE_LEVENSHTEIN / PHENE_STRUCT_GREEDY, per interface.py) instead of
    a Python approximation -- `tests/fakes.py`'s test double implements the same method by
    wrapping src/diversity/* internally, purely so the test suite has something fast and
    dependency-free to exercise this code path against."""
    if measure == DiversityMeasure.GENE:
        return interface.pairwise_dissimilarity(genotypes, encoding, "gene")
    if measure == DiversityMeasure.GRAPH:
        return interface.pairwise_dissimilarity(genotypes, encoding, "graph")
    if measure == DiversityMeasure.HYBRID:
        gene_m = interface.pairwise_dissimilarity(genotypes, encoding, "gene")
        graph_m = interface.pairwise_dissimilarity(genotypes, encoding, "graph")
        n = len(genotypes)
        return [
            [gene_weight * gene_m[i][j] + (1 - gene_weight) * graph_m[i][j] for j in range(n)]
            for i in range(n)
        ]
    raise ValueError(f"Unknown diversity measure {measure}")


def _average_dissimilarity(interface: FramsticksInterface, genotypes: list[str], encoding: str,
                            measure: DiversityMeasure, gene_weight: float = 0.5) -> list[float]:
    """Per-individual average dissimilarity to the rest of the population/front -- the
    diversity objective (engine.py module docstring) and the DE-NSGA-II crowding distance
    formula (Section 2.4) both reduce to this same average-of-a-row computation."""
    n = len(genotypes)
    if n <= 1:
        return [0.0] * n
    matrix = _pairwise_dissimilarity(interface, genotypes, encoding, measure, gene_weight)
    return [sum(row) / (n - 1) for row in matrix]


def fast_non_dominated_sort(individuals: list[Individual]) -> list[list[int]]:
    """Standard fast non-dominated sort (Deb et al., 2002) over `.objectives`
    (both maximised). Returns fronts as lists of indices into `individuals`."""
    n = len(individuals)
    dominates: list[set[int]] = [set() for _ in range(n)]
    domination_count = [0] * n
    fronts: list[list[int]] = [[]]

    def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if _dominates(individuals[i].objectives, individuals[j].objectives):
                dominates[i].add(j)
            elif _dominates(individuals[j].objectives, individuals[i].objectives):
                domination_count[i] += 1
        if domination_count[i] == 0:
            individuals[i].rank = 0
            fronts[0].append(i)

    k = 0
    while fronts[k]:
        next_front = []
        for i in fronts[k]:
            for j in dominates[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    individuals[j].rank = k + 1
                    next_front.append(j)
        k += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


def assign_crowding_distances(
    individuals: list[Individual],
    fronts: list[list[int]],
    strategy: CrowdingStrategy,
    diversity_measure: DiversityMeasure,
    hybrid_weight: float,
    interface: FramsticksInterface,
    adaptive_weight_fn: Callable[[], float] | None = None,
) -> None:
    """Populate `.crowding_dist` on each individual, front by front, using the configured
    strategy. This is the ONLY point of difference between the five algorithm variants."""
    for front in fronts:
        if not front:
            continue
        front_inds = [individuals[i] for i in front]
        objective_values = [ind.objectives for ind in front_inds]

        if strategy == CrowdingStrategy.OBJECTIVE:
            dists = objective_space_crowding_distance(objective_values)

        elif strategy == CrowdingStrategy.DISSIMILARITY:
            genotypes = [ind.genotype for ind in front_inds]
            encoding = front_inds[0].encoding
            diss_matrix = _pairwise_dissimilarity(interface, genotypes, encoding, diversity_measure)
            dists = dissimilarity_crowding_distance(diss_matrix)

        elif strategy in (CrowdingStrategy.HYBRID, CrowdingStrategy.ADAPTIVE):
            genotypes = [ind.genotype for ind in front_inds]
            encoding = front_inds[0].encoding
            diss_matrix = _pairwise_dissimilarity(interface, genotypes, encoding, diversity_measure)
            weight = adaptive_weight_fn() if (strategy == CrowdingStrategy.ADAPTIVE and adaptive_weight_fn) else hybrid_weight
            dists = blended_crowding_distance(objective_values, diss_matrix, gene_weight=weight)

        else:
            raise ValueError(f"Unknown crowding strategy {strategy}")

        for idx, d in zip(front, dists):
            individuals[idx].crowding_dist = d


def _crowded_comparison_key(ind: Individual) -> tuple[int, float]:
    # lower rank better; higher crowding distance better -> sort ascending on (rank, -dist)
    dist = ind.crowding_dist
    neg_dist = -dist if dist != INF else -1e18
    return (ind.rank, neg_dist)


def binary_tournament(pop: list[Individual], rng: random.Random) -> Individual:
    a, b = rng.sample(pop, 2)
    return a if _crowded_comparison_key(a) < _crowded_comparison_key(b) else b


@dataclass
class EngineConfig:
    population_size: int = 100
    n_generations: int = 60
    crossover_prob: float = 0.7
    mutation_prob: float = 0.3
    tournament_size: int = 2
    hybrid_weight: float = 0.5
    adaptive_tau: float = 0.5
    adaptive_steepness: float = 10.0
    eval_budget: int | None = None


def _evaluate(interface: FramsticksInterface, ind: Individual, task: str, world_config: dict | None) -> None:
    if ind.evaluated:
        return
    result = interface.evaluate(ind.genotype, ind.encoding, task, world_config)
    ind.fitness = result.fitness
    ind.evaluated = True


def _make_offspring(interface: FramsticksInterface, parents: list[Individual],
                     encoding: str, config: EngineConfig, rng: random.Random) -> list[Individual]:
    offspring = []
    while len(offspring) < config.population_size:
        p1, p2 = binary_tournament(parents, rng), binary_tournament(parents, rng)
        g1, g2 = p1.genotype, p2.genotype
        if rng.random() < config.crossover_prob:
            g1, g2 = interface.crossover(g1, g2, encoding)
        if rng.random() < config.mutation_prob:
            g1 = interface.mutate(g1, encoding)
        if rng.random() < config.mutation_prob:
            g2 = interface.mutate(g2, encoding)

        for g in (g1, g2):
            if len(offspring) >= config.population_size:
                break
            # Minimal-Viable Morphological Filter (Section 5.5): resample non-viable
            # genotypes rather than letting them consume simulation budget.
            tries = 0
            while not interface.is_viable(g, encoding) and tries < 5:
                g = interface.get_random_genotype(encoding)
                tries += 1
            offspring.append(Individual(genotype=g, encoding=encoding))
    return offspring


def run_evolution(
    interface: FramsticksInterface,
    encoding: str,
    task: str,
    crowding_strategy: CrowdingStrategy,
    diversity_measure: DiversityMeasure,
    config: EngineConfig,
    seed: int,
    world_config: dict | None = None,
) -> RunResult:
    """Runs one full (variant, encoding, task) evolutionary run and returns the final
    population plus per-generation statistics. This is the single function every variant
    module (nsga2_baseline / de_nsga2 / adaptive_de_nsga2) calls, differing only in the
    `crowding_strategy` (and, for Adaptive, in supplying `adaptive_weight_fn`)."""
    rng = random.Random(seed)

    population = [
        Individual(genotype=interface.get_random_genotype(encoding), encoding=encoding)
        for _ in range(config.population_size)
    ]
    # Bootstrap fix (Section 5.5): resample non-viable initial genotypes.
    for ind in population:
        tries = 0
        while not interface.is_viable(ind.genotype, encoding) and tries < 5:
            ind.genotype = interface.get_random_genotype(encoding)
            tries += 1

    history: list[GenerationStats] = []
    n_evaluations = 0
    best_fitness_so_far = float("-inf")
    fitness_at_eval: list[tuple[int, float]] = []  # (n_evaluations, best_fitness_so_far)

    def adaptive_weight_fn_factory(gen: int) -> Callable[[], float]:
        import math
        progress = gen / max(config.n_generations - 1, 1)
        tau = config.adaptive_tau
        k = config.adaptive_steepness
        # sigmoid decay from ~1 (favour dissimilarity/exploration) to ~0 (favour
        # objective-space/exploitation) as training progresses past tau -- Section 5.1,
        # Variant 5 "Adaptive Sigmoidal Selection".
        weight = 1.0 / (1.0 + math.exp(k * (progress - tau)))
        return lambda: weight

    for gen in range(config.n_generations):
        for ind in population:
            _evaluate(interface, ind, task, world_config)
        n_evaluations = interface.n_evaluations if hasattr(interface, "n_evaluations") else n_evaluations + len(population)

        genotypes = [ind.genotype for ind in population]
        div_scores = _average_dissimilarity(interface, genotypes, encoding, diversity_measure)
        for ind, score in zip(population, div_scores):
            ind.diversity_score = score

        fronts = fast_non_dominated_sort(population)
        adaptive_fn = adaptive_weight_fn_factory(gen) if crowding_strategy == CrowdingStrategy.ADAPTIVE else None
        assign_crowding_distances(
            population, fronts, crowding_strategy, diversity_measure,
            hybrid_weight=config.hybrid_weight, interface=interface, adaptive_weight_fn=adaptive_fn,
        )

        gen_best = max(population, key=lambda i: i.fitness)
        if gen_best.fitness > best_fitness_so_far:
            best_fitness_so_far = gen_best.fitness
        fitness_at_eval.append((n_evaluations, best_fitness_so_far))

        history.append(GenerationStats(
            generation=gen,
            n_evaluations=n_evaluations,
            best_fitness=best_fitness_so_far,
            mean_fitness=sum(i.fitness for i in population) / len(population),
            mean_diversity=sum(i.diversity_score for i in population) / len(population),
        ))

        if config.eval_budget and n_evaluations >= config.eval_budget:
            break
        if gen == config.n_generations - 1:
            break

        offspring = _make_offspring(interface, population, encoding, config, rng)
        for ind in offspring:
            _evaluate(interface, ind, task, world_config)
        n_evaluations = interface.n_evaluations if hasattr(interface, "n_evaluations") else n_evaluations + len(offspring)

        combined = population + offspring
        genotypes_c = [ind.genotype for ind in combined]
        div_scores_c = _average_dissimilarity(interface, genotypes_c, encoding, diversity_measure)
        for ind, score in zip(combined, div_scores_c):
            ind.diversity_score = score

        fronts_c = fast_non_dominated_sort(combined)
        assign_crowding_distances(
            combined, fronts_c, crowding_strategy, diversity_measure,
            hybrid_weight=config.hybrid_weight, interface=interface, adaptive_weight_fn=adaptive_fn,
        )

        # elitist replacement: fill next generation front-by-front
        next_pop: list[Individual] = []
        for front in fronts_c:
            if len(next_pop) + len(front) <= config.population_size:
                next_pop.extend(combined[i] for i in front)
            else:
                remaining = config.population_size - len(next_pop)
                front_sorted = sorted(front, key=lambda i: _crowded_comparison_key(combined[i]))
                next_pop.extend(combined[i] for i in front_sorted[:remaining])
                break
        population = next_pop

    best_individual = max(population, key=lambda i: i.fitness)
    final_best = best_individual.fitness
    threshold = 0.9 * final_best if final_best > 0 else 0.0
    convergence_evals = n_evaluations
    for evals, fit in fitness_at_eval:
        if fit >= threshold:
            convergence_evals = evals
            break

    return RunResult(
        population=population,
        history=history,
        n_evaluations=n_evaluations,
        best_individual=best_individual,
        convergence_evals=convergence_evals,
    )

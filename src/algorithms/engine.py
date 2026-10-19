"""
Shared evolutionary engine underlying all five algorithm variants

All variants (Section 5.1) share the same NSGA-II skeleton -- non-dominated sorting,
elitist (mu+lambda) replacement, crowded tournament selection -- and differ ONLY in how
crowding distance is computed within each front. This module implements that skeleton
once; `nsga2_baseline.py`, `de_nsga2.py`, and `adaptive_de_nsga2.py` each configure it with
a different `CrowdingStrategy` / `DiversityMeasure`.

Objectives (both maximised): (1) task fitness from the simulator, (2) a diversity score
= each individual's average dissimilarity to the rest of the population (Section 5.1,
"Diversity objective: average gene dissimilarity to population"), used for all five
variants so that only the crowding/selection mechanism differs.

Variant -> (crowding strategy, dissimilarity measure feeding crowding AND the diversity objective)
    1 nsga2_baseline     OBJECTIVE      GENE
    2 de_nsga2_gene      DISSIMILARITY  GENE
    3 de_nsga2_graph     DISSIMILARITY  GRAPH
    4 de_nsga2_hybrid    DISSIMILARITY  HYBRID  = w*gene_norm + (1-w)*graph_norm, w = config.hybrid_weight
    5 adaptive_de_nsga2  ADAPTIVE       GENE    crowding = s(t)*dissim_norm + (1-s(t))*objective_norm,
                                                s(t) = 1/(1+exp(k(t-tau))) sigmoid schedule

Changes in this revision (each was a real bug found during the Oct 2026 audit):
* `tournament_size` is now honoured (it was ignored -> Phase 2's tournament sweep was a no-op).
* Variant 4 now matches the proposal ("crowding = weighted combination of gene and graph
  dissimilarities, weight tuned per encoding"). Previously it blended a (gene+graph) matrix
  with OBJECTIVE crowding using hybrid_weight, while the gene/graph weight was hard-coded to
  0.5 -- so Phase 2's "gene weight" sweep actually swept something else. Gene and graph
  matrices are now each normalised to [0, 1] before blending (raw Levenshtein distances are
  ~10-100x larger than graph distances, so an unnormalised blend was effectively gene-only).
* Best-ever tracking: dissimilarity crowding does not give boundary solutions infinite
  distance, so DE variants could DROP the best individual during truncation. The run now
  reports the best individual ever evaluated (what the competition scores), not just the
  best in the final population.
* Invalid evaluations (genotypes that cannot be built/simulated in the current world, e.g.
  body-constraint violations in Phase 3) are handled by constraint-domination: they are
  ranked behind every valid individual instead of receiving fitness 0.0 (which beat valid
  creatures with negative velocity_forward fitness).
* Budget exhaustion (interface raises BudgetExhausted) ends the run gracefully.
* Each selection step computes the dissimilarity matrix ONCE for the combined population
  and reuses sub-matrices for per-front crowding (previously recomputed per front --
  roughly doubling graph-dissimilarity cost).
* Per-run fitness cache: evaluation is deterministic (deterministic.sim), so re-evaluating a
  genotype that has already been simulated in this run would only waste budget. Offspring
  identical to an evaluated genotype reuse its fitness (`EngineConfig.cache_fitness`).
* Convergence is reported two ways: the proposal's definition (evaluations to reach 90% of
  the final best fitness) and an improvement-based one (evaluations to achieve 90% of the
  run's total improvement over the initial best). With fitness of the form 1/(1+error) the
  initial population already sits at >90% of the final value, which makes the first
  definition report "generation 0" for almost every run.
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

from src.algorithms.crowding import (
    blended_crowding_distance,
    dissimilarity_crowding_distance,
    objective_space_crowding_distance,
)
from src.framsticks_interface.interface import BudgetExhausted, FramsticksInterface

INF = float("inf")


class DiversityMeasure(str, Enum):
    GENE = "gene"
    GRAPH = "graph"
    HYBRID = "hybrid"


class CrowdingStrategy(str, Enum):
    OBJECTIVE = "objective"          # Variant 1: standard NSGA-II crowding
    DISSIMILARITY = "dissimilarity"  # Variants 2/3/4: DE-NSGA-II crowding (gene / graph / hybrid)
    HYBRID = "hybrid"                # legacy: fixed-weight dissimilarity/objective blend (not used by any variant)
    ADAPTIVE = "adaptive"            # Variant 5: sigmoid-scheduled dissimilarity/objective blend


@dataclass
class Individual:
    genotype: str
    encoding: str
    fitness: float = 0.0
    diversity_score: float = 0.0
    rank: int = -1
    crowding_dist: float = 0.0
    evaluated: bool = False
    valid: bool = True

    @property
    def objectives(self) -> tuple[float, float]:
        return (self.fitness, self.diversity_score)

    def copy(self) -> "Individual":
        return Individual(self.genotype, self.encoding, self.fitness, self.diversity_score,
                          self.rank, self.crowding_dist, self.evaluated, self.valid)


@dataclass
class GenerationStats:
    generation: int
    n_evaluations: int
    best_fitness: float          # best-ever so far
    mean_fitness: float          # over valid individuals of the current population
    mean_diversity: float
    n_valid: int = 0
    crowding_weight: float | None = None  # adaptive schedule value (Variant 5), else None
    elapsed_s: float = 0.0


@dataclass
class RunResult:
    population: list[Individual]
    history: list[GenerationStats]
    n_evaluations: int
    best_individual: Individual
    convergence_evals: int              # proposal definition: evals to reach 90% of final best
    convergence_evals_improvement: int = 0  # evals to achieve 90% of the run's improvement
    stopped_by_budget: bool = False
    n_cache_hits: int = 0
    fitness_trace: list[tuple[int, float]] = field(default_factory=list)  # (evals, best-so-far) at each improvement


# ---------------------------------------------------------------------------- dissimilarity
def _normalise_matrix(m: list[list[float]]) -> list[list[float]]:
    mx = max((v for row in m for v in row), default=0.0)
    if mx <= 0:
        return [[0.0] * len(row) for row in m]
    return [[v / mx for v in row] for row in m]


def _pairwise_dissimilarity(interface: FramsticksInterface, genotypes: list[str], encoding: str,
                            measure: DiversityMeasure, gene_weight: float = 0.5) -> list[list[float]]:
    """Routes through `interface.pairwise_dissimilarity()` so the real backend uses
    Framsticks' own measures. HYBRID = gene_weight * gene/max(gene) + (1-gene_weight) *
    graph/max(graph), normalised over the matrix being computed (the current combined
    population), so the weight means what it says regardless of the raw scales."""
    if measure == DiversityMeasure.GENE:
        return interface.pairwise_dissimilarity(genotypes, encoding, "gene")
    if measure == DiversityMeasure.GRAPH:
        return interface.pairwise_dissimilarity(genotypes, encoding, "graph")
    if measure == DiversityMeasure.HYBRID:
        n = len(genotypes)
        gene_m = _normalise_matrix(interface.pairwise_dissimilarity(genotypes, encoding, "gene")) if gene_weight > 0 else [[0.0] * n for _ in range(n)]
        graph_m = _normalise_matrix(interface.pairwise_dissimilarity(genotypes, encoding, "graph")) if gene_weight < 1 else [[0.0] * n for _ in range(n)]
        return [[gene_weight * gene_m[i][j] + (1 - gene_weight) * graph_m[i][j] for j in range(n)]
                for i in range(n)]
    raise ValueError(f"Unknown diversity measure {measure}")


def _row_means(matrix: list[list[float]]) -> list[float]:
    n = len(matrix)
    if n <= 1:
        return [0.0] * n
    return [sum(row) / (n - 1) for row in matrix]


def _average_dissimilarity(interface: FramsticksInterface, genotypes: list[str], encoding: str,
                           measure: DiversityMeasure, gene_weight: float = 0.5) -> list[float]:
    """Per-individual average dissimilarity to the rest of the population (diversity objective)."""
    if len(genotypes) <= 1:
        return [0.0] * len(genotypes)
    return _row_means(_pairwise_dissimilarity(interface, genotypes, encoding, measure, gene_weight))


def _submatrix(matrix: list[list[float]], idx: list[int]) -> list[list[float]]:
    return [[matrix[i][j] for j in idx] for i in idx]


# ---------------------------------------------------------------------------- sorting
def fast_non_dominated_sort(individuals: list[Individual]) -> list[list[int]]:
    """Fast non-dominated sort (Deb et al., 2002) over `.objectives` (both maximised), with
    constraint-domination: every invalid individual is ranked after all valid ones (they
    form the final front together). Returns fronts as lists of indices."""
    valid_idx = [i for i, ind in enumerate(individuals) if ind.valid]
    invalid_idx = [i for i, ind in enumerate(individuals) if not ind.valid]
    n = len(valid_idx)
    dominates: list[list[int]] = [[] for _ in range(n)]
    domination_count = [0] * n
    fronts_local: list[list[int]] = [[]]

    objs = [individuals[i].objectives for i in valid_idx]

    def _dominates(a: tuple[float, float], b: tuple[float, float]) -> bool:
        return all(x >= y for x, y in zip(a, b)) and any(x > y for x, y in zip(a, b))

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objs[p], objs[q]):
                dominates[p].append(q)
            elif _dominates(objs[q], objs[p]):
                domination_count[p] += 1
        if domination_count[p] == 0:
            fronts_local[0].append(p)

    k = 0
    while fronts_local[k]:
        nxt = []
        for p in fronts_local[k]:
            for q in dominates[p]:
                domination_count[q] -= 1
                if domination_count[q] == 0:
                    nxt.append(q)
        k += 1
        fronts_local.append(nxt)

    fronts = [[valid_idx[p] for p in f] for f in fronts_local if f]
    if invalid_idx:
        fronts.append(invalid_idx)
    for rank, front in enumerate(fronts):
        for i in front:
            individuals[i].rank = rank
    return fronts


def assign_crowding_distances(
    individuals: list[Individual],
    fronts: list[list[int]],
    strategy: CrowdingStrategy,
    diversity_measure: DiversityMeasure,
    hybrid_weight: float,
    interface: FramsticksInterface,
    adaptive_weight_fn: Callable[[], float] | None = None,
    diss_matrix: list[list[float]] | None = None,
) -> None:
    """Populate `.crowding_dist` front by front. This is the ONLY point of difference
    between the five algorithm variants. `diss_matrix` (optional) is the precomputed
    dissimilarity matrix over ALL `individuals`; per-front sub-matrices are taken from it."""
    needs_diss = strategy != CrowdingStrategy.OBJECTIVE
    if needs_diss and diss_matrix is None:
        genotypes = [ind.genotype for ind in individuals]
        enc = individuals[0].encoding if individuals else "f1"
        diss_matrix = _pairwise_dissimilarity(interface, genotypes, enc, diversity_measure, hybrid_weight)

    for front in fronts:
        if not front:
            continue
        front_inds = [individuals[i] for i in front]
        if not front_inds[0].valid:  # constraint-violating front: no crowding preference
            for idx in front:
                individuals[idx].crowding_dist = 0.0
            continue
        objective_values = [ind.objectives for ind in front_inds]

        if strategy == CrowdingStrategy.OBJECTIVE:
            dists = objective_space_crowding_distance(objective_values)
        elif strategy == CrowdingStrategy.DISSIMILARITY:
            dists = dissimilarity_crowding_distance(_submatrix(diss_matrix, front))
        elif strategy in (CrowdingStrategy.HYBRID, CrowdingStrategy.ADAPTIVE):
            weight = adaptive_weight_fn() if (strategy == CrowdingStrategy.ADAPTIVE and adaptive_weight_fn) else hybrid_weight
            dists = blended_crowding_distance(objective_values, _submatrix(diss_matrix, front), gene_weight=weight)
        else:
            raise ValueError(f"Unknown crowding strategy {strategy}")

        for idx, d in zip(front, dists):
            individuals[idx].crowding_dist = d


def _crowded_comparison_key(ind: Individual) -> tuple[int, float]:
    # lower rank better; higher crowding distance better -> sort ascending on (rank, -dist)
    dist = ind.crowding_dist
    neg_dist = -dist if dist != INF else -1e18
    return (ind.rank, neg_dist)


def crowded_tournament(pop: list[Individual], rng: random.Random, k: int = 2) -> Individual:
    """k-way crowded-comparison tournament (k=2 is the classic NSGA-II binary tournament)."""
    k = max(2, min(k, len(pop)))
    contestants = rng.sample(pop, k)
    return min(contestants, key=_crowded_comparison_key)


def binary_tournament(pop: list[Individual], rng: random.Random) -> Individual:
    return crowded_tournament(pop, rng, 2)


# ---------------------------------------------------------------------------- config
@dataclass
class EngineConfig:
    population_size: int = 100
    n_generations: int = 60
    crossover_prob: float = 0.7
    mutation_prob: float = 0.3
    tournament_size: int = 2
    hybrid_weight: float = 0.5        # Variant 4: weight of GENE dissimilarity in the gene/graph blend
    adaptive_tau: float = 0.5         # Variant 5: sigmoid midpoint (fraction of the run)
    adaptive_steepness: float = 10.0  # Variant 5: sigmoid steepness k
    eval_budget: int | None = None
    cache_fitness: bool = True
    verbose: bool = True


def sigmoid_schedule(progress: float, tau: float, steepness: float) -> float:
    """Weight on dissimilarity crowding: ~1 early (explore), ~0 late (exploit)."""
    return 1.0 / (1.0 + math.exp(steepness * (progress - tau)))


# ---------------------------------------------------------------------------- run
class _RunState:
    def __init__(self):
        self.best: Individual | None = None
        self.trace: list[tuple[int, float]] = []
        self.cache: dict[str, tuple[float, bool]] = {}
        self.cache_hits = 0
        self.initial_best: float | None = None


def _n_evals(interface: FramsticksInterface, fallback: int) -> int:
    return interface.n_evaluations if hasattr(interface, "n_evaluations") else fallback


def _evaluate(interface: FramsticksInterface, ind: Individual, task: str, world_config,
              state: _RunState | None = None, use_cache: bool = False) -> None:
    if ind.evaluated:
        return
    if use_cache and state is not None and ind.genotype in state.cache:
        ind.fitness, ind.valid = state.cache[ind.genotype]
        ind.evaluated = True
        state.cache_hits += 1
        return
    result = interface.evaluate(ind.genotype, ind.encoding, task, world_config)
    valid = bool(result.is_valid) and result.fitness is not None and not math.isnan(result.fitness)
    ind.fitness = float(result.fitness) if valid else 0.0
    ind.valid = valid
    ind.evaluated = True
    if state is not None:
        if use_cache:
            state.cache[ind.genotype] = (ind.fitness, ind.valid)
        if valid and (state.best is None or ind.fitness > state.best.fitness):
            state.best = ind.copy()
            state.trace.append((_n_evals(interface, len(state.trace) + 1), ind.fitness))


def _make_offspring(interface: FramsticksInterface, parents: list[Individual],
                    encoding: str, config: EngineConfig, rng: random.Random) -> list[Individual]:
    offspring = []
    while len(offspring) < config.population_size:
        p1 = crowded_tournament(parents, rng, config.tournament_size)
        p2 = crowded_tournament(parents, rng, config.tournament_size)
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


def _select_survivors(combined: list[Individual], fronts: list[list[int]], size: int) -> list[Individual]:
    next_pop: list[Individual] = []
    for front in fronts:
        if len(next_pop) + len(front) <= size:
            next_pop.extend(combined[i] for i in front)
        else:
            remaining = size - len(next_pop)
            front_sorted = sorted(front, key=lambda i: _crowded_comparison_key(combined[i]))
            next_pop.extend(combined[i] for i in front_sorted[:remaining])
            break
    return next_pop


def run_evolution(
    interface: FramsticksInterface,
    encoding: str,
    task: str,
    crowding_strategy: CrowdingStrategy,
    diversity_measure: DiversityMeasure,
    config: EngineConfig,
    seed: int,
    world_config=None,
) -> RunResult:
    """Runs one full (variant, encoding, task[, world]) evolutionary run."""
    rng = random.Random(seed)
    state = _RunState()
    run_start = time.perf_counter()
    stopped_by_budget = False
    history: list[GenerationStats] = []

    def weight_for(gen: int) -> float | None:
        if crowding_strategy != CrowdingStrategy.ADAPTIVE:
            return None
        progress = gen / max(config.n_generations - 1, 1)
        return sigmoid_schedule(progress, config.adaptive_tau, config.adaptive_steepness)

    def score_and_crowd(pop: list[Individual], gen: int) -> list[list[int]]:
        genotypes = [ind.genotype for ind in pop]
        diss = _pairwise_dissimilarity(interface, genotypes, encoding, diversity_measure, config.hybrid_weight) \
            if len(pop) > 1 else [[0.0]]
        for ind, score in zip(pop, _row_means(diss)):
            ind.diversity_score = score
        fronts = fast_non_dominated_sort(pop)
        w = weight_for(gen)
        assign_crowding_distances(pop, fronts, crowding_strategy, diversity_measure,
                                  hybrid_weight=config.hybrid_weight, interface=interface,
                                  adaptive_weight_fn=(lambda: w) if w is not None else None,
                                  diss_matrix=diss)
        return fronts

    # ---- initial population (bootstrap filter, Section 5.5)
    population = []
    for _ in range(config.population_size):
        g = interface.get_random_genotype(encoding)
        tries = 0
        while not interface.is_viable(g, encoding) and tries < 5:
            g = interface.get_random_genotype(encoding)
            tries += 1
        population.append(Individual(genotype=g, encoding=encoding))

    n_evaluations = 0
    for gen in range(config.n_generations):
        # ---- evaluate current population (only generation 0 has unevaluated members)
        try:
            for ind in population:
                _evaluate(interface, ind, task, world_config, state, config.cache_fitness)
        except BudgetExhausted:
            stopped_by_budget = True
            population = [ind for ind in population if ind.evaluated] or population[:1]
        n_evaluations = _n_evals(interface, n_evaluations + len(population))
        if gen == 0:
            state.initial_best = state.best.fitness if state.best else None

        score_and_crowd(population, gen)
        valid = [i for i in population if i.valid]
        history.append(GenerationStats(
            generation=gen,
            n_evaluations=n_evaluations,
            best_fitness=state.best.fitness if state.best else float("nan"),
            mean_fitness=(sum(i.fitness for i in valid) / len(valid)) if valid else float("nan"),
            mean_diversity=sum(i.diversity_score for i in population) / len(population),
            n_valid=len(valid),
            crowding_weight=weight_for(gen),
            elapsed_s=time.perf_counter() - run_start,
        ))
        if config.verbose:
            elapsed = history[-1].elapsed_s
            eta = elapsed / (gen + 1) * (config.n_generations - gen - 1)
            fmt = lambda s: time.strftime("%H:%M:%S", time.gmtime(s))  # noqa: E731
            print(f"[gen {gen + 1}/{config.n_generations}] evals={n_evaluations} "
                  f"best={history[-1].best_fitness:.4f} mean={history[-1].mean_fitness:.4f} "
                  f"valid={len(valid)}/{len(population)} elapsed={fmt(elapsed)} ETA={fmt(eta)}",
                  flush=True)

        if stopped_by_budget or gen == config.n_generations - 1:
            break
        if config.eval_budget and n_evaluations >= config.eval_budget:
            stopped_by_budget = True
            break

        # ---- variation + evaluation of offspring
        offspring = _make_offspring(interface, population, encoding, config, rng)
        try:
            for ind in offspring:
                _evaluate(interface, ind, task, world_config, state, config.cache_fitness)
        except BudgetExhausted:
            stopped_by_budget = True
        offspring = [ind for ind in offspring if ind.evaluated]
        n_evaluations = _n_evals(interface, n_evaluations + len(offspring))

        # ---- elitist (mu + lambda) replacement
        combined = population + offspring
        fronts_c = score_and_crowd(combined, gen)
        population = _select_survivors(combined, fronts_c, config.population_size)
        if stopped_by_budget:
            break

    if state.best is None:  # nothing valid was ever evaluated
        best_individual = max(population, key=lambda i: i.fitness).copy()
        final_best = best_individual.fitness
    else:
        best_individual = state.best
        final_best = best_individual.fitness

    # proposal definition: evals to reach 90% of final best
    threshold = 0.9 * final_best if final_best > 0 else final_best
    convergence_evals = n_evaluations
    for evals, fit in state.trace:
        if fit >= threshold:
            convergence_evals = evals
            break
    # improvement-based definition
    conv_impr = 0
    init = state.initial_best
    if init is not None and final_best > init:
        target = init + 0.9 * (final_best - init)
        conv_impr = n_evaluations
        for evals, fit in state.trace:
            if fit >= target:
                conv_impr = evals
                break

    return RunResult(
        population=population,
        history=history,
        n_evaluations=n_evaluations,
        best_individual=best_individual,
        convergence_evals=convergence_evals,
        convergence_evals_improvement=conv_impr,
        stopped_by_budget=stopped_by_budget,
        n_cache_hits=state.cache_hits,
        fitness_trace=state.trace,
    )

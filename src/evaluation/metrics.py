"""
Evaluation metrics, matching the table in Section 5.4 of the proposal.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.algorithms.engine import Individual, RunResult
from src.framsticks_interface.interface import FramsticksInterface


@dataclass
class RunMetrics:
    mean_best_fitness: float
    gene_diversity: float
    graph_diversity: float
    convergence_evals: int
    generalisation_score: float | None = None


def top_k(population: list[Individual], k: int = 20) -> list[Individual]:
    return sorted(population, key=lambda i: i.fitness, reverse=True)[:k]


def mean_best_fitness(run_results: list[RunResult]) -> float:
    """Average of best fitness across N runs of one (variant, encoding, task) setting."""
    if not run_results:
        return 0.0
    return sum(r.best_individual.fitness for r in run_results) / len(run_results)


def _average_row(matrix: list[list[float]]) -> float:
    n = len(matrix)
    if n <= 1:
        return 0.0
    per_individual = [sum(row) / (n - 1) for row in matrix]
    return sum(per_individual) / len(per_individual)


def gene_diversity_of_best(interface: FramsticksInterface, run_result: RunResult,
                            encoding: str, k: int = 20) -> float:
    """Average gene dissimilarity among the k best genotypes in the final population.
    Routed through `interface.pairwise_dissimilarity()` (rather than calling
    src/diversity/* directly) so a real Framsticks backend reports diversity using
    Framsticks' own GENE_LEVENSHTEIN measure, matching what engine.py now uses internally
    for the diversity objective and DE-NSGA-II crowding distance."""
    best = top_k(run_result.population, k)
    genotypes = [ind.genotype for ind in best]
    if len(genotypes) <= 1:
        return 0.0
    matrix = interface.pairwise_dissimilarity(genotypes, encoding, "gene")
    return _average_row(matrix)


def graph_diversity_of_best(interface: FramsticksInterface, run_result: RunResult,
                             encoding: str, k: int = 20) -> float:
    """Average graph dissimilarity among the k best phenotypic structures. See
    `gene_diversity_of_best` docstring for why this goes through the interface."""
    best = top_k(run_result.population, k)
    genotypes = [ind.genotype for ind in best]
    if len(genotypes) <= 1:
        return 0.0
    matrix = interface.pairwise_dissimilarity(genotypes, encoding, "graph")
    return _average_row(matrix)


def convergence_rate(run_result: RunResult) -> int:
    """Number of evaluations to reach 90% of the final best fitness (already tracked
    during the run by the engine; exposed here for reporting consistency)."""
    return run_result.convergence_evals


def normalised_score(setting_scores: dict[str, float]) -> dict[str, float]:
    """Competition-style normalisation across all submissions/variants for one setting:
    each score is divided by the maximum score achieved by any variant on that setting,
    matching Section 5.4's 'Normalised Score' metric (normalisation across submissions)."""
    if not setting_scores:
        return {}
    max_score = max(setting_scores.values()) or 1.0
    return {k: v / max_score for k, v in setting_scores.items()}


def generalisation_score(interface: FramsticksInterface, encoding: str, task: str, individual: Individual,
                          held_out_world_configs: list[dict]) -> float:
    """Average fitness of one individual re-evaluated on held-out .sim environments not
    seen during training (Section 5.4). Used in Phase 3 clean-environment evaluation."""
    if not held_out_world_configs:
        return individual.fitness
    fitnesses = []
    for wc in held_out_world_configs:
        result = interface.evaluate(individual.genotype, encoding, task, wc)
        fitnesses.append(result.fitness)
    return sum(fitnesses) / len(fitnesses)


def summarise_run(interface: FramsticksInterface, run_result: RunResult, encoding: str) -> RunMetrics:
    return RunMetrics(
        mean_best_fitness=run_result.best_individual.fitness,
        gene_diversity=gene_diversity_of_best(interface, run_result, encoding),
        graph_diversity=graph_diversity_of_best(interface, run_result, encoding),
        convergence_evals=convergence_rate(run_result),
    )

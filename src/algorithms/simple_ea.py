"""
Organisers' baseline: the simple generational EA from framspy's FramsticksEvolution.py
(DEAP `algorithms.eaSimple`), re-implemented here without the DEAP dependency so it runs
through the same interface, budget, seeding, cache and logging as our five variants.

This is the "Simple EA" that the GECCO Automated Design Competition normalises every entry
against (Simple EA = 0 in the 2026 results slides), and the `eaSimpleF0/F1` baseline in
the 2026 winner's thesis (Brassat 2026, Table 5). Defaults are FramsticksEvolution.py's:

  population 50, tournament size 5, mutation probability 0.9, crossover probability 0.2,
  initial population = 50 copies of the simplest genotype of the encoding
  (frams GenMan.getSimplest: "X" for f1, a single stick for f0).

One generation, exactly as DEAP eaSimple + varAnd:
  1. offspring = tournament selection (size 5) of len(pop) individuals, drawn only from
     feasible individuals (framspy's selTournament_only_feasible),
  2. copy; for each consecutive pair (0,1), (2,3), ...: crossover with probability pxov,
  3. every individual is mutated with probability pmut,
  4. only individuals that were changed by 2-3 are (re)evaluated; unchanged copies keep
     their fitness,
  5. the offspring replace the population entirely (no elitism; the best-ever creature is
     kept separately, like DEAP's HallOfFame, and is what the run reports).

Differences from the original, applied identically to all algorithms in this project:
  * genotypes failing `interface.is_viable()` (invalid or over the parts/joints/neurons/
    length caps) are marked infeasible without simulating them, so they cost no budget;
  * the per-run fitness cache (deterministic simulator) re-uses the fitness of a genotype
    already simulated in this run.

Budget: to give the same amount of search as the NSGA-II variants, the run creates the same
number of offspring (`births = population_size x n_generations` of the main GA settings,
30,000 by default), i.e. 600 generations of 50.
"""
from __future__ import annotations

import math
import random
import time

from src.algorithms.engine import (
    EngineConfig, GenerationStats, Individual, RunResult, _evaluate, _n_evals, _RunState,
)
from src.framsticks_interface.interface import BudgetExhausted, FramsticksInterface

VARIANT_NAME = "simple_ea"

DEFAULTS = dict(population_size=50, tournament_size=5, mutation_prob=0.9, crossover_prob=0.2)
DIVERSITY_EVERY = 10  # log population gene diversity every N generations (logging only)


def _simplest(interface: FramsticksInterface, encoding: str) -> str:
    if hasattr(interface, "get_simplest_genotype"):
        return interface.get_simplest_genotype(encoding)
    return interface.get_random_genotype(encoding)  # test doubles without a "simplest" notion


def _tournament_feasible(pop: list[Individual], k: int, tournsize: int, rng: random.Random) -> list[Individual]:
    """DEAP selTournament: each winner is the fittest of `tournsize` aspirants drawn WITH
    replacement (tools.selRandom), here from feasible individuals only."""
    feasible = [i for i in pop if i.valid and i.evaluated] or pop
    return [max((rng.choice(feasible) for _ in range(tournsize)), key=lambda i: i.fitness)
            for _ in range(k)]


def _gene_diversity(interface: FramsticksInterface, pop: list[Individual], encoding: str) -> float:
    g = [i.genotype for i in pop]
    if len(g) < 2:
        return 0.0
    m = interface.pairwise_dissimilarity(g, encoding, "gene")
    n = len(g)
    return sum(sum(row) for row in m) / (n * (n - 1))


def run(interface: FramsticksInterface, encoding: str, task: str, config: EngineConfig,
        seed: int, world_config=None) -> RunResult:
    rng = random.Random(seed)
    state = _RunState()
    t0 = time.perf_counter()
    pop_size = config.population_size
    history: list[GenerationStats] = []
    stopped = False

    def evaluate_all(inds: list[Individual]) -> None:
        for ind in inds:
            if ind.evaluated:
                continue
            if not interface.is_viable(ind.genotype, encoding):
                ind.valid, ind.evaluated, ind.fitness = False, True, 0.0
                continue
            _evaluate(interface, ind, task, world_config, state, config.cache_fitness)

    def log(gen: int, pop: list[Individual], n_evals: int) -> None:
        valid = [i for i in pop if i.valid and i.evaluated]
        div = _gene_diversity(interface, pop, encoding) if (gen % DIVERSITY_EVERY == 0 or gen == config.n_generations - 1) else float("nan")
        history.append(GenerationStats(
            generation=gen, n_evaluations=n_evals,
            best_fitness=state.best.fitness if state.best else float("nan"),
            mean_fitness=(sum(i.fitness for i in valid) / len(valid)) if valid else float("nan"),
            mean_diversity=div, n_valid=len(valid), crowding_weight=None,
            elapsed_s=time.perf_counter() - t0))
        if config.verbose and (gen % 25 == 0 or gen == config.n_generations - 1):
            print(f"[simple_ea gen {gen + 1}/{config.n_generations}] evals={n_evals} best={history[-1].best_fitness:.4f} "
                  f"valid={len(valid)}/{len(pop)}", flush=True)

    simplest = _simplest(interface, encoding)
    population = [Individual(genotype=simplest, encoding=encoding) for _ in range(pop_size)]
    try:
        evaluate_all(population)
    except BudgetExhausted:
        stopped = True
    n_evals = _n_evals(interface, 0)
    state.initial_best = state.best.fitness if state.best else None
    log(0, population, n_evals)

    for gen in range(1, config.n_generations):
        if stopped:
            break
        parents = _tournament_feasible(population, pop_size, config.tournament_size, rng)
        offspring = [p.copy() for p in parents]
        for i in range(1, len(offspring), 2):
            if rng.random() < config.crossover_prob:
                a, b = interface.crossover(offspring[i - 1].genotype, offspring[i].genotype, encoding)
                offspring[i - 1] = Individual(genotype=a, encoding=encoding)
                offspring[i] = Individual(genotype=b, encoding=encoding)
        for i in range(len(offspring)):
            if rng.random() < config.mutation_prob:
                offspring[i] = Individual(genotype=interface.mutate(offspring[i].genotype, encoding), encoding=encoding)
        try:
            evaluate_all(offspring)
        except BudgetExhausted:
            stopped = True
            offspring = [o for o in offspring if o.evaluated] or population
        population = offspring
        n_evals = _n_evals(interface, n_evals)
        log(gen, population, n_evals)
        if config.eval_budget and n_evals >= config.eval_budget:
            stopped = True

    best = state.best or max(population, key=lambda i: i.fitness).copy()
    final_best = best.fitness
    threshold = 0.9 * final_best if final_best > 0 else final_best
    conv = next((e for e, f in state.trace if f >= threshold), n_evals)
    conv_impr = 0
    if state.initial_best is not None and final_best > state.initial_best:
        target = state.initial_best + 0.9 * (final_best - state.initial_best)
        conv_impr = next((e for e, f in state.trace if f >= target), n_evals)
    return RunResult(population=population, history=history, n_evaluations=n_evals, best_individual=best,
                     convergence_evals=conv, convergence_evals_improvement=conv_impr,
                     stopped_by_budget=stopped, n_cache_hits=state.cache_hits, fitness_trace=state.trace)


def config_for(base: EngineConfig, simple_cfg: dict | None = None) -> EngineConfig:
    """Simple-EA settings with the same offspring budget as `base` (pop x generations)."""
    s = {**DEFAULTS, **(simple_cfg or {})}
    births = base.population_size * base.n_generations
    gens = max(2, math.ceil(births / s["population_size"]))
    return EngineConfig(**{**base.__dict__, "population_size": s["population_size"], "n_generations": gens,
                           "tournament_size": s["tournament_size"], "mutation_prob": s["mutation_prob"],
                           "crossover_prob": s["crossover_prob"]})

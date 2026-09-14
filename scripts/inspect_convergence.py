#!/usr/bin/env python3
"""
Run ONE (variant, encoding, task) configuration and print/save its per-generation
history -- best/mean fitness and mean diversity, generation by generation -- so you can
actually see whether fitness is climbing within a run, rather than comparing final
numbers across different independent random seeds (which is very noisy for 1-2 repeats).

For LIVE per-generation progress (not just the final table printed at the end), add the
print statement described in engine.py's run_evolution() loop -- see the ETA snippet
provided alongside this script. That's the only place true live progress can be printed,
since this script only gets `result.history` back after the whole run completes.

Usage:
    python scripts/inspect_convergence.py
    python scripts/inspect_convergence.py --variant de_nsga2_gene --encoding f1 --task velocity_forward
    python scripts/inspect_convergence.py --population-size 30 --generations 40 --eval-budget 5000 --csv results/convergence.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms import adaptive_de_nsga2, de_nsga2, nsga2_baseline  # noqa: E402
from src.algorithms.engine import EngineConfig  # noqa: E402
from src.framsticks_interface.interface import get_interface  # noqa: E402
from src.utils.config import load_config, seed_everything  # noqa: E402

VARIANT_RUNNERS = {
    "nsga2_baseline": lambda interface, encoding, task, config, seed: nsga2_baseline.run(
        interface, encoding, task, config, seed),
    "de_nsga2_gene": lambda interface, encoding, task, config, seed: de_nsga2.run_gene_crowding(
        interface, encoding, task, config, seed),
    "de_nsga2_graph": lambda interface, encoding, task, config, seed: de_nsga2.run_graph_crowding(
        interface, encoding, task, config, seed),
    "de_nsga2_hybrid": lambda interface, encoding, task, config, seed: de_nsga2.run_hybrid_crowding(
        interface, encoding, task, config, seed),
    "adaptive_de_nsga2": lambda interface, encoding, task, config, seed: adaptive_de_nsga2.run(
        interface, encoding, task, config, seed),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="Path to config YAML (default: configs/default_config.yaml)")
    parser.add_argument("--variant", default="nsga2_baseline", choices=list(VARIANT_RUNNERS))
    parser.add_argument("--encoding", default="f1", choices=["f0", "f1"])
    parser.add_argument("--task", default="straight_line_cog",
                         choices=["straight_line_cog", "sinusoidal_bounce", "velocity_forward", "jump_height"])
    parser.add_argument("--population-size", type=int, default=None, help="Overrides config's ga.population_size")
    parser.add_argument("--generations", type=int, default=None, help="Overrides config's ga.n_generations")
    parser.add_argument("--eval-budget", type=int, default=None,
                         help="Overrides config's framsticks.eval_budget. IMPORTANT: this is a hard ceiling on "
                              "total evaluations for the whole run -- if it's lower than roughly "
                              "population_size * n_generations, FramsticksLibCompetition will call sys.exit() "
                              "partway through and you'll lose the run. Rule of thumb: set this comfortably "
                              "above population_size * n_generations.")
    parser.add_argument("--time-budget", type=int, default=None,
                         help="Overrides config's framsticks.time_budget_seconds. Note: this only bounds "
                              "non-simulation overhead time, NOT time spent inside evaluate() -- see chat "
                              "discussion. Raising this alone will NOT stop a run that's slow due to long "
                              "creature lifespans; lower Energy0 in your .sim settings for that instead.")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--csv", default=None, help="Optional path to save per-generation stats as CSV")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["experiment"]["random_seed_base"])
    interface = get_interface(config)

    population_size = args.population_size or config["ga"]["population_size"]
    n_generations = args.generations or config["ga"]["n_generations"]
    eval_budget = args.eval_budget or config["framsticks"]["eval_budget"]

    min_recommended_budget = population_size * n_generations
    if eval_budget < min_recommended_budget:
        print(f"WARNING: eval_budget ({eval_budget}) is lower than population_size * n_generations "
              f"({min_recommended_budget}). This run will very likely be cut short mid-way by "
              f"FramsticksLibCompetition's own budget enforcement (which calls sys.exit()). Consider "
              f"passing --eval-budget {min_recommended_budget * 2} or higher.\n")

    engine_config = EngineConfig(
        population_size=population_size,
        n_generations=n_generations,
        crossover_prob=config["ga"]["crossover_prob"],
        mutation_prob=config["ga"]["mutation_prob"],
        tournament_size=config["ga"]["tournament_size"],
        hybrid_weight=config["hybrid_crowding"]["gene_weight"],
        adaptive_tau=config["adaptive"]["tau"],
        adaptive_steepness=config["adaptive"]["steepness"],
        eval_budget=eval_budget,
    )

    # Also override the real backend's own budget enforcement, not just our engine's --
    # both need to agree, or FramsticksLibCompetition's internal check (which uses its own
    # MAX_EVALUATIONS/MAX_TIME class attributes, set via reset_evaluation_budget()) can
    # still trip independently of engine_config.eval_budget.
    if hasattr(interface, "reset_evaluation_budget"):
        interface.reset_evaluation_budget()
    if hasattr(interface, "_lib"):
        interface._lib.MAX_EVALUATIONS = eval_budget
        if args.time_budget is not None:
            interface._lib.MAX_TIME = args.time_budget

    print(f"Running {args.variant} | {args.encoding} | {args.task} | "
          f"population={engine_config.population_size} generations={engine_config.n_generations} "
          f"eval_budget={eval_budget}\n")

    result = VARIANT_RUNNERS[args.variant](interface, args.encoding, args.task, engine_config, args.seed)

    print(f"\n{'gen':>4}  {'evals':>6}  {'best':>8}  {'mean':>8}  {'mean_diversity':>14}")
    for gen_stat in result.history:
        print(f"{gen_stat.generation:4d}  {gen_stat.n_evaluations:6d}  "
              f"{gen_stat.best_fitness:8.4f}  {gen_stat.mean_fitness:8.4f}  "
              f"{gen_stat.mean_diversity:14.4f}")

    print(f"\nFinal best fitness: {result.best_individual.fitness:.4f}")
    print(f"Best genotype: {result.best_individual.genotype!r}")

    if args.csv:
        out_path = Path(args.csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["generation", "n_evaluations", "best_fitness", "mean_fitness", "mean_diversity"])
            for gen_stat in result.history:
                writer.writerow([gen_stat.generation, gen_stat.n_evaluations, gen_stat.best_fitness,
                                  gen_stat.mean_fitness, gen_stat.mean_diversity])
        print(f"\nSaved per-generation CSV to {out_path}")


if __name__ == "__main__":
    main()

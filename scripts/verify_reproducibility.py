#!/usr/bin/env python3
"""
Reproducibility check split into two safe steps, avoiding calling frams.init() twice in
one process (untested, possibly unsafe):

    1. Run once, save per-generation results to a JSON file:
       python scripts/verify_reproducibility.py --save results/repro_a.json

    2. Run again -- SEPARATE process launch, same command but a different --save path:
       python scripts/verify_reproducibility.py --save results/repro_b.json

    3. Compare the two saved files (no Framsticks needed for this step):
       python scripts/verify_reproducibility.py --compare results/repro_a.json results/repro_b.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def do_run(args):
    from src.algorithms import de_nsga2
    from src.algorithms.engine import EngineConfig
    from src.framsticks_interface.interface import get_interface
    from src.utils.config import load_config, seed_everything

    config = load_config(args.config)
    seed_everything(config["experiment"]["random_seed_base"])
    interface = get_interface(config)

    eval_budget = args.eval_budget or max(config["framsticks"]["eval_budget"],
                                           args.population_size * args.generations * 2)
    engine_config = EngineConfig(
        population_size=args.population_size,
        n_generations=args.generations,
        crossover_prob=config["ga"]["crossover_prob"],
        mutation_prob=config["ga"]["mutation_prob"],
        tournament_size=config["ga"]["tournament_size"],
        eval_budget=eval_budget,
    )

    print(f"Running population={args.population_size} generations={args.generations} seed={args.seed}\n")
    result = de_nsga2.run_gene_crowding(interface, "f1", "straight_line_cog", engine_config, seed=args.seed)

    history = [{"generation": g.generation, "best_fitness": g.best_fitness, "mean_fitness": g.mean_fitness}
               for g in result.history]

    out_path = Path(args.save)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"history": history, "final_genotype": result.best_individual.genotype}, f, indent=2)
    print(f"\nSaved to {out_path}")
    print(f"Final best fitness: {result.best_individual.fitness:.6f}")


def do_compare(paths):
    with open(paths[0]) as f:
        run_a = json.load(f)
    with open(paths[1]) as f:
        run_b = json.load(f)

    hist_a, hist_b = run_a["history"], run_b["history"]
    if len(hist_a) != len(hist_b):
        print(f"MISMATCH: different number of generations ({len(hist_a)} vs {len(hist_b)}) -- "
              f"can't compare directly. Did both runs use the same --generations value?")
        return

    print(f"{'gen':>4}  {'a_best':>10}  {'b_best':>10}  {'match':>6}  "
          f"{'a_mean':>10}  {'b_mean':>10}  {'match':>6}")
    all_match = True
    for a, b in zip(hist_a, hist_b):
        best_match = a["best_fitness"] == b["best_fitness"]
        mean_match = a["mean_fitness"] == b["mean_fitness"]
        if not (best_match and mean_match):
            all_match = False
        print(f"{a['generation']:4d}  {a['best_fitness']:10.6f}  {b['best_fitness']:10.6f}  "
              f"{'YES' if best_match else 'NO':>6}  "
              f"{a['mean_fitness']:10.6f}  {b['mean_fitness']:10.6f}  "
              f"{'YES' if mean_match else 'NO':>6}")

    genotype_match = run_a["final_genotype"] == run_b["final_genotype"]
    print(f"\nFinal best genotype match: {'YES' if genotype_match else 'NO'}")
    if not genotype_match:
        print(f"  a: {run_a['final_genotype']!r}")
        print(f"  b: {run_b['final_genotype']!r}")

    print()
    if all_match:
        print("REPRODUCIBLE: every generation's best/mean fitness matched exactly between runs.")
    else:
        print("NOT FULLY REPRODUCIBLE: at least one generation's fitness differed. Check where "
              "the first mismatch appears above -- an early, small, non-growing gap suggests "
              "residual noise deep in a retry path; a gap that widens over generations suggests "
              "it's compounding and is worth investigating further before trusting comparisons "
              "across separate runs.")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--generations", type=int, default=3)
    parser.add_argument("--eval-budget", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--save", default=None, help="Path to save this run's results as JSON")
    parser.add_argument("--compare", nargs=2, metavar=("RUN_A", "RUN_B"),
                         help="Compare two previously-saved JSON result files (no Framsticks needed)")
    args = parser.parse_args()

    if args.compare:
        do_compare(args.compare)
    elif args.save:
        do_run(args)
    else:
        parser.error("Specify either --save PATH (to run and save) or --compare A B (to diff two saved runs)")


if __name__ == "__main__":
    main()

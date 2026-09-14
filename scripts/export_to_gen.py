#!/usr/bin/env python3
"""
Run ONE (variant, encoding, task) configuration and save its best genotype (and,
optionally, the whole final population) to a .gen file using the same
framsfiles.writer format FramsticksEvolution.py uses for its own Hall of Fame -- so the
output is directly openable in the Framsticks GUI.

To view it:
    1. Launch the Framsticks GUI application (the main executable in your Framsticks55
       folder -- NOT frams-objects.dll, that's the library we script against headlessly).
    2. File -> Load, or drag-and-drop the .gen file onto the GUI's title bar.
    3. Open the "Genotypes" window (Window menu, if not already visible).
    4. Right-click the genotype in the list -> Simulate -> Creatures.
    5. Open the "Artificial world" window (Window menu) and press the play/simulate
       button in the toolbar to watch it move.

Usage:
    python scripts/export_to_gen.py --variant de_nsga2_gene --encoding f1 --task straight_line_cog --out results/best.gen
    python scripts/export_to_gen.py --all-population --out results/final_population.gen
"""
from __future__ import annotations

import argparse
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


def save_genotypes_to_gen(filename: str, entries: list[dict], framspy_path: str) -> None:
    """`entries` is a list of dicts with at least a 'genotype' key; any other keys
    (fitness, rank, etc.) are written alongside as extra fields, same pattern as
    FramsticksEvolution.py's save_genotypes(). Requires framspy_path on sys.path so
    `framsfiles` (part of framspy) is importable."""
    if framspy_path not in sys.path:
        sys.path.append(framspy_path)
    from framsfiles import writer as framswriter

    with open(filename, "w") as outfile:
        for entry in entries:
            outfile.write(framswriter.from_collection({"_classname": "org", **entry}))
            outfile.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--variant", default="nsga2_baseline", choices=list(VARIANT_RUNNERS))
    parser.add_argument("--encoding", default="f1", choices=["f0", "f1"])
    parser.add_argument("--task", default="straight_line_cog",
                         choices=["straight_line_cog", "sinusoidal_bounce", "velocity_forward", "jump_height"])
    parser.add_argument("--population-size", type=int, default=None)
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", default="results/best.gen", help="Output .gen file path")
    parser.add_argument("--all-population", action="store_true",
                         help="Save every individual in the final population, not just the best one")
    parser.add_argument("--top-k", type=int, default=None,
                         help="Save only the top K by fitness (combine with --all-population)")
    args = parser.parse_args()

    config = load_config(args.config)
    seed_everything(config["experiment"]["random_seed_base"])
    interface = get_interface(config)

    engine_config = EngineConfig(
        population_size=args.population_size or config["ga"]["population_size"],
        n_generations=args.generations or config["ga"]["n_generations"],
        crossover_prob=config["ga"]["crossover_prob"],
        mutation_prob=config["ga"]["mutation_prob"],
        tournament_size=config["ga"]["tournament_size"],
        hybrid_weight=config["hybrid_crowding"]["gene_weight"],
        adaptive_tau=config["adaptive"]["tau"],
        adaptive_steepness=config["adaptive"]["steepness"],
        eval_budget=config["framsticks"]["eval_budget"],
    )

    print(f"Running {args.variant} | {args.encoding} | {args.task} ...")
    result = VARIANT_RUNNERS[args.variant](interface, args.encoding, args.task, engine_config, args.seed)
    print(f"Done. Best fitness: {result.best_individual.fitness:.4f}")

    if args.all_population:
        population = sorted(result.population, key=lambda ind: ind.fitness, reverse=True)
        if args.top_k:
            population = population[:args.top_k]
        entries = [{"genotype": ind.genotype, "fitness": ind.fitness} for ind in population]
    else:
        entries = [{"genotype": result.best_individual.genotype, "fitness": result.best_individual.fitness}]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_genotypes_to_gen(str(out_path), entries, config["framsticks"]["framspy_path"])
    print(f"Saved {len(entries)} genotype(s) to {out_path}")
    print("Open this in the Framsticks GUI: File -> Load (or drag-and-drop onto the title bar),")
    print("then right-click the genotype in the Genotypes window -> Simulate -> Creatures.")


if __name__ == "__main__":
    main()

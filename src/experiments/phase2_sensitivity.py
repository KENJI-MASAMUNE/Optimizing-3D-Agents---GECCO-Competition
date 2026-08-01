"""
Phase 2 -- sensitivity analysis (Section 5.3). 

"Vary key parameters: population size (50-200), tournament size, the adaptive switching
threshold tau, and the weight between gene/graph in Variant 4."
"""
from __future__ import annotations

from pathlib import Path

from src.algorithms.engine import EngineConfig
from src.experiments.run_single import run_single
from src.framsticks_interface.interface import get_interface
from src.utils.config import load_config, seed_everything, write_run_record

POPULATION_SIZES = [50, 100, 150, 200]
TOURNAMENT_SIZES = [2, 3, 5]
ADAPTIVE_TAUS = [0.25, 0.5, 0.75]
HYBRID_GENE_WEIGHTS = [0.0, 0.25, 0.5, 0.75, 1.0]


def run_phase2(config_path: str | None = None, smoke: bool = False) -> list[Path]:
    config = load_config(config_path)
    seed_everything(config["experiment"]["random_seed_base"])
    interface = get_interface(config)

    base_kwargs = dict(
        n_generations=config["ga"]["n_generations"] if not smoke else 5,
        crossover_prob=config["ga"]["crossover_prob"],
        mutation_prob=config["ga"]["mutation_prob"],
    )
    task = config["tasks"][0]["name"]
    encoding = config["ga"]["encodings"][0]
    n_repeats = config["experiment"]["n_repeats_smoke"] if smoke else 5  # lighter than Phase 1
    seed_base = config["experiment"]["random_seed_base"]
    out_dir = Path(config["experiment"]["results_dir"]) / "phase2"

    written: list[Path] = []

    pop_sizes = POPULATION_SIZES[:1] if smoke else POPULATION_SIZES
    tourn_sizes = TOURNAMENT_SIZES[:1] if smoke else TOURNAMENT_SIZES
    for pop_size in pop_sizes:
        for tourn in tourn_sizes:
            engine_config = EngineConfig(
                population_size=pop_size, tournament_size=tourn,
                hybrid_weight=config["hybrid_crowding"]["gene_weight"],
                adaptive_tau=config["adaptive"]["tau"],
                adaptive_steepness=config["adaptive"]["steepness"],
                eval_budget=config["framsticks"]["eval_budget"],
                **base_kwargs,
            )
            for repeat in range(n_repeats):
                record = run_single(interface, "de_nsga2_gene", encoding, task,
                                     engine_config, repeat, seed_base)
                record.extra = {"sweep": "population_tournament", "pop_size": pop_size, "tournament": tourn}
                written.append(write_run_record(record, out_dir))

    taus = ADAPTIVE_TAUS[:1] if smoke else ADAPTIVE_TAUS
    for tau in taus:
        engine_config = EngineConfig(
            population_size=config["ga"]["population_size"],
            tournament_size=config["ga"]["tournament_size"],
            hybrid_weight=config["hybrid_crowding"]["gene_weight"],
            adaptive_tau=tau, adaptive_steepness=config["adaptive"]["steepness"],
            eval_budget=config["framsticks"]["eval_budget"], **base_kwargs,
        )
        for repeat in range(n_repeats):
            record = run_single(interface, "adaptive_de_nsga2", encoding, task,
                                 engine_config, repeat, seed_base)
            record.extra = {"sweep": "adaptive_tau", "tau": tau}
            written.append(write_run_record(record, out_dir))

    weights = HYBRID_GENE_WEIGHTS[:1] if smoke else HYBRID_GENE_WEIGHTS
    for weight in weights:
        engine_config = EngineConfig(
            population_size=config["ga"]["population_size"],
            tournament_size=config["ga"]["tournament_size"],
            hybrid_weight=weight, adaptive_tau=config["adaptive"]["tau"],
            adaptive_steepness=config["adaptive"]["steepness"],
            eval_budget=config["framsticks"]["eval_budget"], **base_kwargs,
        )
        for repeat in range(n_repeats):
            record = run_single(interface, "de_nsga2_hybrid", encoding, task,
                                 engine_config, repeat, seed_base)
            record.extra = {"sweep": "hybrid_gene_weight", "gene_weight": weight}
            written.append(write_run_record(record, out_dir))

    print(f"[phase2] wrote {len(written)} run records to {out_dir}")
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 2 sensitivity analysis")
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    run_phase2(args.config, smoke=args.smoke)

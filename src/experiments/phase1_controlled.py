"""
Phase 1 -- controlled internal experiments (Section 5.3).

"Using publicly documented simulation parameters as proxies for competition settings, run
all 5 variants x 2 encodings x 4 proxy tasks x 20 repeats." Runs against the real
Framsticks/framspy install configured in configs/default_config.yaml (see README) until
real competition worlds are available.
"""
from __future__ import annotations

from pathlib import Path

from src.algorithms.engine import EngineConfig
from src.experiments.run_single import run_single
from src.framsticks_interface.interface import get_interface
from src.utils.config import load_config, seed_everything, write_run_record


def run_phase1(config_path: str | None = None, smoke: bool = False) -> list[Path]:
    config = load_config(config_path)
    seed_everything(config["experiment"]["random_seed_base"])
    interface = get_interface(config)

    engine_config = EngineConfig(
        population_size=config["ga"]["population_size"],
        n_generations=config["ga"]["n_generations"] if not smoke else 5,
        crossover_prob=config["ga"]["crossover_prob"],
        mutation_prob=config["ga"]["mutation_prob"],
        tournament_size=config["ga"]["tournament_size"],
        hybrid_weight=config["hybrid_crowding"]["gene_weight"],
        adaptive_tau=config["adaptive"]["tau"],
        adaptive_steepness=config["adaptive"]["steepness"],
        eval_budget=config["framsticks"]["eval_budget"],
    )

    variants = config["variants"] if not smoke else config["variants"][:2]
    encodings = config["ga"]["encodings"]
    tasks = [t["name"] for t in config["tasks"]] if not smoke else [config["tasks"][0]["name"]]
    n_repeats = config["experiment"]["n_repeats_smoke"] if smoke else config["experiment"]["n_repeats"]
    seed_base = config["experiment"]["random_seed_base"]
    out_dir = Path(config["experiment"]["results_dir"]) / "phase1"

    written: list[Path] = []
    total = len(variants) * len(encodings) * len(tasks) * n_repeats
    done = 0
    for variant in variants:
        for encoding in encodings:
            for task in tasks:
                for repeat in range(n_repeats):
                    record = run_single(interface, variant, encoding, task, engine_config,
                                         repeat, seed_base)
                    path = write_run_record(record, out_dir)
                    written.append(path)
                    done += 1
                    print(f"[phase1] {done}/{total} {variant} | {encoding} | {task} | "
                          f"repeat {repeat} -> best_fitness={record.best_fitness:.4f}")
    return written


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 1 controlled experiments")
    parser.add_argument("--config", default=None)
    parser.add_argument("--smoke", action="store_true", help="Quick sanity run (few variants/tasks/repeats)")
    args = parser.parse_args()
    run_phase1(args.config, smoke=args.smoke)

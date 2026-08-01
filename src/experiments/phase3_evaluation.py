"""
Phase 3 -- evaluation and testing (Section 5.3). Owner: Ntsako Sithole

"Select the best-performing variant and encoding per task type. Test in a clean environment
using FramsticksLibCompetition." Also computes the Generalisation Score metric (Section 5.4)
against held-out world configurations not seen during Phase 1/2 training.
"""
from __future__ import annotations

import json
from pathlib import Path

from src.algorithms import adaptive_de_nsga2, de_nsga2, nsga2_baseline
from src.algorithms.engine import EngineConfig
from src.evaluation.metrics import generalisation_score
from src.experiments.run_single import VARIANT_RUNNERS
from src.framsticks_interface.interface import get_interface
from src.utils.config import load_config, seed_everything

# Held-out world configurations, deliberately not used anywhere in Phase 1/2, to measure
# generalisation (Section 5.4) rather than memorisation of training-time environments.
HELD_OUT_WORLD_CONFIGS = [
    {"gravity": 1.2, "water_level": 0.0, "terrain": "flat", "orientation": "upright"},
    {"gravity": 0.8, "water_level": 0.5, "terrain": "hilly", "orientation": "tilted"},
    {"gravity": 1.0, "water_level": 1.0, "terrain": "flat", "orientation": "inverted"},
]


def select_best_per_task(phase1_results_dir: str) -> dict[str, dict]:
    """Reads Phase 1 JSON records and, for each task, returns the (variant, encoding) with
    the highest mean best fitness across repeats."""
    records = []
    for path in Path(phase1_results_dir).glob("*.json"):
        with open(path) as f:
            records.append(json.load(f))

    by_task: dict[str, dict[tuple[str, str], list[float]]] = {}
    for r in records:
        key = (r["variant"], r["encoding"])
        by_task.setdefault(r["task"], {}).setdefault(key, []).append(r["best_fitness"])

    best_per_task = {}
    for task, variant_scores in by_task.items():
        best_key = max(variant_scores, key=lambda k: sum(variant_scores[k]) / len(variant_scores[k]))
        best_per_task[task] = {"variant": best_key[0], "encoding": best_key[1],
                                "mean_best_fitness": sum(variant_scores[best_key]) / len(variant_scores[best_key])}
    return best_per_task


def run_phase3(config_path: str | None = None, phase1_results_dir: str | None = None) -> dict:
    config = load_config(config_path)
    seed_everything(config["experiment"]["random_seed_base"])
    interface = get_interface(config)

    phase1_dir = phase1_results_dir or str(Path(config["experiment"]["results_dir"]) / "phase1")
    best_per_task = select_best_per_task(phase1_dir)

    engine_config = EngineConfig(
        population_size=config["ga"]["population_size"],
        n_generations=config["ga"]["n_generations"],
        crossover_prob=config["ga"]["crossover_prob"],
        mutation_prob=config["ga"]["mutation_prob"],
        tournament_size=config["ga"]["tournament_size"],
        hybrid_weight=config["hybrid_crowding"]["gene_weight"],
        adaptive_tau=config["adaptive"]["tau"],
        adaptive_steepness=config["adaptive"]["steepness"],
        eval_budget=config["framsticks"]["eval_budget"],
    )
    seed_base = config["experiment"]["random_seed_base"] + 1000  # distinct seed space from Phase 1/2

    results = {}
    for task, choice in best_per_task.items():
        runner = VARIANT_RUNNERS[choice["variant"]]
        result = runner(interface, choice["encoding"], task, engine_config, seed_base, None)
        gen_score = generalisation_score(interface, choice["encoding"], task,
                                          result.best_individual, HELD_OUT_WORLD_CONFIGS)
        results[task] = {
            "variant": choice["variant"],
            "encoding": choice["encoding"],
            "clean_best_fitness": result.best_individual.fitness,
            "generalisation_score": gen_score,
        }
        print(f"[phase3] {task}: {choice['variant']}/{choice['encoding']} "
              f"clean_fitness={result.best_individual.fitness:.4f} "
              f"generalisation={gen_score:.4f}")

    out_dir = Path(config["experiment"]["results_dir"]) / "phase3"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "phase3_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Phase 3 clean-environment evaluation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--phase1-results", default=None)
    args = parser.parse_args()
    run_phase3(args.config, args.phase1_results)

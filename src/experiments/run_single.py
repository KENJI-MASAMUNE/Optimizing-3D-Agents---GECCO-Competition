"""
Run a single (variant, encoding, task[, world], repeat) configuration end-to-end.

This is the atomic unit every phase calls. `run_spec()` is the manifest-based entry point
used by the CLI / CHPC jobs; `run_single()` keeps the original signature for existing
callers and tests.
"""
from __future__ import annotations

import json
import os
import platform
import socket
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from src.algorithms import adaptive_de_nsga2, de_nsga2, nsga2_baseline, simple_ea
from src.algorithms.engine import EngineConfig, RunResult
from src.evaluation.metrics import generalisation_details, summarise_run
from src.framsticks_interface.interface import FramsticksInterface
from src.utils.config import RunRecord, timer

VARIANT_RUNNERS = {
    "nsga2_baseline": lambda interface, encoding, task, config, seed, wc: nsga2_baseline.run(
        interface, encoding, task, config, seed, wc),
    "de_nsga2_gene": lambda interface, encoding, task, config, seed, wc: de_nsga2.run_gene_crowding(
        interface, encoding, task, config, seed, wc),
    "de_nsga2_graph": lambda interface, encoding, task, config, seed, wc: de_nsga2.run_graph_crowding(
        interface, encoding, task, config, seed, wc),
    "de_nsga2_hybrid": lambda interface, encoding, task, config, seed, wc: de_nsga2.run_hybrid_crowding(
        interface, encoding, task, config, seed, wc),
    "adaptive_de_nsga2": lambda interface, encoding, task, config, seed, wc: adaptive_de_nsga2.run(
        interface, encoding, task, config, seed, wc),
    # organisers' baseline (framspy FramsticksEvolution.py / DEAP eaSimple), see simple_ea.py
    "simple_ea": lambda interface, encoding, task, config, seed, wc: simple_ea.run(
        interface, encoding, task, config, seed, wc),
}


def _prepare_interface(interface: FramsticksInterface, seed: int, eval_budget: int | None) -> None:
    """Per-run isolation: fresh budget + every RNG reset from this run's seed alone."""
    if hasattr(interface, "set_budget") and eval_budget:
        interface.set_budget(eval_budget=eval_budget)
    if hasattr(interface, "reset_evaluation_budget"):
        interface.reset_evaluation_budget()
    if hasattr(interface, "reseed"):
        interface.reseed(seed)
    else:  # test double without reseed(): at least pin the Python-side RNGs
        import random

        import numpy as np
        random.seed(seed)
        np.random.seed(seed % (2 ** 32))


def run_single(
    interface: FramsticksInterface,
    variant: str,
    encoding: str,
    task: str,
    config: EngineConfig,
    repeat: int,
    seed_base: int,
    world_config=None,
) -> RunRecord:
    """Original API (kept for compatibility). seed = seed_base + repeat."""
    if variant not in VARIANT_RUNNERS:
        raise ValueError(f"Unknown variant '{variant}'. Options: {list(VARIANT_RUNNERS)}")
    seed = seed_base + repeat
    _prepare_interface(interface, seed, config.eval_budget)
    start = timer()
    result: RunResult = VARIANT_RUNNERS[variant](interface, encoding, task, config, seed, world_config)
    wall_time = timer() - start
    metrics = summarise_run(interface, result, encoding)
    return RunRecord(
        variant=variant, encoding=encoding, task=task, repeat=repeat, seed=seed,
        best_fitness=metrics.mean_best_fitness, gene_diversity=metrics.gene_diversity,
        graph_diversity=metrics.graph_diversity, convergence_evals=metrics.convergence_evals,
        n_evaluations=result.n_evaluations, wall_time_s=wall_time,
        extra={"convergence_evals_improvement": metrics.convergence_evals_improvement,
               "stopped_by_budget": result.stopped_by_budget,
               "best_genotype": result.best_individual.genotype},
    )


# ---------------------------------------------------------------------------- manifest API
def engine_config_from(config: dict, overrides: dict[str, Any] | None = None, smoke: bool = False,
                       verbose: bool = True) -> EngineConfig:
    ga = config["ga"]
    kw = dict(
        population_size=ga["population_size"],
        n_generations=ga["n_generations"],
        crossover_prob=ga["crossover_prob"],
        mutation_prob=ga["mutation_prob"],
        tournament_size=ga["tournament_size"],
        hybrid_weight=config["hybrid_crowding"]["gene_weight"],
        adaptive_tau=config["adaptive"]["tau"],
        adaptive_steepness=config["adaptive"]["steepness"],
        eval_budget=config["framsticks"]["eval_budget"],
        cache_fitness=ga.get("cache_fitness", True),
        verbose=verbose,
    )
    if smoke:
        sm = config.get("smoke", {})
        kw["population_size"] = sm.get("population_size", 6)
        kw["n_generations"] = sm.get("n_generations", 3)
    if overrides:
        kw.update(overrides)
        if smoke:  # keep smoke runs tiny even when a sweep overrides pop/gens
            kw["population_size"] = min(kw["population_size"], config.get("smoke", {}).get("population_size", 6))
            kw["n_generations"] = min(kw["n_generations"], config.get("smoke", {}).get("n_generations", 3))
    return EngineConfig(**kw)


def _provenance() -> dict[str, Any]:
    info = {"host": socket.gethostname(), "python": platform.python_version(),
            "pbs_jobid": os.environ.get("PBS_JOBID"), "pbs_array_index": os.environ.get("PBS_ARRAY_INDEX"),
            "slurm_jobid": os.environ.get("SLURM_JOB_ID")}
    try:
        root = Path(__file__).resolve().parents[2]
        info["git_commit"] = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"],
                                            capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        info["git_commit"] = None
    return info


def run_spec(interface: FramsticksInterface, spec, config: dict, results_dir: str | Path,
             smoke: bool = False, verbose: bool = True) -> dict[str, Any]:
    """Runs one RunSpec and atomically writes its JSON record. Returns the record."""
    if spec.variant not in VARIANT_RUNNERS:
        raise ValueError(f"Unknown variant '{spec.variant}'")
    engine_cfg = engine_config_from(config, spec.overrides_dict, smoke=smoke, verbose=verbose)
    if spec.variant == "simple_ea":  # its own GA settings, same offspring budget
        engine_cfg = simple_ea.config_for(engine_cfg, config.get("simple_ea"))
        if smoke:  # keep smoke runs tiny
            sm_pop = config.get("smoke", {}).get("population_size", 6)
            engine_cfg = EngineConfig(**{**engine_cfg.__dict__, "population_size": sm_pop,
                                         "n_generations": config.get("smoke", {}).get("n_generations", 3)})
    _prepare_interface(interface, spec.seed, engine_cfg.eval_budget)

    world = spec.world  # registered world name or None (base training world)
    if hasattr(interface, "set_world"):
        interface.set_world(world)

    t0 = time.perf_counter()
    result: RunResult = VARIANT_RUNNERS[spec.variant](interface, spec.encoding, spec.task, engine_cfg,
                                                      spec.seed, world)
    wall = time.perf_counter() - t0
    metrics = summarise_run(interface, result, spec.encoding)

    record: dict[str, Any] = {
        "phase": spec.phase, "run_id": spec.run_id,
        "variant": spec.variant, "encoding": spec.encoding, "task": spec.task,
        "repeat": spec.repeat, "seed": spec.seed, "group": spec.group,
        "world": spec.world, "setting": spec.setting,
        "sweep": spec.sweep_dict,
        "best_fitness": metrics.mean_best_fitness,
        "best_genotype": result.best_individual.genotype,
        "gene_diversity": metrics.gene_diversity,
        "graph_diversity": metrics.graph_diversity,
        "convergence_evals": metrics.convergence_evals,
        "convergence_evals_improvement": metrics.convergence_evals_improvement,
        "n_evaluations": result.n_evaluations,
        "n_cache_hits": result.n_cache_hits,
        "stopped_by_budget": result.stopped_by_budget,
        "wall_time_s": wall,
        "engine_config": asdict(engine_cfg),
        "history": [asdict(h) for h in result.history],
        "fitness_trace": result.fitness_trace,
        "smoke": smoke,
        "provenance": _provenance(),
    }
    if hasattr(interface, "sim_files"):
        record["sim_files"] = interface.sim_files
    if hasattr(interface, "base_world_values"):
        record["base_world"] = interface.base_world_values

    # Phase 3: generalisation of the best agent to held-out worlds (fresh budget so these
    # few re-evaluations can never be refused by an exhausted run budget).
    if spec.held_out_worlds:
        if hasattr(interface, "reset_evaluation_budget"):
            interface.reset_evaluation_budget()
        details = generalisation_details(interface, spec.encoding, spec.task, result.best_individual,
                                         list(spec.held_out_worlds))
        gen_score = sum(d["fitness"] for d in details) / len(details)
        record["generalisation_score"] = gen_score
        record["generalisation_details"] = details
        best = metrics.mean_best_fitness
        record["generalisation_retention"] = (gen_score / best) if best and best > 0 else None
        if hasattr(interface, "set_world"):
            interface.set_world(None)

    out = spec.output_path(results_dir)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(record, f, indent=1, default=_json_default)
    os.replace(tmp, out)
    return record


def _json_default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except Exception:
        pass
    if isinstance(o, float) and o != o:
        return None
    return str(o)

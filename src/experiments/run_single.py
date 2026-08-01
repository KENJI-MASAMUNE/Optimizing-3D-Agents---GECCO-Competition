"""
Run a single (variant, encoding, task, repeat) configuration end-to-end.

This is the atomic unit every phase script (phase1/2/3) calls repeatedly with different
parameter combinations. Keeping it isolated makes it easy to parallelise later (e.g. with
multiprocessing or a job array) without touching the algorithm code.
"""
from __future__ import annotations

from src.algorithms import adaptive_de_nsga2, de_nsga2, nsga2_baseline
from src.algorithms.engine import EngineConfig, RunResult
from src.evaluation.metrics import summarise_run
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
}


def run_single(
    interface: FramsticksInterface,
    variant: str,
    encoding: str,
    task: str,
    config: EngineConfig,
    repeat: int,
    seed_base: int,
    world_config: dict | None = None,
) -> RunRecord:
    if variant not in VARIANT_RUNNERS:
        raise ValueError(f"Unknown variant '{variant}'. Options: {list(VARIANT_RUNNERS)}")

    # RealFramsticksInterface shares FramsticksLibCompetition's evaluation-count/timer
    # state across runs unless explicitly reset (see interface.py's CRITICAL GOTCHA
    # docstring) -- without this, Phase 1/2's many repeats through one interface would
    # accumulate toward MAX_EVALUATIONS/MAX_TIME and could sys.exit() the whole batch
    # partway through. tests/fakes.py's FakeFramsticksInterface implements the same
    # method for symmetry, so this reset happens for both.
    if hasattr(interface, "reset_evaluation_budget"):
        interface.reset_evaluation_budget()

    seed = seed_base + repeat
    start = timer()
    result: RunResult = VARIANT_RUNNERS[variant](interface, encoding, task, config, seed, world_config)
    wall_time = timer() - start

    metrics = summarise_run(interface, result, encoding)

    return RunRecord(
        variant=variant,
        encoding=encoding,
        task=task,
        repeat=repeat,
        seed=seed,
        best_fitness=metrics.mean_best_fitness,
        gene_diversity=metrics.gene_diversity,
        graph_diversity=metrics.graph_diversity,
        convergence_evals=metrics.convergence_evals,
        n_evaluations=result.n_evaluations,
        wall_time_s=wall_time,
    )

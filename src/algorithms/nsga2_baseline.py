"""
Variant 1 -- standard NSGA-II baseline (Section 5.1)

Objective-space crowding distance over (fitness, diversity_score). Reproduces the baseline
setup from Komosinski and Mensfelt (2025) that DE-NSGA-II is compared against.
"""
from __future__ import annotations

from src.algorithms.engine import (
    CrowdingStrategy,
    DiversityMeasure,
    EngineConfig,
    RunResult,
    run_evolution,
)
from src.framsticks_interface.interface import FramsticksInterface

VARIANT_NAME = "nsga2_baseline"


def run(interface: FramsticksInterface, encoding: str, task: str, config: EngineConfig,
        seed: int, world_config: dict | None = None) -> RunResult:
    return run_evolution(
        interface=interface,
        encoding=encoding,
        task=task,
        crowding_strategy=CrowdingStrategy.OBJECTIVE,
        diversity_measure=DiversityMeasure.GENE,  # diversity objective definition (engine.py docstring)
        config=config,
        seed=seed,
        world_config=world_config,
    )

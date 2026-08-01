"""
Variant 5 -- Adaptive DE-NSGA-II (Section 5.1, novel contribution). Owner: Salmaan Patel

Implements "Adaptive Sigmoidal Selection": a continuous sigmoidal schedule that smoothly
shifts crowding-distance weight from dissimilarity-based crowding (favouring structural
exploration early in the run) to objective-space crowding (favouring fitness exploitation
late in the run), controlled by two hyperparameters:

  tau        - the fraction of total generations at which the schedule crosses 0.5
               (the "switching point" between exploration- and exploitation-dominated
               selection). Default 0.5 = midpoint of the run.
  steepness  - how sharply the transition happens around tau. Higher = closer to a hard
               switch; lower = a very gradual blend across the whole run.

This directly targets Research Objective 3 (Section 3.2) and the corresponding sub-problem
7 (Section 3.1): transitioning from structural diversity to performance exploitation
"without inducing a destructive population selection shock" -- the sigmoid's continuity
(vs. a hard switch at a fixed generation) is exactly what prevents the shock.
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

VARIANT_NAME = "adaptive_de_nsga2"


def sigmoid_weight(progress: float, tau: float, steepness: float) -> float:
    """Standalone helper (exposed for unit testing / plotting the schedule independently
    of a full evolutionary run). `progress` in [0, 1] is generation / (n_generations - 1).
    Returns the dissimilarity-crowding weight in (0, 1): ~1 early, ~0 late, for tau=0.5."""
    import math
    return 1.0 / (1.0 + math.exp(steepness * (progress - tau)))


def run(interface: FramsticksInterface, encoding: str, task: str, config: EngineConfig,
        seed: int, world_config: dict | None = None,
        diversity_measure: DiversityMeasure = DiversityMeasure.GENE) -> RunResult:
    """`diversity_measure` selects which dissimilarity feeds the exploration side of the
    schedule (gene by default, for computational efficiency under the 1-hour competition
    budget -- Section 4.4.2 / Section 5.1 Variant 3 safeguard notes); pass GRAPH or HYBRID
    to test the alternative during Phase 2 sensitivity analysis."""
    return run_evolution(
        interface=interface, encoding=encoding, task=task,
        crowding_strategy=CrowdingStrategy.ADAPTIVE,
        diversity_measure=diversity_measure,
        config=config, seed=seed, world_config=world_config,
    )

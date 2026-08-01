"""
Variants 2-4 -- DE-NSGA-II Gene / Graph / Hybrid crowding (Section 5.1)

Each function is a thin configuration of the shared engine (engine.py) -- the only thing
that changes between them is which dissimilarity measure feeds the crowding distance
calculation. This mirrors the proposal's iterative methodology (Section 5): each variant
isolates exactly one change relative to the previous one.
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

VARIANT_NAME_GENE = "de_nsga2_gene"
VARIANT_NAME_GRAPH = "de_nsga2_graph"
VARIANT_NAME_HYBRID = "de_nsga2_hybrid"


def run_gene_crowding(interface: FramsticksInterface, encoding: str, task: str,
                       config: EngineConfig, seed: int, world_config: dict | None = None) -> RunResult:
    """Variant 2: crowding distance = average pairwise Levenshtein distance."""
    return run_evolution(
        interface=interface, encoding=encoding, task=task,
        crowding_strategy=CrowdingStrategy.DISSIMILARITY,
        diversity_measure=DiversityMeasure.GENE,
        config=config, seed=seed, world_config=world_config,
    )


def run_graph_crowding(interface: FramsticksInterface, encoding: str, task: str,
                        config: EngineConfig, seed: int, world_config: dict | None = None) -> RunResult:
    """Variant 3: crowding distance = average pairwise bounded graph dissimilarity."""
    return run_evolution(
        interface=interface, encoding=encoding, task=task,
        crowding_strategy=CrowdingStrategy.DISSIMILARITY,
        diversity_measure=DiversityMeasure.GRAPH,
        config=config, seed=seed, world_config=world_config,
    )


def run_hybrid_crowding(interface: FramsticksInterface, encoding: str, task: str,
                         config: EngineConfig, seed: int, world_config: dict | None = None,
                         gene_weight: float | None = None) -> RunResult:
    """Variant 4: crowding distance = weighted blend of gene and graph dissimilarity,
    ALSO blended against objective-space crowding via `config.hybrid_weight`
    (see crowding.blended_crowding_distance). `gene_weight` (if given) overrides
    `config.hybrid_weight` for this run -- used by Phase 2 sensitivity analysis
    (Section 5.3) to sweep the gene/graph balance per encoding.
    """
    if gene_weight is not None:
        config = EngineConfig(**{**config.__dict__, "hybrid_weight": gene_weight})
    return run_evolution(
        interface=interface, encoding=encoding, task=task,
        crowding_strategy=CrowdingStrategy.HYBRID,
        diversity_measure=DiversityMeasure.HYBRID,
        config=config, seed=seed, world_config=world_config,
    )

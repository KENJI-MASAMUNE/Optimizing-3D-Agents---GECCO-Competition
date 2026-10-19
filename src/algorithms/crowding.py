"""
Crowding distance mechanisms.

- `objective_space_crowding_distance`: the standard NSGA-II crowding distance (Deb et al.,
  2002), used by Variant 1 (baseline).
- `dissimilarity_crowding_distance`: the DE-NSGA-II crowding distance from Komosinski &
  Mensfelt (2025) / Section 2.4 of the proposal:
      I_i.crowding_dist = (1 / (|I| - 1)) * sum_{j != i} D_ij
  where D_ij is a genetic or graph dissimilarity, computed by src/diversity/*.
- `blended_crowding_distance`: weighted mix of the two (each normalised to [0, 1]), used by
  Variant 5 (Adaptive) with a sigmoid-scheduled weight (see adaptive_de_nsga2.py).
  NOTE: Variant 4 (Hybrid) does NOT use this -- per the proposal it blends GENE and GRAPH
  dissimilarity matrices and then applies `dissimilarity_crowding_distance`
  (see engine._pairwise_dissimilarity / DiversityMeasure.HYBRID).

All functions operate on a list of individuals and return a list of floats aligned by
index, matching the shape DEAP expects when it later sorts by `ind.fitness.crowding_dist`.
"""
from __future__ import annotations

import math

INF = float("inf")


def objective_space_crowding_distance(fitness_values: list[tuple[float, ...]]) -> list[float]:
    """Standard NSGA-II crowding distance over one non-dominated front's objective values.
    `fitness_values[i]` is the tuple of objective values for individual i in the front.
    Boundary individuals (min/max on any objective) get infinite distance, matching the
    original NSGA-II definition so they're always preferred by selection.
    """
    n = len(fitness_values)
    if n == 0:
        return []
    if n <= 2:
        return [INF] * n

    n_obj = len(fitness_values[0])
    distances = [0.0] * n

    for m in range(n_obj):
        idx_sorted = sorted(range(n), key=lambda i: fitness_values[i][m])
        f_min = fitness_values[idx_sorted[0]][m]
        f_max = fitness_values[idx_sorted[-1]][m]
        distances[idx_sorted[0]] = INF
        distances[idx_sorted[-1]] = INF
        if f_max == f_min:
            continue
        for k in range(1, n - 1):
            prev_val = fitness_values[idx_sorted[k - 1]][m]
            next_val = fitness_values[idx_sorted[k + 1]][m]
            if distances[idx_sorted[k]] != INF:
                distances[idx_sorted[k]] += (next_val - prev_val) / (f_max - f_min)

    return distances


def dissimilarity_crowding_distance(dissimilarity_matrix: list[list[float]]) -> list[float]:
    """DE-NSGA-II crowding distance: average pairwise dissimilarity to the rest of the
    population/front. `dissimilarity_matrix[i][j]` = D_ij (gene or graph dissimilarity).
    """
    n = len(dissimilarity_matrix)
    if n <= 1:
        return [0.0] * n
    return [sum(row) / (n - 1) for row in dissimilarity_matrix]


def blended_crowding_distance(
    objective_values: list[tuple[float, ...]],
    dissimilarity_matrix: list[list[float]],
    gene_weight: float,
) -> list[float]:
    """Weighted blend of dissimilarity crowding (weight `gene_weight`, name kept for
    backwards compatibility) and objective-space crowding (weight 1 - gene_weight), each
    normalised by its largest finite value (infinite boundary distances map to 1.0).
    Used by Variant 5 (Adaptive), where the weight follows the sigmoid schedule."""
    obj_dist = objective_space_crowding_distance(objective_values)
    div_dist = dissimilarity_crowding_distance(dissimilarity_matrix)

    def _normalise(vals: list[float]) -> list[float]:
        finite = [v for v in vals if math.isfinite(v)]
        if not finite:
            return [1.0 if math.isinf(v) else 0.0 for v in vals]
        v_max = max(finite) or 1.0
        return [1.0 if math.isinf(v) else v / v_max for v in vals]

    obj_norm = _normalise(obj_dist)
    div_norm = _normalise(div_dist)

    return [
        gene_weight * d + (1.0 - gene_weight) * o
        for d, o in zip(div_norm, obj_norm)
    ]

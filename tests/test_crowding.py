import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms.crowding import (
    dissimilarity_crowding_distance,
    objective_space_crowding_distance,
)


def test_boundary_points_get_infinite_crowding_distance():
    # 3 points on a simple trade-off curve; the two extremes should be boundary points
    fitness_values = [(0.0, 1.0), (0.5, 0.5), (1.0, 0.0)]
    dists = objective_space_crowding_distance(fitness_values)
    assert math.isinf(dists[0])
    assert math.isinf(dists[2])
    assert math.isfinite(dists[1])


def test_two_or_fewer_points_all_infinite():
    assert objective_space_crowding_distance([(0.0, 1.0)]) == [float("inf")]
    dists = objective_space_crowding_distance([(0.0, 1.0), (1.0, 0.0)])
    assert all(math.isinf(d) for d in dists)


def test_dissimilarity_crowding_matches_formula():
    matrix = [
        [0.0, 0.2, 0.4],
        [0.2, 0.0, 0.6],
        [0.4, 0.6, 0.0],
    ]
    dists = dissimilarity_crowding_distance(matrix)
    n = len(matrix)
    expected = [sum(row) / (n - 1) for row in matrix]
    assert dists == expected


def test_empty_front_returns_empty():
    assert objective_space_crowding_distance([]) == []
    assert dissimilarity_crowding_distance([]) == []

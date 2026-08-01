"""
Tests the fitness MATH in task_fitness.py directly against numpy arrays, without needing
the real Framsticks simulator or framspy installed -- so they run in CI and for every team
member regardless of platform, and catch bugs in the fitness formulas before anyone burns
real evaluation budget on them.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.framsticks_interface.task_fitness import (
    extract_cog_trajectory,
    fitness_jump_height,
    fitness_sinusoidal_bounce,
    fitness_straight_line_cog,
    fitness_velocity_forward,
)


def test_extract_cog_trajectory_passes_through_cog_only_format():
    # STORE_ALL_PART_COORDS=0: already (n_steps, 3)
    raw = [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]
    result = extract_cog_trajectory(raw)
    assert result.shape == (3, 3)
    np.testing.assert_array_equal(result, np.array(raw))


def test_extract_cog_trajectory_takes_first_part_from_verbose_format():
    # STORE_ALL_PART_COORDS=1: (n_steps, n_parts+1, 3), COG is index 0 of axis 1
    raw = [
        [[0.0, 0.0, 0.0], [9.0, 9.0, 9.0], [8.0, 8.0, 8.0]],  # step 0: COG, part1, part2
        [[1.0, 1.0, 1.0], [9.1, 9.1, 9.1], [8.1, 8.1, 8.1]],  # step 1
    ]
    result = extract_cog_trajectory(raw)
    assert result.shape == (2, 3)
    np.testing.assert_array_equal(result, np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))


def test_fitness_functions_accept_verbose_format_transparently():
    # The same episode, expressed in both formats, should give identical fitness.
    cog_only = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    verbose = [[list(row), [9.0, 9.0, 9.0]] for row in cog_only]  # COG + one dummy part per step

    assert fitness_straight_line_cog(cog_only) == fitness_straight_line_cog(verbose)
    assert fitness_velocity_forward(cog_only) == fitness_velocity_forward(verbose)
    assert fitness_jump_height(cog_only) == fitness_jump_height(verbose)


def test_straight_line_perfect_path_gives_max_fitness():
    path = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    fitness = fitness_straight_line_cog(path)
    assert fitness == 1.0  # zero RMSE -> 1/(1+0) == 1.0


def test_straight_line_noisy_path_scores_lower():
    perfect = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5], [1.0, 1.0, 1.0]])
    noisy = np.array([[0.0, 0.0, 0.0], [0.9, 0.1, 0.5], [1.0, 1.0, 1.0]])
    assert fitness_straight_line_cog(noisy) < fitness_straight_line_cog(perfect)


def test_sinusoidal_bounce_perfect_match():
    t = np.arange(300)
    z = 0.5 + 0.1 * np.sin(t / 100.0)
    path = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    assert fitness_sinusoidal_bounce(path) == 1.0


def test_velocity_forward_rewards_positive_x_displacement():
    path = np.array([[0.0, 0.0, 0.5], [5.0, 1.0, 0.5], [10.0, -1.0, 0.5]])
    assert fitness_velocity_forward(path) == 10.0


def test_velocity_forward_negative_for_backward_motion():
    path = np.array([[5.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    assert fitness_velocity_forward(path) == -5.0


def test_jump_height_finds_peak_above_start():
    path = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.5]])
    assert fitness_jump_height(path) == 2.0

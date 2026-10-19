"""
Tests the fitness MATH in task_fitness.py directly against numpy arrays, without needing
the real Framsticks simulator or framspy installed -- so they run in CI and for every team
member regardless of platform, and catch bugs in the fitness formulas before anyone burns
real evaluation budget on them.
"""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.framsticks_interface.task_fitness import (
    TARGET_LINE_DISTANCE,
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


def test_straight_line_perfect_follower_of_fixed_target_scores_max():
    # The target line is FIXED (start + TARGET_LINE_DISTANCE along +x), not fit from the
    # creature's own start/end -- so a genotype must actually travel toward that fixed
    # goal to score well, not just move in *some* straight line.
    start = np.array([2.0, 3.0, 0.5])
    t = np.linspace(0, 1, 50)
    perfect = start[None, :] + t[:, None] * np.array([TARGET_LINE_DISTANCE, 0.0, 0.0])[None, :]
    assert fitness_straight_line_cog(perfect) == 1.0


def test_straight_line_static_creature_scores_far_below_max():
    # This is the regression test for the bug: a creature that never moves must NOT score
    # the same as a creature that perfectly follows the target line. Previously this
    # scored a false 1.0 because the old formula fit its "target" from the creature's own
    # (identical) start and end points.
    static = np.tile([2.0, 3.0, 0.5], (50, 1))
    fitness = fitness_straight_line_cog(static)
    assert fitness < 0.5, f"static creature scored {fitness}, expected well below max"


def test_straight_line_wrong_direction_scores_below_perfect():
    # Moving confidently in a straight line, but NOT toward the fixed target direction,
    # must score lower than actually heading toward the target -- also previously false
    # (the old formula only checked "is this internally consistent with itself", not
    # "did you go where you were supposed to").
    start = np.array([2.0, 3.0, 0.5])
    t = np.linspace(0, 1, 50)
    sideways = start[None, :] + t[:, None] * np.array([0.0, TARGET_LINE_DISTANCE, 0.0])[None, :]
    perfect = start[None, :] + t[:, None] * np.array([TARGET_LINE_DISTANCE, 0.0, 0.0])[None, :]
    assert fitness_straight_line_cog(sideways) < fitness_straight_line_cog(perfect)


def test_sinusoidal_bounce_perfect_match():
    t = np.arange(300)
    z = 0.5 + 0.1 * np.sin(t / 100.0)
    path = np.stack([np.zeros_like(z), np.zeros_like(z), z], axis=1)
    assert fitness_sinusoidal_bounce(path) == pytest.approx(1.0)


def test_sinusoidal_bounce_static_creature_scores_below_max():
    static = np.tile([0.0, 0.0, 0.5], (300, 1))  # sits exactly at the target's mean height
    assert fitness_sinusoidal_bounce(static) < 1.0


def test_velocity_forward_rewards_positive_x_displacement():
    path = np.array([[0.0, 0.0, 0.5], [5.0, 1.0, 0.5], [10.0, -1.0, 0.5]])
    assert fitness_velocity_forward(path) == 10.0


def test_velocity_forward_negative_for_backward_motion():
    path = np.array([[5.0, 0.0, 0.5], [0.0, 0.0, 0.5]])
    assert fitness_velocity_forward(path) == -5.0


def test_velocity_forward_static_creature_scores_zero():
    static = np.tile([3.0, 3.0, 0.5], (20, 1))
    assert fitness_velocity_forward(static) == 0.0


def test_jump_height_finds_peak_above_start():
    path = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0], [0.0, 0.0, 0.5]])
    assert fitness_jump_height(path) == 2.0


def test_jump_height_static_creature_scores_zero():
    static = np.tile([0.0, 0.0, 1.0], (20, 1))
    assert fitness_jump_height(static) == 0.0


def test_straight_line_two_point_trajectory_warns(capsys):
    # Documents the separate short-trajectory edge case: even with the fixed-target fix,
    # a 2-point trajectory is too short to be meaningful and should be flagged.
    import src.framsticks_interface.task_fitness as tf
    tf._warned_short_trajectory = False  # reset the once-only warning flag for this test

    path = np.array([[0.0, 0.0, 0.0], [5.0, 3.0, 1.0]])
    fitness_straight_line_cog(path)
    captured = capsys.readouterr()
    assert "WARNING" in captured.out
    assert "2 recorded steps" in captured.out

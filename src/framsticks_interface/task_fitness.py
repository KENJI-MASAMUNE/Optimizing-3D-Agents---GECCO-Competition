"""
Task-specific fitness functions over a COG body-recording path. Owner: Vanessa Rampedi

Deliberately has ZERO dependency on framspy/FramsticksLib -- just numpy -- so these can be
unit-tested on every machine and in CI regardless of whether the real simulator is
installed. `competition_lib.py` (which DOES depend on framspy) imports these; nothing here
imports it back.
"""
from __future__ import annotations

import math

import numpy as np

# Below this many recorded COG steps, several of these fitness definitions become
# mathematically trivial rather than meaningful -- see _warn_if_degenerate() below.
MIN_TRAJECTORY_STEPS = 10

# Fixed distance (world units) the desired straight-line path travels along +x, starting
# from wherever the creature actually begins. Tune this to whatever displacement scale is
# realistic for your creatures' typical lifespan/Energy0 -- too large and even good
# locomotion scores near 0; too small and mediocre locomotion saturates near 1.
TARGET_LINE_DISTANCE = 10.0

_warned_short_trajectory = False


def _warn_if_degenerate(path: np.ndarray, task: str) -> None:
    global _warned_short_trajectory
    if len(path) < MIN_TRAJECTORY_STEPS and not _warned_short_trajectory:
        print(
            f"[task_fitness] WARNING: trajectory for task '{task}' has only {len(path)} "
            f"recorded steps (< {MIN_TRAJECTORY_STEPS}). Fitness values are likely "
            "meaningless/trivial. Check Energy0 in your .sim settings to increase "
            "creature lifespan. This warning only prints once per process."
        )
        _warned_short_trajectory = True


def extract_cog_trajectory(raw_path) -> np.ndarray:
    """Normalises `data->bodyrecording` into a plain (n_steps, 3) COG-only array,
    regardless of the `STORE_ALL_PART_COORDS` setting in `recording-body-coords.sim`:

    - STORE_ALL_PART_COORDS=0 (recommended -- see FramsticksLibCompetition.py's own
      docstring): each step is already just `[cogx, cogy, cogz]` -> shape (n_steps, 3).
    - STORE_ALL_PART_COORDS=1 (verbose): each step is
      `[[cogx,cogy,cogz], [p1x,p1y,p1z], ...]` -> shape (n_steps, n_parts+1, 3), with the
      COG always at index 0 of the second axis (per the .sim file's own comment: "after
      the COG, the list of coordinates of all Parts follows").
    """
    arr = np.array(raw_path)
    if arr.ndim == 3:  # verbose format: (n_steps, n_parts+1, 3) -> take COG (index 0)
        return arr[:, 0, :]
    return arr  # already (n_steps, 3)


def fitness_straight_line_cog(path: np.ndarray) -> float:
    """1.0 / RMSE of COG coordinates vs. a FIXED straight line: starting at the creature's
    actual starting position, travelling TARGET_LINE_DISTANCE along +x.

    IMPORTANT DESIGN NOTE: the target line is fixed/external, NOT fit from the creature's
    own start/end points. An earlier version defined target = interpolation between the
    creature's own start and end position -- which meant a creature that never moved at
    all (start == end) trivially scored a perfect 1.0 (RMSE=0 by construction), identical
    to a creature that travelled flawlessly toward a real goal. That made this fitness
    function unable to distinguish "did nothing" from "did great," which is a serious bug
    for a fitness landscape NSGA-II is supposed to climb. This version can't have that
    problem: a stationary creature is meaningfully far from where the fixed target line
    says it should be by the end of the episode, and scores accordingly low.
    """
    path = extract_cog_trajectory(path)
    _warn_if_degenerate(path, "straight_line_cog")
    start = path[0]
    t = np.linspace(0, 1, len(path))
    target = start[None, :] + t[:, None] * np.array([TARGET_LINE_DISTANCE, 0.0, 0.0])[None, :]
    rmse = math.sqrt(float(np.mean(np.sum((path - target) ** 2, axis=1))))
    return 1.0 / (1.0 + rmse)


def fitness_sinusoidal_bounce(path: np.ndarray) -> float:
    """1.0 / sum(abs diff) of the z-coordinate vs 0.5 + 0.1*sin(t/100.0) (Section 2.2).
    Already correctly defined against a fixed absolute target -- no self-referential bug
    here, a stationary creature scores appropriately low."""
    path = extract_cog_trajectory(path)
    _warn_if_degenerate(path, "sinusoidal_bounce")
    z = path[:, 2]
    t = np.arange(len(path))
    target = 0.5 + 0.1 * np.sin(t / 100.0)
    sum_abs_diff = float(np.sum(np.abs(z - target)))
    return 1.0 / (1.0 + sum_abs_diff)


def fitness_velocity_forward(path: np.ndarray) -> float:
    """Net forward (x-axis) displacement of the COG over the episode. Already correctly
    defined -- a stationary creature scores exactly 0.0, not a false maximum."""
    path = extract_cog_trajectory(path)
    _warn_if_degenerate(path, "velocity_forward")
    return float(path[-1, 0] - path[0, 0])


def fitness_jump_height(path: np.ndarray) -> float:
    """Peak vertical (z-axis) displacement of the COG above its starting height. Already
    correctly defined -- a stationary creature scores exactly 0.0."""
    path = extract_cog_trajectory(path)
    _warn_if_degenerate(path, "jump_height")
    return float(np.max(path[:, 2] - path[0, 2]))


TASK_FITNESS_FUNCTIONS = {
    "straight_line_cog": fitness_straight_line_cog,
    "sinusoidal_bounce": fitness_sinusoidal_bounce,
    "velocity_forward": fitness_velocity_forward,
    "jump_height": fitness_jump_height,
}

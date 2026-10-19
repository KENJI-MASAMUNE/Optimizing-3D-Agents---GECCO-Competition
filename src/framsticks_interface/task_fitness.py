"""Phase 1 proxy-task fitness functions (proposal Section 2.2 / 5.3).

Pure numpy: no framspy / Framsticks dependency, so formulas can be unit-tested on every
machine. `extract_cog_trajectory` normalises both body-recording layouts that
`recording-body-coords.sim` can produce.
"""
from __future__ import annotations

import math
from collections.abc import Callable, Sequence

import numpy as np

ArrayLike = Sequence | np.ndarray

# Fixed distance (world units) the desired straight-line path travels along the target heading.
# A stationary creature is meaningfully far from this target and scores low.
TARGET_LINE_DISTANCE = 10.0

# Once-only flag so a short body recording warns on the first occurrence in a process.
_warned_short_trajectory = False


def _warn_if_degenerate(path: np.ndarray, task: str) -> None:
    """Warn once when a trajectory has fewer than 3 recorded steps.

    A 2-point path is too short for the RMSE and displacement scores to mean anything.
    """
    global _warned_short_trajectory
    n = len(path)
    if n >= 3 or _warned_short_trajectory:
        return
    _warned_short_trajectory = True
    print(
        f"WARNING: {task} received only {n} recorded steps; "
        "a trajectory this short is too short to be meaningful."
    )


def extract_cog_trajectory(raw: ArrayLike) -> np.ndarray:
    """Return an `(n_steps, 3)` array of centre-of-gravity coordinates.

    Accepts:
      - COG-only format (`STORE_ALL_PART_COORDS=0`): already `(n_steps, 3)`
      - verbose format (`STORE_ALL_PART_COORDS=1`): `(n_steps, n_parts+1, 3)`
        where index 0 along the parts axis is the COG.
    """
    arr = np.asarray(raw, dtype=float)
    if arr.size == 0:
        return np.zeros((0, 3), dtype=float)
    if arr.ndim == 2 and arr.shape[-1] == 3:
        return arr
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr[:, 0, :]
    return arr  # already (n_steps, 3)


def _heading_vector(heading_deg: float) -> np.ndarray:
    """Unit vector in the x-y plane for a desired travel heading (0 deg = +x, 90 deg = +y).
    Used by worlds with `target_heading_deg` (deterministic stand-in for a rotated start)."""
    h = math.radians(heading_deg)
    return np.array([math.cos(h), math.sin(h), 0.0])


def fitness_straight_line_cog(path: np.ndarray, target_heading_deg: float = 0.0) -> float:
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
    direction = TARGET_LINE_DISTANCE * _heading_vector(target_heading_deg)
    target = start[None, :] + t[:, None] * direction[None, :]
    rmse = math.sqrt(float(np.mean(np.sum((path - target) ** 2, axis=1))))
    return 1.0 / (1.0 + rmse)


SINUSOID_AMPLITUDE = 0.1   # A in z(t) = c + A*sin(t/100): the competition's example shape
SINUSOID_PERIOD_DIV = 100.0


def fitness_sinusoidal_bounce(path: np.ndarray, **_ignored) -> float:
    """How well the COG height follows the oscillation A*sin(t/100) (A = 0.1, the
    competition's example "0.5 + 0.1*sin(t/100.0)"), scored as 1 / (1 + RMSE / A).

    REVISED (Oct 2026, after Phase 1) -- the previous version, 1/(1 + mean|z - target|)
    against the ABSOLUTE target 0.5 + 0.1*sin(t/100), had a ceiling effect: a creature that
    simply holds its COG still at height 0.5 already scores 1/(1 + 0.0637) = 0.940, and a
    perfect follower only 1.0. In Phase 1 every variant and encoding converged to exactly
    that "stand still at 0.5" solution (mean 0.940, std ~0.001), so the task could not
    separate algorithms. It also rewarded raised terrain / floating in water in Phase 3
    worlds, because the target height was absolute.

    Now:
      * the constant offset c is fitted (least squares) instead of fixed at 0.5, so the
        score measures the OSCILLATION, independent of body height, terrain or water;
      * the error is divided by the amplitude A, so the scale is meaningful:
            perfect follower                      -> 1.0
            stationary creature (any height)      -> 1/(1 + 1/sqrt(2)) ~= 0.586
            right amplitude, opposite phase       -> 1/(1 + sqrt(2))   ~= 0.414
        i.e. anything above ~0.59 means the agent genuinely oscillates in time with the target.
    """
    path = extract_cog_trajectory(path)
    _warn_if_degenerate(path, "sinusoidal_bounce")
    z = path[:, 2]
    t = np.arange(len(path))
    shape = SINUSOID_AMPLITUDE * np.sin(t / SINUSOID_PERIOD_DIV)
    residual = z - shape
    residual = residual - residual.mean()          # best-fitting constant offset c
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    return 1.0 / (1.0 + rmse / SINUSOID_AMPLITUDE)


def fitness_velocity_forward(path: np.ndarray, target_heading_deg: float = 0.0) -> float:
    """Net forward displacement of the COG along the target heading (default +x) over the
    episode. A stationary creature scores exactly 0.0, not a false maximum."""
    path = extract_cog_trajectory(path)
    _warn_if_degenerate(path, "velocity_forward")
    return float(np.dot(path[-1] - path[0], _heading_vector(target_heading_deg)))


def fitness_jump_height(path: np.ndarray, **_ignored) -> float:
    """Peak vertical (z-axis) displacement of the COG above its starting height. Already
    correctly defined -- a stationary creature scores exactly 0.0."""
    path = extract_cog_trajectory(path)
    _warn_if_degenerate(path, "jump_height")
    return float(np.max(path[:, 2] - path[0, 2]))


# ---------------------------------------------------------------- competition examples
# The organisers' public example functions (framspy FramsticksLibCompetition.TEST_FUNCTION
# 3, 4, 5). Not part of our four proxy tasks; registered so results can be compared directly
# with published numbers (Brassat 2026 thesis, Table 5) when run under the reference settings
# (Energy0 10000, 100k evaluations) -- see configs/competition_config.yaml.
def fitness_comp_distance(path: np.ndarray, **_ignored) -> float:
    """TEST_FUNCTION 3: Euclidean distance between the COG at birth and at death."""
    path = extract_cog_trajectory(path)
    return float(np.linalg.norm(path[0] - path[-1]))


def fitness_comp_tall_runner(path: np.ndarray, **_ignored) -> float:
    """TEST_FUNCTION 4: distance travelled x mean height of the COG above ground."""
    path = extract_cog_trajectory(path)
    return float(np.linalg.norm(path[0] - path[-1]) * np.mean(np.maximum(0, path[:, 2])))


def fitness_comp_rising_cog(path: np.ndarray, **_ignored) -> float:
    """TEST_FUNCTION 5: COG height should rise linearly from 0 to 10 over the lifespan;
    1000 - RMSE (offset keeps it positive)."""
    path = extract_cog_trajectory(path)
    return float(1000 - np.linalg.norm(np.linspace(0, 10, len(path), endpoint=True) - path[:, 2]) / np.sqrt(len(path)))


TASK_FITNESS_FUNCTIONS = {
    "comp_distance": fitness_comp_distance,
    "comp_tall_runner": fitness_comp_tall_runner,
    "comp_rising_cog": fitness_comp_rising_cog,
    "straight_line_cog": fitness_straight_line_cog,
    "sinusoidal_bounce": fitness_sinusoidal_bounce,
    "velocity_forward": fitness_velocity_forward,
    "jump_height": fitness_jump_height,
}


def fitness_for_task(task: str, path: ArrayLike) -> float:
    """Dispatch a body-recording path to the named Phase 1 fitness function."""
    try:
        fn = TASK_FITNESS_FUNCTIONS[task]
    except KeyError as exc:
        raise ValueError(
            f"Unknown task '{task}'. Options: {sorted(TASK_FITNESS_FUNCTIONS)}"
        ) from exc
    return fn(path)

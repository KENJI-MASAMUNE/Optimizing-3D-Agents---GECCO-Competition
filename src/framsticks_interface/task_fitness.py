"""Phase 1 proxy-task fitness functions (proposal Section 2.2 / 5.3).

Pure numpy: no framspy / Framsticks dependency, so formulas can be unit-tested on every
machine. `extract_cog_trajectory` normalises both body-recording layouts that
`recording-body-coords.sim` can produce.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

ArrayLike = Sequence | np.ndarray


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
    raise ValueError(
        f"Unexpected body-recording shape {arr.shape}; expected (n_steps, 3) "
        "or (n_steps, n_parts+1, 3)"
    )


# Fixed distance (world units) the desired straight-line path travels along +x.
# A stationary creature is meaningfully far from this target and scores low.
TARGET_LINE_DISTANCE = 10.0


def fitness_straight_line_cog(path: ArrayLike) -> float:
    """Reward a COG trajectory that follows a fixed straight line along +x.

    Target is fixed/external: starting at the creature's actual birth position,
    travelling TARGET_LINE_DISTANCE along +x. A stationary creature cannot trivially
    score 1.0 (unlike a start->end interpolation target).

    Fitness = `1 / (1 + RMSE)` against this fixed target line.
    """
    cog = extract_cog_trajectory(path)
    if len(cog) == 0:
        return 0.0
    t = np.linspace(0.0, 1.0, len(cog))
    target = cog[0] + np.outer(t, np.array([TARGET_LINE_DISTANCE, 0.0, 0.0]))
    rmse = float(np.sqrt(np.mean((cog - target) ** 2)))
    return 1.0 / (1.0 + rmse)


def fitness_sinusoidal_bounce(path: ArrayLike) -> float:
    """Reward a COG z-coordinate matching `0.5 + 0.1 * sin(t / 100)`.

    Fitness = `1 / (1 + sum |z - target_z|)`, so a perfect match scores 1.0.
    """
    cog = extract_cog_trajectory(path)
    if len(cog) == 0:
        return 0.0
    t = np.arange(len(cog), dtype=float)
    target_z = 0.5 + 0.1 * np.sin(t / 100.0)
    abs_diff = float(np.sum(np.abs(cog[:, 2] - target_z)))
    return 1.0 / (1.0 + abs_diff)


def fitness_velocity_forward(path: ArrayLike) -> float:
    """Forward displacement of the COG along x over the episode (`x_end - x_start`)."""
    cog = extract_cog_trajectory(path)
    if len(cog) == 0:
        return 0.0
    return float(cog[-1, 0] - cog[0, 0])


def fitness_jump_height(path: ArrayLike) -> float:
    """Peak vertical COG displacement above the starting height (`max(z) - z_start`)."""
    cog = extract_cog_trajectory(path)
    if len(cog) == 0:
        return 0.0
    return float(np.max(cog[:, 2]) - cog[0, 2])


FITNESS_FUNCTIONS: dict[str, Callable[[ArrayLike], float]] = {
    "straight_line_cog": fitness_straight_line_cog,
    "sinusoidal_bounce": fitness_sinusoidal_bounce,
    "velocity_forward": fitness_velocity_forward,
    "jump_height": fitness_jump_height,
}


def fitness_for_task(task: str, path: ArrayLike) -> float:
    """Dispatch a body-recording path to the named Phase 1 fitness function."""
    try:
        fn = FITNESS_FUNCTIONS[task]
    except KeyError as exc:
        raise ValueError(
            f"Unknown task '{task}'. Options: {sorted(FITNESS_FUNCTIONS)}"
        ) from exc
    return fn(path)

"""
Project-specific subclass of the competition's FramsticksLibCompetition.

FramsticksLibCompetition gives us evaluation-budget bookkeeping
(MAX_EVALUATIONS / MAX_TIME enforcement) and result logging for free -- but its own
`_evaluate_path` only implements three toy TEST_FUNCTION examples (distance between
birth/death COG, etc.), not the actual COG trajectory-following fitness functions our
proposal targets (Section 2.2 / 5.3's four proxy tasks). This subclass overrides
`_evaluate_path` to compute OUR fitness definitions (task_fitness.py) from the COG
body-recording path, while keeping all of FramsticksLibCompetition's budget/logging
machinery completely intact.

This module requires `framspy_path` to already be on `sys.path` (RealFramsticksInterface
in interface.py does that) -- it is NOT importable without the real framspy checked out,
by design. The pure fitness math it delegates to (task_fitness.py) has no such dependency
and is unit-tested independently in tests/test_competition_lib.py.

IMPORTANT: `self._lib.end()` (inherited, unmodified) calls `sys.exit()`. It must be called
EXACTLY ONCE, at the very end of a real competition submission run -- NEVER inside a loop
that does many repeats/variants in one process (i.e. never during Phase 1/2 experiments).
See the warning in RealFramsticksInterface (interface.py).
"""
from __future__ import annotations

import numpy as np
from FramsticksLibCompetition import FramsticksLibCompetition  # requires framspy_path on sys.path

from src.framsticks_interface.task_fitness import TASK_FITNESS_FUNCTIONS


class ProjectFramsticksLibCompetition(FramsticksLibCompetition):
    """`current_task` must be set (by RealFramsticksInterface) before each `evaluate()`
    call, since `_evaluate_path` only receives the path array, not the task name."""

    SIMPLE_FITNESS_FORMAT = False  # we want the full result dict (numparts etc.), not just a float
    COMPETITOR_ID = "DENSGA2Team"  # TODO: replace with your actual competition team/user ID

    current_task: str = "straight_line_cog"

    def _evaluate_path(self, path):
        path = np.array(path)
        fn = TASK_FITNESS_FUNCTIONS.get(self.current_task)
        if fn is None:
            raise ValueError(
                f"Unknown task '{self.current_task}'. Known tasks: {list(TASK_FITNESS_FUNCTIONS)}"
            )
        return fn(path)

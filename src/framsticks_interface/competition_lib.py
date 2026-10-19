"""
Project-specific subclass of the competition's FramsticksLibCompetition.

FramsticksLibCompetition gives us evaluation-budget bookkeeping
(MAX_EVALUATIONS / MAX_TIME enforcement) and result logging for free -- but its own
`_evaluate_path` only implements three toy TEST_FUNCTION examples (distance between
birth/death COG, etc.), not the actual COG trajectory-following fitness functions our
proposal targets (Section 2.2 / 5.3's four proxy tasks). This subclass overrides
`_evaluate_path` to compute OUR fitness definitions (task_fitness.py) from the COG
body-recording path, while keeping all of FramsticksLibCompetition's budget/logging
machinery intact.

It also fixes two things in the upstream library WITHOUT editing framspy itself (so a
fresh, unpatched framspy checkout -- e.g. on the CHPC -- behaves identically to the
patched copy on Salmaan's laptop):

1. `getRandomGenotype()` -- upstream picks its target parts/neurons counts with
   `np.random.default_rng().integers(...)`, i.e. a brand-new generator seeded from OS
   entropy on EVERY call. That silently bypasses all seeding and was the root cause of
   "same seed, different results". Verified on the real simulator (Oct 2026): with the
   upstream line two fresh processes produce different initial populations; with a seeded
   generator they are byte-identical. The override below is a line-for-line copy of the
   upstream method except that it draws the two targets from `self.rng`, a seeded
   `np.random.Generator` that `RealFramsticksInterface.reseed()` resets at the start of
   every run. (Drawing integers uniformly from [min, max] is the same distribution as
   upstream, so this changes reproducibility only, not behaviour.)

2. Budget exhaustion -- upstream calls `self.end()` -> `sys.exit()` from inside
   `_evaluate_single_genotype()` when MAX_EVALUATIONS / MAX_TIME is exceeded. Inside a
   batch of many runs that kills the whole process (and every remaining run). Here we
   raise `BudgetExhausted` instead; the engine catches it and finishes the current run
   cleanly with whatever it has found so far, exactly like a competition run that hits
   its limit. `end()` itself is untouched and can still be called once by a dedicated
   submission script.

This module requires `framspy_path` to already be on `sys.path` (RealFramsticksInterface
in interface.py does that) -- it is NOT importable without framspy, by design. The pure
fitness math it delegates to (task_fitness.py) has no such dependency and is unit-tested
independently in tests/test_competition_lib.py.
"""
from __future__ import annotations

from time import perf_counter

import numpy as np
from FramsticksLibCompetition import FramsticksLibCompetition  # requires framspy_path on sys.path

from src.framsticks_interface.interface import BudgetExhausted  # framspy-free definition
from src.framsticks_interface.task_fitness import TASK_FITNESS_FUNCTIONS


class ProjectFramsticksLibCompetition(FramsticksLibCompetition):
    """Same interface as `FramsticksLibCompetition`, with Phase 1 / competition tasks."""

    SIMPLE_FITNESS_FORMAT = False  # we want the full result dict (numparts etc.), not just a float
    COMPETITOR_ID = "DENSGA2Team"

    current_task: str = "straight_line_cog"
    current_task_params: dict = {}  # e.g. {"target_heading_deg": 90.0}, set per world

    def __init__(self, *args, **kwargs):
        # Instance-level budget state (the parent defines these as CLASS attributes, which
        # would otherwise be shared/accumulated across every run in the process).
        self._best_fitness = None
        self._best_solution = None
        self._evaluation_count = 0
        self._evaluation_time = 0.0
        self._time0 = perf_counter()
        self.rng = np.random.default_rng(0)
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------ fitness
    def _evaluate_path(self, path):
        path = np.array(path)
        fn = TASK_FITNESS_FUNCTIONS.get(self.current_task)
        if fn is None:
            raise ValueError(
                f"Unknown task '{self.current_task}'. Known tasks: {list(TASK_FITNESS_FUNCTIONS)}"
            )
        return fn(path, **(self.current_task_params or {}))

    # ------------------------------------------------------------------ budget
    def budget_exceeded(self) -> bool:
        over_evals = self._evaluation_count >= self.MAX_EVALUATIONS
        over_time = perf_counter() - self._time0 - self._evaluation_time > self.MAX_TIME
        return over_evals or over_time

    def _evaluate_single_genotype(self, genotype):
        # Check BEFORE incrementing so exactly MAX_EVALUATIONS evaluations are allowed
        # (upstream increments first and allows MAX_EVALUATIONS, then exits on the next).
        if self.budget_exceeded():
            raise BudgetExhausted(
                f"budget exhausted after {self._evaluation_count} evaluations "
                f"({perf_counter() - self._time0 - self._evaluation_time:.0f}s non-evaluation time)"
            )
        # Below is the parent's logic minus its end()/sys.exit() branch (re-implemented rather
        # than calling super() so there is no code path left that can exit the process).
        self._evaluation_count += 1
        result = super(FramsticksLibCompetition, self).evaluate([genotype])  # FramsticksLib.evaluate
        valid_result = result[0]["evaluations"] is not None
        fitness = self._evaluate_path(result[0]["evaluations"][""]["data->bodyrecording"]) if valid_result else None
        if fitness is not None and (self._best_fitness is None or self._best_fitness < fitness):
            self._best_fitness = fitness
            self._best_solution = genotype
        if self.SIMPLE_FITNESS_FORMAT:
            return fitness
        if valid_result:
            result[0]["evaluations"][""][self.FITNESS_DICT_KEY] = fitness
        return result[0]

    def evaluate(self, genotype_list):
        # Parent's evaluate() only adds to _evaluation_time on success; make sure time spent
        # in a simulation that is interrupted by BudgetExhausted is still excluded.
        eval_time0 = perf_counter()
        try:
            return super().evaluate(genotype_list)
        except BudgetExhausted:
            self._evaluation_time += perf_counter() - eval_time0
            raise

    # ------------------------------------------------------------------ seeded random genotype
    def getRandomGenotype(self, initial_genotype: str, parts_min: int, parts_max: int,
                          neurons_min: int, neurons_max: int, iter_max: int,
                          return_even_if_failed: bool):
        """Copy of FramsticksLib.getRandomGenotype() with the two target draws taken from the
        seeded `self.rng` (see module docstring, point 1)."""
        import frams

        def estimate_diff(g: str):
            if not self.isValidCreature([g])[0]:
                return None, None
            m = frams.Model.newFromString(g)
            numparts = m.numparts._value()
            numneurons = m.numneurons._value()
            diff_parts = abs(target_parts - numparts)
            diff_neurons = abs(target_neurons - numneurons)
            in_target_range = (parts_min <= numparts <= parts_max) and (neurons_min <= numneurons <= neurons_max)
            return diff_parts + diff_neurons, in_target_range

        target_parts = int(self.rng.integers(parts_min, parts_max + 1))
        target_neurons = int(self.rng.integers(neurons_min, neurons_max + 1))

        if not self.isValidCreature([initial_genotype])[0]:
            raise ValueError("Initial genotype '%s' is invalid" % initial_genotype)

        g = initial_genotype
        for _ in range(iter_max // 2):  # undirected mutations
            g_new = self.mutate([g])[0]
            if self.isValidCreature([g_new])[0]:
                g = g_new

        best_diff, best_in_target_range = estimate_diff(g)
        for _ in range(iter_max // 2):  # mutations accepted only if they approach the targets
            g_new = self.mutate([g])[0]
            diff, in_target_range = estimate_diff(g_new)
            if diff is not None and diff <= best_diff:
                g = g_new
                best_diff = diff
                best_in_target_range = in_target_range

        if best_in_target_range or return_even_if_failed:
            return g
        return None

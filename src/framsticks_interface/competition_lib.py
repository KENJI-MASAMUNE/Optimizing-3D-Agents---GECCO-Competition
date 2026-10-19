"""Project-specific `FramsticksLibCompetition` subclass.

Overrides only `_evaluate_path` (and a thin `_evaluate_single_genotype` stash of the
body recording) so the competition library's evaluation-budget / logging machinery stays
untouched. Fitness math lives in `task_fitness.py`.
"""
from __future__ import annotations

from time import perf_counter

from FramsticksLibCompetition import FramsticksLibCompetition

from src.framsticks_interface.task_fitness import extract_cog_trajectory, fitness_for_task


class ProjectFramsticksLibCompetition(FramsticksLibCompetition):
    """Same interface as `FramsticksLibCompetition`, with Phase 1 / competition tasks."""

    SIMPLE_FITNESS_FORMAT = True
    current_task: str = "straight_line_cog"

    def __init__(self, frams_path, frams_lib_name, sim_settings_files):
        super().__init__(frams_path, frams_lib_name, sim_settings_files)
        self.current_task = "straight_line_cog"
        self.last_eval: dict = {
            "cog_trajectory": [],
            "n_parts": 0,
            "raw": {},
        }

    def _evaluate_path(self, path):
        return fitness_for_task(self.current_task, path)

    def _evaluate_single_genotype(self, genotype):
        """Parent implementation plus stash of COG trajectory / numparts for EvaluationResult."""
        self._evaluation_count += 1
        if (
            self._evaluation_count > self.MAX_EVALUATIONS
            or perf_counter() - self._time0 - self._evaluation_time > self.MAX_TIME
        ):
            print("The allowed time or the maximum number of evaluations exceeded")
            self.end()

        result = super(FramsticksLibCompetition, self).evaluate([genotype])
        valid_result = result[0]["evaluations"] is not None
        bodyrecording = []
        n_parts = 0
        fitness = None
        if valid_result:
            data = result[0]["evaluations"].get("", {})
            bodyrecording = data.get("data->bodyrecording", [])
            n_parts = int(data.get("numparts", 0) or 0)
            fitness = self._evaluate_path(bodyrecording)

        cog = extract_cog_trajectory(bodyrecording) if bodyrecording else extract_cog_trajectory([])
        self.last_eval = {
            "cog_trajectory": [tuple(row) for row in cog.tolist()],
            "n_parts": n_parts,
            "raw": result[0],
        }

        if fitness is not None and (self._best_fitness is None or self._best_fitness < fitness):
            self._best_fitness = fitness
            self._best_solution = genotype

        if self.SIMPLE_FITNESS_FORMAT:
            return fitness

        if valid_result:
            result[0]["evaluations"][""][self.FITNESS_DICT_KEY] = fitness
        return result[0]

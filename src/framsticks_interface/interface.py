"""Abstract FramsticksInterface + RealFramsticksInterface adapter.

Algorithms and experiments depend only on this module. The real simulator is optional
at import time; `get_interface()` constructs `RealFramsticksInterface` from config
and raises if `framspy_path` / `frams_dir` are missing.

CRITICAL GOTCHA: `FramsticksLibCompetition` tracks `_evaluation_count`, `_time0`, and
`_best_fitness` as class attributes. Reusing one interface across Phase 1/2/3 repeats
would accumulate toward `MAX_EVALUATIONS` / `MAX_TIME` and could `sys.exit()` the batch.
Call `reset_evaluation_budget()` before every run (`run_single.py` does this).
"""
from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from time import perf_counter
from typing import Any

from src.framsticks_interface.encodings import format_code


@dataclass
class EvaluationResult:
    fitness: float
    cog_trajectory: list[tuple[float, float, float]]
    n_parts: int
    is_valid: bool
    raw: dict[str, Any] = field(default_factory=dict)


class FramsticksInterface(ABC):
    """Simulator-facing operations used by the evolutionary engine."""

    @abstractmethod
    def get_random_genotype(self, encoding: str) -> str: ...

    @abstractmethod
    def mutate(self, genotype: str, encoding: str) -> str: ...

    @abstractmethod
    def crossover(self, g1: str, g2: str, encoding: str) -> tuple[str, str]: ...

    @abstractmethod
    def is_viable(self, genotype: str, encoding: str) -> bool: ...

    @abstractmethod
    def pairwise_dissimilarity(
        self, genotypes: list[str], encoding: str, measure: str
    ) -> list[list[float]]: ...

    @abstractmethod
    def evaluate(
        self,
        genotype: str,
        encoding: str,
        task: str,
        world_config: dict | None = None,
    ) -> EvaluationResult: ...


class RealFramsticksInterface(FramsticksInterface):
    """Wraps `ProjectFramsticksLibCompetition` (and thus FramsticksLib)."""

    def __init__(
        self,
        framspy_path: str,
        frams_dir: str,
        lib_name: str | None = None,
        sim_files: str | None = None,
        graph_dissimilarity_mode: str = "greedy",
        max_evaluations: int | None = None,
        max_time: float | None = None,
        competitor_id: str | None = None,
        parts_min: int = 2,
        parts_max: int = 8,
        neurons_min: int = 1,
        neurons_max: int = 4,
        random_iter_max: int = 100,
        random_seed: int = 42,
    ):
        self._ensure_framspy_on_path(framspy_path)
        from FramsticksLib import DissimMethod

        from src.framsticks_interface.competition_lib import ProjectFramsticksLibCompetition

        self._DissimMethod = DissimMethod
        self._graph_mode = graph_dissimilarity_mode
        self._parts_min = parts_min
        self._parts_max = parts_max
        self._neurons_min = neurons_min
        self._neurons_max = neurons_max
        self._random_iter_max = random_iter_max

        self.lib = ProjectFramsticksLibCompetition(frams_dir, lib_name, sim_files)
        if max_evaluations is not None:
            self.lib.MAX_EVALUATIONS = int(max_evaluations)
        if max_time is not None:
            self.lib.MAX_TIME = float(max_time)
        if competitor_id is not None:
            self.lib.COMPETITOR_ID = competitor_id
        self._seed_native_rng(random_seed)
        self.reset_evaluation_budget()

    @staticmethod
    def _ensure_framspy_on_path(framspy_path: str) -> None:
        path = str(Path(framspy_path).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)

    @property
    def n_evaluations(self) -> int:
        return int(getattr(self.lib, "_evaluation_count", 0))

    def _seed_native_rng(self, seed: int) -> None:
        """Pin Framsticks' native RNG via `frams.Math.seed` (property assignment).

        Must be called after construction/.sim loading but before any mutation/crossover/
        evaluation. `DETERMINISTIC=True` alone is not sufficient -- confirmed empirically
        that two fresh processes with only DETERMINISTIC=True still produce different
        genotypes. `frams.Math.seed = value` is the actual fix.
        """
        try:
            import frams
            frams.Math.seed = seed
        except Exception:
            pass  # frams not available (e.g. CI / test environment) -- silently skip

    def reset_evaluation_budget(self) -> None:
        """Zero competition counters so batch repeats do not leak evaluations or wall time."""
        self.lib._evaluation_count = 0
        self.lib._evaluation_time = 0
        self.lib._time0 = perf_counter()
        self.lib._best_fitness = None
        self.lib._best_solution = None

    def get_random_genotype(self, encoding: str) -> str:
        initial = self.lib.getSimplest(format_code(encoding))
        genotype = self.lib.getRandomGenotype(
            initial,
            self._parts_min,
            self._parts_max,
            self._neurons_min,
            self._neurons_max,
            self._random_iter_max,
            True,
        )
        return genotype if genotype is not None else initial

    def mutate(self, genotype: str, encoding: str) -> str:
        return self.lib.mutate([genotype])[0]

    def crossover(self, g1: str, g2: str, encoding: str) -> tuple[str, str]:
        # FramsticksLib.crossOver returns one offspring; swap parents for a second child
        # so the engine's (g1, g2) unpacking stays valid.
        child1 = self.lib.crossOver(g1, g2)
        child2 = self.lib.crossOver(g2, g1)
        return child1, child2

    def is_viable(self, genotype: str, encoding: str) -> bool:
        if not self.lib.isValidCreature([genotype])[0]:
            return False
        return self.lib.satisfiesConstraints(
            genotype,
            self._parts_max * 2,   # MAX_NUMPARTS
            self._parts_max * 3,   # MAX_NUMJOINTS
            self._neurons_max * 2, # MAX_NUMNEURONS
            self._neurons_max * 6, # MAX_NUMCONNECTIONS
            500,                   # MAX_NUMGENOCHARS
        )

    def pairwise_dissimilarity(
        self, genotypes: list[str], encoding: str, measure: str
    ) -> list[list[float]]:
        if measure == "gene":
            method = self._DissimMethod.GENE_LEVENSHTEIN
        elif measure == "graph":
            if self._graph_mode == "optim":
                method = self._DissimMethod.PHENE_STRUCT_OPTIM
            else:
                method = self._DissimMethod.PHENE_STRUCT_GREEDY
        else:
            raise ValueError(f"Unknown dissimilarity measure '{measure}'")
        matrix = self.lib.dissimilarity(genotypes, method)
        return matrix.tolist()

    def evaluate(
        self,
        genotype: str,
        encoding: str,
        task: str,
        world_config: dict | None = None,
    ) -> EvaluationResult:
        self.lib.current_task = task
        fitnesses = self.lib.evaluate([genotype])
        fitness = fitnesses[0]
        extras = getattr(self.lib, "last_eval", {})
        is_valid = fitness is not None
        return EvaluationResult(
            fitness=float(fitness) if is_valid else 0.0,
            cog_trajectory=list(extras.get("cog_trajectory", [])),
            n_parts=int(extras.get("n_parts", 0) or 0),
            is_valid=is_valid,
            raw=extras.get("raw", {}),
        )

    def finalize_submission(self) -> None:
        """Call once at the end of a dedicated competition submission process.

        Wraps `FramsticksLibCompetition.end()`, which logs the best solution and
        then `sys.exit()`. Never call this from phase1/2/3 batch scripts.
        """
        self.lib.end()


def compose_sim_files(config: dict) -> str | None:
    """Base `sim_files` plus an optional named world overlay.

    Overlays must be last so they override wrldg / wrldwat / wrldtyp from
    eval-allcriteria.sim. Files have to exist in Framsticks' data/ directory
    (`python scripts/sync_sim_files.py`).
    """
    fs = config.get("framsticks") or {}
    sim_files = fs.get("sim_files")
    active = fs.get("active_environment")
    if not active:
        return sim_files
    overlay = None
    for env in config.get("environments") or []:
        if env.get("name") == active:
            overlay = env.get("sim_overlay")
            break
    if not overlay:
        names = [e.get("name") for e in (config.get("environments") or [])]
        raise ValueError(
            f"Unknown active_environment '{active}'. Known environments: {names}"
        )
    parts = [p for p in str(sim_files or "").split(";") if p]
    if overlay not in parts:
        parts.append(overlay)
    return ";".join(parts)


def get_interface_for_world(config: dict, world_sim_overlay: str) -> RealFramsticksInterface:
    """Construct a fresh interface with a world overlay appended to the base sim_files.

    Used by Phase 3 generalisation evaluation: one interface per held-out world.
    `world_sim_overlay` is just the filename, e.g. 'world-high-gravity.sim'.
    """
    fs = config.get("framsticks") or {}
    base = compose_sim_files(config) or ""
    parts = [p for p in base.split(";") if p]
    if world_sim_overlay not in parts:
        parts.append(world_sim_overlay)
    sim_files = ";".join(parts)
    framspy_path = fs.get("framspy_path")
    frams_dir = fs.get("frams_dir")
    if not framspy_path or not frams_dir:
        raise ValueError(
            "framsticks.framspy_path and framsticks.frams_dir must be set in the "
            "config. See docs/framsticks_integration_notes.md."
        )
    return RealFramsticksInterface(
        framspy_path=framspy_path,
        frams_dir=frams_dir,
        lib_name=fs.get("lib_name"),
        sim_files=sim_files,
        graph_dissimilarity_mode=fs.get("graph_dissimilarity_mode", "greedy"),
        max_evaluations=fs.get("eval_budget"),
        max_time=fs.get("time_budget_seconds"),
        competitor_id=fs.get("competitor_id"),
        random_seed=config.get("experiment", {}).get("random_seed_base", 42),
    )


def get_interface(config: dict) -> RealFramsticksInterface:
    """Always construct the real backend from `config["framsticks"]`."""
    fs = config.get("framsticks") or {}
    framspy_path = fs.get("framspy_path")
    frams_dir = fs.get("frams_dir")
    if not framspy_path or not frams_dir:
        raise ValueError(
            "framsticks.framspy_path and framsticks.frams_dir must be set in the "
            "config (no mock backend). See docs/framsticks_integration_notes.md."
        )
    return RealFramsticksInterface(
        framspy_path=framspy_path,
        frams_dir=frams_dir,
        lib_name=fs.get("lib_name"),
        sim_files=compose_sim_files(config),
        graph_dissimilarity_mode=fs.get("graph_dissimilarity_mode", "greedy"),
        max_evaluations=fs.get("eval_budget"),
        max_time=fs.get("time_budget_seconds"),
        competitor_id=fs.get("competitor_id"),
        random_seed=config.get("experiment", {}).get("random_seed_base", 42),
    )

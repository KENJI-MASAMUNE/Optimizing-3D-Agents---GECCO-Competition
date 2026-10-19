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
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any


class BudgetExhausted(Exception):
    """Raised by an interface's evaluate() when the evaluation-count or time budget for the
    current run is used up. The engine catches it and ends the run gracefully (instead of
    the upstream library's sys.exit(), which would kill a whole batch of runs)."""


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
    def pairwise_dissimilarity(self, genotypes: list[str], encoding: str, measure: str) -> list[list[float]]:
        """Full pairwise dissimilarity matrix for a population. `measure` is 'gene' or
        'graph'. This is what feeds DE-NSGA-II crowding distance (Section 2.4) -- backing
        both algorithms/engine.py's crowding calculation AND its per-generation diversity
        objective."""
        ...


_ENCODING_TO_FRAMS_FORMAT = {"f0": "0", "f1": "1"}

PROJECT_SIM_DIR = Path(__file__).resolve().parents[2] / "framsticks_sim"
DEFAULT_SIM_FILES = "eval-allcriteria.sim;deterministic.sim;recording-cog-only.sim"


def resolve_sim_files(sim_files: str) -> str:
    """Each ';'-separated entry that exists in the repo's `framsticks_sim/` directory is
    replaced by its absolute path, so the evaluation settings are versioned with the code
    instead of depending on whatever happens to be in each machine's Framsticks `data/`
    directory (Salmaan's local eval-allcriteria.sim had been hand-edited, for example).
    Entries not found there are passed through unchanged (Framsticks then looks in its own
    data/ directory, as before)."""
    resolved = []
    for name in sim_files.split(";"):
        name = name.strip()
        if not name:
            continue
        candidate = PROJECT_SIM_DIR / name
        resolved.append(str(candidate) if candidate.is_file() else name)
    return ";".join(resolved)


def _expected_first_gaussian(frams, seed: int) -> float:
    """First value Framsticks' Math.rndGaussStd returns from a CLEAN state after
    `Math.seed = seed`. Framsticks uses the Marsaglia polar method: draw v1, v2 uniform in
    (-1, 1) until 0 < s = v1^2 + v2^2 < 1, return v1*f and CACHE v2*f, f = sqrt(-2 ln s / s)
    (verified against the 5.5 binary). Computed from uniform draws, which never touch the
    Gaussian cache."""
    import math
    frams.Math.seed = int(seed)
    while True:
        v1 = 2.0 * frams.Math.rnd01._double() - 1.0
        v2 = 2.0 * frams.Math.rnd01._double() - 1.0
        rsq = v1 * v1 + v2 * v2
        if 0.0 < rsq < 1.0:
            return v1 * math.sqrt(-2.0 * math.log(rsq) / rsq)


def reset_native_rng(frams, seed: int) -> bool:
    """Puts the native Framsticks RNG into the state "freshly seeded with `seed`, empty
    Gaussian cache", regardless of what happened before.

    WHY: `Math.seed = s` resets the uniform generator but NOT the cached second value of
    the polar-method Gaussian pair (confirmed: seed 42 -> rndGaussStd, repeated, returns
    -0.517, 1.222, -0.517, 1.222, ...). Mutation uses Gaussian draws, so after a reseed
    the first Gaussian a run sees depended on the parity of Gaussian draws made by
    whatever ran earlier in the same process -- e.g. a run gave best fitness 0.148 when
    executed first and 0.219 when executed after a different run. Within one fresh
    process per run (the original verify_reproducibility.py setup) this never showed up,
    which is why the earlier two-process check passed.

    HOW: compute the clean-state first Gaussian X from uniforms, reseed, draw once.
      - result == X  -> cache was empty; the draw generated a new pair and cached its
                        second value, so reseed and draw once more to consume it.
      - result != X  -> the draw consumed a stale cached value without touching the
                        uniform generator; the state is already clean.
    Returns True if the post-condition was verified (next Gaussian after reseed == X)."""
    x = _expected_first_gaussian(frams, seed)
    frams.Math.seed = int(seed)
    g = frams.Math.rndGaussStd._double()
    if abs(g - x) <= 1e-12 * max(1.0, abs(x)):
        frams.Math.seed = int(seed)
        frams.Math.rndGaussStd._double()  # consume the spare generated by our own draw
    # verify: from a clean state the next Gaussian is x; then consume its spare again
    frams.Math.seed = int(seed)
    ok = abs(frams.Math.rndGaussStd._double() - x) <= 1e-12 * max(1.0, abs(x))
    frams.Math.seed = int(seed)
    frams.Math.rndGaussStd._double()  # consumes the spare cached by the verification draw
    frams.Math.seed = int(seed)
    return ok


class RealFramsticksInterface(FramsticksInterface):
    """Adapter over the real `framspy` / `FramsticksLibCompetition` (via our subclass
    `ProjectFramsticksLibCompetition` in competition_lib.py).

    API mapping (confirmed against FramsticksLib.py):
      get_random_genotype -> getRandomGenotype (seeded override in competition_lib.py)
      mutate / crossover  -> mutate / crossOver (crossOver returns ONE child per call)
      is_viable           -> isValidCreature + satisfiesConstraints (no simulation, no budget)
      pairwise_dissimilarity -> dissimilarity(GENE_LEVENSHTEIN | PHENE_STRUCT_GREEDY/OPTIM)
      evaluate            -> evaluate (counts against the budget)

    REPRODUCIBILITY (validated on the real simulator, Oct 2026)
      Three RNGs matter: Python `random` (engine + FramsticksLib.crossOver fallback),
      NumPy (getRandomGenotype targets) and the native Framsticks RNG (mutation, crossover).
      * The native RNG is already deterministic after frams.init() (Math.seed == 0, and
        identical mutation sequences across fresh processes) as long as nothing calls
        Math.randomize() -- which FramsticksLib does unless DETERMINISTIC=True.
      * Assigning `frams.Math.seed = s` (a property, not a method) re-seeds it, and it can
        be re-assigned mid-process: seed 42 -> 30 mutations -> seed 42 -> 30 mutations gives
        identical genotypes.
      * The real cause of "same seed, different results" was upstream getRandomGenotype()
        using np.random.default_rng() (OS entropy). Confirmed: unpatched -> different initial
        populations across processes; patched -> identical. The patch to framspy is
        therefore VALID, but we no longer rely on it: competition_lib.py overrides the
        method with a seeded generator, so an unpatched framspy (e.g. on the CHPC) behaves
        the same.
      * BUT the native Gaussian generator caches the second value of each pair and
        `Math.seed = s` does not clear that cache, so a reseed alone is NOT a full reset
        (found and fixed in this revision -- see reset_native_rng()).
      `reseed(seed)` resets ALL THREE (including the Gaussian cache) plus the
      getRandomGenotype generator. run_single()
      calls it at the start of every run, so each run's result depends only on its own
      seed -- not on which runs happened earlier in the same process. That is what makes
      CHPC job arrays / process pools safe: any run can be re-executed in isolation and
      reproduce bit-for-bit.

    BUDGET
      Budget state is per-instance (the parent defines it at class level) and is reset by
      reset_evaluation_budget(). Exhaustion raises competition_lib.BudgetExhausted (the
      engine stops the run gracefully) instead of sys.exit()-ing the whole batch.

    WORLDS
      `evaluate(..., world_config=...)` now really changes the simulated world (gravity,
      water, terrain, lifespan, body constraints, target heading) -- see worlds.py. `None`
      means the base training world, which is (sim files) + `base_world` overrides from
      the config, snapshotted at construction.
    """

    RANDOM_GENOTYPE_PARTS_RANGE = (3, 15)
    RANDOM_GENOTYPE_NEURONS_RANGE = (1, 10)
    RANDOM_GENOTYPE_ITER_MAX = 100

    # Bloat-prevention caps, enforced in is_viable() via satisfiesConstraints() -- no
    # simulation, no budget cost. None disables a constraint.
    MAX_NUMPARTS: int | None = 30
    MAX_NUMJOINTS: int | None = 40
    MAX_NUMNEURONS: int | None = 20
    MAX_NUMCONNECTIONS: int | None = 60
    # Genotype-length cap PER ENCODING. A single 500-char cap (the previous setting) bound f0
    # far more tightly than f1: f0 spells out every part/joint/neuron on its own line
    # ("p:x, y, z, ..."), so full-size f0 runs ended with best genotypes of 450-499 chars
    # at ~15 parts, while f1 encodes a similar body in well under 200 chars. That made the
    # f0-vs-f1 comparison (research question 4) depend on an arbitrary string limit. The
    # parts/joints/neurons/connections caps above already bound phenotype size (and hence
    # simulation cost) identically for both encodings, so f0 gets a generous char cap.
    MAX_NUMGENOCHARS_BY_ENCODING: dict = {"f0": 3000, "f1": 500}

    def __init__(
        self,
        framspy_path: str,
        frams_dir: str,
        lib_name: str | None = None,
        sim_files: str = DEFAULT_SIM_FILES,
        eval_budget: int = 100_000,
        time_budget_seconds: int = 3600,
        graph_dissimilarity_mode: str = "greedy",  # "greedy" (fast, approximate) or "optim" (slow, exact)
        deterministic: bool = True,
        random_seed: int = 42,
        base_world: dict | None = None,
        quiet: bool = True,
    ):
        import contextlib
        import io

        if framspy_path not in sys.path:
            sys.path.append(framspy_path)
        try:
            from src.framsticks_interface.competition_lib import ProjectFramsticksLibCompetition
        except ImportError as e:
            raise ImportError(
                "Could not import FramsticksLibCompetition from framspy_path="
                f"'{framspy_path}'. Confirm framspy was checked out there and that "
                "FramsticksLib.py / FramsticksLibCompetition.py are directly inside it."
            ) from e

        from FramsticksLib import FramsticksLib
        FramsticksLib.DETERMINISTIC = deterministic  # prevents Math.randomize() in __init__
        self._deterministic = deterministic

        sim_files = resolve_sim_files(sim_files)
        # FramsticksLib prints the full dir(frams) listing and every .sim comment; keep the
        # job logs readable on the cluster unless explicitly asked for.
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf) if quiet else contextlib.nullcontext():
            self._lib = ProjectFramsticksLibCompetition(frams_dir, lib_name, sim_files)
        self.sim_files = sim_files

        import frams
        self._frams = frams

        # Project-wide base world (e.g. Energy0) applied on top of the .sim files so every
        # machine trains in exactly the same world.
        from src.framsticks_interface.worlds import FRAMS_PROPERTY_MAP, FramsWorldApplier, world_from_dict
        if base_world:
            bw = world_from_dict("base_world", base_world)
            for k, v in bw.params.items():
                obj_name, prop = FRAMS_PROPERTY_MAP[k]
                setattr(getattr(frams, obj_name), prop, v)
        self._world_applier = FramsWorldApplier(frams)
        self._current_world = None

        self._eval_budget = eval_budget
        self._time_budget_seconds = time_budget_seconds
        self._graph_dissimilarity_mode = graph_dissimilarity_mode
        self._graph_cache: dict[tuple[str, str], float] = {}
        self._ended = False
        self.current_seed: int | None = None
        self.reseed(random_seed)
        print(f"[RealFramsticksInterface] ready: sims={sim_files} base_world="
              f"{ {k: v for k, v in self._world_applier.base_values.items() if k in ('energy0', 'gravity', 'water_level', 'world_type')} } "
              f"seed={random_seed} deterministic={deterministic}", flush=True)
        self.reset_evaluation_budget()

    # ------------------------------------------------------------------ seeding / budget
    def reseed(self, seed: int) -> None:
        """Puts every RNG that influences a run into a state determined by `seed` alone."""
        import random

        import numpy as np

        random.seed(seed)
        np.random.seed(seed % (2 ** 32))
        self._lib.rng = np.random.default_rng(seed)
        if self._deterministic:
            if not reset_native_rng(self._frams, int(seed)):
                print("[RealFramsticksInterface] WARNING: could not verify the native RNG reset "
                      "(different Framsticks version?) -- runs may depend on execution order.",
                      flush=True)
        self.current_seed = seed
        self.clear_caches()

    def reset_evaluation_budget(self) -> None:
        """Resets evaluation count / timers / best-so-far. Call before every run that reuses
        this interface (run_single() does)."""
        from time import perf_counter

        self._lib._evaluation_count = 0
        self._lib._evaluation_time = 0.0
        self._lib._time0 = perf_counter()
        self._lib._best_fitness = None
        self._lib._best_solution = None
        self._lib.MAX_EVALUATIONS = self._eval_budget
        self._lib.MAX_TIME = self._time_budget_seconds
        self._ended = False

    def set_budget(self, eval_budget: int | None = None, time_budget_seconds: int | None = None) -> None:
        if eval_budget is not None:
            self._eval_budget = eval_budget
            self._lib.MAX_EVALUATIONS = eval_budget
        if time_budget_seconds is not None:
            self._time_budget_seconds = time_budget_seconds
            self._lib.MAX_TIME = time_budget_seconds

    def budget_exceeded(self) -> bool:
        return self._lib.budget_exceeded()

    @property
    def n_evaluations(self) -> int:
        return self._lib._evaluation_count

    @property
    def base_world_values(self) -> dict:
        return self._world_applier.base_values

    # ------------------------------------------------------------------ genetics
    def _frams_format(self, encoding: str) -> str:
        if encoding not in _ENCODING_TO_FRAMS_FORMAT:
            raise ValueError(f"Unknown encoding '{encoding}', expected 'f0' or 'f1'")
        return _ENCODING_TO_FRAMS_FORMAT[encoding]

    def get_random_genotype(self, encoding: str) -> str:
        fmt = self._frams_format(encoding)
        simplest = self._lib.getSimplest(fmt)
        p_min, p_max = self.RANDOM_GENOTYPE_PARTS_RANGE
        n_min, n_max = self.RANDOM_GENOTYPE_NEURONS_RANGE
        return self._lib.getRandomGenotype(
            simplest, p_min, p_max, n_min, n_max,
            self.RANDOM_GENOTYPE_ITER_MAX, return_even_if_failed=True,
        )

    def get_simplest_genotype(self, encoding: str) -> str:
        """Simplest genotype of the encoding (frams GenMan.getSimplest) -- the Simple EA's
        initial population, as in framspy's FramsticksEvolution.py."""
        return self._lib.getSimplest(self._frams_format(encoding))

    def mutate(self, genotype: str, encoding: str) -> str:
        return self._lib.mutate([genotype])[0]

    def crossover(self, g1: str, g2: str, encoding: str) -> tuple[str, str]:
        return self._lib.crossOver(g1, g2), self._lib.crossOver(g1, g2)

    def is_viable(self, genotype: str, encoding: str) -> bool:
        if not self._lib.isValidCreature([genotype])[0]:
            return False
        return self._lib.satisfiesConstraints(
            genotype, self.MAX_NUMPARTS, self.MAX_NUMJOINTS, self.MAX_NUMNEURONS,
            self.MAX_NUMCONNECTIONS, self.MAX_NUMGENOCHARS_BY_ENCODING.get(encoding, 500),
        )

    # ------------------------------------------------------------------ dissimilarity
    def pairwise_dissimilarity(self, genotypes: list[str], encoding: str, measure: str) -> list[list[float]]:
        """Full symmetric dissimilarity matrix.

        gene  -> Levenshtein (raw edit distance; via python-Levenshtein, fast).
        graph -> Framsticks' SimilMeasure (PHENE_STRUCT_GREEDY by default). Implements the
                 proposal's Variant 3 "memorised cache" safeguard: each unordered genotype
                 pair is computed once per run and reused in later generations (elitism
                 keeps most of the population unchanged, so most pairs repeat). The cache
                 is cleared in reset_evaluation_budget()/reseed via clear_caches().
        The matrix is symmetrised ((D + D^T) / 2): Framsticks' structural measure is
        symmetric up to ~1e-15 floating-point noise, which otherwise triggers a warning
        on every call."""
        import numpy as np

        if measure == "gene":
            import Levenshtein
            n = len(genotypes)
            m = np.zeros((n, n))
            for i in range(n):
                for j in range(i + 1, n):
                    m[i, j] = m[j, i] = Levenshtein.distance(genotypes[i], genotypes[j])
            return m.tolist()
        if measure != "graph":
            raise ValueError(f"Unknown dissimilarity measure '{measure}', expected 'gene' or 'graph'")

        frams = self._frams
        frams.SimilMeasure.simil_type = 1 if self._graph_dissimilarity_mode == "optim" else 0
        n = len(genotypes)
        m = np.zeros((n, n))
        genos: dict[str, object] = {}
        evaluate_distance = frams.SimilMeasure.evaluateDistance
        for i in range(n):
            for j in range(i + 1, n):
                a, b = genotypes[i], genotypes[j]
                if a == b:
                    continue
                key = (a, b) if a < b else (b, a)
                d = self._graph_cache.get(key)
                if d is None:
                    for g in key:
                        if g not in genos:
                            genos[g] = frams.Geno.newFromString(g)
                    ga, gb = genos[key[0]], genos[key[1]]
                    d = 0.5 * (evaluate_distance(ga, gb)._double() + evaluate_distance(gb, ga)._double())
                    self._graph_cache[key] = d
                m[i, j] = m[j, i] = d
        if len(self._graph_cache) > 2_000_000:  # memory guard (~200 MB)
            self._graph_cache.clear()
        return m.tolist()

    def clear_caches(self) -> None:
        self._graph_cache.clear()

    # ------------------------------------------------------------------ evaluation
    def set_world(self, world_config) -> None:
        """Applies a world (WorldConfig / dict / registered name / None = base world)."""
        from src.framsticks_interface.worlds import coerce_world

        world = coerce_world(world_config)
        self._world_applier.apply(world)
        self._lib.current_task_params = dict(world.task_params) if world else {}
        self._current_world = world

    def evaluate(self, genotype: str, encoding: str, task: str, world_config=None) -> EvaluationResult:
        if world_config is not self._current_world:
            self.set_world(world_config)

        self._lib.current_task = task
        r = self._lib.evaluate([genotype])[0]  # may raise BudgetExhausted
        evaluations = r.get("evaluations")
        if evaluations is None:
            # Not buildable/simulable in this world (e.g. violates min/max joint length).
            return EvaluationResult(fitness=float("nan"), cog_trajectory=[], n_parts=0, is_valid=False, raw=r)

        data = evaluations[""]
        fitness = data.get(self._lib.FITNESS_DICT_KEY)
        raw_path = data.get("data->bodyrecording", [])
        if raw_path:
            from src.framsticks_interface.task_fitness import extract_cog_trajectory
            trajectory = [tuple(row) for row in extract_cog_trajectory(raw_path)]
        else:
            trajectory = []
        n_parts = data.get("numparts", 0)
        if fitness is None:
            return EvaluationResult(fitness=float("nan"), cog_trajectory=trajectory,
                                    n_parts=int(n_parts or 0), is_valid=False, raw=data)
        return EvaluationResult(
            fitness=float(fitness),
            cog_trajectory=trajectory,
            n_parts=int(n_parts or 0),
            is_valid=True,
            raw=data,
        )

    def finalize_submission(self) -> None:
        """Calls FramsticksLibCompetition.end() (logs the best solution, writes the
        *.results file, then sys.exit()s). ONLY for a dedicated final-submission script."""
        if not self._ended:
            self._ended = True
            self._lib.end()


def get_interface(config: dict, quiet: bool = True) -> FramsticksInterface:
    """Factory: constructs a `RealFramsticksInterface` from config['framsticks']. Paths may
    also come from the environment variables FRAMSPY_PATH / FRAMS_DIR (used on the CHPC so
    the same YAML works on every machine)."""
    import os

    fconf = dict(config["framsticks"])
    fconf["framspy_path"] = os.environ.get("FRAMSPY_PATH") or fconf.get("framspy_path")
    fconf["frams_dir"] = os.environ.get("FRAMS_DIR") or fconf.get("frams_dir")
    missing = [k for k in ("framspy_path", "frams_dir") if not fconf.get(k)]
    if missing:
        raise ValueError(
            f"framsticks.{missing[0]} is not set (and no FRAMSPY_PATH/FRAMS_DIR env var). "
            "Set it in your config (see docs/framsticks_integration_notes.md) or export the "
            "environment variables."
        )
    return RealFramsticksInterface(
        framspy_path=fconf["framspy_path"],
        frams_dir=fconf["frams_dir"],
        lib_name=fconf.get("lib_name"),
        sim_files=fconf.get("sim_files", DEFAULT_SIM_FILES),
        eval_budget=fconf.get("eval_budget", 100_000),
        time_budget_seconds=fconf.get("time_budget_seconds", 3600),
        graph_dissimilarity_mode=fconf.get("graph_dissimilarity_mode", "greedy"),
        deterministic=fconf.get("deterministic", True),
        random_seed=fconf.get("random_seed", config.get("experiment", {}).get("random_seed_base", 42)),
        base_world=fconf.get("base_world"),
        quiet=quiet,
    )

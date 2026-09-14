"""
FramsticksInterface: the single seam every algorithm/experiment module talks to.

Algorithm code (src/algorithms/*) and experiment code (src/experiments/*) must ONLY import
`FramsticksInterface` / `get_interface`, never a framspy binding directly. `RealFramsticksInterface`
is the only production implementation. `tests/fakes.py` has a lightweight test-only double
(not exposed here, not part of the public API) so the test suite can exercise engine.py's
algorithm logic without the real simulator installed.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any


@dataclass
class EvaluationResult:
    """Result of evaluating one genotype in one world/task setting."""

    fitness: float
    cog_trajectory: list[tuple[float, float, float]]
    n_parts: int
    is_valid: bool
    raw: dict[str, Any] | None = None


class FramsticksInterface(ABC):
    """Abstract interface matching the shape of the competition's
    `FramsticksLibCompetition` class (Section 5.2). `RealFramsticksInterface` (below) is
    the only production implementation; `tests/fakes.py::FakeFramsticksInterface` is a
    test-only double used by the test suite."""

    @abstractmethod
    def get_random_genotype(self, encoding: str) -> str:
        ...

    @abstractmethod
    def mutate(self, genotype: str, encoding: str) -> str:
        ...

    @abstractmethod
    def crossover(self, g1: str, g2: str, encoding: str) -> tuple[str, str]:
        ...

    @abstractmethod
    def evaluate(self, genotype: str, encoding: str, task: str, world_config: dict | None = None) -> EvaluationResult:
        ...

    @abstractmethod
    def is_viable(self, genotype: str, encoding: str) -> bool:
        """Minimal-Viable Morphological Filter (Section 5.5): checks a non-zero count of
        interconnected muscles/joints and a balanced geometric layout before the genotype
        is allowed to consume simulation budget."""
        ...

    @abstractmethod
    def pairwise_dissimilarity(self, genotypes: list[str], encoding: str, measure: str) -> list[list[float]]:
        """Full pairwise dissimilarity matrix for a population. `measure` is 'gene' or
        'graph'. This is what feeds DE-NSGA-II crowding distance (Section 2.4) -- backing
        both algorithms/engine.py's crowding calculation AND its per-generation diversity
        objective."""
        ...


_ENCODING_TO_FRAMS_FORMAT = {"f0": "0", "f1": "1"}


class RealFramsticksInterface(FramsticksInterface):
    """Adapter over the real `framspy` / `FramsticksLibCompetition`, now implemented
    against the confirmed `FramsticksLib.py` base-class API (constructor, mutate,
    crossOver, evaluate, isValid/isValidCreature, getRandomGenotype, dissimilarity).

    Wraps `ProjectFramsticksLibCompetition` (src/framsticks_interface/competition_lib.py),
    our subclass that supplies the actual COG-trajectory fitness functions for the four
    Phase 1 proxy tasks (Section 5.3), on top of FramsticksLibCompetition's built-in
    evaluation-budget bookkeeping and result logging.

    RESOLVED (against FramsticksLib.py, confirmed):
    - Constructor: `FramsticksLib(frams_path, frams_lib_name, sim_settings_files)` --
      `frams_path` IS the directory containing the frams-objects library (no trailing
      slash), matching what we assumed. `frams_dir` below maps directly to it.
    - `is_viable()`: now delegates to `isValidCreature()`, which checks buildability
      (parts/joints/neurons actually assemble into a valid Creature) WITHOUT running a
      full simulation and WITHOUT incrementing FramsticksLibCompetition's evaluation
      counter -- exactly the "before consuming simulation budget" property Section 5.5's
      Minimal-Viable Morphological Filter needs.
    - `get_random_genotype()`: now delegates to `getRandomGenotype()`, FramsticksLib's own
      method for generating a genotype within a target parts/neurons range via repeated
      mutation -- a real answer to the locomotion bootstrap problem (Section 3.1,
      sub-problem 6), not just "mutate the simplest genotype a few times".
    - `pairwise_dissimilarity()`: FramsticksLib.dissimilarity() natively computes BOTH our
      diversity measures -- DissimMethod.GENE_LEVENSHTEIN (plain Levenshtein) and
      DissimMethod.PHENE_STRUCT_GREEDY/OPTIM (real phenotypic graph dissimilarity via
      Framsticks' own SimilMeasureHungarian -- GREEDY is "fast but approximate", OPTIM is
      "slower but optimal", which maps directly onto the Variant 3 computational-safeguard
      tradeoff described in Section 5.1). We default to GREEDY for the main GA loop and
      note OPTIM as a one-off post-hoc option.
      NOTE: GENE_LEVENSHTEIN is RAW edit distance, not normalised to [0,1] -- fine for
      within-run relative comparisons (crowding, selection), but be careful comparing raw
      diversity numbers across runs/tasks with very different genotype lengths.
    - REPRODUCIBILITY (`deterministic` + `random_seed` parameters): CORRECTED after direct
      testing -- `FramsticksLib.DETERMINISTIC = True` alone is NOT sufficient. It only
      skips `frams.Math.randomize()`, but Framsticks' internal RNG does not fall back to a
      fixed state when that's skipped -- it still initializes from something
      non-deterministic (confirmed empirically: two fresh processes with DETERMINISTIC=True
      produced different genotypes from identical calls). `dir(frams.Math)` reveals the
      actual mechanism: a dedicated `frams.Math.seed(value)` method, separate from
      `randomize()`. We now call this explicitly, immediately after constructing
      `ProjectFramsticksLibCompetition` (i.e. after `frams.init()` and all `.sim` file
      loading have completed, but before any of our code performs a single mutation/
      crossover/evaluation) -- this is what actually pins the native RNG to a known state.
      `DETERMINISTIC=True` is still set too, defensively, in case anything else in the
      library calls `randomize()` later, but `frams.Math.seed()` is the confirmed fix.
      Combined with Python's own `seed_everything()` (utils/config.py) for
      selection/tournament randomness, this should now give full reproducibility: re-run
      the same script with the same `random_seed` -> identical results, while different
      repeats/generations WITHIN one run still differ naturally (the RNG stream keeps
      advancing, it's just pinned to the same STARTING point every process launch).
      To deliberately sample genuine run-to-run variance instead (e.g. to check how noisy
      an Energy0 comparison really is), pass a different `random_seed` per invocation, or
      set `deterministic=False` to fall back to non-reproducible behaviour entirely.

    STILL OPEN:
    - `world_config` (gravity / water level / terrain / orientation, Section 3.1) is
      IGNORED. Nothing in FramsticksLib.py sets these either -- they're presumably set via
      the `.sim` files passed to the constructor, or via the raw `frams` module directly
      (e.g. `frams.World.<property> = value`, mirroring how EVALUATION_SETTINGS_FILE .sim
      files configure things via `frams.Simulator.ximport(...)`). The likely correct
      design is one `RealFramsticksInterface` instance per world/.sim configuration,
      rather than a runtime dict. Confirm against the actual competition `.sim` files /
      `frams` module docs before running Phase 1 across multiple real environments.

    CRITICAL GOTCHA -- FramsticksLibCompetition's budget-tracking attributes
    (`_evaluation_count`, `_time0`, `_best_fitness`, etc.) are defined as CLASS attributes
    in the uploaded FramsticksLibCompetition.py, not instance attributes. That means if you
    reuse one `RealFramsticksInterface`/`ProjectFramsticksLibCompetition` across multiple
    algorithm runs (exactly what Phase 1/2/3 do -- many repeats through one interface),
    the evaluation count and elapsed-time clock ACCUMULATE ACROSS ALL OF THEM instead of
    resetting per run. Combined with `_evaluate_single_genotype()` calling `self.end()`
    (-> `sys.exit()`) once MAX_EVALUATIONS/MAX_TIME is exceeded, this means a long Phase 1
    batch could silently kill the ENTIRE remaining batch partway through, not just the run
    that happened to hit the limit. `__init__` below shadows these as instance attributes
    to fix the sharing problem, and `reset_evaluation_budget()` must be called before each
    new run when reusing one interface across repeats (see experiments/run_single.py).
    """

    # Bounds passed to FramsticksLib.getRandomGenotype() for initial-population generation.
    # Tune these per encoding/task if Phase 2 sensitivity analysis suggests better ranges.
    RANDOM_GENOTYPE_PARTS_RANGE = (3, 15)
    RANDOM_GENOTYPE_NEURONS_RANGE = (1, 10)
    RANDOM_GENOTYPE_ITER_MAX = 100

    # Bloat-prevention caps, checked in is_viable() via FramsticksLib.satisfiesConstraints()
    # -- no simulation run, no evaluation-budget cost. Nothing in our current fitness
    # functions penalises genotype size, so without a cap, mutation/insertion is free to
    # grow genotypes generation after generation with zero cost -- classic "bloat" in EC
    # terms. This was directly observed: genotypes reaching 30+ parts/joints by mid-run,
    # both slowing simulation (more physics per step) and causing Framsticks' own crossover
    # operator to fail far more often (100 retries against large, dissimilar structures is
    # expensive on its own, then falls back to returning a parent unchanged anyway).
    # These are DELIBERATELY generous defaults, not tuned against real data yet -- treat as
    # a starting point and tighten/loosen based on what Phase 1/2 actually needs. `None`
    # for any of these disables that particular constraint (matches
    # FramsticksLib.satisfiesConstraints()'s own convention).
    MAX_NUMPARTS: int | None = 30
    MAX_NUMJOINTS: int | None = 40
    MAX_NUMNEURONS: int | None = 20
    MAX_NUMCONNECTIONS: int | None = 60
    MAX_NUMGENOCHARS: int | None = 500

    def __init__(
        self,
        framspy_path: str,
        frams_dir: str,
        lib_name: str | None = None,
        sim_files: str = "eval-allcriteria.sim;deterministic.sim;recording-body-coords.sim",
        eval_budget: int = 100_000,
        time_budget_seconds: int = 3600,
        graph_dissimilarity_mode: str = "greedy",  # "greedy" (fast, approximate) or "optim" (slow, exact)
        deterministic: bool = True,  # see class docstring's REPRODUCIBILITY section
        random_seed: int = 42,  # explicit native-RNG seed; see REPRODUCIBILITY section
    ):
        import sys

        if framspy_path not in sys.path:
            sys.path.append(framspy_path)
        try:
            from src.framsticks_interface.competition_lib import ProjectFramsticksLibCompetition
        except ImportError as e:
            raise ImportError(
                "Could not import FramsticksLibCompetition from framspy_path="
                f"'{framspy_path}'. Confirm framspy was checked out there and that "
                "FramsticksLib.py / FramsticksLibCompetition.py are directly inside it "
                "(not in a subdirectory)."
            ) from e

        # MUST be set before ProjectFramsticksLibCompetition() is constructed -- the
        # constructor checks `if not self.DETERMINISTIC: frams.Math.randomize()` during
        # __init__ itself. Kept as a defensive belt-and-braces measure -- see class
        # docstring: this alone was empirically confirmed NOT sufficient for
        # reproducibility, which is why the explicit frams.Math.seed() call below exists.
        from FramsticksLib import FramsticksLib
        FramsticksLib.DETERMINISTIC = deterministic
        self._deterministic = deterministic
        self._random_seed = random_seed

        self._lib = ProjectFramsticksLibCompetition(frams_dir, lib_name, sim_files)

        # THE actual fix (confirmed via `dir(frams.Math)` -> 'seed' method exists,
        # separate from 'randomize'): pin Framsticks' native RNG to a known state, right
        # after construction/`.sim` loading complete but before any mutation/crossover/
        # evaluation call. Must import `frams` directly (not just FramsticksLib) since
        # that's the module-level object FramsticksLib.py itself calls frams.Math.* on.
        if deterministic:
            import frams
            frams.Math.seed = random_seed  # property assignment, NOT a method call --
            # confirmed via TypeError: 'ExtValue' object is not callable when first tried
            # as frams.Math.seed(random_seed). Framsticks' Python binding uses assignment
            # for properties (like frams.World.water_level = 0.5) and parentheses only for
            # actual methods (like frams.Math.randomize()).
            print(f"[RealFramsticksInterface] frams.Math.seed = {random_seed} applied -- "
                  f"native RNG pinned for reproducibility across process restarts.")
        else:
            print("[RealFramsticksInterface] deterministic=False -- native RNG left "
                  "randomized; results will differ across process restarts.")
        self._eval_budget = eval_budget
        self._time_budget_seconds = time_budget_seconds
        self._graph_dissimilarity_mode = graph_dissimilarity_mode
        self._warned_world_config = False
        self._ended = False
        self.reset_evaluation_budget()

    def reset_evaluation_budget(self) -> None:
        """Resets FramsticksLibCompetition's evaluation-count/timer/best-solution tracking
        as INSTANCE attributes (shadowing the class attributes -- see the class docstring's
        CRITICAL GOTCHA). Call this before every new algorithm run when reusing one
        `RealFramsticksInterface` across repeats (experiments/run_single.py does this
        automatically for the real backend)."""
        from time import perf_counter

        self._lib._evaluation_count = 0
        self._lib._evaluation_time = 0
        self._lib._time0 = perf_counter()
        self._lib._best_fitness = None
        self._lib._best_solution = None
        self._lib.MAX_EVALUATIONS = self._eval_budget
        self._lib.MAX_TIME = self._time_budget_seconds
        self._ended = False

    @property
    def n_evaluations(self) -> int:
        """Exposed so engine.py's
        `hasattr(interface, "n_evaluations")` bookkeeping works unchanged for either
        backend."""
        return self._lib._evaluation_count

    def _frams_format(self, encoding: str) -> str:
        if encoding not in _ENCODING_TO_FRAMS_FORMAT:
            raise ValueError(f"Unknown encoding '{encoding}', expected 'f0' or 'f1'")
        return _ENCODING_TO_FRAMS_FORMAT[encoding]

    def get_random_genotype(self, encoding: str) -> str:
        """Delegates to FramsticksLib.getRandomGenotype(): starts from the simplest
        genotype and mutates toward a randomly-chosen target parts/neurons count within
        RANDOM_GENOTYPE_PARTS_RANGE / RANDOM_GENOTYPE_NEURONS_RANGE. Doesn't consume
        evaluation budget (no simulation is run)."""
        fmt = self._frams_format(encoding)
        simplest = self._lib.getSimplest(fmt)
        p_min, p_max = self.RANDOM_GENOTYPE_PARTS_RANGE
        n_min, n_max = self.RANDOM_GENOTYPE_NEURONS_RANGE
        return self._lib.getRandomGenotype(
            simplest, p_min, p_max, n_min, n_max,
            self.RANDOM_GENOTYPE_ITER_MAX, return_even_if_failed=True,
        )

    def mutate(self, genotype: str, encoding: str) -> str:
        return self._lib.mutate([genotype])[0]

    def crossover(self, g1: str, g2: str, encoding: str) -> tuple[str, str]:
        # FramsticksLib.crossOver() returns ONE offspring per call (stochastic) -- call it
        # twice for two children, matching FramsticksEvolution.py's frams_crossover.
        child1 = self._lib.crossOver(g1, g2)
        child2 = self._lib.crossOver(g1, g2)
        return child1, child2

    def is_viable(self, genotype: str, encoding: str) -> bool:
        """isValidCreature() checks genetic validity AND buildability (e.g. rejects
        'X[@][@]'-style genotypes with impossible part overlaps) without running a full
        simulation, and without incrementing the evaluation counter -- exactly what the
        Minimal-Viable Morphological Filter (Section 5.5) needs.

        ALSO enforces the MAX_NUMPARTS/etc. bloat-prevention caps above via
        satisfiesConstraints() -- same no-simulation, no-budget-cost property. This is what
        keeps genotype size (and therefore per-evaluation wall-clock cost) bounded across
        generations; see the class attribute comments for why this exists."""
        if not self._lib.isValidCreature([genotype])[0]:
            return False
        return self._lib.satisfiesConstraints(
            genotype, self.MAX_NUMPARTS, self.MAX_NUMJOINTS, self.MAX_NUMNEURONS,
            self.MAX_NUMCONNECTIONS, self.MAX_NUMGENOCHARS,
        )

    def pairwise_dissimilarity(self, genotypes: list[str], encoding: str, measure: str) -> list[list[float]]:
        # Local import: DissimMethod lives in FramsticksLib.py, only importable once
        # framspy_path is on sys.path (done in __init__).
        from FramsticksLib import DissimMethod

        if measure == "gene":
            method = DissimMethod.GENE_LEVENSHTEIN
        elif measure == "graph":
            method = (DissimMethod.PHENE_STRUCT_OPTIM if self._graph_dissimilarity_mode == "optim"
                      else DissimMethod.PHENE_STRUCT_GREEDY)
        else:
            raise ValueError(f"Unknown dissimilarity measure '{measure}', expected 'gene' or 'graph'")

        matrix = self._lib.dissimilarity(genotypes, method)
        return matrix.tolist()

    def evaluate(self, genotype: str, encoding: str, task: str, world_config: dict | None = None) -> EvaluationResult:
        if world_config and not self._warned_world_config:
            print("[RealFramsticksInterface] WARNING: world_config is currently ignored "
                  "-- see interface.py docstring 'STILL OPEN' (environment is set via .sim "
                  "files / the frams module directly, not per-evaluate()).")
            self._warned_world_config = True

        self._lib.current_task = task
        result = self._lib.evaluate([genotype])  # SIMPLE_FITNESS_FORMAT=False -> full dict per genotype
        r = result[0]
        evaluations = r.get("evaluations")

        if evaluations is None:
            return EvaluationResult(fitness=0.0, cog_trajectory=[], n_parts=0, is_valid=False, raw=r)

        data = evaluations[""]
        fitness = data.get("COGpath", 0.0) or 0.0
        n_parts = data.get("numparts", 0)
        raw_path = data.get("data->bodyrecording", [])
        if raw_path:
            from src.framsticks_interface.task_fitness import extract_cog_trajectory
            trajectory = [tuple(row) for row in extract_cog_trajectory(raw_path)]
        else:
            trajectory = []

        return EvaluationResult(
            fitness=float(fitness),
            cog_trajectory=trajectory,
            n_parts=int(n_parts) if n_parts else 0,
            is_valid=True,
            raw=data,
        )

    def finalize_submission(self) -> None:
        """Calls FramsticksLibCompetition.end() (logs the best solution, writes the
        *.results file, then sys.exit()s the process). ONLY call this from a dedicated
        final-submission script, never from Phase 1/2/3 experiment loops."""
        if not self._ended:
            self._ended = True
            self._lib.end()  # this calls sys.exit() -- process ends here


def get_interface(config: dict) -> FramsticksInterface:
    """Factory: always constructs a `RealFramsticksInterface` from config['framsticks'].
    There is no mock backend in production code -- `tests/fakes.py` has a
    lightweight test-only double used solely by the test suite, so CI can still exercise
    engine.py's algorithm logic without the real simulator installed."""
    fconf = config["framsticks"]
    required = ["framspy_path", "frams_dir"]
    missing = [k for k in required if not fconf.get(k)]
    if missing:
        raise ValueError(
            f"configs/default_config.yaml is missing framsticks.{missing[0]} (and possibly "
            f"others: {missing}). Each team member must set these to their own local "
            "framspy/Framsticks install -- see docs/framsticks_integration_notes.md. "
            "Don't commit real local paths to configs/default_config.yaml; use "
            "configs/local_config.yaml (gitignored) instead and pass --config configs/local_config.yaml."
        )
    return RealFramsticksInterface(
        framspy_path=fconf["framspy_path"],
        frams_dir=fconf["frams_dir"],
        lib_name=fconf.get("lib_name"),
        sim_files=fconf.get("sim_files", "eval-allcriteria.sim;deterministic.sim;recording-body-coords.sim"),
        eval_budget=fconf.get("eval_budget", 100_000),
        time_budget_seconds=fconf.get("time_budget_seconds", 3600),
        graph_dissimilarity_mode=fconf.get("graph_dissimilarity_mode", "greedy"),
        deterministic=fconf.get("deterministic", True),
        random_seed=fconf.get("random_seed", config.get("experiment", {}).get("random_seed_base", 42)),
    )

# Integrating the real Framsticks / `framspy` backend

Status (updated Oct 2026): **world_config is now implemented** -- see src/framsticks_interface/worlds.py,
configs/worlds/phase3_worlds.yaml and docs/AUDIT_2026-10.md. Original status: **implemented and internally consistent**, against the confirmed API of
`FramsticksLib.py`, `FramsticksLibCompetition.py`, and `FramsticksEvolution.py` (all three
uploaded by Salmaan). `RealFramsticksInterface` is now the only production backend -- there
is no mock/fake shipped as a feature (see "No more mock backend" below). One thing remains
genuinely open (world_config / per-setting environments) -- see "STILL OPEN" below --
everything else has a working implementation.

## What's implemented

| File | What it does |
|---|---|
| `task_fitness.py` | Our four Phase 1 proxy-task fitness functions (Section 2.2/5.3), pure numpy, no framspy dependency -- unit-tested in `tests/test_competition_lib.py` on every machine/CI regardless of whether the simulator is installed. |
| `competition_lib.py` | `ProjectFramsticksLibCompetition(FramsticksLibCompetition)` overrides `_evaluate_path` to call `task_fitness.py`, keeping FramsticksLibCompetition's evaluation-budget/logging machinery untouched. |
| `interface.py::RealFramsticksInterface` | Full implementation: `get_random_genotype` (via `FramsticksLib.getRandomGenotype`), `mutate`/`crossover` (direct passthrough), `is_viable` (via `isValidCreature`, no simulation run, no budget cost), `pairwise_dissimilarity` (via `FramsticksLib.dissimilarity()` -- GENE_LEVENSHTEIN / PHENE_STRUCT_GREEDY), `evaluate`, `finalize_submission`. |
| `engine.py`, `metrics.py` | Call `interface.pairwise_dissimilarity()` rather than importing `src/diversity/*` directly, so a real run actually uses Framsticks' own dissimilarity measures, not a Python approximation of them. |

## No more mock backend

`src/framsticks_interface/mock_framsticks.py` has been removed. `RealFramsticksInterface`
is now the only implementation of `FramsticksInterface` in `src/`. `get_interface()` in
`interface.py` no longer branches on a `use_mock` flag -- it always constructs
`RealFramsticksInterface` from `config["framsticks"]`, and raises a clear `ValueError` if
`framspy_path`/`frams_dir` aren't set.

For testing, `tests/fakes.py::FakeFramsticksInterface` is a small test-only double, used
exclusively by `pytest` (not imported anywhere in `src/`) so `engine.py`'s NSGA-II logic
stays covered by CI without needing the real simulator installed there. If you add a new
test that needs a `FramsticksInterface`, import `FakeFramsticksInterface` from
`tests.fakes` -- never reach for the real one in a unit test, and never add mock-like
behaviour back into `src/`.

Practical implication: `scripts/run_phase1.py` (and phase2/phase3) now **require** a real
Framsticks + framspy install configured in `configs/default_config.yaml` (or a
`--config configs/local_config.yaml` override) -- there's no fallback anymore. `pytest
tests/ -q` still works with zero setup, since it never touches `RealFramsticksInterface`.

## Confirmed against FramsticksLib.py (no longer assumptions)

- **Constructor**: `FramsticksLib(frams_path, frams_lib_name, sim_settings_files)` --
  `frams_path` is the directory containing the frams-objects library, no trailing slash
  (matches `-path`'s `type=ensureDir` argparse validator). `frams_dir` in our config maps
  directly to it.
- **`mutate(genotype_list: List[str]) -> List[str]`** and
  **`crossOver(genotype_parent1: str, genotype_parent2: str) -> str`** (singular offspring
  per call) -- exactly what we'd assumed; no code changes needed.
- **`getSimplest(genetic_format: str) -> str`** -- format codes are the encoding letter
  *without* the `f` prefix (`"0"` for f0, `"1"` for f1), confirmed by the constructor's own
  self-test (`getSimplest("1") == "X"`).
- **`isValid(genotype_list) -> List[bool]`** -- genetic-string validity only (no
  build/simulation). **`isValidCreature(genotype_list) -> List[bool]`** -- also checks
  buildability (catches things like `X[@][@]`'s duplicate-muscle problem) without running
  a full simulation and without touching the evaluation counter. `is_viable()` now uses
  `isValidCreature()`, which is the correct match for Section 5.5's Minimal-Viable
  Morphological Filter.
- **`getRandomGenotype(initial_genotype, parts_min, parts_max, neurons_min, neurons_max, iter_max, return_even_if_failed)`**
  -- FramsticksLib's own answer to "give me a random genotype," via targeted mutation
  toward a randomly-chosen parts/neurons count. `get_random_genotype()` now uses this
  directly instead of our earlier ad hoc "mutate the simplest a few times" guess.
- **`dissimilarity(genotype_list, method: DissimMethod) -> np.ndarray`** -- Framsticks
  natively implements BOTH the gene and graph dissimilarity measures the proposal calls
  for: `DissimMethod.GENE_LEVENSHTEIN` and `DissimMethod.PHENE_STRUCT_GREEDY` /
  `PHENE_STRUCT_OPTIM` (fast-approximate vs. slower-exact graph matching -- exactly the
  Variant 3 computational-safeguard tradeoff described in Section 5.1).
  `RealFramsticksInterface.pairwise_dissimilarity()` delegates to this rather than
  reimplementing dissimilarity in Python. **Caveat**: the real `GENE_LEVENSHTEIN` is *raw*
  edit distance, not normalised to `[0,1]` -- fine for relative comparisons within one run
  (crowding, selection), but be careful comparing raw diversity numbers across runs/tasks
  with very different genotype lengths.

## Fixed along the way (not directly about Framsticks, but found while wiring this up)

1. **`engine.py` was missing its `src.diversity.*` imports entirely** -- a pre-existing bug
   that made every non-baseline variant crash with `NameError` the moment it tried to
   compute a diversity score. Caught and fixed while routing diversity calculations through
   the interface.
2. **Evaluation-count "leakage" across runs.** `FramsticksLibCompetition`'s budget-tracking
   attributes (`_evaluation_count`, `_time0`, etc., which are *class* attributes on the
   uploaded `FramsticksLibCompetition.py`, not instance attributes) would otherwise
   accumulate across every repeat in a Phase 1/2/3 batch, since those phases reuse one
   interface instance for the whole batch. Two consequences: (a) `RunRecord.n_evaluations`
   would report a cumulative total instead of one run's count, and (b)
   `EngineConfig.eval_budget` would trip after only a handful of runs cumulatively hit
   100k, silently truncating every run after that to a handful of generations. Fixed via
   `reset_evaluation_budget()`, called automatically by `run_single.py` before every run.
3. **`data->bodyrecording` shape mismatch when `STORE_ALL_PART_COORDS=1`.** If
   `recording-body-coords.sim` has `STORE_ALL_PART_COORDS=1` (verbose: COG + every part's
   coordinates per step) rather than `0` (COG only, as `FramsticksLibCompetition.py`'s own
   docstring recommends), `data->bodyrecording` comes back as a 3D array
   `(n_steps, n_parts+1, 3)` instead of the 2D `(n_steps, 3)` our fitness functions
   assumed, crashing with a numpy broadcast error. Fixed with
   `task_fitness.py::extract_cog_trajectory()`, which normalises either format to plain
   COG coordinates (COG is always index 0 along the parts axis in verbose mode) --
   defensive either way, but **still worth setting `STORE_ALL_PART_COORDS=0` in
   `recording-body-coords.sim` directly**, since verbose mode records far more data than
   needed and will slow down every evaluation.

## STILL OPEN

- **`world_config`** (gravity / water level / terrain / orientation, Section 3.1) is
  ignored by `RealFramsticksInterface.evaluate()` -- it prints a one-time warning and
  proceeds. Nothing in `FramsticksLib.py` sets these as a per-evaluate() parameter; they're
  presumably set via the `.sim` files passed to the constructor, or via the raw `frams`
  module directly (e.g. `frams.World.<property> = value`, mirroring how
  `EVALUATION_SETTINGS_FILE` `.sim` files configure things via `frams.Simulator.ximport(...)`).
  The likely correct design is **one `RealFramsticksInterface` instance per world/`.sim`
  configuration** (i.e. per competition "setting"), rather than a runtime dict. Needs the
  actual competition `.sim` files (or the `frams` module reference docs) to close -- not
  blocked on any more Python source files.

## The `end()` / `sys.exit()` trap

`ProjectFramsticksLibCompetition.end()` (inherited, unmodified) calls `sys.exit()` after
logging the best solution. `RealFramsticksInterface.finalize_submission()` wraps it, but:

- **Never** call `finalize_submission()` from `phase1_controlled.py`, `phase2_sensitivity.py`,
  or `phase3_evaluation.py` -- they each run many combinations in one process.
- **Do** call it from a dedicated, separate `scripts/submit_competition.py` (not yet
  written) that constructs one `RealFramsticksInterface`, runs exactly one algorithm to
  completion, and calls `finalize_submission()` as the very last line.

## Config

```yaml
framsticks:
  framspy_path: "C:/SalmaanP/.../framspy"      # dir containing FramsticksLib.py directly
  frams_dir: "C:/SalmaanP/.../Framsticks55"     # the "-path" argument, no trailing slash
  lib_name: null                                 # leave null for the platform default
  sim_files: "eval-allcriteria.sim;deterministic.sim;recording-body-coords.sim"
  graph_dissimilarity_mode: "greedy"              # "greedy" (fast/approx) or "optim" (slow/exact)
```

Also required, per `FramsticksLibCompetition.py`'s own docstring, before this produces a
valid body-recording path:
- Set `STORE_ALL_PART_COORDS = 0` in `recording-body-coords.sim`.
- Make sure `recording-body-coords.sim` is included in `sim_files` (it is, by default) --
  without it, `data->bodyrecording` won't be populated and `evaluate()` returns an empty
  trajectory.

## Suggested rollout order

1. Get `frams-test.py` and `run-deap-examples.cmd` (or its Linux equivalent) working first
   -- confirms simulator + framspy + DEAP talk to each other before our code enters the
   picture.
2. Fill in `framspy_path`/`frams_dir` in your local config, and in a throwaway script call
   `interface.evaluate()` on one genotype for one task and print the full
   `EvaluationResult` -- confirms `data->bodyrecording`/`numparts` really are populated
   (needs `recording-body-coords.sim` loaded, see above) before trusting a full
   evolutionary run's numbers.
3. Re-run `pytest tests/ -q` (should still pass -- it uses `tests/fakes.py`, not the real
   simulator) then `scripts/run_phase1.py --smoke` with your real paths configured to
   confirm the whole pipeline runs end-to-end before committing to a full Phase 1
   (5 variants x 2 encodings x 4 tasks x 20 repeats). Consider a much smaller
   `population_size`/`n_generations`/`eval_budget` for this first real run -- each
   evaluation against the real simulator is far slower than against the old mock.
4. Resolve `world_config` (see STILL OPEN) before trying to run Phase 1 across more than
   one real environment setting.

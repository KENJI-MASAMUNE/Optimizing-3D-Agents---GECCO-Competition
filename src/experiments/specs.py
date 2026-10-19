"""
Run specifications ("job manifests") for Phases 1-3.

Every experiment is expanded into a flat, deterministic list of `RunSpec`s. A spec fully
describes one evolutionary run (variant, encoding, task, world, repeat, seed and any
hyper-parameter overrides) and has a unique, filesystem-safe `run_id` that determines its
output file. This gives three properties the CHPC workflow relies on:

* Stable indexing -- spec #i is the same run on every machine, so a PBS job array (or a
  process pool) can just be told "run indices a..b".
* No collisions -- sweep parameters are part of run_id, so Phase 2 combinations can no
  longer overwrite each other (the bug that left only 15 files in results/phase2/).
* Resumability -- a run whose output JSON already exists is skipped, so a job that hits
  its walltime can simply be resubmitted.

Seeds: seed = phase seed base + repeat. All variants/encodings/tasks share the same seed for
the same repeat (common random numbers): repeat r of every variant starts from the same
RNG state, which makes the variant comparison paired and reduces variance.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from src.framsticks_interface.worlds import load_worlds

PHASE_SEED_OFFSET = {1: 0, 2: 0, 3: 1000}


@dataclass(frozen=True)
class RunSpec:
    phase: int
    variant: str
    encoding: str
    task: str
    repeat: int
    seed: int
    group: str = ""                       # sub-directory (e.g. Phase 2 sweep name / Phase 3 setting)
    world: str | None = None              # registered world name (None = base training world)
    setting: str | None = None            # Phase 3 setting name
    overrides: tuple = ()                 # sorted (key, value) EngineConfig overrides
    sweep: tuple = ()                     # sorted (key, value) labels describing the sweep point
    held_out_worlds: tuple = ()           # Phase 3: worlds for the generalisation score

    @property
    def overrides_dict(self) -> dict[str, Any]:
        return dict(self.overrides)

    @property
    def sweep_dict(self) -> dict[str, Any]:
        return dict(self.sweep)

    @property
    def run_id(self) -> str:
        parts = [self.variant, self.encoding, self.task]
        if self.setting:
            parts.insert(0, self.setting)
        for k, v in self.sweep:
            parts.append(f"{k}-{_fmt(v)}")
        parts.append(f"r{self.repeat:02d}_s{self.seed}")
        return _safe("__".join(parts))

    def output_path(self, results_dir: str | Path) -> Path:
        base = Path(results_dir) / f"phase{self.phase}"
        if self.group:
            base = base / _safe(self.group)
        return base / f"{self.run_id}.json"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["overrides"] = self.overrides_dict
        d["sweep"] = self.sweep_dict
        d["held_out_worlds"] = list(self.held_out_worlds)
        d["run_id"] = self.run_id
        return d


def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.\-]", "_", s)


def _ov(**kw) -> tuple:
    return tuple(sorted((k, v) for k, v in kw.items() if v is not None))


# ---------------------------------------------------------------------------- Phase 1
def build_phase1_specs(config: dict, smoke: bool = False) -> list[RunSpec]:
    """5 variants x 2 encodings x 4 proxy tasks x n_repeats (proposal Section 5.3)."""
    exp = config["experiment"]
    variants = config["variants"]
    encodings = config["ga"]["encodings"]
    tasks = [t["name"] for t in config["tasks"]]
    n_repeats = exp["n_repeats"]
    if smoke:
        variants, tasks, n_repeats = variants, tasks[:1], exp.get("n_repeats_smoke", 2)
    seed_base = exp["random_seed_base"] + PHASE_SEED_OFFSET[1]
    specs = []
    for variant in variants:
        for encoding in encodings:
            for task in tasks:
                for r in range(n_repeats):
                    specs.append(RunSpec(phase=1, variant=variant, encoding=encoding, task=task,
                                         repeat=r, seed=seed_base + r))
    return specs


# ---------------------------------------------------------------------------- Phase 2
def build_phase2_specs(config: dict, smoke: bool = False) -> list[RunSpec]:
    """One-factor-at-a-time sensitivity sweeps (proposal Section 5.3):
       A) population size x tournament size, at a FIXED evaluation budget
          (n_generations = budget_evals / population_size), so a bigger population is
          not silently given more evaluations;
       B) adaptive switching point tau (and optionally steepness k) for Variant 5;
       C) Variant 4 gene/graph weight, per encoding.
    Everything not being swept stays at the Phase 1 defaults."""
    p2 = config.get("phase2", {})
    exp = config["experiment"]
    ga = config["ga"]
    tasks = p2.get("tasks") or [config["tasks"][0]["name"]]
    n_repeats = p2.get("n_repeats", 10)
    budget_evals = p2.get("budget_evals", ga["population_size"] * ga["n_generations"])
    pop_sizes = p2.get("population_sizes", [50, 100, 150, 200])
    tourn_sizes = p2.get("tournament_sizes", [2, 3, 5])
    taus = p2.get("adaptive_taus", [0.25, 0.5, 0.75])
    steepnesses = p2.get("adaptive_steepness", [config["adaptive"]["steepness"]])
    weights = p2.get("hybrid_gene_weights", [0.0, 0.25, 0.5, 0.75, 1.0])
    pop_variant = p2.get("population_tournament_variant", "de_nsga2_gene")
    enc_ab = p2.get("encoding", ga["encodings"][0])
    enc_c = p2.get("hybrid_encodings", ga["encodings"])

    if smoke:
        tasks, n_repeats = tasks[:1], 1
        pop_sizes, tourn_sizes, taus, steepnesses, weights = pop_sizes[:1], tourn_sizes[:1], taus[:1], steepnesses[:1], weights[:2]
        budget_evals = min(budget_evals, 40)

    seed_base = exp["random_seed_base"] + PHASE_SEED_OFFSET[2]
    specs: list[RunSpec] = []
    for task in tasks:
        for pop in pop_sizes:
            gens = max(2, budget_evals // pop)
            for tourn in tourn_sizes:
                for r in range(n_repeats):
                    specs.append(RunSpec(
                        phase=2, variant=pop_variant, encoding=enc_ab, task=task, repeat=r,
                        seed=seed_base + r, group="A_population_tournament",
                        overrides=_ov(population_size=pop, n_generations=gens, tournament_size=tourn),
                        sweep=(("pop", pop), ("tourn", tourn))))
        for tau in taus:
            for k in steepnesses:
                for r in range(n_repeats):
                    specs.append(RunSpec(
                        phase=2, variant="adaptive_de_nsga2", encoding=enc_ab, task=task, repeat=r,
                        seed=seed_base + r, group="B_adaptive_schedule",
                        overrides=_ov(adaptive_tau=float(tau), adaptive_steepness=float(k)),
                        sweep=(("tau", float(tau)), ("k", float(k)))))
        for enc in enc_c:
            for w in weights:
                for r in range(n_repeats):
                    specs.append(RunSpec(
                        phase=2, variant="de_nsga2_hybrid", encoding=enc, task=task, repeat=r,
                        seed=seed_base + r, group="C_hybrid_weight",
                        overrides=_ov(hybrid_weight=float(w)),
                        sweep=(("w", float(w)),)))
    return specs


# ---------------------------------------------------------------------------- Phase 3
BASELINE_VARIANTS = {"simple_ea"}


def select_best_per_task(phase1_results_dir: str | Path) -> dict[str, dict]:
    """For each task: the (variant, encoding) with the highest mean best fitness in Phase 1."""
    by_task: dict[str, dict[tuple[str, str], list[float]]] = {}
    for path in Path(phase1_results_dir).rglob("*.json"):
        with open(path) as f:
            r = json.load(f)
        if "variant" not in r or "best_fitness" not in r or "run_id" not in r:
            continue  # pre-audit records are ignored
        if r["variant"] in BASELINE_VARIANTS:
            continue  # the organisers' Simple EA is a reference point, never "the winner"
        by_task.setdefault(r["task"], {}).setdefault((r["variant"], r["encoding"]), []).append(r["best_fitness"])
    best = {}
    for task, scores in by_task.items():
        key = max(scores, key=lambda k: sum(scores[k]) / len(scores[k]))
        best[task] = {"variant": key[0], "encoding": key[1],
                      "mean_best_fitness": sum(scores[key]) / len(scores[key]),
                      "n_runs": len(scores[key])}
    return best


def select_tuned_parameters(phase2_results_dir: str | Path) -> dict[str, Any]:
    """Phase 2 -> Phase 3 hand-over: the variant-specific knobs with the best Phase 2 result.

    * adaptive_de_nsga2: (tau, k) from sweep B
    * de_nsga2_hybrid:   gene weight w per encoding from sweep C ("tuned per encoding")
    Scores are task-normalised (each run's best fitness divided by the best mean in its
    task/encoding) before averaging, so tasks with larger fitness scales don't dominate.
    Population/tournament size (sweep A) are reported as a robustness analysis only; they
    are not changed for Phase 3 because they apply to every variant equally."""
    rows = []
    for path in Path(phase2_results_dir).rglob("*.json"):
        with open(path) as f:
            r = json.load(f)
        if r.get("group") in ("B_adaptive_schedule", "C_hybrid_weight") and "best_fitness" in r:
            rows.append(r)
    if not rows:
        return {}

    def best_level(group_rows, key_fn):
        by_task: dict[tuple, dict[tuple, list[float]]] = {}
        for r in group_rows:
            by_task.setdefault((r["task"], r["encoding"]), {}).setdefault(key_fn(r), []).append(r["best_fitness"])
        score: dict[tuple, list[float]] = {}
        for levels in by_task.values():
            means = {lv: sum(v) / len(v) for lv, v in levels.items()}
            top = max(means.values())
            for lv, m in means.items():
                score.setdefault(lv, []).append(m / top if top > 0 else 0.0)
        return max(score, key=lambda lv: sum(score[lv]) / len(score[lv]))

    tuned: dict[str, Any] = {}
    b = [r for r in rows if r["group"] == "B_adaptive_schedule"]
    if b:
        tau, k = best_level(b, lambda r: (r["sweep"]["tau"], r["sweep"]["k"]))
        tuned["adaptive_de_nsga2"] = {"adaptive_tau": float(tau), "adaptive_steepness": float(k)}
    c = [r for r in rows if r["group"] == "C_hybrid_weight"]
    if c:
        tuned["de_nsga2_hybrid"] = {}
        for enc in sorted({r["encoding"] for r in c}):
            w = best_level([r for r in c if r["encoding"] == enc], lambda r: (r["sweep"]["w"],))[0]
            tuned["de_nsga2_hybrid"][enc] = {"hybrid_weight": float(w)}
    return tuned


def _tuned_overrides(tuned: dict | None, variant: str, encoding: str) -> tuple:
    if not tuned or variant not in tuned:
        return ()
    t = tuned[variant]
    if variant == "de_nsga2_hybrid":
        t = t.get(encoding, {})
    return _ov(**t)


def build_phase3_specs(config: dict, selection: dict[str, dict] | None = None,
                       smoke: bool = False, tuned: dict | None = None) -> list[RunSpec]:
    """Competition-style evaluation on the 10 reconstructed settings
    (configs/worlds/phase3_worlds.yaml), each run followed by re-evaluation of its best
    agent in the held-out worlds (Generalisation Score).

    If Phase 2 results exist, `phase3-select` also freezes the tuned variant-specific knobs
    (adaptive tau/k, hybrid weight per encoding) and they are applied here
    (phase3.use_phase2_tuning, default true): Phase 1 = untuned "out-of-the-box"
    comparison, Phase 2 = sensitivity + tuning, Phase 3 = tuned variants on unseen worlds.

    phase3.mode:
      best_per_task (default, = proposal): the Phase 1 winner (variant, encoding) for the
                     setting's task, plus the NSGA-II baseline with the same encoding as a
                     control (so "is the winner still better in new worlds?" is testable).
      all_variants:  every variant in phase3.variants (default: the top-level `variants`
                     list) x all encodings (full comparison + competition-normalised score).
    include_best_per_task: with all_variants, also run each task's Phase 1 winner when it
    is not already in that list. Same output names, so a winner that is already scheduled
    is not run twice.
    In best_per_task mode the organisers' Simple EA (same encoding) is added as a second
    control when it is listed in `variants`, so every setting has a Simple EA = 0 reference.
    """
    p3 = config.get("phase3", {})
    exp = config["experiment"]
    worlds, settings, held_out = load_worlds(p3.get("worlds_file"))
    n_repeats = p3.get("n_repeats", exp["n_repeats"])
    mode = p3.get("mode", "best_per_task")
    if smoke:
        settings, n_repeats = settings[:2], 1

    seed_base = exp["random_seed_base"] + PHASE_SEED_OFFSET[3]
    want_winners = mode != "all_variants" or p3.get("include_best_per_task", False)
    specs: list[RunSpec] = []
    for s in settings:
        combos: list[tuple[str, str]] = []
        if mode == "all_variants":
            combos.extend((v, e) for v in (p3.get("variants") or config["variants"]) for e in config["ga"]["encodings"])
        if want_winners:
            if smoke and (not selection or s.task not in selection):
                selection = dict(selection or {})
                selection[s.task] = {"variant": "de_nsga2_gene", "encoding": "f1"}  # smoke placeholder
            if not selection or s.task not in selection:
                raise ValueError(
                    f"Phase 3 (best_per_task) needs a Phase 1 selection for task '{s.task}'. "
                    "Run `python -m src.experiments.cli phase3-select` after Phase 1, or set "
                    "phase3.include_best_per_task: false.")
            chosen = selection[s.task]
            combos.append((chosen["variant"], chosen["encoding"]))
            if mode != "all_variants":
                if chosen["variant"] != "nsga2_baseline":
                    combos.append(("nsga2_baseline", chosen["encoding"]))
                if "simple_ea" in config["variants"]:
                    combos.append(("simple_ea", chosen["encoding"]))
        seen: set[tuple[str, str]] = set()
        for variant, encoding in combos:
            if (variant, encoding) in seen:
                continue
            seen.add((variant, encoding))
            ov = _tuned_overrides(tuned, variant, encoding) if p3.get("use_phase2_tuning", True) else ()
            for r in range(n_repeats):
                specs.append(RunSpec(phase=3, variant=variant, encoding=encoding, task=s.task,
                                     repeat=r, seed=seed_base + r, group=s.name, world=s.world,
                                     setting=s.name, held_out_worlds=tuple(held_out), overrides=ov))
    return specs


def build_specs(phase: int, config: dict, smoke: bool = False, selection: dict | None = None,
                tuned: dict | None = None) -> list[RunSpec]:
    if phase == 1:
        return build_phase1_specs(config, smoke)
    if phase == 2:
        return build_phase2_specs(config, smoke)
    if phase == 3:
        return build_phase3_specs(config, selection, smoke, tuned)
    raise ValueError(f"Unknown phase {phase}")

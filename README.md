# DE-NSGA-II: Diversity-Enhanced Evolutionary Optimisation of 3D Agents in Framsticks

**COS700 Honours Research Project, University of Pretoria (2026)**

Group: Salmaan Patel, Vanessa Rampedi, Ntsako Sithole
Supervisors: Prof. N. Pillay, Dr. T. Nyathi

This repository implements and evaluates five evolutionary algorithms for evolving Framsticks
agents whose centre of gravity (COG) must move in a required way, using the GECCO Automated
Design Competition protocol as a research benchmark: standard NSGA-II and four
Diversity-Enhancing NSGA-II (DE-NSGA-II) variants (Komosinski & Mensfelt, GECCO 2025) —
Gene, Graph and Hybrid crowding, and the novel **Adaptive DE-NSGA-II** — plus a
re-implementation of the organisers' **Simple EA** baseline.

**Public showcase (summary, figures, poster, animation of the evolved agents):**
https://github.com/KENJI-MASAMUNE/Optimizing-3D-Agents-Showcase

## Results at a glance

All three phases were run on the CHPC Lengau cluster (3,780 runs, ≈1,860 CPU-hours):

| Phase | Runs | Outcome |
|---|---|---|
| 1. Controlled comparison (6 algorithms × f0/f1 × 6 tasks × 20 seeds) | 1440 | No DE variant significantly fitter than NSGA-II on any task; Adaptive significantly better than DE gene across tasks (Wilcoxon p = 0.034) |
| 2. Sensitivity (population, tournament, τ, hybrid weight) | 500 | No significant effects (all p ≥ 0.145) |
| 3. 14 competition-style worlds + 4 held-out worlds | 1840 | Adaptive f1 best general algorithm (best mean in 5/14 settings), not significantly above NSGA-II; Simple EA f0 best in 4/14; agents keep only 32–40% of fitness in unseen worlds |

The full analysis is in the research report (`COS700_Research_Report.docx`, kept outside the
repository) and summarised in the showcase repository.

## Framsticks backend

`RealFramsticksInterface` (`src/framsticks_interface/interface.py`) wraps the
competition library (`FramsticksLibCompetition`, via `framspy`). Every algorithm/experiment module
talks only to the abstract `FramsticksInterface`, never to framspy directly, so if the
adapter ever needs to change, nothing else in `src/` does.

The real simulator (a licensed binary) can't run in CI or be installed via pip, so
`tests/fakes.py` has a small test-only double implementing the same interface, used
exclusively by `pytest` to exercise `engine.py`'s actual NSGA-II logic — non-dominated
sorting, crowding distance, the adaptive sigmoid schedule, budget bookkeeping — without
needing the real simulator installed anywhere. It is not part of `src/`, is never imported
by production code, and nothing it produces is a research result.

## Repository structure and ownership

This split mirrors the Work Plan (Section 7) in the proposal. Each member owns a package and
is the primary reviewer for pull requests touching it — but this is a group project, so
cross-review and pairing are expected, especially at the module boundaries.

```
src/
├── framsticks_interface/   # OWNER: Vanessa Rampedi
│   ├── interface.py           Abstract FramsticksInterface + RealFramsticksInterface adapter
│   ├── competition_lib.py     Our FramsticksLibCompetition subclass (task-specific fitness)
│   ├── task_fitness.py        The six task fitness functions, pure numpy (framspy-free)
│   └── encodings.py           f0 / f1 genotype helpers, random genotype generators
│
├── diversity/               # OWNER: Vanessa Rampedi (with Ntsako on graph measure)
│   ├── gene_dissimilarity.py  Levenshtein distance on genotype strings
│   └── graph_dissimilarity.py Bounded structural graph dissimilarity (f0/f1 -> graph)
│
├── algorithms/              # OWNER: Salmaan Patel
│   ├── crowding.py             Objective-space crowding (NSGA-II) + dissimilarity crowding
│   ├── nsga2_baseline.py       Variant 1: standard NSGA-II (DEAP selNSGA2)
│   ├── de_nsga2.py             Variants 2-4: Gene / Graph / Hybrid crowding
│   └── adaptive_de_nsga2.py    Variant 5: sigmoidal-schedule Adaptive DE-NSGA-II (novel)
│
├── evaluation/               # OWNER: Ntsako Sithole
│   └── metrics.py              Mean best fitness, gene/graph diversity, convergence rate,
│                                generalisation score, normalised competition score
│
├── experiments/              # OWNER: Ntsako Sithole
│   ├── run_single.py           Run one (variant, encoding, task) configuration
│   ├── phase1_controlled.py    Phase 1: 5 variants x 2 encodings x N tasks x 20 repeats
│   ├── phase2_sensitivity.py   Phase 2: population size / tournament / tau / hybrid weight
│   ├── phase3_evaluation.py    Phase 3: clean-environment held-out evaluation
│   └── statistics.py           Kruskal-Wallis + Dunn's post-hoc (alpha = 0.05)
│
└── utils/                    # Shared, all members
    └── config.py               YAML config loader, RNG seeding, run bookkeeping
```

## Getting started

```bash
git clone <this-repo-url>
cd de-nsga2-framsticks
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/ -q     # sanity-check the algorithm/diversity/crowding logic (no Framsticks needed)
```

`pytest` works immediately for everyone — it runs against a small test-only double (see
"Framsticks backend" above), not the real simulator. Actually running the evolutionary
pipeline (`scripts/run_phase1.py` etc.) requires the real Framsticks + `framspy` set up
locally first (next section).

### Local Framsticks/framspy setup (each team member does this individually)

`framspy/` and the Framsticks simulator distribution (e.g. `Framsticks55/`) are **not**
committed to this repo — they're large, platform-specific (`.dll`/`.so`/`.dylib`), and
each team member needs their own local copy anyway. Both are already covered by
`.gitignore`. Follow `docs/framsticks_integration_notes.md` to set these up, then point
`configs/default_config.yaml` at your own local paths:

```yaml
framsticks:
  framspy_path: "/path/to/framspy"        # dir containing FramsticksLib.py directly
  frams_dir: "/path/to/Framsticks55"       # dir containing frams-objects.dll/.so/.dylib
  lib_name: null                            # leave null unless you renamed the library file
  sim_files: "eval-allcriteria.sim;deterministic.sim;recording-body-coords.sim"
```

Since these paths are personal to your machine, **don't commit your edited
`configs/default_config.yaml`** with real paths filled in — either keep the paths `null`
in what you commit and override locally, or add a `configs/local_config.yaml` (gitignored)
and pass it via `--config configs/local_config.yaml` to the scripts below.

## Running the full experimental pipeline

See **docs/PHASES_AND_CHPC.md** for what each phase is for, the CHPC (Lengau) workflow, and
measured run times. See **docs/AUDIT_2026-10.md** for the October 2026 audit (bugs fixed,
determinism validation).

```bash
python scripts/check_framsticks.py --timing          # environment check on a new machine
python scripts/verify_reproducibility.py             # results must depend only on the seed
python -m src.experiments.cli run --phase 1 --workers 8      # 1440 runs (resumable)
python -m src.experiments.cli run --phase 2 --workers 8      # 500 runs (can run alongside Phase 1)
python -m src.experiments.cli phase3-select                  # freeze the Phase 1 winners
python -m src.experiments.cli run --phase 3 --workers 8      # 1840 runs: 14 settings + 4 held-out worlds
python -m src.experiments.cli aggregate --phase 1            # CSVs, Kruskal-Wallis / Dunn, vs Simple EA
python -m src.experiments.cli best-diversity --phase 1       # diversity of the 20 best solutions
python scripts/export_best_genotypes.py results/analysis/phase1/runs.csv best_phase1.gen   # for the Framsticks GUI
python scripts/render_best_agents.py results/analysis/phase1/runs.csv best_agents.gif      # animation of the best agents
```

On the CHPC: `bash chpc/setup_env.sh`, then `bash chpc/submit.sh smoke|1|2|3`.

Each run writes one JSON file (best fitness/genotype, diversity, convergence, per-generation
history, settings, provenance) to `results/phaseN/...`; existing files are skipped, so
interrupted jobs can be resubmitted. Add `--smoke` to any command for a tiny end-to-end
test (written to `results/smoke/`).

## Branching / PR workflow suggestion

- `main` — always runnable, tests passing.
- One feature branch per module/task, e.g. `feature/adaptive-de-nsga2`, `feature/graph-dissimilarity`.
- Small PRs mapped to Work Plan milestones; tag the module owner as reviewer.
- Run `pytest tests/ -q` and `python scripts/run_phase1.py --smoke` before opening a PR.

## Status

- [x] Repository scaffold, RealFramsticksInterface adapter, f0/f1 encodings
- [x] Gene (Levenshtein) and graph (Framsticks SimilMeasure) dissimilarity, with a graph-distance cache
- [x] Variants 1-5 and the organisers' Simple EA baseline
- [x] Evaluation metrics + statistics (Kruskal-Wallis + Dunn, Mann-Whitney vs Simple EA, competition score, generalisation, best-20 diversity)
- [x] Reproducibility: every run depends only on its seed (incl. native Gaussian-cache fix)
- [x] Phase 3 world configurations (14 settings + 4 held-out worlds) wired into the simulator
- [x] Manifest-based runner + CHPC PBS job arrays (chpc/)
- [x] Phase 1-3 experiments executed on the CHPC (October 2026)
- [x] Research report and poster (Poster Day, 29 October 2026)

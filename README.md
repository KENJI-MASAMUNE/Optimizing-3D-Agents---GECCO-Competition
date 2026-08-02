# DE-NSGA-II: Diversity-Enhanced Evolutionary Optimisation of 3D Agents in Framsticks

**COS700 Research Proposal — GECCO 2026 Automated 3D Design Competition**

Group: Salmaan Patel, Vanessa Rampedi, Ntsako Sithole
Supervisors: Prof. N. Pillay, Dr. T. Nyathi

This repository implements the algorithm variants and experimental pipeline described in
`docs/proposal.pdf`: a standard NSGA-II baseline and four Diversity-Enhancing NSGA-II
(DE-NSGA-II) variants — Gene Crowding, Graph Crowding, Hybrid Crowding, and the novel
**Adaptive DE-NSGA-II** — evaluated on centre-of-gravity (COG) trajectory-following tasks in
the Framsticks simulation environment (Komosinski & Mensfelt, GECCO 2025).

## Framsticks backend

`RealFramsticksInterface` (`src/framsticks_interface/interface.py`) wraps the real
competition library (`FramsticksLibCompetition`, via `framspy`) and is the only production
backend — there's no mock/fake shipped as a feature. Every algorithm/experiment module
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
│   ├── task_fitness.py        The four Phase 1 fitness functions, pure numpy (framspy-free)
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

```bash
python scripts/run_phase1.py     # controlled internal experiments (Section 5.3)
python scripts/run_phase2.py     # sensitivity analysis
python scripts/run_phase3.py     # clean-environment held-out evaluation
python -m src.experiments.statistics --results results/phase1/  # stats + Dunn's post-hoc
```

Each run writes a JSON + CSV record per (variant, encoding, task, repeat) into `results/`,
so partial runs can be resumed and results from different lab machines can be merged by
just copying files into the same `results/` tree before running `statistics.py`.

## Branching / PR workflow suggestion

- `main` — always runnable, tests passing.
- One feature branch per module/task, e.g. `feature/adaptive-de-nsga2`, `feature/graph-dissimilarity`.
- Small PRs mapped to Work Plan milestones; tag the module owner as reviewer.
- Run `pytest tests/ -q` and `python scripts/run_phase1.py --smoke` before opening a PR.

## Status

- [x] Repository scaffold, RealFramsticksInterface adapter, f0/f1 encoding stubs
- [] Gene (Levenshtein) and bounded graph dissimilarity
- [] Variant 1: NSGA-II baseline
- [] Variants 2-4: DE-NSGA-II gene / graph / hybrid crowding
- [] Variant 5: Adaptive DE-NSGA-II (sigmoidal schedule)
- [] Evaluation metrics + statistics module
- [x] Real `framspy` / `FramsticksLibCompetition`
- [ ] Phase 1-3 experiment execution on real Framsticks worlds
- [ ] Report write-up

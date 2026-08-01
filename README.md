# DE-NSGA-II: Diversity-Enhanced Evolutionary Optimisation of 3D Agents in Framsticks

**COS700 Research Proposal — GECCO 2026 Automated 3D Design Competition**

Group: Salmaan Patel, Vanessa Rampedi, Ntsako Sithole
Supervisors: Prof. N. Pillay, Dr. T. Nyathi

This repository implements the algorithm variants and experimental pipeline described in
`docs/proposal.pdf`: a standard NSGA-II baseline and four Diversity-Enhancing NSGA-II
(DE-NSGA-II) variants — Gene Crowding, Graph Crowding, Hybrid Crowding, and the novel
**Adaptive DE-NSGA-II** — evaluated on centre-of-gravity (COG) trajectory-following tasks in
the Framsticks simulation environment (Komosinski & Mensfelt, GECCO 2025).

## Why there's a "mock" Framsticks backend

The real Framsticks competition interface (`FramsticksLibCompetition`, via `framspy`) requires
a licensed simulator binary that isn't available in this environment / on a generic CI runner.
So that the codebase is runnable and testable by everyone on the team (and on GitHub Actions)
without that binary, `src/framsticks_interface/mock_framsticks.py` implements the **same
interface** (`evaluate`, `get_random_genotype`, `get_available_stimulators`, world config
loading) but simulates genotypes and fitness synthetically. Swap `USE_MOCK = False` in
`configs/default_config.yaml` once the real Framsticks/`framspy` install is available on a
lab machine — no other code changes are required, because every algorithm module only talks
to `FramsticksInterface`, never to the mock or the real library directly.

## Repository structure and ownership

This split mirrors the Work Plan (Section 7) in the proposal. Each member owns a package and
is the primary reviewer for pull requests touching it — but this is a group project, so
cross-review and pairing are expected, especially at the module boundaries.

```
src/
├── framsticks_interface/   # OWNER: Vanessa Rampedi
│   ├── interface.py           Abstract FramsticksInterface + real-lib adapter stub
│   ├── mock_framsticks.py     Mock simulator (genotypes, COG trajectories, fitness)
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
pytest tests/ -q                       # sanity-check everything works (mock backend)
python scripts/run_phase1.py --smoke   # quick smoke run: 2 variants x 1 task x 2 repeats
```

### Local Framsticks/framspy setup (each team member does this individually)

`framspy/` and the Framsticks simulator distribution (e.g. `Framsticks55/`) are **not**
committed to this repo — they're large, platform-specific (`.dll`/`.so`/`.dylib`), and
each team member needs their own local copy anyway. Both are already covered by
`.gitignore`. Follow `docs/framsticks_integration_notes.md` to set these up, then point
`configs/default_config.yaml` at your own local paths:

```yaml
framsticks:
  use_mock: false
  framspy_path: "/path/to/framspy"        # dir containing FramsticksLib.py directly
  frams_dir: "/path/to/Framsticks55"       # dir containing frams-objects.dll/.so/.dylib
  lib_name: null                            # leave null unless you renamed the library file
  sim_files: "eval-allcriteria.sim;deterministic.sim;recording-body-coords.sim"
```

Since these paths are personal to your machine, **don't commit your edited
`configs/default_config.yaml`** with real paths filled in — either keep `use_mock: true`
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

- [x] Repository scaffold, mock Framsticks backend, f0/f1 encoding stubs
- [x] Gene (Levenshtein) and bounded graph dissimilarity
- [x] Variant 1: NSGA-II baseline
- [x] Variants 2-4: DE-NSGA-II gene / graph / hybrid crowding
- [x] Variant 5: Adaptive DE-NSGA-II (sigmoidal schedule)
- [x] Evaluation metrics + statistics module
- [ ] Real `framspy` / `FramsticksLibCompetition` adapter (blocked on competition package access)
- [ ] Phase 1-3 experiment execution on real Framsticks worlds
- [ ] Report write-up

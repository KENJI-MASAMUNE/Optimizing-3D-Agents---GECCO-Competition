# Contributing (Team Workflow)

This is a 3-person COS700 group project, not an open-source project accepting outside
contributions -- this file just documents how *we* work together.

## Module ownership

See the table in `README.md`. Ownership means "primary reviewer and point of contact for
questions," not "only person allowed to touch it" -- pair up at the module boundaries
(e.g. `algorithms/` calling into `diversity/`).

## Workflow

1. Pull `main`, create a branch: `feature/<short-description>`, e.g. `feature/hybrid-weight-sweep`.
2. Make your change. Add/update tests in `tests/` for anything in `src/`.
3. Before opening a PR:
   ```bash
   pytest tests/ -q
   python scripts/run_phase1.py --smoke
   ```
4. Open a PR against `main`, tag the relevant module owner(s) as reviewer.
5. Squash-merge once approved and CI (`.github/workflows/ci.yml`) is green.

## Commit style

Short, imperative subject line referencing the Work Plan milestone where relevant, e.g.:

```
Implement Variant 3 graph crowding distance (Weeks 7-12)
Add Dunn's post-hoc test to statistics module
Fix bootstrap viability filter re-sampling non-viable f0 genotypes
```

## Coding conventions

- Python 3.11+, type hints on public functions.
- One module = one concern (see the package layout in README.md); avoid new cross-imports
  between `algorithms/` and `experiments/` beyond what's already there -- experiments call
  algorithms, never the reverse.
- New algorithm variants should be added as a thin wrapper around
  `src/algorithms/engine.run_evolution`, not as a copy-pasted GA loop (this is what keeps
  the five variants comparable and auditable per the proposal's Section 5 methodology).
- Every new dissimilarity measure or crowding strategy needs a unit test in `tests/`
  showing the formula gives the expected number on a hand-computable example.

## Data / results

`results/` is gitignored except for `.gitkeep` -- run outputs are large and
machine/seed-specific. If you want to share a results set with the team, zip it and drop it
in the shared drive / Slack, or open a PR adding a specific summary CSV under `docs/results/`.

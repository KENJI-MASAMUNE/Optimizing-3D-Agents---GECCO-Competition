# Running Phases 1–3 (locally and on the CHPC)

## What each phase is for

| Phase | Question it answers | Runs (default config) | Depends on |
|---|---|---|---|
| 1 — Controlled experiments | With standard ("out-of-the-box") settings, which variant × encoding works best on each task? | 6 algorithms (5 variants + Simple EA) × 2 encodings × 6 tasks × 20 repeats = **1440** (completed) | nothing |
| 2 — Sensitivity analysis | Are the Phase 1 conclusions robust to the GA settings, and what are good values for the knobs only some variants have? | **500** (3 sweeps, 2 tasks, 10 repeats; completed) | nothing |
| 3 — Competition-style evaluation | Do Adaptive DE-NSGA-II and each task's Phase 1 winner still beat NSGA-II and the Simple EA in new worlds, and how well do their agents generalise to worlds never seen during optimisation? | 14 settings × (Adaptive, NSGA-II, Simple EA) × 2 encodings × 20 + Phase 1 winners on their own task's settings = **1840** (completed) | Phase 1 (`phase3-select`) |

### The organisers' Simple EA baseline

`simple_ea` (src/algorithms/simple_ea.py) re-implements framspy's `FramsticksEvolution.py`
(DEAP `eaSimple`): population 50, tournament 5, mutation 0.9, crossover 0.2, starting from 50
copies of the simplest genotype, generational replacement. It gets the same number of
offspring as the NSGA-II variants (100 × 300 = 30,000 → 600 generations of 50), the same
constraints, cache and budget. It is listed in `variants`, so Phase 1 now has 960 runs; the
800 finished runs are skipped and only the 160 Simple EA runs are new. It is never chosen as a
Phase 3 "winner". `aggregate` now also writes `competition_scores.csv` (Simple EA = 0,
best = 1 per task, as in the 2026 competition slides) and tests every algorithm against the
Simple EA (Mann–Whitney, Bonferroni) in `stats.json → vs_simple_ea`.

### Tasks

Phase 1 has six tasks: our four proxy tasks from the proposal (straight_line_cog,
sinusoidal_bounce, velocity_forward, jump_height) plus the organisers' public example
functions 3 and 4 from framspy's `FramsticksLibCompetition.py` (`comp_distance`: distance
between the COG at birth and death in any direction; `comp_tall_runner`: that distance times
the mean COG height). None of our proxy tasks is identical to them: velocity_forward only
counts progress along +x, so a creature walking straight forward scores the same on both,
but sideways, backwards or vertical movement only counts for comp_distance. They run under
our Phase 1 settings (Energy0 2500, ~20k evaluations), so their values are not comparable
with published competition numbers. Phase 2 keeps its two tasks.

### Phase 3 (final design, as run in October 2026)

`phase3.mode: all_variants` with `variants: [adaptive_de_nsga2, nsga2_baseline, simple_ea]`
and `include_best_per_task: true`. In every one of the 14 settings, Adaptive DE-NSGA-II,
NSGA-II and the Simple EA run with **both** encodings (14 × 3 × 2 × 20 = 1680 runs), so the
novel variant is tested in every world. `phase3-select` also freezes each task's Phase 1
winner (highest mean best fitness; the Simple EA is never selected) and adds it to that
task's settings when it is not already one of the three (DE gene f1 on the sinusoid and
distance settings, DE graph f0 on tall runner, DE hybrid f0 on jump: 160 runs). Total
**1840 runs**. Phase 2 tuning is **not** applied (`use_phase2_tuning: false`): the tuned
hybrid weight (w = 1) would turn the hybrid jump winner into gene crowding.
`configs/worlds/phase3_worlds.yaml` has the 14 settings (the original 10 plus S11–S14 for
comp_distance and comp_tall_runner) and the 4 held-out worlds.

The proposal's original design (`mode: best_per_task`: winner + NSGA-II + Simple EA in the
winner's encoding only) is still available in the code.

### Why Phase 2 exists

Every evolutionary algorithm has settings that could change the outcome. Phase 2 checks
three of them, one at a time, with everything else held at the Phase 1 defaults:

* **A. Population size (50–200) × tournament size (2, 3, 5)** for DE-NSGA-II gene crowding.
  The evaluation budget is held fixed (population × generations = 6000), so a bigger
  population means fewer generations — the real trade-off you face under the competition
  budget. *Question:* would a reader's conclusion change if we had picked a different
  population or selection pressure? If fitness barely moves across this grid, Phase 1's
  defaults were reasonable and its conclusions are robust. If it moves a lot, that is a
  finding in itself and a caveat for Phase 1.
* **B. Adaptive switching point τ (0.25, 0.5, 0.75)** for Variant 5. τ decides *when* the
  sigmoid hands selection over from structural diversity to fitness. 0.5 was an arbitrary
  choice in the proposal. This is the main hyper-parameter of the novel contribution, so
  the report needs to show how sensitive it is.
* **C. Gene/graph weight w (0, 0.25, 0.5, 0.75, 1)** for Variant 4, separately for f0 and
  f1 ("weight tuned per encoding type" in the proposal). w = 1 is Variant 2 and w = 0 is
  Variant 3, so the sweep also shows whether mixing the two measures beats using either one.

### Shouldn't Phase 2 come before Phase 1?

It's a fair question. Both orders are used in the literature:

* **Tune first, then compare:** the main comparison uses tuned settings. The risk is
  *tuning bias*: the baseline (NSGA-II) has no τ or w to tune, so tuning only the new
  variants on the same tasks you then report makes them look better than they are.
* **Compare with defaults, then analyse sensitivity** (the proposal's order): Phase 1 is a
  fair out-of-the-box comparison where nobody gets tuned. Phase 2 then asks whether those
  conclusions hold and finds good τ and w values.

What makes the proposal's order work is Phase 3. This code now uses a
train → tune → test structure:

1. Phase 1 (proxy tasks, defaults): fair comparison, picks the winner per task.
2. Phase 2 (proxy tasks): robustness + tuning of τ (Variant 5) and w per encoding (Variant 4).
3. Phase 3 (**new** worlds): `phase3-select` freezes the Phase 1 winners **and** the Phase 2
   tuned values, then tests them in the 10 reconstructed competition settings and the
   held-out worlds. Because those worlds were never used for choosing or tuning anything,
   Phase 3 gives an unbiased estimate.

In practice, Phases 1 and 2 don't depend on each other, so on the CHPC you can **submit
them at the same time**. Only Phase 3 has to wait for both.

## Phase 3 worlds

The competition never publishes its 10 settings. `configs/worlds/phase3_worlds.yaml`
rebuilds a 10-setting suite using **only** the parameters the organisers say can vary:
lifespan, world size/heightfield, water level, gravity, body constraints (2023/24), plus
terrain and initial rotation (2025). It also has 4 held-out worlds for the Generalisation
Score. Every world was checked on the real simulator: it loads, it changes the COG path,
and switching back restores the base world exactly. Rotation is emulated by rotating the
target direction (`target_heading_deg`), because Framsticks' own non-zero rotation options
are random and would make evaluation noisy. See the YAML comments for the reasoning
behind each setting. Describe this in the report as a reconstructed suite, not the official one.

## Commands (same locally and on the cluster)

```bash
python -m src.experiments.cli list    --phase 1            # manifest size + first output paths
python -m src.experiments.cli run     --phase 1 --workers 8
python -m src.experiments.cli status  --phase 1
python -m src.experiments.cli aggregate --phase 1          # results/analysis/phase1/*.csv, stats.json
python -m src.experiments.cli phase3-select                # after Phases 1 (+2)
python -m src.experiments.cli run     --phase 3 --workers 8
# add --smoke for a tiny end-to-end test (writes to results/smoke/)
```

Each run writes one JSON file containing: best fitness and genotype, diversity, both
convergence measures, the per-generation history (for convergence plots), engine settings,
sim files, base world, host and git commit. Runs that already have an output file are
skipped, so you can re-run or resubmit any command and it picks up where it stopped.
Every run re-seeds all RNGs from its own seed, so the result is the same no matter which
process or node runs it.

## CHPC (Lengau, PBS Pro) step by step

1. **Upload** (from your laptop; `$USER` = your CHPC username):
   ```bash
   ssh $USER@lengau.chpc.ac.za "mkdir -p /mnt/lustre/users/$USER/cos700"
   # repo (or git clone it on the login node)
   rsync -av --exclude results --exclude .git --exclude Framsticks55 --exclude framspy \
         Optimizing-3D-Agents---GECCO-Competition/ $USER@lengau.chpc.ac.za:/mnt/lustre/users/$USER/cos700/repo/
   rsync -av Optimizing-3D-Agents---GECCO-Competition/framspy/ $USER@lengau.chpc.ac.za:/mnt/lustre/users/$USER/cos700/framspy/
   # only the Linux library + data/ are needed from Framsticks55
   rsync -av Optimizing-3D-Agents---GECCO-Competition/Framsticks55/frams-objects.so \
         Optimizing-3D-Agents---GECCO-Competition/Framsticks55/data \
         $USER@lengau.chpc.ac.za:/mnt/lustre/users/$USER/cos700/Framsticks55/
   ```
   (WinSCP or MobaXterm work too. If you copy the scripts from Windows, make sure they have
   Unix line endings: `.gitattributes` enforces LF in git, otherwise run `dos2unix chpc/*`.)
2. **Edit `chpc/env.sh` once:** set `CHPC_PROJECT` to your project short name. Change `WORK`
   only if you used a different folder.
3. **Set up the environment (offline).** Lengau's login and compute nodes cannot reach
   GitHub or PyPI, and its Python modules stop at 3.9 while framspy needs Python >= 3.12.
   So build an offline bundle on your laptop and upload it:
   ```powershell
   # laptop, PowerShell, in the repo folder
   powershell -ExecutionPolicy Bypass -File chpc\make_bundle.ps1     # creates .\chpc_bundle (~140 MB)
   scp -r chpc_bundle spatel1@lengau.chpc.ac.za:~/lustre/cos700/
   ```
   then on the CHPC run `bash chpc/setup_env.sh`. It unpacks a standalone Linux CPython 3.12
   (glibc 2.17-compatible), creates `~/lustre/cos700/venv312`, installs the wheels offline,
   loads `gcc/9.2.0` (needed for `GLIBCXX_3.4.26`; the default gcc 6.1 is too old), then
   runs `scripts/check_framsticks.py` and the unit tests. It must end with `SETUP OK`.
4. **Smoke test on a compute node:** `bash chpc/submit.sh smoke`, then read
   `cos700_smoke.out`. It must end with `REPRODUCIBLE` and `SMOKE OK`. It also prints the
   per-evaluation timing for that node.
5. **Submit Phases 1 and 2 together:**
   ```bash
   bash chpc/submit.sh 1          # 800 runs -> 8 array elements x 23 cores
   bash chpc/submit.sh 2          # 500 runs -> 5 array elements x 23 cores
   qstat -u $USER
   python -m src.experiments.cli status --phase 1
   ```
   Logs go to `$RESULTS_DIR/logs/`. If an element hits its walltime, resubmit the same
   command; finished runs are skipped.
6. **New Phase 1 runs (Simple EA + the two example functions):** `bash chpc/submit.sh 1 7 12:00:00`
   (640 new runs; the 800 finished ones are skipped), then
   `python -m src.experiments.cli aggregate --phase 1`.
7. **Phase 3:** `python -m src.experiments.cli phase3-select` (check the six printed winners),
   then `bash chpc/submit.sh 3 4 24:00:00`.
8. **Analyse:** `python -m src.experiments.cli aggregate --phase N` for N = 1, 2, 3, then
   `python -m src.experiments.cli best-diversity --phase 1` (and `--phase 3`): Komosinski &
   Mensfelt's diversity test on the 20 best solutions per setup, written to
   `analysis/phaseN/best20_diversity*.csv` and `best20_diversity_stats.json`. Then rsync
   `$RESULTS_DIR` back to your laptop.

### Where things live on Lengau (since 19 Oct 2026)

The code runs from **`~/cos700code`** (home folder), not `$WORK/repo`: during a Lustre
outage (several OSTs inactive) small source files on Lustre hung on read/write. `chpc/env.sh`
sets `REPO=$HOME/cos700code`; `cos700` (in `~/.bashrc`) does
`cd ~/cos700code && source chpc/env.sh`. Everything large stays on Lustre under
`$WORK=/mnt/lustre/users/spatel1/cos700`: `framspy`, `Framsticks55`, `python312`, `venv312`,
`results`. To update the code, upload one archive and extract it in home:

```powershell
# laptop, repo folder
tar --exclude=__pycache__ --exclude=*.pyc -czf repo_update.tgz src configs chpc tests docs
scp repo_update.tgz spatel1@lengau.chpc.ac.za:~/
```
```bash
# Lengau
tar -xzf ~/repo_update.tgz -C ~/cos700code && rm ~/repo_update.tgz && cos700 && python -m pytest tests -q
```
If a command on Lustre hangs, check `lfs df -h $WORK | grep inactive`, use the other login
node (`ssh login1`), and contact helpdesk@chpc.ac.za rather than retrying.

### Sizing (measured, Oct 2026)

Full-size run (population 100, 60 generations, Energy0 2500): **3–8 CPU-minutes** on a
modern core. Graph crowding with f1 is the slowest; about 30% of evaluations are saved by
the fitness cache. Lengau's Haswell cores are probably 1.5–2× slower. Rough totals:

| Phase | Runs | CPU-hours (est.) | With `submit.sh` defaults |
|---|---|---|---|
| 1 | 1440 (640 new) | 250–450 (measured: 15–35 min/run, graph/hybrid f1 ~70 min) | 7 nodes, ~5 h wall for the new runs |
| 2 | 500 | 50–120 | 5 nodes (population 200 runs are slower) |
| 3 (all_variants + winners, as run) | 1840 | ≈ 690 (measured) | 18 array elements × 23 cores, ~5–8 h wall |

Measured totals for the three phases: about 1,860 CPU-hours (Phase 1 ≈ 810, Phase 2 ≈ 350,
Phase 3 ≈ 690). Run `best-diversity --phase 3` as a batch job, not on the login node: it
needs about 18,000 graph comparisons and the login node kills it.

### Queue notes

`chpc/phase.pbs` asks for the `serial` queue with 23 cores (the per-job maximum there) and
24 h walltime. Each array element runs one Python process per core. If your allocation
prefers `smp` (24 cores, longer walltime), change `#PBS -q` and `ncpus`/`NCPUS` together.
Check limits with `qstat -Qf serial` and the CHPC user guide; queue limits change over time.

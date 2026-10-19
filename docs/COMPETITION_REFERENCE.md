# Competition reference results

Raw best-fitness values on the organisers' public example functions, from the 2026 winner's
thesis: A. Brassat, *Disertatie* (2026), Table 5 / Appendix B
(github.com/KebabRonin/Disertatie). The thesis authors ran every algorithm themselves, 20 runs
each, with the reference simulator settings and the competition budget (100,000 evaluations,
1 hour of algorithm time excluding simulation). Mean of the best fitness per run:

| Function | Simple EA f1 (organisers' baseline) | Simple EA f0 | AdaptMut f0 (2024 winner) | DynMut f0 (2026 winner) |
|---|---|---|---|---|
| fn3 `comp_distance` ‖p0 − pT‖ | 87.7 | 171.6 | 249.0 | 361.6 (345.4 without sensor noise) |
| fn4 `comp_tall_runner` ‖p0 − pT‖ · mean(max(0, z)) | 30.3 | 70.5 | 115.7 | 173.6 |
| fn5 `comp_rising_cog` 1000 − RMSE(z, 0→10) | 995.51 | 995.38 | 996.19 | 996.01 |
| fn6 1/Σ\|0.5 + 0.1 sin(t/100) − z\| | 0.00158 | 0.00163 | 0.00162 | not run |

* fn3–fn5 are `TEST_FUNCTION` 3–5 in framspy's `FramsticksLibCompetition.py`; fn6 is in the
  newer framspy the thesis used. fn3–fn5 are registered as tasks in
  `src/framsticks_interface/task_fitness.py`; `configs/competition_config.yaml` +
  `bash chpc/submit.sh comp` reproduce the conditions (Energy0 10000, 100k evaluations, f0).
* Our `simple_ea` f0 should land near 171.6 / 70.5 if the setup matches theirs.
* fn6 shows the same ceiling effect our first sinusoid had: every algorithm scores about the same.
* The competition's own scores are normalised (Simple EA = 0, best entry = 1 per function,
  averaged over 10 hidden functions); `aggregate` writes the same score as
  `competition_scores.csv`.
* Timing (Oct 2026): one evaluation at Energy0 10000 ≈ 0.14 s locally (≈ 4.5× Energy0 2500);
  a 100k-evaluation run ≈ 4 h of simulation locally, 8–12 h on Lengau.

"""
Regression tests for the fixes made in the October 2026 audit (see docs/AUDIT_2026-10.md).
All run against the test double -- no Framsticks needed.
"""
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms import de_nsga2, nsga2_baseline  # noqa: E402
from src.algorithms.engine import (  # noqa: E402
    CrowdingStrategy, DiversityMeasure, EngineConfig, Individual, _pairwise_dissimilarity,
    crowded_tournament, fast_non_dominated_sort, run_evolution,
)
from src.experiments.run_single import run_spec  # noqa: E402
from src.experiments.specs import build_phase1_specs, build_phase2_specs, build_phase3_specs  # noqa: E402
from src.framsticks_interface.task_fitness import (  # noqa: E402
    fitness_sinusoidal_bounce, fitness_straight_line_cog, fitness_velocity_forward, TARGET_LINE_DISTANCE,
)
from src.framsticks_interface.worlds import load_worlds, world_from_dict  # noqa: E402
from tests.fakes import FakeFramsticksInterface  # noqa: E402

CONFIG = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "default_config.yaml"))


def _cfg(**kw):
    base = dict(population_size=8, n_generations=4, crossover_prob=0.7, mutation_prob=0.3,
                tournament_size=2, verbose=False)
    base.update(kw)
    return EngineConfig(**base)


# ------------------------------------------------------------------ selection
def test_tournament_size_is_honoured():
    pop = [Individual(genotype=str(i), encoding="f1", rank=i, crowding_dist=1.0) for i in range(10)]
    rng = random.Random(0)
    # k == population size -> the best-ranked individual must always win
    assert all(crowded_tournament(pop, rng, k=10).rank == 0 for _ in range(20))
    # larger tournaments => stronger selection pressure (lower mean rank of winners)
    mean_rank = lambda k: np.mean([crowded_tournament(pop, random.Random(s), k).rank for s in range(400)])  # noqa: E731
    assert mean_rank(5) < mean_rank(2)


def test_invalid_individuals_ranked_behind_all_valid():
    inds = [Individual("a", "f1", fitness=-5.0, diversity_score=0.1),
            Individual("b", "f1", fitness=0.0, diversity_score=0.0, valid=False),
            Individual("c", "f1", fitness=1.0, diversity_score=0.0)]
    fronts = fast_non_dominated_sort(inds)
    assert fronts[-1] == [1]
    assert inds[1].rank > inds[0].rank and inds[1].rank > inds[2].rank


# ------------------------------------------------------------------ Variant 4 hybrid
def test_hybrid_matrix_is_normalised_blend():
    iface = FakeFramsticksInterface(seed=0)
    gs = [iface.get_random_genotype("f1") for _ in range(6)]
    gene = _pairwise_dissimilarity(iface, gs, "f1", DiversityMeasure.GENE)
    graph = _pairwise_dissimilarity(iface, gs, "f1", DiversityMeasure.GRAPH)
    h1 = _pairwise_dissimilarity(iface, gs, "f1", DiversityMeasure.HYBRID, gene_weight=1.0)
    h0 = _pairwise_dissimilarity(iface, gs, "f1", DiversityMeasure.HYBRID, gene_weight=0.0)
    gmax = max(max(r) for r in gene)
    hmax = max(max(r) for r in graph)
    assert np.allclose(np.array(h1), np.array(gene) / gmax)
    assert np.allclose(np.array(h0), np.array(graph) / hmax)
    assert max(max(r) for r in h1) <= 1.0 + 1e-12


def test_hybrid_weight_changes_the_run():
    r = {}
    for w in (0.0, 1.0):
        iface = FakeFramsticksInterface(seed=3)
        res = de_nsga2.run_hybrid_crowding(iface, "f1", "straight_line_cog", _cfg(hybrid_weight=w), seed=4)
        r[w] = [ind.genotype for ind in res.population]
    assert r[0.0] != r[1.0]


# ------------------------------------------------------------------ best-ever, budget, cache
def test_best_individual_is_best_ever_evaluated():
    iface = FakeFramsticksInterface(seed=1)
    res = de_nsga2.run_gene_crowding(iface, "f1", "velocity_forward", _cfg(n_generations=6), seed=2)
    assert res.best_individual.fitness >= max(i.fitness for i in res.population)
    assert res.best_individual.fitness == max(h.best_fitness for h in res.history)
    assert all(res.history[i].best_fitness <= res.history[i + 1].best_fitness for i in range(len(res.history) - 1))


def test_budget_exhaustion_ends_run_gracefully():
    iface = FakeFramsticksInterface(seed=2, eval_budget=13)
    res = nsga2_baseline.run(iface, "f1", "straight_line_cog", _cfg(n_generations=10), seed=1)
    assert res.stopped_by_budget
    assert iface.n_evaluations == 13
    assert res.best_individual.evaluated


def test_fitness_cache_saves_evaluations_without_changing_results():
    out = {}
    for cache in (False, True):
        iface = FakeFramsticksInterface(seed=5)
        res = nsga2_baseline.run(iface, "f1", "straight_line_cog", _cfg(cache_fitness=cache, n_generations=6), seed=9)
        out[cache] = (res.best_individual.fitness, iface.n_evaluations, res.n_cache_hits)
    assert out[True][0] == out[False][0]
    assert out[True][1] <= out[False][1]


def test_adaptive_weight_recorded_and_decreasing():
    iface = FakeFramsticksInterface(seed=0)
    res = run_evolution(iface, "f1", "straight_line_cog", CrowdingStrategy.ADAPTIVE, DiversityMeasure.GENE,
                        _cfg(n_generations=5), seed=0)
    w = [h.crowding_weight for h in res.history]
    assert all(a >= b for a, b in zip(w, w[1:])) and w[0] > 0.9 and w[-1] < 0.1


# ------------------------------------------------------------------ fitness functions
def test_sinusoid_has_no_ceiling_effect_and_ignores_height():
    t = np.arange(2500)
    wave = 0.1 * np.sin(t / 100.0)
    static = np.tile([0.0, 0.0, 0.5], (2500, 1))
    follower_low = np.stack([np.zeros(2500), np.zeros(2500), 0.05 + wave], axis=1)
    follower_high = np.stack([np.zeros(2500), np.zeros(2500), 3.0 + wave], axis=1)
    anti_phase = np.stack([np.zeros(2500), np.zeros(2500), 0.5 - wave], axis=1)
    assert fitness_sinusoidal_bounce(follower_low) == pytest.approx(1.0)
    assert fitness_sinusoidal_bounce(follower_high) == pytest.approx(1.0)      # height-invariant
    assert fitness_sinusoidal_bounce(static) == pytest.approx(1 / (1 + 2 ** -0.5), abs=0.01)
    assert fitness_sinusoidal_bounce(anti_phase) < fitness_sinusoidal_bounce(static)
    # the old formula gave the stationary creature 0.94 -- now there is real headroom
    assert fitness_sinusoidal_bounce(static) < 0.6


def test_target_heading_rotates_direction():
    start = np.array([1.0, 1.0, 0.0])
    t = np.linspace(0, 1, 50)
    along_y = start + t[:, None] * np.array([0.0, TARGET_LINE_DISTANCE, 0.0])
    assert fitness_straight_line_cog(along_y, target_heading_deg=90) == pytest.approx(1.0)
    assert fitness_straight_line_cog(along_y) < 0.5
    assert fitness_velocity_forward(along_y, target_heading_deg=90) == pytest.approx(TARGET_LINE_DISTANCE)
    assert fitness_velocity_forward(along_y) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------------ worlds
def test_phase3_worlds_file_is_consistent():
    worlds, settings, held_out = load_worlds()
    assert len(settings) == 14
    assert len({s.name for s in settings}) == 14
    assert set(held_out).isdisjoint({s.world for s in settings})
    tasks = {t["name"] for t in CONFIG["tasks"]}
    assert all(s.task in tasks for s in settings)


def test_world_validation():
    with pytest.raises(ValueError):
        world_from_dict("bad", {"gravity": 50})           # outside Framsticks range
    with pytest.raises(ValueError):
        world_from_dict("bad", {"world_type": 2})         # heightfield without a map
    with pytest.raises(ValueError):
        world_from_dict("bad", {"rotation": 2})           # random rotation -> nondeterministic
    with pytest.raises(ValueError):
        world_from_dict("bad", {"wind": 3})               # unknown key
    w = world_from_dict("ok", {"water_level": 0.5, "target_heading_deg": 90})
    assert w.params == {"water_level": 0.5} and w.task_params == {"target_heading_deg": 90.0}


# ------------------------------------------------------------------ manifests
def test_run_ids_unique_and_stable():
    for build in (build_phase1_specs, build_phase2_specs):
        specs = build(CONFIG)
        paths = [s.output_path("results").as_posix() for s in specs]
        assert len(paths) == len(set(paths)), f"{build.__name__} has colliding output paths"
        assert [s.run_id for s in build(CONFIG)] == [s.run_id for s in specs]
    assert len(build_phase1_specs(CONFIG)) == len(CONFIG["variants"]) * 2 * len(CONFIG["tasks"]) * CONFIG["experiment"]["n_repeats"]


def test_phase2_population_sweep_keeps_budget_fixed():
    specs = [s for s in build_phase2_specs(CONFIG) if s.group == "A_population_tournament"]
    budgets = {s.overrides_dict["population_size"] * s.overrides_dict["n_generations"] for s in specs}
    assert max(budgets) - min(budgets) <= max(CONFIG["phase2"]["population_sizes"])


def test_phase3_specs_need_selection_or_all_variants():
    best = dict(CONFIG, phase3=dict(CONFIG["phase3"], mode="best_per_task"))
    with pytest.raises(ValueError):
        build_phase3_specs(best, selection=None)
    sel = {t["name"]: {"variant": "de_nsga2_graph", "encoding": "f0"} for t in CONFIG["tasks"]}
    specs = build_phase3_specs(best, selection=sel)
    assert len(specs) == 14 * 3 * CONFIG["phase3"]["n_repeats"]  # winner + NSGA-II + Simple EA controls
    allv = dict(CONFIG, phase3=dict(CONFIG["phase3"], mode="all_variants"))
    assert len(build_phase3_specs(allv)) == 14 * len(CONFIG["phase3"]["variants"]) * 2 * CONFIG["phase3"]["n_repeats"]
    allv5 = dict(CONFIG, phase3={k: v for k, v in CONFIG["phase3"].items() if k != "variants"} | {"mode": "all_variants"})
    assert len(build_phase3_specs(allv5)) == 14 * len(CONFIG["variants"]) * 2 * CONFIG["phase3"]["n_repeats"]


def test_run_spec_reproducible_and_writes_json(tmp_path):
    spec = build_phase2_specs(CONFIG, smoke=True)[0]
    iface = FakeFramsticksInterface(seed=0)
    r1 = run_spec(iface, spec, CONFIG, tmp_path, smoke=True, verbose=False)
    run_spec(iface, build_phase1_specs(CONFIG, smoke=True)[3], CONFIG, tmp_path, smoke=True, verbose=False)
    r2 = run_spec(iface, spec, CONFIG, tmp_path / "again", smoke=True, verbose=False)
    assert r1["best_fitness"] == r2["best_fitness"] and r1["best_genotype"] == r2["best_genotype"]
    data = json.loads(spec.output_path(tmp_path).read_text())
    assert data["sweep"] and data["history"]
    assert not math.isnan(data["best_fitness"])


# ------------------------------------------------------------------ organisers' Simple EA
def test_simple_ea_runs_reproducibly_with_same_offspring_budget(tmp_path):
    from src.algorithms import simple_ea
    from src.experiments.run_single import engine_config_from
    base = engine_config_from(CONFIG, verbose=False)
    cfg = simple_ea.config_for(base, CONFIG["simple_ea"])
    assert cfg.population_size == 50 and cfg.tournament_size == 5
    assert cfg.population_size * cfg.n_generations == base.population_size * base.n_generations
    small = simple_ea.config_for(_cfg(population_size=10, n_generations=6), {"population_size": 6})
    out = []
    for _ in range(2):
        iface = FakeFramsticksInterface(seed=0)
        iface.reseed(7)
        res = simple_ea.run(iface, "f1", "velocity_forward", small, seed=7)
        out.append((res.best_individual.fitness, res.best_individual.genotype, [h.best_fitness for h in res.history]))
        assert res.best_individual.fitness == max(h.best_fitness for h in res.history)
        assert all(a <= b for a, b in zip(out[-1][2], out[-1][2][1:]))
    assert out[0] == out[1]


def test_simple_ea_only_evaluates_changed_individuals():
    from src.algorithms import simple_ea
    iface = FakeFramsticksInterface(seed=1)
    cfg = _cfg(population_size=8, n_generations=5, mutation_prob=0.0, crossover_prob=0.0, cache_fitness=False)
    res = simple_ea.run(iface, "f1", "jump_height", cfg, seed=3)
    assert iface.n_evaluations == 8          # nothing changes after generation 0 -> no new simulations


def test_simple_ea_is_never_selected_as_phase1_winner(tmp_path):
    from src.experiments.specs import select_best_per_task
    for v, f in (("simple_ea", 99.0), ("nsga2_baseline", 1.0)):
        (tmp_path / f"{v}.json").write_text(json.dumps({"variant": v, "encoding": "f1", "task": "jump_height",
                                                         "best_fitness": f, "run_id": v}))
    assert select_best_per_task(tmp_path)["jump_height"]["variant"] == "nsga2_baseline"


def test_competition_score_simple_ea_zero_best_one():
    import pandas as pd
    from src.experiments.aggregate import competition_scores
    summ = pd.DataFrame({"task": ["t"] * 3, "encoding": ["f0", "f0", "f1"],
                         "variant": ["simple_ea", "nsga2_baseline", "simple_ea"], "best_fitness_mean": [2.0, 6.0, 1.0]})
    c = competition_scores(summ, ["task"]).set_index(["variant", "encoding"])["competition_score"]
    assert c[("simple_ea", "f0")] == 0 and c[("nsga2_baseline", "f0")] == 1 and c[("simple_ea", "f1")] < 0


def test_competition_example_functions():
    from src.framsticks_interface.task_fitness import TASK_FITNESS_FUNCTIONS as T
    path = np.array([[0.0, 0.0, 1.0], [3.0, 4.0, 1.0]])
    assert T["comp_distance"](path) == pytest.approx(5.0)
    assert T["comp_tall_runner"](path) == pytest.approx(5.0)
    rising = np.stack([np.zeros(11), np.zeros(11), np.linspace(0, 10, 11)], axis=1)
    assert T["comp_rising_cog"](rising) == pytest.approx(1000.0)


def test_best_solution_diversity_and_stats():
    import pandas as pd
    from src.evaluation.best_diversity import best_solution_diversity, diversity_stats
    iface = FakeFramsticksInterface(seed=0)
    rows = []
    for v, spread in (("nsga2_baseline", 1), ("de_nsga2_gene", 6)):
        base = iface.get_random_genotype("f1")
        for r in range(8):
            g = base
            for _ in range(spread * (r + 1)):
                g = iface.mutate(g, "f1")
            rows.append({"task": "t", "encoding": "f1", "variant": v, "run_id": f"{v}{r}", "best_genotype": g})
    per = best_solution_diversity(iface, pd.DataFrame(rows), ["task", "encoding", "variant"])
    assert len(per) == 16 and set(per.columns) >= {"gene", "graph"}
    means = per.groupby("variant")["gene"].mean()
    assert means["de_nsga2_gene"] > means["nsga2_baseline"]
    st = diversity_stats(per, ["task", "encoding", "variant"])
    assert "t__f1" in st and "p" in st["t__f1"]["gene"]

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.algorithms import adaptive_de_nsga2, de_nsga2, nsga2_baseline
from src.algorithms.engine import EngineConfig, fast_non_dominated_sort, Individual
from tests.fakes import FakeFramsticksInterface


def _tiny_config():
    return EngineConfig(population_size=8, n_generations=3, crossover_prob=0.7,
                         mutation_prob=0.3, tournament_size=2)


def test_fast_non_dominated_sort_basic():
    inds = [Individual(genotype="a", encoding="f1", fitness=1.0, diversity_score=0.0),
            Individual(genotype="b", encoding="f1", fitness=0.5, diversity_score=0.5),
            Individual(genotype="c", encoding="f1", fitness=0.1, diversity_score=0.1)]
    fronts = fast_non_dominated_sort(inds)
    # 'c' is dominated by both 'a' and 'b' (worse on both objectives); should not be in front 0
    front0_genotypes = {inds[i].genotype for i in fronts[0]}
    assert "c" not in front0_genotypes


def test_nsga2_baseline_runs_and_improves():
    interface = FakeFramsticksInterface(seed=0)
    config = _tiny_config()
    result = nsga2_baseline.run(interface, "f1", "straight_line_cog", config, seed=1)
    assert result.best_individual.fitness > 0.0
    assert len(result.history) > 0
    assert result.n_evaluations > 0


def test_de_nsga2_gene_crowding_runs():
    interface = FakeFramsticksInterface(seed=1)
    config = _tiny_config()
    result = de_nsga2.run_gene_crowding(interface, "f1", "velocity_forward", config, seed=2)
    assert result.best_individual.fitness >= 0.0
    assert len(result.population) == config.population_size


def test_de_nsga2_graph_crowding_runs():
    interface = FakeFramsticksInterface(seed=2)
    config = _tiny_config()
    result = de_nsga2.run_graph_crowding(interface, "f0", "jump_height", config, seed=3)
    assert result.best_individual.fitness >= 0.0


def test_de_nsga2_hybrid_crowding_runs():
    interface = FakeFramsticksInterface(seed=3)
    config = _tiny_config()
    result = de_nsga2.run_hybrid_crowding(interface, "f1", "sinusoidal_bounce", config, seed=4)
    assert result.best_individual.fitness >= 0.0


def test_adaptive_de_nsga2_runs():
    interface = FakeFramsticksInterface(seed=4)
    config = _tiny_config()
    result = adaptive_de_nsga2.run(interface, "f1", "straight_line_cog", config, seed=5)
    assert result.best_individual.fitness >= 0.0


def test_adaptive_sigmoid_weight_monotonic_decreasing():
    weights = [adaptive_de_nsga2.sigmoid_weight(p, tau=0.5, steepness=10.0)
               for p in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert all(weights[i] >= weights[i + 1] for i in range(len(weights) - 1))
    assert weights[0] > 0.9   # early: near-full exploration weight
    assert weights[-1] < 0.1  # late: near-zero exploration weight


def test_population_size_respected_across_generations():
    interface = FakeFramsticksInterface(seed=5)
    config = _tiny_config()
    result = nsga2_baseline.run(interface, "f1", "straight_line_cog", config, seed=6)
    assert len(result.population) == config.population_size

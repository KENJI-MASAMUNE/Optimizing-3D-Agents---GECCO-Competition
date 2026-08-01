import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diversity.graph_dissimilarity import genotype_to_graph, graph_distance


def test_identical_genotype_zero_graph_distance():
    assert graph_distance("XLRNng", "XLRNng", encoding="f1") == 0.0


def test_graph_distance_symmetric():
    a, b = "XLRNng", "QQNTTg"
    assert abs(graph_distance(a, b, "f1") - graph_distance(b, a, "f1")) < 1e-9


def test_graph_distance_bounded():
    d = graph_distance("X", "XLRNngGgTtXXQQNNTTgg", "f1")
    assert 0.0 <= d <= 1.0


def test_f0_graph_has_expected_node_count():
    genotype = "aaaaaaaa bbbbbbbb cccccccc"
    g = genotype_to_graph(genotype, "f0")
    assert g.number_of_nodes() == 3


def test_bounded_node_count_respects_max_nodes():
    from src.diversity.graph_dissimilarity import MAX_NODES
    long_genotype = "X" * 500
    g = genotype_to_graph(long_genotype, "f1")
    assert g.number_of_nodes() <= MAX_NODES

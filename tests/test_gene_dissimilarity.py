import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.diversity.gene_dissimilarity import (
    average_gene_dissimilarity,
    gene_distance,
    pairwise_gene_dissimilarity,
)


def test_identical_genotypes_zero_distance():
    assert gene_distance("XLRN", "XLRN") == 0.0


def test_distance_is_symmetric():
    a, b = "XLRNng", "XRRNnT"
    assert gene_distance(a, b) == gene_distance(b, a)


def test_distance_normalised_bounds():
    d = gene_distance("X", "XLRNngGgTtXX")
    assert 0.0 <= d <= 1.0


def test_pairwise_matrix_symmetric_and_zero_diagonal():
    genotypes = ["XLRN", "XLRG", "NNTT"]
    m = pairwise_gene_dissimilarity(genotypes)
    for i in range(len(genotypes)):
        assert m[i][i] == 0.0
        for j in range(len(genotypes)):
            assert m[i][j] == m[j][i]


def test_average_dissimilarity_matches_manual_formula():
    genotypes = ["XLRN", "NNTT", "GgTtXX"]
    matrix = pairwise_gene_dissimilarity(genotypes)
    avg = average_gene_dissimilarity(genotypes)
    n = len(genotypes)
    for i in range(n):
        expected = sum(matrix[i]) / (n - 1)
        assert abs(avg[i] - expected) < 1e-9


def test_single_individual_zero_diversity():
    assert average_gene_dissimilarity(["XLRN"]) == [0.0]

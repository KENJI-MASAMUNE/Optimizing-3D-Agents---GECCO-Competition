"""
Genetic dissimilarity measure: Levenshtein distance on genotype strings.

Matches Section 2.4 / 4.4.2 of the proposal: "Levenshtein distance computed on genotype
strings -- computationally efficient and encoding-agnostic." Used as the diversity metric
in DE-NSGA-II Variant 2 (Gene Crowding) and as one component of Variant 4 (Hybrid Crowding).
"""
from __future__ import annotations

from functools import lru_cache

try:
    import Levenshtein as _lev

    def _raw_distance(a: str, b: str) -> int:
        return _lev.distance(a, b)
except ImportError:  # pragma: no cover - fallback if python-Levenshtein isn't installed
    def _raw_distance(a: str, b: str) -> int:
        return _python_levenshtein(a, b)


def _python_levenshtein(a: str, b: str) -> int:
    """Pure-python fallback Levenshtein distance (Wagner-Fischer)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def gene_distance(genotype_a: str, genotype_b: str, normalise: bool = True) -> float:
    """Levenshtein edit distance between two genotype strings.

    If `normalise` is True (default), the raw edit distance is divided by the length of the
    longer string, giving a value in [0, 1] that's comparable across genotypes of different
    lengths -- important since f1 genotypes vary substantially in length during evolution.
    """
    if genotype_a == genotype_b:
        return 0.0
    d = _raw_distance(genotype_a, genotype_b)
    if not normalise:
        return float(d)
    max_len = max(len(genotype_a), len(genotype_b), 1)
    return d / max_len


def pairwise_gene_dissimilarity(genotypes: list[str]) -> list[list[float]]:
    """Full pairwise dissimilarity matrix for a population's genotype strings."""
    n = len(genotypes)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = gene_distance(genotypes[i], genotypes[j])
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def average_gene_dissimilarity(genotypes: list[str]) -> list[float]:
    """Per-individual average dissimilarity to the rest of the population -- this IS the
    DE-NSGA-II crowding distance formula from Section 2.4:
        I_i.crowding_dist = (1 / (|I| - 1)) * sum_{j != i} D_ij
    """
    n = len(genotypes)
    if n <= 1:
        return [0.0] * n
    matrix = pairwise_gene_dissimilarity(genotypes)
    return [sum(row) / (n - 1) for row in matrix]

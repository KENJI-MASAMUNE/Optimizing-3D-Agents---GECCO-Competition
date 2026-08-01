"""
Phenotypic (graph) dissimilarity measure. Owner: Ntsako Sithole

Matches Section 2.4 / 4.4.2: "A graph-matching measure comparing 3D structural graphs of
evolved bodies, accounting for vertex degree, neuron counts, geometric positions, and
unmatched parts." Used by DE-NSGA-II Variant 3 (Graph Crowding) and as the second component
of Variant 4 (Hybrid Crowding).

Cost safeguard (Section 5.1, Variant 3): true graph edit distance is exponential in the
worst case, which risks blowing the competition's 1-hour execution budget. This module
therefore:
  1. Builds a bounded-size structural graph per genotype (`genotype_to_graph`), capped at
     `MAX_NODES` nodes.
  2. Uses a cheap, feature-vector-based approximation (degree sequence, neuron counts,
     centroid position, node-count mismatch) rather than exact graph edit distance, wrapped
     in an LRU cache keyed on genotype string so repeated comparisons within a generation
     are free after the first.
  3. Falls back to a fast networkx-based approximation (`graph_edit_distance` with a
     timeout) only when explicitly requested via `exact=True`, since exact GED is only
     tractable for very small graphs.
"""
from __future__ import annotations

from functools import lru_cache

import networkx as nx

MAX_NODES = 20  # structural depth bound


def genotype_to_graph(genotype: str, encoding: str) -> nx.Graph:
    """Build a small structural graph from a genotype string. Nodes represent body parts
    (or f1 structural symbols); edges represent physical/neural connections. This is a
    lightweight stand-in for the real Framsticks phenotype graph (which would come from
    the simulator) -- see mock_framsticks.py docstring for why a mock exists at all.
    """
    g = nx.Graph()

    if encoding == "f1":
        symbols = genotype[:MAX_NODES]
        prev_node = None
        depth_stack = []
        for i, ch in enumerate(symbols):
            node_id = i
            is_neuron = ch in "NnGgTt"
            g.add_node(node_id, kind="neuron" if is_neuron else "part", symbol=ch,
                       pos=(i * 0.3, (i % 3) * 0.2, (i % 5) * 0.1))
            if ch == "L":  # branch open (simplified f1 semantics)
                if prev_node is not None:
                    g.add_edge(prev_node, node_id)
                depth_stack.append(prev_node)
            elif ch == "R" and depth_stack:  # branch close
                anchor = depth_stack.pop()
                if anchor is not None:
                    g.add_edge(anchor, node_id)
            else:
                if prev_node is not None:
                    g.add_edge(prev_node, node_id)
            prev_node = node_id

    elif encoding == "f0":
        tokens = genotype.split(" ")[:MAX_NODES]
        for i, tok in enumerate(tokens):
            is_neuron = "conn" in tok or (tok and tok[0] in "nN")
            x = (sum(ord(c) for c in tok[:3]) % 100) / 100.0 if tok else 0.0
            y = (sum(ord(c) for c in tok[3:6]) % 100) / 100.0 if len(tok) > 3 else 0.0
            z = (sum(ord(c) for c in tok[6:]) % 100) / 100.0 if len(tok) > 6 else 0.0
            g.add_node(i, kind="neuron" if is_neuron else "part", symbol=tok, pos=(x, y, z))
            if i > 0:
                g.add_edge(i - 1, i)
    else:
        raise ValueError(f"Unknown encoding '{encoding}'")

    return g


def _graph_features(g: nx.Graph) -> dict:
    n = g.number_of_nodes()
    degrees = sorted([d for _, d in g.degree()], reverse=True)
    neuron_count = sum(1 for _, data in g.nodes(data=True) if data.get("kind") == "neuron")
    positions = [data.get("pos", (0, 0, 0)) for _, data in g.nodes(data=True)]
    if positions:
        centroid = tuple(sum(p[i] for p in positions) / len(positions) for i in range(3))
    else:
        centroid = (0.0, 0.0, 0.0)
    return {
        "n_nodes": n,
        "degree_seq": degrees,
        "neuron_count": neuron_count,
        "centroid": centroid,
    }


@lru_cache(maxsize=4096)
def _cached_features(genotype: str, encoding: str) -> tuple:
    g = genotype_to_graph(genotype, encoding)
    f = _graph_features(g)
    # tuple-ify for hashability / caching
    return (f["n_nodes"], tuple(f["degree_seq"]), f["neuron_count"], f["centroid"])


def _degree_seq_distance(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Normalised distance between two degree sequences of possibly different length."""
    n = max(len(a), len(b))
    if n == 0:
        return 0.0
    a_pad = list(a) + [0] * (n - len(a))
    b_pad = list(b) + [0] * (n - len(b))
    diff = sum(abs(x - y) for x, y in zip(a_pad, b_pad))
    max_possible = sum(a_pad) + sum(b_pad) or 1
    return diff / max_possible


def graph_distance(genotype_a: str, genotype_b: str, encoding: str, exact: bool = False) -> float:
    """Bounded, cached approximate graph dissimilarity in [0, 1]. Combines: node-count
    mismatch, degree-sequence distance, neuron-count mismatch, and centroid (geometric
    position) distance -- the four factors named in Sections 2.4 / 4.4.2.
    """
    if genotype_a == genotype_b:
        return 0.0

    if exact:
        ga = genotype_to_graph(genotype_a, encoding)
        gb = genotype_to_graph(genotype_b, encoding)
        try:
            ged = nx.graph_edit_distance(ga, gb, timeout=0.25)
            if ged is None:
                exact = False  # fall through to approximation
            else:
                max_size = max(ga.number_of_nodes() + ga.number_of_edges(),
                                gb.number_of_nodes() + gb.number_of_edges(), 1)
                return min(ged / max_size, 1.0)
        except Exception:
            exact = False  # graph too large / timed out -> fall back below

    n_a, deg_a, neur_a, cent_a = _cached_features(genotype_a, encoding)
    n_b, deg_b, neur_b, cent_b = _cached_features(genotype_b, encoding)

    node_mismatch = abs(n_a - n_b) / max(n_a, n_b, 1)
    degree_dist = _degree_seq_distance(deg_a, deg_b)
    neuron_mismatch = abs(neur_a - neur_b) / max(neur_a, neur_b, 1)
    centroid_dist = min(
        sum((x - y) ** 2 for x, y in zip(cent_a, cent_b)) ** 0.5 / 1.732, 1.0
    )  # normalised by max possible Euclidean distance in [0,1]^3

    # equal-weighted combination of the four named structural factors
    return (node_mismatch + degree_dist + neuron_mismatch + centroid_dist) / 4.0


def pairwise_graph_dissimilarity(genotypes: list[str], encoding: str) -> list[list[float]]:
    n = len(genotypes)
    matrix = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            d = graph_distance(genotypes[i], genotypes[j], encoding)
            matrix[i][j] = d
            matrix[j][i] = d
    return matrix


def average_graph_dissimilarity(genotypes: list[str], encoding: str) -> list[float]:
    """Per-individual average graph dissimilarity to the population (DE-NSGA-II crowding
    distance formula, Section 2.4), using graph dissimilarity as D_ij."""
    n = len(genotypes)
    if n <= 1:
        return [0.0] * n
    matrix = pairwise_graph_dissimilarity(genotypes, encoding)
    return [sum(row) / (n - 1) for row in matrix]

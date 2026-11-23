"""Centrality and network influence metrics module for ViralGraph.

This module provides mathematically sound implementations of five canonical node centralities:
1. Degree Centrality (C_D): Local connectivity.
2. Shortest-Path Betweenness Centrality (C_B): Choke-point control and broker influence.
3. Closeness Centrality (C_C): Path efficiency and communication velocity.
4. Eigenvector Centrality (C_E): Prestige and spectral influence.
5. PageRank (PR): Random-walk stationary probability with damping factor d.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import networkx as nx
import numpy as np


@dataclass(frozen=True)
class CentralityProfile:
    """Immutable container holding all five centrality scores for all vertices in a graph."""

    # Node-aligned dictionaries {node_id: float}
    degree: dict[Any, float]
    betweenness: dict[Any, float]
    closeness: dict[Any, float]
    eigenvector: dict[Any, float]
    pagerank: dict[Any, float]

    # Node rankings: sorted lists of node IDs descending by score
    top_degree_nodes: list[Any]
    top_betweenness_nodes: list[Any]
    top_closeness_nodes: list[Any]
    top_eigenvector_nodes: list[Any]
    top_pagerank_nodes: list[Any]

    # Bridge-hub disparity metric: Betweenness-to-Degree ratio
    betweenness_degree_ratios: dict[Any, float]


def compute_degree_centrality(graph: nx.Graph) -> dict[Any, float]:
    """Compute normalized Degree Centrality: C_D(v) = deg(v) / (n - 1).

    Mathematical Formulation:
        Measures the fraction of possible neighbors vertex v is directly connected to.
        C_D(v) in [0.0, 1.0].
    """
    n = graph.number_of_nodes()
    if n <= 1:
        return {node: 0.0 for node in graph.nodes()}
    return {node: deg / (n - 1.0) for node, deg in graph.degree()}


def compute_betweenness_centrality(
    graph: nx.Graph,
    normalized: bool = True,
    k_samples: int | None = None,
    seed: int | None = None,
) -> dict[Any, float]:
    """Compute Shortest-Path Betweenness Centrality: C_B(v) = sum_{s != v != t} (sigma_st(v) / sigma_st).

    Mathematical Formulation:
        Let sigma_st be the total number of shortest paths between vertices s and t.
        Let sigma_st(v) be the number of those shortest paths that pass through vertex v.
        Normalized by 2 / ((n - 1)(n - 2)) for undirected graphs.

    Args:
        graph: Undirected graph.
        normalized: Whether to scale betweenness to [0.0, 1.0].
        k_samples: If specified, uses Brandes' approximation with k source nodes for O(k m) speed.
        seed: Deterministic seed for randomized k-sampling.
    """
    return nx.betweenness_centrality(
        graph, normalized=normalized, k=k_samples, seed=seed, weight=None
    )


def compute_closeness_centrality(graph: nx.Graph) -> dict[Any, float]:
    """Compute Closeness Centrality: C_C(v) = (n - 1) / sum_{u != v} d(v, u).

    Mathematical Formulation:
        Measures the reciprocal of the total shortest path distance from vertex v to all other vertices.
        Uses the Wasserman & Faust formula for disconnected networks:
            C_C(v) = (N_comp - 1) / (n - 1) * (N_comp - 1) / sum_{u in Comp, u != v} d(v, u)
    """
    return nx.closeness_centrality(graph, wf_improved=True)


def compute_eigenvector_centrality(
    graph: nx.Graph,
    max_iter: int = 1000,
    tol: float = 1e-6,
) -> dict[Any, float]:
    """Compute Eigenvector Centrality: A x = lambda x (Principal Eigenvector).

    Mathematical Formulation:
        x_v = (1 / lambda) * sum_{u in N(v)} x_u.
        By the Perron-Frobenius theorem, for a connected graph, the principal eigenvector
        associated with spectral radius mu_1 is non-negative and unique.
        Falls back to PageRank or uniform if power iteration fails to converge on disconnected components.
    """
    n = graph.number_of_nodes()
    if n == 0:
        return {}
    if n == 1:
        return {list(graph.nodes())[0]: 1.0}

    try:
        return nx.eigenvector_centrality(graph, max_iter=max_iter, tol=tol)
    except nx.PowerIterationFailedConvergence:
        # Fallback to NumPy exact eigensolver for robustness
        adj = nx.to_numpy_array(graph)
        evals, evecs = np.linalg.eigh(adj)
        # Principal eigenvector corresponds to largest eigenvalue
        principal_idx = np.argmax(evals)
        principal_vec = np.abs(evecs[:, principal_idx])
        norm = np.linalg.norm(principal_vec)
        if norm > 0:
            principal_vec /= norm
        nodes = list(graph.nodes())
        return {nodes[i]: float(principal_vec[i]) for i in range(n)}


def compute_pagerank(
    graph: nx.Graph,
    alpha: float = 0.85,
    max_iter: int = 500,
    tol: float = 1e-6,
) -> dict[Any, float]:
    """Compute PageRank stationary distribution with damping factor alpha: PR = (1 - alpha)/n * 1 + alpha * M * PR.

    Mathematical Formulation:
        Models a random surfer that transitions along outgoing edges with probability alpha (0.85),
        or teleports uniformly to any node in V with restart probability (1 - alpha).
    """
    return nx.pagerank(graph, alpha=alpha, max_iter=max_iter, tol=tol)


def compute_centrality_profile(
    graph: nx.Graph,
    seed: int | None = None,
) -> CentralityProfile:
    """Compute all five canonical centrality metrics and return an immutable CentralityProfile.

    Also computes the Betweenness-to-Degree ratio (BDR):
        BDR(v) = C_B(v) / (deg(v) + epsilon)
        A high BDR indicates a structural bridge/broker node (high path mediation despite low degree).
    """
    deg_c = compute_degree_centrality(graph)
    bet_c = compute_betweenness_centrality(graph, seed=seed)
    clo_c = compute_closeness_centrality(graph)
    eig_c = compute_eigenvector_centrality(graph)
    pag_c = compute_pagerank(graph)

    # Compute Betweenness-to-Degree Disparity Ratio
    bdr: dict[Any, float] = {}
    for node in graph.nodes():
        deg = graph.degree(node)
        # BDR = C_B / (deg / (n-1)) = betweenness / degree_centrality
        deg_norm = deg_c[node]
        if deg_norm > 0:
            bdr[node] = float(bet_c[node] / deg_norm)
        else:
            bdr[node] = 0.0

    # Top sorted rankings
    top_deg = sorted(deg_c.keys(), key=lambda x: deg_c[x], reverse=True)
    top_bet = sorted(bet_c.keys(), key=lambda x: bet_c[x], reverse=True)
    top_clo = sorted(clo_c.keys(), key=lambda x: clo_c[x], reverse=True)
    top_eig = sorted(eig_c.keys(), key=lambda x: eig_c[x], reverse=True)
    top_pag = sorted(pag_c.keys(), key=lambda x: pag_c[x], reverse=True)

    return CentralityProfile(
        degree=deg_c,
        betweenness=bet_c,
        closeness=clo_c,
        eigenvector=eig_c,
        pagerank=pag_c,
        top_degree_nodes=top_deg,
        top_betweenness_nodes=top_bet,
        top_closeness_nodes=top_clo,
        top_eigenvector_nodes=top_eig,
        top_pagerank_nodes=top_pag,
        betweenness_degree_ratios=bdr,
    )

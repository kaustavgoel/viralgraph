"""Synthetic Graph Generation Module for ViralGraph.

This module provides mathematically sound, reproducible synthetic network generators
grounded in discrete mathematics and complex network theory:
1. Barabási-Albert (Scale-Free networks via preferential attachment)
2. Watts-Strogatz (Small-World networks via lattice rewiring)
3. Stochastic Block Model (Clustered networks with community homophily)

All models generate undirected simple graphs G = (V, E) without self-loops or multi-edges.
"""

from __future__ import annotations

import networkx as nx
import numpy as np


def generate_barabasi_albert(
    n: int,
    m: int,
    seed: int | None = None,
) -> nx.Graph:
    """Generate a scale-free network using the Barabási-Albert preferential attachment model.

    Mathematical Formulation:
        Beginning with an initial connected core of m vertices, at each discrete step
        t = 1, 2, ..., n - m, a new vertex v is added with m edges. The probability
        that the new vertex connects to an existing vertex u is proportional to u's degree:
            Pi(u) = deg(u) / sum_{w in V} deg(w) = deg(u) / (2 * |E|)

        As n -> infinity, the degree distribution follows a power law:
            P(k) ~ k^(-gamma), where gamma = 3.

    Args:
        n: Total number of vertices in the final graph (must be an integer > m).
        m: Number of edges each incoming vertex attaches to existing vertices (m >= 1).
        seed: Optional integer seed for deterministic pseudo-random number generation.

    Returns:
        nx.Graph: An undirected simple graph with |V| = n and exactly |E| = m * (n - m) edges.

    Raises:
        TypeError: If n or m are not integers.
        ValueError: If m < 1 or n <= m.
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"Parameter 'n' must be an integer, got {type(n).__name__}.")
    if not isinstance(m, int) or isinstance(m, bool):
        raise TypeError(f"Parameter 'm' must be an integer, got {type(m).__name__}.")
    if m < 1:
        raise ValueError(f"Parameter 'm' must be >= 1, got {m}.")
    if n <= m:
        raise ValueError(f"Parameter 'n' ({n}) must be strictly greater than 'm' ({m}).")

    graph = nx.barabasi_albert_graph(n=n, m=m, seed=seed)
    graph.name = f"Barabasi-Albert(n={n}, m={m})"
    graph.graph["model"] = "barabasi_albert"
    graph.graph["n"] = n
    graph.graph["m"] = m
    graph.graph["seed"] = seed
    return graph


def generate_watts_strogatz(
    n: int,
    k: int,
    p: float,
    seed: int | None = None,
) -> nx.Graph:
    """Generate a small-world network using the Watts-Strogatz rewiring model.

    Mathematical Formulation:
        Starts from a 1-dimensional regular ring lattice of n vertices where each vertex
        is connected to its k nearest neighbors (k/2 on each side). For each edge (u, v),
        with independent probability p in [0, 1], the edge is rewired to a uniformly
        chosen vertex w in V \\ {u}, avoiding self-loops and multi-edges.

        Properties:
        - When p = 0: High clustering C ~ 3/4, long average path length L ~ n / (2k).
        - When 0 < p << 1: Small-world regime with high clustering and short path lengths.
        - When p = 1: Disordered random graph with low clustering C ~ k/n and short path lengths.

    Args:
        n: Number of vertices in the graph (must be integer > k).
        k: Degree of each vertex in the initial ring lattice (must be an even integer >= 2).
        p: Edge rewiring probability (must be a float in [0.0, 1.0]).
        seed: Optional integer seed for deterministic pseudo-random number generation.

    Returns:
        nx.Graph: An undirected simple graph with |V| = n and |E| = (n * k) // 2 edges.

    Raises:
        TypeError: If n or k are not integers, or p is not a float/int.
        ValueError: If k is odd, k < 2, n <= k, or p is outside [0.0, 1.0].
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"Parameter 'n' must be an integer, got {type(n).__name__}.")
    if not isinstance(k, int) or isinstance(k, bool):
        raise TypeError(f"Parameter 'k' must be an integer, got {type(k).__name__}.")
    if not isinstance(p, (float, int)) or isinstance(p, bool):
        raise TypeError(f"Parameter 'p' must be a float in [0.0, 1.0], got {type(p).__name__}.")

    p = float(p)

    if k < 2:
        raise ValueError(f"Parameter 'k' must be >= 2, got {k}.")
    if k % 2 != 0:
        raise ValueError(f"Parameter 'k' must be an even integer, got {k}.")
    if n <= k:
        raise ValueError(f"Parameter 'n' ({n}) must be strictly greater than 'k' ({k}).")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Parameter 'p' must be in range [0.0, 1.0], got {p}.")

    graph = nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)
    graph.name = f"Watts-Strogatz(n={n}, k={k}, p={p:.3f})"
    graph.graph["model"] = "watts_strogatz"
    graph.graph["n"] = n
    graph.graph["k"] = k
    graph.graph["p"] = p
    graph.graph["seed"] = seed
    return graph


def generate_stochastic_block_model(
    community_sizes: list[int] | tuple[int, ...],
    probability_matrix: list[list[float]] | np.ndarray,
    seed: int | None = None,
) -> tuple[nx.Graph, dict[int, int]]:
    """Generate a clustered community network using the Stochastic Block Model (SBM).

    Mathematical Formulation:
        The vertex set V is partitioned into K mutually disjoint blocks:
            V = V_1 U V_2 U ... U V_K, where |V_r| = n_r and sum n_r = n.
        A symmetric matrix P in [0, 1]^(K x K) governs connection probabilities:
            forall u in V_r, v in V_s (u != v): P((u, v) in E) = P_rs.

        For assortative community structures (homophily):
            P_rr >> P_rs for all r != s.

    Args:
        community_sizes: Sequence of positive integers representing the size of each community block.
        probability_matrix: K x K symmetric matrix of connection probabilities where entry (r, s)
            is the independent Bernoulli probability of an edge between vertices in block r and block s.
        seed: Optional integer seed for deterministic pseudo-random number generation.

    Returns:
        tuple[nx.Graph, dict[int, int]]:
            - nx.Graph: Generated undirected simple graph with node attribute 'block' indicating community.
            - dict[int, int]: Mapping from node_id -> community_index (0 to K - 1).

    Raises:
        TypeError: If community_sizes or probability_matrix have invalid types.
        ValueError: If community_sizes is empty, contains non-positive entries,
            or probability_matrix is not a K x K symmetric matrix with entries in [0, 1].
    """
    if not isinstance(community_sizes, (list, tuple)):
        raise TypeError("Parameter 'community_sizes' must be a list or tuple of integers.")
    if len(community_sizes) == 0:
        raise ValueError("Parameter 'community_sizes' cannot be empty.")
    for idx, size in enumerate(community_sizes):
        if not isinstance(size, int) or isinstance(size, bool):
            raise TypeError(f"Community size at index {idx} must be an integer, got {type(size).__name__}.")
        if size <= 0:
            raise ValueError(f"Community size at index {idx} must be > 0, got {size}.")

    k_blocks = len(community_sizes)

    # Convert and validate probability matrix
    prob_mat = np.asarray(probability_matrix, dtype=float)
    if prob_mat.shape != (k_blocks, k_blocks):
        raise ValueError(
            f"Probability matrix shape must be ({k_blocks}, {k_blocks}), got {prob_mat.shape}."
        )
    if not np.allclose(prob_mat, prob_mat.T):
        raise ValueError("Probability matrix P must be symmetric: P_rs = P_sr.")
    if np.any(prob_mat < 0.0) or np.any(prob_mat > 1.0):
        raise ValueError("All entries in probability matrix must be in [0.0, 1.0].")

    # Generate graph using NetworkX SBM
    graph = nx.stochastic_block_model(
        sizes=list(community_sizes),
        p=prob_mat.tolist(),
        seed=seed,
    )

    # Build node -> block mapping dictionary and attach metadata
    node_community_map: dict[int, int] = {}
    for node, data in graph.nodes(data=True):
        block_idx = int(data.get("block", 0))
        node_community_map[node] = block_idx

    total_n = sum(community_sizes)
    graph.name = f"StochasticBlockModel(K={k_blocks}, N={total_n})"
    graph.graph["model"] = "stochastic_block_model"
    graph.graph["community_sizes"] = list(community_sizes)
    graph.graph["probability_matrix"] = prob_mat.tolist()
    graph.graph["num_communities"] = k_blocks
    graph.graph["seed"] = seed

    return graph, node_community_map


def generate_erdos_renyi(
    n: int,
    p: float,
    seed: int | None = None,
) -> nx.Graph:
    """Generate a random network using the Erdős–Rényi G(n, p) model.

    Mathematical Formulation:
        Given n vertices, each of the n*(n-1)/2 possible unordered pairs {u, v}
        forms an independent edge with probability p.
        Expected edge count: E[|E|] = p * n * (n - 1) / 2.
        Degree distribution: Binomial B(n - 1, p), asymptotically Poisson(lambda)
        where lambda = (n - 1) * p as n -> infinity.

    Args:
        n: Total number of vertices (must be integer >= 1).
        p: Edge formation probability in [0.0, 1.0].
        seed: Optional integer seed for pseudo-random number generator.

    Returns:
        nx.Graph: An undirected simple graph.

    Raises:
        TypeError: If n is not an integer or p is not a float.
        ValueError: If n < 1 or p not in [0.0, 1.0].
    """
    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError(f"Parameter 'n' must be an integer, got {type(n).__name__}.")
    if not isinstance(p, (float, int)) or isinstance(p, bool):
        raise TypeError(f"Parameter 'p' must be a float, got {type(p).__name__}.")
    p = float(p)
    if n < 1:
        raise ValueError(f"Parameter 'n' must be >= 1, got {n}.")
    if not (0.0 <= p <= 1.0):
        raise ValueError(f"Parameter 'p' must be in range [0.0, 1.0], got {p}.")

    graph = nx.erdos_renyi_graph(n=n, p=p, seed=seed)
    graph.name = f"Erdos-Renyi(n={n}, p={p:.3f})"
    graph.graph["model"] = "erdos_renyi"
    graph.graph["n"] = n
    graph.graph["p"] = p
    graph.graph["seed"] = seed
    return graph


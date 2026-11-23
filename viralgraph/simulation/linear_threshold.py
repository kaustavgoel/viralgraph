"""Linear Threshold Model (LTM) simulation engine for ViralGraph.

Mathematical Formulation:
    Receiver-centric discrete-time collective adoption process (Complex Contagion).
    Given graph G = (V, E) and seed set S_0:
    1. Each node v in V is assigned an adoption threshold theta_v ~ Uniform(0, 1).
    2. Directed edges (u, v) have influence weights b_uv >= 0 such that sum_u b_uv <= 1.
       (Default: b_uv = 1 / deg(v)).
    3. At discrete step t >= 1, an inactive node v activates if the cumulative weight
       of its active neighbors exceeds theta_v:
           sum_{u in N(v) and Active(u)} b_uv >= theta_v
    4. Active nodes remain active forever.
    5. The process terminates when no newly activated nodes cross their threshold.
"""

from __future__ import annotations

from typing import Any
import networkx as nx
import numpy as np

from viralgraph.simulation.independent_cascade import CascadeResult


def simulate_linear_threshold(
    graph: nx.Graph,
    seed_nodes: list[Any] | set[Any],
    thresholds: dict[Any, float] | None = None,
    edge_weights: dict[tuple[Any, Any], float] | None = None,
    max_steps: int = 100,
    rng: np.random.Generator | None = None,
) -> CascadeResult:
    """Simulate a single realization of the Linear Threshold Model (LTM).

    Args:
        graph: Undirected or directed NetworkX graph.
        seed_nodes: Initial active seed set S_0.
        thresholds: Optional node -> theta_v in [0, 1]. If None, sampled i.i.d. U(0, 1).
        edge_weights: Optional edge influence weights b_uv. If None, b_uv = 1 / deg(v).
        max_steps: Maximum step horizon.
        rng: NumPy random generator.

    Returns:
        CascadeResult: Trajectory, reach timeline, and transmission tree.
    """
    if rng is None:
        rng = np.random.default_rng()

    seeds = set(seed_nodes)
    if not seeds:
        raise ValueError("seed_nodes cannot be empty.")
    missing = seeds - set(graph.nodes())
    if missing:
        raise ValueError(f"Seed nodes not present in graph: {missing}")

    # Generate thresholds theta_v ~ U(0, 1) if not provided
    if thresholds is None:
        theta: dict[Any, float] = {node: float(rng.uniform(0.0, 1.0)) for node in graph.nodes()}
    else:
        theta = dict(thresholds)

    # Compute influence weights b_uv: default b_uv = 1 / deg(v)
    weights: dict[tuple[Any, Any], float] = {}
    for v in graph.nodes():
        neighbors = list(graph.neighbors(v))
        deg_v = len(neighbors)
        if deg_v > 0:
            default_w = 1.0 / deg_v
            for u in neighbors:
                if edge_weights is not None:
                    w = edge_weights.get((u, v), edge_weights.get((v, u), default_w))
                else:
                    w = default_w
                weights[(u, v)] = w

    active_nodes: set[Any] = set(seeds)
    activation_history: list[set[Any]] = [set(seeds)]
    cumulative_timeline: list[int] = [len(active_nodes)]

    transmission_tree = nx.DiGraph()
    transmission_tree.add_nodes_from(seeds)

    for step in range(1, max_steps + 1):
        newly_activated: set[Any] = set()

        # Check all currently inactive nodes
        inactive_nodes = set(graph.nodes()) - active_nodes
        for v in inactive_nodes:
            # Sum influence of active neighbors
            active_nbrs = [u for u in graph.neighbors(v) if u in active_nodes]
            cum_influence = sum(weights.get((u, v), 0.0) for u in active_nbrs)

            if cum_influence >= theta[v]:
                newly_activated.add(v)
                # In transmission tree, record edge from the active neighbor with highest weight
                if active_nbrs:
                    primary_influencer = max(active_nbrs, key=lambda u: weights.get((u, v), 0.0))
                    transmission_tree.add_edge(primary_influencer, v, step=step)

        if not newly_activated:
            break

        active_nodes.update(newly_activated)
        activation_history.append(newly_activated)
        cumulative_timeline.append(len(active_nodes))

    return CascadeResult(
        seed_nodes=sorted(list(seeds)),
        total_reach=len(active_nodes),
        activation_history=activation_history,
        cumulative_reach_timeline=cumulative_timeline,
        steps_to_termination=len(activation_history) - 1,
        transmission_tree=transmission_tree,
    )

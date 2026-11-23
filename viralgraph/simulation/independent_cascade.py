"""Independent Cascade Model (ICM) simulation engine for ViralGraph.

Mathematical Formulation:
    Sender-centric discrete-time stochastic diffusion process.
    Given graph G = (V, E) and seed set S_0:
    1. At t=0, A_0 = S_0.
    2. At each time step t >= 1, each node u in A_{t-1} attempts to activate each inactive
       neighbor v in N(u) with independent probability p_uv in [0, 1].
    3. Each node u gets exactly ONE attempt to activate neighbor v.
    4. Active nodes remain active monotonically forever.
    5. The process terminates when A_t = empty (no new infections).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import networkx as nx
import numpy as np


@dataclass(frozen=True)
class CascadeResult:
    """Immutable record of a single discrete-time cascade trajectory."""

    seed_nodes: list[Any]
    total_reach: int
    activation_history: list[set[Any]]  # A_0, A_1, ..., A_T
    cumulative_reach_timeline: list[int]  # [|A_<=0|, |A_<=1|, ..., |A_<=T|]
    steps_to_termination: int
    # Transmission tree: directed graph recording who activated whom
    transmission_tree: nx.DiGraph


def simulate_independent_cascade(
    graph: nx.Graph,
    seed_nodes: list[Any] | set[Any],
    p: float = 0.1,
    edge_weights: dict[tuple[Any, Any], float] | None = None,
    max_steps: int = 100,
    rng: np.random.Generator | None = None,
) -> CascadeResult:
    """Simulate a single realization of the Independent Cascade Model (ICM).

    Args:
        graph: Undirected or directed NetworkX graph.
        seed_nodes: Set or sequence of initial activated vertices (S_0).
        p: Baseline transmission probability (used if edge_weights is None).
        edge_weights: Optional mapping (u, v) -> transmission probability p_uv.
        max_steps: Maximum discrete time horizon to prevent infinite loops.
        rng: NumPy random generator for deterministic reproducibility.

    Returns:
        CascadeResult: Trajectory, reach timeline, and transmission tree.

    Raises:
        ValueError: If seed_nodes is empty or contains nodes not in graph.
    """
    if rng is None:
        rng = np.random.default_rng()

    seeds = set(seed_nodes)
    if not seeds:
        raise ValueError("seed_nodes cannot be empty.")
    missing = seeds - set(graph.nodes())
    if missing:
        raise ValueError(f"Seed nodes not present in graph: {missing}")

    # Set of all currently active nodes
    active_nodes: set[Any] = set(seeds)
    # Nodes newly activated at time t-1 that have attempts available at time t
    current_front: set[Any] = set(seeds)

    activation_history: list[set[Any]] = [set(seeds)]
    cumulative_timeline: list[int] = [len(active_nodes)]

    # Build directed transmission DAG
    transmission_tree = nx.DiGraph()
    transmission_tree.add_nodes_from(seeds)

    for step in range(1, max_steps + 1):
        if not current_front:
            break

        new_front: set[Any] = set()

        for u in current_front:
            # For undirected graph, neighbors are out-neighbors
            neighbors = list(graph.neighbors(u))
            for v in neighbors:
                if v not in active_nodes:
                    # Determine transmission probability p_uv
                    if edge_weights is not None:
                        prob = edge_weights.get((u, v), edge_weights.get((v, u), p))
                    else:
                        prob = p

                    # Independent Bernoulli trial
                    if rng.random() < prob:
                        new_front.add(v)
                        active_nodes.add(v)
                        transmission_tree.add_edge(u, v, step=step)

        if not new_front:
            break

        activation_history.append(new_front)
        cumulative_timeline.append(len(active_nodes))
        current_front = new_front

    return CascadeResult(
        seed_nodes=sorted(list(seeds)),
        total_reach=len(active_nodes),
        activation_history=activation_history,
        cumulative_reach_timeline=cumulative_timeline,
        steps_to_termination=len(activation_history) - 1,
        transmission_tree=transmission_tree,
    )

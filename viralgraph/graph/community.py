"""Community structure, modularity, and bridge-node detection module for ViralGraph.

This module provides discrete topological algorithms for community partitioning,
bridge-edge identification, and articulation point discovery:
1. Modularity-maximizing Community Partitioning (Louvain / Greedy Modularity).
2. Bridge-Edge Detection (Cut-edges whose removal increases component count).
3. Articulation Point Detection (Cut-vertices whose removal disconnects the graph).
4. Boundary Edge and Inter-Community Conductance analysis.
5. Bridging Centrality index.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import networkx as nx
import numpy as np


@dataclass(frozen=True)
class CommunityStructure:
    """Immutable profile holding the community partitions and boundary metrics of a graph."""

    # Node -> Community index mapping {node_id: int}
    node_to_community: dict[Any, int]
    # Community index -> list of nodes {int: [node_id, ...]}
    community_to_nodes: dict[int, list[Any]]
    # Total count of detected communities
    num_communities: int
    # Newman's modularity score Q in [-0.5, 1.0]
    modularity: float

    # Bridge and bottleneck topological sets
    bridge_edges: list[tuple[Any, Any]]
    articulation_points: list[Any]
    inter_community_edges: list[tuple[Any, Any]]


def detect_communities(
    graph: nx.Graph,
    seed: int | None = None,
) -> CommunityStructure:
    """Detect communities using modularity maximization (Louvain algorithm).

    Newman's Modularity Q:
        Q = 1 / (2m) * sum_{u, v} [ A_uv - (deg(u)*deg(v)) / (2m) ] * delta(c_u, c_v)
        Q > 0.3 typically indicates strong community structure.

    Args:
        graph: Undirected simple graph.
        seed: Deterministic random seed for Louvain heuristic partitioning.

    Returns:
        CommunityStructure containing partition maps, modularity Q, bridges, and cut-vertices.
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    if n == 0 or m == 0:
        return CommunityStructure(
            node_to_community={},
            community_to_nodes={},
            num_communities=0,
            modularity=0.0,
            bridge_edges=[],
            articulation_points=[],
            inter_community_edges=[],
        )

    # Detect community sets using Louvain algorithm
    try:
        communities = nx.community.louvain_communities(graph, seed=seed)
    except Exception:
        # Fallback to greedy modularity communities if Louvain has issues
        communities = list(nx.community.greedy_modularity_communities(graph))

    # Build node -> comm and comm -> nodes maps
    node_to_comm: dict[Any, int] = {}
    comm_to_nodes: dict[int, list[Any]] = {}

    for comm_idx, node_set in enumerate(communities):
        comm_to_nodes[comm_idx] = sorted(list(node_set))
        for node in node_set:
            node_to_comm[node] = comm_idx

    # Modularity Q
    modularity = float(nx.community.modularity(graph, communities))

    # Bridges: edges whose removal disconnects the component (Tarjan's algorithm O(V + E))
    bridges = list(nx.bridges(graph))

    # Articulation points: vertices whose removal disconnects the component (O(V + E))
    articulation_pts = list(nx.articulation_points(graph))

    # Inter-community edges (edges crossing community boundaries)
    inter_edges: list[tuple[Any, Any]] = []
    for u, v in graph.edges():
        if node_to_comm.get(u) != node_to_comm.get(v):
            inter_edges.append((u, v))

    return CommunityStructure(
        node_to_community=node_to_comm,
        community_to_nodes=comm_to_nodes,
        num_communities=len(communities),
        modularity=modularity,
        bridge_edges=bridges,
        articulation_points=articulation_pts,
        inter_community_edges=inter_edges,
    )


def compute_bridging_centrality(
    graph: nx.Graph,
    betweenness_dict: dict[Any, float] | None = None,
) -> dict[Any, float]:
    """Compute the Bridging Centrality index for each vertex.

    Mathematical Formulation (Hwang et al., 2008):
        Bridging Centrality combines betweenness (global path routing) with the Bridging Coefficient:
            Bridging(v) = C_B(v) * BridgingCoefficient(v)

        where BridgingCoefficient(v) = (1 / deg(v)) * sum_{w in N(v)} [ (1 / deg(w)) / sum_{z in N(v)} (1 / deg(z)) ]
        A high bridging index identifies nodes that connect two or more dense, separate communities.
    """
    n = graph.number_of_nodes()
    if n <= 2:
        return {node: 0.0 for node in graph.nodes()}

    if betweenness_dict is None:
        betweenness_dict = nx.betweenness_centrality(graph)

    bridging_centrality: dict[Any, float] = {}

    for v in graph.nodes():
        deg_v = graph.degree(v)
        if deg_v <= 1:
            bridging_centrality[v] = 0.0
            continue

        neighbors = list(graph.neighbors(v))
        inv_degrees = [1.0 / graph.degree(w) for w in neighbors if graph.degree(w) > 0]
        sum_inv = sum(inv_degrees)

        if sum_inv == 0:
            bc_coeff = 0.0
        else:
            bc_coeff = (1.0 / deg_v) * sum((1.0 / graph.degree(w)) / sum_inv for w in neighbors)

        bridging_centrality[v] = float(betweenness_dict[v] * bc_coeff)

    return bridging_centrality

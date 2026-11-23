"""Comparative topological analysis engine for ViralGraph.

Runs systematic cross-network comparisons across:
- Barabási–Albert (Scale-Free)
- Watts–Strogatz (Small-World)
- Stochastic Block Model (Clustered Communities)

Computes full metric vectors spanning:
1. Degree sequence statistics & power-law scaling.
2. Distance & path metrics.
3. Clustering & transitivity.
4. Centrality distributions and Bridge-to-Hub ratios.
5. Community structure, modularity Q, and cut-edges.
6. Spectral radius and algebraic connectivity.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any
import networkx as nx
import numpy as np

from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)
from viralgraph.graph.centrality import compute_centrality_profile
from viralgraph.graph.community import detect_communities
from viralgraph.graph.degree_analysis import analyze_degree_distribution
from viralgraph.graph.representation import GraphRepresentation


@dataclass(frozen=True)
class NetworkComparisonRecord:
    """Consolidated record of all discrete mathematical invariants for a network."""

    model_name: str
    nodes: int
    edges: int
    density: float

    # Degree stats
    mean_degree: float
    variance_degree: float
    std_degree: float
    max_degree: int
    min_degree: int
    skewness: float
    assortativity: float
    estimated_power_law_gamma: float

    # Clustering & distances
    avg_clustering: float
    transitivity: float
    avg_shortest_path: float
    diameter: int

    # Spectral properties
    spectral_radius: float
    algebraic_connectivity: float

    # Communities & Bridges
    num_communities: int
    modularity_q: float
    num_bridge_edges: int
    num_articulation_points: int
    num_inter_community_edges: int
    max_betweenness_degree_ratio: float


def analyze_single_network(
    name: str,
    graph: nx.Graph,
    seed: int = 42,
) -> NetworkComparisonRecord:
    """Compute the complete topological and discrete math profile for a single graph."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    density = float(nx.density(graph))

    # 1. Degree analysis
    deg_prof = analyze_degree_distribution(graph)

    # 2. Centrality analysis
    cent_prof = compute_centrality_profile(graph, seed=seed)
    max_bdr = float(max(cent_prof.betweenness_degree_ratios.values())) if cent_prof.betweenness_degree_ratios else 0.0

    # 3. Community and bridges
    comm_struct = detect_communities(graph, seed=seed)

    # 4. Matrix & Spectral analysis
    rep = GraphRepresentation(graph)
    spec_prof = rep.compute_spectral_profile()

    # 5. Clustering and distance
    avg_clust = float(nx.average_clustering(graph))
    transitivity = float(nx.transitivity(graph))

    # Path length and diameter on Giant Component
    components = list(nx.connected_components(graph))
    giant = graph.subgraph(max(components, key=len))
    avg_path = float(nx.average_shortest_path_length(giant))
    diameter = int(nx.diameter(giant))

    return NetworkComparisonRecord(
        model_name=name,
        nodes=n,
        edges=m,
        density=density,
        mean_degree=deg_prof.mean_degree,
        variance_degree=deg_prof.variance_degree,
        std_degree=deg_prof.std_degree,
        max_degree=deg_prof.max_degree,
        min_degree=deg_prof.min_degree,
        skewness=deg_prof.skewness,
        assortativity=deg_prof.assortativity,
        estimated_power_law_gamma=deg_prof.estimated_power_law_gamma,
        avg_clustering=avg_clust,
        transitivity=transitivity,
        avg_shortest_path=avg_path,
        diameter=diameter,
        spectral_radius=spec_prof.spectral_radius,
        algebraic_connectivity=spec_prof.algebraic_connectivity,
        num_communities=comm_struct.num_communities,
        modularity_q=comm_struct.modularity,
        num_bridge_edges=len(comm_struct.bridge_edges),
        num_articulation_points=len(comm_struct.articulation_points),
        num_inter_community_edges=len(comm_struct.inter_community_edges),
        max_betweenness_degree_ratio=max_bdr,
    )


def run_benchmark_comparison(
    n_nodes: int = 300,
    seed: int = 42,
) -> list[NetworkComparisonRecord]:
    """Run standardized benchmark comparison across BA, WS, and SBM models."""
    ba = generate_barabasi_albert(n=n_nodes, m=3, seed=seed)
    ws = generate_watts_strogatz(n=n_nodes, k=6, p=0.08, seed=seed)

    # 3 equal blocks of n_nodes / 3
    b_size = n_nodes // 3
    sbm_sizes = [b_size, b_size, n_nodes - 2 * b_size]
    sbm_prob = [
        [0.080, 0.004, 0.002],
        [0.004, 0.070, 0.004],
        [0.002, 0.004, 0.060],
    ]
    sbm, _ = generate_stochastic_block_model(sbm_sizes, sbm_prob, seed=seed)

    records = [
        analyze_single_network("Barabási–Albert (Scale-Free)", ba, seed=seed),
        analyze_single_network("Watts–Strogatz (Small-World)", ws, seed=seed),
        analyze_single_network("Stochastic Block Model (Communities)", sbm, seed=seed),
    ]
    return records

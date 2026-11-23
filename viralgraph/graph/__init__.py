"""Discrete mathematics and graph theory engine.

Provides graph representations, matrix algebra, spectral properties,
centrality metrics, community detection, bridge analysis, and degree statistics.
"""

from viralgraph.graph.centrality import (
    CentralityProfile,
    compute_betweenness_centrality,
    compute_centrality_profile,
    compute_closeness_centrality,
    compute_degree_centrality,
    compute_eigenvector_centrality,
    compute_pagerank,
)
from viralgraph.graph.community import (
    CommunityStructure,
    compute_bridging_centrality,
    detect_communities,
)
from viralgraph.graph.degree_analysis import (
    DegreeProfile,
    analyze_degree_distribution,
)
from viralgraph.graph.representation import (
    GraphRepresentation,
    SpectralProfile,
    is_graphical_havel_hakimi,
)

__all__ = [
    "GraphRepresentation",
    "SpectralProfile",
    "is_graphical_havel_hakimi",
    "CentralityProfile",
    "compute_centrality_profile",
    "compute_degree_centrality",
    "compute_betweenness_centrality",
    "compute_closeness_centrality",
    "compute_eigenvector_centrality",
    "compute_pagerank",
    "CommunityStructure",
    "detect_communities",
    "compute_bridging_centrality",
    "DegreeProfile",
    "analyze_degree_distribution",
]

"""What-If Analysis and Structural Intervention Engine for ViralGraph.

Simulates network interventions to study resilience, containment, and misinformation mitigation:
1. Targeted Hub Inoculation (Degree-based node removal).
2. Targeted Bridge Inoculation (Betweenness/Bridging-based node removal).
3. Random Inoculation (Uniform baseline node removal).
4. Community Quarantine (Severing inter-community cut-edges).
5. Before-and-After Cascade Reach & Spectral Connectivity Evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import networkx as nx
import numpy as np

from viralgraph.graph.centrality import compute_betweenness_centrality, compute_degree_centrality
from viralgraph.graph.community import detect_communities
from viralgraph.graph.representation import GraphRepresentation
from viralgraph.simulation.monte_carlo import run_monte_carlo_simulation


@dataclass(frozen=True)
class InterventionComparison:
    """Pre- and Post-intervention comparative assessment."""

    intervention_type: str
    num_nodes_removed: int
    num_edges_removed: int

    # Structural impact
    baseline_gcc_ratio: float
    intervened_gcc_ratio: float
    baseline_algebraic_connectivity: float
    intervened_algebraic_connectivity: float

    # Cascade containment impact
    baseline_expected_reach: float
    intervened_expected_reach: float
    reach_reduction_percentage: float
    containment_efficiency: float  # Absolute reach reduction per immunized node


def apply_node_immunization(
    graph: nx.Graph,
    strategy: str = "targeted_hub",
    budget: int = 5,
    seed: int = 42,
) -> tuple[nx.Graph, list[Any]]:
    """Create an immunized copy of the graph by removing high-influence or random nodes.

    Args:
        graph: Undirected graph.
        strategy: "targeted_hub", "targeted_bridge", or "random".
        budget: Number of nodes to immunize (remove from transmission).
        seed: Random seed for tie-breaking or random selection.

    Returns:
        tuple[nx.Graph, list[Any]]: Perturbed graph and list of immunized nodes.
    """
    g_intervened = graph.copy()
    budget = min(budget, graph.number_of_nodes())

    if strategy == "targeted_hub":
        deg_dict = compute_degree_centrality(graph)
        immunized_nodes = sorted(deg_dict.keys(), key=lambda n: deg_dict[n], reverse=True)[:budget]

    elif strategy == "targeted_bridge":
        bet_dict = compute_betweenness_centrality(graph, seed=seed)
        immunized_nodes = sorted(bet_dict.keys(), key=lambda n: bet_dict[n], reverse=True)[:budget]

    elif strategy == "random":
        rng = np.random.default_rng(seed)
        nodes = list(graph.nodes())
        immunized_nodes = list(rng.choice(nodes, size=budget, replace=False))

    else:
        raise ValueError(f"Unknown immunization strategy: '{strategy}'.")

    g_intervened.remove_nodes_from(immunized_nodes)
    return g_intervened, immunized_nodes


def apply_community_quarantine(
    graph: nx.Graph,
    quarantined_community_idx: int,
    node_to_community: dict[Any, int],
) -> tuple[nx.Graph, list[tuple[Any, Any]]]:
    """Sever all boundary edges connecting the quarantined community to the exterior graph."""
    g_intervened = graph.copy()
    quarantined_nodes = {n for n, comm in node_to_community.items() if comm == quarantined_community_idx}

    severed_edges: list[tuple[Any, Any]] = []
    for u, v in graph.edges():
        u_in = u in quarantined_nodes
        v_in = v in quarantined_nodes
        if (u_in and not v_in) or (v_in and not u_in):
            severed_edges.append((u, v))

    g_intervened.remove_edges_from(severed_edges)
    return g_intervened, severed_edges


def evaluate_intervention_impact(
    graph: nx.Graph,
    intervention_type: str = "targeted_hub",
    budget: int = 5,
    p_transmission: float = 0.12,
    mc_simulations: int = 100,
    seed: int = 42,
) -> InterventionComparison:
    """Evaluate structural and cascade containment impact of an intervention."""
    n_orig = graph.number_of_nodes()
    m_orig = graph.number_of_edges()

    # Pre-intervention spectral & structural baseline
    rep_base = GraphRepresentation(graph)
    base_spec = rep_base.compute_spectral_profile()
    base_gcc = len(max(nx.connected_components(graph), key=len)) / float(n_orig)

    # Pre-intervention cascade baseline (from top degree node not among immunized nodes)
    g_intervened, removed_nodes = apply_node_immunization(
        graph, strategy=intervention_type, budget=budget, seed=seed
    )

    # Choose seed node that exists in both baseline and intervened graphs
    remaining_nodes = list(g_intervened.nodes())
    if not remaining_nodes:
        raise ValueError("Budget equals or exceeds total network order.")

    # Select common seed: highest degree node among remaining nodes
    deg_rem = dict(g_intervened.degree())
    common_seed = max(deg_rem.keys(), key=lambda n: deg_rem[n])

    # Run baseline Monte Carlo
    base_mc = run_monte_carlo_simulation(
        graph=graph,
        seed_nodes=[common_seed],
        model="ICM",
        p=p_transmission,
        num_simulations=mc_simulations,
        seed=seed,
    )

    # Post-intervention spectral & structural
    n_post = g_intervened.number_of_nodes()
    m_post = g_intervened.number_of_edges()
    rep_post = GraphRepresentation(g_intervened)
    post_spec = rep_post.compute_spectral_profile()
    post_gcc = (
        len(max(nx.connected_components(g_intervened), key=len)) / float(n_post)
        if n_post > 0
        else 0.0
    )

    # Run post-intervention Monte Carlo
    post_mc = run_monte_carlo_simulation(
        graph=g_intervened,
        seed_nodes=[common_seed],
        model="ICM",
        p=p_transmission,
        num_simulations=mc_simulations,
        seed=seed,
    )

    # Calculate containment metrics
    reduction = max(0.0, base_mc.expected_reach - post_mc.expected_reach)
    pct_red = (reduction / base_mc.expected_reach) * 100.0 if base_mc.expected_reach > 0 else 0.0
    eff = reduction / float(budget) if budget > 0 else 0.0

    return InterventionComparison(
        intervention_type=intervention_type,
        num_nodes_removed=len(removed_nodes),
        num_edges_removed=m_orig - m_post,
        baseline_gcc_ratio=base_gcc,
        intervened_gcc_ratio=post_gcc,
        baseline_algebraic_connectivity=base_spec.algebraic_connectivity,
        intervened_algebraic_connectivity=post_spec.algebraic_connectivity,
        baseline_expected_reach=base_mc.expected_reach,
        intervened_expected_reach=post_mc.expected_reach,
        reach_reduction_percentage=pct_red,
        containment_efficiency=eff,
    )

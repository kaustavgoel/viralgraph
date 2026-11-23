"""Unit tests for the What-If intervention and network immunization engine."""

import networkx as nx
import pytest

from viralgraph.data.synthetic_generator import generate_barabasi_albert, generate_stochastic_block_model
from viralgraph.simulation.intervention_engine import (
    apply_community_quarantine,
    apply_node_immunization,
    evaluate_intervention_impact,
)


def test_node_immunization_reduction():
    """Verify that node immunization strictly reduces vertex and edge counts."""
    g = generate_barabasi_albert(n=80, m=3, seed=42)
    budget = 4

    g_hub, removed_hubs = apply_node_immunization(g, strategy="targeted_hub", budget=budget)
    assert g_hub.number_of_nodes() == 80 - budget
    assert len(removed_hubs) == budget
    assert g_hub.number_of_edges() < g.number_of_edges()


def test_community_quarantine():
    """Verify that community quarantine severs inter-community boundary edges."""
    sizes = [40, 40]
    prob_mat = [[0.15, 0.02], [0.02, 0.15]]
    g, node_map = generate_stochastic_block_model(sizes, prob_mat, seed=42)

    g_quarantined, severed = apply_community_quarantine(g, quarantined_community_idx=0, node_to_community=node_map)

    # After full quarantine of block 0, there must be NO edges between block 0 and block 1
    for u, v in g_quarantined.edges():
        assert not (node_map[u] == 0 and node_map[v] == 1)
        assert not (node_map[u] == 1 and node_map[v] == 0)

    assert len(severed) > 0


def test_intervention_impact_evaluation():
    """Evaluate that targeted hub immunization reduces expected reach on BA network."""
    g = generate_barabasi_albert(n=100, m=3, seed=42)
    res = evaluate_intervention_impact(
        graph=g,
        intervention_type="targeted_hub",
        budget=3,
        p_transmission=0.15,
        mc_simulations=30,
        seed=42,
    )
    assert res.num_nodes_removed == 3
    assert res.baseline_expected_reach >= res.intervened_expected_reach

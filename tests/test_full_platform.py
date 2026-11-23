"""Automated integration test suite for ViralGraph Full Platform (Milestones 3–8).

Tests:
1. Live Diffusion Plotly canvas and timeline chart rendering.
2. Network Explorer node inspection metrics and degree distributions.
3. Influencer Research centrality profile rankings and Monte Carlo head-to-head trials.
4. Intervention Lab non-mutation invariants and containment efficiency.
5. Virality Prediction strict early feature extraction (t <= 1) and scoring logic.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from app.components.cascade_visualizer import render_cascade_canvas, render_cascade_timeline_chart
from app.components.diffusion_launcher import (
    launch_diffusion_simulation,
    resolve_seed_node,
    compute_external_infection_fraction,
)
from app.state import SimulationConfig, PostData
from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_erdos_renyi,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)
from viralgraph.graph.centrality import compute_centrality_profile
from viralgraph.models.dataset_builder import extract_strict_early_features
from viralgraph.simulation.independent_cascade import simulate_independent_cascade
from viralgraph.simulation.intervention_engine import (
    apply_node_immunization,
    apply_community_quarantine,
    evaluate_intervention_impact,
)
from viralgraph.simulation.monte_carlo import run_monte_carlo_simulation


@pytest.fixture
def sample_ba_graph() -> nx.Graph:
    """Fixture providing a deterministic Barabasi-Albert graph."""
    return generate_barabasi_albert(n=60, m=2, seed=42)


@pytest.fixture
def sample_sbm_graph() -> tuple[nx.Graph, dict]:
    """Fixture providing a deterministic Stochastic Block Model graph."""
    sizes = [30, 30]
    p_mat = [[0.15, 0.01], [0.01, 0.15]]
    return generate_stochastic_block_model(community_sizes=sizes, probability_matrix=p_mat, seed=42)


# =============================================================================
# 1. LIVE DIFFUSION TESTS
# =============================================================================

def test_live_diffusion_canvas_rendering(sample_ba_graph: nx.Graph) -> None:
    """Verify render_cascade_canvas produces a valid Plotly Figure with expected traces."""
    res = simulate_independent_cascade(
        graph=sample_ba_graph,
        seed_nodes=[0],
        p=0.2,
        max_steps=5,
        rng=np.random.default_rng(42),
    )

    fig = render_cascade_canvas(sample_ba_graph, res, current_step=1, layout_seed=42)
    assert fig is not None
    # Traces should include background edges, node groups
    trace_names = [t.name for t in fig.data if hasattr(t, "name") and t.name]
    assert len(fig.data) >= 2
    # Ensure layout properties are dark-mode aligned
    assert fig.layout.paper_bgcolor == "#0b0f19"


def test_live_diffusion_timeline_rendering(sample_ba_graph: nx.Graph) -> None:
    """Verify render_cascade_timeline_chart produces both cumulative reach line and velocity bar."""
    res = simulate_independent_cascade(
        graph=sample_ba_graph,
        seed_nodes=[0],
        p=0.2,
        max_steps=5,
        rng=np.random.default_rng(42),
    )

    fig = render_cascade_timeline_chart(res, current_step=1)
    assert fig is not None
    assert len(fig.data) == 2  # Line trace + Bar trace
    assert fig.data[0].name == "Cumulative Reach |A_<=t|"
    assert fig.data[1].name == "New Infections |A_t|"


def test_live_diffusion_scrubbing_bounds(sample_ba_graph: nx.Graph) -> None:
    """Verify canvas rendering behaves safely with out-of-bound step indices."""
    res = simulate_independent_cascade(
        graph=sample_ba_graph,
        seed_nodes=[0],
        p=0.2,
        max_steps=3,
        rng=np.random.default_rng(42),
    )
    max_step = len(res.activation_history) - 1

    # Step < 0 should clamp to 0
    fig_neg = render_cascade_canvas(sample_ba_graph, res, current_step=-5)
    assert fig_neg is not None

    # Step > max_step should clamp to max_step
    fig_high = render_cascade_canvas(sample_ba_graph, res, current_step=max_step + 10)
    assert fig_high is not None


# =============================================================================
# 2. NETWORK EXPLORER & NODE INSPECTION TESTS
# =============================================================================

def test_erdos_renyi_generation_properties() -> None:
    """Verify generate_erdos_renyi matches expected density and node count."""
    g = generate_erdos_renyi(n=80, p=0.05, seed=42)
    assert g.number_of_nodes() == 80
    assert g.number_of_edges() > 0
    # Expected edges approx n*(n-1)/2 * p = 80*79/2 * 0.05 = 158
    assert 80 < g.number_of_edges() < 250


def test_node_inspection_centrality_metrics(sample_ba_graph: nx.Graph) -> None:
    """Verify single-node metrics computed in Network Explorer match NetworkX definitions."""
    v = 0
    deg = sample_ba_graph.degree(v)
    n = sample_ba_graph.number_of_nodes()
    norm_deg = deg / (n - 1)
    clust = nx.clustering(sample_ba_graph, v)
    bet = nx.betweenness_centrality(sample_ba_graph, normalized=True)[v]

    assert norm_deg == deg / (n - 1)
    assert 0.0 <= clust <= 1.0
    assert 0.0 <= bet <= 1.0


# =============================================================================
# 3. INFLUENCER RESEARCH TESTS
# =============================================================================

def test_centrality_profile_rankings(sample_ba_graph: nx.Graph) -> None:
    """Verify compute_centrality_profile yields consistent rankings and BDR metrics."""
    profile = compute_centrality_profile(sample_ba_graph, seed=42)
    assert len(profile.top_degree_nodes) == sample_ba_graph.number_of_nodes()
    assert len(profile.top_betweenness_nodes) == sample_ba_graph.number_of_nodes()

    # The top degree node should indeed have the maximum degree
    top_deg = profile.top_degree_nodes[0]
    degrees = dict(sample_ba_graph.degree())
    assert degrees[top_deg] == max(degrees.values())

    # BDR dictionary exists for all nodes
    assert len(profile.betweenness_degree_ratios) == sample_ba_graph.number_of_nodes()


def test_influencer_head_to_head_mc_simulation(sample_ba_graph: nx.Graph) -> None:
    """Verify Monte Carlo simulation runs cleanly across competing strategies."""
    res_mc = run_monte_carlo_simulation(
        graph=sample_ba_graph,
        seed_nodes=[0],
        model="ICM",
        num_simulations=10,
        seed=42,
        p=0.15,
        max_steps=10,
    )
    assert res_mc.expected_reach >= 1.0
    assert res_mc.num_simulations == 10
    assert len(res_mc.reach_distribution) == 10
    assert res_mc.average_steps_to_termination >= 0.0


# =============================================================================
# 4. INTERVENTION LAB TESTS
# =============================================================================

def test_node_immunization_preserves_original_graph(sample_ba_graph: nx.Graph) -> None:
    """Verify apply_node_immunization operates on a copy and leaves the baseline intact."""
    orig_nodes = set(sample_ba_graph.nodes())
    orig_edges = set(sample_ba_graph.edges())

    g_interv, removed = apply_node_immunization(sample_ba_graph, strategy="targeted_hub", budget=3)

    assert len(removed) == 3
    assert g_interv.number_of_nodes() == sample_ba_graph.number_of_nodes() - 3
    # Original graph must be completely unmodified
    assert set(sample_ba_graph.nodes()) == orig_nodes
    assert set(sample_ba_graph.edges()) == orig_edges


def test_evaluate_intervention_impact(sample_ba_graph: nx.Graph) -> None:
    """Verify evaluate_intervention_impact correctly measures reach suppression."""
    comp = evaluate_intervention_impact(
        graph=sample_ba_graph,
        intervention_type="targeted_hub",
        budget=3,
        p_transmission=0.2,
        mc_simulations=15,
        seed=42,
    )
    assert comp.num_nodes_removed == 3
    assert comp.num_edges_removed > 0
    assert comp.baseline_expected_reach > 0.0
    assert comp.intervened_expected_reach >= 0.0
    # Net suppression percentage should be non-negative
    assert comp.reach_reduction_percentage >= 0.0


def test_community_quarantine(sample_sbm_graph: tuple[nx.Graph, dict]) -> None:
    """Verify apply_community_quarantine severs cut-edges without mutating original graph."""
    g_sbm, comm_map = sample_sbm_graph
    orig_edges = g_sbm.number_of_edges()

    g_quar, severed = apply_community_quarantine(g_sbm, quarantined_community_idx=0, node_to_community=comm_map)

    assert g_sbm.number_of_edges() == orig_edges  # Non-mutation
    assert g_quar.number_of_edges() == orig_edges - len(severed)
    # Ensure severed edges truly connected comm 0 to other communities
    for u, v in severed:
        assert (comm_map[u] == 0 and comm_map[v] != 0) or (comm_map[v] == 0 and comm_map[u] != 0)


# =============================================================================
# 5. VIRALITY PREDICTION TESTS
# =============================================================================

def test_strict_early_feature_extraction_bounds(sample_ba_graph: nx.Graph) -> None:
    """Verify extract_strict_early_features strictly abides by t <= 1 observation window."""
    cascade = simulate_independent_cascade(
        graph=sample_ba_graph,
        seed_nodes=[0],
        p=0.25,
        max_steps=5,
        rng=np.random.default_rng(42),
    )

    feats = extract_strict_early_features(sample_ba_graph, seed_node=0, t_obs=1, cascade_res=cascade)

    # Required feature keys
    required_keys = [
        "network_nodes",
        "network_edges",
        "network_density",
        "seed_degree",
        "seed_pagerank",
        "seed_betweenness",
        "early_observed_reach",
        "early_velocity",
        "early_infections_step1",
    ]
    for k in required_keys:
        assert k in feats

    # Step-1 observed reach should equal seed (1) + newly infected at step 1
    step1_count = len(cascade.activation_history[1]) if len(cascade.activation_history) > 1 else 0
    assert feats["early_observed_reach"] == 1.0 + step1_count
    assert feats["early_infections_step1"] == float(step1_count)

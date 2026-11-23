"""Unit tests for graph analytics, centralities, community detection, and bridge analysis.

Tests:
1. Centrality metrics on Star graph S_n (center node dominates all centralities).
2. Bridge edge and articulation point detection on Barbell topology.
3. Hub vs. Bridge node identification via betweenness-to-degree ratio.
4. Community detection and modularity Q on Stochastic Block Model.
5. Degree distribution moments, assortativity, and power-law scaling estimation.
"""

import networkx as nx
import numpy as np
import pytest

from viralgraph.analytics.influencer_research import (
    AggregatedStrategyOutcome,
    StrategyInstanceMeasurement,
    compare_strategies_paired,
    compute_rigorous_community_metrics,
    compute_student_t_confidence_interval,
)
from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)
from viralgraph.graph.centrality import (
    compute_betweenness_centrality,
    compute_centrality_profile,
    compute_closeness_centrality,
    compute_degree_centrality,
    compute_eigenvector_centrality,
    compute_pagerank,
)
from viralgraph.graph.community import (
    compute_bridging_centrality,
    detect_communities,
)
from viralgraph.graph.degree_analysis import analyze_degree_distribution


class TestCentralityMetrics:
    """Mathematical validation of node centralities."""

    def test_star_graph_centrality_extremes(self):
        """On a Star Graph S_10 (center node 0 connected to 9 leaves):
        - Center node 0 must have Degree Centrality = 1.0 (leaves = 1/9).
        - Center node 0 must have Betweenness Centrality = 1.0 (leaves = 0.0).
        - Center node 0 must have Closeness Centrality = 1.0.
        - Center node 0 must have highest PageRank and Eigenvector centrality.
        """
        star = nx.star_graph(9)  # 10 nodes: 0 is center, 1..9 are leaves
        profile = compute_centrality_profile(star)

        assert profile.degree[0] == 1.0
        assert profile.betweenness[0] == 1.0
        assert profile.closeness[0] == 1.0

        for leaf in range(1, 10):
            assert profile.betweenness[leaf] == 0.0
            assert profile.degree[leaf] == 1.0 / 9.0
            assert profile.closeness[leaf] < 1.0

        assert profile.top_degree_nodes[0] == 0
        assert profile.top_betweenness_nodes[0] == 0
        assert profile.top_pagerank_nodes[0] == 0

    def test_pagerank_probability_distribution(self):
        """PageRank scores must sum to 1.0 across all nodes."""
        g = generate_watts_strogatz(n=80, k=4, p=0.1, seed=42)
        pr = compute_pagerank(g)
        assert np.isclose(sum(pr.values()), 1.0, atol=1e-5)

    def test_eigenvector_centrality_positivity(self):
        """For a connected graph, all eigenvector centrality values must be strictly positive."""
        g = generate_barabasi_albert(n=60, m=3, seed=42)
        eig = compute_eigenvector_centrality(g)
        assert all(val > 0.0 for val in eig.values())


class TestCommunityAndBridgeDetection:
    """Validation of community partitioning, bridges, and articulation points."""

    def test_bridge_and_articulation_detection_on_barbell(self):
        """Create two 5-cliques joined by a single bridge edge (4, 5).
        Vertices 4 and 5 must be articulation points, and (4, 5) must be the bridge edge.
        """
        g = nx.Graph()
        # Clique 1: nodes 0..4
        for u in range(5):
            for v in range(u + 1, 5):
                g.add_edge(u, v)
        # Clique 2: nodes 5..9
        for u in range(5, 10):
            for v in range(u + 1, 10):
                g.add_edge(u, v)
        # Bridge edge connecting the two cliques
        g.add_edge(4, 5)

        comm_struct = detect_communities(g)

        # Bridge detection
        bridge_edges_normalized = [tuple(sorted(e)) for e in comm_struct.bridge_edges]
        assert (4, 5) in bridge_edges_normalized

        # Articulation points
        assert 4 in comm_struct.articulation_points
        assert 5 in comm_struct.articulation_points

        # High modularity Q
        assert comm_struct.modularity > 0.35
        assert comm_struct.num_communities == 2

    def test_hub_vs_bridge_disparity_ratio(self):
        """Verify that bridge nodes exhibit exceptionally high Betweenness-to-Degree ratio."""
        # Two large stars whose leaves are cross-connected by a single bridge node B
        g = nx.Graph()
        # Star 1: hub 100 with 15 leaves (1..15)
        for i in range(1, 16):
            g.add_edge(100, i)
        # Star 2: hub 200 with 15 leaves (21..35)
        for i in range(21, 36):
            g.add_edge(200, i)
        # Bridge node 0 connects leaf 1 and leaf 21
        g.add_edge(0, 1)
        g.add_edge(0, 21)

        profile = compute_centrality_profile(g)

        # Node 0 has degree 2, while hubs have degree 15
        assert g.degree(0) == 2
        assert g.degree(100) == 15

        # Node 0's betweenness must be positive and its Betweenness-to-Degree ratio high
        assert profile.betweenness[0] > 0.3
        assert profile.betweenness_degree_ratios[0] > profile.betweenness_degree_ratios[100]

    def test_bridging_centrality_index(self):
        """Bridging centrality must rank boundary connector nodes above isolated peripheral leaves."""
        g = nx.barbell_graph(m1=8, m2=2)
        bc = compute_bridging_centrality(g)
        # Central path nodes must have higher bridging centrality than clique nodes
        assert bc[8] > bc[0]


class TestDegreeDistributionAnalysis:
    """Statistical validation of degree distributions."""

    def test_scale_free_heavy_tail_properties(self):
        """Barabási-Albert model must be classified as heavy-tailed with positive skewness."""
        g = generate_barabasi_albert(n=400, m=3, seed=42)
        profile = analyze_degree_distribution(g)

        assert profile.is_heavy_tailed is True
        assert profile.skewness > 1.5
        assert profile.variance_degree > profile.mean_degree
        # Clauset MLE exponent for BA is typically around 2.5 - 3.2
        assert 2.0 <= profile.estimated_power_law_gamma <= 4.0

    def test_small_world_homogeneous_distribution(self):
        """Watts-Strogatz model must exhibit a narrow, low-variance distribution."""
        g = generate_watts_strogatz(n=300, k=6, p=0.05, seed=42)
        profile = analyze_degree_distribution(g)

        assert profile.std_degree < 1.5
        assert abs(profile.skewness) < 1.0
        assert profile.variance_degree < profile.mean_degree


class TestStatisticalUncertaintyAndCrossCommunityMetrics:
    """Rigorous audit tests for Task 8 statistical and community metrics."""

    def test_student_t_confidence_interval_small_sample(self):
        """Verify Student's t CI uses t_crit with df = N - 1, not normal 1.96."""
        # 5 samples
        data = np.array([10.0, 12.0, 11.0, 13.0, 14.0])
        mean, s, se, (ci_low, ci_high) = compute_student_t_confidence_interval(data, confidence=0.95)

        assert mean == 12.0
        assert np.isclose(s, np.std(data, ddof=1))
        assert np.isclose(se, s / np.sqrt(5))

        # Critical t for df = 4 at 95% is ~2.7764
        import scipy.stats as stats
        t_crit = stats.t.ppf(0.975, df=4)
        assert np.isclose(t_crit, 2.776445, atol=1e-4)

        expected_margin = t_crit * se
        assert np.isclose(ci_low, 12.0 - expected_margin)
        assert np.isclose(ci_high, 12.0 + expected_margin)

        # Confirm normal 1.96 would have underestimated margin by ~41%
        normal_margin = 1.96 * se
        assert expected_margin > normal_margin * 1.40

    def test_single_sample_confidence_interval(self):
        """Single sample (N=1) must return zero standard error without error."""
        data = np.array([42.0])
        mean, s, se, (ci_low, ci_high) = compute_student_t_confidence_interval(data)
        assert mean == 42.0
        assert s == 0.0
        assert se == 0.0
        assert ci_low == 42.0
        assert ci_high == 42.0

    def test_external_infection_fraction_known_communities(self):
        """Verify rho_external = sum_{v not in C_s} P(v) / sum_{v in V} P(v)."""
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
        # Partition: Comm 0: {0, 1}, Comm 1: {2, 3}, Comm 2: {4}
        comm_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}

        # Seed is 0 (home community C_s = 0)
        # Activation probabilities: P(0)=1.0, P(1)=0.8, P(2)=0.5, P(3)=0.2, P(4)=0.1
        probs = {0: 1.0, 1: 0.8, 2: 0.5, 3: 0.2, 4: 0.1}
        # Total mass = 1.0 + 0.8 + 0.5 + 0.2 + 0.1 = 2.6
        # Outside comm 0: nodes 2, 3, 4 with mass 0.5 + 0.2 + 0.1 = 0.8
        # Expected rho_external = 0.8 / 2.6 ~ 0.30769
        # Outside communities: Comm 1 and Comm 2 (both penetrated since max P >= 0.02)
        # Pi_external = 2 / 2 = 1.0

        pi_ext, rho_ext = compute_rigorous_community_metrics(
            graph=g, seed_node=0, node_to_community=comm_map, activation_probs=probs
        )

        assert np.isclose(rho_ext, 0.8 / 2.6, atol=1e-5)
        assert np.isclose(pi_ext, 1.0, atol=1e-5)

    def test_external_metrics_boundary_single_community(self):
        """When K <= 1 (only 1 community), rho_ext and Pi_ext must return 0.0."""
        g = nx.path_graph(5)
        comm_map = {i: 0 for i in range(5)}  # All nodes in community 0
        probs = {0: 1.0, 1: 0.9, 2: 0.5, 3: 0.3, 4: 0.1}

        pi_ext, rho_ext = compute_rigorous_community_metrics(
            graph=g, seed_node=0, node_to_community=comm_map, activation_probs=probs
        )

        assert pi_ext == 0.0
        assert rho_ext == 0.0

    def test_external_metrics_boundary_empty_cascade(self):
        """When total activation mass is 0, return 0.0 for both."""
        g = nx.path_graph(3)
        comm_map = {0: 0, 1: 1, 2: 1}
        probs = {0: 0.0, 1: 0.0, 2: 0.0}

        pi_ext, rho_ext = compute_rigorous_community_metrics(
            graph=g, seed_node=0, node_to_community=comm_map, activation_probs=probs
        )

        assert pi_ext == 0.0
        assert rho_ext == 0.0

    def test_external_metrics_seed_only_infected(self):
        """When only the seed node is infected (P(s)=1.0, others 0), rho_ext must be 0.0."""
        g = nx.path_graph(3)
        comm_map = {0: 0, 1: 1, 2: 1}
        probs = {0: 1.0, 1: 0.0, 2: 0.0}

        pi_ext, rho_ext = compute_rigorous_community_metrics(
            graph=g, seed_node=0, node_to_community=comm_map, activation_probs=probs
        )

        assert pi_ext == 0.0
        assert rho_ext == 0.0

    def test_paired_strategy_comparison(self):
        """Test paired t-test computation between strategies."""
        # Create mock instance measurements for 5 graph instances
        meas_a = [
            StrategyInstanceMeasurement(
                instance_id=i,
                seed_node=i,
                is_hub=True,
                seed_degree=15.0,
                seed_betweenness=0.5,
                seed_bdr=0.03,
                mc_expected_reach=10.0 + i,
                mc_variance_reach=1.0,
                external_comm_penetration=0.8,
                external_infection_fraction=0.2,
            )
            for i in range(5)
        ]
        meas_b = [
            StrategyInstanceMeasurement(
                instance_id=i,
                seed_node=i + 10,
                is_hub=False,
                seed_degree=8.0,
                seed_betweenness=0.6,
                seed_bdr=0.07,
                mc_expected_reach=7.0 + i,  # Difference is always +3.0
                mc_variance_reach=1.0,
                external_comm_penetration=0.9,
                external_infection_fraction=0.4,
            )
            for i in range(5)
        ]

        outcomes = [
            AggregatedStrategyOutcome(
                strategy_name="highest_degree",
                num_instances=5,
                num_mc_per_instance=40,
                mean_seed_degree=15.0,
                mean_seed_betweenness=0.5,
                mean_seed_bdr=0.03,
                mean_expected_reach=12.0,
                std_expected_reach=1.58,
                se_expected_reach=0.707,
                ci95_expected_reach=(10.0, 14.0),
                within_graph_mc_std=1.0,
                mean_external_comm_penetration=0.8,
                ci95_external_comm_penetration=(0.7, 0.9),
                mean_external_infection_fraction=0.2,
                ci95_external_infection_fraction=(0.15, 0.25),
                overlap_with_hub_rate=1.0,
                instance_measurements=tuple(meas_a),
            ),
            AggregatedStrategyOutcome(
                strategy_name="community_bridge",
                num_instances=5,
                num_mc_per_instance=40,
                mean_seed_degree=8.0,
                mean_seed_betweenness=0.6,
                mean_seed_bdr=0.07,
                mean_expected_reach=9.0,
                std_expected_reach=1.58,
                se_expected_reach=0.707,
                ci95_expected_reach=(7.0, 11.0),
                within_graph_mc_std=1.0,
                mean_external_comm_penetration=0.9,
                ci95_external_comm_penetration=(0.8, 1.0),
                mean_external_infection_fraction=0.4,
                ci95_external_infection_fraction=(0.35, 0.45),
                overlap_with_hub_rate=0.0,
                instance_measurements=tuple(meas_b),
            ),
        ]

        comp = compare_strategies_paired(outcomes, "highest_degree", "community_bridge", metric="reach")
        assert comp.mean_difference == 3.0
        assert comp.std_difference == 0.0
        assert comp.degrees_of_freedom == 4
        assert comp.is_statistically_significant is False  # SE is 0 (identical diffs)

        comp_ext = compare_strategies_paired(
            outcomes, "community_bridge", "highest_degree", metric="external_fraction"
        )
        assert np.isclose(comp_ext.mean_difference, 0.2)


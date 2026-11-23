"""Unit tests for Milestone 2: Post Studio and Content Injection Architecture.

Tests:
1. PostData creation, editing, and field persistence.
2. Erdős–Rényi graph generator validation and error handling.
3. Seed selection strategies (highest degree, highest betweenness, random, manual).
4. Validation guards for empty graphs and non-existent seed nodes.
5. ICM simulation execution and deterministic reproducibility.
6. LTM simulation execution with random and constant thresholds.
7. External community infection fraction calculation.
8. SimulationRecord generation and SessionManager history tracking.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from app.components.diffusion_launcher import (
    compute_external_infection_fraction,
    launch_diffusion_simulation,
    resolve_seed_node,
)
from app.state import PostData, SessionManager, SimulationConfig, SimulationRecord
from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_erdos_renyi,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)


class TestPostComposerAndState:
    """Validation of PostData payload modeling and session state."""

    def test_post_data_defaults_and_mutation(self):
        """Verify default post initialization and field updates."""
        post = PostData()
        assert "Bridging Echo Chambers" in post.title
        assert "Network Science" in post.topic
        assert post.description == ""

        # Update fields
        post.title = "Viral Disinformation Outbreak"
        post.content = "Tracking rumor spread across polarized communities."
        post.topic = "Social Policy & Governance"
        post.description = "Test run with ICM p=0.15"

        assert post.title == "Viral Disinformation Outbreak"
        assert post.topic == "Social Policy & Governance"
        assert post.description == "Test run with ICM p=0.15"

    def test_session_manager_post_roundtrip(self):
        """Verify roundtrip get and set in SessionManager."""
        custom_post = PostData(
            title="Vaccine Efficacy Diffusion",
            content="Simulating health information cascades.",
            topic="Public Health & Epidemiology",
        )
        SessionManager.set_current_post(custom_post)
        retrieved = SessionManager.get_current_post()

        assert retrieved.title == "Vaccine Efficacy Diffusion"
        assert retrieved.topic == "Public Health & Epidemiology"


class TestErdosRenyiGenerator:
    """Validation of the Erdős–Rényi G(n, p) generator."""

    def test_erdos_renyi_generation(self):
        """Verify G(n, p) generates valid simple graph with expected edge density."""
        n = 100
        p = 0.08
        g = generate_erdos_renyi(n=n, p=p, seed=42)

        assert g.number_of_nodes() == n
        # Expected edges ~ p * n * (n - 1) / 2 = 0.08 * 4950 = 396
        # Allow +/- 3 std deviations
        std_edges = np.sqrt(p * (1 - p) * n * (n - 1) / 2)
        assert abs(g.number_of_edges() - 396) < 4 * std_edges
        assert g.graph["model"] == "erdos_renyi"

    def test_erdos_renyi_parameter_validation(self):
        """Verify parameter validation for ER generator."""
        with pytest.raises(TypeError):
            generate_erdos_renyi(n="100", p=0.1)  # type: ignore
        with pytest.raises(ValueError):
            generate_erdos_renyi(n=0, p=0.1)
        with pytest.raises(ValueError):
            generate_erdos_renyi(n=50, p=1.5)


class TestSeedSelectionStrategies:
    """Validation of seed resolution logic."""

    def test_highest_degree_seed_resolution(self):
        """On a star graph, highest degree strategy must resolve center node 0."""
        star = nx.star_graph(10)  # Center node 0 with degree 10
        seed_node, metrics = resolve_seed_node(star, strategy="highest_degree")

        assert seed_node == 0
        assert metrics["degree"] == 10
        assert metrics["normalized_degree"] == 1.0

    def test_highest_betweenness_seed_resolution(self):
        """On two cliques joined by bridge (4, 5), bridge nodes must have peak betweenness."""
        g = nx.barbell_graph(m1=6, m2=2)  # Nodes 5, 6, 7 are bridge path
        seed_node, metrics = resolve_seed_node(g, strategy="highest_betweenness")

        # Central node of the bridge path has maximum shortest paths
        assert seed_node in [6, 5, 7]
        assert metrics["degree"] >= 2

    def test_random_seed_reproducibility(self):
        """Random seed selection must be deterministic given the same seed."""
        ba = generate_barabasi_albert(n=80, m=3, seed=42)
        node1, _ = resolve_seed_node(ba, strategy="random", seed=1007)
        node2, _ = resolve_seed_node(ba, strategy="random", seed=1007)
        assert node1 == node2

    def test_manual_seed_selection_and_validation(self):
        """Manual seed selection must succeed for existing node, fail for missing."""
        g = nx.cycle_graph(10)
        seed_node, metrics = resolve_seed_node(g, strategy="manual", manual_node=5)
        assert seed_node == 5
        assert metrics["degree"] == 2

        with pytest.raises(ValueError, match="does not exist in graph"):
            resolve_seed_node(g, strategy="manual", manual_node=999)

    def test_empty_graph_seed_selection_guard(self):
        """Empty graph must raise ValueError."""
        empty = nx.Graph()
        with pytest.raises(ValueError, match="empty graph"):
            resolve_seed_node(empty, strategy="highest_degree")


class TestDiffusionExecutionAndIntegration:
    """Validation of simulation launch, result storage, and reproducibility."""

    def test_icm_simulation_launch_and_reproducibility(self):
        """Verify ICM execution with deterministic seed yields identical reach."""
        g = generate_barabasi_albert(n=100, m=3, seed=42)
        post = PostData(title="Reproducibility Test")
        config = SimulationConfig(
            model="ICM",
            transmission_prob=0.15,
            max_steps=30,
            seed_strategy="highest_degree",
            selected_seed_node=0,
            random_seed=123,
        )

        res1, rec1 = launch_diffusion_simulation(g, seed_node=0, config=config, post=post)
        res2, rec2 = launch_diffusion_simulation(g, seed_node=0, config=config, post=post)

        assert res1.total_reach == res2.total_reach
        assert res1.steps_to_termination == res2.steps_to_termination
        assert res1.total_reach >= 1
        assert rec1.total_reach == res1.total_reach

        # Check SessionManager was updated
        assert SessionManager.get_simulation_results() == res2
        history = SessionManager.get_simulation_history()
        assert len(history) >= 2
        assert history[0].post_title == "Reproducibility Test"

    def test_ltm_simulation_launch_random_and_constant(self):
        """Verify LTM execution under both random uniform and constant thresholds."""
        g = generate_watts_strogatz(n=80, k=6, p=0.08, seed=42)
        post = PostData(title="LTM Test")

        # 1. Random uniform thresholds
        config_rand = SimulationConfig(
            model="LTM",
            ltm_threshold_mode="random_uniform",
            max_steps=20,
            random_seed=42,
        )
        res_rand, rec_rand = launch_diffusion_simulation(g, seed_node=0, config=config_rand, post=post)
        assert res_rand.total_reach >= 1

        # 2. Constant uniform threshold
        config_const = SimulationConfig(
            model="LTM",
            ltm_threshold_mode="constant_uniform",
            ltm_constant_threshold=0.3,
            max_steps=20,
            random_seed=42,
        )
        res_const, rec_const = launch_diffusion_simulation(g, seed_node=0, config=config_const, post=post)
        assert res_const.total_reach >= 1

    def test_external_community_fraction_calculation(self):
        """Verify external infection fraction calculation with known community map."""
        g = nx.path_graph(6)
        # Block 0: {0, 1, 2}, Block 1: {3, 4, 5}
        comm_map = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1}
        # Infected nodes: {0, 1, 3, 4} (total 4, 2 outside home community 0)
        infected = {0, 1, 3, 4}

        frac = compute_external_infection_fraction(g, seed_node=0, infected_nodes=infected, community_map=comm_map)
        # Seed 0 is home; outside infected: {3, 4} (2 nodes)
        # Fraction = 2 / 4 = 0.5
        assert np.isclose(frac, 0.5)

    def test_invalid_seed_diffusion_guard(self):
        """Simulation launch with non-existent seed must raise ValueError."""
        g = nx.cycle_graph(5)
        config = SimulationConfig(model="ICM")
        post = PostData()

        with pytest.raises(ValueError, match="not present in the graph"):
            launch_diffusion_simulation(g, seed_node=999, config=config, post=post)

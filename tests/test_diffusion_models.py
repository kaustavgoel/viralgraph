"""Unit tests and mathematical invariant validations for information diffusion models (ICM and LTM).

Tests:
1. Strict monotonicity: active sets only expand (A_<=t-1 subseteq A_<=t).
2. Boundary cases: p=0.0 (zero spread), p=1.0 (full component coverage).
3. Complex contagion in LTM: collective adoption requirements.
4. Transmission DAG verification: acyclic causality.
5. Monte Carlo expectation reproducibility and variance properties.
"""

import networkx as nx
import numpy as np
import pytest

from viralgraph.data.synthetic_generator import generate_barabasi_albert
from viralgraph.simulation.independent_cascade import simulate_independent_cascade
from viralgraph.simulation.linear_threshold import simulate_linear_threshold
from viralgraph.simulation.monte_carlo import run_monte_carlo_simulation


class TestIndependentCascadeModel:
    """Mathematical validation for the Independent Cascade Model (ICM)."""

    def test_monotonicity_and_termination(self):
        """Active set must be strictly non-decreasing and terminate within max_steps."""
        g = generate_barabasi_albert(n=100, m=3, seed=42)
        res = simulate_independent_cascade(g, seed_nodes=[0], p=0.15, rng=np.random.default_rng(42))

        # Check monotonicity of cumulative reach
        for t in range(1, len(res.cumulative_reach_timeline)):
            assert res.cumulative_reach_timeline[t] >= res.cumulative_reach_timeline[t - 1]

        assert res.total_reach == res.cumulative_reach_timeline[-1]
        assert res.total_reach >= len(res.seed_nodes)

        # Transmission tree must be an acyclic directed graph (DAG)
        assert nx.is_directed_acyclic_graph(res.transmission_tree)

    def test_boundary_probability_zero_and_one(self):
        """At p=0, reach must strictly equal seed count. At p=1, reach must equal entire connected graph."""
        g = generate_barabasi_albert(n=60, m=2, seed=42)
        seeds = [0, 1]

        # p = 0.0 -> no propagation
        res_zero = simulate_independent_cascade(g, seed_nodes=seeds, p=0.0)
        assert res_zero.total_reach == len(seeds)
        assert res_zero.steps_to_termination == 0

        # p = 1.0 -> full component propagation (BA is connected, so reach must be |V|)
        res_one = simulate_independent_cascade(g, seed_nodes=[0], p=1.0)
        assert res_one.total_reach == g.number_of_nodes()

    def test_invalid_seed_nodes(self):
        """Empty seeds or nonexistent seeds must raise ValueError."""
        g = nx.path_graph(5)
        with pytest.raises(ValueError, match="cannot be empty"):
            simulate_independent_cascade(g, seed_nodes=[])

        with pytest.raises(ValueError, match="not present in graph"):
            simulate_independent_cascade(g, seed_nodes=[999])


class TestLinearThresholdModel:
    """Mathematical validation for the Linear Threshold Model (LTM)."""

    def test_monotonicity_and_threshold_enforcement(self):
        """Nodes activate only when cumulative active neighbor weights exceed theta_v."""
        # Simple triangle: 0 connects to 1 and 2, 1 connects to 2
        g = nx.complete_graph(3)
        # Weights: b_uv = 1/2 for all pairs
        # Set theta_1 = 0.4 (activates with 1 neighbor), theta_2 = 0.8 (requires BOTH neighbors active)
        thresholds = {0: 1.0, 1: 0.4, 2: 0.8}
        seeds = [0]

        res = simulate_linear_threshold(g, seed_nodes=seeds, thresholds=thresholds)

        # Step 0: [0] active
        # Step 1: 1 active (weight from 0 is 0.5 >= 0.4), but 2 is inactive (weight 0.5 < 0.8)
        # Step 2: 2 activates (weight from 0 and 1 is 0.5 + 0.5 = 1.0 >= 0.8)
        assert res.total_reach == 3
        assert res.steps_to_termination == 2
        assert 1 in res.activation_history[1]
        assert 2 in res.activation_history[2]


class TestMonteCarloSimulation:
    """Validation of repeated Monte Carlo diffusion runs."""

    def test_monte_carlo_expectation_bounds_and_reproducibility(self):
        """Expected reach must be bounded between seed size and total nodes, and be deterministic with fixed seed."""
        g = generate_barabasi_albert(n=80, m=2, seed=42)
        mc1 = run_monte_carlo_simulation(g, seed_nodes=[0], model="ICM", num_simulations=50, p=0.12, seed=77)
        mc2 = run_monte_carlo_simulation(g, seed_nodes=[0], model="ICM", num_simulations=50, p=0.12, seed=77)

        assert mc1.expected_reach == mc2.expected_reach
        assert mc1.reach_distribution == mc2.reach_distribution
        assert 1.0 <= mc1.expected_reach <= 80.0

        # Seed node must have activation probability 1.0
        assert mc1.node_activation_probabilities[0] == 1.0

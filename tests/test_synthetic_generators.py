"""Unit tests and mathematical invariant validations for synthetic graph generators.

Tests:
1. Exact vertex and edge counts against theoretical formulas.
2. Euler's Handshaking Lemma: sum(deg(v)) == 2 * |E|.
3. Structural connectivity and component properties.
4. Scale-free hub existence in Barabási-Albert.
5. Small-world clustering and edge preservation in Watts-Strogatz.
6. Community partitioning and homophily in Stochastic Block Model.
7. Random seed determinism and parameter validation.
"""

import networkx as nx
import numpy as np
import pytest

from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)


class TestBarabasiAlbertGenerator:
    """Mathematical validation for Barabási-Albert model."""

    def test_vertex_and_edge_counts(self):
        """Verify that |V| = n and |E| = m * (n - m) holds exactly."""
        n = 100
        m = 3
        g = generate_barabasi_albert(n=n, m=m, seed=42)

        assert g.number_of_nodes() == n
        # In standard NetworkX BA model, edge count is strictly m * (n - m)
        expected_edges = m * (n - m)
        assert g.number_of_edges() == expected_edges

    def test_handshaking_lemma(self):
        """Verify Euler's Handshaking Lemma: sum(deg(v)) = 2 * |E|."""
        g = generate_barabasi_albert(n=150, m=4, seed=123)
        degrees = [d for _, d in g.degree()]
        assert sum(degrees) == 2 * g.number_of_edges()

    def test_minimum_degree_and_connectivity(self):
        """For m >= 1, min degree >= m and graph must be connected (|C| = 1)."""
        m = 3
        g = generate_barabasi_albert(n=80, m=m, seed=99)
        degrees = [d for _, d in g.degree()]

        assert min(degrees) >= m
        assert nx.is_connected(g)
        assert nx.number_connected_components(g) == 1

    def test_hub_emergence(self):
        """Scale-free networks must exhibit high maximum degree relative to mean degree."""
        n = 500
        m = 2
        g = generate_barabasi_albert(n=n, m=m, seed=42)
        degrees = [d for _, d in g.degree()]

        mean_degree = float(np.mean(degrees))
        max_degree = max(degrees)
        # In a 500-node BA network with m=2, max hub degree is typically > 5x the mean degree
        assert max_degree > 3.0 * mean_degree

    def test_seed_reproducibility(self):
        """Identical seeds must produce identical graphs; differing seeds must diverge."""
        g1 = generate_barabasi_albert(n=50, m=2, seed=101)
        g2 = generate_barabasi_albert(n=50, m=2, seed=101)
        g3 = generate_barabasi_albert(n=50, m=2, seed=202)

        assert sorted(g1.edges()) == sorted(g2.edges())
        assert sorted(g1.edges()) != sorted(g3.edges())

    def test_parameter_validation(self):
        """Ensure invalid parameters raise descriptive exceptions."""
        with pytest.raises(ValueError, match="must be >= 1"):
            generate_barabasi_albert(n=50, m=0)
        with pytest.raises(ValueError, match="strictly greater than 'm'"):
            generate_barabasi_albert(n=5, m=5)
        with pytest.raises(TypeError, match="must be an integer"):
            generate_barabasi_albert(n=50.5, m=3)  # type: ignore


class TestWattsStrogatzGenerator:
    """Mathematical validation for Watts-Strogatz model."""

    def test_vertex_and_edge_counts(self):
        """Verify |V| = n and edge count is exactly (n * k) // 2."""
        n = 120
        k = 6
        g = generate_watts_strogatz(n=n, k=k, p=0.1, seed=42)

        assert g.number_of_nodes() == n
        expected_edges = (n * k) // 2
        assert g.number_of_edges() == expected_edges

    def test_handshaking_lemma(self):
        """Verify sum(deg(v)) = 2 * |E|."""
        g = generate_watts_strogatz(n=100, k=4, p=0.2, seed=42)
        degrees = [d for _, d in g.degree()]
        assert sum(degrees) == 2 * g.number_of_edges()

    def test_lattice_limit_clustering(self):
        """At p=0 (pure ring lattice), clustering coefficient is high C ~ 3(k-2)/(4(k-1))."""
        n = 200
        k = 6
        g = generate_watts_strogatz(n=n, k=k, p=0.0, seed=42)
        clustering = nx.average_clustering(g)

        # Theoretical clustering for 1D ring lattice: C = 3*(k-2) / (4*(k-1))
        theoretical_c = (3.0 * (k - 2)) / (4.0 * (k - 1))  # for k=6, 3*4 / 4*5 = 0.60
        assert np.isclose(clustering, theoretical_c, atol=1e-3)

    def test_random_rewiring_reduces_clustering(self):
        """Clustering at p=0.0 must be significantly higher than at p=0.8."""
        g_ordered = generate_watts_strogatz(n=150, k=6, p=0.0, seed=42)
        g_random = generate_watts_strogatz(n=150, k=6, p=0.8, seed=42)

        c_ordered = nx.average_clustering(g_ordered)
        c_random = nx.average_clustering(g_random)
        assert c_ordered > 3.0 * c_random

    def test_seed_reproducibility(self):
        """Verify deterministic generation with identical seeds."""
        g1 = generate_watts_strogatz(n=60, k=4, p=0.15, seed=77)
        g2 = generate_watts_strogatz(n=60, k=4, p=0.15, seed=77)
        assert sorted(g1.edges()) == sorted(g2.edges())

    def test_parameter_validation(self):
        """Ensure invalid inputs trigger appropriate errors."""
        with pytest.raises(ValueError, match="must be an even integer"):
            generate_watts_strogatz(n=50, k=5, p=0.1)
        with pytest.raises(ValueError, match="must be >= 2"):
            generate_watts_strogatz(n=50, k=0, p=0.1)
        with pytest.raises(ValueError, match="must be strictly greater"):
            generate_watts_strogatz(n=6, k=6, p=0.1)
        with pytest.raises(ValueError, match="range \\[0.0, 1.0\\]"):
            generate_watts_strogatz(n=50, k=4, p=1.5)


class TestStochasticBlockModelGenerator:
    """Mathematical validation for Stochastic Block Model."""

    def test_community_partition_and_sizes(self):
        """Verify total nodes match sum of community blocks and attributes are attached."""
        sizes = [40, 50, 60]
        prob_mat = [
            [0.10, 0.01, 0.005],
            [0.01, 0.12, 0.01],
            [0.005, 0.01, 0.08],
        ]
        g, node_map = generate_stochastic_block_model(
            community_sizes=sizes, probability_matrix=prob_mat, seed=42
        )

        assert g.number_of_nodes() == sum(sizes)
        assert len(node_map) == sum(sizes)

        # Check block sizes match requested sizes
        block_counts = [0] * len(sizes)
        for _, b_idx in node_map.items():
            block_counts[b_idx] += 1
        assert block_counts == sizes

    def test_handshaking_lemma(self):
        """Verify Euler's Handshaking Lemma on SBM graph."""
        sizes = [30, 30]
        prob_mat = [[0.15, 0.02], [0.02, 0.15]]
        g, _ = generate_stochastic_block_model(
            community_sizes=sizes, probability_matrix=prob_mat, seed=42
        )
        degrees = [d for _, d in g.degree()]
        assert sum(degrees) == 2 * g.number_of_edges()

    def test_homophily_density_contrast(self):
        """In an assortative SBM, intra-block edge density must exceed inter-block edge density."""
        sizes = [50, 50]
        p_in = 0.20
        p_out = 0.01
        prob_mat = [[p_in, p_out], [p_out, p_in]]
        g, node_map = generate_stochastic_block_model(
            community_sizes=sizes, probability_matrix=prob_mat, seed=42
        )

        intra_edges = 0
        inter_edges = 0
        for u, v in g.edges():
            if node_map[u] == node_map[v]:
                intra_edges += 1
            else:
                inter_edges += 1

        # Max possible intra edges = 2 * (50 * 49 / 2) = 2450
        # Max possible inter edges = 50 * 50 = 2500
        intra_density = intra_edges / (2 * (50 * 49 / 2))
        inter_density = inter_edges / (50 * 50)

        assert intra_density > 5.0 * inter_density

    def test_parameter_validation(self):
        """Ensure asymmetric matrices, out-of-range probabilities, or invalid sizes fail cleanly."""
        # Non-symmetric matrix
        with pytest.raises(ValueError, match="must be symmetric"):
            generate_stochastic_block_model(
                community_sizes=[30, 30],
                probability_matrix=[[0.1, 0.05], [0.01, 0.1]],
            )

        # Probability out of bounds
        with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
            generate_stochastic_block_model(
                community_sizes=[20, 20],
                probability_matrix=[[1.2, 0.0], [0.0, 0.5]],
            )

        # Mismatched dimension
        with pytest.raises(ValueError, match="shape must be"):
            generate_stochastic_block_model(
                community_sizes=[20, 20, 20],
                probability_matrix=[[0.1, 0.0], [0.0, 0.1]],
            )

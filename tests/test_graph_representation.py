"""Unit tests and mathematical invariant validations for GraphRepresentation.

Verifies:
1. Adjacency Matrix symmetry, diagonal zero, binary entries, and Handshaking lemma.
2. Degree Matrix diagonal structure and trace sum.
3. Graph Laplacian row-sum property (L * 1 = 0) and positive semi-definiteness.
4. Walk Counting Theorem (A^k)_uv against analytical manual enumerations.
5. Trace(A^3) / 6 triangle count theorem.
6. Spectral theorems: Real eigenvalues, zero eigenvalue multiplicity == |C|,
   algebraic connectivity (Fiedler value), and spectral radius bounds.
"""

import networkx as nx
import numpy as np
import pytest

from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)
from viralgraph.graph.representation import (
    GraphRepresentation,
    SpectralProfile,
    is_graphical_havel_hakimi,
)


class TestAdjacencyAndDegreeMatrices:
    """Mathematical validation for Adjacency and Degree matrices."""

    def test_adjacency_symmetry_and_diagonal(self):
        """A = A^T and diag(A) = 0 for simple undirected graphs."""
        g = generate_barabasi_albert(n=60, m=3, seed=42)
        rep = GraphRepresentation(g)
        a = rep.get_adjacency_matrix(sparse=False)

        # Symmetry: A - A^T == 0
        assert np.array_equal(a, a.T)
        # No self-loops: diagonal is strictly zeros
        assert np.all(np.diag(a) == 0)
        # Binary entries: A_uv in {0, 1}
        assert np.all(np.isin(a, [0.0, 1.0]))

    def test_handshaking_lemma_on_matrices(self):
        """Verify Euler's Handshaking lemma on both A and D: sum(A) = Tr(D) = 2|E|."""
        g = generate_watts_strogatz(n=80, k=4, p=0.1, seed=42)
        rep = GraphRepresentation(g)
        a = rep.get_adjacency_matrix(sparse=False)
        d = rep.get_degree_matrix(sparse=False)

        total_edges = g.number_of_edges()
        assert np.sum(a) == 2 * total_edges
        assert np.trace(d) == 2 * total_edges

    def test_degree_matrix_diagonal_structure(self):
        """Degree matrix D must be strictly diagonal with D_uu = deg(u)."""
        g = generate_barabasi_albert(n=50, m=2, seed=123)
        rep = GraphRepresentation(g)
        d = rep.get_degree_matrix(sparse=False)
        degrees = rep.raw_degrees

        # Off-diagonal elements must be zero
        off_diag = d - np.diag(np.diag(d))
        assert np.all(off_diag == 0.0)

        # Diagonal entries must match vertex degrees
        assert np.array_equal(np.diag(d), degrees)

    def test_havel_hakimi_theorem(self):
        """Verify graphicality criteria under Havel-Hakimi theorem."""
        # Valid graphical sequence: Complete graph K_4 has sequence (3, 3, 3, 3)
        assert is_graphical_havel_hakimi([3, 3, 3, 3]) is True

        # Invalid: odd sum of degrees violates Handshaking lemma
        assert is_graphical_havel_hakimi([3, 3, 3]) is False

        # Invalid: degree exceeds n-1
        assert is_graphical_havel_hakimi([4, 1, 1, 1]) is False

        # Test on synthesized graph
        g = generate_barabasi_albert(n=50, m=3, seed=42)
        rep = GraphRepresentation(g)
        assert rep.is_graphical is True

    def test_sparse_and_dense_equivalence(self):
        """Sparse CSR matrices must match dense arrays exactly."""
        g = generate_barabasi_albert(n=40, m=2, seed=77)
        rep = GraphRepresentation(g)

        a_dense = rep.get_adjacency_matrix(sparse=False)
        a_sparse = rep.get_adjacency_matrix(sparse=True)
        assert np.array_equal(a_dense, a_sparse.toarray())

        d_dense = rep.get_degree_matrix(sparse=False)
        d_sparse = rep.get_degree_matrix(sparse=True)
        assert np.array_equal(d_dense, d_sparse.toarray())


class TestGraphLaplacian:
    """Mathematical validation of the Graph Laplacian operators."""

    def test_laplacian_row_sum_zero(self):
        """The Combinatorial Laplacian L = D - A must satisfy L * 1 = 0."""
        g = generate_barabasi_albert(n=70, m=3, seed=42)
        rep = GraphRepresentation(g)
        l_mat = rep.get_laplacian_matrix(normalized=None, sparse=False)

        ones = np.ones((rep.num_vertices, 1))
        row_sums = l_mat @ ones
        assert np.allclose(row_sums, 0.0, atol=1e-10)

    def test_laplacian_positive_semidefiniteness(self):
        """L must be positive semi-definite: for any vector x, x^T L x >= 0."""
        g = generate_watts_strogatz(n=50, k=4, p=0.2, seed=42)
        rep = GraphRepresentation(g)
        l_mat = rep.get_laplacian_matrix(normalized=None, sparse=False)

        # Test quadratic form on 20 random vectors
        rng = np.random.default_rng(42)
        for _ in range(20):
            x = rng.standard_normal(rep.num_vertices)
            quadratic_form = float(x @ l_mat @ x)
            assert quadratic_form >= -1e-10

    def test_symmetric_and_random_walk_normalized_laplacians(self):
        """Verify L_sym = D^(-1/2) L D^(-1/2) and L_rw = D^(-1) L."""
        g = generate_barabasi_albert(n=40, m=2, seed=99)
        rep = GraphRepresentation(g)

        l_comb = rep.get_laplacian_matrix(normalized=None, sparse=False)
        d_diag = rep.raw_degrees.astype(float)
        d_inv_sqrt = np.diag(1.0 / np.sqrt(d_diag))
        d_inv = np.diag(1.0 / d_diag)

        expected_l_sym = d_inv_sqrt @ l_comb @ d_inv_sqrt
        actual_l_sym = rep.get_laplacian_matrix(normalized="symmetric", sparse=False)
        assert np.allclose(actual_l_sym, expected_l_sym, atol=1e-10)

        expected_l_rw = d_inv @ l_comb
        actual_l_rw = rep.get_laplacian_matrix(normalized="random_walk", sparse=False)
        assert np.allclose(actual_l_rw, expected_l_rw, atol=1e-10)


class TestWalkCountingTheorem:
    """Verification of the Walk Counting Theorem against analytical small graphs."""

    def test_walk_counting_on_path_graph_p4(self):
        """Analytical verification on Path graph P_4 (0-1-2-3):
        Walks of length 3:
        - 0 to 3: exactly 1 walk (0->1->2->3).
        - 0 to 1: exactly 2 walks (0->1->0->1, 0->1->2->1).
        - 0 to 0: 0 walks (bipartite odd-length closed walk is 0).
        """
        p4 = nx.path_graph(4)
        rep = GraphRepresentation(p4)
        a3 = rep.count_walks(k=3, sparse=False)

        assert a3[0, 3] == 1
        assert a3[0, 1] == 2
        assert a3[0, 0] == 0

    def test_walk_counting_on_triangle_k3(self):
        """Analytical verification on Complete Triangle K_3:
        A = [[0, 1, 1], [1, 0, 1], [1, 1, 0]]
        A^2: diagonal must be 2 (deg=2), off-diagonal must be 1.
        A^3: diagonal must be 2, off-diagonal must be 3.
        Trace(A^3) = 2 + 2 + 2 = 6 = 6 * 1 triangle.
        """
        k3 = nx.complete_graph(3)
        rep = GraphRepresentation(k3)

        a2 = rep.count_walks(k=2, sparse=False)
        expected_a2 = np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        assert np.array_equal(a2, expected_a2)

        a3 = rep.count_walks(k=3, sparse=False)
        expected_a3 = np.array([[2, 3, 3], [3, 2, 3], [3, 3, 2]])
        assert np.array_equal(a3, expected_a3)
        assert rep.count_triangles() == 1

    def test_triangle_counting_theorem_on_k4_and_k5(self):
        """Trace(A^3)/6 theorem on K_4 (4 triangles) and K_5 (10 triangles)."""
        k4 = nx.complete_graph(4)
        assert GraphRepresentation(k4).count_triangles() == 4

        k5 = nx.complete_graph(5)
        assert GraphRepresentation(k5).count_triangles() == 10

    def test_binary_exponentiation_boundary_cases(self):
        """Verify k=0 returns Identity matrix, k=1 returns A."""
        g = generate_barabasi_albert(n=30, m=2, seed=42)
        rep = GraphRepresentation(g)

        a0 = rep.count_walks(k=0, sparse=False)
        assert np.array_equal(a0, np.eye(30))

        a1 = rep.count_walks(k=1, sparse=False)
        assert np.array_equal(a1, rep.get_adjacency_matrix(sparse=False))


class TestSpectralTheorems:
    """Validation of Spectral Graph Theory theorems."""

    def test_eigenvalue_reality_and_laplacian_properties(self):
        """Laplacian eigenvalues must be non-negative real numbers starting at 0."""
        g = generate_barabasi_albert(n=60, m=3, seed=42)
        rep = GraphRepresentation(g)
        profile = rep.compute_spectral_profile()

        # Real numbers
        assert np.isreal(profile.laplacian_eigenvalues).all()
        assert np.isreal(profile.adjacency_eigenvalues).all()

        # Smallest eigenvalue lambda_1 == 0
        assert np.isclose(profile.laplacian_eigenvalues[0], 0.0, atol=1e-8)

        # Monotonically non-decreasing
        assert np.all(np.diff(profile.laplacian_eigenvalues) >= -1e-10)

    def test_multiplicity_of_zero_eigenvalue_equals_components(self):
        """Theorem: Multiplicity of lambda=0 in Laplacian equals |C| (number of connected components)."""
        # 1. Connected graph: |C| = 1 => exactly one zero eigenvalue, algebraic connectivity > 0
        g_conn = generate_watts_strogatz(n=60, k=4, p=0.1, seed=42)
        profile_conn = GraphRepresentation(g_conn).compute_spectral_profile()
        assert profile_conn.zero_eigenvalue_multiplicity == 1
        assert profile_conn.algebraic_connectivity > 0.0

        # 2. Disconnected graph with 2 components: K_3 disjoint union K_4 => |C| = 2
        g_disconn = nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(4))
        profile_disconn = GraphRepresentation(g_disconn).compute_spectral_profile()
        assert profile_disconn.zero_eigenvalue_multiplicity == 2
        assert np.isclose(profile_disconn.algebraic_connectivity, 0.0, atol=1e-8)

    def test_spectral_radius_bounds(self):
        """Theorem: d_avg <= mu_1 <= d_max for the adjacency matrix spectral radius."""
        g = generate_barabasi_albert(n=80, m=2, seed=42)
        rep = GraphRepresentation(g)
        profile = rep.compute_spectral_profile()

        d_avg = float(np.mean(rep.raw_degrees))
        d_max = float(np.max(rep.raw_degrees))
        mu_1 = profile.spectral_radius

        assert d_avg <= mu_1 + 1e-8
        assert mu_1 <= d_max + 1e-8


class TestErrorHandlingAndValidation:
    """Ensure invalid graphs or parameters raise clear exceptions."""

    def test_reject_directed_graphs(self):
        dg = nx.DiGraph([(0, 1), (1, 2)])
        with pytest.raises(ValueError, match="requires undirected graphs"):
            GraphRepresentation(dg)

    def test_reject_self_loops(self):
        g = nx.Graph([(0, 1)])
        g.add_edge(0, 0)
        with pytest.raises(ValueError, match="without self-loops"):
            GraphRepresentation(g)

    def test_reject_negative_walk_length(self):
        g = nx.path_graph(3)
        rep = GraphRepresentation(g)
        with pytest.raises(ValueError, match="must be non-negative"):
            rep.count_walks(k=-1)

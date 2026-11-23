"""Graph representations and matrix theory module for ViralGraph.

This module provides mathematically sound matrix representations, spectral decompositions,
and walk-counting operators for complex social networks:
1. Adjacency Matrix (A): Dense (NumPy) and Sparse (SciPy CSR).
2. Degree Matrix (D): Diagonal matrix of vertex degrees.
3. Graph Laplacian (L = D - A) and Normalized Laplacians (L_sym, L_rw).
4. Matrix powers (A^k) and walk counting theorem.
5. Spectral analysis: Eigenvalues and eigenvectors of A and L, algebraic connectivity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal
import networkx as nx
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


def is_graphical_havel_hakimi(sequence: list[int] | np.ndarray) -> bool:
    """Verify whether a non-negative integer sequence is graphical using the Havel-Hakimi algorithm.

    Havel-Hakimi Theorem (Discrete Mathematics):
        A finite sequence of non-negative integers s = (d_1 >= d_2 >= ... >= d_n)
        is graphical (realizable by an undirected simple graph) if and only if
        the reduced sequence s' obtained by removing d_1 and subtracting 1 from
        the next d_1 terms is graphical.

    Args:
        sequence: Sequence of candidate vertex degrees.

    Returns:
        bool: True if the sequence can form an undirected simple graph without self-loops or multi-edges.
    """
    # Filter negative entries immediately
    seq = [int(x) for x in sequence]
    if any(x < 0 for x in seq):
        return False

    while True:
        # Sort non-increasing and prune zeros
        seq = sorted([d for d in seq if d > 0], reverse=True)
        if not seq:
            return True
        d1 = seq.pop(0)
        if d1 > len(seq):
            return False
        for i in range(d1):
            seq[i] -= 1
            if seq[i] < 0:
                return False


@dataclass(frozen=True)
class SpectralProfile:
    """Immutable container holding spectral invariants of a graph."""

    # Adjacency matrix eigenvalues (sorted descending: mu_1 >= mu_2 >= ... >= mu_n)
    adjacency_eigenvalues: np.ndarray
    spectral_radius: float

    # Laplacian eigenvalues (sorted ascending: 0 = lambda_1 <= lambda_2 <= ... <= lambda_n)
    laplacian_eigenvalues: np.ndarray
    algebraic_connectivity: float  # Fiedler eigenvalue lambda_2
    zero_eigenvalue_multiplicity: int  # Multiplicity of eigenvalue 0 == |C|


class GraphRepresentation:
    """Comprehensive discrete matrix representation and algebraic operator for an undirected simple graph.

    Mathematical Formulation:
        Let G = (V, E) be an undirected simple graph with |V| = n, |E| = m.
        - Vertices are indexed V = {0, 1, ..., n - 1}.
        - Adjacency Matrix A in {0, 1}^(n x n): A_uv = 1 if (u, v) in E, else 0.
        - Degree Matrix D in Z^(n x n): D_uu = deg(u) = sum_v A_uv, D_uv = 0 for u != v.
        - Combinatorial Laplacian L in Z^(n x n): L = D - A.
        - Normalized Symmetric Laplacian L_sym = D^(-1/2) L D^(-1/2) = I - D^(-1/2) A D^(-1/2).
        - Random Walk Laplacian L_rw = D^(-1) L = I - D^(-1) A = I - P.
    """

    def __init__(self, graph: nx.Graph) -> None:
        """Initialize matrix representations from a NetworkX undirected graph.

        Args:
            graph: NetworkX graph. Must be undirected and simple (no self-loops or multi-edges).

        Raises:
            TypeError: If graph is not a networkx.Graph.
            ValueError: If graph is directed or contains self-loops.
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError(f"Expected networkx.Graph, got {type(graph).__name__}.")
        if graph.is_directed():
            raise ValueError("GraphRepresentation currently requires undirected graphs.")
        if nx.number_of_selfloops(graph) > 0:
            raise ValueError("GraphRepresentation requires simple graphs without self-loops.")

        self._graph: nx.Graph = graph
        self._n: int = graph.number_of_nodes()
        self._m: int = graph.number_of_edges()

        # Consistent deterministic vertex ordering: 0, 1, ..., n-1
        # If node labels are not 0..n-1, we map them preserving their sorted order
        self._nodes: list[Any] = sorted(list(graph.nodes()))
        self._node_to_idx: dict[Any, int] = {node: idx for idx, node in enumerate(self._nodes)}
        self._idx_to_node: dict[int, Any] = {idx: node for idx, node in enumerate(self._nodes)}

        # Precompute sparse CSR matrices for memory and computational efficiency
        self._sparse_adj: sp.csr_matrix = nx.to_scipy_sparse_array(
            graph, nodelist=self._nodes, format="csr", dtype=np.float64
        )
        self._degrees: np.ndarray = np.array(
            [graph.degree(node) for node in self._nodes], dtype=np.int64
        )

    # -------------------------------------------------------------------------
    # Properties & Basic Counts
    # -------------------------------------------------------------------------

    @property
    def num_vertices(self) -> int:
        """Number of vertices |V| = n."""
        return self._n

    @property
    def num_edges(self) -> int:
        """Number of edges |E| = m."""
        return self._m

    @property
    def degree_sequence(self) -> np.ndarray:
        """Degree sequence sorted in non-increasing order: (k_(1) >= k_(2) >= ... >= k_(n))."""
        return np.sort(self._degrees)[::-1]

    @property
    def raw_degrees(self) -> np.ndarray:
        """Array of degrees aligned with canonical vertex indices 0, 1, ..., n-1."""
        return self._degrees.copy()

    @property
    def is_graphical(self) -> bool:
        """Verify that the graph's degree sequence satisfies the Havel-Hakimi theorem."""
        return is_graphical_havel_hakimi(self.degree_sequence)

    @property
    def node_to_index(self) -> dict[Any, int]:
        """Dictionary mapping canonical node IDs to matrix row/column indices [0, n-1]."""
        return self._node_to_idx.copy()

    @property
    def index_to_node(self) -> dict[int, Any]:
        """Dictionary mapping matrix row/column index to original node ID."""
        return self._idx_to_node.copy()

    # -------------------------------------------------------------------------
    # Matrix Representations
    # -------------------------------------------------------------------------

    def get_adjacency_matrix(self, sparse: bool = False) -> np.ndarray | sp.csr_matrix:
        """Get the Adjacency Matrix A.

        Mathematical Invariants:
            - A = A^T (symmetric for undirected graphs).
            - diag(A) = 0 (no self-loops).
            - A_uv in {0, 1} for simple graphs.
            - sum_uv A_uv = 2 * |E| (Handshaking lemma).

        Args:
            sparse: If True, returns scipy.sparse.csr_matrix. Else returns dense numpy.ndarray.
        """
        if sparse:
            return self._sparse_adj.copy()
        return self._sparse_adj.toarray()

    def get_degree_matrix(self, sparse: bool = False) -> np.ndarray | sp.csr_matrix:
        """Get the Degree Matrix D = diag(deg(v_0), deg(v_1), ..., deg(v_(n-1))).

        Mathematical Invariants:
            - D is a diagonal matrix: D_uv = 0 for u != v.
            - Trace(D) = sum deg(v_i) = 2 * |E|.
        """
        if sparse:
            return sp.diags(self._degrees, format="csr", dtype=np.float64)
        return np.diag(self._degrees.astype(np.float64))

    def get_laplacian_matrix(
        self,
        normalized: Literal[None, "symmetric", "random_walk"] = None,
        sparse: bool = False,
    ) -> np.ndarray | sp.csr_matrix:
        """Get the Graph Laplacian matrix L.

        Variants:
            - None: Standard combinatorial Laplacian L = D - A.
                Row-sum: L * 1 = 0.
                Quadratic form:
                    x^T L x = 1/2 * sum_u sum_v A_uv (x_u - x_v)^2
                            = sum_{{u, v} in E} (x_u - x_v)^2 >= 0 (Positive semi-definite).
            - "symmetric": L_sym = D^(-1/2) L D^(-1/2) = I - D^(-1/2) A D^(-1/2).
            - "random_walk": L_rw = D^(-1) L = I - D^(-1) A = I - P.

        Args:
            normalized: None, 'symmetric', or 'random_walk'.
            sparse: If True, returns SciPy CSR matrix.
        """
        d_sparse = sp.diags(self._degrees, format="csr", dtype=np.float64)
        l_combinatorial = d_sparse - self._sparse_adj

        if normalized is None:
            if sparse:
                return l_combinatorial
            return l_combinatorial.toarray()

        if normalized == "symmetric":
            # For isolated vertices (deg=0), define d^(-1/2) = 0 to avoid division by zero
            with np.errstate(divide="ignore", invalid="ignore"):
                d_inv_sqrt = np.where(self._degrees > 0, 1.0 / np.sqrt(self._degrees), 0.0)
            d_inv_sqrt_mat = sp.diags(d_inv_sqrt, format="csr", dtype=np.float64)
            l_sym = d_inv_sqrt_mat @ l_combinatorial @ d_inv_sqrt_mat
            if sparse:
                return l_sym
            return l_sym.toarray()

        if normalized == "random_walk":
            with np.errstate(divide="ignore", invalid="ignore"):
                d_inv = np.where(self._degrees > 0, 1.0 / self._degrees, 0.0)
            d_inv_mat = sp.diags(d_inv, format="csr", dtype=np.float64)
            l_rw = d_inv_mat @ l_combinatorial
            if sparse:
                return l_rw
            return l_rw.toarray()

        raise ValueError(f"Unknown normalization mode: '{normalized}'. Use None, 'symmetric', or 'random_walk'.")

    # -------------------------------------------------------------------------
    # Walk Counting Theorem via Matrix Exponentiation
    # -------------------------------------------------------------------------

    def count_walks(self, k: int, sparse: bool = False) -> np.ndarray | sp.csr_matrix:
        """Compute the number of walks of exact length k between all pairs of vertices.

        Walk Counting Theorem (Discrete Mathematics):
            Let A be the adjacency matrix of a simple graph G. For any positive integer k,
            the entry (A^k)_uv equals the exact number of distinct walks of length k
            from vertex u to vertex v.

            Corollaries:
            1. (A^2)_uu = deg(u): A walk of length 2 from u to u is a traversal u -> v -> u for any neighbor v.
            2. Trace(A^2) = sum_u deg(u) = 2 * |E|.
            3. Trace(A^3) = 6 * (number of triangles in G), because each triangle {u, v, w}
               generates 6 directed closed walks of length 3:
               (u->v->w->u, u->w->v->u, and cyclic permutations).

        Args:
            k: Length of walks (k >= 0).
            sparse: If True, returns sparse matrix.

        Raises:
            ValueError: If k < 0.
        """
        if not isinstance(k, int) or isinstance(k, bool):
            raise TypeError(f"Walk length k must be an integer, got {type(k).__name__}.")
        if k < 0:
            raise ValueError(f"Walk length k must be non-negative, got {k}.")

        if k == 0:
            # A^0 = Identity matrix I_n
            i_mat = sp.eye(self._n, format="csr", dtype=np.int64)
            return i_mat if sparse else i_mat.toarray()

        if k == 1:
            a_k = self._sparse_adj.copy()
            return a_k if sparse else a_k.toarray()

        # Binary matrix exponentiation for O(n^3 log k) complexity
        result = sp.eye(self._n, format="csr", dtype=np.float64)
        base = self._sparse_adj.copy()
        exp = k

        while exp > 0:
            if exp % 2 == 1:
                result = result @ base
            base = base @ base
            exp //= 2

        if sparse:
            return result
        return result.toarray()

    def count_triangles(self) -> int:
        """Count total triangles in G using the matrix trace theorem: Triangles = Trace(A^3) / 6."""
        if self._n < 3 or self._m < 3:
            return 0
        a3 = self.count_walks(k=3, sparse=True)
        trace_a3 = float(a3.diagonal().sum())
        return int(round(trace_a3 / 6.0))

    # -------------------------------------------------------------------------
    # Spectral Decomposition
    # -------------------------------------------------------------------------

    def compute_spectral_profile(self) -> SpectralProfile:
        """Compute the complete eigenvalue spectrum for the Adjacency and Laplacian matrices.

        Spectral Theorems (Spectral Graph Theory):
            1. Real Eigenvalues: A and L are real symmetric, hence all eigenvalues are real:
               lambda_i in R, mu_i in R.
            2. Laplacian Spectrum:
               0 = lambda_1 <= lambda_2 <= ... <= lambda_n.
               The multiplicity of eigenvalue 0 equals the number of connected components |C|.
            3. Algebraic Connectivity: lambda_2 (Fiedler value).
               lambda_2 > 0 if and only if G is connected.
            4. Spectral Radius: mu_1 = max |mu_i| = lambda_max(A).
               d_avg <= mu_1 <= d_max.
        """
        if self._n == 0:
            return SpectralProfile(
                adjacency_eigenvalues=np.array([]),
                spectral_radius=0.0,
                laplacian_eigenvalues=np.array([]),
                algebraic_connectivity=0.0,
                zero_eigenvalue_multiplicity=0,
            )

        if self._n == 1:
            return SpectralProfile(
                adjacency_eigenvalues=np.array([0.0]),
                spectral_radius=0.0,
                laplacian_eigenvalues=np.array([0.0]),
                algebraic_connectivity=0.0,
                zero_eigenvalue_multiplicity=1,
            )

        # Dense symmetric eigensolver (scipy.linalg.eigh / numpy.linalg.eigh)
        adj_dense = self.get_adjacency_matrix(sparse=False)
        lap_dense = self.get_laplacian_matrix(normalized=None, sparse=False)

        # eigh returns eigenvalues in ascending order
        adj_evals = np.linalg.eigh(adj_dense)[0]
        # Sort adjacency eigenvalues descending: mu_1 >= mu_2 >= ... >= mu_n
        adj_evals_desc = np.sort(adj_evals)[::-1]
        spectral_radius = float(np.max(np.abs(adj_evals_desc)))

        lap_evals = np.sort(np.linalg.eigh(lap_dense)[0])
        # Force numerical zero for machine precision artifacts around lambda_1
        lap_evals = np.where(np.abs(lap_evals) < 1e-10, 0.0, lap_evals)

        # Multiplicity of eigenvalue 0
        zero_eval_multiplicity = int(np.sum(np.isclose(lap_evals, 0.0, atol=1e-8)))

        # Fiedler eigenvalue lambda_2 (algebraic connectivity)
        fiedler_val = float(lap_evals[1]) if len(lap_evals) > 1 else 0.0

        return SpectralProfile(
            adjacency_eigenvalues=adj_evals_desc,
            spectral_radius=spectral_radius,
            laplacian_eigenvalues=lap_evals,
            algebraic_connectivity=fiedler_val,
            zero_eigenvalue_multiplicity=zero_eval_multiplicity,
        )

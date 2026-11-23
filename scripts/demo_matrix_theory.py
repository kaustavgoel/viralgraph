"""Demonstration of Graph Representations and Matrix Theory in ViralGraph.

Runs matrix operations, walk counting, and spectral decompositions on:
1. Canonical toy graphs (K_3, P_4) for analytical verification.
2. Complex networks (Barabási-Albert, Watts-Strogatz, Stochastic Block Model).
"""

import numpy as np
import networkx as nx

from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_watts_strogatz,
    generate_stochastic_block_model,
)
from viralgraph.graph.representation import GraphRepresentation


def demonstrate_toy_graphs():
    print("=" * 70)
    print("PART 1: ANALYTICAL TOY GRAPH VERIFICATION")
    print("=" * 70)

    # 1. Triangle K_3
    k3 = nx.complete_graph(3)
    rep_k3 = GraphRepresentation(k3)
    print("\n--- Triangle Graph K_3 ---")
    print("Adjacency Matrix A:\n", rep_k3.get_adjacency_matrix())
    print("Degree Matrix D:\n", rep_k3.get_degree_matrix())
    print("Combinatorial Laplacian L = D - A:\n", rep_k3.get_laplacian_matrix())
    print("Walks of length 2 (A^2):\n", rep_k3.count_walks(k=2))
    print("Walks of length 3 (A^3):\n", rep_k3.count_walks(k=3))
    print(f"Total Triangles = Tr(A^3) / 6: {rep_k3.count_triangles()} (Expected: 1)")

    # 2. Path Graph P_4 (0-1-2-3)
    p4 = nx.path_graph(4)
    rep_p4 = GraphRepresentation(p4)
    print("\n--- Path Graph P_4 (0-1-2-3) ---")
    a3 = rep_p4.count_walks(k=3)
    print("Walks of length 3 (A^3):\n", a3)
    print(f"Walks 0 -> 3: {a3[0, 3]} (Expected: 1 via 0-1-2-3)")
    print(f"Walks 0 -> 1: {a3[0, 1]} (Expected: 2 via 0-1-0-1 and 0-1-2-1)")


def demonstrate_complex_topologies():
    print("\n" + "=" * 70)
    print("PART 2: SPECTRAL & MATRIX ANALYSIS OF COMPLEX NETWORKS")
    print("=" * 70)

    graphs = [
        ("Barabási–Albert (Scale-Free, N=150, m=3)", generate_barabasi_albert(n=150, m=3, seed=42)),
        ("Watts–Strogatz (Small-World, N=150, k=6, p=0.08)", generate_watts_strogatz(n=150, k=6, p=0.08, seed=42)),
        ("Stochastic Block Model (K=3, N=150)", generate_stochastic_block_model([50, 50, 50], [[0.12, 0.005, 0.002], [0.005, 0.10, 0.005], [0.002, 0.005, 0.10]], seed=42)[0]),
    ]

    for name, g in graphs:
        rep = GraphRepresentation(g)
        profile = rep.compute_spectral_profile()
        triangles = rep.count_triangles()
        a2 = rep.count_walks(k=2)
        total_2_walks = int(np.sum(a2))

        print(f"\n--- {name} ---")
        print(f"  Vertices (|V|)             : {rep.num_vertices}")
        print(f"  Edges (|E|)                : {rep.num_edges}")
        print(f"  Handshaking Sum(A) == 2|E| : {int(np.sum(rep.get_adjacency_matrix()))} == {2 * rep.num_edges}")
        print(f"  Total Closed 2-Walks Tr(A2): {int(np.trace(a2))} == 2|E|")
        print(f"  Total Triangles Tr(A3)/6   : {triangles}")
        print(f"  Spectral Radius mu_1(A)    : {profile.spectral_radius:.4f}")
        print(f"  Algebraic Connectivity l_2 : {profile.algebraic_connectivity:.4f} (Fiedler value)")
        print(f"  Zero Eigenvalue Mult |C|   : {profile.zero_eigenvalue_multiplicity}")
        print(f"  Top 3 Adjacency Eigenvals  : {np.round(profile.adjacency_eigenvalues[:3], 3)}")
        print(f"  First 3 Laplacian Eigenvals: {np.round(profile.laplacian_eigenvalues[:3], 3)}")


if __name__ == "__main__":
    demonstrate_toy_graphs()
    demonstrate_complex_topologies()

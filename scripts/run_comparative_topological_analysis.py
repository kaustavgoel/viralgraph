"""Executable script for Task 5: Comparative Topological Analysis.

Computes the metric profiles for BA, WS, and SBM, outputs the academic Markdown table,
and renders a 4-panel scientific comparison figure saved to docs/comparative_topological_analysis.png.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from viralgraph.analytics.topological_comparator import run_benchmark_comparison
from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)
from viralgraph.graph.centrality import compute_centrality_profile
from viralgraph.graph.representation import GraphRepresentation


def generate_diagnostic_plots(output_path: str = "docs/comparative_topological_analysis.png"):
    seed = 42
    n = 300
    ba = generate_barabasi_albert(n=n, m=3, seed=seed)
    ws = generate_watts_strogatz(n=n, k=6, p=0.08, seed=seed)
    sbm, _ = generate_stochastic_block_model([100, 100, 100], [[0.08, 0.004, 0.002], [0.004, 0.07, 0.004], [0.002, 0.004, 0.06]], seed=seed)

    graphs = [
        ("Barabási–Albert", ba, "#3B82F6"),
        ("Watts–Strogatz", ws, "#10B981"),
        ("Stochastic Block", sbm, "#F59E0B"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), dpi=150)
    fig.patch.set_facecolor("#0e1117")

    for ax in axes.flat:
        ax.set_facecolor("#151d2f")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.2, color="#94a3b8")

    # 1. Degree Distribution (Log-Log)
    ax1 = axes[0, 0]
    ax1.set_title("Degree Distribution P(k) [Log-Log]", color="#f8fafc", fontsize=13, fontweight="bold")
    for name, g, color in graphs:
        deg = [d for _, d in g.degree()]
        unique, counts = np.unique(deg, return_counts=True)
        ax1.scatter(unique, counts / len(deg), label=name, color=color, s=40, alpha=0.85)
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Degree (k)", color="#cbd5e1")
    ax1.set_ylabel("Probability P(k)", color="#cbd5e1")
    ax1.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    # 2. Betweenness vs Degree (Bridge Node Detection)
    ax2 = axes[0, 1]
    ax2.set_title("Betweenness Centrality vs. Degree (Hubs vs. Bridges)", color="#f8fafc", fontsize=13, fontweight="bold")
    for name, g, color in graphs:
        cent = compute_centrality_profile(g, seed=seed)
        degs = [g.degree(node) for node in g.nodes()]
        bets = [cent.betweenness[node] for node in g.nodes()]
        ax2.scatter(degs, bets, label=name, color=color, s=25, alpha=0.6)
    ax2.set_xlabel("Degree k", color="#cbd5e1")
    ax2.set_ylabel("Betweenness Centrality C_B", color="#cbd5e1")
    ax2.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    # 3. Local Clustering vs Degree C(k)
    ax3 = axes[1, 0]
    ax3.set_title("Clustering Coefficient vs. Degree C(k)", color="#f8fafc", fontsize=13, fontweight="bold")
    for name, g, color in graphs:
        degs = np.array([g.degree(node) for node in g.nodes()])
        clusts = np.array([nx.clustering(g, node) for node in g.nodes()])
        unique_k = np.unique(degs)
        mean_c_k = [np.mean(clusts[degs == k]) for k in unique_k]
        ax3.plot(unique_k, mean_c_k, marker="o", label=name, color=color, linewidth=2, markersize=5)
    ax3.set_xlabel("Degree k", color="#cbd5e1")
    ax3.set_ylabel("Average Clustering C(k)", color="#cbd5e1")
    ax3.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    # 4. Laplacian Eigenvalue Spectrum (Algebraic Connectivity)
    ax4 = axes[1, 1]
    ax4.set_title("Laplacian Eigenvalue Spectrum (Sorted ascending)", color="#f8fafc", fontsize=13, fontweight="bold")
    for name, g, color in graphs:
        rep = GraphRepresentation(g)
        evals = rep.compute_spectral_profile().laplacian_eigenvalues
        ax4.plot(np.linspace(0, 1, len(evals)), evals, label=f"{name} (λ₂={evals[1]:.2f})", color=color, linewidth=2)
    ax4.set_xlabel("Normalized Eigenvalue Index (i / N)", color="#cbd5e1")
    ax4.set_ylabel("Laplacian Eigenvalue λ_i", color="#cbd5e1")
    ax4.legend(facecolor="#1e293b", edgecolor="#475569", labelcolor="#f8fafc")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def main():
    records = run_benchmark_comparison(n_nodes=300, seed=42)
    generate_diagnostic_plots("/Users/kaustavgoel/.gemini/antigravity/scratch/viralgraph/docs/comparative_topological_analysis.png")

    print("=" * 80)
    print("TASK 5: COMPARATIVE TOPOLOGICAL BENCHMARK (N=300)")
    print("=" * 80)
    for r in records:
        print(f"\nModel: {r.model_name}")
        print(f"  Nodes / Edges / Density   : {r.nodes} / {r.edges} / {r.density:.4f}")
        print(f"  Degree Mean / Std / Max   : {r.mean_degree:.2f} / {r.std_degree:.2f} / {r.max_degree}")
        print(f"  Skewness / Assortativity  : {r.skewness:.2f} / {r.assortativity:.3f}")
        print(f"  Clustering / Transitivity : {r.avg_clustering:.3f} / {r.transitivity:.3f}")
        print(f"  Shortest Path / Diameter  : {r.avg_shortest_path:.2f} / {r.diameter}")
        print(f"  Spectral Radius / Fiedler : {r.spectral_radius:.2f} / {r.algebraic_connectivity:.4f}")
        print(f"  Communities / Modularity  : {r.num_communities} / Q={r.modularity_q:.3f}")
        print(f"  Bridges / Articulations   : {r.num_bridge_edges} / {r.num_articulation_points}")
        print(f"  Max Betweenness/Degree BDR: {r.max_betweenness_degree_ratio:.2f}")


if __name__ == "__main__":
    main()

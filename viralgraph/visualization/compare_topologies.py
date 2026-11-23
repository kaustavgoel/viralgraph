"""Comparative topological analysis and visualization for synthetic graph models.

This module generates canonical instances of Barabási-Albert, Watts-Strogatz,
and Stochastic Block Model graphs, computes their discrete mathematical invariants,
and plots their network layouts and degree distributions for structural comparison.
"""

from __future__ import annotations

import os
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless rendering
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)


def compute_discrete_math_profile(graph: nx.Graph) -> dict[str, float | int | str]:
    """Compute discrete mathematical topological invariants for a graph."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = [deg for _, deg in graph.degree()]

    # Handshaking lemma assertion
    assert sum(degrees) == 2 * m, "Handshaking lemma failed: sum(deg(v)) != 2|E|"

    # Connected components
    components = list(nx.connected_components(graph))
    num_components = len(components)
    giant_comp_nodes = max(components, key=len)
    giant_subgraph = graph.subgraph(giant_comp_nodes)

    # Average clustering coefficient
    avg_clustering = float(nx.average_clustering(graph))

    # Average shortest path length & diameter on giant component
    avg_path_length = float(nx.average_shortest_path_length(giant_subgraph))
    diameter = int(nx.diameter(giant_subgraph))

    return {
        "model_name": graph.name,
        "nodes": n,
        "edges": m,
        "min_degree": int(np.min(degrees)),
        "max_degree": int(np.max(degrees)),
        "mean_degree": float(np.mean(degrees)),
        "std_degree": float(np.std(degrees)),
        "avg_clustering": avg_clustering,
        "avg_path_length": avg_path_length,
        "diameter": diameter,
        "connected_components": num_components,
        "giant_component_ratio": float(len(giant_comp_nodes) / n),
    }


def plot_synthetic_comparison(output_path: str = "docs/synthetic_topologies_comparison.png") -> None:
    """Generate canonical synthetic graphs and render a visual comparison panel."""
    # 1. Barabási-Albert
    ba_graph = generate_barabasi_albert(n=300, m=3, seed=42)

    # 2. Watts-Strogatz
    ws_graph = generate_watts_strogatz(n=300, k=6, p=0.08, seed=42)

    # 3. Stochastic Block Model
    sbm_sizes = [100, 100, 100]
    sbm_prob = [
        [0.080, 0.004, 0.002],
        [0.004, 0.070, 0.004],
        [0.002, 0.004, 0.060],
    ]
    sbm_graph, sbm_map = generate_stochastic_block_model(
        community_sizes=sbm_sizes, probability_matrix=sbm_prob, seed=42
    )

    graphs = [
        ("Barabási-Albert (Scale-Free)", ba_graph, None),
        ("Watts-Strogatz (Small-World)", ws_graph, None),
        ("Stochastic Block Model (Communities)", sbm_graph, sbm_map),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=150)
    fig.patch.set_facecolor("#0e1117")

    palette = ["#4A90E2", "#E94B35", "#50E3C2", "#F5A623"]

    for ax, (title, g, node_map) in zip(axes, graphs):
        ax.set_facecolor("#0e1117")
        ax.set_title(title, color="#FAFAFA", fontsize=14, fontweight="bold", pad=12)

        # Compute layout using spring algorithm with fixed seed for aesthetics
        pos = nx.spring_layout(g, seed=42, k=0.15, iterations=40)

        # Node colors and sizes
        if node_map is not None:
            # Color by community block
            node_colors = [palette[node_map[node] % len(palette)] for node in g.nodes()]
            node_sizes = [30 for _ in g.nodes()]
        else:
            # Color and size by degree
            degrees = np.array([d for _, d in g.degree()])
            node_sizes = 20 + 3.0 * degrees
            node_colors = degrees

        # Draw edges with low opacity
        nx.draw_networkx_edges(
            g,
            pos,
            ax=ax,
            alpha=0.25,
            edge_color="#7F8C8D",
            width=0.6,
        )

        # Draw nodes
        nx.draw_networkx_nodes(
            g,
            pos,
            ax=ax,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.viridis if node_map is None else None,
            alpha=0.85,
        )

        ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()


def run_demonstration() -> list[dict[str, float | int | str]]:
    """Run full demonstration, computing metrics and generating visual comparison."""
    ba = generate_barabasi_albert(n=300, m=3, seed=42)
    ws = generate_watts_strogatz(n=300, k=6, p=0.08, seed=42)
    sbm_sizes = [100, 100, 100]
    sbm_prob = [
        [0.080, 0.004, 0.002],
        [0.004, 0.070, 0.004],
        [0.002, 0.004, 0.060],
    ]
    sbm, _ = generate_stochastic_block_model(
        community_sizes=sbm_sizes, probability_matrix=sbm_prob, seed=42
    )

    profiles = [
        compute_discrete_math_profile(ba),
        compute_discrete_math_profile(ws),
        compute_discrete_math_profile(sbm),
    ]

    plot_synthetic_comparison(
        output_path="/Users/kaustavgoel/.gemini/antigravity/scratch/viralgraph/docs/synthetic_topologies_comparison.png"
    )
    return profiles


if __name__ == "__main__":
    results = run_demonstration()
    for res in results:
        print(f"\n--- {res['model_name']} ---")
        for k, v in res.items():
            if k != "model_name":
                print(f"  {k:25s}: {v}")

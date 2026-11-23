"""Interactive network graph visualization component using Plotly and NetworkX layout engines.

Provides responsive 2D visual representation of complex networks with hover metadata,
community color coding, and scale-free hub size adjustments.
"""

from __future__ import annotations

import networkx as nx
import numpy as np
import plotly.graph_objects as go


def render_network_graph(
    graph: nx.Graph,
    model_type: str,
    node_community_map: dict[int, int] | None = None,
    layout_seed: int = 42,
) -> go.Figure:
    """Render an interactive 2D Plotly visualization of a complex network.

    Args:
        graph: NetworkX undirected simple graph.
        model_type: Generative model string ("barabasi_albert", "watts_strogatz", "stochastic_block_model").
        node_community_map: Optional mapping of node_id -> community_id (for SBM).
        layout_seed: Random seed for deterministic spring layout positioning.

    Returns:
        go.Figure: Interactive Plotly figure with zoom, pan, and hover capabilities.
    """
    n_nodes = graph.number_of_nodes()

    # Select optimal layout algorithm based on topology
    if model_type == "watts_strogatz" and n_nodes <= 400:
        # Circular layout preserves the ring geometry with visible rewired chords
        pos = nx.circular_layout(graph)
    else:
        # Fruchterman-Reingold / Spring layout for organic cluster and hub discovery
        # Adjust k (optimal distance between nodes) based on density
        k_param = 1.8 / np.sqrt(max(n_nodes, 1))
        pos = nx.spring_layout(graph, seed=layout_seed, k=k_param, iterations=50)

    # 1. Build Edge Traces
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []

    for u, v in graph.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.6, color="rgba(160, 174, 192, 0.3)"),
        hoverinfo="none",
        mode="lines",
        name="Edges",
        showlegend=False,
    )

    # 2. Build Node Coordinates and Attributes
    node_x: list[float] = []
    node_y: list[float] = []
    node_sizes: list[float] = []
    node_colors: list[float | int | str] = []
    hover_texts: list[str] = []

    degrees = dict(graph.degree())
    clustering = nx.clustering(graph)

    # Palette for discrete communities
    community_palette = [
        "#3B82F6",  # Blue
        "#EF4444",  # Red
        "#10B981",  # Emerald Green
        "#F59E0B",  # Amber
        "#8B5CF6",  # Purple
        "#EC4899",  # Pink
    ]

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        deg = degrees[node]
        clust = clustering[node]

        if model_type == "stochastic_block_model" and node_community_map is not None:
            comm_id = node_community_map.get(node, 0)
            color = community_palette[comm_id % len(community_palette)]
            node_colors.append(color)
            node_sizes.append(9.0 + 1.2 * np.sqrt(deg))
            hover_text = (
                f"<b>Vertex ID:</b> {node}<br>"
                f"<b>Community:</b> Block {comm_id + 1}<br>"
                f"<b>Degree:</b> {deg}<br>"
                f"<b>Clustering Coeff:</b> {clust:.3f}"
            )
        elif model_type == "barabasi_albert":
            # Scale-free: color by continuous degree and size by degree
            node_colors.append(deg)
            # Hub nodes become distinctly larger
            node_sizes.append(7.0 + 2.2 * np.sqrt(deg))
            hover_text = (
                f"<b>Vertex ID:</b> {node}<br>"
                f"<b>Degree:</b> {deg} {'🔥 (Hub)' if deg > 15 else ''}<br>"
                f"<b>Clustering Coeff:</b> {clust:.3f}"
            )
        else:  # watts_strogatz
            # High-clustering small-world ring
            node_colors.append(clust)
            node_sizes.append(8.0 + 1.5 * np.sqrt(deg))
            hover_text = (
                f"<b>Vertex ID:</b> {node}<br>"
                f"<b>Degree:</b> {deg}<br>"
                f"<b>Local Clustering:</b> {clust:.3f}"
            )

        hover_texts.append(hover_text)

    # Node Trace
    marker_config: dict = dict(
        showscale=False,
        size=node_sizes,
        line=dict(width=0.8, color="#FFFFFF"),
        opacity=0.9,
    )

    if model_type == "stochastic_block_model":
        marker_config["color"] = node_colors
    elif model_type == "barabasi_albert":
        marker_config["color"] = node_colors
        marker_config["colorscale"] = "Plasma"
        marker_config["showscale"] = True
        marker_config["colorbar"] = dict(
            title=dict(text="Degree", font=dict(color="#FFFFFF", size=11)),
            thickness=12,
            len=0.7,
            tickcolor="#FFFFFF",
            tickfont=dict(color="#CBD5E1", size=10),
        )
    else:  # watts_strogatz
        marker_config["color"] = node_colors
        marker_config["colorscale"] = "Viridis"
        marker_config["showscale"] = True
        marker_config["colorbar"] = dict(
            title=dict(text="Clustering", font=dict(color="#FFFFFF", size=11)),
            thickness=12,
            len=0.7,
            tickcolor="#FFFFFF",
            tickfont=dict(color="#CBD5E1", size=10),
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=hover_texts,
        marker=marker_config,
        name="Vertices",
    )

    # 3. Assemble Plotly Figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            showlegend=False,
            hovermode="closest",
            margin=dict(b=10, l=10, r=10, t=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            dragmode="pan",
            height=600,
        ),
    )

    return fig

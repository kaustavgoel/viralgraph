"""Graph statistics and degree distribution analysis panel for ViralGraph.

Computes 10 discrete mathematical graph invariants and renders degree distribution plots
with strict mathematical handling for disconnected networks.
"""

from __future__ import annotations

from typing import Any
import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st


def compute_comprehensive_metrics(graph: nx.Graph) -> dict[str, Any]:
    """Compute the 10 discrete mathematical invariants of the network.

    Strictly adheres to graph theory: for disconnected graphs, global distance
    metrics (shortest path length and diameter) are mathematically undefined (infinity).
    In such cases, we compute them on the Giant Connected Component (GCC) and clearly flag it.

    Args:
        graph: NetworkX undirected simple graph.

    Returns:
        dict[str, Any]: Dictionary containing all 10 topological statistics and metadata.
    """
    n = graph.number_of_nodes()
    m = graph.number_of_edges()
    degrees = [deg for _, deg in graph.degree()]

    if n == 0:
        return {"nodes": 0, "edges": 0}

    # Degrees
    min_deg = int(np.min(degrees))
    max_deg = int(np.max(degrees))
    mean_deg = float(np.mean(degrees))
    std_deg = float(np.std(degrees))

    # Handshaking lemma assertion
    assert sum(degrees) == 2 * m, f"Handshaking invariant violated: {sum(degrees)} != {2 * m}"

    # Clustering Coefficient
    avg_clustering = float(nx.average_clustering(graph))

    # Connected Components
    components = list(nx.connected_components(graph))
    num_components = len(components)
    is_connected = (num_components == 1)

    giant_nodes = max(components, key=len)
    giant_size = len(giant_nodes)
    giant_ratio = giant_size / n
    giant_subgraph = graph.subgraph(giant_nodes)

    # Shortest path and diameter
    if is_connected:
        avg_path_length = float(nx.average_shortest_path_length(graph))
        diameter = int(nx.diameter(graph))
        path_metric_scope = "Global (Fully Connected)"
    else:
        # Undefined for whole graph; compute on GCC if GCC has > 1 node
        if giant_size > 1:
            avg_path_length = float(nx.average_shortest_path_length(giant_subgraph))
            diameter = int(nx.diameter(giant_subgraph))
            path_metric_scope = f"GCC only ({giant_size}/{n} nodes, {giant_ratio*100:.1f}%)"
        else:
            avg_path_length = None
            diameter = None
            path_metric_scope = "Undefined (No connected component > 1 node)"

    return {
        "nodes": n,
        "edges": m,
        "mean_degree": mean_deg,
        "min_degree": min_deg,
        "max_degree": max_deg,
        "std_degree": std_deg,
        "avg_clustering": avg_clustering,
        "num_components": num_components,
        "is_connected": is_connected,
        "avg_path_length": avg_path_length,
        "diameter": diameter,
        "path_metric_scope": path_metric_scope,
        "giant_size": giant_size,
        "giant_ratio": giant_ratio,
        "degrees": degrees,
    }


def render_metrics_cards(metrics: dict[str, Any]) -> None:
    """Render top summary metric cards in Streamlit."""
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Vertices (|V|)", f"{metrics['nodes']:,}")
    with col2:
        st.metric("Edges (|E|)", f"{metrics['edges']:,}")
    with col3:
        st.metric("Mean Degree ⟨k⟩", f"{metrics['mean_degree']:.2f}")
    with col4:
        st.metric("Avg Clustering ⟨C⟩", f"{metrics['avg_clustering']:.3f}")
    with col5:
        comp_str = f"{metrics['num_components']}"
        if not metrics["is_connected"]:
            comp_str += " ⚠️ (Disconnected)"
        st.metric("Components |C|", comp_str)


def render_detailed_statistics(metrics: dict[str, Any]) -> None:
    """Render the 10 detailed metrics table and degree distribution chart."""
    st.markdown("### 📊 Discrete Graph Invariants")

    path_display = (
        f"{metrics['avg_path_length']:.3f} hops"
        if metrics["avg_path_length"] is not None
        else "N/A"
    )
    diam_display = (
        f"{metrics['diameter']} hops"
        if metrics["diameter"] is not None
        else "N/A"
    )

    if not metrics["is_connected"]:
        st.warning(
            f"**Note on Disconnected Topology**: The graph contains {metrics['num_components']} disjoint components. "
            f"Global distance metrics are mathematically undefined ($\\infty$). Distance values below are computed on the "
            f"**Giant Connected Component (GCC)** consisting of {metrics['giant_size']}/{metrics['nodes']} vertices "
            f"({metrics['giant_ratio']*100:.1f}% coverage)."
        )

    # Metrics Table
    stats_data = {
        "Metric": [
            "1. Vertices (|V|)",
            "2. Edges (|E|)",
            "3. Average Degree ⟨k⟩",
            "4. Minimum Degree min(k)",
            "5. Maximum Degree max(k)",
            "6. Degree Standard Deviation σ_k",
            "7. Average Clustering ⟨C⟩",
            "8. Connected Components |C|",
            "9. Avg Shortest Path ⟨L⟩",
            "10. Graph Diameter D",
        ],
        "Mathematical Value": [
            f"{metrics['nodes']}",
            f"{metrics['edges']}",
            f"{metrics['mean_degree']:.3f}",
            f"{metrics['min_degree']}",
            f"{metrics['max_degree']}",
            f"{metrics['std_degree']:.3f}",
            f"{metrics['avg_clustering']:.4f}",
            f"{metrics['num_components']}",
            f"{path_display} ({metrics['path_metric_scope']})",
            f"{diam_display} ({metrics['path_metric_scope']})",
        ],
    }

    st.table(stats_data)

    # Degree Distribution Chart
    st.markdown("### 📈 Degree Distribution P(k)")
    degrees = metrics["degrees"]
    unique_degrees, counts = np.unique(degrees, return_counts=True)
    pk = counts / len(degrees)

    use_log_scale = st.checkbox("Log-Log Scale (View Power-Law Tail)", value=False)

    fig = go.Figure()

    # Bar chart for degree frequencies
    fig.add_trace(
        go.Bar(
            x=unique_degrees,
            y=pk,
            marker_color="#3B82F6",
            opacity=0.85,
            name="P(k)",
            hovertemplate="Degree k: %{x}<br>Probability P(k): %{y:.4f}<extra></extra>",
        )
    )

    # Mean degree vertical line
    fig.add_vline(
        x=metrics["mean_degree"],
        line_width=2,
        line_dash="dash",
        line_color="#EF4444",
        annotation_text=f"⟨k⟩ = {metrics['mean_degree']:.2f}",
        annotation_position="top right",
        annotation_font_color="#EF4444",
    )

    fig.update_layout(
        paper_bgcolor="#0e1117",
        plot_bgcolor="#0e1117",
        font=dict(color="#CBD5E1"),
        height=320,
        margin=dict(l=40, r=20, t=30, b=40),
        xaxis=dict(
            title="Degree (k)",
            type="log" if use_log_scale else "linear",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        yaxis=dict(
            title="Frequency P(k)",
            type="log" if use_log_scale else "linear",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)

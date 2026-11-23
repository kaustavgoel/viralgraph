"""Network Explorer page for ViralGraph (Interactive Synthetic Graph Engine & Node Inspection)."""

from __future__ import annotations

from typing import Any
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import numpy as np

from app.components.graph_controls import render_graph_controls
from app.components.graph_visualization import render_network_graph
from app.components.header import render_page_header
from app.components.metrics_panel import (
    compute_comprehensive_metrics,
    render_detailed_statistics,
    render_metrics_cards,
)
from app.state import SessionManager
from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_erdos_renyi,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)
from viralgraph.graph.centrality import compute_centrality_profile


def render_network_explorer_page() -> None:
    """Render interactive Network Explorer page."""
    render_page_header(
        title="Network Explorer",
        subtitle="Synthesize complex synthetic graph topologies, analyze degree distributions, inspect spectral invariants, and drill into individual node profiles.",
        badge="Topological Laboratory",
        category="NETWORK LAB",
    )

    # Render sidebar controls
    controls = render_graph_controls()

    # Handle generation trigger
    if controls.get("generate_triggered", False):
        model = controls["model"]
        params = controls["params"]
        seed = controls["seed"]

        try:
            with st.spinner("Synthesizing complex network topology..."):
                if model == "barabasi_albert":
                    g = generate_barabasi_albert(n=params["n"], m=params["m"], seed=seed)
                    comm_map = None
                elif model == "watts_strogatz":
                    g = generate_watts_strogatz(n=params["n"], k=params["k"], p=params["p"], seed=seed)
                    comm_map = None
                elif model == "stochastic_block_model":
                    g, comm_map = generate_stochastic_block_model(
                        community_sizes=params["community_sizes"],
                        probability_matrix=params["probability_matrix"],
                        seed=seed,
                    )
                elif model == "erdos_renyi":
                    g = generate_erdos_renyi(n=params["n"], p=params["p"], seed=seed)
                    comm_map = None
                else:
                    raise ValueError(f"Unknown graph model: {model}")

                # Update shared session state
                SessionManager.set_current_graph(
                    graph=g,
                    model_type=model,
                    model_label=controls["model_label"],
                    params=params,
                    community_map=comm_map,
                )
                st.toast(f"Generated {controls['model_label']} with {g.number_of_nodes()} nodes!", icon="✅")

        except Exception as exc:
            st.error(f"**Mathematical Generation Error**: {exc}")

    # Active Parameters Banner
    model_label = st.session_state.get("model_label", "Scale-Free (BA)")
    current_params = st.session_state.get("current_params", {})
    param_summary = ", ".join(f"**{k}** = {v}" for k, v in current_params.items() if k != "probability_matrix")
    st.info(f"📍 **Active Topology**: {model_label} | {param_summary}")

    # Top Metric Cards
    metrics = SessionManager.get_metrics()
    if metrics:
        render_metrics_cards(metrics)

    st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)

    # Two-Column Layout: Left = Canvas, Right = Statistics
    col_viz, col_stats = st.columns([7, 5], gap="large")

    graph = SessionManager.get_current_graph()
    model_type = st.session_state.get("model_type", "barabasi_albert")
    comm_map = st.session_state.get("node_community_map")
    seed = current_params.get("seed", 42)

    with col_viz:
        st.markdown("### 🌐 Interactive Topological Canvas")
        st.caption("Drag to pan, scroll to zoom. Hover over vertices to inspect degree and community IDs.")

        fig = render_network_graph(
            graph=graph,
            model_type=model_type,
            node_community_map=comm_map,
            layout_seed=seed,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_stats:
        if metrics:
            render_detailed_statistics(metrics)

    # -------------------------------------------------------------------------
    # INTERACTIVE NODE INSPECTION PANEL
    # -------------------------------------------------------------------------
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 24px 0;' />", unsafe_allow_html=True)
    st.markdown("### 🔍 Vertex Micro-Inspection Panel")
    st.caption("Inspect individual vertex invariants, shortest-path centrality scores, and local neighborhood topology.")

    nodes_list = sorted(list(graph.nodes()))
    if nodes_list:
        # Node selector with quick options
        col_sel, col_quick = st.columns([4, 8], gap="medium")
        with col_sel:
            selected_node = st.selectbox(
                "Select Vertex to Inspect (v)",
                options=nodes_list,
                index=0,
                format_func=lambda n: f"Vertex {n} (Degree: {graph.degree(n)})",
                key="node_inspector_sel",
            )

        with col_quick:
            # Calculate top degree and top betweenness nodes for quick jump
            degrees = dict(graph.degree())
            max_deg_node = max(degrees.keys(), key=lambda k: degrees[k])
            st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)
            col_b1, col_b2, col_b3 = st.columns(3)
            with col_b1:
                if st.button(f"🎯 Highest Hub (v_{max_deg_node})", use_container_width=True):
                    st.session_state["node_inspector_sel"] = max_deg_node
                    st.rerun()

        # Compute or retrieve centrality profile
        if selected_node in graph:
            deg_v = graph.degree(selected_node)
            n_total = graph.number_of_nodes()
            norm_deg = deg_v / (n_total - 1) if n_total > 1 else 0.0
            clust_v = float(nx.clustering(graph, selected_node)) if deg_v >= 2 else 0.0
            comm_id = comm_map.get(selected_node, "N/A") if comm_map else "Single Component"

            # Compute local centralities
            bet_v = float(nx.betweenness_centrality(graph, normalized=True).get(selected_node, 0.0))
            clo_v = float(nx.closeness_centrality(graph).get(selected_node, 0.0))
            bdr_v = float(bet_v / norm_deg) if norm_deg > 0 else 0.0

            # Render node metric badges
            st.markdown(
                f"""
                <div class="vg-card" style="margin-top: 10px; margin-bottom: 16px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.07); padding-bottom: 8px;">
                        <span style="font-size: 1.1rem; font-weight: 700; color: #ffffff;">
                            Vertex Profile: <span style="color: #60a5fa;">v_{selected_node}</span>
                        </span>
                        <span class="vg-badge" style="background: rgba(59, 130, 246, 0.15); color: #60a5fa; border-color: rgba(59, 130, 246, 0.3);">
                            Community: {comm_id}
                        </span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; font-size: 0.88rem;">
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Degree deg(v)</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc;">{deg_v} <span style="font-size: 0.8rem; color: #64748b;">(C_D={norm_deg:.4f})</span></div>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Betweenness C_B(v)</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #34d399;">{bet_v:.5f}</div>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Closeness C_C(v)</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #38bdf8;">{clo_v:.4f}</div>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Clustering Coeff C(v)</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #f59e0b;">{clust_v:.4f}</div>
                        </div>
                        <div style="background: rgba(15, 23, 42, 0.6); padding: 10px; border-radius: 6px;">
                            <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Brokerage Ratio (BDR)</div>
                            <div style="font-size: 1.25rem; font-weight: 700; color: #a78bfa;">{bdr_v:.3f}</div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            # Neighborhood list
            neighbors = sorted(list(graph.neighbors(selected_node)))
            with st.expander(f"Inspect Neighbors of v_{selected_node} ({len(neighbors)} direct adjacent vertices)", expanded=False):
                if neighbors:
                    nb_badges = [f"<span style='background: rgba(255, 255, 255, 0.05); padding: 3px 8px; border-radius: 4px; font-family: monospace; font-size: 0.8rem; margin: 2px;'>v_{u} (d={graph.degree(u)})</span>" for u in neighbors[:40]]
                    st.markdown(" ".join(nb_badges) + (f" ... and {len(neighbors)-40} more" if len(neighbors) > 40 else ""), unsafe_allow_html=True)
                else:
                    st.caption("Vertex is an isolated singleton with zero adjacent edges.")

    # -------------------------------------------------------------------------
    # ANALYTICAL DISTRIBUTION CHARTS
    # -------------------------------------------------------------------------
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 24px 0;' />", unsafe_allow_html=True)
    st.markdown("### 📊 Topological Distribution Analytics")

    col_d1, col_d2 = st.columns(2, gap="large")

    with col_d1:
        st.markdown("#### Degree Distribution P(k)")
        degrees = [d for _, d in graph.degree()]
        if degrees:
            fig_deg = go.Figure()
            fig_deg.add_trace(
                go.Histogram(
                    x=degrees,
                    nbinsx=max(15, len(set(degrees))),
                    marker=dict(color="#3b82f6", line=dict(color="#1d4ed8", width=1)),
                    opacity=0.85,
                    name="Frequency",
                )
            )
            fig_deg.update_layout(
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#151e32",
                margin=dict(l=20, r=20, t=20, b=20),
                height=260,
                xaxis=dict(
                    title=dict(text="Node Degree (k)", font=dict(color="#94a3b8")),
                    tickfont=dict(color="#94a3b8"),
                    gridcolor="rgba(255, 255, 255, 0.05)",
                ),
                yaxis=dict(
                    title=dict(text="Node Count", font=dict(color="#94a3b8")),
                    tickfont=dict(color="#94a3b8"),
                    gridcolor="rgba(255, 255, 255, 0.05)",
                ),
            )
            st.plotly_chart(fig_deg, use_container_width=True)

    with col_d2:
        st.markdown("#### Shortest-Path Betweenness Distribution")
        bet_dict = nx.betweenness_centrality(graph, normalized=True)
        bet_vals = list(bet_dict.values())
        if bet_vals:
            fig_bet = go.Figure()
            fig_bet.add_trace(
                go.Histogram(
                    x=bet_vals,
                    nbinsx=25,
                    marker=dict(color="#10b981", line=dict(color="#047857", width=1)),
                    opacity=0.85,
                    name="Betweenness",
                )
            )
            fig_bet.update_layout(
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#151e32",
                margin=dict(l=20, r=20, t=20, b=20),
                height=260,
                xaxis=dict(
                    title=dict(text="Betweenness Centrality C_B(v)", font=dict(color="#94a3b8")),
                    tickfont=dict(color="#94a3b8"),
                    gridcolor="rgba(255, 255, 255, 0.05)",
                ),
                yaxis=dict(
                    title=dict(text="Node Count", font=dict(color="#94a3b8")),
                    tickfont=dict(color="#94a3b8"),
                    gridcolor="rgba(255, 255, 255, 0.05)",
                ),
            )
            st.plotly_chart(fig_bet, use_container_width=True)

    # -------------------------------------------------------------------------
    # DISCRETE MATHEMATICS FORMULATIONS
    # -------------------------------------------------------------------------
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 24px 0;' />", unsafe_allow_html=True)
    with st.expander("📖 Discrete Mathematics & Theoretical Properties of Active Model"):
        if model_type == "barabasi_albert":
            st.markdown(
                """
                **Barabási–Albert Model (Scale-Free Topologies via Preferential Attachment)**:
                - **Generative Rule**: At each step, a new vertex $v$ attaches to $m$ existing vertices with probability 
                  $$\\Pi(u) = \\frac{\\deg(u)}{\\sum \\deg(w)} = \\frac{\\deg(u)}{2|E|}$$
                - **Asymptotic Degree Distribution**: Follows a power law $P(k) \\sim k^{-3}$ in the thermodynamic limit ($n \\to \\infty$).
                - **Diffusion Significance**: Extremely vulnerable to targeted seed infection (super-hubs act as cascade accelerators), but highly resilient to random failure.
                """
            )
        elif model_type == "watts_strogatz":
            st.markdown(
                """
                **Watts–Strogatz Model (Small-World Networks via Random Rewiring)**:
                - **Generative Rule**: Starts as a 1D ring lattice where each node is connected to $k$ nearest neighbors. Each edge is rewired with probability $p$.
                - **Small-World Regime**: For small $p \\in (0.01, 0.1)$, the network achieves high local clustering $\\langle C \\rangle \\approx 3/4$ alongside ultra-short average path length $\\langle L \\rangle \\sim \\ln n$.
                - **Diffusion Significance**: Simulates corporate communication where information circulates rapidly inside triangles before crossing shortcut bridges.
                """
            )
        elif model_type == "stochastic_block_model":
            st.markdown(
                """
                **Stochastic Block Model (Clustered Communities and Homophily)**:
                - **Generative Rule**: Vertex set $V$ is partitioned into $K$ blocks. Edges occur independently with probability $P_{rs}$ between block $r$ and $s$.
                - **Homophily Regime**: $P_{rr} \\gg P_{rs}$, producing dense intra-cluster connections and sparse inter-cluster bottleneck edges.
                - **Diffusion Significance**: Essential for modeling polarization, filter bubbles, and identifying structural bridge nodes for information intervention.
                """
            )
        else:
            st.markdown(
                """
                **Erdős–Rényi Model (Homogeneous Random Graph Ensemble $G(n, p)$)**:
                - **Generative Rule**: Every pair of vertices $\{u, v\} \\subset V$ is joined by an edge with independent probability $p$.
                - **Degree Distribution**: Follows a binomial distribution $\\text{Binomial}(n-1, p)$, which approaches a Poisson distribution $\\text{Poisson}(\\lambda)$ with $\\lambda = (n-1)p$ as $n \\to \\infty$.
                - **Diffusion Significance**: Represents null-hypothesis baseline topology devoid of preferential attachment or homophily.
                """
            )

"""Interactive Plotly cascade visualizer for Live Diffusion in ViralGraph."""

from __future__ import annotations

from typing import Any
import networkx as nx
import plotly.graph_objects as go

from viralgraph.simulation.independent_cascade import CascadeResult


def render_cascade_canvas(
    graph: nx.Graph,
    cascade_res: CascadeResult,
    current_step: int,
    layout_seed: int = 42,
) -> go.Figure:
    """Render Plotly network canvas reflecting node and edge infection states at discrete step t.

    Node States:
    - Initial Seed (S_0): Gold/Amber (#F59E0B, size 16)
    - Newly Activated at step t (A_t): Bright Emerald (#10B981, size 14)
    - Previously Activated (< t): Vibrant Blue (#3B82F6, size 11)
    - Susceptible / Inactive: Dark Slate (#334155, size 7)

    Transmission Edges:
    - Non-transmission edges: Faint transparent white (rgba(255, 255, 255, 0.05))
    - Active transmission pathways: Bright Cyan (#38BDF8, width 2.5)
    """
    n_nodes = graph.number_of_nodes()
    if n_nodes == 0:
        fig = go.Figure()
        fig.update_layout(
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#0b0f19",
            title=dict(text="Graph has 0 nodes", font=dict(color="#94a3b8")),
        )
        return fig

    # 1. Deterministic Layout Positioning
    if n_nodes > 300:
        pos = nx.spring_layout(graph, seed=layout_seed, k=1.0 / (n_nodes**0.5), iterations=30)
    else:
        pos = nx.spring_layout(graph, seed=layout_seed, k=0.35, iterations=50)

    # 2. Extract activation sets at current step t
    max_history_step = len(cascade_res.activation_history) - 1
    t = min(max(0, current_step), max_history_step)

    seeds = set(cascade_res.seed_nodes)
    cum_active = set().union(*cascade_res.activation_history[: t + 1])
    newly_active_at_t = cascade_res.activation_history[t] if t <= max_history_step else set()

    # Active transmission tree edges activated at or before step t
    active_transmission_edges: set[tuple[Any, Any]] = set()
    tree = cascade_res.transmission_tree
    for u, v in tree.edges():
        if v in cum_active:
            active_transmission_edges.add((u, v))
            active_transmission_edges.add((v, u))

    # 3. Build Edge Coordinates
    bg_edge_x: list[float | None] = []
    bg_edge_y: list[float | None] = []

    tx_edge_x: list[float | None] = []
    tx_edge_y: list[float | None] = []

    for u, v in graph.edges():
        if u not in pos or v not in pos:
            continue
        x0, y0 = pos[u]
        x1, y1 = pos[v]

        if (u, v) in active_transmission_edges or (v, u) in active_transmission_edges:
            tx_edge_x.extend([x0, x1, None])
            tx_edge_y.extend([y0, y1, None])
        else:
            bg_edge_x.extend([x0, x1, None])
            bg_edge_y.extend([y0, y1, None])

    traces: list[go.Scatter] = []

    # Background edges trace
    if bg_edge_x:
        traces.append(
            go.Scatter(
                x=bg_edge_x,
                y=bg_edge_y,
                mode="lines",
                line=dict(width=0.8, color="rgba(255, 255, 255, 0.06)"),
                hoverinfo="none",
                showlegend=False,
            )
        )

    # Active transmission pathways trace
    if tx_edge_x:
        traces.append(
            go.Scatter(
                x=tx_edge_x,
                y=tx_edge_y,
                mode="lines",
                line=dict(width=2.5, color="#38bdf8"),
                name="Transmission Path",
                hoverinfo="none",
                showlegend=True,
            )
        )

    # 4. Partition Nodes into 4 State Categories for Distinct Visual Rendering
    # Categorize each node
    state_groups: dict[str, dict[str, list[Any]]] = {
        "Susceptible": {"x": [], "y": [], "text": []},
        "Previously Active": {"x": [], "y": [], "text": []},
        "Newly Activated": {"x": [], "y": [], "text": []},
        "Seed Node": {"x": [], "y": [], "text": []},
    }

    # Find step of activation for hover text
    node_step_map: dict[Any, int] = {}
    for step_idx, step_nodes in enumerate(cascade_res.activation_history):
        for n in step_nodes:
            if n not in node_step_map:
                node_step_map[n] = step_idx

    for node in graph.nodes():
        if node not in pos:
            continue
        x, y = pos[node]
        deg = graph.degree(node)
        step_act = node_step_map.get(node)

        if node in seeds:
            grp = "Seed Node"
            status_desc = f"Initial Seed (Injected at t=0) | Degree: {deg}"
        elif node in newly_active_at_t and t > 0:
            grp = "Newly Activated"
            status_desc = f"Newly Infected at Step {t}! | Degree: {deg}"
        elif node in cum_active:
            grp = "Previously Active"
            status_desc = f"Active (Infected at Step {step_act}) | Degree: {deg}"
        else:
            grp = "Susceptible"
            status_desc = f"Susceptible (Uninfected) | Degree: {deg}"

        hover_info = f"<b>Vertex {node}</b><br>{status_desc}"
        state_groups[grp]["x"].append(x)
        state_groups[grp]["y"].append(y)
        state_groups[grp]["text"].append(hover_info)

    # Add Node Traces by Group
    styling_map = {
        "Susceptible": dict(color="#334155", size=7, symbol="circle", opacity=0.55),
        "Previously Active": dict(color="#3b82f6", size=11, symbol="circle", opacity=0.9),
        "Newly Activated": dict(color="#10b981", size=14, symbol="circle", opacity=1.0),
        "Seed Node": dict(color="#f59e0b", size=16, symbol="diamond", opacity=1.0),
    }

    for grp_name, grp_data in state_groups.items():
        if grp_data["x"]:
            st_style = styling_map[grp_name]
            traces.append(
                go.Scatter(
                    x=grp_data["x"],
                    y=grp_data["y"],
                    mode="markers",
                    name=f"{grp_name} ({len(grp_data['x'])})",
                    hoverinfo="text",
                    hovertext=grp_data["text"],
                    marker=dict(
                        color=st_style["color"],
                        size=st_style["size"],
                        symbol=st_style["symbol"],
                        opacity=st_style["opacity"],
                        line=dict(width=1.5, color="#ffffff" if grp_name in ["Seed Node", "Newly Activated"] else "#0b0f19"),
                    ),
                    showlegend=True,
                )
            )

    fig = go.Figure(data=traces)
    fig.update_layout(
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#0b0f19",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#94a3b8", size=11),
            bgcolor="rgba(15, 23, 42, 0.7)",
            bordercolor="rgba(255, 255, 255, 0.08)",
            borderwidth=1,
        ),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=520,
    )
    return fig


def render_cascade_timeline_chart(
    cascade_res: CascadeResult,
    current_step: int,
) -> go.Figure:
    """Render cumulative reach and marginal infection velocity curve across timesteps."""
    steps = list(range(len(cascade_res.activation_history)))
    cum_reach = cascade_res.cumulative_reach_timeline
    step_velocity = [len(h) for h in cascade_res.activation_history]

    fig = go.Figure()

    # Cumulative reach line
    fig.add_trace(
        go.Scatter(
            x=steps,
            y=cum_reach,
            mode="lines+markers",
            name="Cumulative Reach |A_<=t|",
            line=dict(color="#3b82f6", width=3),
            marker=dict(size=8, color="#60a5fa"),
        )
    )

    # Step velocity bar chart
    fig.add_trace(
        go.Bar(
            x=steps,
            y=step_velocity,
            name="New Infections |A_t|",
            marker=dict(color="rgba(16, 185, 129, 0.65)"),
            yaxis="y2",
        )
    )

    # Vertical line highlighting current step
    fig.add_vline(
        x=current_step,
        line_width=2,
        line_dash="dash",
        line_color="#f59e0b",
        annotation_text=f"t = {current_step}",
        annotation_position="top right",
        annotation_font=dict(color="#f59e0b", size=11),
    )

    fig.update_layout(
        paper_bgcolor="#0b0f19",
        plot_bgcolor="#151e32",
        margin=dict(l=20, r=20, t=30, b=20),
        height=260,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            font=dict(color="#94a3b8", size=11),
        ),
        xaxis=dict(
            title=dict(text="Discrete Timestep (t)", font=dict(color="#94a3b8")),
            tickfont=dict(color="#94a3b8"),
            gridcolor="rgba(255, 255, 255, 0.05)",
            dtick=1,
        ),
        yaxis=dict(
            title=dict(text="Cumulative Reach", font=dict(color="#60a5fa")),
            tickfont=dict(color="#60a5fa"),
            gridcolor="rgba(255, 255, 255, 0.05)",
        ),
        yaxis2=dict(
            title=dict(text="Velocity (New)", font=dict(color="#34d399")),
            tickfont=dict(color="#34d399"),
            overlaying="y",
            side="right",
            showgrid=False,
        ),
    )
    return fig

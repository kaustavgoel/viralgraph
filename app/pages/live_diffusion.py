"""Live Diffusion page for ViralGraph (Milestone 3).

Provides step-by-step discrete-time cascade playback, active infection frontiers,
transmission tree visualization, and dynamic propagation curves.
"""

from __future__ import annotations

from typing import Any
import streamlit as st
import networkx as nx

from app.components.header import render_page_header
from app.components.cascade_visualizer import render_cascade_canvas, render_cascade_timeline_chart
from app.components.diffusion_launcher import launch_diffusion_simulation, resolve_seed_node
from app.state import SessionManager, SimulationConfig, PostData


def render_live_diffusion_page() -> None:
    """Render interactive Live Diffusion page."""
    render_page_header(
        title="Live Diffusion",
        subtitle="Step-by-step discrete-time cascade playback, active infection frontiers, and transmission pathways.",
        badge="Milestone 3",
        category="SIMULATION STUDIO",
    )

    graph = SessionManager.get_current_graph()
    results = SessionManager.get_simulation_results()
    post = SessionManager.get_current_post()
    config = SessionManager.get_simulation_config()

    # -------------------------------------------------------------------------
    # EMPTY STATE HANDLING
    # -------------------------------------------------------------------------
    if results is None:
        st.markdown(
            """
            <div class="vg-card-elevated" style="text-align: center; padding: 40px 20px;">
                <div style="font-size: 2.2rem; margin-bottom: 12px;">⚡</div>
                <div style="font-size: 1.25rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">
                    No Simulation Trajectory Available
                </div>
                <p style="font-size: 0.92rem; color: #94a3b8; max-width: 580px; margin: 0 auto 24px auto; line-height: 1.6;">
                    Live Diffusion visualizes the temporal evolution of stochastic cascade steps.
                    Launch a diffusion experiment from the Post Studio or generate a quick demonstration run on the active network.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height: 16px;'></div>", unsafe_allow_html=True)
        col_act1, col_act2, col_act3 = st.columns([3, 4, 3])
        with col_act2:
            quick_demo = st.button("🚀 Run Quick Demo Cascade (ICM, p=0.15)", type="primary", use_container_width=True)
            if quick_demo:
                try:
                    seed_node, _ = resolve_seed_node(graph, "highest_degree")
                    demo_config = SimulationConfig(
                        model="ICM",
                        transmission_prob=0.15,
                        seed_strategy="highest_degree",
                        max_steps=20,
                        random_seed=42,
                    )
                    demo_post = PostData(
                        title="Quantum Algorithms in Network Dynamics",
                        content="Simulating discrete information cascading across scale-free topologies.",
                        category="Technology",
                    )
                    comm_map = st.session_state.get("node_community_map")
                    res, record = launch_diffusion_simulation(
                        graph=graph,
                        seed_node=seed_node,
                        config=demo_config,
                        post=demo_post,
                        community_map=comm_map,
                    )
                    SessionManager.set_playback_step(0)
                    st.session_state["live_diff_slider"] = 0
                    st.rerun()
                except Exception as exc:
                    st.error(f"Error launching demo: {exc}")

            if st.button("📝 Open Post Studio", use_container_width=True):
                try:
                    st.switch_page("post_studio")
                except Exception:
                    pass
        return

    # -------------------------------------------------------------------------
    # ACTIVE CASCADE SUMMARY CARD
    # -------------------------------------------------------------------------
    max_step = len(results.activation_history) - 1
    total_nodes = graph.number_of_nodes()
    total_reach = results.total_reach
    reach_pct = (total_reach / total_nodes) * 100.0 if total_nodes > 0 else 0.0
    seed_node = results.seed_nodes[0] if results.seed_nodes else "N/A"
    model_label = "Independent Cascade (ICM)" if config.model == "ICM" else "Linear Threshold (LTM)"

    st.markdown(
        f"""
        <div class="vg-card" style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px; border-bottom: 1px solid rgba(255, 255, 255, 0.07); padding-bottom: 10px;">
                <div>
                    <span style="font-size: 0.8rem; text-transform: uppercase; color: #94a3b8; letter-spacing: 0.05em;">Active Experiment Payload</span>
                    <div style="font-size: 1.15rem; font-weight: 700; color: #ffffff;">{post.title}</div>
                </div>
                <div style="text-align: right;">
                    <span class="vg-badge" style="background: rgba(59, 130, 246, 0.15); color: #60a5fa; border-color: rgba(59, 130, 246, 0.3);">
                        {model_label}
                    </span>
                </div>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; font-size: 0.88rem;">
                <div>Initial Seed Node: <b style="color: #f59e0b;">Vertex {seed_node}</b></div>
                <div>Network Order: <b style="color: #cbd5e1;">|V| = {total_nodes}</b></div>
                <div>Network Size: <b style="color: #cbd5e1;">|E| = {graph.number_of_edges()}</b></div>
                <div>Final Reach: <b style="color: #34d399;">{total_reach} nodes ({reach_pct:.1f}%)</b></div>
                <div>Duration: <b style="color: #38bdf8;">{results.steps_to_termination} discrete steps</b></div>
                <div>Transmission Edges: <b style="color: #a78bfa;">{results.transmission_tree.number_of_edges()} pathways</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # PLAYBACK TIMESTEP CONTROLLER
    # -------------------------------------------------------------------------
    # Ensure slider state is synchronized
    if "live_diff_slider" not in st.session_state:
        st.session_state["live_diff_slider"] = min(SessionManager.get_playback_step(), max_step)
    elif st.session_state["live_diff_slider"] > max_step:
        st.session_state["live_diff_slider"] = max_step

    # Control buttons handler
    col_c1, col_c2, col_c3, col_c4, col_c5 = st.columns([2, 2, 2, 2, 4], gap="small")
    with col_c1:
        if st.button("⏮ Beginning", use_container_width=True, disabled=(st.session_state["live_diff_slider"] == 0)):
            st.session_state["live_diff_slider"] = 0
            SessionManager.set_playback_step(0)
            st.rerun()
    with col_c2:
        if st.button("◀ Step Back", use_container_width=True, disabled=(st.session_state["live_diff_slider"] == 0)):
            st.session_state["live_diff_slider"] = max(0, st.session_state["live_diff_slider"] - 1)
            SessionManager.set_playback_step(st.session_state["live_diff_slider"])
            st.rerun()
    with col_c3:
        if st.button("▶ Step Next", use_container_width=True, disabled=(st.session_state["live_diff_slider"] >= max_step)):
            st.session_state["live_diff_slider"] = min(max_step, st.session_state["live_diff_slider"] + 1)
            SessionManager.set_playback_step(st.session_state["live_diff_slider"])
            st.rerun()
    with col_c4:
        if st.button("⏭ Termination", use_container_width=True, disabled=(st.session_state["live_diff_slider"] >= max_step)):
            st.session_state["live_diff_slider"] = max_step
            SessionManager.set_playback_step(max_step)
            st.rerun()
    with col_c5:
        # Re-run simulation with incremented random seed
        if st.button("🎲 Re-run with New Random Seed", use_container_width=True):
            try:
                new_seed = config.random_seed + 1
                config.random_seed = new_seed
                comm_map = st.session_state.get("node_community_map")
                res, record = launch_diffusion_simulation(
                    graph=graph,
                    seed_node=seed_node,
                    config=config,
                    post=post,
                    community_map=comm_map,
                )
                st.session_state["live_diff_slider"] = 0
                SessionManager.set_playback_step(0)
                st.rerun()
            except Exception as exc:
                st.error(f"Re-run failed: {exc}")

    # Slider control
    current_step = st.slider(
        "Discrete Cascade Epoch (t)",
        min_value=0,
        max_value=max_step,
        value=st.session_state["live_diff_slider"],
        key="live_diff_slider",
        help="Scrub through discrete simulation epochs. Visualizes node states and transmission DAG at step t.",
    )
    SessionManager.set_playback_step(current_step)

    # -------------------------------------------------------------------------
    # STEP METRIC CARDS
    # -------------------------------------------------------------------------
    cum_active_set = set().union(*results.activation_history[: current_step + 1])
    newly_active_set = results.activation_history[current_step]
    cum_reach_at_t = len(cum_active_set)
    pct_reach_at_t = (cum_reach_at_t / total_nodes) * 100.0 if total_nodes > 0 else 0.0
    susceptible_at_t = total_nodes - cum_reach_at_t

    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 18px;">
            <div style="background: rgba(15, 23, 42, 0.7); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.06);">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Current Epoch</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #ffffff;">
                    t = {current_step} <span style="font-size: 0.85rem; font-weight: 500; color: #64748b;">/ {max_step}</span>
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.7); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.06);">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Cumulative Reach |A_≤t|</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #60a5fa;">
                    {cum_reach_at_t} <span style="font-size: 0.85rem; font-weight: 500; color: #94a3b8;">({pct_reach_at_t:.1f}%)</span>
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.7); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.06);">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Newly Infected |A_t|</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #34d399;">
                    +{len(newly_active_set)} <span style="font-size: 0.85rem; font-weight: 500; color: #94a3b8;">nodes</span>
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.7); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.06);">
                <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Susceptible Nodes</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #94a3b8;">
                    {susceptible_at_t} <span style="font-size: 0.85rem; font-weight: 500; color: #64748b;">nodes</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # INTERACTIVE NETWORK CANVAS & TIMELINE CHARTS
    # -------------------------------------------------------------------------
    col_vis, col_side = st.columns([8, 4], gap="medium")

    with col_vis:
        st.markdown(
            f"""
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                <span style="font-size: 0.95rem; font-weight: 700; color: #f8fafc;">
                    🌐 Network Infection State at Epoch t = {current_step}
                </span>
                <span style="font-size: 0.78rem; color: #94a3b8;">
                    Hover over vertices for infection details
                </span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        canvas_fig = render_cascade_canvas(
            graph=graph,
            cascade_res=results,
            current_step=current_step,
            layout_seed=42,
        )
        st.plotly_chart(canvas_fig, use_container_width=True)

        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-top: 14px; margin-bottom: 6px;">
                📈 Cascade Trajectory & Marginal Velocity Curve
            </div>
            """,
            unsafe_allow_html=True,
        )
        timeline_fig = render_cascade_timeline_chart(
            cascade_res=results,
            current_step=current_step,
        )
        st.plotly_chart(timeline_fig, use_container_width=True)

    with col_side:
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-bottom: 8px;">
                🎯 Epoch Event Log
            </div>
            """,
            unsafe_allow_html=True,
        )

        if current_step == 0:
            st.markdown(
                f"""
                <div class="vg-card" style="padding: 12px; margin-bottom: 12px;">
                    <div style="font-size: 0.82rem; font-weight: 700; color: #f59e0b; margin-bottom: 4px;">
                        Epoch t = 0: Seed Injection
                    </div>
                    <p style="font-size: 0.82rem; color: #cbd5e1; margin: 0; line-height: 1.5;">
                        Contagion injected into initial seed vertex <b>{seed_node}</b> (Degree = {graph.degree(seed_node)}).
                        All neighbors are now exposed to transmission attempts in epoch t=1.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            if newly_active_set:
                nodes_sorted = sorted(list(newly_active_set), key=lambda u: graph.degree(u), reverse=True)
                sample_nodes = nodes_sorted[:12]
                st.markdown(
                    f"""
                    <div class="vg-card" style="padding: 12px; margin-bottom: 12px;">
                        <div style="font-size: 0.82rem; font-weight: 700; color: #34d399; margin-bottom: 6px;">
                            Epoch t = {current_step}: {len(newly_active_set)} Vertices Infected
                        </div>
                        <div style="display: flex; flex-wrap: wrap; gap: 6px;">
                            {''.join([f'<span style="background: rgba(16, 185, 129, 0.2); color: #34d399; padding: 2px 8px; border-radius: 4px; font-size: 0.78rem; font-family: monospace;">v_{n} (d={graph.degree(n)})</span>' for n in sample_nodes])}
                            {f'<span style="color: #94a3b8; font-size: 0.78rem;">+{len(nodes_sorted) - 12} more</span>' if len(nodes_sorted) > 12 else ''}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"""
                    <div class="vg-card" style="padding: 12px; margin-bottom: 12px;">
                        <div style="font-size: 0.82rem; font-weight: 700; color: #94a3b8; margin-bottom: 4px;">
                            Epoch t = {current_step}: Quiescence
                        </div>
                        <p style="font-size: 0.82rem; color: #64748b; margin: 0; line-height: 1.5;">
                            No new vertices were activated in this epoch. The cascade frontier has ceased expanding.
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        # Transmission Pathway Details
        st.markdown(
            """
            <div style="font-size: 0.95rem; font-weight: 700; color: #f8fafc; margin-top: 14px; margin-bottom: 8px;">
                🌳 Transmission Directed Graph
            </div>
            """,
            unsafe_allow_html=True,
        )

        tree = results.transmission_tree
        edges_in_tree = list(tree.edges())
        st.markdown(
            f"""
            <div class="vg-card" style="font-size: 0.84rem; color: #cbd5e1; line-height: 1.6;">
                <div>Directed Arcs: <b>{tree.number_of_edges()}</b></div>
                <div>Tree Depth: <b>{results.steps_to_termination}</b></div>
                <div>Branching Factor: <b>{tree.number_of_edges() / max(1, len(results.seed_nodes)):.2f}</b></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if edges_in_tree:
            st.markdown("<div style='height: 8px;'></div>", unsafe_allow_html=True)
            with st.expander("View Active Transmission Arcs (u → v)", expanded=False):
                edge_text_list = [f"`v_{u}` ➔ `v_{v}`" for u, v in edges_in_tree[:30]]
                st.markdown(", ".join(edge_text_list) + (f" ... and {len(edges_in_tree)-30} more" if len(edges_in_tree) > 30 else ""))

        # Jump to Post Studio or Influencer Research
        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        col_nav1, col_nav2 = st.columns(2)
        with col_nav1:
            if st.button("📝 Post Studio", use_container_width=True):
                try:
                    st.switch_page("post_studio")
                except Exception:
                    pass
        with col_nav2:
            if st.button("🌟 Influencers", use_container_width=True):
                try:
                    st.switch_page("influencer_research")
                except Exception:
                    pass

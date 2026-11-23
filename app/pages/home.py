"""Home / Overview landing page for ViralGraph."""

from __future__ import annotations

import streamlit as st
from app.state import SessionManager


def render_home_page() -> None:
    """Render the primary ViralGraph Home landing page."""
    # 1. Hero Banner with Welcome Heading & Brand
    st.markdown(
        """
        <div class="vg-hero">
            <div class="vg-breadcrumb">DISCRETE MATHEMATICS & NETWORK INTELLIGENCE</div>
            <h1>Explore how ideas spread.</h1>
            <p>
                ViralGraph is an advanced research and simulation platform that studies, simulates,
                and explains how information travels through complex social structures. Rather than treating
                virality as an isolated property of content, ViralGraph demonstrates how network geometry,
                high-degree hubs, and community bridges govern the dynamics of contagion.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 2. Prominent Primary Call-to-Action
    col_cta1, col_cta2, col_cta3 = st.columns([3, 3, 4], gap="medium")
    with col_cta1:
        if st.button("🚀 Start a Simulation", type="primary", use_container_width=True):
            try:
                st.switch_page("post-studio")
            except Exception:
                st.info("Navigate to **Post Studio** in the sidebar to configure your first simulation.")
    with col_cta2:
        if st.button("🌐 Explore Network Topologies", use_container_width=True):
            try:
                st.switch_page("network-explorer")
            except Exception:
                st.info("Navigate to **Network Explorer** in the sidebar.")
    with col_cta3:
        if st.button("🔬 View Research Hypotheses", use_container_width=True):
            try:
                st.switch_page("influencer-research")
            except Exception:
                st.info("Navigate to **Influencer Research** in the sidebar.")

    st.markdown("<div style='height: 24px;'></div>", unsafe_allow_html=True)

    # 3. Quick-Access Workflow Cards
    st.markdown(
        """
        <div class="vg-section-header">
            <span class="vg-section-title">Quick-Access Workflows</span>
            <span style="font-size: 0.85rem; color: #94a3b8;">End-to-End Contagion Pipeline</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_w1, col_w2, col_w3 = st.columns(3, gap="medium")

    with col_w1:
        st.markdown(
            """
            <div class="vg-card-interactive">
                <div style="font-size: 1.5rem; margin-bottom: 8px;">🌐</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #f1f5f9; margin-bottom: 6px;">
                    Network Explorer
                </div>
                <p style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5; margin-bottom: 16px;">
                    Synthesize complex synthetic topologies (Barabási–Albert, Watts–Strogatz, SBM).
                    Inspect degree spectra, algebraic connectivity, and interactive Plotly graph canvases.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Launch Explorer", key="home_btn_explorer", use_container_width=True):
            try:
                st.switch_page("network-explorer")
            except Exception:
                pass

    with col_w2:
        st.markdown(
            """
            <div class="vg-card-interactive">
                <div style="font-size: 1.5rem; margin-bottom: 8px;">✍️</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #f1f5f9; margin-bottom: 6px;">
                    Post Studio
                </div>
                <p style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5; margin-bottom: 16px;">
                    Draft simulated social posts, configure topic tags and virality factors,
                    and target initial seed influencers (degree hubs, bridges, or custom vertices).
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Post Studio", key="home_btn_studio", use_container_width=True):
            try:
                st.switch_page("post-studio")
            except Exception:
                pass

    with col_w3:
        st.markdown(
            """
            <div class="vg-card-interactive">
                <div style="font-size: 1.5rem; margin-bottom: 8px;">🔬</div>
                <div style="font-size: 1.05rem; font-weight: 700; color: #f1f5f9; margin-bottom: 6px;">
                    Influencer Research
                </div>
                <p style="font-size: 0.88rem; color: #94a3b8; line-height: 1.5; margin-bottom: 16px;">
                    Test whether community bridges escape echo chambers better than high-degree hubs.
                    Backed by small-sample Student's t-distributions and paired hypothesis tests.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button("Open Research Lab", key="home_btn_research", use_container_width=True):
            try:
                st.switch_page("influencer-research")
            except Exception:
                pass

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 4. Project Capability Overview (4 Core Pillars)
    st.markdown(
        """
        <div class="vg-section-header">
            <span class="vg-section-title">Core Platform Capabilities</span>
            <span style="font-size: 0.85rem; color: #94a3b8;">Mathematical Rigor & Scalable Engineering</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    cap_col1, cap_col2 = st.columns(2, gap="medium")

    with cap_col1:
        st.markdown(
            """
            <div class="vg-card">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 1.3rem;">📐</span>
                    <span style="font-size: 1.0rem; font-weight: 700; color: #f8fafc;">1. Synthetic Graph Engine</span>
                </div>
                <p style="font-size: 0.86rem; color: #94a3b8; line-height: 1.5; margin: 0;">
                    Construct mathematically verified graph generators with full parameter controls:
                    Barabási–Albert (Scale-Free preferential attachment), Watts–Strogatz (Small-World rewired lattice),
                    and Stochastic Block Model (Clustered homophily communities).
                </p>
            </div>
            <div class="vg-card">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 1.3rem;">⚡</span>
                    <span style="font-size: 1.0rem; font-weight: 700; color: #f8fafc;">2. Stochastic Diffusion Dynamics</span>
                </div>
                <p style="font-size: 0.86rem; color: #94a3b8; line-height: 1.5; margin: 0;">
                    Simulate discrete-time Markovian contagions via the Independent Cascade Model (ICM)
                    and Linear Threshold Model (LTM) with exact transmission tree reconstruction
                    and Monte Carlo expectation bounds.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cap_col2:
        st.markdown(
            """
            <div class="vg-card">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 1.3rem;">🔍</span>
                    <span style="font-size: 1.0rem; font-weight: 700; color: #f8fafc;">3. Topological Intelligence & Centralities</span>
                </div>
                <p style="font-size: 0.86rem; color: #94a3b8; line-height: 1.5; margin: 0;">
                    Compute full node centrality spectra (Degree, Betweenness, Closeness, Eigenvector, PageRank),
                    Betweenness-to-Degree Ratios (BDR) to isolate structural bridges, and Louvain community partitions.
                </p>
            </div>
            <div class="vg-card">
                <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
                    <span style="font-size: 1.3rem;">📈</span>
                    <span style="font-size: 1.0rem; font-weight: 700; color: #f8fafc;">4. Out-of-Graph Machine Learning</span>
                </div>
                <p style="font-size: 0.86rem; color: #94a3b8; line-height: 1.5; margin: 0;">
                    Forecast cascade virality from early observation windows (t ≤ 1) using leakage-free
                    inner-fold thresholding and GroupKFold cross-validation evaluated across entirely unseen graph topologies.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # 5. Recent Simulations Section (With Authentic Empty State)
    st.markdown(
        """
        <div class="vg-section-header">
            <span class="vg-section-title">Recent Cascade Simulations</span>
            <span style="font-size: 0.85rem; color: #94a3b8;">Active Session Feed</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    history = SessionManager.get_simulation_history()

    if not history:
        # Authentic empty state — No fabricated history
        st.markdown(
            """
            <div class="vg-empty-state">
                <div class="vg-empty-icon">📊</div>
                <div class="vg-empty-title">No simulations executed in this session</div>
                <div class="vg-empty-desc">
                    Cascade runs, propagation timelines, and infection penetration metrics
                    will appear here automatically once you launch a simulation.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        col_c1, col_c2, col_c3 = st.columns([4, 4, 4])
        with col_c2:
            if st.button("Start First Simulation", key="home_btn_first_sim", type="primary", use_container_width=True):
                try:
                    st.switch_page("post-studio")
                except Exception:
                    st.info("Navigate to **Post Studio** in the sidebar to configure a simulation.")
    else:
        # Display authentic completed simulations
        st.write(f"Showing **{len(history)}** completed simulation(s) in this session:")
        for rec in history[:5]:
            st.markdown(
                f"""
                <div class="vg-card" style="padding: 14px 18px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <div>
                            <b>{rec.sim_id}</b> — {rec.topology_name} ({rec.model_name})
                            <div style="font-size: 0.82rem; color: #94a3b8;">
                                Seed: Node {rec.seed_node} | Reached: <b>{rec.total_reach}</b> nodes ({rec.steps_taken} steps)
                            </div>
                        </div>
                        <span class="vg-badge">{rec.timestamp}</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

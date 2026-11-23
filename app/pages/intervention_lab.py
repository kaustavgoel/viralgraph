"""Intervention Lab page for ViralGraph (Structural Countermeasures & What-If Studio)."""

from __future__ import annotations

from typing import Any
import streamlit as st
import networkx as nx
import plotly.graph_objects as go
import pandas as pd
import numpy as np

from app.components.header import render_page_header
from app.state import SessionManager
from viralgraph.simulation.intervention_engine import (
    apply_node_immunization,
    apply_community_quarantine,
    evaluate_intervention_impact,
    InterventionComparison,
)
from viralgraph.simulation.monte_carlo import run_monte_carlo_simulation
from viralgraph.graph.representation import GraphRepresentation


def render_intervention_lab_page() -> None:
    """Render interactive Intervention Lab page."""
    render_page_header(
        title="Intervention Lab",
        subtitle="Simulate structural countermeasures: targeted node immunization, community quarantine, and network resilience analytics.",
        badge="Countermeasure Studio",
        category="NETWORK LAB",
    )

    graph = SessionManager.get_current_graph()
    comm_map = st.session_state.get("node_community_map")
    n_nodes = graph.number_of_nodes()

    if n_nodes < 5:
        st.warning("Active graph must have at least 5 vertices to conduct structural intervention experiments.")
        return

    # -------------------------------------------------------------------------
    # HERO OVERVIEW
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="vg-hero" style="padding: 22px 26px; margin-bottom: 20px;">
            <div class="vg-breadcrumb">COUNTERFACTUAL RESILIENCE EXPERIMENTATION</div>
            <h2 style="color: #ffffff; font-size: 1.45rem; font-weight: 700; margin: 0 0 8px 0;">
                "How effectively can structural interventions suppress viral cascades?"
            </h2>
            <p style="margin: 0; color: #cbd5e1; font-size: 0.92rem; line-height: 1.6;">
                Evaluate targeted immunization and quarantine policies without mutating your active graph.
                Quantify cascade suppression $\\Delta R$, containment efficiency, Giant Connected Component (GCC) reduction,
                and algebraic connectivity degradation $\\lambda_2(L)$.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # INTERVENTION CONFIGURATION PANEL
    # -------------------------------------------------------------------------
    st.markdown("### 🛠️ Countermeasure Configuration")

    col_cfg1, col_cfg2, col_cfg3 = st.columns([4, 4, 4], gap="medium")

    with col_cfg1:
        strategy_options = {
            "Targeted Hub Inoculation (Top-k Degree)": "targeted_hub",
            "Targeted Bridge Inoculation (Top-k Betweenness)": "targeted_bridge",
            "Random Inoculation (Uniform Baseline)": "random",
        }
        strat_label = st.selectbox(
            "Intervention Policy",
            options=list(strategy_options.keys()),
            index=0,
            help="Select which topological criteria determines the vertices to immunize.",
        )
        strategy = strategy_options[strat_label]

    with col_cfg2:
        budget = st.slider(
            "Immunization Budget (k vertices)",
            min_value=1,
            max_value=min(20, n_nodes - 2),
            value=min(5, max(1, n_nodes // 10)),
            step=1,
            help="Number of vertices to completely immunize/quarantine from the transmission graph.",
        )

    with col_cfg3:
        p_trans = st.slider(
            "Cascade Transmission Prob (p)",
            min_value=0.05,
            max_value=0.35,
            value=0.15,
            step=0.01,
            help="Edge infection probability in the Independent Cascade Model (ICM).",
        )

    col_btn, col_info = st.columns([4, 8], gap="medium")
    with col_btn:
        run_intervention = st.button("🛡️ Evaluate Structural Intervention", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # INTERVENTION EVALUATION EXECUTION
    # -------------------------------------------------------------------------
    interv_key = f"interv_{strategy}_{budget}_{p_trans}_{n_nodes}"
    
    if run_intervention or interv_key not in st.session_state:
        with st.spinner("Executing counterfactual intervention and Monte Carlo evaluations..."):
            try:
                comp = evaluate_intervention_impact(
                    graph=graph,
                    intervention_type=strategy,
                    budget=budget,
                    p_transmission=p_trans,
                    mc_simulations=30,
                    seed=42,
                )
                st.session_state[interv_key] = comp
            except Exception as exc:
                st.error(f"Intervention evaluation error: {exc}")
                return

    comp: InterventionComparison = st.session_state.get(interv_key)
    if comp is None:
        return

    # -------------------------------------------------------------------------
    # IMPACT SCORECARDS
    # -------------------------------------------------------------------------
    st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Counterfactual Impact Assessment")

    abs_reduction = comp.baseline_expected_reach - comp.intervened_expected_reach

    st.markdown(
        f"""
        <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 20px;">
            <div style="background: rgba(15, 23, 42, 0.7); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.06);">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Reach Suppression (ΔR)</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #34d399;">
                    -{comp.reach_reduction_percentage:.1f}%
                </div>
                <div style="font-size: 0.78rem; color: #64748b; margin-top: 2px;">
                    {comp.baseline_expected_reach:.1f} → {comp.intervened_expected_reach:.1f} nodes
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.7); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.06);">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Containment Efficiency</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #60a5fa;">
                    {comp.containment_efficiency:.2f}
                </div>
                <div style="font-size: 0.78rem; color: #64748b; margin-top: 2px;">
                    Infections prevented per immunized node
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.7); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.06);">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Edges Severed</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #f59e0b;">
                    {comp.num_edges_removed}
                </div>
                <div style="font-size: 0.78rem; color: #64748b; margin-top: 2px;">
                    Removed from transmission graph
                </div>
            </div>
            <div style="background: rgba(15, 23, 42, 0.7); padding: 12px 14px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.06);">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase;">Algebraic Connectivity λ_2</div>
                <div style="font-size: 1.4rem; font-weight: 800; color: #a78bfa;">
                    {comp.intervened_algebraic_connectivity:.4f}
                </div>
                <div style="font-size: 0.78rem; color: #64748b; margin-top: 2px;">
                    Baseline: {comp.baseline_algebraic_connectivity:.4f}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # COMPARATIVE VISUALIZATION: BASELINE VS. INTERVENED
    # -------------------------------------------------------------------------
    col_v1, col_v2 = st.columns([6, 6], gap="large")

    with col_v1:
        st.markdown("#### Cascade Reach: Baseline vs. Post-Intervention")
        fig_cmp = go.Figure()
        fig_cmp.add_trace(
            go.Bar(
                x=["Baseline (Unprotected)", f"Intervened (k={budget})"],
                y=[comp.baseline_expected_reach, comp.intervened_expected_reach],
                marker=dict(color=["#ef4444", "#10b981"]),
                text=[f"{comp.baseline_expected_reach:.1f} nodes", f"{comp.intervened_expected_reach:.1f} nodes"],
                textposition="auto",
            )
        )
        fig_cmp.update_layout(
            paper_bgcolor="#0b0f19",
            plot_bgcolor="#151e32",
            margin=dict(l=20, r=20, t=20, b=20),
            height=260,
            yaxis=dict(title=dict(text="Expected Cascade Reach", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8"), gridcolor="rgba(255, 255, 255, 0.05)"),
            xaxis=dict(tickfont=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig_cmp, use_container_width=True)

    with col_v2:
        st.markdown("#### Network Integrity Comparison")
        st.markdown(
            f"""
            <div class="vg-card" style="font-size: 0.88rem; line-height: 1.8;">
                <div style="display: flex; justify-content: space-between; border-bottom: 1px solid rgba(255, 255, 255, 0.06); padding-bottom: 6px; margin-bottom: 6px;">
                    <span style="color: #94a3b8;">Metric</span>
                    <span style="color: #ef4444; font-weight: 600;">Baseline</span>
                    <span style="color: #10b981; font-weight: 600;">Intervened</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #cbd5e1;">Graph Order |V|</span>
                    <span style="color: #f8fafc;">{n_nodes}</span>
                    <span style="color: #34d399;">{n_nodes - comp.num_nodes_removed}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #cbd5e1;">Graph Size |E|</span>
                    <span style="color: #f8fafc;">{graph.number_of_edges()}</span>
                    <span style="color: #34d399;">{graph.number_of_edges() - comp.num_edges_removed}</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #cbd5e1;">Giant Component Ratio (GCC)</span>
                    <span style="color: #f8fafc;">{comp.baseline_gcc_ratio*100:.1f}%</span>
                    <span style="color: #34d399;">{comp.intervened_gcc_ratio*100:.1f}%</span>
                </div>
                <div style="display: flex; justify-content: space-between;">
                    <span style="color: #cbd5e1;">Algebraic Connectivity λ_2(L)</span>
                    <span style="color: #f8fafc;">{comp.baseline_algebraic_connectivity:.4f}</span>
                    <span style="color: #34d399;">{comp.intervened_algebraic_connectivity:.4f}</span>
                </div>
                <div style="display: flex; justify-content: space-between; border-top: 1px solid rgba(255, 255, 255, 0.06); padding-top: 6px; margin-top: 6px;">
                    <span style="color: #cbd5e1; font-weight: 700;">Net Reach Suppression</span>
                    <span colspan="2" style="color: #34d399; font-weight: 800;">-{comp.reach_reduction_percentage:.1f}%</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # -------------------------------------------------------------------------
    # POLICY COMPARISON: HUBS VS BRIDGES VS RANDOM
    # -------------------------------------------------------------------------
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 24px 0;' />", unsafe_allow_html=True)
    st.markdown("### ⚖️ Multi-Policy Containment Efficiency")
    st.caption("How do different immunization policies compare at identical budget allocations?")

    sweep_btn = st.button("📈 Run Multi-Policy Comparative Benchmark (k = 1, 3, 5, 8)", use_container_width=False)
    
    sweep_key = f"sweep_{n_nodes}_{graph.number_of_edges()}"
    if sweep_btn or sweep_key in st.session_state:
        if sweep_key not in st.session_state:
            with st.spinner("Benchmarking multi-policy immunization curves across budget budgets..."):
                budgets = [1, 3, 5, min(8, n_nodes - 2)]
                sweep_records = []
                for b_val in budgets:
                    for pol, pol_name in [("targeted_hub", "Targeted Hub"), ("targeted_bridge", "Targeted Bridge"), ("random", "Random")]:
                        try:
                            c = evaluate_intervention_impact(
                                graph=graph,
                                intervention_type=pol,
                                budget=b_val,
                                p_transmission=p_trans,
                                mc_simulations=20,
                                seed=42,
                            )
                            sweep_records.append({
                                "Budget": b_val,
                                "Policy": pol_name,
                                "Reach Suppression (%)": c.reach_reduction_percentage,
                                "Efficiency": c.containment_efficiency,
                            })
                        except Exception:
                            pass
                st.session_state[sweep_key] = sweep_records

        sweep_data = st.session_state.get(sweep_key, [])
        if sweep_data:
            df_sweep = pd.DataFrame(sweep_data)
            fig_sw = go.Figure()
            for pol_name, col in [("Targeted Hub", "#ef4444"), ("Targeted Bridge", "#10b981"), ("Random", "#64748b")]:
                sub = df_sweep[df_sweep["Policy"] == pol_name]
                if not sub.empty:
                    fig_sw.add_trace(
                        go.Scatter(
                            x=sub["Budget"],
                            y=sub["Reach Suppression (%)"],
                            mode="lines+markers",
                            name=pol_name,
                            line=dict(color=col, width=2.5),
                            marker=dict(size=8),
                        )
                    )
            fig_sw.update_layout(
                paper_bgcolor="#0b0f19",
                plot_bgcolor="#151e32",
                margin=dict(l=20, r=20, t=20, b=20),
                height=260,
                xaxis=dict(title=dict(text="Immunization Budget (k)", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8"), gridcolor="rgba(255, 255, 255, 0.05)", dtick=1),
                yaxis=dict(title=dict(text="Reach Suppression (%)", font=dict(color="#94a3b8")), tickfont=dict(color="#94a3b8"), gridcolor="rgba(255, 255, 255, 0.05)"),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(color="#94a3b8")),
            )
            st.plotly_chart(fig_sw, use_container_width=True)

    # -------------------------------------------------------------------------
    # DISCRETE MATHEMATICS & PERCOLATION THEORY
    # -------------------------------------------------------------------------
    st.markdown("<hr style='border: none; border-top: 1px solid rgba(255, 255, 255, 0.08); margin: 24px 0;' />", unsafe_allow_html=True)
    with st.expander("📖 Percolation Theory & Submodular Containment Proofs"):
        st.markdown(
            """
            **Theoretical Foundations of Network Inoculation (Albert, Jeong & Barabási 2000; Callaway et al. 2000)**:
            - **Percolation Threshold on Scale-Free Topologies**: For scale-free networks with degree exponent $\\gamma \\le 3$,
              the critical percolation threshold under random node removal vanishes in the infinite graph limit:
              $$f_c = 1 - \\frac{1}{\\frac{\\langle k^2 \\rangle}{\\langle k \\rangle} - 1} \\to 1$$
              Scale-free networks are **infinitely resilient to random failure**.
            - **Targeted Hub Vulnerability**: Under targeted removal of vertices ordered by degree, the critical threshold
              drops drastically:
              $$f_c^{\\text{targeted}} \\sim n^{-(3-\\gamma)/(\\gamma-1)} \\ll 1$$
              Removing a tiny fraction (often $< 3\\%$) of top-degree hubs disintegrates the Giant Connected Component (GCC).
            - **Submodularity of Cascade Spread**: The influence spread function $\\sigma(S)$ is submodular:
              $$\\sigma(S \\cup \\{v\\}) - \\sigma(S) \\ge \\sigma(T \\cup \\{v\\}) - \\sigma(T) \\quad \\forall S \\subseteq T$$
              This guarantees that the marginal benefit of each additional immunized node exhibits diminishing returns.
            """
        )

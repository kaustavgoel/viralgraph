"""Shared sidebar component for ViralGraph."""

from __future__ import annotations

import streamlit as st
from app.state import SessionManager


def render_shared_sidebar() -> None:
    """Render shared sidebar elements, status badges, and active graph overview."""
    with st.sidebar:
        st.markdown(
            """
            <div style="padding: 12px 0 16px 0; border-bottom: 1px solid rgba(255, 255, 255, 0.08); margin-bottom: 16px;">
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="font-size: 1.6rem;">🕸️</span>
                    <span style="font-size: 1.3rem; font-weight: 800; color: #ffffff; letter-spacing: -0.02em;">ViralGraph</span>
                </div>
                <div style="font-size: 0.78rem; color: #60a5fa; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-top: 4px;">
                    Social Diffusion Platform
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Active Network Status Card
        model_label = st.session_state.get("model_label", "Scale-Free (BA)")
        metrics = SessionManager.get_metrics()
        n_nodes = metrics.get("nodes", 150)
        n_edges = metrics.get("edges", 441)
        is_conn = "Connected" if metrics.get("is_connected", True) else "Disconnected"

        st.markdown(
            f"""
            <div style="background-color: #151e32; border: 1px solid rgba(255, 255, 255, 0.08); border-radius: 10px; padding: 12px 14px; margin-bottom: 16px;">
                <div style="font-size: 0.72rem; font-weight: 700; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px;">
                    Active Network Topology
                </div>
                <div style="font-size: 0.92rem; font-weight: 600; color: #f1f5f9; margin-bottom: 6px;">
                    {model_label}
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.78rem; color: #cbd5e1;">
                    <span>|V| = <b>{n_nodes}</b></span>
                    <span>|E| = <b>{n_edges}</b></span>
                    <span style="color: #34d399;">● {is_conn}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Session Simulation Statistics
        history = SessionManager.get_simulation_history()
        history_count = len(history)

        st.markdown(
            f"""
            <div style="background-color: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255, 255, 255, 0.05); border-radius: 8px; padding: 8px 12px; margin-bottom: 16px; font-size: 0.8rem; color: #94a3b8;">
                Session Simulations: <b style="color: #60a5fa;">{history_count}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div style="margin-top: 24px; padding-top: 14px; border-top: 1px solid rgba(255, 255, 255, 0.08); font-size: 0.72rem; color: #64748b; line-height: 1.4;">
                <b>ViralGraph Research Suite</b><br>
                Discrete Mathematics & Graph Contagion<br>
                Milestone 1 Foundation
            </div>
            """,
            unsafe_allow_html=True,
        )

"""Reusable page header component for ViralGraph."""

from __future__ import annotations

import streamlit as st


def render_page_header(
    title: str,
    subtitle: str,
    badge: str | None = None,
    category: str | None = None,
) -> None:
    """Render a clean, consistent header across all ViralGraph pages.

    Args:
        title: Main page title (e.g. "Post Studio").
        subtitle: Concise one-sentence explanation of page utility.
        badge: Optional status tag (e.g. "Milestone 2", "Interactive Engine").
        category: Optional uppercase breadcrumb (e.g. "SIMULATION STUDIO").
    """
    badge_html = f"<span class='vg-badge'>{badge}</span>" if badge else ""
    cat_html = f"<div class='vg-breadcrumb'>{category}</div>" if category else ""

    st.markdown(
        f"""
        <div style="margin-bottom: 20px;">
            {cat_html}
            <h1 style="display: flex; align-items: center; margin: 0 0 6px 0; font-size: 2.0rem; font-weight: 700; color: #f8fafc;">
                {title} {badge_html}
            </h1>
            <p style="margin: 0; color: #94a3b8; font-size: 0.98rem; line-height: 1.5;">
                {subtitle}
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='margin: 0 0 24px 0; border: none; border-top: 1px solid rgba(255, 255, 255, 0.08);' />", unsafe_allow_html=True)

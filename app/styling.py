"""Shared styling tokens and dark theme CSS for ViralGraph."""

import streamlit as st


def apply_dark_theme() -> None:
    """Inject custom CSS to establish a cohesive, research-grade dark theme."""
    st.markdown(
        """
        <style>
        /* Base Application Background */
        .stApp {
            background-color: #0b0f19;
            color: #f8fafc;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        }

        /* Sidebar Styling */
        section[data-testid="stSidebar"] {
            background-color: #0f172a;
            border-right: 1px solid rgba(255, 255, 255, 0.08);
        }

        /* Top Header Area Clean-up & Hide Fork/Toolbar */
        header[data-testid="stHeader"] {
            background: transparent;
        }
        #MainMenu {visibility: hidden; display: none;}
        footer {visibility: hidden; display: none;}
        .stDeployButton {visibility: hidden; display: none;}
        [data-testid="stToolbar"] {visibility: hidden; display: none;}
        [data-testid="stDecoration"] {visibility: hidden; display: none;}
        [data-testid="stStatusWidget"] {visibility: hidden; display: none;}
        div[data-testid="stToolbarActions"] {visibility: hidden; display: none;}

        /* Reusable Card Containers */
        .vg-card {
            background-color: #151e32;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px 24px;
            margin-bottom: 16px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.25);
        }

        .vg-card-elevated {
            background-color: #1e293b;
            border: 1px solid rgba(59, 130, 246, 0.25);
            border-radius: 12px;
            padding: 22px 26px;
            margin-bottom: 20px;
        }

        .vg-card-interactive {
            background-color: #151e32;
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 12px;
            padding: 20px;
            transition: all 0.2s ease-in-out;
            height: 100%;
        }
        .vg-card-interactive:hover {
            border-color: rgba(59, 130, 246, 0.45);
            transform: translateY(-2px);
            box-shadow: 0 8px 24px rgba(59, 130, 246, 0.12);
        }

        /* Category Breadcrumb & Badge */
        .vg-breadcrumb {
            font-size: 0.75rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #60a5fa;
            margin-bottom: 4px;
        }

        .vg-badge {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            background-color: rgba(59, 130, 246, 0.15);
            color: #60a5fa;
            border: 1px solid rgba(59, 130, 246, 0.3);
            vertical-align: middle;
            margin-left: 8px;
        }

        .vg-badge-neutral {
            display: inline-block;
            padding: 3px 10px;
            border-radius: 12px;
            font-size: 0.78rem;
            font-weight: 600;
            letter-spacing: 0.02em;
            background-color: rgba(255, 255, 255, 0.08);
            color: #94a3b8;
            border: 1px solid rgba(255, 255, 255, 0.15);
            vertical-align: middle;
            margin-left: 8px;
        }

        /* Hero Banner */
        .vg-hero {
            background: linear-gradient(135deg, #151e32 0%, #0f172a 100%);
            border: 1px solid rgba(59, 130, 246, 0.2);
            border-radius: 14px;
            padding: 32px 36px;
            margin-bottom: 28px;
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        }

        .vg-hero h1 {
            color: #ffffff;
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 12px;
            letter-spacing: -0.02em;
        }

        .vg-hero p {
            color: #cbd5e1;
            font-size: 1.05rem;
            line-height: 1.6;
            max-width: 820px;
            margin-bottom: 20px;
        }

        /* Empty State */
        .vg-empty-state {
            background-color: rgba(21, 30, 50, 0.6);
            border: 1px dashed rgba(255, 255, 255, 0.15);
            border-radius: 12px;
            padding: 40px 24px;
            text-align: center;
            margin: 16px 0;
        }

        .vg-empty-icon {
            font-size: 2.5rem;
            color: #64748b;
            margin-bottom: 12px;
        }

        .vg-empty-title {
            font-size: 1.1rem;
            font-weight: 600;
            color: #e2e8f0;
            margin-bottom: 6px;
        }

        .vg-empty-desc {
            font-size: 0.9rem;
            color: #94a3b8;
            max-width: 500px;
            margin: 0 auto 16px auto;
            line-height: 1.5;
        }

        /* Section Headings */
        .vg-section-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin: 28px 0 16px 0;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            padding-bottom: 8px;
        }

        .vg-section-title {
            font-size: 1.25rem;
            font-weight: 700;
            color: #f1f5f9;
        }

        /* Button Polish */
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
            transition: all 0.15s ease-in-out;
        }

        /* Subtle scrollbars */
        ::-webkit-scrollbar {
            width: 6px;
            height: 6px;
        }
        ::-webkit-scrollbar-track {
            background: #0b0f19;
        }
        ::-webkit-scrollbar-thumb {
            background: #1e293b;
            border-radius: 3px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #334155;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

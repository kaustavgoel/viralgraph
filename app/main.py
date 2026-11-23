"""ViralGraph: Information Diffusion & Network Analysis Platform.

Main entry point for the multipage Streamlit research dashboard.
Orchestrates central navigation, shared dark theme styling, persistent session state,
and page execution across all eight core platform modules.
"""

from __future__ import annotations

import streamlit as st

from app.components.sidebar import render_shared_sidebar
from app.pages.home import render_home_page
from app.pages.influencer_research import render_influencer_research_page
from app.pages.intervention_lab import render_intervention_lab_page
from app.pages.live_diffusion import render_live_diffusion_page
from app.pages.network_explorer import render_network_explorer_page
from app.pages.post_studio import render_post_studio_page
from app.pages.research_methodology import render_research_methodology_page
from app.pages.virality_prediction import render_virality_prediction_page
from app.state import SessionManager
from app.styling import apply_dark_theme

# 1. Global Page Configuration
st.set_page_config(
    page_title="ViralGraph | Social Information Diffusion Platform",
    page_icon="🕸️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject Polished Research-Dashboard Dark Theme
apply_dark_theme()

# 3. Initialize Shared Session State
SessionManager.initialize()

# 4. Central Multipage Navigation Architecture (All 8 Pages)
pages = {
    "Overview": [
        st.Page(render_home_page, title="Home", icon=":material/home:", default=True, url_path="home"),
    ],
    "Simulation Studio": [
        st.Page(render_post_studio_page, title="Post Studio", icon=":material/edit_note:", url_path="post-studio"),
        st.Page(render_live_diffusion_page, title="Live Diffusion", icon=":material/play_circle:", url_path="live-diffusion"),
    ],
    "Network Lab": [
        st.Page(render_network_explorer_page, title="Network Explorer", icon=":material/hub:", url_path="network-explorer"),
        st.Page(render_influencer_research_page, title="Influencer Research", icon=":material/biotech:", url_path="influencer-research"),
        st.Page(render_intervention_lab_page, title="Intervention Lab", icon=":material/shield:", url_path="intervention-lab"),
        st.Page(render_virality_prediction_page, title="Virality Prediction", icon=":material/insights:", url_path="virality-prediction"),
    ],
    "Methodology": [
        st.Page(render_research_methodology_page, title="Research & Methodology", icon=":material/menu_book:", url_path="methodology"),
    ],
}

nav = st.navigation(pages)

# 5. Shared Persistent Sidebar
render_shared_sidebar()

# 6. Execute Selected Page
nav.run()

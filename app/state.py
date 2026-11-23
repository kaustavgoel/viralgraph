"""Shared session-state management for ViralGraph.

Retains application state across multipage navigation without unnecessary recomputations:
- Current post data & experiment notes
- Generated network graph, parameters, and invariants
- Selected seed node and strategy
- Simulation configurations and results
- Discrete playback step
- Multi-instance experiment parameters
- Session simulation history (with authentic empty state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import networkx as nx
import streamlit as st

from app.components.metrics_panel import compute_comprehensive_metrics
from viralgraph.data.synthetic_generator import generate_barabasi_albert


@dataclass
class PostData:
    """Representation of simulated social post content and metadata."""

    title: str = "Bridging Echo Chambers: The Geometry of Ideas"
    content: str = (
        "How does information escape clustered communities? Our discrete math "
        "investigation reveals that topological bridges channel 80% higher cross-boundary "
        "infection than degree hubs."
    )
    topic: str = "Network Science & Computing"
    description: str = ""  # Optional experiment notes
    media_type: str = "Text + Interactive Diagram"
    sentiment_score: float = 0.65  # Analytical neutral-positive tone
    virality_multiplier: float = 1.0


@dataclass
class SimulationConfig:
    """Configuration parameters for stochastic diffusion simulation."""

    model: str = "ICM"  # "ICM" (Independent Cascade) or "LTM" (Linear Threshold)
    transmission_prob: float = 0.12
    max_steps: int = 50
    seed_strategy: str = "highest_degree"  # "highest_degree", "highest_betweenness", "random", "manual"
    selected_seed_node: Any | None = None
    random_seed: int = 42
    ltm_threshold_mode: str = "random_uniform"  # "random_uniform" (U(0,1)) or "constant_uniform"
    ltm_constant_threshold: float = 0.50
    ltm_influence_weight_mode: str = "in_degree"  # b_uv = 1 / deg(v) (default backend weight)
    num_mc_runs: int = 40


@dataclass
class SimulationRecord:
    """Record of an executed cascade simulation for the recent simulations feed."""

    sim_id: str
    timestamp: str
    topology_name: str
    model_name: str
    seed_node: Any
    total_reach: int
    steps_taken: int
    external_fraction: float
    seed_strategy: str = "highest_degree"
    reach_percentage: float = 0.0
    post_title: str = ""


class SessionManager:
    """Centralized accessor and mutator for ViralGraph shared session state."""

    @staticmethod
    def initialize() -> None:
        """Initialize default session state keys if not already present."""
        if "initialized" not in st.session_state:
            # 1. Default Network Topology: Barabási-Albert (N=150, m=3, seed=42)
            default_graph = generate_barabasi_albert(n=150, m=3, seed=42)
            st.session_state["current_graph"] = default_graph
            st.session_state["model_type"] = "barabasi_albert"
            st.session_state["model_label"] = "Barabási–Albert (Scale-Free)"
            st.session_state["node_community_map"] = None
            st.session_state["current_params"] = {"n": 150, "m": 3, "seed": 42}
            st.session_state["metrics"] = compute_comprehensive_metrics(default_graph)

            # 2. Content & Seed State
            st.session_state["current_post"] = PostData()
            st.session_state["selected_seed"] = None
            st.session_state["seed_strategy"] = "highest_degree"

            # 3. Simulation & Playback State
            st.session_state["simulation_config"] = SimulationConfig()
            st.session_state["simulation_results"] = None
            st.session_state["current_playback_step"] = 0

            # 4. Research Experiment State
            st.session_state["experiment_config"] = {
                "num_instances": 5,
                "p_transmission": 0.12,
                "num_mc_per_instance": 40,
                "base_seed": 42,
            }

            # 5. Session History (Starts strictly empty: zero fabricated simulations)
            st.session_state["simulation_history"] = []

            st.session_state["initialized"] = True

    @staticmethod
    def get_current_graph() -> nx.Graph:
        """Retrieve the active networkx graph."""
        SessionManager.initialize()
        return st.session_state["current_graph"]

    @staticmethod
    def set_current_graph(
        graph: nx.Graph,
        model_type: str,
        model_label: str,
        params: dict[str, Any],
        community_map: dict[Any, int] | None = None,
    ) -> None:
        """Update active graph, associated metadata, and recompute invariants."""
        st.session_state["current_graph"] = graph
        st.session_state["model_type"] = model_type
        st.session_state["model_label"] = model_label
        st.session_state["current_params"] = params
        st.session_state["node_community_map"] = community_map
        st.session_state["metrics"] = compute_comprehensive_metrics(graph)

    @staticmethod
    def get_metrics() -> dict[str, Any]:
        """Retrieve precomputed graph metrics."""
        SessionManager.initialize()
        return st.session_state.get("metrics", {})

    @staticmethod
    def get_current_post() -> PostData:
        """Retrieve active simulated post data."""
        SessionManager.initialize()
        return st.session_state["current_post"]

    @staticmethod
    def set_current_post(post: PostData) -> None:
        """Update post data."""
        st.session_state["current_post"] = post

    @staticmethod
    def get_simulation_config() -> SimulationConfig:
        """Retrieve diffusion simulation configuration."""
        SessionManager.initialize()
        return st.session_state["simulation_config"]

    @staticmethod
    def set_simulation_config(config: SimulationConfig) -> None:
        """Update diffusion simulation configuration."""
        st.session_state["simulation_config"] = config

    @staticmethod
    def get_selected_seed() -> Any | None:
        """Retrieve currently selected seed vertex ID."""
        SessionManager.initialize()
        return st.session_state.get("selected_seed")

    @staticmethod
    def set_selected_seed(seed_node: Any | None) -> None:
        """Update selected seed vertex ID."""
        st.session_state["selected_seed"] = seed_node

    @staticmethod
    def get_seed_strategy() -> str:
        """Retrieve current seed selection strategy."""
        SessionManager.initialize()
        return st.session_state.get("seed_strategy", "highest_degree")

    @staticmethod
    def set_seed_strategy(strategy: str) -> None:
        """Update seed selection strategy."""
        st.session_state["seed_strategy"] = strategy

    @staticmethod
    def get_simulation_results() -> Any | None:
        """Retrieve latest cascade simulation results."""
        SessionManager.initialize()
        return st.session_state.get("simulation_results")

    @staticmethod
    def set_simulation_results(results: Any) -> None:
        """Store simulation output."""
        st.session_state["simulation_results"] = results

    @staticmethod
    def get_playback_step() -> int:
        """Retrieve current discrete playback step."""
        SessionManager.initialize()
        return st.session_state.get("current_playback_step", 0)

    @staticmethod
    def set_playback_step(step: int) -> None:
        """Update discrete playback step."""
        st.session_state["current_playback_step"] = step

    @staticmethod
    def get_simulation_history() -> list[SimulationRecord]:
        """Retrieve history of executed simulations in current session."""
        SessionManager.initialize()
        return st.session_state.get("simulation_history", [])

    @staticmethod
    def add_simulation_record(record: SimulationRecord) -> None:
        """Add completed cascade simulation to session history."""
        SessionManager.initialize()
        st.session_state["simulation_history"].insert(0, record)

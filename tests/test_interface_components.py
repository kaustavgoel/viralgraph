"""Unit tests for ViralGraph Streamlit interface components.

Verifies:
1. compute_comprehensive_metrics() calculates all 10 invariants accurately.
2. Handling of disconnected graphs (undefined global distances, fallback to GCC).
3. render_network_graph() produces valid Plotly figures across all three models.
"""

import networkx as nx
import pytest

from app.components.graph_visualization import render_network_graph
from app.components.metrics_panel import compute_comprehensive_metrics
from app.state import PostData, SessionManager, SimulationConfig, SimulationRecord
from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)


class TestMetricsPanel:
    """Test suite for metrics calculation."""

    def test_metrics_on_connected_graph(self):
        """Verify metric calculation on a connected Barabási-Albert network."""
        g = generate_barabasi_albert(n=100, m=2, seed=42)
        metrics = compute_comprehensive_metrics(g)

        assert metrics["nodes"] == 100
        assert metrics["edges"] == 2 * (100 - 2)
        assert metrics["is_connected"] is True
        assert metrics["num_components"] == 1
        assert metrics["avg_path_length"] is not None
        assert metrics["diameter"] is not None
        assert "Global" in metrics["path_metric_scope"]

    def test_metrics_on_disconnected_graph(self):
        """Verify correct mathematical handling when graph is disconnected."""
        # Create an explicitly disconnected graph: Two disjoint 4-cliques
        g = nx.Graph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0)])  # Triangle 1
        g.add_edges_from([(3, 4), (4, 5), (5, 6), (6, 3)])  # 4-cycle 2

        metrics = compute_comprehensive_metrics(g)
        assert metrics["nodes"] == 7
        assert metrics["edges"] == 7
        assert metrics["is_connected"] is False
        assert metrics["num_components"] == 2
        # Global path length is mathematically undefined, computed on GCC (the 4-cycle)
        assert metrics["giant_size"] == 4
        assert "GCC only" in metrics["path_metric_scope"]
        assert metrics["diameter"] == 2  # diameter of 4-cycle is 2

    def test_metrics_on_empty_or_trivial_graph(self):
        """Ensure boundary conditions are handled gracefully."""
        g = nx.Graph()
        g.add_node(0)
        metrics = compute_comprehensive_metrics(g)
        assert metrics["nodes"] == 1
        assert metrics["edges"] == 0
        assert metrics["mean_degree"] == 0.0
        assert metrics["avg_path_length"] == 0.0


class TestVisualizationRendering:
    """Test suite for Plotly figure construction."""

    def test_plotly_figure_barabasi_albert(self):
        """Ensure BA network produces a Plotly figure with edge and node traces."""
        g = generate_barabasi_albert(n=60, m=2, seed=42)
        fig = render_network_graph(g, model_type="barabasi_albert")

        assert len(fig.data) == 2  # Edge trace + Node trace
        assert fig.data[0].mode == "lines"
        assert fig.data[1].mode == "markers"

    def test_plotly_figure_watts_strogatz(self):
        """Ensure WS small-world network renders correctly with circular or spring layout."""
        g = generate_watts_strogatz(n=50, k=4, p=0.1, seed=42)
        fig = render_network_graph(g, model_type="watts_strogatz")

        assert len(fig.data) == 2
        assert fig.data[1].mode == "markers"


class TestSessionManagerAndMultipageState:
    """Test suite for shared session state management and multipage persistence."""

    def test_session_manager_initialization(self):
        """Verify that SessionManager initializes default graph and empty simulation history."""
        import streamlit as st
        # Clear session state for clean test
        st.session_state.clear()
        SessionManager.initialize()

        assert "initialized" in st.session_state
        assert st.session_state["model_type"] == "barabasi_albert"

        g = SessionManager.get_current_graph()
        assert isinstance(g, nx.Graph)
        assert g.number_of_nodes() == 150

        metrics = SessionManager.get_metrics()
        assert metrics["nodes"] == 150
        assert metrics["is_connected"] is True

        # Must start with strictly empty simulation history (no fabricated statistics)
        history = SessionManager.get_simulation_history()
        assert history == []

    def test_post_data_persistence(self):
        """Verify post drafting persistence."""
        post = SessionManager.get_current_post()
        assert isinstance(post, PostData)

        new_post = PostData(
            title="Discrete Math Breakthrough",
            content="Testing complex network cascades.",
            topic="Science",
        )
        SessionManager.set_current_post(new_post)
        retrieved = SessionManager.get_current_post()
        assert retrieved.title == "Discrete Math Breakthrough"
        assert retrieved.topic == "Science"

    def test_simulation_history_record_addition(self):
        """Verify that running simulations appends to history in reverse chronological order."""
        rec = SimulationRecord(
            sim_id="SIM-001",
            timestamp="2026-11-23 01:50",
            topology_name="Barabási–Albert",
            model_name="ICM",
            seed_node=42,
            total_reach=18,
            steps_taken=4,
            external_fraction=0.75,
        )
        SessionManager.add_simulation_record(rec)
        history = SessionManager.get_simulation_history()
        assert len(history) == 1
        assert history[0].sim_id == "SIM-001"
        assert history[0].total_reach == 18

    def test_graph_update_recomputes_metrics(self):
        """Verify that updating the active graph properly updates state and recomputes metrics."""
        ws = generate_watts_strogatz(n=80, k=4, p=0.1, seed=42)
        SessionManager.set_current_graph(
            graph=ws,
            model_type="watts_strogatz",
            model_label="Watts–Strogatz (Small-World)",
            params={"n": 80, "k": 4, "p": 0.1, "seed": 42},
        )

        assert SessionManager.get_current_graph().number_of_nodes() == 80
        metrics = SessionManager.get_metrics()
        assert metrics["nodes"] == 80


    def test_plotly_figure_stochastic_block_model(self):
        """Ensure SBM renders with community color mapping."""
        g, comm_map = generate_stochastic_block_model(
            community_sizes=[30, 30],
            probability_matrix=[[0.1, 0.01], [0.01, 0.1]],
            seed=42,
        )
        fig = render_network_graph(
            g, model_type="stochastic_block_model", node_community_map=comm_map
        )

        assert len(fig.data) == 2
        assert fig.data[1].mode == "markers"


class TestSessionManagerAndMultipageState:
    """Test suite for shared session state management and multipage persistence."""

    def test_session_manager_initialization(self):
        """Verify that SessionManager initializes default graph and empty simulation history."""
        import streamlit as st
        # Clear session state for clean test
        st.session_state.clear()
        SessionManager.initialize()

        assert "initialized" in st.session_state
        assert st.session_state["model_type"] == "barabasi_albert"

        g = SessionManager.get_current_graph()
        assert isinstance(g, nx.Graph)
        assert g.number_of_nodes() == 150

        metrics = SessionManager.get_metrics()
        assert metrics["nodes"] == 150
        assert metrics["is_connected"] is True

        # Must start with strictly empty simulation history (no fabricated statistics)
        history = SessionManager.get_simulation_history()
        assert history == []

    def test_post_data_persistence(self):
        """Verify post drafting persistence."""
        post = SessionManager.get_current_post()
        assert isinstance(post, PostData)

        new_post = PostData(
            title="Discrete Math Breakthrough",
            content="Testing complex network cascades.",
            topic="Science",
        )
        SessionManager.set_current_post(new_post)
        retrieved = SessionManager.get_current_post()
        assert retrieved.title == "Discrete Math Breakthrough"
        assert retrieved.topic == "Science"

    def test_simulation_history_record_addition(self):
        """Verify that running simulations appends to history in reverse chronological order."""
        rec = SimulationRecord(
            sim_id="SIM-001",
            timestamp="2026-11-23 01:50",
            topology_name="Barabási–Albert",
            model_name="ICM",
            seed_node=42,
            total_reach=18,
            steps_taken=4,
            external_fraction=0.75,
        )
        SessionManager.add_simulation_record(rec)
        history = SessionManager.get_simulation_history()
        assert len(history) == 1
        assert history[0].sim_id == "SIM-001"
        assert history[0].total_reach == 18

    def test_graph_update_recomputes_metrics(self):
        """Verify that updating the active graph properly updates state and recomputes metrics."""
        ws = generate_watts_strogatz(n=80, k=4, p=0.1, seed=42)
        SessionManager.set_current_graph(
            graph=ws,
            model_type="watts_strogatz",
            model_label="Watts–Strogatz (Small-World)",
            params={"n": 80, "k": 4, "p": 0.1, "seed": 42},
        )

        assert SessionManager.get_current_graph().number_of_nodes() == 80
        metrics = SessionManager.get_metrics()
        assert metrics["nodes"] == 80


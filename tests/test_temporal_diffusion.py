"""Unit tests for temporal diffusion dynamics and observation window leakage prevention."""

import networkx as nx
import pytest

from viralgraph.analytics.temporal_diffusion import (
    analyze_temporal_cascade,
    extract_temporal_observation_window,
)
from viralgraph.simulation.independent_cascade import simulate_independent_cascade


def test_temporal_cascade_metrics():
    """Verify velocity, peak infection, and half-saturation calculations."""
    g = nx.path_graph(10)
    # At p=1.0 on path graph starting from node 0, deterministic cascade: 1 new node per step
    res = simulate_independent_cascade(g, seed_nodes=[0], p=1.0)
    metrics = analyze_temporal_cascade(res)

    assert metrics.final_reach == 10
    assert metrics.total_duration == 9
    assert metrics.early_velocity_step1 == 1.0
    assert metrics.early_velocity_step2 == 1.0
    assert metrics.new_infections_per_step == [1] * 10
    assert metrics.half_saturation_step == 4  # reach at t=4 is 5 >= 0.5 * 10


def test_temporal_window_leakage_guard():
    """Verify observation window strictly bounds features to t <= t_obs without future leakage."""
    g = nx.star_graph(20)
    res = simulate_independent_cascade(g, seed_nodes=[0], p=1.0)

    window = extract_temporal_observation_window(res, t_obs=1, virality_threshold=15)
    assert window.observation_horizon == 1
    assert window.observed_reach == 21
    assert len(window.observed_new_infections) == 2  # t=0, t=1
    assert window.is_viral is True

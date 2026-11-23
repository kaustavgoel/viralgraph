"""Monte Carlo simulation harness for stochastic diffusion models in ViralGraph.

Runs repeated stochastic realizations of ICM and LTM to compute:
1. Expected cascade reach E[R_infinity] = sigma(S_0).
2. Cascade reach standard deviation and distribution.
3. Node-level activation probabilities P(node v in Cascade).
4. Average propagation timelines.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable
import networkx as nx
import numpy as np

from viralgraph.simulation.independent_cascade import (
    CascadeResult,
    simulate_independent_cascade,
)
from viralgraph.simulation.linear_threshold import simulate_linear_threshold


@dataclass(frozen=True)
class MonteCarloResult:
    """Aggregated statistical outcome across M Monte Carlo cascade trials."""

    model_name: str
    seed_nodes: list[Any]
    num_simulations: int
    expected_reach: float  # sigma(S_0)
    std_reach: float
    min_reach: int
    max_reach: int
    reach_distribution: list[int]  # List of reach outcomes across M runs
    node_activation_probabilities: dict[Any, float]  # P(v in Cascade)
    average_steps_to_termination: float
    mean_cumulative_timeline: list[float]


def run_monte_carlo_simulation(
    graph: nx.Graph,
    seed_nodes: list[Any] | set[Any],
    model: str = "ICM",
    num_simulations: int = 100,
    seed: int = 42,
    p: float = 0.1,
    **kwargs: Any,
) -> MonteCarloResult:
    """Execute M independent Monte Carlo trials of information diffusion.

    Args:
        graph: Undirected or directed NetworkX graph.
        seed_nodes: Initial active seed set S_0.
        model: "ICM" (Independent Cascade) or "LTM" (Linear Threshold).
        num_simulations: Number of Monte Carlo iterations M (default 100).
        seed: Master random seed for reproducible random streams.
        p: Transmission probability for ICM.
        **kwargs: Additional parameters passed to diffusion simulators.

    Returns:
        MonteCarloResult containing empirical expected reach, variance, and activation fields.
    """
    rng = np.random.default_rng(seed)

    reach_outcomes: list[int] = []
    step_counts: list[int] = []
    node_activation_counts: dict[Any, int] = {node: 0 for node in graph.nodes()}
    all_timelines: list[list[int]] = []

    sim_func: Callable
    if model.upper() == "ICM":
        sim_func = lambda r: simulate_independent_cascade(
            graph=graph, seed_nodes=seed_nodes, p=p, rng=r, **kwargs
        )
    elif model.upper() == "LTM":
        sim_func = lambda r: simulate_linear_threshold(
            graph=graph, seed_nodes=seed_nodes, rng=r, **kwargs
        )
    else:
        raise ValueError(f"Unknown diffusion model: '{model}'. Choose 'ICM' or 'LTM'.")

    for _ in range(num_simulations):
        # Generate independent sub-generator for each trial
        trial_rng = np.random.default_rng(rng.integers(0, 2**31 - 1))
        res: CascadeResult = sim_func(trial_rng)

        reach_outcomes.append(res.total_reach)
        step_counts.append(res.steps_to_termination)
        all_timelines.append(res.cumulative_reach_timeline)

        # Record infected nodes
        for node_set in res.activation_history:
            for v in node_set:
                node_activation_counts[v] += 1

    # Compute node activation probabilities
    act_probs = {node: count / float(num_simulations) for node, count in node_activation_counts.items()}

    # Compute mean cumulative timeline (aligning varying step lengths by padding with final reach)
    max_len = max(len(tl) for tl in all_timelines)
    padded_matrix = np.zeros((num_simulations, max_len), dtype=float)
    for i, tl in enumerate(all_timelines):
        padded_matrix[i, : len(tl)] = tl
        padded_matrix[i, len(tl) :] = tl[-1]  # Hold final reach
    mean_timeline = list(np.mean(padded_matrix, axis=0))

    return MonteCarloResult(
        model_name=model.upper(),
        seed_nodes=sorted(list(seed_nodes)),
        num_simulations=num_simulations,
        expected_reach=float(np.mean(reach_outcomes)),
        std_reach=float(np.std(reach_outcomes)),
        min_reach=int(np.min(reach_outcomes)),
        max_reach=int(np.max(reach_outcomes)),
        reach_distribution=reach_outcomes,
        node_activation_probabilities=act_probs,
        average_steps_to_termination=float(np.mean(step_counts)),
        mean_cumulative_timeline=mean_timeline,
    )

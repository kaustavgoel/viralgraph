"""Simulation execution and seed selection engine for ViralGraph Post Studio.

Integrates the UI configuration with the existing mathematical diffusion backend:
- Independent Cascade Model (ICM) via simulate_independent_cascade()
- Linear Threshold Model (LTM) via simulate_linear_threshold()
- Seed node resolution across multiple topological strategies (Hub, Betweenness, Random, Manual)
- External community infection calculation
- Safe session state persistence without duplicate execution
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
import networkx as nx
import numpy as np

from app.state import PostData, SessionManager, SimulationConfig, SimulationRecord
from viralgraph.graph.centrality import compute_betweenness_centrality
from viralgraph.simulation.independent_cascade import CascadeResult, simulate_independent_cascade
from viralgraph.simulation.linear_threshold import simulate_linear_threshold


def resolve_seed_node(
    graph: nx.Graph,
    strategy: str,
    manual_node: Any | None = None,
    seed: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """Resolve initial seed vertex and its local centrality properties.

    Args:
        graph: Active NetworkX graph.
        strategy: 'highest_degree', 'highest_betweenness', 'random', or 'manual'.
        manual_node: Node identifier if strategy is 'manual'.
        seed: Random seed for deterministic random selection.

    Returns:
        tuple[Any, dict[str, Any]]: (seed_node_id, metrics_dict)

    Raises:
        ValueError: If graph has no nodes or manual node is invalid.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot select a seed node from an empty graph (|V| = 0).")

    nodes = list(graph.nodes())

    if strategy == "highest_degree":
        selected_node = max(nodes, key=lambda u: graph.degree(u))
    elif strategy == "highest_betweenness":
        # Compute shortest-path betweenness centrality
        bet = compute_betweenness_centrality(graph, seed=seed)
        selected_node = max(nodes, key=lambda u: bet.get(u, 0.0))
    elif strategy == "random":
        rng = np.random.default_rng(seed)
        selected_node = nodes[int(rng.integers(0, len(nodes)))]
    elif strategy == "manual":
        if manual_node is None or manual_node not in graph:
            # Fallback to first node if manual_node is invalid
            raise ValueError(f"Manual seed node '{manual_node}' does not exist in graph.")
        selected_node = manual_node
    else:
        raise ValueError(f"Unknown seed selection strategy: '{strategy}'.")

    # Compute node-level metrics for UI display
    deg = graph.degree(selected_node)
    n = graph.number_of_nodes()
    norm_deg = deg / (n - 1) if n > 1 else 0.0

    # Local clustering coefficient
    clust = float(nx.clustering(graph, selected_node)) if deg >= 2 else 0.0

    # Community assignment if available in graph metadata
    comm_map = graph.graph.get("node_community_map")
    comm_id = comm_map.get(selected_node, "N/A") if comm_map else "N/A"

    metrics = {
        "node_id": selected_node,
        "degree": deg,
        "normalized_degree": norm_deg,
        "clustering": clust,
        "community": comm_id,
        "strategy": strategy,
    }

    return selected_node, metrics


def compute_external_infection_fraction(
    graph: nx.Graph,
    seed_node: Any,
    infected_nodes: set[Any] | list[Any],
    community_map: dict[Any, int] | None = None,
) -> float:
    """Compute fraction of infected cascade vertices outside the seed's home community.

    Formula:
        rho_external = |Cascade cap (V \\ C_s)| / |Cascade|
        Seed node is in C_s, so it contributes 0 to the numerator.
    """
    total_infected = len(infected_nodes)
    if total_infected <= 1 or not community_map:
        return 0.0

    seed_comm = community_map.get(seed_node)
    if seed_comm is None:
        return 0.0

    outside_infected = sum(
        1 for v in infected_nodes if community_map.get(v) != seed_comm and v != seed_node
    )
    return float(outside_infected / total_infected)


def launch_diffusion_simulation(
    graph: nx.Graph,
    seed_node: Any,
    config: SimulationConfig,
    post: PostData,
    community_map: dict[Any, int] | None = None,
) -> tuple[CascadeResult, SimulationRecord]:
    """Execute stochastic diffusion simulation on active topology using existing backend engines.

    Args:
        graph: Active NetworkX graph.
        seed_node: Valid initial seed vertex.
        config: Simulation parameters (ICM or LTM).
        post: Post payload metadata.
        community_map: Optional node -> community index mapping.

    Returns:
        tuple[CascadeResult, SimulationRecord]: The raw trajectory and session record.
    """
    if graph.number_of_nodes() == 0:
        raise ValueError("Cannot simulate diffusion on an empty graph.")
    if seed_node not in graph:
        raise ValueError(f"Seed node '{seed_node}' is not present in the graph.")

    rng = np.random.default_rng(config.random_seed)

    # 1. Execute diffusion model
    if config.model == "ICM":
        result = simulate_independent_cascade(
            graph=graph,
            seed_nodes=[seed_node],
            p=config.transmission_prob,
            max_steps=config.max_steps,
            rng=rng,
        )
    elif config.model == "LTM":
        # Handle LTM threshold configuration
        if config.ltm_threshold_mode == "constant_uniform":
            thresholds = {node: float(config.ltm_constant_threshold) for node in graph.nodes()}
        else:
            thresholds = None  # Backend samples i.i.d. U(0, 1)

        result = simulate_linear_threshold(
            graph=graph,
            seed_nodes=[seed_node],
            thresholds=thresholds,
            max_steps=config.max_steps,
            rng=rng,
        )
    else:
        raise ValueError(f"Unsupported diffusion model: '{config.model}'.")

    # 2. Extract infected nodes set across all steps
    all_infected = set().union(*result.activation_history)

    # 3. Compute cross-community transmission
    ext_frac = compute_external_infection_fraction(
        graph=graph,
        seed_node=seed_node,
        infected_nodes=all_infected,
        community_map=community_map,
    )

    # 4. Construct SimulationRecord for history
    history = SessionManager.get_simulation_history()
    sim_id = f"SIM-{len(history) + 1:03d}"
    timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    reach_pct = (result.total_reach / graph.number_of_nodes()) * 100.0 if graph.number_of_nodes() > 0 else 0.0

    record = SimulationRecord(
        sim_id=sim_id,
        timestamp=timestamp_str,
        topology_name=graph.name if hasattr(graph, "name") and graph.name else "Custom Network",
        model_name=config.model,
        seed_node=seed_node,
        seed_strategy=config.seed_strategy,
        total_reach=result.total_reach,
        reach_percentage=reach_pct,
        steps_taken=result.steps_to_termination,
        external_fraction=ext_frac,
        post_title=post.title,
    )

    # 5. Store in SessionManager
    SessionManager.set_simulation_results(result)
    SessionManager.set_simulation_config(config)
    SessionManager.set_current_post(post)
    SessionManager.set_selected_seed(seed_node)
    SessionManager.add_simulation_record(record)

    return result, record

"""Audited Simulation Dataset Generation Pipeline for ViralGraph.

Scientific Audit Enforcements:
1. Grouped Architecture: Each sample belongs to a distinct `graph_id` so train/test sets can be split across unseen graphs.
2. Strict Temporal Horizon (t <= t_obs): Features reference only events observed up to t_obs.
3. Label Independence: Does NOT compute percentile thresholds globally; emits continuous targets
   so the classifier can compute the threshold strictly on training folds to prevent target leakage.
"""

from __future__ import annotations

from typing import Any
import networkx as nx
import numpy as np
import pandas as pd

from viralgraph.analytics.temporal_diffusion import extract_temporal_observation_window
from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)
from viralgraph.graph.centrality import compute_centrality_profile
from viralgraph.graph.representation import GraphRepresentation
from viralgraph.simulation.independent_cascade import simulate_independent_cascade


def extract_strict_early_features(
    graph: nx.Graph,
    seed_node: Any,
    t_obs: int = 1,
    cascade_res: Any | None = None,
) -> dict[str, float]:
    """Extract strictly observed features for a given (network, seed, cascade) tuple at t <= t_obs."""
    n = graph.number_of_nodes()
    m = graph.number_of_edges()

    # Network-level static features
    density = float(nx.density(graph))
    avg_clustering = float(nx.average_clustering(graph))

    # Node-level structural features
    cent = compute_centrality_profile(graph)
    deg_v = graph.degree(seed_node)
    c_d = cent.degree[seed_node]
    c_b = cent.betweenness[seed_node]
    c_c = cent.closeness[seed_node]
    c_e = cent.eigenvector[seed_node]
    pr = cent.pagerank[seed_node]
    bdr = cent.betweenness_degree_ratios.get(seed_node, 0.0)
    clust_v = float(nx.clustering(graph, seed_node))

    # Early temporal cascade features strictly bounded to t <= t_obs
    if cascade_res is not None:
        win = extract_temporal_observation_window(cascade_res, t_obs=t_obs)
        obs_reach = float(win.observed_reach)
        obs_velocity = float(win.observed_velocity)
        inf_step1 = float(win.observed_new_infections[1]) if len(win.observed_new_infections) > 1 else 0.0
    else:
        obs_reach = 1.0
        obs_velocity = 0.0
        inf_step1 = 0.0

    return {
        "network_nodes": float(n),
        "network_edges": float(m),
        "network_density": density,
        "network_avg_clustering": avg_clustering,
        "seed_degree": float(deg_v),
        "seed_degree_centrality": c_d,
        "seed_betweenness": c_b,
        "seed_closeness": c_c,
        "seed_eigenvector": c_e,
        "seed_pagerank": pr,
        "seed_bdr": bdr,
        "seed_clustering": clust_v,
        "early_observed_reach": obs_reach,
        "early_velocity": obs_velocity,
        "early_infections_step1": inf_step1,
    }


def generate_grouped_cascade_dataset(
    num_graphs: int = 20,
    cascades_per_graph: int = 15,
    n_nodes: int = 120,
    t_obs: int = 1,
    p_transmission: float = 0.12,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate a grouped tabular dataset for machine learning evaluation.

    Returns:
        tuple[pd.DataFrame, pd.Series, pd.Series]:
            - X: DataFrame containing features and a 'graph_id' column for grouped splitting.
            - y_reach: Continuous final cascade reach (target).
            - graph_ids: Series containing integer graph IDs for GroupKFold splitting.
    """
    rng = np.random.default_rng(seed)
    feature_rows: list[dict[str, Any]] = []
    final_reaches: list[int] = []
    graph_ids: list[int] = []

    models = ["ba", "ws", "sbm"]

    for g_idx in range(num_graphs):
        model_choice = models[g_idx % len(models)]
        g_seed = int(rng.integers(1, 100000))

        if model_choice == "ba":
            g = generate_barabasi_albert(n=n_nodes, m=3, seed=g_seed)
        elif model_choice == "ws":
            g = generate_watts_strogatz(n=n_nodes, k=6, p=0.08, seed=g_seed)
        else:
            b_size = n_nodes // 3
            sizes = [b_size, b_size, n_nodes - 2 * b_size]
            prob = [[0.10, 0.005, 0.002], [0.005, 0.10, 0.005], [0.002, 0.005, 0.10]]
            g, _ = generate_stochastic_block_model(sizes, prob, seed=g_seed)

        nodes_list = list(g.nodes())

        for _ in range(cascades_per_graph):
            seed_node = rng.choice(nodes_list)

            cascade = simulate_independent_cascade(
                g, seed_nodes=[seed_node], p=p_transmission, rng=rng
            )

            feats = extract_strict_early_features(
                graph=g, seed_node=seed_node, t_obs=t_obs, cascade_res=cascade
            )
            feats["graph_id"] = g_idx
            feature_rows.append(feats)
            final_reaches.append(cascade.total_reach)
            graph_ids.append(g_idx)

    df_x = pd.DataFrame(feature_rows)
    s_reach = pd.Series(final_reaches, name="final_reach")
    s_groups = pd.Series(graph_ids, name="graph_id")

    return df_x, s_reach, s_groups

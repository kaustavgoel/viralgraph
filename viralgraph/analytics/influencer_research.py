"""Influencer vs. Bridge-Node Empirical Research Engine for ViralGraph (Audited Version).

Addresses the central research hypothesis:
    "Can a community-bridging node generate a larger or more cross-cutting
     information cascade than a high-degree hub?"

Scientific & Statistical Audit Enhancements:
1. Multi-instance graph sampling: Primary sampling units are N_instances independent network realizations.
2. Small-sample uncertainty estimation: Reports graph-level sample means, standard errors,
   and 95% Confidence Intervals derived from Student's t-distribution with df = N_instances - 1
   (e.g., t_crit = 2.776 for df = 4), avoiding normal-approximation undercoverage.
3. Separation of variance sources: Explicitly isolates between-graph topological variance
   from within-graph Monte Carlo simulation variance.
4. Formal paired hypothesis testing: Evaluates paired differences (Delta = Reach_hub - Reach_bridge)
   on identical graph realizations via paired Student's t-tests, reporting t-statistics, p-values,
   and paired Cohen's d effect sizes.
5. Rigorous mathematical definitions for cross-community diffusion:
   - External Infection Fraction:
     rho_external = sum_{v in V \\ C_s} P(v) / sum_{v in V} P(v)
     Denominator includes seed activation P(s)=1.0; returns 0.0 when K <= 1 or cascade is empty.
   - External Community Penetration:
     Pi_external = |{c in P \\ {C_s} : exists v in V_c with P(v) >= theta}| / (K - 1)
     Returns 0.0 when K <= 1 (no outside communities exist).
6. Community detection provenance: Ground-truth block partitions for SBM; Louvain modularity
   optimization heuristic for BA and WS (which lack planted ground truth).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import networkx as nx
import numpy as np
import scipy.stats as stats

from viralgraph.graph.centrality import compute_centrality_profile
from viralgraph.graph.community import compute_bridging_centrality, detect_communities
from viralgraph.simulation.monte_carlo import run_monte_carlo_simulation


@dataclass(frozen=True)
class StrategyInstanceMeasurement:
    """Measurement for a seed strategy on a single independent graph realization."""

    instance_id: int
    seed_node: Any
    is_hub: bool
    seed_degree: float
    seed_betweenness: float
    seed_bdr: float
    mc_expected_reach: float
    mc_variance_reach: float
    external_comm_penetration: float
    external_infection_fraction: float


@dataclass(frozen=True)
class AggregatedStrategyOutcome:
    """Consolidated performance profile for a seed strategy across multiple graph instances."""

    strategy_name: str
    num_instances: int
    num_mc_per_instance: int

    # Mean structural properties of selected seeds
    mean_seed_degree: float
    mean_seed_betweenness: float
    mean_seed_bdr: float

    # Cascade reach statistics across graph realizations (primary sampling units)
    mean_expected_reach: float
    std_expected_reach: float  # Between-graph topological standard deviation (s)
    se_expected_reach: float  # Standard error of the mean (s / sqrt(N))
    ci95_expected_reach: tuple[float, float]  # Student's t 95% Confidence Interval
    within_graph_mc_std: float  # Mean within-graph Monte Carlo standard deviation

    # Cross-community transmission metrics
    mean_external_comm_penetration: float  # Pi_ext in [0, 1]
    ci95_external_comm_penetration: tuple[float, float]
    mean_external_infection_fraction: float  # rho_ext in [0, 1]
    ci95_external_infection_fraction: tuple[float, float]

    # Strategy identity overlap rate (frequency of selecting the exact same node as highest_degree)
    overlap_with_hub_rate: float

    # Raw per-instance measurements for paired hypothesis testing
    instance_measurements: tuple[StrategyInstanceMeasurement, ...]


@dataclass(frozen=True)
class PairedStrategyComparison:
    """Rigorous paired statistical comparison between two strategies across identical graph instances."""

    strategy_a: str
    strategy_b: str
    metric_name: str
    mean_difference: float  # mean(A - B)
    std_difference: float
    se_difference: float
    ci95_difference: tuple[float, float]
    t_statistic: float
    p_value: float
    degrees_of_freedom: int
    cohens_d: float
    is_statistically_significant: bool  # p < 0.05


def compute_student_t_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> tuple[float, float, float, tuple[float, float]]:
    """Compute sample mean, sample std, standard error, and Student's t confidence interval.

    Args:
        data: 1D array of observations from independent primary sampling units.
        confidence: Desired confidence level (default 0.95).

    Returns:
        tuple: (mean, sample_std, standard_error, (ci_lower, ci_upper))
    """
    n = len(data)
    mean = float(np.mean(data))
    if n <= 1:
        return mean, 0.0, 0.0, (mean, mean)

    s = float(np.std(data, ddof=1))
    se = s / np.sqrt(n)
    df = n - 1
    t_crit = float(stats.t.ppf((1.0 + confidence) / 2.0, df=df))
    margin = t_crit * se
    return mean, s, se, (mean - margin, mean + margin)


def select_seed_with_overlap_detection(
    graph: nx.Graph,
    strategy: str,
    node_to_community: dict[Any, int] | None = None,
    seed: int = 42,
) -> tuple[Any, bool]:
    """Select a seed node and flag whether it is identical to the highest-degree hub."""
    cent = compute_centrality_profile(graph, seed=seed)
    hub_node = cent.top_degree_nodes[0]

    if strategy == "highest_degree":
        selected = hub_node
    elif strategy == "highest_betweenness":
        selected = cent.top_betweenness_nodes[0]
    elif strategy == "highest_pagerank":
        selected = cent.top_pagerank_nodes[0]
    elif strategy == "community_bridge":
        bdr = cent.betweenness_degree_ratios
        candidates = [n for n in graph.nodes() if graph.degree(n) >= 2]
        if candidates:
            # Sort by BDR descending, tie-break by betweenness
            selected = max(candidates, key=lambda n: (bdr.get(n, 0.0), cent.betweenness[n]))
        else:
            selected = cent.top_betweenness_nodes[0]
    elif strategy == "random_baseline":
        rng = np.random.default_rng(seed)
        nodes = list(graph.nodes())
        selected = nodes[int(rng.integers(0, len(nodes)))]
    else:
        raise ValueError(f"Unknown seed strategy: '{strategy}'.")

    is_identical_to_hub = bool(selected == hub_node)
    return selected, is_identical_to_hub


def compute_rigorous_community_metrics(
    graph: nx.Graph,
    seed_node: Any,
    node_to_community: dict[Any, int],
    activation_probs: dict[Any, float],
    prob_threshold: float = 0.02,
) -> tuple[float, float]:
    """Compute mathematically rigorous cross-community diffusion metrics.

    Mathematical Formulation:
        Let P = {C_1, ..., C_K} be the partition of V into K communities.
        Let s in V be the seed node, with home community C_s in P.

        1. External Infection Fraction:
           rho_external = |Cascade cap (V \\ C_s)| / |Cascade|
           In Monte Carlo expectation with node activation probabilities P(v):
           rho_external = sum_{v in V \\ C_s} P(v) / sum_{v in V} P(v)

           - Denominator sum_{v} P(v) includes P(s) = 1.0 (seed node).
           - Since s in C_s, s contributes 0 to the numerator.
           - If sum_{v} P(v) == 0 (empty cascade), returns 0.0.
           - If K <= 1 (only 1 community in graph, V \\ C_s = empty), returns 0.0.

        2. External Community Penetration:
           Pi_external = |{c in P \\ {C_s} : exists v in V_c s.t. P(v) >= prob_threshold}| / (K - 1)
           - Returns 0.0 when K <= 1 (no external communities exist).

    Args:
        graph: Underlying networkx Graph.
        seed_node: Chosen seed vertex s.
        node_to_community: Map from node identifier to community integer id.
        activation_probs: Dict mapping each node v to its marginal activation probability P(v).
        prob_threshold: Probability threshold to declare an external community penetrated.

    Returns:
        tuple[float, float]: (Pi_external, rho_external)
    """
    total_prob_mass = sum(activation_probs.values())
    if total_prob_mass <= 0.0:
        return 0.0, 0.0

    # Ensure seed node has a community assignment
    if seed_node not in node_to_community:
        # Fallback: assign seed unique identifier if unassigned
        seed_comm = -999
    else:
        seed_comm = node_to_community[seed_node]

    # Gather distinct communities across the network
    all_communities = set(node_to_community.values())
    outside_comms = all_communities - {seed_comm}
    num_outside_comms = len(outside_comms)

    # Edge case: K <= 1 (no foreign communities exist)
    if num_outside_comms <= 0:
        return 0.0, 0.0

    # Calculate outside probability mass and penetrated outside communities
    outside_prob_mass = 0.0
    penetrated_outside = set()

    for node, p_act in activation_probs.items():
        comm = node_to_community.get(node, -1)
        if comm != seed_comm:
            outside_prob_mass += p_act
            if p_act >= prob_threshold and comm in outside_comms:
                penetrated_outside.add(comm)

    # Pi_external = (distinct external communities reached) / (K - 1)
    ext_penetration = len(penetrated_outside) / float(num_outside_comms)
    # rho_external = (mass outside home community) / (total cascade mass)
    ext_infection_fraction = outside_prob_mass / total_prob_mass

    return float(ext_penetration), float(ext_infection_fraction)


def compare_strategies_paired(
    aggregated_outcomes: list[AggregatedStrategyOutcome],
    strat_a: str,
    strat_b: str,
    metric: str = "reach",
) -> PairedStrategyComparison:
    """Perform formal paired hypothesis testing between two strategies on identical graph instances.

    Args:
        aggregated_outcomes: List of outcomes from run_multi_instance_influencer_experiment.
        strat_a: Name of first strategy (e.g., 'highest_degree').
        strat_b: Name of second strategy (e.g., 'community_bridge').
        metric: Metric to compare ('reach' or 'external_fraction').

    Returns:
        PairedStrategyComparison with t-statistic, p-value, and effect size.
    """
    outcomes_dict = {o.strategy_name: o for o in aggregated_outcomes}
    if strat_a not in outcomes_dict or strat_b not in outcomes_dict:
        raise ValueError(f"Both strategies must be in outcomes: '{strat_a}', '{strat_b}'.")

    meas_a = outcomes_dict[strat_a].instance_measurements
    meas_b = outcomes_dict[strat_b].instance_measurements

    if len(meas_a) != len(meas_b):
        raise ValueError("Measurements must have identical number of graph instances.")

    if metric == "reach":
        vals_a = np.array([m.mc_expected_reach for m in meas_a])
        vals_b = np.array([m.mc_expected_reach for m in meas_b])
    elif metric == "external_fraction":
        vals_a = np.array([m.external_infection_fraction for m in meas_a])
        vals_b = np.array([m.external_infection_fraction for m in meas_b])
    elif metric == "penetration":
        vals_a = np.array([m.external_comm_penetration for m in meas_a])
        vals_b = np.array([m.external_comm_penetration for m in meas_b])
    else:
        raise ValueError(f"Unknown metric for paired comparison: '{metric}'.")

    # Paired differences Delta_i = A_i - B_i
    diffs = vals_a - vals_b
    mean_diff, std_diff, se_diff, ci95_diff = compute_student_t_confidence_interval(diffs)

    n = len(diffs)
    df = n - 1

    if se_diff > 1e-12:
        t_stat = mean_diff / se_diff
        p_val = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df=df)))
        cohens_d = mean_diff / std_diff
    else:
        t_stat = 0.0
        p_val = 1.0
        cohens_d = 0.0

    return PairedStrategyComparison(
        strategy_a=strat_a,
        strategy_b=strat_b,
        metric_name=metric,
        mean_difference=mean_diff,
        std_difference=std_diff,
        se_difference=se_diff,
        ci95_difference=ci95_diff,
        t_statistic=t_stat,
        p_value=p_val,
        degrees_of_freedom=df,
        cohens_d=cohens_d,
        is_statistically_significant=bool(p_val < 0.05),
    )


def run_multi_instance_influencer_experiment(
    graph_generator: Any,
    graph_gen_kwargs: dict[str, Any],
    num_instances: int = 5,
    num_mc_per_instance: int = 40,
    p_transmission: float = 0.12,
    base_seed: int = 42,
) -> list[AggregatedStrategyOutcome]:
    """Execute audited multi-instance research experiment across 5 seed strategies.

    Args:
        graph_generator: Graph generation callable.
        graph_gen_kwargs: Parameters for generator.
        num_instances: Number of independent graph realizations (primary sampling units, default 5).
        num_mc_per_instance: Monte Carlo cascade simulations per realization (default 40).
        p_transmission: ICM transmission probability.
        base_seed: Master random seed.

    Returns:
        list[AggregatedStrategyOutcome]: Aggregated results with Student's t 95% CIs.
    """
    strategies = [
        "highest_degree",
        "highest_betweenness",
        "highest_pagerank",
        "community_bridge",
        "random_baseline",
    ]

    # Collect per-instance measurements: {strategy: list[StrategyInstanceMeasurement]}
    raw_measurements: dict[str, list[StrategyInstanceMeasurement]] = {strat: [] for strat in strategies}

    for inst_idx in range(num_instances):
        inst_seed = base_seed + inst_idx * 1007
        gen_out = graph_generator(**graph_gen_kwargs, seed=inst_seed)
        if isinstance(gen_out, tuple):
            g = gen_out[0]
            comm_map = gen_out[1]
        else:
            g = gen_out
            # BA / WS lack planted partitions; use Louvain modularity optimization heuristic
            comm_struct = detect_communities(g, seed=inst_seed)
            comm_map = comm_struct.node_to_community

        cent = compute_centrality_profile(g, seed=inst_seed)

        for strat in strategies:
            seed_node, is_hub = select_seed_with_overlap_detection(
                g, strategy=strat, node_to_community=comm_map, seed=inst_seed
            )

            # Run Monte Carlo cascades
            mc_res = run_monte_carlo_simulation(
                graph=g,
                seed_nodes=[seed_node],
                model="ICM",
                p=p_transmission,
                num_simulations=num_mc_per_instance,
                seed=inst_seed,
            )

            # Compute cross-community metrics
            ext_pen, ext_frac = compute_rigorous_community_metrics(
                graph=g,
                seed_node=seed_node,
                node_to_community=comm_map,
                activation_probs=mc_res.node_activation_probabilities,
            )

            meas = StrategyInstanceMeasurement(
                instance_id=inst_idx,
                seed_node=seed_node,
                is_hub=is_hub,
                seed_degree=float(g.degree(seed_node)),
                seed_betweenness=float(cent.betweenness[seed_node]),
                seed_bdr=float(cent.betweenness_degree_ratios.get(seed_node, 0.0)),
                mc_expected_reach=float(mc_res.expected_reach),
                mc_variance_reach=float(mc_res.std_reach**2),
                external_comm_penetration=ext_pen,
                external_infection_fraction=ext_frac,
            )
            raw_measurements[strat].append(meas)

    aggregated: list[AggregatedStrategyOutcome] = []

    for strat in strategies:
        data = raw_measurements[strat]
        reaches = np.array([m.mc_expected_reach for m in data])
        ext_pens = np.array([m.external_comm_penetration for m in data])
        ext_fracs = np.array([m.external_infection_fraction for m in data])
        degs = np.array([m.seed_degree for m in data])
        bets = np.array([m.seed_betweenness for m in data])
        bdrs = np.array([m.seed_bdr for m in data])
        hubs = np.array([1.0 if m.is_hub else 0.0 for m in data])
        mc_stds = np.array([np.sqrt(m.mc_variance_reach) for m in data])

        # Small-sample Student's t 95% Confidence Intervals (df = N_instances - 1)
        m_reach, s_reach, se_reach, ci_reach = compute_student_t_confidence_interval(reaches)
        # Bounded reach CI [1.0, N]
        ci_reach = (max(1.0, ci_reach[0]), ci_reach[1])

        m_pen, s_pen, se_pen, ci_pen = compute_student_t_confidence_interval(ext_pens)
        ci_pen = (max(0.0, ci_pen[0]), min(1.0, ci_pen[1]))

        m_frac, s_frac, se_frac, ci_frac = compute_student_t_confidence_interval(ext_fracs)
        ci_frac = (max(0.0, ci_frac[0]), min(1.0, ci_frac[1]))

        aggregated.append(
            AggregatedStrategyOutcome(
                strategy_name=strat,
                num_instances=num_instances,
                num_mc_per_instance=num_mc_per_instance,
                mean_seed_degree=float(np.mean(degs)),
                mean_seed_betweenness=float(np.mean(bets)),
                mean_seed_bdr=float(np.mean(bdrs)),
                mean_expected_reach=m_reach,
                std_expected_reach=s_reach,
                se_expected_reach=se_reach,
                ci95_expected_reach=ci_reach,
                within_graph_mc_std=float(np.mean(mc_stds)),
                mean_external_comm_penetration=m_pen,
                ci95_external_comm_penetration=ci_pen,
                mean_external_infection_fraction=m_frac,
                ci95_external_infection_fraction=ci_frac,
                overlap_with_hub_rate=float(np.mean(hubs)),
                instance_measurements=tuple(data),
            )
        )

    return aggregated

"""Degree distribution and statistical power-law analysis module for ViralGraph.

This module provides mathematical tools to evaluate:
1. Degree distribution moments: Mean, Variance, Skewness, Kurtosis.
2. Degree Assortativity Coefficient (Newman's r): Homophilic degree correlation.
3. Power-law scaling parameter estimation (Clauset-Shalizi-Newman MLE).
4. Goodness-of-fit evaluation and academic discussion of finite-size cutoffs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import networkx as nx
import numpy as np
import scipy.stats as stats


@dataclass(frozen=True)
class DegreeProfile:
    """Immutable statistical profile of a graph's degree distribution."""

    # Fundamental moments
    mean_degree: float
    variance_degree: float
    std_degree: float
    skewness: float
    kurtosis: float

    # Extrema
    min_degree: int
    max_degree: int

    # Assortativity coefficient r in [-1.0, 1.0]
    assortativity: float

    # Power-law estimates (P(k) ~ k^(-gamma))
    estimated_power_law_gamma: float
    k_min_cutoff: int
    is_heavy_tailed: bool  # Variance > Mean (over-dispersion)


def analyze_degree_distribution(graph: nx.Graph) -> DegreeProfile:
    """Perform comprehensive statistical analysis of the graph's degree sequence.

    Mathematical Formulation:
        - Mean: <k> = (1/n) * sum k_i = 2m / n.
        - Variance: sigma_k^2 = (1/n) * sum (k_i - <k>)^2.
        - Skewness: gamma_1 = (1/n) * sum [ (k_i - <k>) / sigma_k ]^3. (Positive indicates heavy right tail).
        - Degree Assortativity: Pearson correlation coefficient of degrees at either ends of an edge:
            r = sum_{jk} jk (e_jk - q_j q_k) / sigma_q^2.
            Social networks are typically assortative (r > 0: popular nodes follow popular nodes).
        - Power-Law Scaling Parameter (Clauset et al., 2009):
            gamma_hat = 1 + n_tail * [ sum_{i: k_i >= k_min} ln(k_i / (k_min - 0.5)) ]^(-1).
    """
    n = graph.number_of_nodes()
    if n == 0:
        return DegreeProfile(
            mean_degree=0.0,
            variance_degree=0.0,
            std_degree=0.0,
            skewness=0.0,
            kurtosis=0.0,
            min_degree=0,
            max_degree=0,
            assortativity=0.0,
            estimated_power_law_gamma=0.0,
            k_min_cutoff=0,
            is_heavy_tailed=False,
        )

    degrees = np.array([deg for _, deg in graph.degree()], dtype=float)

    mean_deg = float(np.mean(degrees))
    var_deg = float(np.var(degrees))
    std_deg = float(np.std(degrees))
    skew = float(stats.skew(degrees)) if std_deg > 0 else 0.0
    kurt = float(stats.kurtosis(degrees)) if std_deg > 0 else 0.0

    min_deg = int(np.min(degrees))
    max_deg = int(np.max(degrees))

    # Assortativity coefficient r
    if graph.number_of_edges() > 0:
        assortativity = float(nx.degree_assortativity_coefficient(graph))
        if np.isnan(assortativity):
            assortativity = 0.0
    else:
        assortativity = 0.0

    # Over-dispersion / Heavy-tail indicator: Variance substantially exceeds Mean
    is_heavy_tailed = bool(var_deg > mean_deg and skew > 1.0)

    # Estimate power-law exponent gamma using discrete MLE on upper tail
    # Default k_min cutoff set to max(min_deg, 2) or 25th percentile
    k_min = int(max(min_deg, 2))
    tail_degrees = degrees[degrees >= k_min]
    if len(tail_degrees) > 10 and max_deg > k_min:
        # Clauset-Shalizi-Newman discrete approximation
        log_ratios = np.log(tail_degrees / (k_min - 0.5))
        sum_log = float(np.sum(log_ratios))
        if sum_log > 0:
            gamma_mle = float(1.0 + len(tail_degrees) / sum_log)
        else:
            gamma_mle = 0.0
    else:
        gamma_mle = 0.0

    return DegreeProfile(
        mean_degree=mean_deg,
        variance_degree=var_deg,
        std_degree=std_deg,
        skewness=skew,
        kurtosis=kurt,
        min_degree=min_deg,
        max_degree=max_deg,
        assortativity=assortativity,
        estimated_power_law_gamma=gamma_mle,
        k_min_cutoff=k_min,
        is_heavy_tailed=is_heavy_tailed,
    )

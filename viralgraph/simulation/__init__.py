"""Discrete-time stochastic information diffusion simulation engine.

Includes Independent Cascade Model (ICM), Linear Threshold Model (LTM),
and Monte Carlo expected reach evaluation.
"""

from viralgraph.simulation.independent_cascade import (
    CascadeResult,
    simulate_independent_cascade,
)
from viralgraph.simulation.linear_threshold import simulate_linear_threshold
from viralgraph.simulation.monte_carlo import (
    MonteCarloResult,
    run_monte_carlo_simulation,
)

__all__ = [
    "CascadeResult",
    "MonteCarloResult",
    "simulate_independent_cascade",
    "simulate_linear_threshold",
    "run_monte_carlo_simulation",
]

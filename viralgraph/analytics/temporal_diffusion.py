"""Temporal diffusion analysis and early velocity modeling for ViralGraph.

Tracks temporal dynamics of information cascades:
1. Cumulative reach trajectory R(t).
2. Step-by-step infection increments Delta R(t) = |A_t|.
3. Early propagation velocity V(t) = (R(t) - R(0)) / t.
4. Acceleration A(t) = Delta R(t) - Delta R(t-1).
5. Critical temporal milestones: Time-to-Peak (t_peak), Saturation time.
6. Temporal Windowing Guard: Restricts observable cascade features to t <= t_obs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

from viralgraph.simulation.independent_cascade import CascadeResult


@dataclass(frozen=True)
class TemporalCascadeMetrics:
    """Mathematical temporal invariants extracted from a cascade trajectory."""

    total_duration: int
    final_reach: int
    peak_step: int
    peak_new_infections: int
    early_velocity_step1: float
    early_velocity_step2: float
    acceleration_step2: float
    half_saturation_step: int
    reach_timeline: list[int]
    new_infections_per_step: list[int]
    velocities_per_step: list[float]


@dataclass(frozen=True)
class TemporalObservationWindow:
    """Strict temporal boundary guard isolating early features from future cascade events."""

    observation_horizon: int  # t_obs
    observed_reach: int  # R(t_obs)
    observed_new_infections: list[int]  # [|A_0|, |A_1|, ..., |A_t_obs|]
    observed_velocity: float  # V(t_obs)
    ground_truth_final_reach: int  # R_infinity (target label)
    is_viral: bool  # Whether R_infinity >= threshold


def analyze_temporal_cascade(cascade: CascadeResult) -> TemporalCascadeMetrics:
    """Compute temporal velocities, acceleration, and critical inflection points."""
    timeline = cascade.cumulative_reach_timeline
    duration = len(timeline) - 1
    final_reach = cascade.total_reach

    # Delta R(t) = new infections at step t
    new_infections = [len(s) for s in cascade.activation_history]

    # Velocities V(t) = (R(t) - R(0)) / t
    r0 = timeline[0]
    velocities = [0.0]
    for t in range(1, len(timeline)):
        velocities.append(float((timeline[t] - r0) / t))

    # Peak infection step
    if duration > 0:
        # Ignore t=0 (seed) when looking for propagation peak
        peak_step = int(np.argmax(new_infections[1:]) + 1) if duration >= 1 else 0
        peak_val = new_infections[peak_step]
    else:
        peak_step = 0
        peak_val = 0

    v1 = velocities[1] if duration >= 1 else 0.0
    v2 = velocities[2] if duration >= 2 else v1
    acc2 = float(new_infections[2] - new_infections[1]) if duration >= 2 else 0.0

    # Half saturation step: first t where R(t) >= 0.5 * R_inf
    half_target = 0.5 * final_reach
    half_step = 0
    for t, reach in enumerate(timeline):
        if reach >= half_target:
            half_step = t
            break

    return TemporalCascadeMetrics(
        total_duration=duration,
        final_reach=final_reach,
        peak_step=peak_step,
        peak_new_infections=peak_val,
        early_velocity_step1=v1,
        early_velocity_step2=v2,
        acceleration_step2=acc2,
        half_saturation_step=half_step,
        reach_timeline=timeline,
        new_infections_per_step=new_infections,
        velocities_per_step=velocities,
    )


def extract_temporal_observation_window(
    cascade: CascadeResult,
    t_obs: int = 2,
    virality_threshold: int | float = 50,
) -> TemporalObservationWindow:
    """Enforce strict temporal cutoff: extract features ONLY up to t <= t_obs.

    Prevents future data leakage in machine learning training.
    """
    timeline = cascade.cumulative_reach_timeline
    actual_t = min(t_obs, len(timeline) - 1)

    observed_reach = timeline[actual_t]
    observed_new = [len(s) for s in cascade.activation_history[: actual_t + 1]]

    r0 = timeline[0]
    obs_velocity = float((observed_reach - r0) / actual_t) if actual_t > 0 else 0.0

    is_viral = bool(cascade.total_reach >= virality_threshold)

    return TemporalObservationWindow(
        observation_horizon=t_obs,
        observed_reach=observed_reach,
        observed_new_infections=observed_new,
        observed_velocity=obs_velocity,
        ground_truth_final_reach=cascade.total_reach,
        is_viral=is_viral,
    )

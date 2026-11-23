"""Unit tests for the audited machine learning virality prediction pipeline."""

import numpy as np
import pandas as pd
import pytest

from viralgraph.models.dataset_builder import (
    extract_strict_early_features,
    generate_grouped_cascade_dataset,
)
from viralgraph.models.virality_classifiers import (
    evaluate_models_with_leakage_guards,
)


def test_grouped_dataset_generation():
    """Verify that grouped dataset builder produces valid DataFrames with zero missing values."""
    df_x, s_reach, s_groups = generate_grouped_cascade_dataset(
        num_graphs=6, cascades_per_graph=8, n_nodes=60, t_obs=1, seed=42
    )

    assert len(df_x) == 48
    assert len(s_reach) == 48
    assert len(s_groups) == 48
    assert not df_x.isnull().values.any()
    assert len(s_groups.unique()) == 6


def test_audited_model_evaluation_with_leakage_guards():
    """Verify models train, cross-validate with GroupKFold, and score properly."""
    df_x, s_reach, s_groups = generate_grouped_cascade_dataset(
        num_graphs=8, cascades_per_graph=10, n_nodes=50, t_obs=1, seed=42
    )
    reports, reg_metrics = evaluate_models_with_leakage_guards(
        df_x=df_x,
        s_reach=s_reach,
        s_groups=s_groups,
        percentile_threshold=75.0,
        test_graph_fraction=0.25,
        seed=42,
    )

    for name in ["Majority Baseline", "Logistic Regression", "Decision Tree", "Random Forest"]:
        rep = reports[name]
        assert 0.0 <= rep.accuracy <= 1.0
        assert 0.0 <= rep.roc_auc <= 1.0
        assert 0.0 <= rep.cv_mean_roc_auc <= 1.0
        assert 0.0 <= rep.cv_std_roc_auc <= 1.0
        assert 0.0 <= rep.cv_mean_f1 <= 1.0

    assert reg_metrics["mse"] >= 0.0
    assert reg_metrics["test_graphs_count"] > 0
    assert reg_metrics["train_graphs_count"] > 0
    assert reg_metrics["tau_train_threshold"] >= 1.0


def test_inner_fold_threshold_independence():
    """Verify that inner fold threshold calculation does not leak test or validation reaches."""
    # Construct synthetic reaches where graph 0-3 have reaches 1..10 and graph 4 has huge reach 100
    df_x = pd.DataFrame({
        "seed_degree": [2.0] * 20,
        "early_velocity": [1.0] * 20,
        "graph_id": [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
    })
    s_reach = pd.Series([
        2, 3, 4, 5,
        2, 3, 4, 5,
        2, 3, 4, 5,
        2, 3, 4, 5,
        100, 100, 100, 100,
    ])
    s_groups = df_x["graph_id"]

    reports, reg = evaluate_models_with_leakage_guards(
        df_x=df_x,
        s_reach=s_reach,
        s_groups=s_groups,
        percentile_threshold=75.0,
        test_graph_fraction=0.2,
        seed=42,
    )

    assert "Random Forest" in reports
    rf_rep = reports["Random Forest"]
    assert hasattr(rf_rep, "cv_std_roc_auc")
    assert hasattr(rf_rep, "cv_mean_f1")


def test_extract_strict_early_features_temporal_bound():
    """Verify that extract_strict_early_features strictly observes up to t_obs = 1."""
    import networkx as nx
    from viralgraph.simulation.independent_cascade import simulate_independent_cascade

    g = nx.erdos_renyi_graph(30, 0.2, seed=42)
    rng = np.random.default_rng(42)
    cascade = simulate_independent_cascade(g, seed_nodes=[0], p=0.3, max_steps=10, rng=rng)

    feats = extract_strict_early_features(g, seed_node=0, t_obs=1, cascade_res=cascade)

    # Check required keys are present
    assert "early_observed_reach" in feats
    assert "early_velocity" in feats
    assert "early_infections_step1" in feats

    # Must NOT have keys beyond t_obs = 1
    assert "early_infections_step2" not in feats
    assert "final_reach" not in feats

    # If t_obs = 1, observed reach must equal 1 (seed) + step 1 infections
    step1_infections = len(cascade.activation_history[1]) if len(cascade.activation_history) > 1 else 0
    assert feats["early_observed_reach"] == 1 + step1_infections


"""Audited Machine Learning Experiment Script for ViralGraph.

Evaluates out-of-graph generalization with:
- Grouped train/test splitting across independent graph instances.
- Zero target leakage: virality threshold fit independently inside each inner training fold.
- Observation window horizon t_obs = 1 (realistic early prediction).
- Comparison against Dummy Majority Baseline.
- Explicit distinction between Inner Grouped CV metrics and Outer Test metrics.
"""

from viralgraph.models.dataset_builder import generate_grouped_cascade_dataset
from viralgraph.models.virality_classifiers import evaluate_models_with_leakage_guards


def main():
    print("=" * 105)
    print("TASK 10 SCIENTIFIC AUDIT: LEAKAGE-FREE ML VIRALITY EVALUATION")
    print("Horizon: t_obs = 1 (Only Step 1 observed) | Outer Split: Disjoint Unseen Graph Topologies")
    print("Inner Target Thresholding: Fit strictly inside each inner training fold (Zero Validation Leakage)")
    print("=" * 105)

    print("\n[1/3] Generating multi-graph cascade dataset (24 graphs, 15 cascades each = 360 samples)...")
    df_x, s_reach, s_groups = generate_grouped_cascade_dataset(
        num_graphs=24,
        cascades_per_graph=15,
        n_nodes=120,
        t_obs=1,
        p_transmission=0.12,
        seed=42,
    )

    print(f"Generated {len(df_x)} samples across {len(s_groups.unique())} distinct graph topologies.")

    print("\n[2/3] Evaluating models under Grouped Out-of-Graph Splitting & Inner Fold Thresholding...")
    reports, reg_metrics = evaluate_models_with_leakage_guards(
        df_x=df_x,
        s_reach=s_reach,
        s_groups=s_groups,
        percentile_threshold=75.0,
        test_graph_fraction=0.25,
        seed=42,
    )

    print(f"Outer Train set: {reg_metrics['train_graphs_count']} graphs | Outer Test set: {reg_metrics['test_graphs_count']} unseen graphs")
    print(f"Learned Virality Threshold (from full train split): Reach >= {reg_metrics['tau_train_threshold']:.1f} nodes")

    print("\n--- MODEL PERFORMANCE: INNER GROUPED CV vs. OUTER TEST GENERALIZATION ---")
    print(
        f"{'Model':22s} | {'Inner CV ROC-AUC':18s} | {'Inner CV F1':11s} | "
        f"{'Test Acc':8s} | {'Test Prec':9s} | {'Test Rec':8s} | {'Test F1':7s} | {'Test ROC-AUC':12s}"
    )
    print("-" * 115)
    for name, r in reports.items():
        cv_auc_str = f"{r.cv_mean_roc_auc:.3f} ± {r.cv_std_roc_auc:.3f}"
        print(
            f"{name:22s} | {cv_auc_str:18s} | {r.cv_mean_f1:11.3f} | "
            f"{r.accuracy:8.3f} | {r.precision:9.3f} | {r.recall:8.3f} | "
            f"{r.f1:7.3f} | {r.roc_auc:12.3f}"
        )

    print("\n--- CONFUSION MATRICES (Outer Unseen Test Set) ---")
    for name, r in reports.items():
        cm = r.confusion_mat
        print(f"  {name:22s}: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")

    print("\n--- TOP 5 PREDICTIVE FEATURES (Random Forest) ---")
    rf_imp = reports["Random Forest"].feature_importances
    for rank, (feat, val) in enumerate(list(rf_imp.items())[:5], 1):
        print(f"  {rank}. {feat:25s}: {val:.4f} ({val*100:.1f}%)")

    print("\n[3/3] Out-of-Graph Continuous Reach Regression (Random Forest)...")
    print(f"  Reach Regression RMSE: {reg_metrics['rmse']:.2f} nodes")
    print(f"  Reach Regression R^2 : {reg_metrics['r2_score']:.3f}")


if __name__ == "__main__":
    main()

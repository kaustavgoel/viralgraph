"""Executable script for Task 8 Research Experiment (Audited Multi-Instance Version).

Evaluates the central question:
"Can a community-bridging node generate a larger or more cross-cutting cascade than a high-degree hub?"
Across 5 independent graph instances per topology, with 40 Monte Carlo runs per instance.

Scientific Audit Enhancements:
- Primary sampling units are independent graph realizations (N=5).
- Student's t-distribution 95% Confidence Intervals with df = 4 (t_crit = 2.776).
- Isolates topological variation (s_topo) from within-graph Monte Carlo variance (s_MC).
- Formal paired t-tests on identical graph realizations for Hub vs. Bridge comparison.
- Exact formulas for external infection fraction and external community penetration.
"""

from viralgraph.analytics.influencer_research import (
    compare_strategies_paired,
    run_multi_instance_influencer_experiment,
)
from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)


def main():
    base_seed = 42
    n = 150
    p_transmission = 0.12
    num_instances = 5
    num_mc = 40

    print("=" * 115)
    print("TASK 8 SCIENTIFIC AUDIT: MULTI-INSTANCE INFLUENCER VS. BRIDGE EXPERIMENT")
    print(f"Network Order N={n}, Transmission p={p_transmission}, Instances={num_instances}, MC Trials/Inst={num_mc}")
    print("Inference: Small-sample Student's t 95% CIs (df=4, t_crit=2.776) | Primary Sampling Unit = Graph Instance")
    print("=" * 115)

    experiments = [
        (
            "Barabási–Albert (Scale-Free)",
            generate_barabasi_albert,
            {"n": n, "m": 3},
            "Detected via Louvain heuristic modularity optimization (no planted partition)",
        ),
        (
            "Watts–Strogatz (Small-World)",
            generate_watts_strogatz,
            {"n": n, "k": 6, "p": 0.08},
            "Detected via Louvain heuristic modularity optimization (no planted partition)",
        ),
        (
            "Stochastic Block Model (Communities)",
            generate_stochastic_block_model,
            {
                "community_sizes": [50, 50, 50],
                "probability_matrix": [
                    [0.10, 0.005, 0.002],
                    [0.005, 0.10, 0.005],
                    [0.002, 0.005, 0.10],
                ],
            },
            "Ground-truth planted block partition (K=3 blocks)",
        ),
    ]

    for net_title, gen_fn, gen_kwargs, comm_prov in experiments:
        print(f"\n>>> TOPOLOGY: {net_title}")
        print(f"    Community Provenance: {comm_prov}")
        print(
            f"{'Strategy':20s} | {'Mean Deg':8s} | {'Mean BDR':8s} | {'Hub Overlap':11s} | "
            f"{'E[Reach] ± 95% t-CI':22s} | {'s_topo (SE)':13s} | {'s_MC':6s} | "
            f"{'Ext Comm Pen':12s} | {'Ext Inf Frac':12s}"
        )
        print("-" * 125)

        outcomes = run_multi_instance_influencer_experiment(
            graph_generator=gen_fn,
            graph_gen_kwargs=gen_kwargs,
            num_instances=num_instances,
            num_mc_per_instance=num_mc,
            p_transmission=p_transmission,
            base_seed=base_seed,
        )

        for o in outcomes:
            ci_reach_str = f"{o.mean_expected_reach:.2f} [{o.ci95_expected_reach[0]:.2f}, {o.ci95_expected_reach[1]:.2f}]"
            topo_str = f"{o.std_expected_reach:.2f} ({o.se_expected_reach:.2f})"
            pen_str = f"{o.mean_external_comm_penetration*100:5.1f}%"
            frac_str = f"{o.mean_external_infection_fraction*100:5.1f}%"
            overlap_str = f"{o.overlap_with_hub_rate*100:5.1f}%"
            print(
                f"{o.strategy_name:20s} | {o.mean_seed_degree:8.1f} | {o.mean_seed_bdr:8.2f} | "
                f"{overlap_str:11s} | {ci_reach_str:22s} | {topo_str:13s} | {o.within_graph_mc_std:6.2f} | "
                f"{pen_str:12s} | {frac_str:12s}"
            )

        # Formal Paired Hypothesis Testing: Hub vs. Community Bridge
        print("\n    [Formal Paired Tests across N=5 identical graph realizations (df=4)]")
        pair_reach = compare_strategies_paired(
            outcomes, "highest_degree", "community_bridge", metric="reach"
        )
        print(
            f"    - Reach Difference (Hub - Bridge): Delta = {pair_reach.mean_difference:+.2f} "
            f"[{pair_reach.ci95_difference[0]:+.2f}, {pair_reach.ci95_difference[1]:+.2f}], "
            f"t({pair_reach.degrees_of_freedom}) = {pair_reach.t_statistic:.2f}, "
            f"p = {pair_reach.p_value:.4f}, paired Cohen's d = {pair_reach.cohens_d:.2f} "
            f"({'Significant' if pair_reach.is_statistically_significant else 'Not Significant at alpha=0.05'})"
        )

        pair_ext = compare_strategies_paired(
            outcomes, "community_bridge", "highest_degree", metric="external_fraction"
        )
        print(
            f"    - External Infection Difference (Bridge - Hub): Delta_rho = {pair_ext.mean_difference*100:+.1f}% "
            f"[{pair_ext.ci95_difference[0]*100:+.1f}%, {pair_ext.ci95_difference[1]*100:+.1f}%], "
            f"t({pair_ext.degrees_of_freedom}) = {pair_ext.t_statistic:.2f}, "
            f"p = {pair_ext.p_value:.4f}, paired Cohen's d = {pair_ext.cohens_d:.2f} "
            f"({'Significant' if pair_ext.is_statistically_significant else 'Not Significant at alpha=0.05'})"
        )


if __name__ == "__main__":
    main()

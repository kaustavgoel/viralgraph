"""Post Studio page for ViralGraph (Milestone 2 Content & Diffusion Architecture)."""

from __future__ import annotations

import streamlit as st
import networkx as nx

from app.components.diffusion_launcher import launch_diffusion_simulation, resolve_seed_node
from app.components.header import render_page_header
from app.state import PostData, SessionManager, SimulationConfig
from viralgraph.data.synthetic_generator import (
    generate_barabasi_albert,
    generate_erdos_renyi,
    generate_stochastic_block_model,
    generate_watts_strogatz,
)


def render_post_studio_page() -> None:
    """Render the complete Post Studio and content injection interface."""
    render_page_header(
        title="Post Studio",
        subtitle="Compose hypothetical social media content, configure active graph topology, assign seed injection nodes, and launch discrete-time diffusion.",
        badge="Milestone 2 Active",
        category="SIMULATION STUDIO",
    )

    # Retrieve current session state
    post = SessionManager.get_current_post()
    sim_config = SessionManager.get_simulation_config()
    graph = SessionManager.get_current_graph()

    # -------------------------------------------------------------------------
    # SECTION A: POST COMPOSER
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="vg-section-header">
            <span class="vg-section-title">A. Post Composer & Payload Configuration</span>
            <span style="font-size: 0.82rem; color: #94a3b8;">Simulated Message Payload</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_edit, col_prev = st.columns([7, 5], gap="large")

    with col_edit:
        new_title = st.text_input(
            "Post Title / Headline",
            value=post.title,
            max_chars=120,
            help="Headline of the simulated social media post acting as the contagion payload.",
        )
        new_content = st.text_area(
            "Post Body / Content",
            value=post.content,
            height=110,
            max_chars=400,
            help="Main message text being diffused across network edges.",
        )

        col_top, col_med = st.columns(2)
        with col_top:
            topic_options = [
                "Network Science & Computing",
                "Public Health & Epidemiology",
                "Financial Markets & Economics",
                "Social Policy & Governance",
                "Culture & Entertainment",
                "General Discussion",
            ]
            current_topic_idx = topic_options.index(post.topic) if post.topic in topic_options else 0
            new_topic = st.selectbox("Topic / Category", options=topic_options, index=current_topic_idx)
        with col_med:
            media_options = ["Text Only", "Text + Interactive Diagram", "Text + Data Table", "Video / Visual Clip"]
            current_med_idx = media_options.index(post.media_type) if post.media_type in media_options else 1
            new_media = st.selectbox("Media Format", options=media_options, index=current_med_idx)

        new_desc = st.text_area(
            "Experiment Notes / Hypothesis (Optional)",
            value=post.description,
            height=60,
            placeholder="e.g., Testing whether bridge seeding outperforms hub seeding under ICM with p=0.12...",
        )

        # Update post in session state if changed
        if (
            new_title != post.title
            or new_content != post.content
            or new_topic != post.topic
            or new_desc != post.description
            or new_media != post.media_type
        ):
            post = PostData(
                title=new_title,
                content=new_content,
                topic=new_topic,
                description=new_desc,
                media_type=new_media,
            )
            SessionManager.set_current_post(post)

    with col_prev:
        st.markdown("**📱 Social Card Preview**")
        st.markdown(
            f"""
            <div class="vg-card" style="border-left: 4px solid #3b82f6; margin-bottom: 8px;">
                <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 8px;">
                    <div style="display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.5rem;">👤</span>
                        <div>
                            <div style="font-weight: 700; color: #f8fafc; font-size: 0.92rem;">ViralGraph Researcher</div>
                            <div style="font-size: 0.75rem; color: #64748b;">@diffusion_lab · Seed Node Active</div>
                        </div>
                    </div>
                    <span class="vg-badge">{post.topic}</span>
                </div>
                <div style="font-size: 1.02rem; font-weight: 700; color: #ffffff; margin-bottom: 6px;">
                    {post.title if post.title else 'Untitled Post'}
                </div>
                <p style="color: #cbd5e1; font-size: 0.88rem; line-height: 1.5; margin-bottom: 12px;">
                    {post.content if post.content else 'Enter content in the composer to view preview...'}
                </p>
                <div style="font-size: 0.76rem; color: #94a3b8; border-top: 1px solid rgba(255, 255, 255, 0.08); padding-top: 8px; display: flex; justify-content: space-between;">
                    <span>Format: <b>{post.media_type}</b></span>
                    <span>Payload ID: <b>VG-PAYLOAD-01</b></span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.caption(
            "📌 **Methodological Note**: The post functions as the informational payload. "
            "In accordance with mathematical rigor, message wording does not arbitrarily mutate "
            "edge transmission probabilities or thresholds, which are governed strictly by the structural diffusion models."
        )

    # -------------------------------------------------------------------------
    # SECTION B: NETWORK SELECTION
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="vg-section-header">
            <span class="vg-section-title">B. Network Topology Selection</span>
            <span style="font-size: 0.82rem; color: #94a3b8;">Underlying Contagion Graph</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    net_mode = st.radio(
        "Network Source",
        options=["Use Active Network from Session", "Synthesize New Network"],
        horizontal=True,
    )

    if net_mode == "Use Active Network from Session":
        model_label = st.session_state.get("model_label", "Custom Network")
        metrics = SessionManager.get_metrics()
        n_nodes = graph.number_of_nodes()
        n_edges = graph.number_of_edges()
        mean_k = metrics.get("mean_degree", 0.0)
        clustering = metrics.get("avg_clustering", 0.0)
        is_conn = "Connected" if metrics.get("is_connected", True) else "Disconnected"

        st.markdown(
            f"""
            <div class="vg-card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">📍 Active Network: {model_label}</span>
                    <span class="vg-badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.3);">● {is_conn}</span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-size: 0.88rem; color: #cbd5e1;">
                    <div>Vertices (|V|): <b style="color: #60a5fa;">{n_nodes}</b></div>
                    <div>Edges (|E|): <b style="color: #60a5fa;">{n_edges}</b></div>
                    <div>Mean Degree (⟨k⟩): <b>{mean_k:.2f}</b></div>
                    <div>Avg Clustering (⟨C⟩): <b>{clustering:.3f}</b></div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="font-size: 0.86rem; color: #cbd5e1; margin-bottom: 12px;">
                Configure and generate a new graph. <b>Note</b>: Generating a new network will update the active graph across all platform pages (including Network Explorer).
            </div>
            """,
            unsafe_allow_html=True,
        )

        col_gen1, col_gen2 = st.columns([4, 8], gap="medium")
        with col_gen1:
            gen_choice = st.selectbox(
                "Graph Model Generator",
                options=[
                    "Barabási–Albert (Scale-Free)",
                    "Watts–Strogatz (Small-World)",
                    "Stochastic Block Model (Communities)",
                    "Erdős–Rényi (Random Graph)",
                ],
            )
            gen_seed = st.number_input("Generator Seed", min_value=1, max_value=999999, value=42, step=1)

        with col_gen2:
            if "Barabási" in gen_choice:
                c1, c2 = st.columns(2)
                with c1:
                    ba_n = st.slider("Vertices (N)", min_value=30, max_value=500, value=150, step=10)
                with c2:
                    ba_m = st.slider("Edges per new node (m)", min_value=1, max_value=8, value=3, step=1)
            elif "Watts" in gen_choice:
                c1, c2, c3 = st.columns(3)
                with c1:
                    ws_n = st.slider("Vertices (N)", min_value=30, max_value=500, value=150, step=10)
                with c2:
                    ws_k = st.slider("Lattice Degree (k, even)", min_value=2, max_value=12, value=6, step=2)
                with c3:
                    ws_p = st.slider("Rewiring Prob (p)", min_value=0.0, max_value=1.0, value=0.08, step=0.01)
            elif "Block Model" in gen_choice:
                c1, c2, c3 = st.columns(3)
                with c1:
                    sbm_n = st.slider("Vertices per block (K=3)", min_value=15, max_value=100, value=50, step=5)
                with c2:
                    sbm_pin = st.slider("Intra-Block Prob (P_in)", min_value=0.02, max_value=0.30, value=0.10, step=0.01)
                with c3:
                    sbm_pout = st.slider("Inter-Block Prob (P_out)", min_value=0.001, max_value=0.03, value=0.005, step=0.001)
            else:
                c1, c2 = st.columns(2)
                with c1:
                    er_n = st.slider("Vertices (N)", min_value=20, max_value=500, value=120, step=10)
                with c2:
                    er_p = st.slider("Edge Probability (p)", min_value=0.01, max_value=0.30, value=0.05, step=0.01)

        if st.button("⚡ Generate and Set as Active Network", type="secondary"):
            try:
                with st.spinner("Generating new network topology..."):
                    if "Barabási" in gen_choice:
                        new_g = generate_barabasi_albert(n=ba_n, m=ba_m, seed=gen_seed)
                        model_type = "barabasi_albert"
                        model_label = "Barabási–Albert (Scale-Free)"
                        params = {"n": ba_n, "m": ba_m, "seed": gen_seed}
                        comm_map = None
                    elif "Watts" in gen_choice:
                        new_g = generate_watts_strogatz(n=ws_n, k=ws_k, p=ws_p, seed=gen_seed)
                        model_type = "watts_strogatz"
                        model_label = "Watts–Strogatz (Small-World)"
                        params = {"n": ws_n, "k": ws_k, "p": ws_p, "seed": gen_seed}
                        comm_map = None
                    elif "Block Model" in gen_choice:
                        sizes = [sbm_n, sbm_n, sbm_n]
                        prob_matrix = [
                            [sbm_pin, sbm_pout, sbm_pout],
                            [sbm_pout, sbm_pin, sbm_pout],
                            [sbm_pout, sbm_pout, sbm_pin],
                        ]
                        new_g, comm_map = generate_stochastic_block_model(
                            community_sizes=sizes,
                            probability_matrix=prob_matrix,
                            seed=gen_seed,
                        )
                        model_type = "stochastic_block_model"
                        model_label = "Stochastic Block Model (Communities)"
                        params = {"community_sizes": sizes, "probability_matrix": prob_matrix, "seed": gen_seed}
                    else:
                        new_g = generate_erdos_renyi(n=er_n, p=er_p, seed=gen_seed)
                        model_type = "erdos_renyi"
                        model_label = "Erdős–Rényi (Random Graph)"
                        params = {"n": er_n, "p": er_p, "seed": gen_seed}
                        comm_map = None

                    SessionManager.set_current_graph(
                        graph=new_g,
                        model_type=model_type,
                        model_label=model_label,
                        params=params,
                        community_map=comm_map,
                    )
                    graph = new_g
                    st.success(f"Generated {model_label} (|V| = {new_g.number_of_nodes()}, |E| = {new_g.number_of_edges()})! Active graph updated.")
            except Exception as exc:
                st.error(f"Network generation error: {exc}")

    # -------------------------------------------------------------------------
    # SECTION C: SEED NODE SELECTION
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="vg-section-header">
            <span class="vg-section-title">C. Seed Node Selection</span>
            <span style="font-size: 0.82rem; color: #94a3b8;">Initial Contagion Injection Vertex</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if graph.number_of_nodes() == 0:
        st.warning("⚠️ The active network has 0 vertices. Please select or generate a valid network above.")
        seed_resolved = None
        seed_metrics = {}
    else:
        col_strat, col_manual = st.columns([7, 5], gap="large")

        with col_strat:
            strategy_choices = {
                "highest_degree": "Highest-Degree Node (Local Hub)",
                "highest_betweenness": "Highest-Betweenness Node (Structural Bridge)",
                "random": "Random Node (Baseline)",
                "manual": "Manual Node Identifier",
            }
            current_strat = SessionManager.get_seed_strategy()
            strat_keys = list(strategy_choices.keys())
            strat_idx = strat_keys.index(current_strat) if current_strat in strat_keys else 0

            chosen_strat = st.radio(
                "Seed Selection Strategy",
                options=strat_keys,
                format_func=lambda k: strategy_choices[k],
                index=strat_idx,
                horizontal=False,
            )
            SessionManager.set_seed_strategy(chosen_strat)

            seed_rand_int = st.number_input(
                "Seed Selection Random State", min_value=1, max_value=999999, value=42, step=1,
                help="Seed used for tie-breaking and random seed selection."
            )

        with col_manual:
            all_node_list = sorted(list(graph.nodes()))
            if chosen_strat == "manual":
                manual_input = st.selectbox(
                    "Select Exact Node Identifier",
                    options=all_node_list,
                    index=0,
                    help="Choose an existing vertex ID from the active graph.",
                )
            else:
                manual_input = all_node_list[0] if all_node_list else None
                st.caption(f"Strategy will dynamically resolve the optimal vertex across {len(all_node_list)} nodes.")

        # Resolve the seed node
        try:
            seed_resolved, seed_metrics = resolve_seed_node(
                graph=graph,
                strategy=chosen_strat,
                manual_node=manual_input,
                seed=seed_rand_int,
            )
            SessionManager.set_selected_seed(seed_resolved)

            # Display seed properties
            st.markdown(
                f"""
                <div class="vg-card" style="border-left: 4px solid #10b981; margin-top: 8px;">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 6px;">
                        <span style="font-weight: 700; color: #f8fafc; font-size: 1.05rem;">
                            🎯 Resolved Seed: Vertex {seed_resolved}
                        </span>
                        <span class="vg-badge">{strategy_choices[chosen_strat]}</span>
                    </div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; font-size: 0.86rem; color: #cbd5e1;">
                        <div>Degree: <b style="color: #60a5fa;">{seed_metrics['degree']}</b></div>
                        <div>Normalized Degree: <b>{seed_metrics['normalized_degree']:.3f}</b></div>
                        <div>Local Clustering: <b>{seed_metrics['clustering']:.3f}</b></div>
                        <div>Community: <b>{seed_metrics['community']}</b></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        except Exception as exc:
            st.error(f"Seed resolution error: {exc}")
            seed_resolved = None
            seed_metrics = {}

    # -------------------------------------------------------------------------
    # SECTION D: DIFFUSION MODEL CONFIGURATION
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="vg-section-header">
            <span class="vg-section-title">D. Diffusion Model Configuration</span>
            <span style="font-size: 0.82rem; color: #94a3b8;">Discrete-Time Mathematical Engine</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_m1, col_m2 = st.columns([4, 8], gap="large")

    with col_m1:
        model_choice = st.radio(
            "Diffusion Engine",
            options=["ICM", "LTM"],
            format_func=lambda m: "Independent Cascade Model (ICM)" if m == "ICM" else "Linear Threshold Model (LTM)",
            index=0 if sim_config.model == "ICM" else 1,
            help="ICM: Sender-centric Bernoulli contagion. LTM: Receiver-centric complex contagion.",
        )
        sim_seed = st.number_input(
            "Stochastic Simulation Seed",
            min_value=1,
            max_value=999999,
            value=sim_config.random_seed,
            step=1,
            help="Ensures reproducible diffusion trajectories.",
        )

    with col_m2:
        if model_choice == "ICM":
            st.markdown("**ICM Parameters (Sender-Centric Discrete Contagion)**")
            col_p, col_steps = st.columns(2)
            with col_p:
                p_trans = st.slider(
                    "Transmission Probability (p)",
                    min_value=0.01,
                    max_value=1.00,
                    value=float(sim_config.transmission_prob),
                    step=0.01,
                    help="Independent probability that an active vertex infects each inactive neighbor in step t.",
                )
            with col_steps:
                max_s = st.slider(
                    "Maximum Steps (T_max)",
                    min_value=5,
                    max_value=100,
                    value=int(sim_config.max_steps),
                    step=5,
                    help="Step cutoff horizon to guarantee termination.",
                )

            updated_config = SimulationConfig(
                model="ICM",
                transmission_prob=p_trans,
                max_steps=max_s,
                seed_strategy=chosen_strat,
                selected_seed_node=seed_resolved,
                random_seed=sim_seed,
            )
            SessionManager.set_simulation_config(updated_config)

        else:
            st.markdown("**LTM Parameters (Receiver-Centric Complex Contagion)**")
            col_th_mode, col_th_val = st.columns(2)
            with col_th_mode:
                th_mode = st.radio(
                    "Adoption Thresholds (theta_v)",
                    options=["random_uniform", "constant_uniform"],
                    format_func=lambda m: "Sampled i.i.d. U(0, 1) [Default]" if m == "random_uniform" else "Constant Uniform Threshold",
                    index=0 if sim_config.ltm_threshold_mode == "random_uniform" else 1,
                )
            with col_th_val:
                if th_mode == "constant_uniform":
                    const_th = st.slider(
                        "Constant Threshold (theta)",
                        min_value=0.05,
                        max_value=1.00,
                        value=float(sim_config.ltm_constant_threshold),
                        step=0.05,
                    )
                else:
                    const_th = 0.50
                    st.caption("Each vertex draws independent threshold theta_v ~ Uniform(0, 1).")

            max_s = st.slider(
                "Maximum Steps (T_max)",
                min_value=5,
                max_value=100,
                value=int(sim_config.max_steps),
                step=5,
            )

            st.caption("Influence Weights: b_uv = 1 / deg(v) (Uniform in-degree normalization supported by backend).")

            updated_config = SimulationConfig(
                model="LTM",
                transmission_prob=0.12,
                max_steps=max_s,
                seed_strategy=chosen_strat,
                selected_seed_node=seed_resolved,
                random_seed=sim_seed,
                ltm_threshold_mode=th_mode,
                ltm_constant_threshold=const_th,
            )
            SessionManager.set_simulation_config(updated_config)

    # -------------------------------------------------------------------------
    # SECTION E: EXPERIMENT SUMMARY & LAUNCH
    # -------------------------------------------------------------------------
    st.markdown(
        """
        <div class="vg-section-header">
            <span class="vg-section-title">E. Pre-Launch Experiment Summary</span>
            <span style="font-size: 0.82rem; color: #94a3b8;">Review & Validate Parameters</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Validation checks
    is_valid = True
    validation_errors = []

    if not post.title.strip():
        is_valid = False
        validation_errors.append("Post title cannot be empty.")
    if graph.number_of_nodes() < 2:
        is_valid = False
        validation_errors.append("Active graph must have at least 2 vertices.")
    if seed_resolved is None or seed_resolved not in graph:
        is_valid = False
        validation_errors.append(f"Selected seed node '{seed_resolved}' does not exist in the active graph.")

    # Summary table card
    model_name_str = "Independent Cascade (ICM)" if updated_config.model == "ICM" else "Linear Threshold (LTM)"
    param_str = f"p = {updated_config.transmission_prob:.2f}" if updated_config.model == "ICM" else f"Threshold: {updated_config.ltm_threshold_mode}"

    st.markdown(
        f"""
        <div class="vg-card">
            <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; font-size: 0.88rem; color: #cbd5e1;">
                <div>Payload Title: <b style="color: #ffffff;">{post.title}</b></div>
                <div>Network Topology: <b>{st.session_state.get('model_label', 'Active Graph')}</b></div>
                <div>Graph Order / Size: <b>|V| = {graph.number_of_nodes()}, |E| = {graph.number_of_edges()}</b></div>
                <div>Seed Node: <b style="color: #34d399;">Vertex {seed_resolved}</b></div>
                <div>Seed Strategy: <b>{updated_config.seed_strategy}</b></div>
                <div>Diffusion Model: <b style="color: #60a5fa;">{model_name_str}</b></div>
                <div>Model Parameter: <b>{param_str}</b></div>
                <div>Horizon (T_max): <b>{updated_config.max_steps} steps</b></div>
                <div>Simulation Seed: <b>{updated_config.random_seed}</b></div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not is_valid:
        for err in validation_errors:
            st.error(f"⚠️ Validation Error: {err}")

    # Launch Simulation Button
    col_run, col_status = st.columns([4, 8], gap="medium")

    with col_run:
        launch_clicked = st.button(
            "🚀 Launch Diffusion Simulation",
            type="primary",
            disabled=not is_valid,
            use_container_width=True,
        )

    if launch_clicked:
        try:
            with st.spinner(f"Simulating {model_name_str} cascade starting from seed {seed_resolved}..."):
                comm_map = st.session_state.get("node_community_map")
                res, record = launch_diffusion_simulation(
                    graph=graph,
                    seed_node=seed_resolved,
                    config=updated_config,
                    post=post,
                    community_map=comm_map,
                )

            st.toast(f"Simulation completed! Reached {res.total_reach} nodes in {res.steps_to_termination} steps.", icon="✅")

        except Exception as exc:
            st.error(f"Simulation execution error: {exc}")

    # -------------------------------------------------------------------------
    # DISPLAY RECENT COMPLETED SIMULATION OUTCOME
    # -------------------------------------------------------------------------
    res = SessionManager.get_simulation_results()
    history = SessionManager.get_simulation_history()

    if res is not None and history:
        latest_rec = history[0]
        n_total = graph.number_of_nodes()
        reach_pct = (res.total_reach / n_total) * 100.0 if n_total > 0 else 0.0

        st.markdown("<div style='height: 12px;'></div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="vg-card-elevated" style="border-left: 4px solid #34d399;">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                    <span style="font-weight: 700; color: #ffffff; font-size: 1.1rem;">
                        🎉 Simulation {latest_rec.sim_id} Executed Successfully
                    </span>
                    <span class="vg-badge" style="background: rgba(16, 185, 129, 0.15); color: #34d399; border-color: rgba(16, 185, 129, 0.3);">
                        Completed in {res.steps_to_termination} Epochs
                    </span>
                </div>
                <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; margin-bottom: 12px;">
                    <div style="background: rgba(15, 23, 42, 0.7); padding: 10px 14px; border-radius: 8px;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Final Cascade Reach</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #60a5fa;">
                            {res.total_reach} <span style="font-size: 0.85rem; font-weight: 500; color: #94a3b8;">/ {n_total} ({reach_pct:.1f}%)</span>
                        </div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.7); padding: 10px 14px; border-radius: 8px;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Steps to Termination</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #34d399;">
                            {res.steps_to_termination} <span style="font-size: 0.85rem; font-weight: 500; color: #94a3b8;">steps</span>
                        </div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.7); padding: 10px 14px; border-radius: 8px;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">Transmission Edges</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #f59e0b;">
                            {res.transmission_tree.number_of_edges()} <span style="font-size: 0.85rem; font-weight: 500; color: #94a3b8;">edges</span>
                        </div>
                    </div>
                    <div style="background: rgba(15, 23, 42, 0.7); padding: 10px 14px; border-radius: 8px;">
                        <div style="font-size: 0.75rem; color: #94a3b8; text-transform: uppercase;">External Infection Frac</div>
                        <div style="font-size: 1.3rem; font-weight: 800; color: #a78bfa;">
                            {latest_rec.external_fraction*100:.1f}%
                        </div>
                    </div>
                </div>
                <div style="font-size: 0.84rem; color: #94a3b8; display: flex; justify-content: space-between; align-items: center;">
                    <span>Results saved in session state. Available for Live Diffusion step-by-step playback in Milestone 3.</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

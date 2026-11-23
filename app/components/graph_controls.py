"""Graph control panel component for ViralGraph Streamlit interface.

Handles user input for synthetic graph generation parameters with strict validation
matching the mathematical invariants in viralgraph.data.synthetic_generator.
"""

from __future__ import annotations

import random
from typing import Any
import streamlit as st


def render_graph_controls() -> dict[str, Any]:
    """Render sidebar controls for graph model selection and parameters.

    Returns:
        dict[str, Any]: Configuration dictionary containing:
            - "model": Selected model name ("barabasi_albert", "watts_strogatz", "stochastic_block_model")
            - "params": Model-specific parameter dictionary
            - "seed": Random seed integer
            - "generate_triggered": Boolean indicating whether the generate button was pressed
    """
    st.sidebar.markdown("## 🕸️ Network Architecture")
    st.sidebar.caption("Select a generative discrete graph model and configure its structural parameters.")

    model_options = {
        "Barabási–Albert (Scale-Free)": "barabasi_albert",
        "Watts–Strogatz (Small-World)": "watts_strogatz",
        "Stochastic Block Model (Communities)": "stochastic_block_model",
        "Erdős–Rényi (Random Graph)": "erdos_renyi",
    }

    selected_label = st.sidebar.selectbox(
        "Generative Graph Model",
        options=list(model_options.keys()),
        index=0,
        help="Select which complex network topology to synthesize.",
    )
    model_type = model_options[selected_label]

    params: dict[str, Any] = {}

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### Parameters: {selected_label.split(' ')[0]}")

    if model_type == "barabasi_albert":
        st.sidebar.info(
            "**Scale-Free Network**: Grows by preferential attachment $\\Pi(u) \\propto \\deg(u)$. "
            "Produces heavy-tailed degree distributions ($P(k) \\sim k^{-3}$) with super-hubs."
        )
        n = st.sidebar.slider(
            "Total Vertices (n)",
            min_value=50,
            max_value=1000,
            value=250,
            step=25,
            help="Total number of nodes in the final graph.",
        )
        m = st.sidebar.slider(
            "Edges per Incoming Node (m)",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Number of edges each new node attaches to existing nodes (m < n).",
        )
        params = {"n": n, "m": m}

    elif model_type == "watts_strogatz":
        st.sidebar.info(
            "**Small-World Network**: Starts as a 1D ring lattice where each node links to $k$ nearest "
            "neighbors. Randomly rewires edges with probability $p$ to create short-range and long-range bridges."
        )
        n = st.sidebar.slider(
            "Total Vertices (n)",
            min_value=50,
            max_value=1000,
            value=250,
            step=25,
            help="Total number of vertices.",
        )
        # k must be even
        k = st.sidebar.slider(
            "Initial Lattice Degree (k)",
            min_value=2,
            max_value=20,
            value=6,
            step=2,
            help="Degree of each vertex in initial ring lattice (must be an even integer).",
        )
        p = st.sidebar.slider(
            "Rewiring Probability (p)",
            min_value=0.0,
            max_value=1.0,
            value=0.08,
            step=0.01,
            help="Probability of rewiring each edge to a random node. 0 = pure lattice, 1 = random graph.",
        )
        params = {"n": n, "k": k, "p": p}

    elif model_type == "stochastic_block_model":
        st.sidebar.info(
            "**Stochastic Block Model**: Partitions $V$ into $K$ community blocks. "
            "Edge existence follows block probabilities ($P_{rr} \\gg P_{rs}$ in homophily)."
        )
        num_communities = st.sidebar.slider(
            "Number of Communities (K)",
            min_value=2,
            max_value=5,
            value=3,
            step=1,
            help="Number of disjoint community blocks.",
        )

        community_sizes: list[int] = []
        st.sidebar.markdown("**Community Block Sizes**")
        cols = st.sidebar.columns(num_communities)
        default_sizes = [80, 80, 90, 75, 75]
        for i in range(num_communities):
            with cols[i]:
                size = st.number_input(
                    f"C{i+1}",
                    min_value=10,
                    max_value=500,
                    value=default_sizes[i] if i < len(default_sizes) else 80,
                    step=10,
                    key=f"sbm_size_{i}",
                )
                community_sizes.append(int(size))

        p_in = st.sidebar.slider(
            "Intra-Community Probability ($P_{in}$)",
            min_value=0.01,
            max_value=0.30,
            value=0.08,
            step=0.005,
            help="Connection probability between nodes in the SAME community block ($P_{rr}$).",
        )
        p_out = st.sidebar.slider(
            "Inter-Community Probability ($P_{out}$)",
            min_value=0.001,
            max_value=0.05,
            value=0.004,
            step=0.001,
            format="%.4f",
            help="Connection probability between nodes in DIFFERENT community blocks ($P_{rs}$).",
        )

        # Build K x K symmetric probability matrix
        prob_matrix = [[p_out for _ in range(num_communities)] for _ in range(num_communities)]
        for r in range(num_communities):
            prob_matrix[r][r] = p_in

        params = {
            "community_sizes": community_sizes,
            "probability_matrix": prob_matrix,
            "p_in": p_in,
            "p_out": p_out,
            "num_communities": num_communities,
        }

    elif model_type == "erdos_renyi":
        st.sidebar.info(
            "**Erdős–Rényi Network**: $G(n, p)$ random ensemble where each pair of vertices has independent "
            "edge probability $p$. Yields a binomial/Poisson degree distribution without clustering."
        )
        n = st.sidebar.slider(
            "Total Vertices (n)",
            min_value=50,
            max_value=1000,
            value=250,
            step=25,
            help="Total number of nodes.",
        )
        p = st.sidebar.slider(
            "Edge Probability (p)",
            min_value=0.005,
            max_value=0.20,
            value=0.03,
            step=0.005,
            format="%.3f",
            help="Independent edge existence probability between any two vertices.",
        )
        params = {"n": n, "p": p}

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Reproducibility Control")
    seed_col1, seed_col2 = st.sidebar.columns([3, 2])
    with seed_col1:
        seed = st.number_input(
            "Random Seed",
            min_value=0,
            max_value=999999,
            value=42,
            step=1,
            help="Fixed integer seed for exact pseudo-random reproducibility.",
        )
    with seed_col2:
        st.markdown("<div style='height: 28px'></div>", unsafe_allow_html=True)
        if st.button("🎲 Random", help="Set a random seed"):
            seed = random.randint(1, 99999)
            st.session_state["random_seed_override"] = seed

    if "random_seed_override" in st.session_state:
        seed = st.session_state["random_seed_override"]

    params["seed"] = int(seed)

    st.sidebar.markdown("---")
    generate_btn = st.sidebar.button(
        "🚀 Generate Network",
        type="primary",
        use_container_width=True,
        help="Synthesize graph and update all metrics and visualizations.",
    )

    return {
        "model": model_type,
        "model_label": selected_label,
        "params": params,
        "seed": int(seed),
        "generate_triggered": generate_btn,
    }

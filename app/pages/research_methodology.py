"""Research & Methodology page for ViralGraph (Discrete Mathematics Foundations & Academic Proofs)."""

from __future__ import annotations

import streamlit as st
from app.components.header import render_page_header


def render_research_methodology_page() -> None:
    """Render Research & Methodology theoretical documentation page."""
    render_page_header(
        title="Research & Methodology",
        subtitle="Rigorous discrete mathematics foundations, spectral proofs, stochastic contagion theorems, and statistical methodology.",
        badge="Discrete Math Compendium",
        category="THEORETICAL FOUNDATIONS",
    )

    st.markdown(
        """
        <div class="vg-hero" style="padding: 24px 28px; margin-bottom: 24px;">
            <div class="vg-breadcrumb">THEORETICAL COMPENDIUM</div>
            <h2 style="color: #ffffff; font-size: 1.45rem; font-weight: 700; margin: 0 0 8px 0;">
                Mathematical & Methodological Foundations of ViralGraph
            </h2>
            <p style="margin: 0; color: #cbd5e1; font-size: 0.92rem; line-height: 1.6;">
                Every analytical calculation, diffusion simulator, and machine learning pipeline in ViralGraph is grounded in
                formal discrete mathematics, algebraic graph theory, stochastic processes, and rigorous statistical estimation.
                Below is the comprehensive theoretical specification prepared for academic peer review and demonstration.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # SECTION A: NETWORK SCIENCE FUNDAMENTALS
    # -------------------------------------------------------------------------
    with st.expander("Section A: Network Science Fundamentals & Graph Invariants", expanded=True):
        st.markdown(
            """
            ### 1. Formal Graph Definitions
            Let $G = (V, E)$ be an undirected, simple, finite graph with vertex set $V = \\{v_0, v_1, \\dots, v_{n-1}\\}$ of order $|V| = n$
            and edge set $E \\subseteq \\{\\{u, v\\} : u, v \\in V, u \\neq v\\}$ of size $|E| = m$.
            
            **Euler's Handshaking Lemma**:
            $$\\sum_{u \\in V} \\deg(u) = 2|E| = 2m$$
            *Proof*: Each edge $e = \\{u, v\\} \\in E$ contributes exactly $+1$ to $\\deg(u)$ and $+1$ to $\\deg(v)$. Summing over all vertices counts each edge exactly twice. $\\blacksquare$

            ### 2. Adjacency Matrix & Walk Counting Theorem
            Let $A \\in \\{0, 1\\}^{n \\times n}$ be the symmetric adjacency matrix:
            $$A_{uv} = \\begin{cases} 1 & \\text{if } \\{u, v\\} \\in E \\\\ 0 & \\text{otherwise} \\end{cases}$$
            
            **Theorem (Walk Counting via Matrix Powers)**:
            The $(u, v)$-th entry of matrix power $A^k$ ($k \\in \\mathbb{N}$) equals the number of distinct walks of length $k$ between vertex $u$ and vertex $v$:
            $$(A^k)_{uv} = \\sum_{w_1, \\dots, w_{k-1} \\in V} A_{u w_1} A_{w_1 w_2} \\cdots A_{w_{k-1} v}$$
            
            **Corollaries**:
            - Closed 2-walks: $\\text{Tr}(A^2) = \\sum_{u \\in V} (A^2)_{uu} = \\sum_{u \\in V} \\deg(u) = 2m$.
            - Triangle Counting: Every triangle $\\{u, v, w\\}$ contains 6 closed 3-walks (2 orientations $\\times$ 3 starting vertices). Thus:
              $$T(G) = \\frac{1}{6} \\text{Tr}(A^3)$$

            ### 3. Graph Density & Local Clustering Coefficient
            - **Graph Density**: Fraction of potential edges realized:
              $$\\rho(G) = \\frac{2m}{n(n - 1)}$$
            - **Local Clustering Coefficient (Watts & Strogatz 1998)**:
              For vertex $v$ with $\\deg(v) \\ge 2$:
              $$C(v) = \\frac{2 T(v)}{\\deg(v)(\\deg(v) - 1)} = \\frac{(A^3)_{vv}}{\\deg(v)(\\deg(v) - 1)}$$
              where $T(v)$ is the number of edges connecting pairs of neighbors of $v$. For $\\deg(v) < 2$, $C(v) \\equiv 0$.
            """
        )

    # -------------------------------------------------------------------------
    # SECTION B: GENERATIVE GRAPH ENSEMBLES
    # -------------------------------------------------------------------------
    with st.expander("Section B: Generative Graph Models & Thermodynamic Regimes", expanded=True):
        st.markdown(
            """
            ### 1. Erdős–Rényi Model $G(n, p)$ (Homogeneous Random Graphs)
            Each of the $\\binom{n}{2}$ possible edges is included independently with probability $p \\in [0, 1]$.
            - **Degree Distribution**: Binomial $\\text{Binomial}(n - 1, p)$. In the thermodynamic limit ($n \\to \\infty, np = \\lambda$), approaches Poisson:
              $$P(k) = e^{-\\lambda} \\frac{\\lambda^k}{k!}$$
            - **Phase Transition**: At $p_c = 1/n$ (mean degree $\\langle k \\rangle = 1$), a giant connected component (GCC) emerges containing $\\Theta(n)$ vertices.

            ### 2. Barabási–Albert Model (Scale-Free Topologies via Preferential Attachment)
            Begins with an initial clique $m_0$. At each time step $t$, a new vertex arrives and connects to $m \\le m_0$ existing vertices with probability:
            $$\\Pi(u) = \\frac{\\deg(u)}{\\sum_{w \\in V} \\deg(w)} = \\frac{\\deg(u)}{2m t}$$
            - **Asymptotic Degree Distribution**: Power law $P(k) \\sim 2m^2 k^{-3}$ with degree exponent $\\gamma = 3$.
            - **Ultra-Small-World Property**: Average shortest path length scales as $\\langle L \\rangle \\sim \\frac{\\ln n}{\\ln \\ln n}$.

            ### 3. Watts–Strogatz Model (Small-World Networks)
            Begins with a 1D ring lattice of $n$ vertices where each node connects to $k$ nearest neighbors ($k$ even, $k \\ll n$). Each edge is rewired with probability $p \\in [0, 1]$.
            - **Small-World Regime ($p \\in [0.01, 0.1]$)**: Local clustering remains high ($\\\\langle C \\rangle \\approx \\frac{3(k-2)}{4(k-1)}$) while average path length collapses exponentially to $\\langle L \\rangle \\sim \\ln n$ due to long-range shortcuts.

            ### 4. Stochastic Block Model (Clustered Communities & Homophily)
            Vertex set $V$ is partitioned into $K$ disjoint community blocks $V_1, \\dots, V_K$. Edge presence between $u \\in V_r$ and $v \\in V_s$ is an independent Bernoulli trial with probability $P_{rs}$.
            - **Homophily Condition**: $P_{rr} \\gg P_{rs}$ ($r \\neq s$), generating dense intra-community clusters linked by sparse bridge edges.
            """
        )

    # -------------------------------------------------------------------------
    # SECTION C: STOCHASTIC DIFFUSION MODELS
    # -------------------------------------------------------------------------
    with st.expander("Section C: Stochastic Information Diffusion Models", expanded=True):
        st.markdown(
            """
            ### 1. Independent Cascade Model (ICM) — Simple Contagion
            At discrete time step $t=0$, an initial seed set $S_0 \\subset V$ is activated.
            At step $t \\ge 1$, each vertex $u \\in A_{t-1}$ (activated in epoch $t-1$) attempts to infect each susceptible neighbor $v \\in V \\setminus \\bigcup_{\\tau < t} A_\\tau$ independently with transmission probability $p_{uv} = p$.
            - **One-Chance Rule**: Vertex $u$ has exactly **one** chance to activate each neighbor $v$. Regardless of outcome, $u$ cannot attempt to activate $v$ in subsequent epochs.
            - **Process Termination**: Cascade terminates at epoch $T_{\\text{term}}$ when $A_t = \\emptyset$.
            - **Live-Edge Equivalence**: The probability distribution of active sets in ICM is isomorphic to the reach distribution in a random sub-graph $G'$ where each edge $e \\in E$ is retained independently with probability $p$.

            ### 2. Linear Threshold Model (LTM) — Complex Contagion
            Each vertex $v \\in V$ is assigned an adoption threshold $\\theta_v \\sim \\text{Uniform}(0, 1)$.
            Edges $(u, v)$ carry normalized influence weights $b_{uv} = 1 / \\deg(v)$ such that $\\sum_{u \\in N(v)} b_{uv} = 1$.
            - **Activation Rule**: A susceptible vertex $v$ becomes permanently activated at step $t$ if and only if the aggregate fraction of its activated neighbors exceeds its threshold:
              $$\\sum_{u \\in N(v) \\cap A_{\\le t-1}} b_{uv} \\ge \\theta_v$$
            - **Complex Contagion Characteristic**: Unlike ICM, a single active neighbor cannot trigger adoption if $\\theta_v > 1/\\deg(v)$. Requires social reinforcement and multiple exposures.

            ### 3. Kempe-Kleinberg-Tardos Submodularity Theorem
            Let $\\sigma(S) = \\mathbb{E}[|R_\\infty(S)|]$ be the expected cascade reach of seed set $S$.
            
            **Theorem (Kempe, Kleinberg & Tardos 2003)**:
            Under both the Independent Cascade Model and the Linear Threshold Model, the influence function $\\sigma: 2^V \\to \\mathbb{R}_{\\ge 0}$ is:
            1. **Monotone**: $S \\subseteq T \\implies \\sigma(S) \\le \\sigma(T)$
            2. **Submodular**: For all $S \\subseteq T \\subset V$ and $v \\in V \\setminus T$:
               $$\\sigma(S \\cup \\{v\\}) - \\sigma(S) \\ge \\sigma(T \\cup \\{v\\}) - \\sigma(T)$$
            - **Algorithmic Consequence**: The greedy seed selection algorithm achieves a guaranteed $(1 - 1/e) \\approx 63.2\\%$ approximation factor to the NP-hard optimal influence maximization problem.
            """
        )

    # -------------------------------------------------------------------------
    # SECTION D: CENTRALITY INVARIANTS & SPECTRAL GRAPH THEORY
    # -------------------------------------------------------------------------
    with st.expander("Section D: Centrality Metrics & Spectral Graph Theory", expanded=True):
        st.markdown(
            """
            ### 1. Canonical Centrality Metrics
            - **Normalized Degree Centrality**:
              $$C_D(v) = \\frac{\\deg(v)}{n - 1}$$
            - **Shortest-Path Betweenness Centrality (Brandes 2001)**:
              $$C_B(v) = \\sum_{s \\neq v \\neq t \\in V} \\frac{\\sigma_{st}(v)}{\\sigma_{st}}$$
              Normalized by $\\frac{2}{(n-1)(n-2)}$ for undirected graphs. Computed in $O(nm)$ using Brandes' dependency accumulation.
            - **Closeness Centrality (Wasserman & Faust 1994)**:
              For disconnected graphs with component $C_v$ containing vertex $v$:
              $$C_C(v) = \\frac{|C_v| - 1}{n - 1} \\cdot \\frac{|C_v| - 1}{\\sum_{u \\in C_v, u \\neq v} d(v, u)}$$
            - **Eigenvector Centrality**:
              Principal eigenvector of adjacency matrix: $A x = \\lambda_1 x$. By the Perron-Frobenius theorem, for a connected graph, $x_v > 0$ for all $v$.
            - **Betweenness-to-Degree Ratio (BDR)**:
              $$\\text{BDR}(v) = \\frac{C_B(v)}{C_D(v)}$$
              Identifies structural bridges: vertices with moderate degree but disproportionate shortest-path brokerage.

            ### 2. Spectral Graph Theory & Combinatorial Laplacian
            Let $D = \\text{diag}(\\deg(v_0), \\dots, \\deg(v_{n-1}))$. The combinatorial Laplacian matrix is $L = D - A$.
            - **Quadratic Form (Unordered Edges)**:
              $$x^T L x = \\sum_{\\{u, v\\} \\in E} (x_u - x_v)^2$$
            - **Quadratic Form (Ordered Vertex Pairs)**:
              $$x^T L x = \\frac{1}{2} \\sum_{u \\in V} \\sum_{v \\in V} A_{uv} (x_u - x_v)^2$$
            - **Positive Semi-Definiteness**: Because $(x_u - x_v)^2 \\ge 0$, $x^T L x \\ge 0 \\implies L \\succeq 0$. Eigenvalues satisfy:
              $$0 = \\lambda_1 \\le \\lambda_2 \\le \\dots \\le \\lambda_n$$
            - **Algebraic Connectivity $\\lambda_2$ (Fiedler 1973)**: $\\lambda_2 > 0 \\iff G$ is connected.
            - **Cheeger's Inequality**: Let $h(G) = \\min_{|S| \\le n/2} \\frac{|\\partial(S)|}{|S|}$ be the conductance:
              $$\\frac{\\lambda_2}{2} \\le h(G) \\le \\sqrt{2 \\Delta \\lambda_2}$$
            """
        )

    # -------------------------------------------------------------------------
    # SECTION E: EMPIRICAL & STATISTICAL METHODOLOGY
    # -------------------------------------------------------------------------
    with st.expander("Section E: Empirical & Statistical Research Methodology", expanded=True):
        st.markdown(
            """
            ### 1. Multi-Instance Hierarchical Sampling
            When testing generalization across network topologies, individual Monte Carlo runs on a single graph cannot be treated as independent observations.
            ViralGraph enforces a hierarchical design:
            - **Primary Sampling Unit (PSU)**: $N_{\\text{inst}} = 5$ independent synthetic graph instances per topology.
            - **Secondary Sampling Unit**: $M = 40$ Monte Carlo cascade realizations per instance.
            - **Graph-Level Mean**: For instance $i$, compute sample mean $\\bar{R}_i = \\frac{1}{M} \\sum_{j=1}^M R_{i, j}$.
            - **Uncertainty Estimation**: Because $N_{\\text{inst}} = 5$ is small, confidence intervals use Student's $t$-distribution with $df = N_{\\text{inst}} - 1 = 4$:
              $$\\bar{R} \\pm t_{0.025, 4} \\cdot \\frac{s}{\\sqrt{N_{\\text{inst}}}}, \\quad t_{0.025, 4} \\approx 2.776$$

            ### 2. Grouped Out-of-Graph Cross-Validation (Zero Target Leakage)
            In virality forecasting, standard K-Fold random sampling leaks graph topology (models memorize vertex IDs).
            ViralGraph employs:
            - **GroupShuffleSplit**: Partitions graph topologies into disjoint train and test graphs (6 unseen test graphs).
            - **Inner-Fold Thresholding**: Virality threshold $\\tau$ is fit strictly within inner training folds. Validation and test folds never influence $\\tau$.
            - **Strict Early Temporal Horizon**: Features are limited strictly to events observed at $t \\le 1$ ($`early_velocity`$, $`early_infections_step1`$), eliminating lookahead bias.
            """
        )

    # -------------------------------------------------------------------------
    # SECTION F: SCIENTIFIC LIMITATIONS & SCOPE BOUNDARIES
    # -------------------------------------------------------------------------
    with st.expander("Section F: Scientific Limitations & Scope Boundaries", expanded=True):
        st.markdown(
            """
            ### 1. Static Graph Topology Abstraction
            ViralGraph abstracts social networks as static discrete graphs $G = (V, E)$. Real-world interactions occur over time-varying temporal hypergraphs where edge existence is bursty and ephemeral.

            ### 2. Homogeneous Transmission Probabilities
            Unless explicitly parameterized, synthetic simulations assume uniform transmission probabilities $p_{uv} = p$. Real social influence exhibits heavy asymmetry, homophilic affinity, and topic-dependent susceptibility.

            ### 3. No Unvalidated Content Manipulation
            ViralGraph explicitly decouples post wording and topic categorization from transmission probabilities $p_{uv}$ in the absence of a validated, empirical content-transmission function. All virality predictions are structural forecasts based strictly on early topological propagation dynamics.
            """
        )

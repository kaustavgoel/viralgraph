# ViralGraph — Professor Demonstration Manual & Architectural Compendium
**BTech Mathematics & Computing Project Demonstration Edition**

---

## 1. Executive Summary & Academic Purpose

**ViralGraph** is an interactive, dark-mode-first research and simulation platform designed to study information diffusion, stochastic contagion dynamics, and structural interventions across complex networks.

Built specifically for academic demonstration, the platform bridges **discrete mathematics, spectral graph theory, stochastic processes, and machine learning**. It delivers a complete end-to-end laboratory where theoretical theorems are directly visualized and empirically tested through live simulations.

### Key Architectural Pillars
- **Zero Fabrication**: No simulated cascades, centralities, or prediction scores are hardcoded or fabricated. All calculations execute live using NetworkX, NumPy, SciPy, and Scikit-Learn.
- **Strict Methodological Discipline**: Post titles and wording do not artificially manipulate edge transmission probabilities $p_{uv}$. Content is treated as experimental metadata, ensuring strict adherence to network science principles.
- **Multi-Instance Uncertainty**: When evaluating network generalization, ViralGraph uses hierarchical multi-instance sampling ($N_{\text{inst}} = 5$ independent graph instances per topology, $M = 40$ Monte Carlo runs/instance) with Student's $t$-distributions ($df = 4$).
- **Leakage-Free ML Forecasting**: Out-of-graph grouped cross-validation across 6 unseen topologies, inner-fold target thresholding, and strict early observation horizons ($t \le 1$).

---

## 2. Theoretical & Mathematical Foundations

### 2.1 Graph Invariants & Euler's Handshaking Lemma
For an undirected simple graph $G = (V, E)$ with order $|V| = n$ and size $|E| = m$:
$$\sum_{u \in V} \deg(u) = 2|E| = 2m$$
Every edge $e = \{u, v\} \in E$ contributes $+1$ to $\deg(u)$ and $+1$ to $\deg(v)$.

### 2.2 Walk Counting & Triangle Enumeration
Let $A \in \{0, 1\}^{n \times n}$ be the symmetric adjacency matrix. The $(u, v)$-th entry of matrix power $A^k$ ($k \in \mathbb{N}$) enumerates distinct walks of length $k$ between $u$ and $v$:
$$(A^k)_{uv} = \sum_{w_1, \dots, w_{k-1}} A_{u w_1} A_{w_1 w_2} \cdots A_{w_{k-1} v}$$
- Closed 2-walks: $\text{Tr}(A^2) = \sum_{u \in V} \deg(u) = 2m$.
- Triangle counting: $T(G) = \frac{1}{6} \text{Tr}(A^3)$.

### 2.3 Combinatorial Laplacian & Positive Semi-Definiteness
Let $D = \text{diag}(\deg(v_0), \dots, \deg(v_{n-1}))$. The combinatorial Laplacian matrix is $L = D - A$.
- **Quadratic Form (Unordered Edges)**:
  $$x^T L x = \sum_{\{u, v\} \in E} (x_u - x_v)^2$$
- **Quadratic Form (Ordered Vertex Pairs)**:
  $$x^T L x = \frac{1}{2} \sum_{u \in V} \sum_{v \in V} A_{uv} (x_u - x_v)^2$$
Because $(x_u - x_v)^2 \ge 0$, $L \succeq 0$, and all eigenvalues are non-negative real numbers:
$$0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_n$$
- The algebraic connectivity $\lambda_2 > 0 \iff G$ is connected (Fiedler value).
- The multiplicity of eigenvalue $0$ strictly equals the number of connected components.

### 2.4 Cheeger's Inequality & Conductance Bounds
Let $h(G) = \min_{|S| \le n/2} \frac{|\partial(S)|}{|S|}$ be the bottleneck conductance ratio:
$$\frac{\lambda_2}{2} \le h(G) \le \sqrt{2 \Delta \lambda_2}$$
where $\Delta = \max_{u \in V} \deg(u)$.

### 2.5 Stochastic Diffusion Models
1. **Independent Cascade Model (ICM)**:
   - At epoch $t=0$, seed set $S_0$ is activated.
   - At epoch $t \ge 1$, each newly active vertex $u \in A_{t-1}$ attempts to infect each susceptible neighbor $v \in N(u) \setminus A_{\le t-1}$ with independent probability $p_{uv} = p$.
   - Vertices have exactly one attempt to infect neighbors.
2. **Linear Threshold Model (LTM)**:
   - Vertices have adoption thresholds $\theta_v \sim \text{Uniform}(0, 1)$ and incoming edge weights $b_{uv} = 1/\deg(v)$.
   - Vertex $v$ activates when $\sum_{u \in N(v) \cap A_{\le t-1}} b_{uv} \ge \theta_v$.
3. **Kempe-Kleinberg-Tardos Submodularity (2003)**:
   The expected cascade reach $\sigma(S) = \mathbb{E}[|R_\infty(S)|]$ is monotone submodular:
   $$\sigma(S \cup \{v\}) - \sigma(S) \ge \sigma(T \cup \{v\}) - \sigma(T) \quad \forall S \subseteq T$$
   Guarantees a $(1 - 1/e) \approx 63.2\%$ approximation factor for greedy seed selection.

---

## 3. Platform Architecture: 8 Functional Pages

ViralGraph is implemented as a cohesive, dark-mode multipage application using Streamlit:

| Page | Route / File | Core Academic Functionality |
| :--- | :--- | :--- |
| **1. Home** | `app/pages/home.py` | Platform orientation, 4 capability pillars, live session simulation feed, quick-start entry cards. |
| **2. Post Studio** | `app/pages/post_studio.py` | Experiment composer (title, body, category), real-time post card preview, network selector with overwrite protection, seed selection strategies (Hub, Bridge, Closeness, Manual), ICM/LTM parameter controls, validation summary, and launch engine. |
| **3. Live Diffusion** | `app/pages/live_diffusion.py` | Step-by-step discrete-time cascade scrubbing ($t \in [0, T_{\text{term}}]$), interactive Plotly network canvas distinguishing Seed (Gold), Newly Activated (Emerald), Previously Active (Blue), and Susceptible (Slate), active transmission DAG edges, cumulative reach line, and marginal infection velocity bar chart. |
| **4. Network Explorer** | `app/pages/network_explorer.py` | Generative graph synthesis (BA, WS, SBM, ER), 2D topological canvas, global metric cards ($|V|, |E|, \langle k \rangle, \rho, \langle C \rangle, \lambda_2$), **Vertex Micro-Inspection Panel** (individual node degree, normalized degree, betweenness, closeness, clustering, BDR, community, and neighborhood list), Degree Distribution $P(k)$ histogram, and Betweenness distribution histogram. |
| **5. Influencer Research** | `app/pages/influencer_research.py` | Top-10 influencer rankings across all 5 canonical centralities, Degree vs. Betweenness quadrant scatter plot highlighting Top Hub vs. Strategic Bridge, live head-to-head Monte Carlo comparison trial on active graph, and audited multi-instance benchmark results ($t(4) = 6.53, p = 0.0028, d = 2.92$). |
| **6. Intervention Lab** | `app/pages/intervention_lab.py` | Countermeasure studio: Targeted Hub Inoculation, Targeted Bridge Inoculation, Random Inoculation, and Community Quarantine. Evaluates reach suppression ($\Delta R$), containment efficiency, GCC ratio reduction, and algebraic connectivity degradation on a non-mutating `graph.copy()`. Multi-budget sweep curve ($k \in [1, 8]$). |
| **7. Virality Prediction** | `app/pages/virality_prediction.py` | Supervised ML early virality estimator ($t \le 1$), predicted virality class and probability score, continuous reach regression interval, audited benchmark across 6 unseen test topologies, Random Forest confusion matrix, and feature importance breakdown ($>72\%$ importance in early velocity/reach). |
| **8. Research & Methodology** | `app/pages/research_methodology.py` | Full theoretical compendium covering Sections A–F: Graph Fundamentals, Generative Models, Stochastic Diffusion, Centrality Invariants, Statistical Methodology, and Scope Boundaries with formal LaTeX proofs. |

---

## 4. Professor Live Demonstration Walkthrough Script

Follow this structured 8-stage sequence during a faculty evaluation:

### Stage 1: Platform Overview & Theoretical Alignment (Home)
1. Open `http://localhost:8501`.
2. Point out the dark-mode aesthetic (`#0B0F19`), the clean header, and the four capability pillars.
3. Note the simulation history feed (starts completely empty with authentic zero-fabrication logging).

### Stage 2: Generative Complex Networks (Network Explorer)
1. Navigate to **Network Explorer**.
2. Select **Barabási–Albert (Scale-Free)** with $n=250, m=3$. Click **🚀 Generate Network**.
3. Inspect the top metric cards: $|V| = 250, |E| = 741, \langle k \rangle = 5.93, \rho = 0.0238$.
4. Scroll down to the **Vertex Micro-Inspection Panel**:
   - Click **🎯 Highest Hub**.
   - Show the professor how vertex $v_0$ has degree $\approx 45$, normalized degree $C_D \approx 0.18$, high betweenness $C_B \approx 0.15$, and local clustering $C(v)$.
   - Inspect the Degree Distribution $P(k)$ histogram illustrating power-law heavy tails.

### Stage 3: Designing an Information Cascade (Post Studio)
1. Navigate to **Post Studio**.
2. In **Section A (Composer)**, enter:
   - Title: `Quantum Computing Breakthrough in Cryptography`
   - Content: `Evaluating stochastic cascade dynamics across scale-free topologies.`
   - Category: `Technology`
   - Note the real-time social card preview.
3. In **Section C (Seed Node)**, select `Highest Degree (Hub)`. Note that vertex $v_0$ is automatically resolved with degree $\approx 45$.
4. In **Section D (Diffusion Model)**, select `Independent Cascade (ICM)` with $p = 0.18$.
5. In **Section E**, click **🚀 Launch Diffusion Simulation**.
6. Observe the completion notification and recent simulation scorecard showing total reached vertices, steps to termination, and transmission pathways.

### Stage 4: Discrete Temporal Playback (Live Diffusion)
1. Navigate to **Live Diffusion**.
2. Explain that the active simulation trajectory is loaded directly from session state.
3. Scrub the **Discrete Cascade Epoch slider** from $t=0$ to $t=1, 2, \dots$:
   - At $t=0$: Initial seed vertex $v_0$ is highlighted in amber/gold.
   - At $t=1$: First generation of newly infected vertices appears in bright emerald green.
   - At $t \ge 2$: Previously infected vertices transition to blue, new frontier nodes in emerald, and cyan directed transmission edges trace the exact propagation pathways.
4. Point out the bottom charts: the **Cumulative Reach Curve** and the **Marginal Infection Velocity Bar Chart**.

### Stage 5: The Hub vs. Bridge Research Hypothesis (Influencer Research)
1. Navigate to **Influencer Research**.
2. State the central research question: *"Can a community-bridging node generate a larger or more cross-cutting cascade than a high-degree hub?"*
3. Show the **Top-Ranked Structural Influencers** table comparing top nodes across Degree, Betweenness, PageRank, Closeness, and Eigenvector centralities.
4. Examine the **Degree vs. Shortest-Path Betweenness Scatter Plot**:
   - Point out the red diamond (Top Hub) and the green star (Strategic Bridge with high BDR).
5. Click **🚀 Run Head-to-Head Comparative Trial**:
   - The engine runs 25 Monte Carlo cascades for Hub, Bridge, PageRank, and Random baseline.
   - Show the professor the empirical findings: Hubs win on raw volume ($\mathbb{E}[R_\infty]$), but in clustered networks (SBM), bridges achieve $+80.2\%$ higher cross-community infection fraction ($\rho_{\text{ext}}$).

### Stage 6: Structural Countermeasures (Intervention Lab)
1. Navigate to **Intervention Lab**.
2. Pose the what-if question: *"How can we suppress a viral cascade with minimal network disruption?"*
3. Select **Targeted Hub Inoculation** with budget $k=5$ vertices.
4. Click **🛡️ Evaluate Structural Intervention**.
5. Show the comparison metrics:
   - **Reach Suppression ($\Delta R$)**: Typically $-50\%$ to $-80\%$ reduction.
   - **Containment Efficiency**: $>3$ infections prevented per immunized vertex.
   - **Algebraic Connectivity ($\lambda_2$)**: Drops, demonstrating network fragmentation.
6. Emphasize that all computations execute on `graph.copy()`, preserving baseline network integrity.

### Stage 7: Leakage-Free Early Virality Forecasting (Virality Prediction)
1. Navigate to **Virality Prediction**.
2. Explain the early observation window constraint ($t \le 1$): no lookahead bias.
3. Click **🔮 Forecast Virality from t ≤ 1**:
   - The model evaluates the seed's step-1 velocity, degree, PageRank, and density.
   - Displays the predicted virality classification, probability score, and continuous reach interval.
4. Show the **Out-of-Graph Benchmark Table**:
   - Explain why grouped cross-validation across 6 unseen graphs is necessary to prevent topology memorization.
   - Point out the Random Forest test ROC-AUC ($0.945$) and the Confusion Matrix.
   - Note the feature importance chart: early velocity and early reach account for $>72\%$ of model predictive power.

### Stage 8: Mathematical Grounding (Research & Methodology)
1. Navigate to **Research & Methodology**.
2. Open Section A (Euler's Handshaking Lemma & walk counting).
3. Open Section C (ICM Bernoulli formulation & Kempe-Kleinberg-Tardos submodularity).
4. Open Section D (Laplacian quadratic forms, Cheeger's inequality, and algebraic connectivity).
5. Open Section E (Student's $t$ uncertainty estimation with $df = 4$).
6. Return to **Home** to show the persistent simulation history feed with full session provenance.

---

## 5. Automated Verification & Code Health

The entire backend and interface are protected by an automated test suite of **96 passing unit and integration tests**:

```bash
$ .venv/bin/pytest tests/
======================== 96 passed, 8 warnings in 2.07s ========================
```

| Test Module | Coverage Scope | Status |
| :--- | :--- | :--- |
| `tests/test_full_platform.py` | Live Diffusion Plotly canvas/timeline, Node Micro-Inspection, Monte Carlo trials, Intervention non-mutation, Virality strict feature bounds ($t \le 1$). | **11 Passing** |
| `tests/test_post_studio.py` | Post Studio composer, seed resolution, overwrite guards, ICM/LTM execution, SessionManager persistence. | **13 Passing** |
| `tests/test_diffusion_models.py` | ICM Bernoulli trials, LTM threshold adoption, transmission DAG acyclicity, step monotonicity. | **5 Passing** |
| `tests/test_graph_analytics.py` | Centralities (Degree, Betweenness, Closeness, Eigenvector, PageRank, BDR), Student's $t$ confidence intervals ($df=4$). | **15 Passing** |
| `tests/test_graph_representation.py` | Laplacian quadratic forms ($x^T L x$), walk counting ($A^k$), algebraic connectivity ($\lambda_2$), spectral bounds. | **18 Passing** |
| `tests/test_synthetic_generators.py` | ER, BA, WS, SBM mathematical parameter validation and invariant testing. | **16 Passing** |
| `tests/test_interventions.py` | Targeted hub, targeted bridge, random immunization, community quarantine, GCC reduction. | **3 Passing** |
| `tests/test_ml_pipeline.py` | Grouped train-test splitting, inner-fold thresholding, zero target leakage, out-of-graph generalization. | **4 Passing** |
| `tests/test_interface_components.py` | Metric cards, formatting, state managers. | **9 Passing** |
| `tests/test_temporal_diffusion.py` | Strict temporal observation windows ($t \le t_{\text{obs}}$). | **2 Passing** |

---
*ViralGraph — Developed for BTech Mathematics & Computing Project Demonstration.*

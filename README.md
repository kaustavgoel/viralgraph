# ViralGraph

> **Intelligent Social Information Diffusion & Network Analysis Platform**  
> *A Discrete Mathematics & Graph Theory Research Laboratory for Information Spreading, Prediction, Simulation, and Topological Intelligence.*

---

## 🌐 Live Web Application

The interactive platform is deployed and publicly accessible online:

👉 **[https://viralgraph.streamlit.app/](https://viralgraph.streamlit.app/)**

---

## 📖 About the Project

**ViralGraph** studies, simulates, predicts, and explains how information cascades travel across complex social topologies. Rather than treating virality as an isolated property of text or media, ViralGraph treats information diffusion as an **emergent phenomenon** governed by network geometry, seed placement, discrete-time stochastic transitions, and community boundaries.

### Core Objectives
1. **Explain**: Decompose network structure into adjacency matrices, degree distributions, centrality spectra, and community partitions.
2. **Simulate**: Execute discrete-time stochastic models (Independent Cascade Model, Linear Threshold Model) to explore information cascades and structural bottlenecks.
3. **Predict**: Forecast final cascade reach and binary virality without temporal data leakage using strict early observation horizons ($t \le 1$).
4. **Intervene (What-If)**: Model counterfactual scenarios, such as targeted hub/bridge node immunization, seed variation, and community quarantine.

---

## 🔬 Core Platform Modules

The platform features eight interconnected research modules:

1. **Home / Overview**: Central platform orientation, capability pillars, and live session provenance audit feed.
2. **Post Studio**: Experiment composer with real-time card preview, network selector with overwrite protection, automated seed targeting (Hub, Bridge, Closeness, or Manual), diffusion parameter controls (ICM and LTM), pre-launch validation, and instant simulation scorecard.
3. **Live Diffusion**: Discrete-time cascade temporal scrubber ($t \in [0, T_{\text{term}}]$), interactive Plotly network canvas distinguishing Seed (Gold), Newly Activated (Emerald), Previously Active (Blue), and Susceptible (Slate) vertices, active transmission DAG edges, cumulative reach trajectory, and marginal velocity curves.
4. **Network Explorer**: Generative synthetic graph engine (ER, BA, WS, SBM), 2D topological canvas, global graph invariants, **Vertex Micro-Inspection Panel** (evaluating individual node degree, normalized degree, betweenness, closeness, clustering, BDR, community block, and neighbor lists), and Plotly degree distribution $P(k)$ histograms.
5. **Influencer Research**: Top-10 influencer rankings across all 5 canonical centralities, Degree vs. Betweenness scatter plot highlighting Top Hubs vs. Strategic Bridges, live head-to-head Monte Carlo comparison trial on the active graph, and audited multi-instance benchmark results ($t(4) = 6.53, p = 0.0028, d = 2.92$).
6. **Intervention Lab**: Structural countermeasure studio: Targeted Hub Inoculation, Targeted Bridge Inoculation, Random Inoculation, and Community Quarantine. Evaluates reach suppression ($\Delta R$), containment efficiency, GCC ratio reduction, and algebraic connectivity degradation on a non-mutating `graph.copy()`.
7. **Virality Prediction**: Supervised ML early virality estimator ($t \le 1$), predicted virality class and probability score, continuous reach regression interval, audited benchmark across 6 unseen test topologies, Random Forest confusion matrix, and feature importance breakdown ($>72\%$ importance in early velocity/reach).
8. **Research & Methodology**: Complete discrete mathematics compendium covering Sections A–F: Graph Fundamentals, Generative Models, Stochastic Diffusion, Centrality Invariants, Statistical Methodology, and Scope Boundaries with formal LaTeX proofs.

---

## 📐 Mathematical Foundations

The platform is grounded in formal Discrete Mathematics, Matrix Theory, and Spectral Graph Theory:

### 1. Graph Invariants & Euler's Handshaking Lemma
For an undirected simple graph $G = (V, E)$ with order $|V| = n$ and size $|E| = m$:
$$\sum_{u \in V} \deg(u) = 2|E| = 2m$$
Every edge $e = \{u, v\} \in E$ contributes $+1$ to $\deg(u)$ and $+1$ to $\deg(v)$.

### 2. Walk Counting & Triangle Enumeration
Let $A \in \{0, 1\}^{n \times n}$ be the symmetric adjacency matrix. The $(u, v)$-th entry of matrix power $A^k$ ($k \in \mathbb{N}$) enumerates distinct walks of length $k$ between $u$ and $v$:
$$(A^k)_{uv} = \sum_{w_1, \dots, w_{k-1}} A_{u w_1} A_{w_1 w_2} \cdots A_{w_{k-1} v}$$
- Closed 2-walks: $\text{Tr}(A^2) = \sum_{u \in V} \deg(u) = 2m$.
- Triangle counting: $T(G) = \frac{1}{6} \text{Tr}(A^3)$.

### 3. Combinatorial Laplacian & Positive Semi-Definiteness
Let $D = \text{diag}(\deg(v_0), \dots, \deg(v_{n-1}))$. The combinatorial Laplacian matrix is $L = D - A$.
- **Quadratic Form (Unordered Edges)**:
  $$x^T L x = \sum_{\{u, v\} \in E} (x_u - x_v)^2$$
- **Quadratic Form (Ordered Vertex Pairs)**:
  $$x^T L x = \frac{1}{2} \sum_{u \in V} \sum_{v \in V} A_{uv} (x_u - x_v)^2$$
Because $(x_u - x_v)^2 \ge 0$, $L \succeq 0$, and all eigenvalues are non-negative real numbers:
$$0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_n$$
- The algebraic connectivity $\lambda_2 > 0 \iff G$ is connected (Fiedler value).
- The multiplicity of eigenvalue $0$ strictly equals the number of connected components.

### 4. Cheeger's Conductance Inequality
Let $h(G) = \min_{|S| \le n/2} \frac{|\partial(S)|}{|S|}$ be the bottleneck conductance ratio:
$$\frac{\lambda_2}{2} \le h(G) \le \sqrt{2 \Delta \lambda_2}$$
where $\Delta = \max_{u \in V} \deg(u)$.

### 5. Stochastic Diffusion & Influence Submodularity
- **Independent Cascade Model (ICM)**: Live-edge sub-graph equivalence with Bernoulli trials $p_{uv} = p$.
- **Linear Threshold Model (LTM)**: Threshold adoption $\sum_{u \in N(v) \cap A_{\le t-1}} b_{uv} \ge \theta_v$ modeling complex contagion.
- **Kempe-Kleinberg-Tardos Theorem (2003)**: The influence function $\sigma(S) = \mathbb{E}[|R_\infty(S)|]$ is monotone submodular:
  $$\sigma(S \cup \{v\}) - \sigma(S) \ge \sigma(T \cup \{v\}) - \sigma(T) \quad \forall S \subseteq T$$
  Guarantees a $(1 - 1/e) \approx 63.2\%$ approximation factor for greedy seed selection.

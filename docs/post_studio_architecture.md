# Post Studio & Content Injection Architecture (Milestone 2)

## 1. Overview & Purpose

**Post Studio** serves as the campaign authoring and diffusion configuration workbench of the ViralGraph platform. It enables researchers and users to compose simulated social media messages, calibrate underlying complex network topologies, assign seed injection nodes according to discrete graph-theoretic strategies, and execute discrete-time stochastic diffusion simulations using the verified mathematical backend.

---

## 2. Payload Representation: The Simulated Post

A social post in ViralGraph is modeled as an informational payload encapsulated in the `PostData` dataclass:

```python
@dataclass
class PostData:
    title: str            # Message headline / subject
    content: str          # Text body
    topic: str            # Categorical topic vector (e.g. Science, Governance)
    description: str      # Optional hypothesis or trial notes
    media_type: str       # Text Only, Text + Interactive Diagram, Video
    sentiment_score: float # Base sentiment rating [0.0, 1.0]
    virality_multiplier: float # Multiplicative boost
```

### Methodological Invariant & No-Magic Principle
> **Strict Scientific Boundary**: The post text and topic serve as the payload being propagated. Message wording and sentiment do **not** arbitrarily alter transmission probabilities $p_{uv}$ or threshold parameters $\theta_v$. Contagion dynamics are governed strictly by the structural topology and diffusion model equations.

---

## 3. Network Selection & Active State Synchronization

Post Studio supports two network modes:
1. **Use Active Network**: Reuses the network currently loaded in `SessionManager` from previous interactions (e.g., from Network Explorer), preventing unnecessary re-generation.
2. **Synthesize New Network**: Provides parameter controls for four generative graph models:
   - **Barabási–Albert (BA)**: Scale-Free networks via preferential attachment ($n, m$).
   - **Watts–Strogatz (WS)**: Small-World networks via ring lattice rewiring ($n, k, p$).
   - **Stochastic Block Model (SBM)**: Clustered homophily communities ($K$ blocks, $P_{\text{in}}, P_{\text{out}}$).
   - **Erdős–Rényi (ER)**: Random Bernoulli graphs $G(n, p)$.

### Safeguards Against Silent Overwrite
When a new network is synthesized in Post Studio, the user must explicitly click `⚡ Generate and Set as Active Network`. The new graph is stored in `SessionManager.set_current_graph()`, updating its topological invariants, degree distributions, and spectral metrics across all eight platform pages.

---

## 4. Seed Node Selection Strategies

Diffusion begins from an initial seed vertex $s \in V$ ($S_0 = \{s\}$). Post Studio provides four selection strategies:

1. **Highest-Degree Node (Local Hub)**:
   $$s = \arg\max_{u \in V} \deg(u)$$
   Maximizes immediate local branching in step $t=1$.
2. **Highest-Betweenness Node (Structural Bridge)**:
   $$s = \arg\max_{u \in V} C_B(u) = \arg\max_{u \in V} \sum_{s \neq u \neq t} \frac{\sigma_{st}(u)}{\sigma_{st}}$$
   Targeted at vertices that intermediate shortest paths between distinct network clusters.
3. **Random Node (Baseline)**:
   $$s \sim \text{Uniform}(V)$$
   Provides a baseline benchmark for evaluating the lift provided by topological targeting.
4. **Manual Identifier**:
   Allows researchers to pick an exact integer vertex identifier from the graph, validated to ensure $s \in V$.

If the graph is empty ($|V| = 0$), simulation execution is automatically disabled.

---

## 5. Diffusion Models: ICM vs. LTM

Post Studio connects directly to the two stochastic diffusion models implemented in `viralgraph/simulation/`:

### 5.1 Independent Cascade Model (ICM)
- **Nature**: Sender-centric discrete-time contagion.
- **Process**: At time $t$, each newly activated node $u \in A_{t-1}$ gets a single attempt to activate each inactive neighbor $v \in \mathcal{N}(u)$ with independent probability $p_{uv} = p$.
- **Configurable Parameters**:
  - Transmission Probability $p \in [0.01, 1.00]$ (default $0.12$).
  - Maximum Step Horizon $T_{\text{max}} \in [5, 100]$ (default $50$).
  - Random Seed for deterministic reproducibility.

### 5.2 Linear Threshold Model (LTM)
- **Nature**: Receiver-centric complex contagion (peer reinforcement).
- **Process**: Each vertex $v$ has an intrinsic threshold $\theta_v$. Node $v$ activates at step $t$ if and only if the cumulative influence of its active neighbors reaches or exceeds $\theta_v$:
  $$\sum_{u \in \mathcal{N}(v) \cap A_{\le t-1}} b_{uv} \ge \theta_v$$
- **Configurable Parameters**:
  - Threshold Mode:
    - *Random Uniform* (default): $\theta_v \sim \text{Uniform}(0, 1)$ i.i.d. for each node.
    - *Constant Uniform*: $\theta_v = \theta \in [0.05, 1.00]$ for all nodes.
  - Influence Weights: Default uniform in-degree normalization $b_{uv} = \frac{1}{\deg(v)}$ satisfying $\sum_u b_{uv} \le 1$.
  - Maximum Step Horizon $T_{\text{max}}$.
  - Random Seed for deterministic reproducibility.

---

## 6. Backend Integration & Execution Flow

```
[Post Studio UI]
       │
       ▼
[Validate Post Title, Active Graph |V| >= 2, Seed s in V]
       │
       ▼
[resolve_seed_node(graph, strategy, seed)]
       │
       ▼
[launch_diffusion_simulation()]
       ├── ICM: simulate_independent_cascade(graph, [s], p, max_steps, rng)
       └── LTM: simulate_linear_threshold(graph, [s], thresholds, max_steps, rng)
       │
       ▼
[CascadeResult]
       ├── total_reach: Final cascade size |A_<=T|
       ├── steps_to_termination: Discrete epochs taken
       ├── activation_history: [A_0, A_1, ..., A_T]
       ├── cumulative_reach_timeline: [|A_<=0|, ..., |A_<=T|]
       └── transmission_tree: nx.DiGraph recording who infected whom
       │
       ▼
[SessionManager.set_simulation_results()]
[SessionManager.add_simulation_record()]
       │
       ▼
[Results Display in Post Studio + Visible in Home Recent Simulations Feed]
```

### Persistence Without Recomputation
When the user runs a simulation, results are stored in `st.session_state["simulation_results"]` and appended to `st.session_state["simulation_history"]`. Navigating to other pages (Home, Network Explorer, etc.) and returning to Post Studio retains the completed result card without triggering a rerun.

---

## 7. Known Limitations & Scientific Assumptions

1. **Static Graph Edges**: Network topology is treated as static throughout cascade execution; dynamic tie creation or decay is not modeled during a single cascade.
2. **Homogeneous Transmission**: In the baseline ICM configuration, $p_{uv} = p$ is uniform across all edges.
3. **Single Seed Injection**: Post Studio focuses on single-vertex injection ($|S_0| = 1$) to compare hub vs. bridge efficacy cleanly. Multi-seed set optimization belongs to advanced research modules.

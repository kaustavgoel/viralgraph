# ViralGraph

> **Intelligent Social Information Diffusion Platform**  
> *A Discrete Mathematics & Graph Theory Research Architecture for Information Spreading, Prediction, Simulation, and Topological Intelligence.*

---

## 1. Project Overview

**ViralGraph** studies, simulates, predicts, and explains how information cascades travel across complex social topologies. Rather than treating virality as an isolated property of text or media, ViralGraph treats information diffusion as an **emergent phenomenon** governed by network geometry, seed placement, discrete-time stochastic transitions, and community boundaries.

### Core Objectives
1. **Explain**: Decompose network structure into adjacency matrices, degree distributions, centrality spectra, and community partitions.
2. **Simulate**: Execute discrete-time stochastic models (Independent Cascade Model, Linear Threshold Model) to explore information cascades and structural bottlenecks.
3. **Predict**: Forecast final cascade reach and binary virality without temporal data leakage ($t \le t_{\text{obs}}$).
4. **Intervene (What-If)**: Model counterfactual scenarios, such as bridge inoculation, seed variation, and community isolation.

---

## 2. Mathematical Foundation

The platform is grounded in core Discrete Mathematics disciplines:
- **Set Theory & Relations**: Vertex sets $V$, Edge relations $E \subseteq V \times V$, Neighborhood mappings $\mathcal{N}(v)$.
- **Graph Theory**: Directed/undirected graphs, adjacency matrices $A \in \{0, 1\}^{n \times n}$, walk lengths via $A^k$, Laplacians $L = D - A$.
- **Centrality Spectra**: Degree, Closeness, Shortest-path Betweenness, PageRank eigenvector centrality.
- **Stochastic Processes**: Discrete-time Markovian transitions, submodular influence maximization under Independent Cascade Dynamics.

---

## 3. Repository Architecture

```
viralgraph/
├── configs/                     # Central YAML configurations and simulation presets
├── data/                        # Datasets (raw, processed, and synthetic benchmarks)
├── docs/                        # Formal mathematical formulations, literature review, and reports
├── notebooks/                   # Reproducible exploratory Jupyter notebooks
├── tests/                       # Unit tests validating mathematical invariants and algorithms
├── viralgraph/                  # Core Python Package
│   ├── data/                    # Synthetic graph generators (BA, WS, SBM) and loaders
│   ├── graph/                   # Graph representation, adjacency matrices, and centralities
│   ├── simulation/              # Stochastic diffusion engines (ICM, LTM, What-If)
│   ├── models/                  # ML virality classifiers, regressors, and leakage guards
│   ├── analytics/               # Influencer intelligence, bridges, and integrity interventions
│   └── visualization/           # Graph layout plotting and cascade progression charts
└── app/                         # Interactive Streamlit Web Application
```

---

## 4. Milestone Status

| Milestone | Description | Status |
| :--- | :--- | :--- |
| **Task 1** | Project Scaffolding, Architecture & Environment | 🟢 Completed |
| **Task 2** | Synthetic Graph Engine (BA, WS, SBM) | 🟢 Completed |
| **Task 2.5** | Initial Interactive Streamlit Interface | 🟢 Completed |
| **Task 3** | Graph Representations, Matrix Theory & Spectral Properties | 🟢 Completed |
| **Task 4** | Graph Analytics, Centralities & Community Detection | 🟢 Completed |
| **Task 5** | Comparative Topological Analysis & Benchmarking | 🟢 Completed |
| **Task 6** | Information Diffusion Models (ICM, LTM & Monte Carlo) | 🟢 Completed |
| **Task 7** | Temporal Diffusion Dynamics & Leakage-Free Observation Windows | 🟢 Completed |
| **Task 8** | Influencer vs. Bridge-Node Empirical Research Experiments | 🟢 Completed |
| **Task 9** | What-If Analysis, Inoculation & Structural Interventions | 🟢 Completed |
| **Task 10** | Machine Learning Virality Prediction & Feature Importance | 🟢 Completed |
| **Task 11** | Research Documentation, Proofs & Scientific Evaluation | 🟢 Completed |

| **Milestone 1** | Multipage Research Dashboard & Foundation | 🟢 Completed |
| **Milestone 2** | Post Studio & Content Injection Architecture | 🟢 Completed |
| **Milestone 3** | Live Diffusion Interactive Temporal Player | ⚪ Pending |

---

## 5. Development & Reproduction

```bash
# 1. Activate Virtual Environment
source .venv/bin/activate

# 2. Run Comprehensive Mathematical & Statistical Test Suite (85 Tests)
pytest tests/ -v

# 3. Execute Scientific Research Scripts
python scripts/demo_matrix_theory.py
python scripts/run_comparative_topological_analysis.py
python scripts/run_influencer_experiment.py
python scripts/run_ml_experiment.py

# 4. Launch Interactive Streamlit Interface
streamlit run app/main.py
```

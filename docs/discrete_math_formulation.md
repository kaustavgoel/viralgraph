# Discrete Mathematics & Theoretical Formulations in ViralGraph

This document provides formal mathematical definitions, proofs, and theoretical derivations underpinning the ViralGraph platform.

---

## 1. Graph Fundamentals & Set-Theoretic Definitions

Let $G = (V, E)$ be an **undirected simple graph**:
- $V = \{v_0, v_1, \dots, v_{n-1}\}$ is a finite set of $n = |V|$ vertices.
- $E \subseteq \{\{u, v\} : u, v \in V, u \neq v\}$ is a set of $m = |E|$ unordered 2-element subsets (edges).
- The open neighborhood of $u \in V$ is $\mathcal{N}(u) = \{v \in V : \{u, v\} \in E\}$.
- The degree of $u$ is $\deg(u) = |\mathcal{N}(u)|$.

### Theorem 1.1 (Euler's Handshaking Lemma)
For every graph $G = (V, E)$:
$$\sum_{u \in V} \deg(u) = 2|E|$$
*Proof*: Every edge $e = \{u, v\} \in E$ contributes exactly $1$ to $\deg(u)$ and $1$ to $\deg(v)$. Summing across all $u \in V$ counts every edge exactly twice. $\blacksquare$

---

## 2. Matrix Representations

### 2.1 Adjacency Matrix $A$
The adjacency matrix $A \in \{0, 1\}^{n \times n}$ has entries:
$$A_{uv} = \begin{cases} 1 & \text{if } \{u, v\} \in E \\ 0 & \text{otherwise} \end{cases}$$
- **Symmetry**: $A = A^T$ (undirected).
- **Zero Diagonal**: $A_{uu} = 0$ (no self-loops).
- $\sum_{v} A_{uv} = \deg(u)$ and $\sum_{u, v} A_{uv} = 2m$.

### 2.2 Walk Counting Theorem
**Theorem**: Let $A$ be the adjacency matrix of $G$. For any $k \in \mathbb{N}$, $(A^k)_{uv}$ equals the number of distinct walks of length $k$ from $u$ to $v$.

*Proof by Induction*:
- For $k=1$: $(A^1)_{uv} = A_{uv}$, which is 1 if an edge (walk of length 1) exists, 0 otherwise.
- Inductive hypothesis: Assume true for $k$.
- For $k+1$: $(A^{k+1})_{uv} = \sum_{w \in V} (A^k)_{uw} A_{wv}$.
  A walk of length $k+1$ from $u$ to $v$ consists of a walk of length $k$ from $u$ to some intermediate vertex $w$, followed by an edge $\{w, v\}$. Summing over all $w \in \mathcal{N}(v)$ partitions all possible length-$(k+1)$ walks from $u$ to $v$. $\blacksquare$

**Corollaries**:
1. $(A^2)_{uu} = \deg(u)$, and $\text{Tr}(A^2) = 2m$.
2. $\text{Tr}(A^3) = 6 \times (\text{Number of Triangles in } G)$.

---

## 3. Spectral Graph Theory & The Graph Laplacian

### 3.1 Combinatorial Laplacian $L$
$$L = D - A$$
where $D = \text{diag}(\deg(v_0), \dots, \deg(v_{n-1}))$.

### 3.2 Positive Semi-Definiteness
**Theorem**: The combinatorial Laplacian $L$ is symmetric and positive semi-definite ($L \succeq 0$).

*Proof*: For any vector $x \in \mathbb{R}^n$:
$$x^T L x = x^T D x - x^T A x = \sum_{u \in V} \deg(u) x_u^2 - \sum_{u \in V} \sum_{v \in V} A_{uv} x_u x_v$$
Since $\deg(u) = \sum_{v \in V} A_{uv}$:
$$\sum_{u \in V} \deg(u) x_u^2 = \sum_{u \in V} \sum_{v \in V} A_{uv} x_u^2 = \frac{1}{2} \sum_{u \in V} \sum_{v \in V} A_{uv} x_u^2 + \frac{1}{2} \sum_{u \in V} \sum_{v \in V} A_{uv} x_v^2$$
Therefore, summing over ordered pairs $(u, v) \in V \times V$:
$$x^T L x = \frac{1}{2} \sum_{u \in V} \sum_{v \in V} A_{uv} \left( x_u^2 - 2x_u x_v + x_v^2 \right) = \frac{1}{2} \sum_{u \in V} \sum_{v \in V} A_{uv} (x_u - x_v)^2$$
Equivalently, summing over each unordered edge $\{u, v\} \in E$ exactly once:
$$x^T L x = \sum_{\{u, v\} \in E} (x_u - x_v)^2 \ge 0$$
Because $(x_u - x_v)^2 \ge 0$ for all real components $x_u, x_v$, $x^T L x \ge 0$ for all $x \in \mathbb{R}^n$.
Thus, all eigenvalues of $L$ are non-negative real numbers: $0 = \lambda_1 \le \lambda_2 \le \dots \le \lambda_n$. $\blacksquare$

### 3.3 Algebraic Connectivity & Cheeger's Inequality
- $\lambda_1 = 0$ with eigenvector $\mathbf{1} = [1, 1, \dots, 1]^T$.
- $\text{dim}(\ker(L)) = |\mathcal{C}|$ (multiplicity of $0$ equals the number of connected components).
- $\lambda_2 > 0 \iff G$ is connected (Fiedler value).
- **Cheeger's Inequality**: Let $h(G) = \min_{\emptyset \subset S \subset V, |S| \le n/2} \frac{|\partial(S)|}{|S|}$ be the conductance of $G$, and $\Delta = \max_{u \in V} \deg(u)$ be the maximum degree. Then:
$$\frac{\lambda_2}{2} \le h(G) \le \sqrt{2 \Delta \lambda_2}$$
For the normalized Laplacian $L_{\text{sym}} = D^{-1/2} L D^{-1/2}$, with normalized conductance $h_{\text{norm}}(G)$:
$$\frac{\lambda_2^{\text{sym}}}{2} \le h_{\text{norm}}(G) \le \sqrt{2 \lambda_2^{\text{sym}}}$$

---

## 4. Stochastic Information Diffusion Processes

### 4.1 Independent Cascade Model (ICM)
- At $t=0$, $A_0 = S_0 \subset V$.
- At $t \ge 1$, each $u \in A_{t-1}$ attempts to activate inactive neighbor $v$ with probability $p_{uv}$.
- Each node gets **exactly one attempt**.
- Monotonicity: $A_{\le t-1} \subseteq A_{\le t}$.
- Expected reach function $\sigma(S_0) = \mathbb{E}[|A_{\le \infty}| \mid S_0]$ is non-negative, monotone, and **submodular**:
$$\sigma(S \cup \{v\}) - \sigma(S) \ge \sigma(T \cup \{v\}) - \sigma(T) \quad \forall S \subseteq T \subseteq V$$

### 4.2 Linear Threshold Model (LTM)
- Each node $v$ has an intrinsic threshold $\theta_v \sim U(0, 1)$.
- Influence weights $b_{uv} \ge 0$ satisfy $\sum_{u \in \mathcal{N}(v)} b_{uv} \le 1$.
- At step $t$, inactive $v$ activates if and only if:
$$\sum_{u \in \mathcal{N}(v) \cap A_{\le t-1}} b_{uv} \ge \theta_v$$
- Models complex contagion (social proof, peer reinforcement).

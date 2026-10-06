# Audit Report: Modules 4–7 LaTeX & Formatting Integrity
**Target File:** `/home/nilesh/code/transformers-from-scratch/Transformers_From_Scratch.md`  
**Scope:** Module 4, Module 5, Module 6, and Module 7 (Lines 1170 to 1719)  
**Auditor:** Technical Editor & LaTeX Auditor  
**Date:** 2026-10-06  

---

## Executive Summary

A comprehensive audit was performed across Modules 4 through 7 of *Transformers: The Atomic Dissection*. The audit scrutinized three critical dimensions:
1. **Mathematical Formulas & KaTeX Compatibility:** Every single inline (`$...$`) and display (`$$...$$`) LaTeX expression was extracted and rendered through KaTeX (v0.19.0) in strict mode. Delimiter parity, escape sequences, bracket matching, and multi-line formatting were validated.
2. **Diagram Integrity & ASCII-Art Elimination:** All 12 code blocks across Modules 4–7 were categorized. 4 blocks use native responsive SVG/XML diagrams, while 8 blocks remain as raw monospaced ASCII-art / text schematics.
3. **Variable Definitions & 2-Sentence Summaries:** All 12 subsections were verified for explicit variable breakdowns under `#### 📐 The Mathematics & Working` and compared against the 2-sentence summary requirement. In Modules 4–7, all summaries are present and strictly 2 sentences, using the header `**2-Sentence Working:**` (contrasting with `**2-Sentence Plain Lingo**:` in Modules 2–3). Minor variable coverage gaps (where secondary formula terms are explained in surrounding prose rather than in the bulleted definition list) are identified below with precise patch recommendations.

---

## 1. Mathematical Formulas & KaTeX Audit

### 1.1 Test Methodology & Quantitative Results
* **Total Display Equations (`$$...$$`):** 28 instances across Modules 4–7.
* **Total Inline Equations (`$...$`):** 170 instances across Modules 4–7.
* **Total LaTeX Expressions Evaluated:** 198 expressions.
* **KaTeX Execution Result (Node.js + KaTeX 0.19.0, `throwOnError: true, strict: 'error'`):** **0 errors / 0 failures**.
* **Delimiter Parity:** All opening and closing `$` and `$$` delimiters match cleanly. No unescaped currency dollar signs or broken markdown wrappers exist in the mathematical text.
* **HTML/PDF Pipeline Check:** Tested against `/home/nilesh/code/transformers-from-scratch/build_perfect_book.js`. Global KaTeX pre-rendering succeeds with 0 errors across the entire book.

### 1.2 Line-by-Line Inventory of Display Equations

| Line | Subsection | Formula Preview | KaTeX Status | Notes |
| :--- | :--- | :--- | :---: | :--- |
| **1183** | 4.1 | `$$\mathbf{x}_{\text{out}} = \mathbf{x}_{\text{in}} + \text{Sublayer}(\mathbf{x}_{\text{in}})$$` | Pass | Clean |
| **1186** | 4.1 | `$$\frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{in}}} = \frac{\partial \mathcal{L}}{\partial \mathbf{x}_{\text{out}}} \cdot \left( \mathbf{I} + \frac{\partial \text{Sublayer}(\mathbf{x}_{\text{in}})}{\partial \mathbf{x}_{\text{in}}} \right) = \dots$$` | Pass | Correct `\left( \dots \right)` grouping |
| **1208** | 4.1 | `$$\mathbf{x}^{(l)} = \mathbf{x}^{(l-1)} + f(\mathbf{x}^{(l-1)})$$` | Pass | Clean |
| **1231** | 4.2 | `$$\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$` | Pass | Standard LayerNorm form |
| **1235** | 4.2 | `$$\text{RMSNorm}(\mathbf{x}) = \frac{\mathbf{x}}{\text{RMS}(\mathbf{x})} \odot \gamma, \quad \text{where } \text{RMS}(\mathbf{x}) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$` | Pass | Correct `\quad` and `\text{where}` |
| **1244** | 4.2 | `$$\bar{x}_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}} \cdot \gamma_i$$` | Pass | Elementwise RMSNorm scalar |
| **1268** | 4.3 | `$$\mathbf{x}^{(l)} = \text{Norm}(\mathbf{x}^{(l-1)} + \text{Sublayer}(\mathbf{x}^{(l-1)}))$$` | Pass | Post-LN formulation |
| **1272** | 4.3 | `$$\mathbf{x}^{(l)} = \mathbf{x}^{(l-1)} + \text{Sublayer}(\text{Norm}(\mathbf{x}^{(l-1)}))$$` | Pass | Pre-LN formulation |
| **1290** | 4.3 | `$$\mathbf{x}^{(l)} = \mathbf{x}^{(l-1)} + \text{FFN}(\text{RMSNorm}(\mathbf{x}^{(l-1)}))$$` | Pass | Pre-LN SwiGLU highway |
| **1317** | 5.1 | `$$\text{FFN}(\mathbf{x}) = \sigma(\mathbf{x} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2$$` | Pass | Classic MLP formulation |
| **1342** | 5.1 | `$$\mathbf{y} = \sigma(\mathbf{x} W_{\text{gate}}) W_{\text{down}}$$` | Pass | Intermediate expansion |
| **1366** | 5.2 | `$$\text{SwiGLU}(\mathbf{x}) = \left( \text{Swish}(\mathbf{x} W_{\text{gate}}) \odot (\mathbf{x} W_{\text{up}}) \right) W_{\text{down}}$$` | Pass | Clean `\odot` and operator names |
| **1388** | 5.2 | `$$\mathbf{h}_{\text{ffn}} = \left( (\mathbf{x} W_{\text{gate}} \cdot \text{sigmoid}(\mathbf{x} W_{\text{gate}})) \odot (\mathbf{x} W_{\text{up}}) \right) W_{\text{down}}$$` | Pass | Expanded SwiGLU equation |
| **1414** | 6.1 | `$$z_i = \mathbf{h}_{\text{final}} \cdot \mathbf{w}_i$$` | Pass | Vocabulary dot product |
| **1434** | 6.1 | `$$\mathbf{z} = \text{RMSNorm}(\mathbf{h}_{\text{final}}) W_U$$` | Pass | Full unembedding step |
| **1459** | 6.2 | `$$P(\text{token}_i) = \frac{\exp(z_i / T)}{\sum_{j=1}^V \exp(z_j / T)}$$` | Pass | Temperature Softmax |
| **1473** | 6.2 | `$$P_i(T) = \frac{\exp(z_i / T)}{\sum_{k=1}^V \exp(z_k / T)}$$` | Pass | Calibrated sampling probability |
| **1500** | 6.3 | `$$\sum_{i \in V^{(p)}} P_{(i)} \ge p$$` | Pass | Top-p cumulative inequality |
| **1519** | 6.3 | `$$P'(w_i) = \begin{cases} \frac{P(w_i)}{\sum_{w_j \in V^{(p)}} P(w_j)} & \text{if } w_i \in V^{(p)} \\ 0 & \text{otherwise} \end{cases}$$` | Pass | KaTeX `cases` environment renders properly |
| **1549** | 7.1 | `$$\mathcal{L}_{\text{SFT}}(\theta) = -\sum_{t=1}^{|y|} \log P_\theta(y_t \mid x, y_{<t})$$` | Pass | Correct `\mid` usage |
| **1564** | 7.1 | `$$\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \sum_{t=1}^T \log \pi_\theta(y_t \mid x, y_{<t}) \right]$$` | Pass | Expectation over dataset |
| **1590** | 7.2 | `$$\mathcal{L}_{RM}(\phi) = -\log \sigma(R_\phi(x, y_w) - R_\phi(x, y_l))$$` | Pass | Bradley-Terry loss |
| **1593** | 7.2 | `$$\text{Reward}_{\text{total}} = R_\phi(x, y) - \beta \, \mathbb{D}_{\text{KL}}(\pi_\theta(y \mid x) \,\|\, \pi_{\text{ref}}(y \mid x))$$` | Pass | Valid `\|\` KL separator |
| **1614** | 7.2 | `$$\max_\theta \, \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ R_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right] - \gamma_{\text{entropy}} \mathcal{H}(\pi_\theta)$$` | Pass | PPO RLHF objective |
| **1637** | 7.3 | `$$r(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)}$$` | Pass | Implicit reward closed form |
| **1640** | 7.3 | `$$\mathcal{L}_{\text{DPO}}(\theta) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} \right) \right]$$` | Pass | Full DPO loss |
| **1658** | 7.3 | `$$\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \, \sigma(\hat{r}_l - \hat{r}_w) \left[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right]$$` | Pass | DPO gradient equation |
| **1684** | 7.4 | `$$A_i = \frac{r_i - \text{mean}(\{r_1 \dots r_G\})}{\text{std}(\{r_1 \dots r_G\}) + \epsilon}$$` | Pass | Advantage baseline |
| **1709** | 7.4 | `$$\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \left[ \min\left( \frac{\pi_\theta(o_i \mid q)}{\pi_{\text{old}}(o_i \mid q)} A_i, \, \text{clip}\left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\text{old}}(o_i \mid q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta \, \mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) \right]$$` | Pass | Full GRPO objective |

---

## 2. Diagram Integrity & ASCII-Art Audit

In Modules 4–7, diagrams appear in 12 distinct code blocks. 
* **4 diagrams are rendered as high-resolution inline SVGs (`xml` code fences).**
* **8 diagrams remain as raw monospaced ASCII-art / box-drawing text schematics.**

### 2.1 Complete Inventory of Code Blocks in Modules 4–7

| Subsection | Line Range | Format / Lang | Description / Content | Status |
| :--- | :--- | :--- | :--- | :--- |
| **4.1** | **1190–1205** | ```` ``` ```` | Box-drawing ASCII flowchart of the Residual Stream Highway | **ASCII Art** |
| **4.2** | **1238–1241** | ```` ``` ```` | 2-line arrow schematic comparing LayerNorm vs RMSNorm pipeline | **ASCII / Text** |
| **4.3** | **1275–1287** | ```` ``` ```` | Side-by-side block diagram comparing Post-LN (2017) vs Pre-LN (Modern) | **ASCII Art** |
| **5.1** | **1322–1339** | ```` ``` ```` | Vertical expansion-contraction pipeline of 2-layer FFN ($W_1 \to \sigma \to W_2$) | **ASCII Art** |
| **5.2** | **1375–1385** | ```` ``` ```` | Dual-branch SwiGLU gating flowchart ($W_{\text{gate}}$ and $W_{\text{up}}$ into $\odot$) | **ASCII Art** |
| **6.1** | **1418–1431** | ```` ```xml ```` | Scalable SVG: $h_{\text{final}} \to W_U \text{ Head} \to \text{Vocab Logits } z$ | **Rendered SVG** |
| **6.2** | **1464–1470** | ```` ``` ```` | Monospaced probability shift table for Cold ($T=0.2$), Normal ($T=1.0$), Hot ($T=3.0$) | **Monospace Text** |
| **6.3** | **1506–1516** | ```` ``` ```` | Monospaced trace of Top-$p$ cutoff accumulation ($0.88 + 0.05 \ge 0.90$) | **Monospace Text** |
| **7.1** | **1551–1561** | ```` ```xml ```` | Scalable SVG: Prompt Masking (Weight = 0.0) vs Target Response Cross-Entropy | **Rendered SVG** |
| **7.2** | **1595–1611** | ```` ```xml ```` | Scalable SVG: 3-branch RLHF pipeline (Prompt $\to \pi_\theta$, $\pi_{\text{ref}}$, $R_\phi$, KL Penalty) | **Rendered SVG** |
| **7.3** | **1644–1655** | ```` ``` ```` | Flowchart of DPO objective: preference pairs $\to$ Log-Ratios $\to$ Delta $\to$ Loss | **ASCII Art** |
| **7.4** | **1690–1706** | ```` ```xml ```` | Scalable SVG: GRPO pipeline (Math/Code $Q \to G$ Rollouts $\to$ Rule Verifiers) | **Rendered SVG** |

### 2.2 Recommendation for Book PDF Publication
The 4 SVGs (Sections 6.1, 7.1, 7.2, 7.4) render with modern styled pill containers and high aesthetic fidelity in both HTML and PDF. The 8 ASCII blocks currently render cleanly as `<pre class="code-block"><code>...</code></pre>`. However, for a cohesive, professional textbook look matching the rest of the book's architecture diagrams, replacing the 5 structural ASCII box diagrams (1190–1205, 1275–1287, 1322–1339, 1375–1385, and 1644–1655) with responsive SVG blocks (using the color palette `#EEF2F6`, `#FEF3C7`, `#DCFCE7`, `#FEE2E2`) is strongly recommended.

---

## 3. Variable Definitions & 2-Sentence Plain-Lingo Summaries

### 3.1 Structural Standard Comparison
* In **Modules 2 and 3**, the summary header is:
  `**2-Sentence Plain Lingo**:`
* In **Modules 4, 5, 6, and 7**, the summary header is:
  `**2-Sentence Working:**`
Both serve the exact same pedagogical role: providing a 2-sentence non-jargon intuitive explanation immediately below the variable breakdown. Every subsection in Modules 4–7 possesses exactly one summary block, and every single summary block contains **exactly two complete sentences**.

### 3.2 Detailed Audit by Subsection

#### Subsection 4.1: Residual Skip Connections ($x + \text{Sublayer}(x)$)
* **Lines:** 1174–1221
* **Main Equation (L1208):** $\mathbf{x}^{(l)} = \mathbf{x}^{(l-1)} + f(\mathbf{x}^{(l-1)})$
* **Variables Defined (L1209–1211):**
  * $\mathbf{x}^{(l-1)} \in \mathbb{R}^{N \times d_{model}}$: Input residual stream state entering layer $l$.
  * $f(\cdot)$: The sublayer transformation (Multi-Head Attention or Feed-Forward Network).
  * $\mathbf{x}^{(l)} \in \mathbb{R}^{N \times d_{model}}$: Updated residual stream state passed to layer $l+1$.
* **Definition Completeness:** **Complete.** All variables in the main formula are explicitly defined. The earlier backprop derivation (L1186) also explicitly names $\mathbf{I}$ and $\mathcal{L}$ in prose (L1185, 1188).
* **Summary (L1213–1214):**
  * *Header:* `**2-Sentence Working:**`
  * *Sentence 1:* "Instead of forcing each layer to generate a brand new token vector from scratch, the architecture keeps the original vector intact and merely adds the layer's new edits on top of it."
  * *Sentence 2:* "This creates a friction-free express lane that lets error signals travel thousands of steps backwards during training without fading away."
  * *Evaluation:* **Pass (2 sentences, clear, non-jargon).**

---

#### Subsection 4.2: Layer Normalization vs. RMSNorm
* **Lines:** 1222–1258
* **Main Equation (L1244):** $\bar{x}_i = \frac{x_i}{\sqrt{\frac{1}{d}\sum_{j=1}^d x_j^2 + \epsilon}} \cdot \gamma_i$
* **Variables Defined (L1245–1248):**
  * $x_i$: The $i$-th coordinate of the token vector $\mathbf{x} \in \mathbb{R}^{d_{model}}$.
  * $d$: Total hidden dimension $d_{model}$ (e.g. 4096).
  * $\epsilon$: Small constant (e.g. $10^{-6}$) preventing division by zero.
  * $\gamma_i$: Learnable gain parameter restoring representational flexibility.
* **Definition Completeness:** **Complete.** All symbols in the scalar equation are rigorously defined with dimensional types.
* **Summary (L1250–1251):**
  * *Header:* `**2-Sentence Working:**`
  * *Sentence 1:* "The model calculates the average power across all coordinates of a single word's vector and divides the vector by that number to keep its length strictly in check."
  * *Sentence 2:* "It then multiplies the result by a learned volume dial so important features can still stand out when needed."
  * *Evaluation:* **Pass (2 sentences, intuitive analogy).**

---

#### Subsection 4.3: Pre-LN vs. Post-LN Architecture
* **Lines:** 1259–1303
* **Main Equation (L1290):** $\mathbf{x}^{(l)} = \mathbf{x}^{(l-1)} + \text{FFN}(\text{RMSNorm}(\mathbf{x}^{(l-1)}))$
* **Variables Defined (L1291–1293):**
  * $\mathbf{x}^{(l-1)}$: Clean, unnormalized residual state.
  * $\text{RMSNorm}(\mathbf{x}^{(l-1)})$: Normalized temporary copy sent only into the sublayer.
  * $\text{FFN}(\cdot)$: The feed-forward non-linear network.
* **Definition Completeness:** **Complete.** 
* **Summary (L1295–1296):**
  * *Header:* `**2-Sentence Working:**`
  * *Sentence 1:* "Instead of washing and resizing the main information highway at every toll booth, the model makes a temporary normalized copy of the data and feeds that copy into the processing engine."
  * *Sentence 2:* "The resulting output is then added straight back onto the raw, uninterrupted highway."
  * *Evaluation:* **Pass (2 sentences, highway toll booth analogy).**

---

#### Subsection 5.1: The Two-Layer FFN Expansion ($W_1, W_2$)
* **Lines:** 1308–1356
* **Main Equation (L1342):** $\mathbf{y} = \sigma(\mathbf{x} W_{\text{gate}}) W_{\text{down}}$
* **Variables Defined (L1343–1346):**
  * $\mathbf{x} \in \mathbb{R}^{1 \times d_{model}}$: Input token vector.
  * $W_{\text{gate}} \in \mathbb{R}^{d_{model} \times d_{ff}}$: Expansion matrix mapping to the wider intermediate dimension.
  * $\sigma(\cdot)$: Non-linear activation function.
  * $W_{\text{down}} \in \mathbb{R}^{d_{ff} \times d_{model}}$: Contraction matrix mapping back to residual dimension.
* **Definition Completeness:** **Complete.** (Matches the notation of modern FFN implementations; the classical $W_1, W_2, \mathbf{b}_1, \mathbf{b}_2$ notation from L1317 is also fully defined in lines 1318–1320).
* **Summary (L1348–1349):**
  * *Header:* `**2-Sentence Working:**`
  * *Sentence 1:* "The model takes the word vector and blows it up into a four-times wider array where thousands of individual concept detectors can scan it and light up."
  * *Sentence 2:* "Whichever detectors fire then pour their stored factual knowledge back down into the word's normal vector size."
  * *Evaluation:* **Pass (2 sentences, clear explanation of key-value memory behavior).**

---

#### Subsection 5.2: Modern Gated Activation: SwiGLU (Shazeer, 2020)
* **Lines:** 1357–1400
* **Main Equation (L1388):** $\mathbf{h}_{\text{ffn}} = \left( (\mathbf{x} W_{\text{gate}} \cdot \text{sigmoid}(\mathbf{x} W_{\text{gate}})) \odot (\mathbf{x} W_{\text{up}}) \right) W_{\text{down}}$
* **Variables Defined (L1389–1391):**
  * $\odot$: Elementwise Hadamard product.
  * $\text{sigmoid}(z) = \frac{1}{1 + e^{-z}}$: Smooth probability gate.
  * $W_{\text{gate}}, W_{\text{up}}, W_{\text{down}}$: The three learned linear projections of the SwiGLU block.
* **Definition Completeness:** **Minor Observation:** $\mathbf{x}$ and $\mathbf{h}_{\text{ffn}}$ are described in the preceding text and Section 5.1, but adding explicit bullet entries for $\mathbf{x} \in \mathbb{R}^{1 \times d_{model}}$ and $\mathbf{h}_{\text{ffn}} \in \mathbb{R}^{1 \times d_{model}}$ enhances mathematical self-containment.
* **Summary (L1393–1394):**
  * *Header:* `**2-Sentence Working:**`
  * *Sentence 1:* "Instead of abruptly shutting off negative numbers with a hard switch, the model computes both a proposed thought and a smooth dial that controls how much of that thought should be allowed through."
  * *Sentence 2:* "Multiplying the thought by its dial lets the network dynamically amplify or silence individual concepts with extreme precision."
  * *Evaluation:* **Pass (2 sentences, smooth dial concept).**

---

#### Subsection 6.1: The Unembedding Head ($W_U$) & Logits
* **Lines:** 1405–1447
* **Main Equation (L1434):** $\mathbf{z} = \text{RMSNorm}(\mathbf{h}_{\text{final}}) W_U$
* **Variables Defined (L1435–1437):**
  * $\mathbf{z} \in \mathbb{R}^{1 \times V}$: Raw output logit vector across vocabulary $V$.
  * $\mathbf{h}_{\text{final}} \in \mathbb{R}^{1 \times d_{model}}$: Final residual stream state for the current token position.
  * $W_U \in \mathbb{R}^{d_{model} \times V}$: Unembedding projection parameter matrix.
* **Definition Completeness:** **Complete.** (The scalar projection $z_i = \mathbf{h}_{\text{final}} \cdot \mathbf{w}_i$ on L1414 is also fully defined in lines 1413–1416).
* **Summary (L1439–1440):**
  * *Header:* `**2-Sentence Working:**`
  * *Sentence 1:* "The model takes the final mathematical summary vector of the sentence and compares it against every single word in its dictionary using dot products."
  * *Sentence 2:* "The higher the resulting score for a word, the more confident the model is that this word should come next."
  * *Evaluation:* **Pass (2 sentences, intuitive dot-product dictionary comparison).**

---

#### Subsection 6.2: Temperature Scaling ($T$) & Entropy
* **Lines:** 1448–1486
* **Main Equation (L1473):** $P_i(T) = \frac{\exp(z_i / T)}{\sum_{k=1}^V \exp(z_k / T)}$
* **Variables Defined (L1474–1476):**
  * $z_i$: Unnormalized logit for vocabulary token $i$.
  * $T \in (0, \infty)$: Temperature hyperparameter.
  * $P_i(T)$: Calibrated next-token probability.
* **Definition Completeness:** **Complete.** $V$ is well-known from 6.1, though adding $* V$: Total vocabulary size would guarantee 100% self-containment.
* **Summary (L1478–1479):**
  * *Header:* `**2-Sentence Working:**`
  * *Sentence 1:* "Temperature acts like a contrast slider for the model's confidence scores before turning them into probabilities."
  * *Sentence 2:* "Turning the temperature down makes the top choice drown out all competitors, while turning it up gives unusual and risky words a fighting chance to be picked."
  * *Evaluation:* **Pass (2 sentences, contrast slider analogy).**

---

#### Subsection 6.3: Nucleus Sampling (Top-$p$) & Top-$k$
* **Lines:** 1487–1531
* **Main Equation (L1519):** $P'(w_i) = \begin{cases} \frac{P(w_i)}{\sum_{w_j \in V^{(p)}} P(w_j)} & \text{if } w_i \in V^{(p)} \\ 0 & \text{otherwise} \end{cases}$
* **Variables Defined (L1520–1521):**
  * $V^{(p)}$: The smallest subset of tokens such that $\sum_{w_i \in V^{(p)}} P(w_i) \ge p$.
  * $p \in (0, 1]$: Cumulative probability cutoff (typically $0.85$ to $0.95$).
* **Definition Completeness:** **Minor Observation:** Defining $w_i$ (token candidate) and $P(w_i)$ (original Softmax probability) in the bullet list explicitly would make the formula block fully self-contained.
* **Summary (L1523–1524):**
  * *Header:* `**2-Sentence Working:**`
  * *Sentence 1:* "Top-p sampling draws a cutoff line right where the cumulative probability of the most likely words hits 90%, immediately discarding the long tail of bizarre words."
  * *Sentence 2:* "It then stretches the remaining sensible candidates so they fill 100% of the wheel and spins to pick the winner."
  * *Evaluation:* **Pass (2 sentences, roulette wheel stretching analogy).**

---

#### Subsection 7.1: Supervised Fine-Tuning (SFT) & Instruction Tuning
* **Lines:** 1536–1577
* **Main Equation (L1564):** $\mathcal{L}_{\text{SFT}} = -\mathbb{E}_{(x, y) \sim \mathcal{D}} \left[ \sum_{t=1}^T \log \pi_\theta(y_t \mid x, y_{<t}) \right]$
* **Variables Defined (L1565–1567):**
  * $(x, y) \sim \mathcal{D}$: High-quality curated instruction dataset.
  * $\pi_\theta$: The model parameterized by weights $\theta$.
  * $y_t$: Target response token at step $t$.
* **Definition Completeness:** **Complete.** ($x$ is the prompt sequence, $T = |y|$ is response token length; both defined in L1545–1549).
* **Summary (L1569–1570):**
  * *Header:* `**2-Sentence Working:**`
  * *Sentence 1:* "The model is fed thousands of pristine examples of ideal questions followed by ideal answers, and its weights are updated only on the answer portion."
  * *Sentence 2:* "This conditions the model to transition from continuing random web pages to behaving as a helpful, obedient assistant."
  * *Evaluation:* **Pass (2 sentences, clear contrast between base model and assistant).**

---

#### Subsection 7.2: Reward Modeling & RLHF (PPO)
* **Lines:** 1578–1627
* **Main Equation (L1614):** $\max_\theta \, \mathbb{E}_{x \sim \mathcal{D}, y \sim \pi_\theta} \left[ R_\phi(x, y) - \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\text{ref}}(y \mid x)} \right] - \gamma_{\text{entropy}} \mathcal{H}(\pi_\theta)$
* **Variables Defined (L1615–1617):**
  * $R_\phi(x, y)$: Learned scalar reward predicting human approval.
  * $\pi_{\text{ref}}$: Frozen reference SFT model preventing catastrophic forgetting.
  * $\beta$: Regularization coefficient controlling the strength of the KL leash.
* **Definition Completeness:** **Minor Observation:** The entropy term $-\gamma_{\text{entropy}} \mathcal{H}(\pi_\theta)$ appears in the formula; defining $\mathcal{H}(\pi_\theta)$ (policy entropy) and $\gamma_{\text{entropy}}$ in the bullet list completes all terms.
* **Summary (L1619–1620):**
  * *Header:* `**2-Sentence Working:**`
  * *Sentence 1:* "A scoring model acts as a teacher assigning points to the assistant's answers based on human preferences, and the assistant uses trial and error to learn what phrasing earns the highest score."
  * *Sentence 2:* "A strict leash tethered to the original model prevents the assistant from finding bizarre verbal tricks that fool the score keeper."
  * *Evaluation:* **Pass (2 sentences, scorekeeper teacher & leash analogy).**

---

#### Subsection 7.3: Direct Preference Optimization (DPO - Rafailov et al. 2023)
* **Lines:** 1628–1671
* **Main Equation (L1658):** $\nabla_\theta \mathcal{L}_{\text{DPO}} = -\beta \, \sigma(\hat{r}_l - \hat{r}_w) \left[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \right]$
* **Variables Defined (L1659–1661):**
  * $\hat{r}_w = \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}$: Implicit reward for the winning response.
  * $\hat{r}_l = \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}$: Implicit reward for the losing response.
  * $\beta$: Regularization hyperparameter (controls conservatism).
* **Definition Completeness:** **Complete.** ($y_w, y_l$ are the winning/losing answers, and $\sigma$ is the sigmoid function, defined in L1632–1640).
* **Summary (L1663–1664):**
  * *Header:* `**2-Sentence Working:**`
  * *Sentence 1:* "DPO compares how much the model prefers the good answer versus the bad answer relative to a baseline model, and increases the probability of the good answer while directly pushing down the bad answer."
  * *Sentence 2:* "It achieves the exact same results as complex reinforcement learning with the simplicity of standard supervised training."
  * *Evaluation:* **Pass (2 sentences, concise and accurate).**

---

#### Subsection 7.4: Reasoning RL & Verifiable Rewards (GRPO / DeepSeek-R1 / o1)
* **Lines:** 1672–1719
* **Main Equation (L1709):** $\mathcal{L}_{\text{GRPO}}(\theta) = -\frac{1}{G} \sum_{i=1}^G \left[ \min\left( \frac{\pi_\theta(o_i \mid q)}{\pi_{\text{old}}(o_i \mid q)} A_i, \, \text{clip}\left(\frac{\pi_\theta(o_i \mid q)}{\pi_{\text{old}}(o_i \mid q)}, 1-\epsilon, 1+\epsilon\right) A_i \right) - \beta \, \mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}}) \right]$
* **Variables Defined (L1710–1712):**
  * $G$: Number of parallel reasoning rollouts sampled per question.
  * $A_i$: Normalized advantage score computed relative to the peer group.
  * $\text{clip}(\dots)$: PPO-style clipping preventing destructive policy updates.
* **Definition Completeness:** **Minor Observation:** $\beta$ and $\mathbb{D}_{\text{KL}}(\pi_\theta \,\|\, \pi_{\text{ref}})$ appear in the equation and are defined in 7.2; including a reminder bullet for $\beta$ and $\pi_{\text{old}}$ (the reference policy before the current update step) adds total clarity.
* **Summary (L1714–1715):**
  * *Header:* `**2-Sentence Working:**`
  * *Sentence 1:* "The model generates eight different attempts at solving a math or coding problem, and an automated computer program checks which ones got the exact right answer."
  * *Sentence 2:* "The steps that led to the correct answers are rewarded and strengthened, teaching the model to double-check its work and reason through complex problems step-by-step."
  * *Evaluation:* **Pass (2 sentences, clear RL-from-verifiable-feedback explanation).**

---

## 4. Suggested Editorial Improvements & Corrections

The document currently renders without technical error. To reach textbook publication perfection across all modules, the following targeted editorial refinements are recommended:

### 4.1 Header Consistency
* **Observation:** Modules 2 and 3 use `**2-Sentence Plain Lingo**:` (e.g. Lines 577, 629, 687, 743, 794, 863, 924, 1081, 1162), whereas Modules 4, 5, 6, and 7 use `**2-Sentence Working:**` (Lines 1213, 1250, 1295, 1348, 1393, 1439, 1478, 1523, 1569, 1619, 1663, 1714).
* **Recommendation:** Standardize the phrasing across all modules. If `Plain Lingo` is preferred book-wide, rename the headers in Modules 4–7 to `**2-Sentence Plain Lingo**:`.

### 4.2 Comprehensive Variable Bullet Additions
To make every equation in `#### 📐 The Mathematics & Working` 100% self-contained without consulting previous sections:

1. **Section 5.2 (L1391):** Add bullets for the input and output vectors:
   ```markdown
   * $\mathbf{x} \in \mathbb{R}^{1 \times d_{model}}$: Input residual activation vector.
   * $\mathbf{h}_{\text{ffn}} \in \mathbb{R}^{1 \times d_{model}}$: Output vector injected back into the residual highway.
   ```
2. **Section 6.3 (L1521):** Add bullets for candidates and base probabilities:
   ```markdown
   * $w_i$: Vocabulary candidate token.
   * $P(w_i)$: Original probability assigned to token $w_i$ prior to truncation.
   ```
3. **Section 7.2 (L1617):** Add bullets for policy entropy:
   ```markdown
   * $\mathcal{H}(\pi_\theta)$: Policy entropy encouraging exploration.
   * $\gamma_{\text{entropy}}$: Weight of the entropy regularization bonus.
   ```
4. **Section 7.4 (L1712):** Add bullets for policy ratios and KL divergence:
   ```markdown
   * $\pi_{\text{old}}$: Policy parameters prior to the current optimization epoch.
   * $\beta \, \mathbb{D}_{\text{KL}}$: KL penalty keeping the policy bounded to the reference base model.
   ```

### 4.3 Visual Modernization of ASCII Blocks
Replace the 5 structural ASCII box diagrams (Residual Stream Highway L1190–1205, Pre-LN vs Post-LN L1275–1287, FFN Expansion L1322–1339, SwiGLU Gating L1375–1385, and DPO Flowchart L1644–1655) with responsive inline SVG code blocks (` ```xml `), aligning them with the style used in Sections 6.1, 7.1, 7.2, and 7.4.

---

## 5. Audit Conclusion

Modules 4, 5, 6, and 7 exhibit **outstanding technical rigor and formatting health**:
* **0 broken KaTeX formulas.**
* **100% compliant 2-sentence intuitive summaries in all 12 subsections.**
* **Mathematical definitions are accurate, robust, and aligned with modern frontier LLM literature (LLaMA-3, SwiGLU, RMSNorm, DPO, DeepSeek GRPO).**
* **The HTML and PDF generation pipelines build cleanly without math rendering warnings.**

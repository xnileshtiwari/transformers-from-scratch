# 🧠 Transformers: From Scratch
> **An Irreducible, First-Principles Guide to Modern LLM Architecture & Post-Training**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PDF Available](https://img.shields.io/badge/Format-PDF%20%2863%20Pages%29-red)](Transformers_From_Scratch.pdf)
[![HTML Ready](https://img.shields.io/badge/Format-Interactive%20HTML-blue)](index.html)
[![AI Generated Diagrams](https://img.shields.io/badge/Diagrams-Nano%20Banana%20%28Gemini%203.1%29-emerald)](#-visual-architectural-gallery)

---

## 📌 Overview

Most Transformer tutorials present the architecture as a single monolithic block diagram. In contrast, **Transformers: From Scratch** breaks down modern Large Language Models into **23 irreducible atomic pieces** across **8 core modules**—from raw static Word2Vec dictionary embeddings to modern reasoning reinforcement learning (GRPO / DeepSeek-R1 / OpenAI o1).

Every single atomic piece adheres strictly to an unyielding 4-part pedagogical contract:
1. 💡 **`Why?` Block:** Explains the exact mathematical or functional necessity, its architectural role, and what specific LLM behavior breaks if omitted.
2. 📖 **`Definition & Architecture`:** Formal explanation grounded in top-tier literature (Vaswani et al., RoFormer, LLaMA, Chinchilla, DPO, GRPO) paired with high-resolution visual diagrams.
3. 📐 **`Mathematics & Working`:** Exact formulas with **every variable explicitly defined**, concluded with a **2-sentence plain-lingo summary** explaining the mechanical intuition.
4. 🌍 **`Real-World & NLP Behaviors` (2 Examples):** Concrete production NLP case studies, failure modes, and industry engineering practices.

---

## 🗺️ Complete Architecture Dependency Graph

```mermaid
graph TD
    subgraph S0 [Module 0: The Bridge from Word2Vec]
        M0_1[0.1 Polysemy & Word2Vec Collapse] --> M0_2[0.2 Token Embedding Matrix & Residual Stream]
    end

    subgraph S1 [Module 1: Spatial & Order Foundations]
        M1_1[1.1 Permutation Invariance] --> M1_2[1.2 Absolute Sinusoidal PE]
        M1_1 --> M1_3[1.3 Rotary Position Embedding RoPE]
    end

    M0_2 --> STREAM[The Residual Stream Highway]
    M1_2 -.-> STREAM
    M1_3 -.-> M2_1

    subgraph S2 [Module 2: Self-Attention Atomic Engine]
        STREAM --> M2_1[2.1 Linear Projections: Q, K, V]
        M2_1 --> M2_2[2.2 Dot-Product Compatibility: Q x K^T]
        M2_2 --> M2_3[2.3 Variance Scaling: 1 / sqrt dk]
        M2_3 --> M2_4[2.4 Softmax Probability Normalization]
        M2_4 --> M2_5[2.5 Value Aggregation: Softmax x V]
    end

    subgraph S3 [Module 3: Multi-Head & Causal Decoding]
        M2_5 --> M3_1[3.1 Multi-Head Subspace Splitting]
        M3_1 --> M3_2[3.2 Output Projection Matrix: W_O]
        M2_3 --> M3_3[3.3 Causal Attention Masking]
        M3_3 --> M2_4
        M2_1 --> M3_4[3.4 KV-Cache Dynamic Memory]
    end

    subgraph S4 [Module 4: Signal Integrity & Normalization]
        STREAM --> M4_1[4.1 Residual Skip Connections]
        M3_2 --> M4_1
        M4_1 --> M4_2[4.2 Normalization: LayerNorm vs RMSNorm]
        M4_2 --> M4_3[4.3 Placement Dynamics: Pre-LN vs Post-LN]
    end

    subgraph S5 [Module 5: Factual Memory & Computation FFN]
        M4_3 --> M5_1[5.1 Two-Layer Expansion FFN]
        M5_1 --> M5_2[5.2 Gated Activations: SwiGLU]
    end

    M5_2 --> RES2[Post-FFN Residual Accumulator]

    subgraph S6 [Module 6: Vocabulary Projection & Sampling]
        RES2 --> M6_1[6.1 Unembedding Head: W_U Logits]
        M6_1 --> M6_2[6.2 Logit Temperature Scaling]
        M6_2 --> M6_3[6.3 Nucleus & Top-k Sampling]
    end

    subgraph S7 [Module 7: Post-Training, Alignment & Reasoning RL]
        M6_3 --> M7_1[7.1 Supervised Fine-Tuning SFT]
        M7_1 --> M7_2[7.2 Reward Modeling & RLHF PPO]
        M7_1 --> M7_3[7.3 Direct Preference Optimization DPO]
        M7_1 --> M7_4[7.4 Reasoning RL & Verifiable Rewards GRPO]
    end
```

---

## 📚 Curriculum & Module Breakdown

### [Module 0: The Bridge from Word2Vec](Transformers_From_Scratch.md#module-0-the-bridge-from-word2vec-to-dynamic-representations)
* **0.1 Word2Vec's Polysemy Collapse:** Why static dictionaries fail when `"apple"` can mean fruit or technology; how dynamic contextual attention constructs sentence-specific vector trajectories.
* **0.2 Token Embedding Lookup ($W_E$) & Residual Stream:** Projecting discrete integer vocabulary tokens into $\mathbb{R}^{d_{model}}$, introducing the persistent vector accumulator.

### [Module 1: Spatial & Order Foundations](Transformers_From_Scratch.md#module-1-spatial--order-foundations-why-transformers-need-position)
* **1.1 Permutation Invariance of Set Operations:** Proof that raw attention treats input sequences as unordered sets (`"dog bites man"` = `"man bites dog"` without positional bias).
* **1.2 Absolute Sinusoidal Positional Encoding (Vaswani 2017):** Trigonometric wavelength spectrums ranging from local bigram transitions to macro-document pacing.
* **1.3 Rotary Position Embedding (RoPE):** Complex 2D coordinate plane rotation where inner products depend strictly on relative displacement $(m - n)$, enabling context extrapolation to 128K+ tokens.

### [Module 2: Self-Attention Dissected](Transformers_From_Scratch.md#module-2-self-attention-dissected-the-core-router)
* **2.1 The Three Projections ($W_Q, W_K, W_V$):** Decoupling token roles into Searcher ($Q$), Matcher ($K$), and Payload ($V$) to unlock asymmetric, directional syntactic routing.
* **2.2 Raw Compatibility Scoring ($Q K^T$):** Pairwise inner-product affinity metrics.
* **2.3 Variance Scaling Factor ($1/\sqrt{d_k}$):** Statistical proof of dot-product variance expansion and how dividing by $\sqrt{d_k}$ prevents Softmax saturation and zero-gradient collapse.
* **2.4 Softmax Probability Normalization:** Row-stochastic probability simplex generation.
* **2.5 Weighted Value Aggregation ($\text{Softmax} \times V$):** Barycentric interpolation assembling the final contextualized token representation.

### [Module 3: Multi-Head & Causal Decoding](Transformers_From_Scratch.md#module-3-multi-head-architecture--causal-decoding)
* **3.1 Multi-Head Subspace Splitting ($h$ Heads):** Tracking simultaneous orthogonal linguistic features (syntax, coreference, factual induction).
* **3.2 Output Linear Projection ($W_O$):** Cross-head reconciliation and residual stream dimension matching.
* **3.3 Causal Triangular Attention Masking:** Upper-triangular $-\infty$ masking preventing future token cheating in autoregressive generation.
* **3.4 The KV-Cache Mechanism:** Caching historical Keys and Values in VRAM to drop decode step compute from $O(N^2)$ to $O(1)$.

### [Module 4: Signal Integrity & Normalization](Transformers_From_Scratch.md#module-4-signal-integrity--normalization-the-residual-highway--stability)
* **4.1 Residual Skip Connections ($x + \text{Sublayer}(x)$):** The uninterrupted identity highway that solves vanishing gradients across 32–80 layers.
* **4.2 Layer Normalization vs. RMSNorm:** Why dropping the mean $\mu$ and bias terms in RMSNorm optimizes GPU memory bandwidth without sacrificing training stability.
* **4.3 Pre-LN Architecture:** Preserving clean residual signals to enable training without fragile warmup schedules.

### [Module 5: Factual Memory & Computation (FFN)](Transformers_From_Scratch.md#module-5-factual-memory--computation-the-feed-forward-network--ffn)
* **5.1 Two-Layer FFN Expansion ($W_1, W_2$):** Expanding into $4\times d_{model}$ as an associative key-value memory bank storing world facts.
* **5.2 Modern Gated Activation (SwiGLU):** Multiplicative gating ($\text{Swish}(x W_{\text{gate}}) \odot x W_{\text{up}}$) for smooth, continuous feature selection.

### [Module 6: Token Emission & Sampling](Transformers_From_Scratch.md#module-6-token-emission--sampling-from-residual-vector-to-text)
* **6.1 The Unembedding Head ($W_U$) & Logits:** Projecting the final hidden state back to vocabulary size $|\mathcal{V}|$.
* **6.2 Temperature Scaling ($T$):** Contrast control adjusting distribution entropy between deterministic precision ($T=0.2$) and creative diversity ($T=1.0$).
* **6.3 Nucleus (Top-$p$) & Top-$k$ Sampling:** Dynamic cumulative probability thresholds that eliminate the long tail of nonsensical words without entering repetitive greedy loops.

### [Module 7: Post-Training, Alignment & Reasoning RL](Transformers_From_Scratch.md#module-7-post-training-alignment--reasoning-rl)
* **7.1 Supervised Fine-Tuning (SFT):** Masked instruction tuning transforming document predictors into obedient assistants.
* **7.2 Reward Modeling & RLHF (PPO):** Bradley-Terry human preference models with KL-divergence regularization.
* **7.3 Direct Preference Optimization (DPO):** Closed-form implicit reward substitution eliminating separate reward models and PPO training instability.
* **7.4 Reasoning RL & Verifiable Rewards (GRPO / DeepSeek-R1 / o1):** Rule-based automated verifiers and group-relative policy updates creating autonomous self-verification, backtracking, and long chain-of-thought reasoning.

---

## 🎨 Visual Architectural Gallery

All architectural figures were generated via **Google's Nano Banana 2 (Gemini 3.1 Flash Image)** to ensure clear, high-contrast, professional textbook clarity:

| Concept | Architectural Figure |
|---|---|
| **0.1 Word2Vec vs. Transformer** | ![Polysemy Trajectories](assets/diagram_0_polysemy.jpg) |
| **1.3 Rotary Position Embedding (RoPE)** | ![RoPE 2D Complex Plane](assets/diagram_1_rope.jpg) |
| **2.1 Self-Attention Dataflow** | ![Scaled Dot-Product Attention](assets/diagram_2_attention.jpg) |
| **3.4 Causal Masking & KV-Cache** | ![Causal Mask & KV-Cache](assets/diagram_3_kv_cache.jpg) |
| **4.1 Residual Stream & Pre-LN RMSNorm** | ![Residual Highway](assets/diagram_4_residual_norm.jpg) |
| **5.2 SwiGLU Gated Activation** | ![SwiGLU Architecture](assets/diagram_5_swiglu.jpg) |
| **6.2 Temperature & Top-p Sampling** | ![Token Sampling Calibration](assets/diagram_6_sampling.jpg) |
| **7.4 Reasoning RL & Verifiable Rewards** | ![Post-Training Alignment & GRPO](assets/diagram_7_post_training.jpg) |

---

## 📖 Downloads & Formats

* **Complete PDF (63 Pages):** [Download `Transformers_From_Scratch.pdf`](Transformers_From_Scratch.pdf)
* **Interactive HTML Edition:** Open [`index.html`](index.html) in any browser (typeset with Bookerly font, KaTeX vector equations, and embedded figures).
* **Raw Markdown Source:** Read [`Transformers_From_Scratch.md`](Transformers_From_Scratch.md).

---

## 📄 License
This repository is released under the [MIT License](LICENSE).

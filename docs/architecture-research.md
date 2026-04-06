# Architecture Research — Modern LLM Inference Engine

This document summarizes six influential research papers and their relevance to building a modern, portable C++ LLM inference engine (hesa-llm), targeting a **Gemma-4 class** model (approximately 27B parameters, efficient inference, strong reasoning).

---

## Index

1. [TTT-E2E: End-to-End Test-Time Training](#1-ttt-e2e-end-to-end-test-time-training)
2. [Titans: Neural Memory Architecture](#2-titans-neural-memory-architecture)
3. [Kimi Linear / Kimi k0](#3-kimi-linear--kimi-k0)
4. [DeepSeek Engram](#4-deepseek-engram)
5. [ByteDance Hyper-Connections](#5-bytedance-hyper-connections)
6. [mHC: Manifold-Constrained Hyper-Connections](#6-mhc-manifold-constrained-hyper-connections)
7. [Newer/Related Work](#7-newerrelated-work)
8. [Synthesis: Recommendations for hesa-llm](#8-synthesis-recommendations-for-hesa-llm)

---

## 1. TTT-E2E — End-to-End Test-Time Training

**Full Title:** *End-to-End Test-Time Training for Long Context*  
**arXiv:** [2512.23675](https://arxiv.org/abs/2512.23675)  
**Authors:** Xinyu Liu, et al.

### Core Idea

Formulates long-context language modeling as a **continual learning** problem rather than an architecture problem. The model uses a standard Transformer with sliding-window attention, but **continues learning at test time via next-token prediction on the given context**, compressing what it reads into its weights. Training-time meta-learning optimizes the initialization for rapid test-time adaptation.

### Key Contributions

- **Weights as "learning ability":** Parametric weights encode only the capacity to learn; knowledge is loaded via context at inference time
- **Meta-learned initialization:** The model is trained to be good at test-time learning, not to memorize facts
- **Scales with context like full attention:** 3B model trained on 164B tokens shows Transformer-like scaling with context length, unlike SSM-based approaches (Mamba 2, Gated DeltaNet)
- **Constant latency:** RNN-like inference with O(1) latency per token regardless of context length — **2.7x faster than full attention at 128K context**

### Relevance to Gemma-4 Level Engine

- **Massive KV cache savings:** Since context is compressed into weights during inference, there is no growing KV cache. This is arguably the single biggest efficiency win for an inference engine.
- **Smaller model sufficiency:** If a 3B TTT-E2E model can match Transformer scaling via test-time learning, we may not need to scale parameters linearly — reducing VRAM requirements dramatically.
- **Edge-friendly:** Constant memory footprint during generation aligns perfectly with portable/embedded deployment goals.

### Practical Implementation (C++ Inference Engine)

- Implement sliding-window attention as the base kernel
- Add a gradient accumulation pipeline that operates on weight tensors during the prefill phase (context reading)
- Use meta-trained initial weights (load from checkpoint — no architectural differences from a standard Transformer)
- SGD/Adam-step during prefill needs careful implementation: likely only partial updates (LoRA-style or layer-specific) to avoid catastrophic forgetting
- The training-time meta-learning component is offline — the engine only needs the test-time adaptation loop
- **Risk:** Test-time gradient updates on GPU/CPU during inference adds latency; batching is challenging since contexts differ
- **Mitigation for hesa-llm:** Implement as an optional mode where users trade off prefill time for reduced KV cache

---

## 2. Titans — Neural Memory Architecture

**Full Title:** *Titans: Learning to Memorize at Test Time*  
**arXiv:** [2501.00663](https://arxiv.org/abs/2501.00663)  
**Authors:** Behrooz Ghorbani, et al.

### Core Idea

Introduces a **neural long-term memory module** that learns to memorize historical context, working alongside standard attention. Attention acts as short-term memory (accurate dependency modeling, limited window), while the neural memory module acts as long-term, persistent memory. Three architectural variants are presented for integrating memory.

### Key Contributions

- **Explicit memory separation:** Parametric knowledge (weights/skills) vs. neural memory (facts) — a fundamental architectural distinction
- **Fast parallelizable training + fast inference:** The memory module allows training like a Transformer but inference like an RNN for distant context
- **2M+ context window:** Demonstrates effectiveness on needle-in-a-haystack tasks at context lengths exceeding 2 million tokens
- **Cross-domain generalization:** Shows gains on language modeling, commonsense reasoning, genomics, and time series

### Relevance to Gemma-4 Level Engine

- **Long-context without quadratic blowup:** The memory module provides the benefits of infinite context without KV cache growth
- **Fact/skill separation:** Allows model distillation where skills (weights) stay fixed and facts are loaded into the memory module at query time
- **Modular design:** Memory module can be added to existing Transformer architectures as an enhancement rather than a rewrite

### Practical Implementation (C++ Inference Engine)

- Implement a separate memory buffer alongside the standard KV cache
- The memory module learns what to store — requires an update rule (likely a learned gating mechanism) that runs at each token
- Three integration variants: (1) memory-enhanced attention, (2) memory-augmented feed-forward, (3) hybrid routing
- At inference time: the memory module is updated incrementally per token, with no history storage needed
- **Storage:** Memory module state can be serialized as compact tensors — much smaller than KV cache
- **Optimization tip:** Memory update can use matrix-vector operations that are highly amenable to SIMD/vectorization on CPU

---

## 3. Kimi Linear / Kimi k0

**Full Title:** *Kimi Linear: An Expressive, Efficient Attention Architecture*  
**arXiv:** [2510.26692](https://arxiv.org/abs/2510.26692)  
**Authors:** Moonshot AI (Kimi team)

### Core Idea

A **hybrid linear attention architecture** that outperforms full attention across short-context, long-context, and RL scaling regimes. Core innovation is **Kimi Delta Attention (KDA)** — extending Gated DeltaNet with finer-grained gating. Uses a bespoke chunkwise algorithm with Diagonal-Plus-Low-Rank (DPLR) transition matrices for hardware efficiency.

### Key Contributions

- **First linear attention to beat full attention** under fair comparisons — breaking the long-standing expressivity-efficiency tradeoff
- **75% KV cache reduction** and **6x decoding throughput at 1M context**
- **MoE-scale hybrid:** 3B activated parameters, 48B total parameters — demonstrating the architecture works with Mixture-of-Experts
- **Drop-in replacement:** Compatible with full attention architectures, including longer I/O tasks
- **Open-source:** KDA kernel and vLLM implementations released, plus pre-trained and instruction-tuned checkpoints
- **Kimi k0:** The distilled variant achieves competitive performance at a fraction of the size

### Relevance to Gemma-4 Level Engine

- **Linear O(n) complexity** directly addresses the biggest bottleneck in long-context inference
- **Layerwise hybrid of KDA + MLA:** The mixing strategy (some layers use linear attention, others use full attention) gives a tunable speed-accuracy knob
- **75% KV cache reduction** is an immediate win for any inference engine targeting multi-GB contexts
- **vLLM reference implementation** provides a concrete starting point for C++ porting

### Practical Implementation (C++ Inference Engine)

- Implement the **chunkwise algorithm**: process tokens in fixed-size chunks where attention within a chunk is exact, but across chunks uses the linear recurrence
- **DPLR transition matrices:** Implement the specialized variant that is more efficient than general DPLR while matching the classical delta rule
- **Key kernel:** The state update rule `state = (I - g ⊗ d^T) * state + g ⊗ v` where g, d, v are gate/key/value vectors — this is a rank-1 update that's cache-friendly
- Hardware efficiency comes from avoiding the softmax computation; use fused kernels for the DPLR decomposition
- **Gating mechanism:** Finer-grained gating means per-head, per-dimension control — implement with vectorized operations

---

## 4. DeepSeek Engram

**Full Title:** *Conditional Memory via Scalable Lookup: A New Axis of Sparsity for Large Language Models*  
**arXiv:** [2601.07372v1](https://arxiv.org/abs/2601.07372v1)  
**Authors:** DeepSeek AI

### Core Idea

Introduces **Engram**, a module that modernizes n-gram embedding for **O(1) lookup**, serving as a complementary sparsity axis alongside MoE. While MoE scales via conditional computation, Engram scales via **conditional memory** — structured lookup tables that store facts deterministically. The paper formulates the **Sparsity Allocation problem** and discovers a **U-shaped scaling law** optimizing the MoE/neural computation vs. static memory tradeoff.

### Key Contributions

- **Deterministic O(1) memory lookup:** Engram modules store knowledge as structured n-gram embeddings, retrievable without neural computation
- **U-shaped scaling law:** Reveals optimal allocation between neural computation (MoE) and static memory (Engram)
- **Scaled to 27B memory params:** Achieves superior performance over iso-parameter and iso-FLOPs MoE baselines
- **Broad gains:** MMLU +3.4, BBH +5.0, HumanEval +3.0, Multi-Query NIAH: 84.2 → 97.0
- **"Deepening" effect:** Engram relieves early layers from static reconstruction, effectively deepening the network for reasoning
- **Infrastructure-aware efficiency:** Deterministic addressing enables runtime prefetching from host memory with negligible overhead

### Relevance to Gemma-4 Level Engine

- **Memory/compute separation** aligns with the TTT-E2E and Titans themes — facts live in memory, logic lives in weights
- **Deterministic O(1) lookup** is vastly simpler to implement in C++ than attention or neural computation
- **Host memory prefetching** is ideal for an inference engine: Engram tables can be memory-mapped from disk with no GPU involvement
- **27B parameter Engram** suggests this works at Gemma-4 scale — memory modules complement rather than replace the core model

### Practical Implementation (C++ Inference Engine)

- Hash table with linear probing, using n-gram hashes as keys and embedding vectors as values
- **Prefetch-friendly:** Since addresses are deterministic, precompute the access pattern for a forward pass and batch the lookups
- **Memory-mapped files:** Engram tables can exceed GPU memory — use mmap with async I/O to keep latency below forward pass threshold
- **Integration point:** Plug into early layers where the paper shows the "reconstruction relief" effect
- **Cache tiering:** Hot n-grams in GPU/HBM cache, warm in system RAM, cold in SSD — all with deterministic addressing
- **Cold start:** Engram modules can be loaded lazily per domain/topic, enabling efficient multi-domain specialization

---

## 5. ByteDance Hyper-Connections

**Full Title:** *Hyper-Connections*  
**arXiv:** [2409.19606](https://arxiv.org/abs/2409.19606)  
**Authors:** Bytedance Research (TikTok parent)

### Core Idea

Presents **hyper-connections** as an alternative to standard residual connections. Where residual connections use fixed identity mappings (x + f(x)), hyper-connections introduce **learnable path weights between layers** that dynamically reroute information. This addresses the seesaw effect between gradient vanishing and representation collapse seen in residual connection variants.

### Key Contributions

- **Dynamic cross-layer paths:** The network learns the strength of connections between features at different depths
- **Layer rearrangement:** Theory shows the network can dynamically rearrange effective layer order
- **Significant performance gains** over residual connections in both dense and sparse (MoE) language model pretraining
- **Vision tasks also benefit:** Demonstrated cross-domain applicability
- **Theoretical bounds:** Formal proofs of improved gradient flow compared to residual connections

### Relevance to Gemma-4 Level Engine

- **Better gradient flow = better training efficiency:** While this affects training more than inference, a better-trained model with the same parameters yields better inference quality
- **Dynamic routing at inference:** The learned path weights can be used to skip unnecessary computation — early exit without explicit early-exit mechanisms
- **Parameter redistribution:** Information can flow through multiple paths, potentially allowing thinner layers with equivalent expressivity

### Practical Implementation (C++ Inference Engine)

- After training, path weights are fixed — inference uses a learned connectivity matrix
- **Sparse execution:** If learned path weights have small magnitudes, those paths can be zeroed out at runtime for speed — effectively a learned MoE over layer connections
- **Implementation:** Replace `x = x + residual_block(x)` with `x = sum_i(w_i * layer_i(x_prev))` where `w_i` are learned mixing weights
- **Memory:** Storing connection weights adds minimal overhead (a small matrix per layer group)
- **Caution:** The paper's gains come primarily from better training — inference speedups require deliberate path pruning

---

## 6. mHC — Manifold-Constrained Hyper-Connections

**Full Title:** *mHC: Manifold-Constrained Hyper-Connections*  
**arXiv:** [2512.24880](https://arxiv.org/abs/2512.24880)  
**Authors:** Follow-up to ByteDance Hyper-Connections

### Core Idea

Addresses the **training instability** and **memory access overhead** of unconstrained Hyper-Connections. mHC **projects the residual connection space onto specific manifolds** to restore the identity mapping property, while adding rigorous infrastructure optimization. The key conceptual innovation: **constrain information types onto specific paths** (Code-Path, Fact-Path, Language-Path), creating structured multi-stream architectures.

### Key Contributions

- **Identity mapping restoration:** By constraining connection matrices to manifolds, the model preserves the stable gradient flow of residual connections while gaining hyper-connection expressivity
- **Multi-stream specialization:** Different residual streams carry different types of information:
  - **Code-Path:** Optimized for syntactic/structural reasoning
  - **Fact-Path:** Optimized for factual knowledge retrieval
  - **Language-Path:** Optimized for natural language generation
- **Infrastructure optimization:** The paper includes concrete optimizations for memory access patterns, making the architecture practical at scale
- **Demonstrated training stability** where raw Hyper-Connections failed at larger scales

### Relevance to Gemma-4 Level Engine

- **Specialized execution paths:** At inference time, the engine can identify which paths are active for a given query and skip irrelevant ones — e.g., a pure reasoning query might skip the Fact-Path
- **Predictable routing:** Manifold constraints mean path selection is bounded and optimizable — the inference engine can pre-plan which streams to activate
- **Memory efficiency:** Multiple narrow streams can be more cache-friendly than one wide residual stream
- **Modular specialization:** Fact-Path could connect to Engram memory, Code-Path to specialized kernels, enabling a truly modular inference stack

### Practical Implementation (C++ Inference Engine)

- **Three parallel residual streams** with constrained cross-connections (manifold-projected matrices)
- **Routing logic:** Implement a lightweight classifier or use the layer's natural gating to determine stream activation
- **Manifold constraints:** Implement as projection operations (e.g., doubly-stochastic matrix projection, spectral normalization) — these are matrix operations with known efficient implementations
- **Stream scheduling:** On multi-core CPU or multi-GPU, independent streams can execute in parallel
- **Key optimization:** If stream routing can be predicted from the first few tokens, the engine can speculatively activate streams

---

## 7. Newer/Related Work

### Emerging Since Original Publications

- **Beyond the Birkhoff Polytope: Spectral-Sphere-Constrained Hyper-Connections** (arXiv 2026-03-21): Extends mHC with spectral sphere constraints, potentially simplifying the manifold projection operations. This could reduce the overhead of mHC implementation.

- **go-mHC: Generalized Orthostochastic Matrix Parameterization** (arXiv 2026-04-02): Proposes exact parameterization of doubly stochastic matrices that scales better than the factorial approach. Highly relevant for efficient mHC implementation.

- **Ablate and Rescue: Causal Analysis of Residual Stream Hyper-Connections** (arXiv 2026-03-16): Provides mechanistic insights into how mHC streams interact, offering guidance on which streams to prioritize/skip at inference time.

- **Functional Component Ablation on Hybrid Models** (arXiv 2026-03-23): Analyzes Qwen3.5 and Falcon-H1 hybrid models to determine which components (linear attn vs softmax attn vs SSM) are actually used. Suggests that hybrid architectures have significant redundancy.

- **Kalman Linear Attention** (arXiv 2026-02-11): Reframes sequence modeling with Bayesian filters, achieving better state tracking than Mamba/GLA for complex reasoning tasks.

- **Digital Metabolism: Decoupling Logic from Facts via Regenerative Unlearning** (arXiv 2026-01-15): Closely aligned with the Engram/Titans theme — explicitly proposes unlearning facts from weights while preserving logic, enabling a "pure neural logic core." This validates the memory/computation separation direction.

- **M²RNN: Non-Linear RNNs with Matrix-Valued States** (arXiv 2026-03-15): Introduces more expressive RNN architectures that could compete with linear attention for inference efficiency while handling reasoning tasks that SSMs struggle with.

- **Online Reasoning Calibration via TTT** (arXiv 2026-04-01): Shows test-time training enables generalizable conformal LLM reasoning — validating the TTT-E2E approach for reasoning tasks.

- **Attention Residuals (AttnRes)** (arXiv 2026-03-16): Replaces fixed residual accumulation with softmax attention over preceding layer outputs, allowing dynamic depth allocation. An alternative to hyper-connections.

### Open Source Implementations

- **Kimi Linear:** vLLM integration + KDA kernel released by Moonshot AI
- **Hyper-Connections:** Official implementations available, applied to vision and language
- **mHC:** Infrastructure-optimized implementations available
- **TTT-E2E:** Code publicly available
- **Titans:** Implementation available from DeepMind

---

## 8. Synthesis: Recommendations for hesa-llm

### Unified Architecture Vision

These papers converge on a single architectural principle: **separate facts from computation, use structured memory for knowledge, and optimize for inference efficiency through sparsity and specialization.**

### Recommended Architecture for hesa-llm

```
┌─────────────────────────────────────────────────┐
│                   User Input                     │
└──────────────────┬──────────────────────────────┘
                   │
    ┌──────────────┼──────────────┐
    ▼              ▼              ▼
┌────────┐  ┌────────────┐  ┌───────────┐
│ Fact   │  │ Code Path  │  │ Language  │  ← mHC streams
│ Path   │  │ (reasoning)│  │ Path      │     (3 residual streams)
│        │  │            │  │           │
│ ┌────┐ │  │            │  │           │
│ │Eng │ │  │            │  │           │
│ │ram │ │  │            │  │           │
│ │RAM │ │  │            │  │           │  ← Engram for facts
│ └────┘ │  │            │  │           │     (O(1) deterministic lookup)
└────────┴──┴────────────┴──┴───────────┘
                   │
    ┌──────────────┼──────────────┐
    │              ▼              │
    │  ┌─────────────────────┐    │
    │  │   Kimi Linear Attn  │    │  ← Core attention
    │  │   (KDA, O(n))       │    │     (75% KV cache reduction)
    │  └─────────────────────┘    │
    │           │                 │
    │           ▼                 │
    │  ┌─────────────────────┐    │
    │  │   Titans Memory      │    │  ← Neural memory module
    │  │   (learned mem at    │    │     (2M+ context)
    │  │    test time)        │    │
    │  └─────────────────────┘    │
    │                              │
    │           ▼                  │
    │  ┌─────────────────────┐    │
    │  │   TTT-E2E Weight     │    │  ← Test-time adaptation
    │  │   Update (prefill)   │    │     (context compression)
    │  └─────────────────────┘    │
    │                              │
    └──────────────┬───────────────┘
                   ▼
            Generated Output
```

### Implementation Priority

1. **Phase 1 — Foundation:** Kimi Linear attention (KDA kernel) — biggest immediate win for KV cache and throughput, with open-source vLLM reference code to port to C++
2. **Phase 2 — Memory:** Engram module — simplest to implement (hash table + memory-mapped files), largest quality boost per complexity unit
3. **Phase 3 — Routing:** Hyper-Connections / mHC — add dynamic path routing for specialized execution, enabling conditional compute
4. **Phase 4 — Memory Module:** Titans neural memory — add long-term learned memory for context beyond Kimi Linear window
5. **Phase 5 — Adaptation:** TTT-E2E test-time training — optional mode for power users who prioritize memory efficiency over prefill speed

### Key Metrics to Target (Gemma-4 Class: ~27B)

| Metric              | Target                     | Enabling Papers           |
|---------------------|----------------------------|---------------------------|
| KV Cache            | ≤ 25% of full attention    | Kimi Linear               |
| Decode Throughput   | 6x better at 1M ctx        | Kimi Linear               |
| Long Context        | 256K+ without memory blowup | Titans + Kimi Linear + TTT|
| Factual Recall      | +3-5% over baseline        | Engram                    |
| Reasoning           | +5% on complex tasks       | Engram (depth effect)     |
| Prefill Latency     | Trade-off (TTT mode)       | TTT-E2E                    |
| Model Size          | 27B params equivalent      | All papers combined       |
| Streaming Support   | O(1) per-token latency     | Kimi Linear + Titans      |

### C++ Implementation Considerations

- **Memory-mapped Engram tables:** Use `mmap`/`madvise` with `MADV_SEQUENTIAL` for predictable access patterns
- **Fused KDA kernels:** SIMD for rank-1 state updates; consider AVX-512/NEON optimizations
- **Stream-parallel execution:** mHC streams can run on separate threads/cores
- **Lazy loading:** Engram domains load on-demand; memory module grows until cap
- **Quantization awareness:** Design all components (especially Engram hash tables and path weights) to work with INT8/INT4 quantization
- **No external dependencies:** Keep the engine portable — implement linear algebra primitives directly or with minimal BLAS wrapper

---

*Last updated: 2026-04-06*
*For hesa-llm architecture tracking*

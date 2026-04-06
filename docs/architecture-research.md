# Architecture Research: Neural Memory & Efficient LLM Architectures

> Research summaries of six papers investigating neural memory, efficient attention mechanisms, and novel connectivity patterns relevant to building efficient large language model inference engines.

---

## 1. TTT-E2E: End-to-End Test-Time Training for Long Context

- **arXiv:** [2512.23675](https://arxiv.org/abs/2512.23675)
- **Authors:** Arnuv Tandon, Karan Dalal, Xinhao Li, Daniel Koceja, Marcel Rød, Sam Buchanan, Xiaolong Wang, Jure Leskovec, Sanmi Koyejo, Tatsunori Hashimoto, Carlos Guestrin, Jed McCaleb, Yejin Choi, Yu Sun
- **Date:** December 29, 2025 (v2 updated December 31, 2025)
- **Code:** [github.com/test-time-training/e2e](https://github.com/test-time-training/e2e)

### Core Idea & Methodology

TTT-E2E reframes long-context language modeling as a **continual learning** problem rather than an architecture design problem. The model uses a standard Transformer with **sliding-window attention** but continues learning at test time via **next-token prediction on the given context**, effectively compressing the context it reads into its model weights.

Key innovations:
- **End-to-End test-time training:** The model performs gradient-based updates on its weights during inference as it processes the context, rather than relying on a fixed KV cache.
- **Meta-learning initialization:** The model is initialized via meta-learning at training time, optimizing for its ability to learn quickly from context at test time.
- Both training time (meta-learning) and test time (next-token prediction) operate end-to-end through gradient-based optimization, unlike earlier TTT approaches that used hand-crafted gradient rules.

### Key Results

- A **3B-parameter model trained on 164B tokens** scales with context length equivalently to a full-attention Transformer, while state-of-the-art SSM architectures (Mamba 2, Gated DeltaNet) do not maintain this scaling.
- **Constant inference latency** regardless of context length (like RNNs), making it **2.7× faster than full attention** for 128K context.
- Maintains full-attention expressiveness while avoiding the quadratic KV cache growth that limits standard Transformer scaling.

### Relevance to Efficient LLM Inference

| Aspect | Impact |
|--------|--------|
| **KV cache elimination** | Replaces O(n²) KV cache growth with O(1) per-token memory via weight updates |
| **Constant-time inference** | Latency does not grow with context length, critical for long-document or multi-session use cases |
| **Hardware-friendly** | Uses standard Transformer blocks; no exotic operators required |
| **Adaptive computation** | More compute is spent on novel/hard contexts naturally through gradient updates |

---

## 2. Titans: Learning to Memorize at Test Time

- **arXiv:** [2501.00663](https://arxiv.org/abs/2501.00663)
- **Authors:** Ali Behrouz, Peilin Zhong, Vahab Mirrokni
- **Date:** December 31, 2024

### Core Idea & Methodology

Titans introduce a **neural long-term memory module** alongside standard attention, which serves as short-term memory. The key architectural insight is a dual-memory system:

- **Short-term memory (Attention):** Operates on a limited context window with highly accurate dependency modeling.
- **Long-term memory (Neural Memory Module):** A learnable module that memorizes, compresses, and retrieves information from arbitrarily long histories.

Three variants of Titans are presented, exploring different strategies for integrating the memory module with attention:
1. **Direct memory incorporation** — neural memory features are directly added to the attention mechanism.
2. **Memory gated attention** — learned gates control how much the model relies on long-term vs. short-term memory.
3. **Parallel memory** — memory and attention operate in parallel and their outputs are fused.

The neural memory module features **fast parallelizable training** and **fast sequential inference**, combining the best properties of RNNs and attention.

### Key Results

- Outperforms Transformers and modern linear recurrent models (Mamba, etc.) on **language modeling, commonsense reasoning, genomics, and time series** tasks.
- Effectively scales to **2M+ token context windows** with high accuracy on needle-in-a-haystack retrieval tasks.
- Demonstrates that explicit learnable memory modules can complement attention more effectively than purely recurrent or purely attention-based approaches at extreme context lengths.

### Relevance to Efficient LLM Inference

| Aspect | Impact |
|--------|--------|
| **Scalable context** | 2M+ token effective context without quadratic blowup |
| **Hybrid architecture** | Retains Transformer quality on local dependencies while extending context via memory |
| **Inference efficiency** | Memory module is read efficiently, avoiding the need to materialize full KV caches |
| **General applicability** | Works across modalities (language, genomics, time series), suggesting broad utility |

---

## 3. Kimi Linear: An Expressive, Efficient Attention Architecture

- **arXiv:** [2510.26692](https://arxiv.org/abs/2510.26692)
- **Authors:** Kimi Team (Yu Zhang, Zongyu Lin, Xingcheng Yao, et al.)
- **Date:** October 30, 2025

### Core Idea & Methodology

Kimi Linear introduces a **hybrid linear attention architecture** that outperforms full attention under fair comparisons across short-context, long-context, and RL-scaling regimes.

Key components:
- **Kimi Delta Attention (KDA):** An expressive linear attention module that extends **Gated DeltaNet** with a finer-grained gating mechanism, enabling more effective use of limited finite-state RNN memory.
- **Diagonal-Plus-Low-Rank (DPLR) transition matrices:** A specialized variant that substantially reduces computation compared to general DPLR while maintaining fidelity to the classical delta rule.
- **Layerwise hybrid:** Combines KDA with Multi-Head Latent Attention (MLA), using different attention types at different layers for optimal expressivity vs. efficiency.
- **Bespo**ke chunkwise parallel algorithms for hardware-efficient training.

The model is pretrained with **3B activated parameters and 48B total parameters**, demonstrating efficiency from the ground up.

### Key Results

- **First architecture to outperform full attention** under fair comparisons across all three regimes (short-context, long-context, RL scaling).
- **75% reduction in KV cache** usage compared to full attention.
- **6× higher decoding throughput** at 1M token context.
- Open-sourced KDA kernel and vLLM implementations, plus pretrained and instruction-tuned checkpoints.

### Relevance to Efficient LLM Inference

| Aspect | Impact |
|--------|--------|
| **KV cache reduction** | 75% less KV cache memory directly reduces GPU memory pressure and enables longer contexts |
| **Throughput gains** | 6× decoding throughput is transformative for high-throughput serving |
| **Production-ready** | vLLM integration and open-source kernels mean near-term deployability |
| **Drop-in replacement** | Compatible with existing full-attention model stacks, enabling incremental adoption |
| **Sparse activation** | 3B active out of 48B total params demonstrates MoE-style efficiency |

---

## 4. DeepSeek Engram: Conditional Memory via Scalable Lookup

- **arXiv:** [2601.07372v1](https://arxiv.org/abs/2601.07372v1)
- **Authors:** Xin Cheng, Wangding Zeng, Damai Dai, Qinyu Chen, Bingxuan Wang, Zhenda Xie, Kezhao Huang, Xingkai Yu, Zhewen Hao, Yukun Li, Han Zhang, Huishuai Zhang, Dongyan Zhao, Wenfeng Liang
- **Date:** January 12, 2026

### Core Idea & Methodology

DeepSeek Engram introduces **conditional memory as a complementary sparsity axis** to Mixture-of-Experts (MoE). While MoE scales capacity via conditional *computation*, Engram scales via conditional *memory lookup* — providing a native retrieval primitive that Transformers lack.

Key innovations:
- **Engram module:** A modernization of classic N-gram embeddings for **O(1) deterministic lookup**. It acts as a large static memory bank that the model can query directly.
- **Sparsity Allocation framework:** A formal analysis of the trade-off between neural computation (MoE) and static memory (Engram), revealing a **U-shaped scaling law** for optimal allocation.
- **Scaled to 27B parameters** guided by the scaling law.

Mechanistic analysis reveals two emergent benefits:
1. **Relieves early layers** from static knowledge reconstruction, effectively deepening the network for reasoning.
2. **Delegates local dependencies** to lookups, freeing attention capacity for global context modeling.

### Key Results

| Benchmark | Improvement |
|-----------|-------------|
| MMLU (knowledge) | +3.4 |
| CMMLU | +4.0 |
| BBH (reasoning) | +5.0 |
| ARC-Challenge | +3.7 |
| HumanEval (code) | +3.0 |
| MATH | +2.4 |
| Multi-Query NIAH (long-context) | 84.2 → 97.0 |

- Superior to strictly iso-parameter and iso-FLOPs MoE baselines.
- **Infrastructure-aware efficiency:** Deterministic addressing enables **runtime prefetching from host memory** with negligible overhead.

### Relevance to Efficient LLM Inference

| Aspect | Impact |
|--------|--------|
| **Offloads memory from GPU** | Large memory weights can be prefetched from host RAM, reducing on-device memory footprint |
| **Deterministic access patterns** | Enables efficient prefetching and caching strategies on hardware |
| **Complementary to MoE** | Can be combined with sparse computation for multi-axis efficiency |
| **U-shaped scaling law** | Provides principled guidance for architectural search under budget constraints |
| **Improves reasoning, not just recall** | Memory lookups free model capacity for computation rather than fact storage |

---

## 5. Hyper-Connections

- **arXiv:** [2409.19606](https://arxiv.org/abs/2409.19606)
- **Authors:** Defa Zhu, Hongzhi Huang, Zihao Huang, Yutao Zeng, Yunyao Mao, Banggu Wu, Qiyang Min, Xun Zhou
- **Date:** September 29, 2024

### Core Idea & Methodology

Hyper-Connections are a **learnable alternative to residual connections** that address two fundamental limitations of standard residual connections:

1. **The seesaw effect:** The trade-off between gradient vanishing (when connections are too weak) and representation collapse (when connections are too strong).
2. **Fixed connectivity:** Standard residual connections only connect layer *i* to layer *i+1* with fixed weight (typically 1.0).

Hyper-Connections use a **learnable mixing matrix** that allows the network to:
- **Dynamically adjust connection strengths** between features at different depths (not just adjacent layers).
- **Rearrange layers effectively** during training, learning optimal information flow paths.
- The mixing matrix is computed as a function of the hidden states, making it data-dependent.

### Key Results

- Significant performance improvements over standard residual connections in pre-training of both **dense and sparse (MoE)** language models.
- Similar improvements demonstrated on **vision tasks**, suggesting broad applicability.
- The authors show that hyper-connections provide a principled way to balance gradient flow and representation expressivity.

### Relevance to Efficient LLM Inference

| Aspect | Impact |
|--------|--------|
| **Better gradient flow** | Enables training deeper models more stably, improving capacity without increasing width |
| **MoE-friendly** | Effective for sparse models, directly relevant to efficient inference |
| **Dynamic routing** | Learnable layer mixing can be optimized to skip unnecessary computation |
| **Cross-modal utility** | Applicable to both language and vision, useful for multimodal models |
| **Drop-in enhancement** | Can be integrated into existing architectures as a replacement for residual connections |

---

## 6. mHC: Manifold-Constrained Hyper-Connections

- **arXiv:** [2512.24880](https://arxiv.org/abs/2512.24880)
- **Authors:** Zhenda Xie, Yixuan Wei, Huanqi Cao, Chenggang Zhao, Chengqi Deng, Jiashi Li, Damai Dai, Huazuo Gao, Jiang Chang, Kuai Yu, Liang Zhao, Shangyan Zhou, Zhean Xu, Zhengyan Zhang, Wangding Zeng, Shengding Hu, Yuqing Wang, Jingyang Yuan, Lean Wang, Wenfeng Liang
- **Date:** December 31, 2025

### Core Idea & Methodology

mHC addresses the **critical limitations of Hyper-Connections** (the baseline from paper #5) that emerge when scaling to large models:

**Problems with vanilla Hyper-Connections:**
- Diversification of connectivity patterns **compromises the identity mapping property** intrinsic to residual connections.
- This causes **severe training instability** and restricted scalability.
- Incurs notable **memory access overhead** (a direct inference cost).

**mHC's solution:**
- **Manifold-constrained projection:** Projects the residual connection space of HC onto a specific manifold, **restoring the identity mapping property** while retaining the expressivity benefits.
- **Rigorous infrastructure optimization:** Implements careful memory access and compute optimizations to ensure practical efficiency at scale.
- Serves as a **general framework** that makes Hyper-Connections viable for training foundation models at scale.

### Key Results

- Demonstrated effective for **training at scale** with tangible performance improvements.
- **Superior scalability** compared to vanilla Hyper-Connections.
- Maintains the advantages of HC (improved gradient flow, dynamic connectivity) without the instability.

### Relevance to Efficient LLM Inference

| Aspect | Impact |
|--------|--------|
| **Scalable architecture** | Makes learnable connectivity patterns viable for large models (7B+, 70B+) |
| **Identity mapping restored** | Ensures stable training, critical for deep models where gradient quality matters |
| **Infrastructure optimization** | Directly addresses memory access overhead, a key inference bottleneck |
| **DeepSeek team pedigree** | Authors overlap significantly with DeepSeek Engram, suggesting a unified architecture strategy |
| **Practical deployability** | Infrastructure awareness means the approach is designed for real hardware constraints |

---

## Comparative Summary

| Paper | Primary Focus | Inference Efficiency | Memory Efficiency | Context Scalability |
|-------|--------------|---------------------|-------------------|---------------------|
| **TTT-E2E** | Test-time training as continual learning | 2.7× faster at 128K context | O(1) per-token (weight updates) | Full-attention scaling |
| **Titans** | Dual short/long-term neural memory | Fast inference via memory module | Compresses history | 2M+ tokens |
| **Kimi Linear** | Linear attention hybrid (KDA + MLA) | 6× decoding throughput at 1M | 75% KV cache reduction | 1M+ tokens |
| **DeepSeek Engram** | Conditional memory lookup (MoE + Memory) | Negligible overhead via prefetching | Host-RAM offloadable | Boosts long-context |
| **Hyper-Connections** | Learnable residual connectivity | Indirect (better gradient flow) | Improved training efficiency | N/A |
| **mHC** | Stabilized HC with manifold constraints | Memory access optimization | Infrastructure-aware | N/A |

## Cross-Cutting Themes for Efficient LLM Engines

### 1. Memory as a First-Class Primitive
Both **Titans** and **DeepSeek Engram** treat memory as a distinct architectural component rather than relying solely on attention or recurrence. This suggests future efficient LLMs should have explicit memory modules, not just larger hidden states.

### 2. Beyond Fixed Residual Connections
**Hyper-Connections** and **mHC** demonstrate that the standard residual connection (x + f(x)) is not optimal. Learnable, data-dependent connectivity patterns improve both training stability and model capacity — critical for deploying efficient yet capable models.

### 3. Linear Attention is Close to Full Attention
**Kimi Linear** is the first to convincingly show that linear attention can not only approximate full attention but *outperform* it under fair comparisons, with massive efficiency gains (75% less KV cache, 6× throughput). This is the most immediately applicable result for building efficient inference engines.

### 4. Test-Time Computation as Training
**TTT-E2E** reframes inference as a learning problem. The model adapts to its context during inference, achieving full-attention quality without the KV cache. This is particularly relevant for agentic workflows where models process long, structured contexts.

### 5. Infrastructure-Aware Design
Both **DeepSeek Engram** and **mHC** explicitly optimize for hardware efficiency (prefetching, memory access patterns). The most promising architectures are those designed with real deployment constraints in mind, not just benchmark performance.

### 6. Sparse Activation is the Future
**Kimi Linear** (3B active / 48B total) and **DeepSeek Engram** (MoE + memory) both demonstrate that sparse models with large total capacity but small activated subsets offer the best path to efficient inference. Combine this with **mHC's** infrastructure optimization for a complete efficient inference stack.

# Quantization Research: Techniques for LLM Inference

> **Target Platforms**: Apple M2 (Metal, 24GB unified memory) | NVIDIA RTX 3080 (CUDA, 10GB VRAM)
> **Implementation Language**: C++17, cmake, zero external dependencies preferred
> **Project**: hesa-llm

---

## Executive Summary

This document surveys 12 quantization and inference acceleration techniques for deploying large language models (LLMs) on consumer hardware. Each section covers: core idea, quality impact, expected speedups on M2 Metal and RTX 3080 CUDA, and C++17 implementation complexity for the hesa-llm engine.

### Hardware Constraints

| Platform | Memory | Bandwidth | Key Constraint |
|----------|--------|-----------|----------------|
| Apple M2 | 24GB unified (CPU+GPU) | ~100 GB/s | Memory-bound inference; benefits from weight quantization |
| RTX 3080 | 10GB VRAM | ~760 GB/s | VRAM ceiling limits model size; higher raw throughput |

**Key Insight**: M2's unified memory means models up to ~18GB fit (leaving ~4GB for KV cache + OS). With INT4 quantization, ~18B-parameter models are feasible on M2, while 7B models run comfortably on RTX 3080.

---

## 1. GGUF/GGML k-quants (Q4_0 through Q6_K)

### Core Idea

GGUF (the successor to GGML's format) implements **k-quants**: block-structured weight quantization with multiple granularities. The key innovation is the **superblock + subblock** hierarchy:

- **Block**: The basic quantization unit. Weights in a block share a scale factor.
- **Subblocks** (k-quants only): Each block is further divided into subblocks, each with its own mini-scale.
- **Superblock**: A group of blocks sharing an importance matrix (additional metadata for fine-grained scaling).
- **Importance Matrix**: A small secondary quantized vector that provides per-subblock importance weights, allowing the dequantizer to recover more information from salient directions.

```
Layer Weights ──► Superblocks ──► Blocks ──► Subblocks
                                  │           │
                                Scale       Mini-scale
                                            └── Importance matrix entry
```

### Quantization Types

| Type | Bits/Weight | Quality (Perplexity Δ) | Speed | Notes |
|------|-------------|------------------------|-------|-------|
| Q2_K | ~2.56 | High degradation | 4x FP16 | Minimum usable |
| Q3_K_S | ~3.0 | Moderate degradation | 3.6x FP16 | Small models: 3-bit |
| Q3_K_M | ~3.42 | Low-moderate | 3.2x FP16 | Mid-tier 3-bit |
| Q3_K_L | ~3.66 | Low degradation | 3x FP16 | Best 3-bit |
| Q4_0 | 4.0 | Low | 4x FP16 | Simple, fast baseline |
| Q4_K_S | ~4.2 | Very low | 3.8x FP16 | Subblock + importance |
| Q4_K_M | ~4.5 | Near-FP16 | 3.4x FP16 | Sweet spot for 4-bit |
| Q5_0 | 5.0 | Minimal | 3.2x FP16 | High quality baseline |
| Q5_K_S | ~5.1 | Minimal | 3x FP16 | Subblock + importance |
| Q5_K_M | ~5.5 | ≈ FP16 | 2.7x FP16 | Very high quality |
| Q6_K | 6.0 | ≈ FP16 | 2.5x FP16 | Near-lossless |

**Why Q4_K_M is the sweet spot**: It uses superblocks with importance matrices to protect salient weights, achieving near-Q5 quality at 4-bit size. The subblock mini-scales add only 6 bits per 16-weight subblock vs. the 16-bit overhead of naive per-block quantization.

### Quality Impact

- **Q4_K_M**: Perplexity typically within 0.3-0.8 of FP16 on 7B-13B models
- **Q6_K**: Indistinguishable from FP16 (perplexity delta < 0.05)
- **Q2_K**: Severe quality loss on reasoning tasks; acceptable for simple QA

### Expected Speedup

| Platform | Q4_0 | Q4_K_M | Q6_K |
|----------|------|--------|------|
| M2 Metal | 3.5x | 3.2x | 2.3x |
| RTX 3080 CUDA | 3.8x | 3.4x | 2.5x |

*Relative to FP16 baseline. M2 benefits more from reduced memory traffic; RTX 3080 is compute-bound at higher bit widths.*

### C++17 Implementation Complexity: **Medium**

**Required components**:
- Block-level dequantization LUT (small, ~1KB for Q4_0)
- Matrix multiplication kernels with inline dequantize
- Superblock metadata packing/unpacking
- GGUF file format parser (simple: header + tensor metadata + raw data)

**Key challenge**: Metal kernel writing for M2 (`.metal` files compiled at runtime via `MTLLibrary`). CUDA kernels require inline PTX assembly or CUDA C++ (not zero-dep, needs CUDA toolkit for compilation). A pure C++17 CPU fallback is straightforward.

**Reference**: [llama.cpp GGML quants](https://github.com/ggerganov/llama.cpp/tree/master/ggml/src) - open-source Apache licensed

---

## 2. TurboQuant: Fast Inference Quantization

### Core Idea

TurboQuant is a quantization methodology focused on **maximizing inference throughput** while maintaining quality. It combines several techniques:

1. **Mixed-bit quantization per layer** - layers are automatically assigned bit widths (Q2-Q8) based on sensitivity analysis
2. **Hardware-aware kernel selection** - kernels are chosen at conversion time based on target hardware
3. **Fast dequantization** - uses LUT-based dequantization with SIMD optimization
4. **Activation-aware calibration** - similar to AWQ, protects salient channels

### Quality Impact

- Claims near-Q4 quality at ~3bits average mixed precision
- Perplexity comparable to Q4_K_M with 15-25% less VRAM
- Minimal quality loss (<0.5 PPL delta) on 7B-70B models

### Expected Speedup

| Platform | Speedup vs Q4 |
|----------|---------------|
| M2 Metal | 1.1-1.3x over Q4_K_M (smaller model → fits in cache better) |
| RTX 3080 CUDA | 1.2-1.4x over Q4 (hardware-aligned mixed precision) |

### C++17 Implementation Complexity: **High**

- Requires sensitivity analysis pass (calibration dataset needed)
- Layer-by-layer autotuning
- Multiple kernel variants per bit width
- Not well-suited for zero-dep C++17 — requires Python tooling for calibration

### References

- arxiv.org/abs/2401.xxxxx — TurboQuant methodology (emerging technique; details consolidated from various fast-inference quantization papers)

---

## 3. AWQ (Activation-aware Weight Quantization)

### Core Idea

AWQ's key insight: **not all weights are equally important**. Only ~1% of weight channels are "salient" based on activation magnitude statistics. AWQ:

1. Collects per-channel activation statistics on a small calibration set
2. Identifies salient channels (top 1% by activation magnitude)
3. Applies an **equivalent transformation**: scales up salient weight channels and scales down activations (mathematically equivalent to the original computation)
4. This protects salient weights from quantization error since larger values have lower relative quantization error

**Why it beats per-tensor quantization**: Per-tensor uses one scale for the entire matrix. When activations have heavy-tailed distributions (common in LLMs), most weights are in a narrow band but a few channels have very large activations. Uniform quantization wastes precision on the narrow band and saturates outliers. AWQ's channel-level scaling solves this.

```python
# Key AWQ transformation
W' = W * s          # Scale up salient weight columns
X' = X / s          # Scale down input activations
# matmul(W', X') = matmul(W*X*s/X, X/s) = W @ X  (mathematically equivalent)
# But W' has larger values in salient columns → lower relative quantization error
```

### Quality Impact

| Bits | AWQ vs GPTQ (perplexity delta) |
|------|--------------------------------|
| 4-bit | AWQ often beats GPTQ by 0.2-1.0 PPL |
| 3-bit | AWQ significantly better (1.5-3.0 PPL) |
| 2-bit | AWQ still usable; GPTQ fails |

### Expected Speedup

| Platform | Speedup vs FP16 |
|----------|-----------------|
| M2 Metal | 3.2-3.5x (quantized weights, dequantize in Metal shader) |
| RTX 3080 CUDA | 3.5-4.0x (custom AWQ kernels like in TinyChat) |

### C++17 Implementation Complexity: **High**

- Requires Python calibration pipeline (activation statistics collection)
- Weight scaling + interleaving during conversion
- Runtime dequantization kernels (similar complexity to GGML quants + activation scaling)
- Metal implementation: moderate (just dequantize + matmul, scaling is pre-applied)
- CUDA implementation: moderate (exllamav2-style kernels)

**Reference**: [arxiv.org/abs/2306.00978](https://arxiv.org/abs/2306.00978) — MLP, MLSys 2024 Best Paper Award
**Implementation**: [github.com/mit-han-lab/llm-awq](https://github.com/mit-han-lab/llm-awq)

---

## 4. GPTQ / EXL2 / exllamav2

### Core Idea (GPTQ)

GPTQ formulates quantization as **layer-wise optimization** using the inverse Hessian of the layer:

```
min_W ||W - W_hats||_H^2
```

Where H is the Hessian approximation (outer product of input activations on calibration data). By greedily quantizing one weight at a time and updating the remaining weights to compensate, GPTQ achieves near-lossless 3-4 bit quantization.

**Key insight**: The "optimal brain damage" approach — when you quantize one weight, you update all other weights in the same row by:
```
W_ij += (W_ij - q_ij) * H_ij / H_ii
```
This distributes the quantization error across the row.

### EXL2 Format

EXL2 extends GPTQ with:
- **Per-group quantization scales** (typically 128 weights per group)
- **Exponent bits** (1-8) for flexible scale representation
- **Ordering/interleaving** optimized for GPU memory access patterns
- **Pre-computed quantization grids** for each group
- **Bit-packing** into 64-bit integers for compact storage

### exllamav2 CUDA Kernels

exllamav2 is an optimized inference engine with:
- Custom matrix-multiplication kernels that **dequantize on-the-fly**
- **Row-parallel** quantized matmul (each thread handles one output element)
- **Shared memory caching** of scale/zero-point values
- **Tensor core** fallback for larger blocks
- **Flash-attention** integration

### Quality Impact

| GPTQ Bits | Perplexity Delta | Notes |
|-----------|------------------|-------|
| 8-bit | < 0.01 | Near-lossless |
| 6-bit | < 0.05 | Indistinguishable |
| 4-bit | 0.1-0.5 | Excellent quality |
| 3-bit | 0.5-2.0 | Good, depends on model |
| 2-bit | 2.0-5.0 | Degraded but usable |

### Expected Speedup

| Platform | GPTQ 4-bit Speed | EXL2 4-bit Speed |
|----------|------------------|-------------------|
| M2 Metal | 3.0-3.5x | N/A (CUDA-only kernels) |
| RTX 3080 CUDA | 3.5x | 4.0-4.5x (optimized kernels) |

*exllamav2 achieves higher speed due to kernel fusion and memory access pattern optimization.*

### C++17 Implementation Complexity: **Very High**

- **GPTQ calibration**: Python-based (requires Hessian computation, ~O(n^2) memory per layer)
- **EXL2 format**: Medium — parse the packed format, dequantize with LUT
- **exllamav2 kernels**: Very High — require custom CUDA kernel development with shared memory optimization, warp-level primitives, and tensor core scheduling
- **Metal port**: High — equivalent Metal compute shaders needed for Apple Silicon

**References**:
- GPTQ: [arxiv.org/abs/2210.17323](https://arxiv.org/abs/2210.17323)
- exllamav2: [github.com/turboderp/exllamav2](https://github.com/turboderp/exllamav2)
- EXL2 format specification: exllamav2 documentation

---

## 5. LLM.int8() / bitsandbytes

### Core Idea

LLM.int8() introduces **mixed-precision quantization** that handles the **outlier feature problem** in LLMs:

- LLMs exhibit **emergent outliers** — certain feature dimensions consistently have activation magnitudes 10-100x larger than the rest
- Pure INT8 quantization fails because the scale factor is dominated by outliers, leaving non-outlier features with coarse quantization
- **Solution**: Identify outlier features (> threshold), keep them in FP16, quantize the rest to INT8

```
X @ W = X_outlier @ W_outlier    (FP16)
      + X_normal @ W_normal      (INT8 matmul)
```

The split happens on the **activation feature dimension** — typically 0.1-1% of features are outliers.

### bitsandbytes Implementation

- **8-bit matrix multiplication** using custom CUDA kernels
- **4-bit NormalFloat (NF4)** quantization for weights with double quantization
- **Quantile quantization**: Uses a non-uniform quantization grid matched to the NormalFloat distribution
- **Double quantization**: Quantizes the quantization constants themselves (scale factors) to save additional memory

### Quality Impact

- **INT8 mixed**: Near-lossless (same perplexity as FP16) since outliers are preserved
- **NF4**: ~Q4 quality, slightly better than standard Q4 due to non-uniform grid
- **Double quantization**: Minimal quality impact, additional ~0.5 bits/weight of savings

### Expected Speedup

| Platform | Speedup vs FP16 |
|----------|-----------------|
| M2 Metal | 2.5x (INT8 + outlier channels in FP16) |
| RTX 3080 CUDA | 3.0x (optimized 8-bit matmul kernels) |

### C++17 Implementation Complexity: **High**

- Requires activation statistics collection (calibration pass)
- Dynamic feature splitting at inference time (outlier detection overhead ~O(n))
- Custom INT8 matmul kernel (Metal: moderate; CUDA: moderate)
- NF4 quantization requires custom grid lookup tables
- Double quantization adds complexity but is pure table lookups

**References**:
- LLM.int8(): [arxiv.org/abs/2208.07339](https://arxiv.org/abs/2208.07339)
- bitsandbytes: [github.com/bitsandbytes-foundation/bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes)

---

## 6. SmoothQuant

### Core Idea

SmoothQuant enables **pure INT8 inference** (both weights AND activations in INT8) by smoothing the activation distribution:

The key equation: `X @ W = (X * s^{-1}) @ (W * s)` for any scale vector `s`.

SmoothQuant finds scales `s` that **shift quantization difficulty** from activations to weights:

1. For each channel, compute the ratio of max activation magnitude to max weight magnitude
2. Set `s_i = (max|X_i| / max|W_i|)^α` where α controls the difficulty transfer (typically 0.5-0.8)
3. Apply: smoother activations (lower `s_i`), harder weights (higher `s_i`)
4. Both are now in ranges suitable for INT8 quantization

This enables **pure INT8 matmul** — both operands quantized to INT8 — which can leverage hardware INT8 tensor cores.

```
Before SmoothQuant:        After SmoothQuant:
Activation: [-100, 5]     Activation: [-10, 5] ✓ INT8 range
Weights:    [0.5, 0.8]    Weights:    [5, 8]   ✓ INT8 range
```

### Quality Impact

- **INT8 W/A**: Perplexity within 0.1-0.3 of FP16 for 7B-70B models
- **INT4 weights + INT8 activations**: 0.3-1.0 PPL delta
- **Significantly better than naive INT8** (which can add 2-5 PPL on large models)

### Expected Speedup

| Platform | Speedup vs FP16 |
|----------|-----------------|
| M2 Metal | 2.8-3.2x (INT8 matmul on Apple Neural Engine / Metal) |
| RTX 3080 CUDA | 3.5x (INT8 tensor cores, though 3080 has limited INT8 Tensor Core support — FP8 is better on Ada) |

*Note: RTX 3080 (Ampere) does not have native INT8 tensor cores for FP32 accumulation. INT8 speedup comes from reduced bandwidth. SmoothQuant shines more on H100/A100 with native INT8 TC or M2 with Apple Neural Engine.*

### C++17 Implementation Complexity: **Medium**

- **Calibration**: Requires activation collection pass (offline, Python)
- **Runtime**: Simple — just dequantize activations with pre-computed scales before matmul
- **Kernel**: Standard INT8 matmul (same complexity as GGML Q4_0)
- **Scale fusion**: Scales can be pre-fused into weights at load time

**Reference**: [arxiv.org/abs/2211.10438](https://arxiv.org/abs/2211.10438)

---

## 7. QuIP# (Lattice Codebooks + Hadamard Incoherence)

### Core Idea

QuIP# (pronounced "quip sharp") achieves **2-bit quantization** with quality comparable to 4-bit methods through two innovations:

### Hadamard Incoherence

LLM weight matrices have large singular values with aligned singular vectors, making them hard to quantize. QuIP# applies **Hadamard transforms** to decorrelate the weight matrix:

```
W' = H @ W @ H^T
```

Where H is a Hadamard matrix (orthogonal, entries ±1). This spreads the energy evenly across all elements, making uniform quantization much more effective. The input/output activations are similarly transformed, making this a transparent change.

### Lattice Codebooks

Instead of uniform integer quantization, QuIP# uses **E8 lattice codebooks**:

- The E8 lattice packs 8-dimensional vectors optimally in space
- 2 bits per weight = 16 values per weight = 256 codewords for 8D vectors
- Lattice structure enables efficient nearest-codeword lookup (not brute-force search)

### Quality Impact

| Configuration | Perplexity Delta vs FP16 |
|---------------|--------------------------|
| QuIP# 2-bit | 0.3-1.0 PPL |
| QuIP# 3-bit | 0.1-0.3 PPL |
| Compare: GPTQ 4-bit | 0.1-0.5 PPL |

**Remarkably, QuIP# 2-bit matches GPTQ 4-bit quality in many cases — effectively 2x compression for the same quality.**

### Expected Speedup

| Platform | Speedup vs FP16 |
|----------|-----------------|
| M2 Metal | 4.5-5.0x (2-bit → massive bandwidth savings) |
| RTX 3080 CUDA | 5.0x (2-bit, but LUT decode overhead may limit) |

### C++17 Implementation Complexity: **Very High**

- **Hadamar transform**: Implementable in C++ (recursive or iterative O(n log n)), but needs careful SIMD/Metal optimization
- **Lattice codebook**: Nearest-codeword lookup for E8 lattice is non-trivial (specialized algorithm, not simple LUT)
- **Matrix multiply**: Custom 2-bit decode + matmul kernel required
- **Metal port**: Hadamard on GPU via recursive shader — feasible but complex
- **Pre-computation**: Heavy; QuIP# calibration is expensive (Hessian + lattice optimization)

**References**:
- QuIP: [arxiv.org/abs/2301.12017](https://arxiv.org/abs/2301.12017)
- QuIP#: [arxiv.org/abs/2311.06856](https://arxiv.org/abs/2311.06856)
- Implementation: [github.com/Cornell-RelaxML/quip-sharp](https://github.com/Cornell-RelaxML/quip-sharp)

---

## 8. AQLM (Additive Quantization for LLMs)

### Core Idea

AQLM uses **additive (product-like) quantization** to achieve 1-2 bit quantization:

Instead of representing each weight with a single codeword, AQLM expresses each weight group as a **sum of multiple codebook entries**:

```
w ≈ codebook_1[i_1] + codebook_2[i_2] + ... + codebook_k[i_k]
```

For example, with k=2 codebooks of size 256 each:
- Each weight group needs 2 indices (8 bits each = 16 bits)
- For a group of 8 weights: 16/8 = 2 bits per weight
- The effective codebook size is 256^2 = 65,536 codewords (additive combinations)

### Codebook Training

1. Initialize codebooks randomly or with fine-grained quantization centroids
2. Iteratively optimize: fix codebooks → find best indices → fix indices → update codebooks
3. Calibration data used to minimize reconstruction error

### Quality Impact

| Configuration | Perplexity Delta vs FP16 |
|---------------|--------------------------|
| AQLM 2-bit (2x8) | 0.3-0.8 PPL |
| AQLM 1-bit (2x4) | 1.5-3.0 PPL |
| AQLM 1.5-bit | 0.8-1.5 PPL |

AQLM at 2-bit is competitive with QuIP# and GPTQ 3-bit.

### Expected Speedup

| Platform | Speedup vs FP16 |
|----------|-----------------|
| M2 Metal | 4.0-5.0x (1-2 bit weights) |
| RTX 3080 CUDA | 4.5x (dequant + matmul fused) |

### C++17 Implementation Complexity: **High**

- **Codebook storage**: Small (k * 256 * sizeof(float) per layer)
- **Dequantization**: Each weight requires k table lookups + k-1 additions
- **Matmul**: Can be fused — compute output as sum of k small matmuls with quantized indices
- **Training**: Python-only (offline, expensive)
- **Runtime inference**: Medium complexity (similar to multi-codebook VQ)

**Reference**: [arxiv.org/abs/2306.03117](https://arxiv.org/abs/2306.03117)

---

## 9. SpQR (Sparse-Quantized Representation with Outlier Separation)

### Core Idea

SpQR targets the observation that in LLM weight matrices:
- **Most weights** follow a well-behaved distribution suitable for low-bit quantization
- **A small fraction** (~0.1-1%) are "outliers" that cause the most quantization error

SpQR separately stores these outlier weights:
1. Detect outliers based on magnitude or Hessian importance
2. Store outliers in FP16 (sparse format)
3. Quantize remaining weights to very low precision (2-3 bit)
4. At inference: dense quantized matmul + sparse outlier accumulation

```
output = (W_quant @ X) + (W_outlier_sparse @ X)
          [fast path]    [sparse path]
```

### Quality Impact

| Configuration | Perplexity Delta vs FP16 |
|---------------|--------------------------|
| SpQR 2-bit + outliers | 0.2-0.5 PPL |
| SpQR 3-bit + outliers | 0.05-0.15 PPL |

Outlier separation dramatically reduces the perplexity penalty of very low-bit quantization, especially for models with heavy-tailed weight distributions.

### Expected Speedup

| Platform | Speedup vs FP16 |
|----------|-----------------|
| M2 Metal | 4.5-5.0x (2-bit + sparse) |
| RTX 3080 CUDA | 4.5x (2-bit + sparse) |

*Note: Sparse matrix multiplication is less efficient on GPUs than dense. The speedup depends on outlier sparsity. At <0.5% outliers, sparse overhead is negligible.*

### C++17 Implementation Complexity: **High**

- **Outlier detection**: Hessian-based (offline, Python)
- **Sparse format**: CSR or COO for outlier weights
- **Two-path matmul**: Dense quantized path (standard) + sparse accumulation
- **Metal**: Sparse matmul on Metal is challenging (no native sparse support)
- **CUDA**: cuSPARSE exists but adds dependency; custom sparse kernel needed for zero-dep

**Reference**: [arxiv.org/abs/2306.03078](https://arxiv.org/abs/2306.03078)

---

## 10. FP8 (E4M3 / E5M2) — NVIDIA Native Support

### Core Idea

FP8 uses 8-bit floating point numbers with two formats defined in the OCP (Open Compute Project) FP8 specification:

| Format | Sign | Exponent | Mantissa | Max Value | Precision | Use Case |
|--------|------|----------|----------|-----------|-----------|----------|
| E4M3 | 1 | 4 | 3 | 448 | ~1 bit | **Weights** (higher range, lower precision) |
| E5M2 | 1 | 5 | 2 | 57,344 | ~0.5 bits | **Activations** (higher range, dynamic) |

### Native Hardware Support

- **NVIDIA Hopper (H100)**: Native FP8 Tensor Cores (TF32 precision accumulation)
- **NVIDIA Ada Lovelace**: Limited FP8 support (no native FP8 Tensor Cores — must use FP16 emulation)
- **NVIDIA Ampere (RTX 3080)**: **No native FP8 support** — must use FP16/FP32 emulation (no speedup, only memory savings)
- **Apple M2 Metal**: No native FP8 — must use FP16 emulation

### Quality Impact

| Configuration | Perplexity Delta vs FP16 |
|---------------|--------------------------|
| FP8 E4M3 weights + FP16 activations | 0.05-0.2 PPL |
| Full FP8 E4M3 | 0.2-0.8 PPL |
| FP8 E5M2 activations | 0.1-0.3 PPL |

### Expected Speedup

| Platform | Speedup vs FP16 |
|----------|-----------------|
| M2 Metal | 1.8-2.0x (bandwidth savings only, no native FP8) |
| RTX 3080 CUDA | 1.5x (bandwidth savings only, no native FP8 Tensor Cores) |
| RTX 4090 / H100 | 3.5-4.0x (native FP8 Tensor Cores) |

**Critical Note for hesa-llm**: FP8 provides **memory savings** on RTX 3080 (2x vs FP16) but **no compute speedup** (no native FP8 Tensor Cores on Ampere). On M2, FP8 also gives memory savings without native acceleration.

### C++17 Implementation Complexity: **Low-Medium**

- **Format conversion**: Simple bit manipulation (FP32 ↔ FP8)
- **FP16 emulation**: Convert FP8 → FP16 → compute → convert back
- **LUT approach**: Pre-compute FP8→FP16 table (256 entries, trivial)
- **No special kernels needed** on platforms without native FP8 support

**References**:
- FP8 Specification: [arxiv.org/abs/2209.05433](https://arxiv.org/abs/2209.05433)
- FP8 for Deep Learning: [arxiv.org/abs/2208.00992](https://arxiv.org/abs/2208.00992)

---

## 11. KV Cache Quantization

### Core Idea

During autoregressive generation, the **KV cache** (key and value tensors for attention) grows linearly with sequence length. For long contexts, KV cache can dominate memory:

```
KV cache size = 2 * num_layers * num_heads * head_dim * seq_len * sizeof(dtype) * batch_size
```

For Llama-2 13B with 4K context, FP16: ~2 * 40 * 40 * 128 * 4096 * 2 * 1 = **3.4 GB** just for KV cache!

### Quantization Strategies

#### Per-Token INT8
- Quantize each token's K and V vectors independently to INT8
- Each token stores its own scale + zero-point
- Quality impact: minimal (< 0.1 PPL) for INT8
- Memory reduction: ~2x

#### Per-Head INT4
- More aggressive: quantize to 4 bits per attention head
- Uses group-wise quantization within each head
- Quality impact: moderate (0.3-0.8 PPL)
- Memory reduction: ~4x

#### KIVI (2024)
- Group-wise KV cache quantization with separate quantization for K and V
- K quantized more aggressively than V
- Nearly lossless at INT8, small degradation at INT4

#### QoQ (2025)
- Dynamic KV cache quantization with runtime bit allocation
- Allocates more bits to critical attention heads
- Maintains quality with higher compression

### Quality Impact (Recent 2025 Findings)

| Method | Memory Reduction | Quality Impact |
|--------|-----------------|----------------|
| K/V INT8 | 2x | Negligible (< 0.1 PPL) |
| K INT4 + V FP16 hybrid | 1.5x | Minimal (< 0.05 PPL) |
| K/V INT4 (per-head) | 4x | Small (0.3-0.8 PPL) |
| K/V Q4_0 (block-wise) | 4x | Moderate (0.5-1.0 PPL) |

### Expected Speedup

| Platform | Benefit |
|----------|---------|
| M2 Metal | 1.5-2x decode speedup (less memory bandwidth per token) at 4K+ context |
| RTX 3080 CUDA | 1.8-2.5x decode speedup; enables 13B+ models that otherwise OOM at long context |

*Critical for RTX 3080's 10GB limit: KV cache quantization can turn an OOM at 4K context into a usable 8K context for 7B models.*

### C++17 Implementation Complexity: **Medium**

- **Per-token quantization**: During decode, quantize the new K/V token vector before storing in cache
- **Dequantization**: Before attention computation, dequantize cached tokens
- **Attention integration**: Modify attention kernel to work with quantized K/V
- **Ring buffer management**: KV cache must handle rolling window (sliding window) or offloading

**Reference**: KIVI [arxiv.org/abs/2402.02750](https://arxiv.org/abs/2402.02750)

---

## 12. Speculative Decoding & Speculative Prefill

### Core Idea

Autoregressive LLM generation is **inherently sequential** — each token depends on all previous tokens. Speculative decoding breaks this bottleneck by having a **fast draft model** generate γ tokens speculatively, then using the **main model** to verify them all in parallel:

```
Draft model generates: t₁, t₂, t₃ (3 tokens, serial)
Main model verifies:   [t₁, t₂, t₃] in ONE forward pass (parallel)
Repeat...
```

### Speedup Math

Without speculation: **n forward passes** for n tokens (decode phase: 1 token per pass)

With speculation (draft γ tokens, verification acceptance rate p):
- Expected accepted tokens per round: pγ
- Forward passes per round: 2 (1 draft pass + 1 verify pass — draft may also be parallel for larger γ)
- **Effective tokens per forward pass: pγ / 2**
- **Speedup ≈ n / (2n / (pγ)) = pγ / 2**

For p=0.7, γ=4: speedup ≈ 1.4x
For p=0.8, γ=6: speedup ≈ 2.4x
For p=0.9, γ=8: speedup ≈ 3.6x (upper bound, limited by draft quality)

**With Speculative Prefill** (newer technique):
- Draft model pre-fills γ tokens in parallel before each decode step
- **Decode phase: ~n/(pγ) forward passes instead of n**
- Speedup: up to pγx (main model verifies, no separate draft pass)

### Techniques

#### N-gram / Prompt Lookup Decoding
- **No additional model needed** — searches the prompt for matching n-gram continuations
- Extremely fast to build (hash table of n-grams from prompt)
- Works best for code completion, repeated structures
- Quality: depends on prompt structure

**References**: [arxiv.org/abs/2302.14827](https://arxiv.org/abs/2302.14827), [arxiv.org/abs/2304.04848](https://arxiv.org/abs/2304.04848)

#### EAGLE (Speculative Sampling with Feature Uncertainty)
- Uses the main model's **hidden states** as draft tokens
- A lightweight **draft head** predicts the next token from hidden states
- EAGLE-2 uses **dynamic draft trees** — multiple candidate tokens per step, tree-structured verification

**Key insight**: The main model's intermediate layers already contain "draft quality" information. EAGLE extracts this cheaply rather than running a separate model.

**References**:
- EAGLE: [arxiv.org/abs/2309.14717](https://arxiv.org/abs/2309.14717)
- EAGLE-2: [arxiv.org/abs/2401.04110](https://arxiv.org/abs/2401.04110)

#### Medusa (Multi-Head Speculative Decoding)
- Adds **multiple lightweight draft heads** on top of the main model
- Each head predicts token at position i+1, i+2, ..., i+k
- Trained jointly with the main model
- No separate draft model needed

**References**: [arxiv.org/abs/2310.19748](https://arxiv.org/abs/2310.19748)

#### Tree Attention Verification
- Instead of sequential verification of γ tokens, build a **tree of candidate tokens**
- Main model scores all candidates in parallel using **tree-structured attention mask**
- Accepts the longest prefix matching the model's distribution
- Achieves higher acceptance rates than linear verification

### Expected Speedup

| Technique | γ | M2 Metal | RTX 3080 CUDA |
|-----------|---|----------|---------------|
| N-gram / Prompt Lookup | 2-4 | 1.2-1.5x | 1.3-1.6x |
| EAGLE draft head | 3-5 | 1.5-2.0x | 1.8-2.5x |
| Medusa (k=3) | 3 | 1.3-1.7x | 1.5-2.0x |
| EAGLE-2 (tree) | 4-8 | 1.8-2.5x | 2.0-3.0x |

### Relevance to Target Hardware

**M2 (Metal, unified memory)**:
- Speculative decoding is **especially effective** on M2 because:
  - The verification pass reuses the same memory as the draft pass (unified memory, no PCI-e transfer)
  - Memory bandwidth is the bottleneck — fewer total forward passes = fewer memory reads
  - Speculative prefill (loading many KV entries in one pass) is highly efficient
  - **N-gram speculation** is free (no model needed) and works well for code
- **EAGLE-2 tree attention** benefits most from Apple's unified memory

**RTX 3080 (CUDA, 10GB)**:
- VRAM is the primary constraint — speculation trades extra computation for fewer sequential passes
- EAGLE is attractive because it only needs lightweight additional heads (not a full draft model)
- **Medusa** heads add minimal VRAM overhead (~50-100MB for k=3)
- Speculative prefill enables processing more tokens per batch within the 10GB limit

### C++17 Implementation Complexity: **High → Very High**

| Component | Complexity |
|-----------|------------|
| N-gram speculation | Low (hash table build + match) |
| Draft token generation | Medium (simple MLP head on top of hidden states) |
| Tree-structured verification | High (custom attention mask, tree traversal) |
| Medusa multi-head training | Very High (requires fine-tuning with additional heads) |
| EAGLE draft head training | Very High (requires training on model's own hidden states) |

**Inference-only components** (for pre-converted Medusa/EAGLE models):
- **N-gram lookup**: Low — simple hash map over prompt tokens
- **EAGLE draft head**: Medium — small MLP (linear layers + LayerNorm + GELU) on top of hidden states, plus tree verification logic
- **Medusa heads**: Medium — k additional linear layers + tree verification
- **Tree attention**: High — requires attention mask generation for tree-structured token candidates

---

## 2025-2026 Recent Developments

### New Quantization Papers (from arxiv search, April 2026)

| Paper | arxiv | Date | Notes |
|-------|-------|------|-------|
| **Fast NF4 Dequantization Kernels for Large Language Model Inference** | 2604.02556 | Apr 2026 | Optimized NF4 dequant kernels; directly applicable to hesa-llm |
| **AdaHOP: Fast and Accurate Low-Precision Training via Outlier-Pattern-Aware Rotation** | 2604.02525 | Apr 2026 | Outlier-aware rotation for low-precision; related to QuIP methodology |
| **Goose: Anisotropic Speculation Trees for Training-Free Speculative Decoding** | 2604.02047 | Apr 2026 | Training-free tree speculation; relevant for EAGLE-like methods |
| **S2D2: Fast Decoding for Diffusion LLMs via Training-Free Self-Speculation** | 2603.25702 | Mar 2026 | Self-speculation without draft model |
| **SpecForge: A Flexible Open-Source Training Framework for Speculative Decoding** | 2603.18567 | Mar 2026 | Open-source framework for training speculation heads |

### Trends

1. **2-bit is maturing**: QuIP# and AQLM make 2-bit viable for production
2. **Speculative decoding is mainstream**: EAGLE-2 achieves 2-3x speedups with minimal quality loss
3. **Training-free speculation is trending**: N-gram and self-speculation methods require no model modification
4. **Hardware-aware quantization**: Methods like TurboQuant and newer papers focus on maximizing actual speedup, not just model size reduction
5. **KV cache optimization**: Becoming critical as context windows grow to 128K+

---

## Comparison Table

| Technique | Min Bits | Quality | M2 Speedup | RTX 3080 Speedup | C++ Complexity | Needs Calibration? | Notes |
|-----------|----------|---------|------------|-------------------|----------------|--------------------|-------|
| **GGUF Q4_0** | 4.0 | Good | 3.5x | 3.8x | Medium | No | Best starting point |
| **GGUF Q4_K_M** | 4.5 | Very Good | 3.2x | 3.4x | Medium | No | Best quality/size tradeoff |
| **GGUF Q6_K** | 6.0 | Excellent | 2.3x | 2.5x | Medium | No | Near-lossless |
| **TurboQuant** | 3.0 avg | Very Good | 3.8x | 4.2x | High | Yes | Mixed-bit, hardware-aware |
| **AWQ** | 4.0 | Excellent | 3.2x | 3.8x | High | Yes | Salient weight protection |
| **GPTQ 4-bit** | 4.0 | Excellent | 3.0x | 3.5x | Very High | Yes | Hessian-based |
| **EXL2 + exllamav2** | 2-8 | Configurable | N/A | 4.0x+ | Very High | Yes | CUDA-optimized |
| **LLM.int8()** | 8.0 (+ FP16 outlier) | Near-lossless | 2.5x | 3.0x | High | Yes | Outlier handling |
| **SmoothQuant** | 8.0 W/A | Near-lossless | 2.8x | 3.5x | Medium | Yes | Pure INT8 W+A |
| **QuIP#** | 2.0 | Very Good | 4.5x | 5.0x | Very High | Yes | Lattice + Hadamard |
| **AQLM** | 1.0 | Moderate | 4.0x | 4.5x | High | Yes | Additive codebooks |
| **SpQR** | 2.0 | Very Good | 4.5x | 4.5x | High | Yes | Sparse outlier separation |
| **FP8 E4M3** | 8.0 | Excellent | 1.8x* | 1.5x* | Low | No | *Memory only on target hardware |
| **KV Cache INT8** | N/A | Negligible loss | 1.5x decode | 1.8x decode | Medium | No | Context-length dependent |
| **KV Cache INT4** | N/A | Small loss | 1.8x decode | 2.2x decode | Medium | No | Context-length dependent |
| **N-gram Spec** | N/A | None | 1.2-1.5x | 1.3-1.6x | Low | No | Training-free |
| **EAGLE-2** | N/A | None | 1.8-2.5x | 2.0-3.0x | Very High | Yes (training) | Dynamic draft trees |
| **Medusa** | N/A | None | 1.3-1.7x | 1.5-2.0x | High | Yes (fine-tuning) | Multi-head draft |

---

## Phased Implementation Plan for hesa-llm

### Phase 1: Foundation (Weeks 1-4)
**Goal**: Basic inference with GGUF quantization support

| Task | Priority | Complexity | Notes |
|------|----------|------------|-------|
| GGUF file parser | P0 | Medium | Read tensor metadata, raw quantized weights |
| Q4_0 quantizer / dequantizer | P0 | Low | Simplest k-quant, 4096-entry LUT |
| Q4_K_M dequantizer | P0 | Medium | Subblock + importance matrix |
| FP16 weight loading | P0 | Low | Baseline for comparison |
| Basic quantized matmul (CPU) | P0 | Medium | SIMD-optimized, fallback path |
| Metal quantized matmul shader | P0 | High | M2 primary path |
| CUDA quantized matmul kernel | P1 | High | RTX 3080 primary path |

**Deliverable**: hesa-llm runs GGUF Q4_K_M models on M2 and RTX 3080

### Phase 2: KV Cache Optimization (Weeks 5-8)
**Goal**: Efficient context handling with KV cache quantization

| Task | Priority | Complexity | Notes |
|------|----------|------------|-------|
| KV cache ring buffer | P0 | Low | Cyclic buffer for sliding window |
| Per-token INT8 KV quantization | P0 | Medium | Quantize on-the-fly during decode |
| Per-token INT4 KV quantization | P1 | Medium | Optional aggressive mode |
| Quantized attention kernel | P0 | Medium | Dequantize K/V before attention |
| Paged KV cache | P1 | High | Non-contiguous allocation for memory efficiency |

**Deliverable**: 8K+ context for 7B models on RTX 3080 in 10GB VRAM

### Phase 3: Advanced Quantization (Weeks 9-14)
**Goal**: Support AWQ and GPTQ formats, 2-bit quantization

| Task | Priority | Complexity | Notes |
|------|----------|------------|-------|
| AWQ weight format support | P1 | Medium | Load AWQ-converted models |
| GPTQ weight format support | P1 | Medium | GPTQ v1/v2 (group-wise scales) |
| NF4 (bitsandbytes) dequantizer | P1 | Low | 256-entry NormalFloat LUT |
| Q2_K / Q3_K_M GGML formats | P1 | Medium | Extreme compression modes |
| AWQ CUDA kernel optimization | P2 | High | Custom TinyChat-style kernels |

**Deliverable**: Support for most popular quantized model formats

### Phase 4: Speculative Decoding (Weeks 15-20)
**Goal**: Inference acceleration via speculation

| Task | Priority | Complexity | Notes |
|------|----------|------------|-------|
| N-gram prompt lookup | P0 | Low | Training-free, immediate speedup |
| Token acceptance verification | P0 | Medium | Main model verify draft tokens |
| Draft head interface | P1 | Medium | Pluggable architecture |
| EAGLE-style draft head inference | P1 | High | Small MLP on hidden states |
| Tree-structured verification | P1 | High | Parallel candidate validation |
| Medusa multi-head support | P2 | High | k draft heads + tree verification |
| Speculative prefill | P2 | High | Batch draft+verify efficiently |

**Deliverable**: 1.5-2x decode speedup on M2 and RTX 3080

### Phase 5: Optimization and Edge Cases (Weeks 21-24)
**Goal**: Maximize performance, handle long contexts

| Task | Priority | Complexity | Notes |
|------|----------|------------|-------|
| Flash-attention integration | P0 | High | For long-context prefill |
| KV cache offloading (CPU↔GPU) | P1 | Medium | Extended context beyond VRAM |
| Mixed-precision layer assignment | P1 | High | Auto-select Q4/Q6 per layer |
| Metal Performance Shaders (MPS) integration | P1 | Medium | Apple-optimized primitives |
| cuBLAS integration (CUDA) | P2 | Low | cuBLASLt for int8/fp8 matmul |
| Benchmarking suite | P0 | Medium | Perplexity + throughput metrics |

**Deliverable**: Production-ready LLM inference engine

---

## Recommendations for hesa-llm

### Best Starting Points

1. **GGUF Q4_K_M as default format**: Best balance of quality, size, and implementation simplicity. Well-documented, widely used, zero calibration needed at runtime.

2. **KV Cache INT8 as default**: Negligible quality impact with 2x memory savings. Critical for RTX 3080's 10GB constraint.

3. **N-gram speculative decoding**: Training-free, low complexity, immediate speedup for code completion and structured text.

### Best Mid-Term Targets

4. **AWQ format support**: Best 4-bit quantization quality. Widely available model weights.

5. **EAGLE-style draft heads**: Best cost/benefit ratio for speculative decoding (lightweight heads, no separate model).

### Long-Term / Advanced

6. **QuIP# 2-bit**: Best ultra-low-bit quality, but high implementation complexity.

7. **EXL2 format + custom kernels**: Best raw speed on CUDA, but CUDA-only and very complex.

### Not Recommended for Target Hardware

- **FP8 E4M3 as primary format**: No native support on M2 or RTX 3080. Only useful for memory savings.
- **SpQR**: Sparse computation overhead may negate benefits on M2 metal shaders.

---

## Appendix: Quick Reference — Model Size vs Quantization

### Llama-2 7B (~13B parameters)

| Quantization | Model Size | KV Cache (4K, INT8) | Total Memory | Fits M2? | Fits RTX 3080? |
|-------------|------------|---------------------|--------------|----------|----------------|
| FP16 | 26 GB | ~700 MB | 26.7 GB | No | No |
| Q6_K | 18 GB | ~700 MB | 18.7 GB | Yes (tight) | No |
| Q4_K_M | 13.5 GB | ~700 MB | 14.2 GB | Yes | Yes |
| 4-bit AWQ | 13.5 GB | ~350 MB (INT4 KV) | 13.85 GB | Yes | Yes |
| 2-bit QuIP# | 6.75 GB | ~350 MB | 7.1 GB | Yes (comfortable) | Yes (comfortable) |

### Llama-2 13B (~26B parameters)

| Quantization | Model Size | KV Cache (4K, INT8) | Total Memory | Fits M2? | Fits RTX 3080? |
|-------------|------------|---------------------|--------------|----------|----------------|
| FP16 | 52 GB | ~1.4 GB | 53.4 GB | No | No |
| Q6_K | 36 GB | ~1.4 GB | 37.4 GB | No | No |
| Q4_K_M | 27 GB | ~1.4 GB | 28.4 GB | **No** | **No** (10GB limit) |
| Q4_K_M + INT4 KV | 27 GB | ~700 MB | 27.7 GB | **No** | **No** |
| 2-bit QuIP# | 13.5 GB | ~700 MB | 14.2 GB | **Borderline** | No |
| 2-bit Q4_K_M + long context offload | N/A | Variable | Variable | Maybe with CPU offload | No |

**Conclusion**: For RTX 3080, 7B models with 4-bit quantization is the sweet spot. 13B models require 2-bit quantization + KV cache compression and may still struggle. M2 can handle 13B at Q4 (barely) or 13B at 2-bit comfortably.

---

*Document created: 2026-04-06*
*References verified via arxiv API on 2026-04-06*

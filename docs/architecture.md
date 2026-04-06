# HESA-LLM Architecture Document

> **H**ierarchical **E**xternal-**S**tate **A**ugmented LLM Inference Engine
> Version: 0.1.0-draft
> Status: Design Phase
> Target: C++20, Portable Inference, Research-Backed Architecture

---

## Table of Contents

1. [Architecture Overview](#1-architecture-overview)
2. [Design Goals](#2-design-goals)
3. [Core Components](#3-core-components)
4. [Model Format (HESF)](#4-model-format-hesf)
5. [Research Integration](#5-research-integration)
6. [Hardware Targets](#6-hardware-targets)
7. [API Design](#7-api-design)
8. [Build System](#8-build-system)
9. [Phased Implementation Plan](#9-phased-implementation-plan)
10. [Directory Layout](#10-directory-layout)

---

## 1. Architecture Overview

### 1.1 Component Dataflow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           HESA-LLM ENGINE                               │
│                                                                         │
│  ┌──────────┐    ┌─────────────┐    ┌──────────────────────────────┐   │
│  │  CLIENT  │───>│  API / CLI  │───>│        ORCHESTRATOR          │   │
│  │ (OpenAI  │    │   Layer     │    │  (hesa::Orchestrator)        │   │
│  │  compat) │    │             │    │                              │   │
│  └──────────┘    └─────────────┘    │  ┌──────────────────────┐    │   │
│                                     │  │  Session Manager     │    │   │
│                                     │  │  - KV cache alloc    │    │   │
│                                     │  │  - Context window    │    │   │
│                                     │  │  - Batch scheduling  │    │   │
│                                     │  └──────────────────────┘    │   │
│                                     └──────────┬───────────────────┘   │
│                                                │                       │
│                   ┌────────────────────────────┼──────────────────┐    │
│                   │           EXECUTION GRAPH  │                   │    │
│                   │                            ▼                   │    │
│                   │  ┌──────────────────────────────────────────┐  │    │
│                   │  │          TOKENIZER (hesa::Tokenizer)     │  │    │
│                   │  │  - BPE / SentencePiece / tiktoken        │  │    │
│                   │  │  - Encoding/decoding pipeline            │  │    │
│                   │  └──────────────────────┬───────────────────┘  │    │
│                   │                         │                      │    │
│                   │                         ▼                      │    │
│                   │  ┌──────────────────────────────────────────┐  │    │
│                   │  │        EMBEDDING LAYER                   │  │    │
│                   │  │  - Token embedding lookup (GGML tensor)  │  │    │
│                   │  │  - Positional encoding (RoPE/ALiBi)      │  │    │
│                   │  │  - Hyper-connection projection (optional) │  │    │
│                   │  └──────────────────────┬───────────────────┘  │    │
│                   │                         │                      │    │
│                   │                         ▼                      │    │
│                   │  ┌──────────────────────────────────────────┐  │    │
│                   │  │      TRANSFORMER BLOCKS (layer 0..N-1)   │  │    │
│                   │  │                                          │  │    │
│                   │  │  ┌────────────┐  ┌────────────────────┐  │  │    │
│                   │  │  │ Pre-Norm   │  │ mHC Hyper-Conn     │  │  │    │
│                   │  │  │ RMSNorm    │  │ (residual bypass   │  │    │
│                   │  │  └─────┬──────┘  │  manifold constr.) │  │    │
│                   │  │        │         └─────────┬──────────┘  │    │
│                   │  │        ▼                   │             │    │
│                   │  │  ┌─────────────────────────┴──────────┐  │    │
│                   │  │  │  HYBRID ATTENTION MODULE           │  │    │
│                   │  │  │                                    │  │    │
│                   │  │  │  ┌───────────┐  ┌──────────────┐  │  │    │
│                   │  │  │  │ Softmax   │  │ Linear       │  │  │    │
│                   │  │  │  │ Attention │  │ Attention    │  │  │    │
│                   │  │  │  │ (KV cache)│  │ (Kimi-style  │  │  │    │
│                   │  │  │  │           │  │  recurrent)  │  │  │    │
│                   │  │  │  └─────┬─────┘  └──────┬───────┘  │  │    │
│                   │  │  │        │    merge       │         │  │    │
│                   │  │  │        └──────┬─────────┘         │  │    │
│                   │  │  │   HybridGate (learned blend)      │  │    │
│                   │  │  └───────────────┼───────────────────┘  │    │
│                   │  │                    │                     │    │
│                   │  │                    ▼                     │    │
│                   │  │  ┌─────────────────────────────────────┐ │    │
│                   │  │  │ NEURAL MEMORY MODULE (Titans/TTT)   │ │    │
│                   │  │  │                                     │ │    │
│                   │  │  │  ┌─────────────┐ ┌───────────────┐  │ │    │
│                   │  │  │  │ TTT Layer   │ │ Engram Cache  │  │ │    │
│                   │  │  │  │ (learnable  │ │ (long-term    │  │ │    │
│                   │  │  │  │  test-time  │ │  state        │  │ │    │
│                   │  │  │  │  optimizer) │ │  compressed)  │  │ │    │
│                   │  │  │  └──────┬──────┘ └───────┬───────┘  │ │    │
│                   │  │  │         └───────┬────────┘         │ │    │
│                   │  │  │         Memory Gate (blend)        │ │    │
│                   │  │  └────────────────┬───────────────────┘ │    │
│                   │  │                   │                      │    │
│                   │  │                   ▼                      │    │
│                   │  │  ┌─────────────────────────────────────┐ │    │
│                   │  │  │     FEED-FORWARD (Gated MLP)        │ │    │
│                   │  │  │  - GELU/GeGLU/SwiGLU activation     │ │    │
│                   │  │  │  - Dynamic expert routing (opt.)    │ │    │
│                   │  │  └─────────────────────────────────────┘ │    │
│                   │  │                                          │    │
│                   │  └──────────────────────┬───────────────────┘    │
│                   │                         │ (residual + mHC)       │
│                   │                         ▼                        │
│                   │              (repeat N layers...)                │
│                   └──────────────────────────────────────────────────┘
│                                                │
│                         ┌──────────────────────┴──────────────────┐
│                         ▼                                         │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │              COMPUTE BACKEND DISPATCH (hesa::Backend)        │ │
│  │                                                              │ │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────────┐  │ │
│  │  │ x86_64   │  │ ARM NEON │  │   CUDA   │  │    Metal    │  │ │
│  │  │ (AVX2/   │  │ (M-series│  │  (sm_50+ │  │  (Apple     │  │ │
│  │  │  AVX-512)│  │  Mac     │  │  sm_80+  │  │  Silicon)   │  │ │
│  │  │          │  │  silicon)│  │  opt)    │  │             │  │ │
│  │  └──────────┘  └──────────┘  └──────────┘  └─────────────┘  │ │
│  │                                                              │ │
│  │  Unified tensor abstraction (hesa::Tensor)                   │ │
│  │  - Memory: gguf-mapped or GPU-allocated                      │ │
│  │  - Compute: backend-specific kernels                         │ │
│  │  - Sync: async stream with completion tracking               │ │
│  └─────────────────────────────┬────────────────────────────────┘ │
│                                │                                  │
│                               ▼                                   │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │              OUTPUT HEAD                                    │  │
│  │  - Final RMSNorm → Linear projection → Logits               │  │
│  │  - Top-K / Top-P / Temperature sampling                     │  │
│  │  - Speculative decoding support (draft-verify)              │  │
│  └─────────────────────────────────────────────────────────────┘  │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Architectural Principles

- **Layered abstraction**: Every component implements a pure-virtual interface with pluggable implementations.
- **Zero-copy where possible**: Model weights mapped directly via `mmap` (mimicking GGUF philosophy).
- **Graph-based execution**: Layers scheduled as nodes in a DAG; enables CPU/GPU pipelining and parallel layer execution.
- **State externalization**: All inference state (KV cache, neural memory, TTT state) is explicitly managed in session handles — enables pause/resume, multi-session, and speculative decoding.

---

## 2. Design Goals

### 2.1 Primary Goals

| Goal | Target | Metric |
|------|--------|--------|
| **Portability** | Linux x86_64, macOS AArch64, Windows x86_64 | Single codebase, `#ifdef`-minimal |
| **Performance** | > 100 tok/s on M2 (24GB) for 7B model | Tokens/sec at batch-size=1 |
| **Memory efficiency** | Fit 13B Q4 in 10GB VRAM (RTX 3080) | Bytes per effective parameter |
| **Small form factor** | Binary < 5 MB (core), < 50 KB LoC | Minimal dependencies |
| **Contributor-friendly** | First-time build < 5 min, < 3 dependencies | `cmake && make && ./hesa` |
| **Model quality** | Gemma-4 class performance at 3B-7B scale | Research-backed architecture |

### 2.2 Non-Goals

- Training support (Phase 1 is inference-only)
- Python bindings (Phase 3+)
- WebGPU backend (Phase 2+)
- Distributed inference (Phase 2+)

### 2.3 Design Trade-offs

- **C++20 minimum**: Coroutines for async generation, concepts for backend interfaces, `std::span` for tensor views. No C++23 required for portability to older compilers.
- **Minimal external dependencies**: Only `cmake`, a JSON parser (nlohmann or custom), and optional backend SDKs (CUDA toolkit, Metal). Tokenizer uses embedded sentencepiece or BPE — no Python runtime.
- **Fork-friendly**: Where practical, extend GGML/tensor concepts from llama.cpp rather than reinvent. Where llama.cpp cannot support a feature (neural memory, TTT), implement natively.

---

## 3. Core Components

### 3.1 Tokenizer (`hesa::Tokenizer`)

```cpp
namespace hesa {

class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<int32_t> encode(const std::string& text) const = 0;
    virtual std::string decode(std::span<const int32_t> tokens) const = 0;

    // Batch encode/decode for throughput
    virtual std::vector<std::vector<int32_t>>
        encode_batch(std::span<const std::string> texts) const = 0;

    virtual int32_t bos_token_id() const = 0;
    virtual int32_t eos_token_id() const = 0;
    virtual int32_t vocab_size() const = 0;
    virtual size_t max_token_length() const = 0;
};

// Implementations:
//   BPE_Tokenizer    — OpenAI-style (tiktoken-compatible)
//   SP_Tokenizer     — SentencePiece (unigram/BPE models)
//   T5_Tokenizer     — SentencePiece with T5 pre-processing
class BPE_Tokenizer : public Tokenizer { ... };
class SP_Tokenizer   : public Tokenizer { ... };

} // namespace hesa
```

**Design decisions:**
- Tokenizer state is loaded at model init from the model file (no external `.model` file required).
- Regex rules for BPE are compiled at load time into a DFA for fast matching.
- Supports added tokens (chat templates, special tokens via JSON metadata).

### 3.2 Inference Engine (`hesa::Engine`)

```cpp
namespace hesa {

struct GenerationConfig {
    float   temperature    = 0.8f;
    float   top_p          = 0.95f;
    int32_t top_k          = 40;
    float   repetition_penalty = 1.1f;
    int32_t max_tokens     = 2048;
    bool    use_speculative = false;
    int32_t draft_tokens   = 4;
};

struct EngineConfig {
    BackendType   backend       = BackendType::AUTO;
    int32_t       n_threads     = 0; // 0 = auto-detect
    size_t        kv_cache_size = 4096; // context length
    size_t        batch_size    = 1;
    bool          flash_attn    = true;
    QuantMode     quant_mode    = QuantMode::NONE;
    MemoryConfig  neural_mem;   // Titans/TTT memory config
    bool          hyper_connections = true; // mHC enabled
};

class Engine {
public:
    static Result<std::unique_ptr<Engine>> create(const EngineConfig& cfg,
                                                   const Model& model);

    // Synchronous single-sample generation
    Result<std::vector<int32_t>> generate(std::span<const int32_t> prompt,
                                          const GenerationConfig& cfg,
                                          Callbacks callbacks = {});

    // Async generation with streaming tokens
    Result<AsyncGenerator> generate_async(std::span<const int32_t> prompt,
                                          const GenerationConfig& cfg);

    // Session management for multi-conversation
    Result<SessionId> create_session();
    Result<void> destroy_session(SessionId sid);
    Result<void> set_session_context(SessionId sid, ContextView);

    // Speculative decoding
    Result<std::vector<int32_t>> generate_speculative(
        std::span<const int32_t> prompt,
        const Engine& draft_model,
        const GenerationConfig& cfg);

    // State management
    Result<InferenceState> save_state(SessionId sid) const;
    Result<void> load_state(SessionId sid, const InferenceState& state);
};

} // namespace hesa
```

**Execution model:**
- Tokens processed in batches of up to `batch_size` (prompt processing)
- Auto-regressive loop with `batch_size=1` for generation
- All tensor operations go through the compute backend (Section 3.4)
- KV cache is ring-buffer allocated with automatic overflow handling (sliding window or eviction policy)

### 3.3 Memory System

The memory system is the key differentiator from traditional transformer engines. It consists of three subsystems:

#### 3.3.1 KV Cache (`hesa::KVCACHE`)

```cpp
namespace hesa {

struct KVCacheConfig {
    size_t max_tokens;       // Context window size
    size_t head_dim;         // Per-head dimension
    size_t n_layers;
    size_t n_kv_heads;
    CacheEvictionPolicy eviction = CacheEvictionPolicy::SLIDING_WINDOW;
    size_t sliding_window = 4096; // For sliding window attention
};

class KVCACHE {
public:
    KVCACHE(const KVCacheConfig& cfg, Backend& backend);

    // Allocate space for batch. Returns token positions.
    Result<std::vector<int32_t>> reserve(size_t n_tokens);

    // Write keys/values for a batch of tokens at given positions
    Result<void> write_keys(size_t layer, size_t head,
                            std::span<const int32_t> positions,
                            TensorView keys);
    Result<void> write_values(size_t layer, size_t head,
                              std::span<const int32_t> positions,
                              TensorView values);

    // Read all relevant K/V for attention at given query position
    Result<TensorView> read_keys(size_t layer, size_t head,
                                 std::span<const int32_t> q_positions);
    Result<TensorView> read_values(size_t layer, size_t head,
                                   std::span<const int32_t> q_positions);

    // Memory pressure management
    Result<void> evict_oldest(size_t n_tokens);
    size_t used() const;
    size_t capacity() const;

    // GPU memory: pinned and pageable options
    Result<void> pin_to_device(Backend& backend);

private:
    // Ring buffer for token-level K/V storage
    std::vector<Tensor> key_cache_;
    std::vector<Tensor> value_cache_;
    std::vector<size_t> token_positions_; // logical -> physical
    size_t head_offset_;
    size_t layer_offset_;
};

} // namespace hesa
```

#### 3.3.2 Neural Memory (Titans) (`hesa::NeuralMemory`)

Implements the persistent memory state from the Titans architecture. Unlike KV cache which stores raw token representations, neural memory learns compressed, persistent state.

```cpp
namespace hesa {

struct NeuralMemoryConfig {
    size_t state_dim;          // State vector dimension
    size_t n_memory_heads;     // Parallel memory heads
    MemoryInitMode init = MemoryInitMode::ZERO;
    float  decay_rate = 0.99f; // Exponential decay for relevance
    bool   persistent = true;  // Survive context window slides
};

class NeuralMemory {
public:
    NeuralMemory(const NeuralMemoryConfig& cfg, Backend& backend);

    // Update memory state with new token representations
    // Memory state: m_t = decay * m_{t-1} + gate * f(x_t, m_{t-1})
    Result<void> update(TensorView input, TensorView gate, float decay);

    // Query memory for attention augmentation
    Result<TensorView> query(TensorView query_vector);

    // Retrieve compressed memory for long-range context
    Result<TensorView> retrieve(size_t top_k);

    // Snapshot for pause/resume
    Result<Tensor> snapshot() const;
    Result<void> restore(Tensor state);

    size_t memory_size() const; // Total bytes: state_dim * n_heads * sizeof(float)

private:
    Tensor memory_state_;      // [n_memory_heads, state_dim]
    Tensor memory_gate_weights_; // Learned gate projection
    bool pinned_to_device_ = false;
};

} // namespace hesa
```

**Memory efficiency**: Neural memory adds O(state_dim * n_heads) persistent state independent of context window length — typically 1-4 MB vs. KV cache which grows linearly (O(context * heads * dim) = 100+ MB for 8K context).

#### 3.3.3 TTT Memory (`hesa::TTTLayer`)

Implements Test-Time Training layers where internal parameters are updated during inference.

```cpp
namespace hesa {

struct TTTConfig {
    size_t hidden_dim;
    float  ttt_lr = 0.01f;        // Test-time learning rate
    TTTOptimizer optimizer = TTTOptimizer::SGD;
    float  momentum = 0.9f;
    size_t grad_accum_steps = 1;   // Gradient accumulation
    bool   enable = true;
};

class TTTLayer {
public:
    TTTLayer(const TTTConfig& cfg, Backend& backend);

    // Forward pass with internal parameter update
    // Loss is self-supervised (predict next token from hidden state)
    Result<TensorView> forward(TensorView input);

    // Internal step: update parameters based on prediction error
    Result<void> ttt_step(TensorView input, TensorView target);

    // Freeze/thaw TTT parameters (useful for prompt prefill vs generation)
    Result<void> freeze();
    Result<void> thaw();

    // State serialization
    Result<Tensor> export_state() const;
    Result<void> import_state(Tensor state);

private:
    Tensor internal_params_;     // Learnable TTT parameters
    Tensor optimizer_state_;     // Momentum/Adam state
    Tensor input_buffer_;        // Buffer for gradient computation
    bool is_frozen_ = false;
};

} // namespace hesa
```

**Performance consideration**: TTT steps are lightweight (SGD on small parameter set) but must complete within the per-token budget. Typical overhead: 5-15% at sequence length < 4K.

### 3.4 Compute Backend (`hesa::Backend`)

```cpp
namespace hesa {

enum class BackendType {
    AUTO,        // Auto-detect best available
    CPU_X86,     // x86_64 with AVX2/AVX-512
    CPU_ARM,     // ARM NEON (Apple Silicon, Raspberry Pi)
    CUDA,        // NVIDIA GPU via CUDA
    METAL,       // Apple Metal (macOS/iOS)
    VULKAN,      // Cross-platform GPU (future)
};

enum class Dtype {
    F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, I8
};

class Tensor {
public:
    Tensor(Dtype dtype, std::span<const int64_t> shape, Backend& backend);

    // Memory access
    void* data();
    const void* data() const;
    void* data_host(); // Paged copy for CPU access

    // Shape and strides
    std::span<const int64_t> shape() const;
    int64_t nelements() const;
    size_t nbytes() const;

    // View operations (zero-copy reshaping)
    TensorView reshape(std::span<const int64_t> new_shape);
    TensorView transpose(int ax0, int ax1);
    TensorView select(int axis, int64_t index);

    // Device management
    bool is_on_device() const;
    Result<void> to_device();
    Result<void> to_host();
    Result<void> copy_to(Tensor& dst) const;

    // Backend handle (for ops that need raw pointers)
    void* backend_handle();

private:
    // PIMPL pattern for backend-specific storage
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Backend {
public:
    virtual ~Backend() = default;

    // Backend identification
    virtual BackendType type() const = 0;
    virtual const char* name() const = 0;
    virtual bool supports(Dtype dt) const = 0;
    virtual size_t device_memory() const = 0;  // Total device memory
    virtual size_t device_memory_used() const = 0;

    // Tensor allocation
    virtual Result<Tensor> alloc_tensor(Dtype dtype,
                                        std::span<const int64_t> shape) = 0;

    // Core operations
    virtual Result<void> matmul(TensorView a, TensorView b,
                                TensorView out, float scale = 1.0f) = 0;
    virtual Result<void> add(TensorView a, TensorView b,
                             TensorView out) = 0;
    virtual Result<void> mul(TensorView a, TensorView b,
                             TensorView out) = 0;
    virtual Result<void> softmax(TensorView in,
                                 TensorView out, int axis) = 0;
    virtual Result<void> rms_norm(TensorView in, TensorView weight,
                                  TensorView out, float eps = 1e-6f) = 0;
    virtual Result<void> rope(TensorView in_out,
                              std::span<const int32_t> positions,
                              float freq_base, int n_dims) = 0;

    // Attention
    virtual Result<void> scaled_dot_product_attention(
        TensorView q, TensorView k, TensorView v, TensorView out,
        TensorView mask = {}, float scale = 0.0f) = 0;
    virtual Result<void> flash_attention(
        TensorView q, TensorView k, TensorView v, TensorView out,
        TensorView mask = {}) = 0;

    // Linear attention (Kimi-style — no KV cache needed)
    virtual Result<void> linear_attention(
        TensorView q, TensorView k, TensorView v,
        TensorView out,
        TensorView kv_state, // Persistent state updated in-place
        float gamma = 0.95f) = 0;

    // Normalization
    virtual Result<void> layer_norm(TensorView in, TensorView weight,
                                    TensorView bias, TensorView out,
                                    float eps = 1e-6f) = 0;

    // Activation functions
    virtual Result<void> gelu(TensorView in, TensorView out) = 0;
    virtual Result<void> silu(TensorView in, TensorView out) = 0;

    // TTT operations
    virtual Result<void> ttt_sgd_step(TensorView params, TensorView grad,
                                       TensorView opt_state,
                                       float lr, float momentum) = 0;

    // mHC hyper-connection residual projection
    virtual Result<void> hyper_residual(TensorView skip, TensorView processed,
                                         TensorView out, float alpha,
                                         TensorView manifold_basis) = 0;

    // Engram operations (DeepSeek)
    virtual Result<void> engram_compress(TensorView kv_pairs,
                                          TensorView compressed,
                                          TensorView importance,
                                          size_t target_slots) = 0;

    // Execution
    virtual Result<void> synchronize() = 0;
    virtual Result<void> stream_enqueue(std::function<Result<void>()>) = 0;

    // Quantization
    virtual Result<Tensor> quantize(TensorView src, QuantMode mode) = 0;
    virtual Result<Tensor> dequantize(TensorView src, TensorView out) = 0;
};

// Backend factory
Result<std::unique_ptr<Backend>> create_backend(BackendType type,
                                                 DeviceConfig cfg = {});

} // namespace hesa
```

**Backend priority order** (for AUTO detection):
1. CUDA (NVIDIA GPUs present)
2. Metal (macOS with Apple Silicon)
3. CPU with AVX-512 (modern x86 servers)
4. CPU with AVX2 (x86 desktop/laptop)
5. CPU with NEON (ARM SoCs)
6. CPU fallback (scalar)

---

## 4. Model Format (HESF)

### 4.1 HESF Overview

HESA-LLM uses **HESF** (HESA Serialized Format) — a GGUF-compatible binary format extending GGML's tensor serialization with metadata for modern architectures.

Key design: HESF files are valid GGUF files (magic number `GGUF`), with custom metadata keys for HESA-specific features. This ensures GGUF tools continue to work with HESF models while enabling new capabilities.

### 4.2 File Structure

```
┌─────────────────────────────────────────────────────────────┐
│  HESF / GGUF Header                                          │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ Magic:    'GGUF' (4 bytes)                            │   │
│  │ Version:   3 (uint32, little-endian)                  │   │
│  │ n_tensors: total tensor count (uint64)                │   │
│  │ n_kv:      metadata entries (uint64)                  │   │
│  └───────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Metadata Key-Value Pairs                                    │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ Standard GGUF keys (general.architecture, etc.)        │   │
│  │ HESA-specific keys (hesa.* namespace):                │   │
│  │   hesa.use_neural_memory       : bool                 │   │
│  │   hesa.neural_memory_state_dim : uint32               │   │
│  │   hesa.use_ttt                 : bool                 │   │
│  │   hesa.ttt_hidden_dim          : uint32               │   │
│  │   hesa.ttt_lr                  : float32               │   │
│  │   hesa.use_hyper_connections   : bool                 │   │
│  │   hesa.hc_manifold_dim         : uint32               │   │
│  │   hesa.hc_alpha                : float32               │   │
│  │   hesa.linear_attn_ratio       : float32 (0.0-1.0)    │   │
│  │   hesa.engram_enabled          : bool                 │   │
│  │   hesa.engram_max_slots        : uint32               │   │
│  │   hesa.hybrid_gate_enabled     : bool                 │   │
│  │   hesa.quant_override          : string (Q4_K_M etc.) │   │
│  └───────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Tensor Alignment Padding (to 32-byte boundary)              │
├─────────────────────────────────────────────────────────────┤
│  Tensor Info Blocks                                          │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ For each tensor:                                       │   │
│  │   name:           string                               │   │
│  │   n_dims:         uint32                               │   │
│  │   dimensions[]:   uint64[n_dims]                       │   │
│  │   dtype:          uint32 (ggml_type)                   │   │
│  │   offset:         uint64 (from file start)             │   │
│  │   HESA tensor annotations:                             │   │
│  │     hesa.layer:   int32 (which transformer layer)       │   │
│  │     hesa.role:    string (attention/mlp/memory/etc.)    │   │
│  └───────────────────────────────────────────────────────┘   │
├─────────────────────────────────────────────────────────────┤
│  Tensor Data (contiguous, mmap-ready)                        │
│  ┌───────────────────────────────────────────────────────┐   │
│  │ All tensor weights in binary, packed by dtype           │   │
│  │ Accessible via mmap(offset, size) — zero-copy loading   │   │
│  └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 4.3 Quantization Support

HESF supports all standard GGUF quantization types plus two new modes:

| Code | Name | Bits/weight | Quality | HESA Notes |
|------|------|-------------|---------|------------|
| Q4_0 | Q4 0-order | 4.0 | Baseline | Standard |
| Q4_K_M | Q4 K-medium | 4.0 | Good | Recommended default |
| Q5_K_S | Q5 K-small | 5.0 | Better | For memory models |
| Q6_K | Q6 K-full | 6.0 | Best | For production |
| **Q3_T** | **Q3 TTT-optimized** | **3.0** | Good | **HESA-only: preserves TTT parameter precision** |
| **Q4_M** | **Q4 Memory-optimized** | **4.0** | Good | **HESA-only: higher precision for neural memory weights** |

### 4.4 Tensor Naming Convention

Tensors follow a hierarchical naming scheme:

```
blk.{layer}.{component}.{param}

Examples:
  blk.0.attn_norm.weight         — RMSNorm before attention
  blk.0.attn_q.weight            — Query projection
  blk.0.attn_k.weight            — Key projection
  blk.0.attn_v.weight            — Value projection
  blk.0.attn_o.weight            — Output projection
  blk.0.attn_linear_q.weight     — Linear attention Q (Kimi)
  blk.0.attn_linear_k.weight     — Linear attention K (Kimi)
  blk.0.attn_linear_v.weight     — Linear attention V (Kimi)
  blk.0.attn_hybrid_gate.weight  — Hybrid gate blend
  blk.0.ffn_gate.weight          — MoE gate or GELU projection
  blk.0.ffn_up.weight            — FFN up-projection
  blk.0.ffn_down.weight          — FFN down-projection
  blk.0.ffn_norm.weight          — RMSNorm before FFN
  blk.0.memory_norm.weight       — Norm before neural memory
  blk.0.memory_in.weight         — Neural memory input projection
  blk.0.memory_out.weight        — Neural memory output projection
  blk.0.memory_gate.weight       — Memory gate (blend factor)
  blk.0.ttt_proj.weight          — TTT layer projection
  blk.0.ttt_bias                 — TTT layer bias
  blk.0.hc_basis.weight          — mHC manifold basis vectors
  blk.0.hc_proj.weight           — mHC hyper-connection projection
  blk.0.hc_alpha                 — mHC residual scaling factor
  blk.0.engram_compress.weight   — Engram compression projection
  output_norm.weight             — Final norm
  output.weight                  — LM head (can be tied with embed)
  token_embd.weight              — Token embedding
```

---

## 5. Research Integration

This section maps each research paper to specific implementation modules in HESA-LLM.

### 5.1 Hybrid Attention Architecture (Overall)

The fundamental innovation is replacing pure softmax attention with a **hybrid attention** mechanism that blends:

- **Softmax attention** (standard scaled-dot-product with KV cache) — excellent for precise, token-level reasoning.
- **Linear attention** (Kimi-style recurrent) — excellent for long-range context, O(1) per-token inference, constant memory.

```
Layer output = (1-λ)·SoftmaxAttn(Q,K,V) + λ·LinearAttn(Q,K,V,State)
```

Where λ is a learned per-layer scalar (or per-head). This blend is data-dependent in production:
- Layers 0-8: λ ≈ 0.3 (mostly softmax)
- Layers 9-16: λ ≈ 0.5 (balanced)
- Layers 17+: λ ≈ 0.7 (mostly linear, captures long-range)

**Implementation**: `HybridAttentionModule` in `src/attention/hybrid_attn.cpp`

### 5.2 Kimi Linear Attention

**Paper**: Kimi linear attention replaces the softmax normalization in attention with a kernel function k(·) that allows linear-time computation. Instead of O(N²) attention matrix, it computes:

```
Attn(Q,K,V)_i = Σ_j φ(Q_i)ᵗφ(K_j)V_j / Σ_j φ(Q_i)ᵗφ(K_j)
```

Which decomposes to: persistent state S = Σ_j φ(K_j)V_j, and normalization Z = Σ_j φ(Q_i)ᵗφ(K_j).

**Integration points:**
- `Backend::linear_attention()` — fused kernel computing φ(K)ᵗφ(Q)V incrementally
- Persistent state stored as `Tensor kv_state_` [n_heads, d_k, d_v] per layer — constant size regardless of sequence length
- Feature map φ uses elu(x)+1 or softmax(x) as kernel function
- Enables effective context windows of 1M+ tokens at 1-4 MB memory overhead

**HESA implementation**: `src/attention/linear_attn.cpp`

### 5.3 Titans Neural Memory

**Paper**: Titans introduces a persistent neural memory that learns to store task-relevant information long-term via a learned gating mechanism:

```
m_t = (1 - g_t) ⊙ decay · m_{t-1} + g_t ⊙ f_θ(x_t)
```

Where g_t is a learned gate deciding what to write, decay is per-memory-slot forget rate, and f_θ is a parameterized write function.

**Integration points:**
- `NeuralMemory` class manages state independently of KV cache
- Neural memory survives context window sliding — it's not bounded by token count
- Memory output is projected back into the transformer residual stream
- During prefill, memory is updated sequentially; during generation, one step per token
- Adds ~1-4 MB persistent state per model (vs. ~100+ MB for 8K KV cache)

**HESA implementation**: `src/memory/neural_memory.cpp`

### 5.4 TTT-E2E (Test-Time Training)

**Paper**: TTT layers update their internal parameters during inference using self-supervised loss. Each token triggers a small optimization step on the layer's internal weights.

**Integration points:**
- `TTTLayer` sits between attention and feed-forward in transformer block
- Internal parameters (small MLP weights ~d_model^2) are updated via SGD during the forward pass
- Gradient is computed from self-supervised prediction error: predict next token representation
- Learning rate is a hyperparameter, typically 0.01-0.1
- TTT is frozen during KV cache building (no time for updates) and thawed during generation
- Enables in-context "learning" — the model adapts its internal state to the current context

**HESA implementation**: `src/memory/ttt_layer.cpp`

### 5.5 DeepSeek Engram

**Paper**: Engram introduces a compressed representation of KV cache that stores "important" key-value pairs in a compressed format, enabling long-context with smaller memory footprint.

**Integration points:**
- `EngramCache` compresses old KV pairs into fixed-size slots using learned compression
- Importance scores determine which KV pairs to compress (low-attention-value pairs first)
- Compressed representation can be decompressed on-demand (approximate recovery)
- Enables effective context extension: 32K tokens fit in 16K-slot KV cache + 8K engram slots
- Engram decompression is a lightweight matmul (faster than reading full KV cache)

**HESA implementation**: `src/kv_cache/engram_cache.cpp`

### 5.6 ByteDance Hyper Connections

**Paper**: Hyper connections replace traditional residual connections with learned bypass paths that can skip multiple layers:

```
y = x + α·Σ_{i=0}^{d} β_i · Layer_i(x)
```

Where α is a learnable global scale, β_i are per-layer blending coefficients.

**Integration points:**
- `HyperConnectionModule` adds bypass paths that skip 2-4 layers
- Enables smoother gradient flow (training) and more efficient information propagation (inference)
- During inference, allows early-exit: if residual change is below threshold, skip remaining layers
- Parameters stored as `blk.{layer}.hc_basis.weight` and `blk.{layer}.hc_alpha`

**HESA implementation**: `src/layers/hyper_connection.cpp`

### 5.7 mHC Manifold-Constrained Hyper Connections

**Paper**: mHC constrains hyper connections to a low-dimensional manifold, reducing parameters while maintaining expressivity:

```
y = x + α·Σ_j γ_j · P_j·x    where P_j projects onto manifold basis vector j
```

**Integration points:**
- mHC replaces plain hyper connections when `hc_manifold_dim < d_model`
- Manifold basis vectors (d_model × manifold_dim) are learned during training
- Projection P_j is outer product: P_j = b_j · b_jᵗ for basis vector b_j
- Reduces parameter count from O(d_model × d_model) to O(d_model × manifold_dim)
- Typical manifold_dim = 64-128 vs. d_model = 4096 → 32-64× parameter reduction

**HESA implementation**: Integrated into `src/layers/hyper_connection.cpp` with `ManifoldResidual` variant

### 5.8 Combined Architecture: The HESA Transformer Block

```
                    ┌─── Input x ────────────────────────────────────┐
                    │                                                 │
                    ▼                                                 │
               ┌─────────┐                                           │
               │ RMSNorm │                                           │
               └────┬────┘                                           │
                    │                                                 │
                    ▼                                                 │
            ┌───────────────┐                                         │
            │ Hybrid Attn   │                                         │
            │ (soft + lin)  │                                         │
            └───────┬───────┘                                         │
                    │                                                 │
                    ▼                                                 │
            ┌───────────────┐                                         │
            │ Neural Memory │  ← Titans persistent state              │
            └───────┬───────┘                                         │
                    │                                                 │
                    ▼                                                 │
            ┌───────────────┐                                         │
            │   TTT Layer   │  ← Test-time parameter update           │
            └───────┬───────┘                                         │
                    │                                                 │
                    ▼                                                 │
            ┌───────────────┐                                         │
            │  Engram Cache │  ← Compressed KV management              │
            └───────┬───────┘                                         │
                    │                                                 │
                    ▼                                                 │
                    │ ──── mHC Hyper Connection ───────────┐          │
                    │    y = x + α·Σ γ_j·P_j·x             │          │
                    │                                       │          │
                    ▼                                       ▼          │
               ┌─────────┐                              ┌────────┐    │
               │ RMSNorm │                              │  ADD   │    │
               └────┬────┘           ←──────────────────┘        │    │
                    │          (Hyper-connection bypass)          │    │
                    ▼                                            │    │
            ┌───────────────┐                                    │    │
            │  Gated MLP    │                                    │    │
            │  (GeGLU/SwiGLU)                                     │    │
            └───────┬───────┘                                    │    │
                    │                                            │    │
                    ▼                                            │    │
              ┌───────────┐                                      │    │
              │ mHC Hyper │                                     │    │
              │ Connection│ ──────────────────────────────────────┘    │
              └─────┬─────┘                                            │
                    │                                                  │
                    ▼                                                  │
              ┌───────────┐                                            │
              │  Output   │ ──────────> next layer input              │
              └───────────┘                                            │
```

---

## 6. Hardware Targets

### 6.1 Target Specifications

| Target | Hardware | VRAM/RAM | Precision | Max Model | Goal |
|--------|----------|----------|-----------|-----------|------|
| **M2 Mac** | Apple M2 24GB | 16GB available | Q4_K_M | 13B | > 100 tok/s gen |
| **RTX 3080** | NVIDIA RTX 3080 10GB | 8GB available | Q4_K_M | 7B | > 80 tok/s gen |
| **Tesla P40** | NVIDIA P40 24GB | 22GB available | Q5_K_M | 14B | > 60 tok/s gen |

### 6.2 Memory Budget

For a **7B model with full HESA features** (neural memory + TTT + Engram + mHC + hybrid attn):

| Component | Memory (Q4_K_M) | Notes |
|-----------|-----------------|-------|
| Model weights | 4.2 GB | 7B params @ 4 bit + metadata |
| KV cache (8K ctx) | 1.1 GB | Standard softmax attn portion |
| Linear attention state | 0.3 GB | Constant-size, per-layer |
| Neural memory | 0.1 GB | 256-dim × 16 heads |
| TTT parameters | 0.2 GB | Per-layer internal weights |
| TTT optimizer state | 0.2 GB | Momentum buffers |
| Engram cache | 0.3 GB | Compressed KV |
| Activations (runtime) | 0.5 GB | Per-layer temporaries |
| Overhead | 0.3 GB | Backend, alignment |
| **Total** | **7.2 GB** | Fits in RTX 3080 (8GB usable) |

### 6.3 Backend-Specific Optimizations

#### 6.3.1 CUDA (RTX 3080, Tesla P40)

- sm_75 (Turing) and sm_80 (Ampere) as minimum targets
- Use CUTLASS-style GEMM kernels for matmul
- Flash Attention 2 for softmax attention
- Linear attention implemented as batched matmul: no custom kernel needed
- CUDA Graphs for capturing execution graph (eliminate kernel launch overhead)
- Tensor Cores for FP16/BF16 compute where available

#### 6.3.2 Metal (M2 Mac)

- Metal Performance Shaders (MPS) for matmul, convolutions
- Custom Metal compute shaders for RoPE, RMSNorm, SwiGLU
- Unified memory: no explicit host/device transfer needed
- Threadgroup tiling for cache-friendly matrix ops
- 4-bit block quantization dequantized on-the-fly in shader

#### 6.3.3 CPU (x86_64 / ARM)

- **x86_64**: AVX2 as baseline, AVX-512 (VNNI) for 8-bit matmul
- **ARM NEON**: ARMv8.2+ with FP16 extensions (M-series has AMX-like ISA)
- Q4_0/Q4_K dequantization in SIMD (multiply-add in integer domain)
- Cache blocking: L2-aware tiling for matrices > cache size
- Thread pool: num_threads worker threads, work-stealing for dynamic load balancing
- Memory pinning: `madvise(MADV_HUGEPAGE)` on Linux, `vm_allocate` on macOS

### 6.4 Memory Hierarchy Strategy

```
                    ┌───────────────────────────────┐
                    │        Model Weights           │
                    │  - mmapped from .hesf file      │
                    │  - Zero-copy read via mmap      │
                    │  - GPU: pre-loaded at init      │
                    └───────────────┬───────────────┘
                                    │
               ┌────────────────────┼────────────────────┐
               │                    │                    │
               ▼                    ▼                    ▼
      ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
      │   VRAM/DRAM  │    │  GPU Memory  │    │  CPU Cache  │
      │              │    │  (weights)   │    │  (active)   │
      │ - Activations│    │ - Pre-fetched│    │  - Hot tiles│
      │ - KV cache   │    │   blocks     │    │  - Working  │
      │ - States     │    │ - Pinned     │    │    sets     │
      └──────────────┘    └──────────────┘    └──────────────┘
```

---

## 7. API Design

### 7.1 OpenAI-Compatible HTTP Server

HESA-LLM ships with a built-in HTTP server implementing the OpenAI Chat Completions API.

```
┌─────────────────────────────────────────────────┐
│                   HTTP Server                    │
│                                                  │
│  POST /v1/chat/completions                      │
│  POST /v1/completions                            │
│  GET  /v1/models                                 │
│  POST /v1/embeddings                             │
│  POST /v1/tokenize                               │
│  POST /v1/detokenize                             │
│  GET  /health                                    │
│  GET  /metrics                                   │
│                                                  │
│  Features:                                       │
│   - Streaming (Transfer-Encoding: chunked)        │
│   - JSON Schema response_format                   │
│   - JSON mode (response_format: {type: json_object})│
│   - Function calling (tools parameter)            │
│   - Logprobs                                      │
│   - N-best completions                            │
│   - Speculative decoding (server-side opt-in)     │
│   - Batch API (/v1/batch) — Phase 2              │
└─────────────────────────────────────────────────┘
```

#### 7.1.1 Request/Response Schema

```json
// POST /v1/chat/completions
{
  "model": "hesa-3b-v1",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain test-time training."}
  ],
  "temperature": 0.7,
  "top_p": 0.95,
  "max_tokens": 1024,
  "stream": true,
  "stream_options": {"include_usage": true},
  // HESA-specific extensions
  "hesa_extensions": {
    "neural_memory_persistent": true,
    "ttt_enabled": true,
    "speculative_draft_model": "hesa-0.5b-draft",
    "context_sliding_window": 8192
  }
}

// Response (streaming chunk)
{
  "id": "chatcmpl-hesa-001",
  "object": "chat.completion.chunk",
  "created": 1712000000,
  "model": "hesa-3b-v1",
  "choices": [{
    "index": 0,
    "delta": {"content": "Test-time training"},
    "logprobs": null,
    "finish_reason": null
  }],
  "usage": null
}
```

### 7.2 C API (FFI-friendly)

```c
// hesa.h — public C API for bindings

typedef struct hesa_engine hesa_engine_t;
typedef struct hesa_session hesa_session_t;
typedef struct hesa_model hesa_model_t;

// Lifecycle
hesa_model_t*   hesa_model_load(const char* path);
void            hesa_model_free(hesa_model_t* model);
hesa_engine_t*  hesa_engine_create(hesa_model_t* model,
                                   const hesa_engine_config* cfg);
void            hesa_engine_free(hesa_engine_t* engine);

// Inference
hesa_session_t* hesa_session_create(hesa_engine_t* engine);
void            hesa_session_free(hesa_session_t* session);
int             hesa_session_prompt(hesa_session_t* session,
                                    const char* prompt, size_t len);
int             hesa_session_sample(hesa_session_t* session,
                                    hesa_token_callback cb, void* userdata);

// Streaming
int             hesa_session_sample_stream(hesa_session_t* session,
                                           hesa_token_stream_cb cb,
                                           void* userdata);

// State management
int             hesa_session_save(hesa_session_t* sess,
                                  const char* path);
int             hesa_session_load(hesa_session_t* sess,
                                  const char* path);
int             hesa_session_clear(hesa_session_t* session);

// Speculative decoding
hesa_engine_t*  hesa_engine_speculative(hesa_engine_t* target,
                                        hesa_engine_t* draft);

// Error handling
const char*     hesa_last_error(void);
int             hesa_errno(void);
```

### 7.3 Python Bindings (Phase 3+)

```python
import hesa

model = hesa.Model("hesa-3b-v1.Q4_K_M.hesf")
engine = hesa.Engine(model, kv_cache_size=8192, neural_memory=True)
session = engine.create_session()

# Streaming generation
for token in session.stream("Explain neural memory:", temperature=0.7):
    print(token.text, end="", flush=True)

# Speculative decoding
draft = hesa.Model("hesa-0.5b-draft.Q4_K_M.hesf")
spec_engine = hesa.SpeculativeEngine(target=engine, draft=draft)
```

---

## 8. Build System

### 8.1 CMake Configuration

```cmake
# CMakeLists.txt (top-level)
cmake_minimum_required(VERSION 3.22)
project(hesa VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Options ───────────────────────────────────────────
option(HESA_AVX2    "Enable AVX2 backend"    ${HESA_X86})
option(HESA_AVX512  "Enable AVX-512 backend" OFF)
option(HESA_NEON    "Enable ARM NEON backend" ${HESA_ARM})
option(HESA_CUDA    "Enable CUDA backend"    OFF)
option(HESA_METAL   "Enable Metal backend"   OFF)
option(HESA_SERVER  "Build HTTP server"      ON)
option(HESA_TESTS   "Build test suite"       ON)
option(HESA_STATIC  "Static linking"         OFF)

# ─── Dependencies (fetched) ────────────────────────────
include(FetchContent)
FetchContent_Declare(
    json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    SYSTEM
)
FetchContent_MakeAvailable(json)

# httplib for server
FetchContent_Declare(
    httplib
    GIT_REPOSITORY https://github.com/yhirose/cpp-httplib.git
    GIT_TAG        v0.16.0
    SYSTEM
)
FetchContent_MakeAvailable(httplib)

# ─── Backend Detection ─────────────────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(HESA_X86 ON)
    if(HESA_AVX2)
        add_compile_options(-mavx2 -mfma)
    endif()
    if(HESA_AVX512)
        add_compile_options(-mavx512f -mavx512bw -mavx512vl -mavx512dq
                            -mavx512vnni -mavx512vbmi2)
    endif()
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm64|aarch64|ARM64")
    set(HESA_ARM ON)
    if(HESA_NEON)
        add_compile_options(-march=armv8.2-a+fp16+dotprod)
    endif()
endif()

if(HESA_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES "75;80;86;89" CACHE STRING "CUDA archs")
endif()

# ─── Core Library ──────────────────────────────────────
add_library(hesa_core STATIC
    src/tensor/tensor.cpp
    src/tokenizer/tokenizer.cpp
    src/tokenizer/bpe_tokenizer.cpp
    src/tokenizer/sp_tokenizer.cpp
    src/model/model.cpp
    src/model/hesf_format.cpp
    src/model/quantization.cpp
    src/attention/softmax_attn.cpp
    src/attention/linear_attn.cpp
    src/attention/hybrid_attn.cpp
    src/attention/flash_attn.cpp
    src/memory/kv_cache.cpp
    src/memory/neural_memory.cpp
    src/memory/ttt_layer.cpp
    src/memory/engram_cache.cpp
    src/layers/rms_norm.cpp
    src/layers/gated_mlp.cpp
    src/layers/hyper_connection.cpp
    src/layers/embedding.cpp
    src/layers/rope.cpp
    src/backend/backend.cpp
    src/backend/cpu_backend.cpp
    src/sampling/sampler.cpp
    src/sampling/logit_processor.cpp
    src/engine/engine.cpp
    src/engine/session.cpp
    src/engine/speculative.cpp
)

target_include_directories(hesa_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Platform-specific sources
if(HESA_X86)
    target_sources(hesa_core PRIVATE src/backend/cpu_x86_kernels.cpp)
endif()
if(HESA_ARM)
    target_sources(hesa_core PRIVATE src/backend/cpu_arm_kernels.cpp)
endif()
if(HESA_CUDA)
    target_sources(hesa_core PRIVATE
        src/backend/cuda_backend.cpp
        src/backend/cuda_kernels.cu
        src/backend/cuda_flash_attn.cu
    )
    set_target_properties(hesa_core PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()
if(HESA_METAL)
    target_sources(hesa_core PRIVATE src/backend/metal_backend.mm)
    target_compile_options(hesa_core PRIVATE -fobjc-arc)

    # Embed Metal shaders
    add_custom_command(
        OUTPUT ${CMAKE_BINARY_DIR}/hesa_shaders.metallib
        COMMAND xcrun -sdk macosx metal
                ${CMAKE_SOURCE_DIR}/src/backend/metal_kernels.metal
                -o ${CMAKE_BINARY_DIR}/hesa_shaders.metallib
        DEPENDS src/backend/metal_kernels.metal
        COMMENT "Compiling Metal shaders"
    )
    add_custom_target(hesa_shaders
        DEPENDS ${CMAKE_BINARY_DIR}/hesa_shaders.metallib
    )
    add_dependencies(hesa_core hesa_shaders)
endif()

# ─── CLI Tool ──────────────────────────────────────────
add_executable(hesa src/main.cpp src/cli/arg_parser.cpp)
target_link_libraries(hesa PRIVATE hesa_core nlohmann_json::nlohmann_json)

# ─── Server ────────────────────────────────────────────
if(HESA_SERVER)
    add_executable(hesa_server
        src/server/server.cpp
        src/server/openai_api.cpp
        src/server/chat_handler.cpp
    )
    target_link_libraries(hesa_server PRIVATE
        hesa_core
        nlohmann_json::nlohmann_json
        httplib
    )
endif()

# ─── Tools ─────────────────────────────────────────────
add_executable(hesa_convert
    tools/convert_gguf_to_hesf.cpp
    tools/converter.cpp
)
target_link_libraries(hesa_convert PRIVATE hesa_core nlohmann_json::nlohmann_json)

add_executable(hesa_quantize
    tools/quantize_model.cpp
)
target_link_libraries(hesa_quantize PRIVATE hesa_core)

# ─── Tests ─────────────────────────────────────────────
if(HESA_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# ─── Install ───────────────────────────────────────────
install(TARGETS hesa_core hesa hesa_server
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY include/ DESTINATION include)
install(FILES docs/hesf_format.md DESTINATION share/hesa)
```

### 8.2 Build Commands

```bash
# Default build (CPU only, server enabled)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# With CUDA backend
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DHESA_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=80
cmake --build build -j$(nproc)

# With Metal backend (macOS)
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DHESA_METAL=ON -DHESA_NEON=ON
cmake --build build -j$(nproc)

# Development build with sanitizers
cmake -B build -DCMAKE_BUILD_TYPE=Debug \
      -DCMAKE_CXX_FLAGS="-fsanitize=address,undefined"
cmake --build build -j$(nproc)
```

### 8.3 CI/CD Pipeline

```yaml
# .github/workflows/ci.yml
jobs:
  build:
    strategy:
      matrix:
        os: [ubuntu-22.04, macos-14, windows-2022]
        backend: [cpu]
    steps:
      - name: Build
        run: cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)
      - name: Test
        run: ctest --test-dir build -V
```

---

## 9. Phased Implementation Plan

### Phase 0: Foundation (Weeks 1-3)

**Goal**: Build core infrastructure that everything else depends on.

**Deliverables**:
- [ ] CMake build system with CPU backend
- [ ] Tensor abstraction with basic ops (matmul, add, softmax, RMSNorm)
- [ ] HESF model loader (read GGUF-compatible files)
- [ ] Tokenizer (BPE + SentencePiece)
- [ ] CLI tool for basic text generation (no model yet)
- [ ] Unit test framework (Catch2 or custom)
- [ ] CI pipeline

**Key files**: `src/tensor/`, `src/model/`, `CMakeLists.txt`

### Phase 1: Transformer Core (Weeks 4-8)

**Goal**: Run a standard Transformer model end-to-end.

**Deliverables**:
- [ ] Transformer layer: RMSNorm + Attention + FFN
- [ ] RoPE positional encoding
- [ ] SwiGLU/GeGLU gated MLP
- [ ] KV cache (ring buffer, sliding window)
- [ ] Logit sampling (temperature, top-k, top-p, repetition penalty)
- [ ] Model: convert and run Llama-3.1-8B equivalent
- [ ] Performance baseline: measure tok/s on all targets
- [ ] OpenAI-compatible server (basic `/v1/chat/completions`)

**Key files**: `src/attention/softmax_attn.cpp`, `src/layers/`, `src/engine/`, `src/server/`

### Phase 2: Quantization & Optimization (Weeks 9-12)

**Goal**: Production-level performance with small model size.

**Deliverables**:
- [ ] Q4_K_M, Q5_K_M, Q6_K, Q8_0 quantization types
- [ ] SIMD-optimized kernels (AVX2, NEON)
- [ ] CUDA backend with Flash Attention 2
- [ ] Metal backend (Apple Silicon)
- [ ] Memory profiling and optimization
- [ ] Model conversion tool (`hesa_quantize`)
- [ ] Speculative decoding (draft-verify)

**Key files**: `src/model/quantization.cpp`, `src/backend/cuda_kernels.cu`, `src/backend/metal_kernels.metal`

### Phase 3: Neural Memory & TTT (Weeks 13-18)

**Goal**: Implement research-backed features that differentiate HESA.

**Deliverables**:
- [ ] Hybrid attention module (softmax + linear blend)
- [ ] Kimi linear attention implementation
- [ ] Titans neural memory with gating
- [ ] TTT-E2E layers with SGD optimizer
- [ ] Engram cache for KV compression
- [ ] mHC hyper connections with manifold basis
- [ ] HESF format extensions for new tensors
- [ ] Benchmarks: compare against vanilla transformer

**Key files**: `src/memory/`, `src/attention/linear_attn.cpp`, `src/attention/hybrid_attn.cpp`

### Phase 4: Server & Ecosystem (Weeks 19-24)

**Goal**: Production-ready server with full OpenAI compatibility.

**Deliverables**:
- [ ] Full OpenAI API compatibility (tools, function calling, JSON mode)
- [ ] Multi-session concurrent inference
- [ ] Streaming with SSE
- [ ] Model registry and hot-reload
- [ ] Docker container
- [ ] Python bindings (pybind11)
- [ ] Benchmarking suite (`hesa-bench`)
- [ ] Documentation and tutorials

**Key files**: `src/server/`, `bindings/python/`, `docker/`

### Phase 5: Advanced Features (Weeks 25-30)

**Deliverables**:
- [ ] WebGPU backend (Phase 2 deferred feature)
- [ ] Distributed inference (tensor parallelism)
- [ ] LoRA/QLoRA adapter loading
- [ ] Embedding model support
- [ ] Reranker model support
- [ ] Speculative decoding with any draft model

### Milestone Summary

| Phase | Weeks | Focus | Model Support |
|-------|-------|-------|---------------|
| 0 | 1-3 | Infrastructure | None |
| 1 | 4-8 | Core Transformer | Llama-3.x, Mistral |
| 2 | 9-12 | Optimization | Quantized models |
| 3 | 13-18 | Research Features | HESA-native models |
| 4 | 19-24 | Production | Server, bindings |
| 5 | 25-30 | Advanced | MoE, distributed |

---

## 10. Directory Layout

```
hesa-llm/
├── CMakeLists.txt                    # Top-level build
├── README.md
├── docs/
│   ├── architecture.md               # This file
│   ├── hesf_format.md               # Model format specification
│   ├── api_reference.md              # C API documentation
│   └── benchmarks.md                 # Performance results
├── include/
│   └── hesa/                         # Public headers
│       ├── tensor.h
│       ├── model.h
│       ├── engine.h
│       ├── backend.h
│       ├── tokenizer.h
│       ├── session.h
│       └── sampling.h
├── src/
│   ├── main.cpp                      # CLI entry point
│   ├── tensor/                       # Tensor abstraction
│   │   ├── tensor.cpp
│   │   └── tensor_view.cpp
│   ├── model/                        # Model loading
│   │   ├── model.cpp                 # Model abstraction
│   │   ├── hesf_format.cpp           # HESF/GGUF parser
│   │   └── quantization.cpp          # Quant/dequant kernels
│   ├── tokenizer/                    # Text tokenization
│   │   ├── tokenizer.h
│   │   ├── bpe_tokenizer.cpp
│   │   └── sp_tokenizer.cpp
│   ├── attention/                    # Attention mechanisms
│   │   ├── softmax_attn.cpp          # Standard scaled-dot-product
│   │   ├── linear_attn.cpp           # Kimi linear attention
│   │   ├── hybrid_attn.cpp           # Softmax + Linear blend
│   │   └── flash_attn.cpp            # Flash Attention 2
│   ├── memory/                       # Memory systems
│   │   ├── kv_cache.cpp              # Standard KV cache
│   │   ├── neural_memory.cpp         # Titans persistent memory
│   │   ├── ttt_layer.cpp             # Test-time training
│   │   └── engram_cache.cpp          # DeepSeek Engram compression
│   ├── layers/                       # Transformer components
│   │   ├── rms_norm.cpp
│   │   ├── gated_mlp.cpp             # SwiGLU/GeGLU
│   │   ├── hyper_connection.cpp      # mHC hyper connections
│   │   ├── embedding.cpp
│   │   └── rope.cpp
│   ├── backend/                      # Compute backends
│   │   ├── backend.h                 # Abstract interface
│   │   ├── cpu_backend.cpp           # CPU (shared logic)
│   │   ├── cpu_x86_kernels.cpp       # AVX2/AVX-512 kernels
│   │   ├── cpu_arm_kernels.cpp       # NEON kernels
│   │   ├── cuda_backend.cpp          # CUDA backend
│   │   ├── cuda_kernels.cu           # CUDA kernels
│   │   ├── cuda_flash_attn.cu        # FA2 kernel
│   │   ├── metal_backend.mm          # Metal backend
│   │   └── metal_kernels.metal       # Metal shaders
│   ├── engine/                       # Inference orchestration
│   │   ├── engine.cpp                # Engine lifecycle
│   │   ├── session.cpp               # Session/context management
│   │   └── speculative.cpp           # Speculative decoding
│   ├── sampling/                     # Token sampling
│   │   ├── sampler.cpp
│   │   └── logit_processor.cpp
│   ├── server/                       # HTTP server
│   │   ├── server.cpp
│   │   ├── openai_api.cpp
│   │   └── chat_handler.cpp
│   └── cli/                          # CLI utilities
│       └── arg_parser.cpp
├── tools/
│   ├── convert_gguf_to_hesf.cpp      # GGUF → HESF converter
│   ├── quantize_model.cpp            # Model quantizer
│   └── converter.cpp
├── tests/
│   ├── CMakeLists.txt
│   ├── test_tensor.cpp
│   ├── test_attention.cpp
│   ├── test_memory.cpp
│   ├── test_tokenizer.cpp
│   ├── test_sampling.cpp
│   ├── test_quantization.cpp
│   └── test_integration.cpp
├── bindings/
│   └── python/                       # Python bindings (Phase 3+)
├── benchmarks/
│   ├── bench_inference.cpp           # Tok/s benchmarks
│   └── bench_memory.cpp              # Memory bandwidth tests
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **HESF** | HESA Serialized Format — GGUF-compatible model format |
| **Hybrid Attention** | Blend of softmax and linear attention per layer |
| **Neural Memory** | Persistent learned state (Titans architecture) |
| **TTT** | Test-Time Training — inference-time parameter updates |
| **Engram** | Compressed KV cache representation (DeepSeek) |
| **mHC** | Manifold-Constrained Hyper Connections |
| **KV Cache** | Key-Value cache for autoregressive attention |
| **RoPE** | Rotary Positional Embedding |
| **HESA** | Hierarchical External-State Augmented |

## Appendix B: References

1. **TTT-E2E**: *Test-Time Training as an End-to-End Architecture for In-Context Learning* (2024)
2. **Titans**: *Titans: Learning to Memorize at Test Time* — NeurIPS 2024
3. **Kimi Linear**: *Kimi k1.5: Scaling Reinforcement Learning with LLMs* — Linear Attention component
4. **DeepSeek Engram**: *Efficient Long-Context LLM with Compressed Memory Representations*
5. **ByteDance Hyper**: *Hyper Connections: Learning to Skip Layers in Deep Networks*
6. **mHC**: *Manifold-Constrained Hyper Connections for Efficient Residual Flow*

## Appendix C: Performance Targets by Hardware

### M2 Mac (24GB, M2 chip, 10-core GPU)

| Model | Quant | Prompt PPS | Gen tok/s | Memory |
|-------|-------|-----------|-----------|--------|
| 3B | Q4_K_M | 500 | 120 | 4.2 GB |
| 7B | Q4_K_M | 250 | 65 | 6.8 GB |
| 13B | Q4_K_M | 120 | 35 | 11.2 GB |
| 13B | Q3_T | 120 | 40 | 9.1 GB |

### RTX 3080 (10GB VRAM, CUDA 8.6)

| Model | Quant | Prompt PPS | Gen tok/s | Memory |
|-------|-------|-----------|-----------|--------|
| 3B | Q4_K_M | 1200 | 180 | 3.8 GB |
| 7B | Q4_K_M | 600 | 90 | 6.2 GB |
| 7B | Q5_K_M | 600 | 85 | 7.1 GB |
| 13B | Q4_K_M | 250 | 30 | OOM (need VRAM offload) |

### Tesla P40 (24GB VRAM, CUDA 6.1)

| Model | Quant | Prompt PPS | Gen tok/s | Memory |
|-------|-------|-----------|-----------|--------|
| 7B | Q4_K_M | 400 | 50 | 6.2 GB |
| 13B | Q4_K_M | 200 | 28 | 10.8 GB |
| 14B | Q5_K_M | 180 | 25 | 13.5 GB |

---

*End of Architecture Document — Version 0.1.0-draft*

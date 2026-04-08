# hesa-llm

**hesa-llm** is a modern, portable LLM inference engine in C++23, targeting Gemma-4 class performance in a smaller form factor.

## Goals

- **Efficiency**: High-quality inference with minimal resource consumption
- **Portability**: Linux, macOS, Windows -- single codebase, platform-specific backends
- **Simplicity**: Clean C++23 with modern CMake, minimal dependencies
- **Extensibility**: Modular architecture for new models, backends, and kernels

## Architecture

```
CLI (hesa-llm) → Engine → Transformer Blocks → Backend (NEON/AVX2/CUDA/Metal)
                 ↑                                    ↑
           GGUF Loader (mmap, zero-copy)        SIMD kernels (dequant + matvec)
           Tokenizer (BPE / SentencePiece)      KV Cache (ring buffer)
```

## Features

- C++23 codebase; FetchContent for optional deps (nlohmann/json)
- Cross-platform build (CMake 3.20+); gcc-14, clang support
- GGUF v3 full parser: all 13 metadata types, array types, tensor loading
- Zero-copy mmap-based weight loading (quantized tensors stay in-place)
- NEON SIMD (AArch64) with AVX2/AVX512 dispatch paths
- Dequantization: Q4_0, Q5_0, Q8_0, Q4_K, Q5_K, Q6_K
- Transformer components: RMSNorm, RoPE, SwiGLU FFN, attention (MHA + GQA)
- KV Cache with ring buffer and sliding-window eviction
- Tokenizer: BPE (vocab, merges, scores from GGUF metadata)
- Sampling: greedy, temperature, top-k, top-p, repetition penalty
- 42/42 unit tests passing

## Building

```bash
git clone https://github.com/HermestoAizales/hesa-llm.git
cd hesa-llm

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)

# Run (requires GGUF model file)
./build/hesa-llm -m model.gguf -p "Hello" -n 64
```

## Current Status

**Phase 1-2.x complete.** Core inference pipeline is built and passes all unit tests.

| Component | Status |
|-----------|--------|
| GGUF v3 Loader | Complete -- all metadata types, tensor info, mmap loading |
| Tensor + Views | Complete -- F32/F16/BF16 + all GGML quant types |
| SIMD Kernels | Complete -- NEON + AVX2/AVX512 dispatch |
| Dequant + Matvec | Complete -- Q4_0/5_0/8_0, Q4_K/5_K/6_K; Q5_0 scaling fixed (NaN issue resolved) |
| Transformer Block | Complete -- Attention, FFN, RoPE, RMSNorm |
| KV Cache | Complete -- ring buffer, sliding window |
| Tokenizer | Complete -- BPE from GGUF metadata |
| Engine | Complete -- generate loop, logits, sampling |
| CLI | Complete -- args, generation |
| **End-to-End Inference** | 10/10 complete, End-to-End functional (Debugging → DONE) |



## License

MIT

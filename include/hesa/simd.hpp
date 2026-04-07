#ifndef HESA_SIMD_HPP
#define HESA_SIMD_HPP

/**
 * SIMD acceleration layer for CPU backends.
 * Provides compile-time detection and runtime-dispatched kernels
 * for dot-product, matmul, RMSNorm, softmax, RoPE, and activation functions.
 *
 * Target backends:
 *   - ARM NEON (Apple M-series, Raspberry Pi, ARM servers)
 *   - x86_64 AVX2 (Intel Haswell+, AMD Zen+)
 *   - x86_64 AVX-512 (Intel Skylake-X+, AMD Zen4+)
 *   - Scalar fallback (portable C++)
 *
 * Usage:
 *   #include "hesa/simd.hpp"
 *   simd::dot_f32(a, b, n)  — auto-dispatches best kernel
 */

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <algorithm>

// ─── Compile-time SIMD detection ─────────────────────────────────

#if defined(__AVX512F__)
  #define HESA_SIMD AVX512
#elif defined(__AVX2__) && defined(__FMA__)
  #define HESA_SIMD AVX2_FMA
#elif defined(__AVX2__)
  #define HESA_SIMD AVX2
#elif defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
  #define HESA_SIMD NEON
#else
  #define HESA_SIMD SCALAR
#endif

// ─── Runtime CPU feature detection ───────────────────────────────

namespace hesa { namespace simd {

/** Runtime-detected SIMD level. Set at startup. */
enum class Level : int8_t {
    Scalar   = 0,
    AVX2     = 1,
    AVX2_FMA = 2,
    AVX512   = 3,
    NEON     = 4,
};

Level detect_level();
Level current_level();  // Returns the globally set level

/**
 * Set the SIMD level to use. Call once at startup.
 * A value of Level::Scalar forces scalar fallback even if SIMD is available.
 */
void set_level(Level level);

// ─── Dot Product: float * float -> float ────────────────────────

/** Compute dot(a, b) for n elements. */
float dot_f32(const float* a, const float* b, size_t n);

// ─── Fused Multiply-Add: out[i] = a[i] * b[i] + c[i] ─────────────

void fma_f32(const float* a, const float* b, const float* c, float* out, size_t n);

// ─── Fused Multiply-Add (single scalar c): out[i] = a[i] * b[i] + c ───

void fma_scalar_f32(const float* a, const float* b, float c, float* out, size_t n);

// ─── Element-wise: a[i] * b[i] ──────────────────────────────────

void mul_f32(const float* a, const float* b, float* out, size_t n);

// ─── Element-wise: a[i] + b[i] ─────────────────────────────────

void add_f32(const float* a, const float* b, float* out, size_t n);

// ─── Element-wise: a[i] + s ─────────────────────────────────────

void add_scalar_f32(const float* a, float s, float* out, size_t n);

// ─── RMSNorm ─────────────────────────────────────────────────────

/**
 * RMS-normalize: out[i] = (in[i] / sqrt(mean(in^2) + eps)) * weight[i]
 * Input: n_rows × row_size matrix.
 */
void rms_norm_f32(const float* in, const float* weight, float* out,
                  size_t n_rows, size_t row_size, float eps);

// ─── Softmax ─────────────────────────────────────────────────────

/**
 * Softmax: out[i] = exp(in[i] - max) / sum(exp(...))
 * Applied per row for shape [n_rows, row_size].
 */
void softmax_f32(const float* in, float* out,
                 size_t n_rows, size_t row_size);

// ─── SiLU / Sigmoid-Linear Unit ──────────────────────────────────

/** out[i] = in[i] * sigmoid(in[i]) = in[i] / (1 + exp(-in[i])) */
void silu_f32(const float* in, float* out, size_t n);

// ─── GELU ────────────────────────────────────────────────────────

/** Approximate GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3))) */
void gelu_f32(const float* in, float* out, size_t n);

// ─── RoPE (single position, single head, n_dims must be even) ────

/**
 * Apply RoPE rotation to vec of length n_dims in-place.
 * Uses precomputed cos/sin tables indexed by position and dimension pair.
 */
void rope_apply(float* vec, int32_t position, int n_dims,
                const float* cos_table, const float* sin_table,
                int freq_stride);

// ─── Dequantization ──────────────────────────────────────────────

/**
 * Dequantize Q4_0 block into 32 float values.
 * Block layout: [d: float16][m: float16][16 bytes packed weights]
 */
void dequantize_q4_0(const uint8_t* block, float* out, int n_blocks);

/**
 * Dequantize Q8_0 block into 32 float values.
 * Block layout: [d: float16][32 bytes int8 weights]
 */
void dequantize_q8_0(const uint8_t* block, float* out, int n_blocks);

/**
 * Dequantize Q4_K block into 256 float values.
 * Block layout: [scales(6 bytes)][mins(6 bytes)][128 bytes packed weights]
 */
void dequantize_q4_k(const uint8_t* block, float* out, int n_blocks);

/**
 * Dequantize Q5_K block into 256 float values.
 * Block layout: [scales(6)] [mins(6)] [qs(128)] [qh(32)]
 */
void dequantize_q5_k(const uint8_t* block, float* out, int n_blocks);

/**
 * Dequantize Q6_K block into 256 float values.
 * Block layout: [scales(16)] [ql(128)] [qh(64)] [8 int8 scales]
 */
void dequantize_q6_k(const uint8_t* block, float* out, int n_blocks);

// ─── Quantization ────────────────────────────────────────────────

/**
 * Quantize 32 float values into a Q4_0 block.
 * out must be at least 18 bytes per block.
 */
void quantize_q4_0(const float* src, uint8_t* out, int n_blocks);

/**
 * Quantize 32 float values into a Q8_0 block.
 * out must be at least 34 bytes per block.
 */
void quantize_q8_0(const float* src, uint8_t* out, int n_blocks);

// ─── Matrix-vector multiply (optimized) ──────────────────────────

/**
 * Compute y = A @ x + y, where A is [M, N] stored row-major.
 * A and x are F32.
 */
void matvec_f32(const float* A, const float* x, float* y, int M, int N);

/**
 * Dequantize A (quantized) and compute y = dequant(A) @ x + y.
 * Supports Q4_0, Q4_1, Q5_0, Q5_1, Q8_0.
 */
void matvec_dequant(const uint8_t* A_q, const float* x, float* y,
                    int M, int N, int block_size, int qtype);

// ─── Inline helpers ──────────────────────────────────────────────

static inline float fp16_to_f32(uint16_t h) {
    uint32_t sign = (h >> 15) & 0x1;
    int32_t exp   = ((h >> 10) & 0x1F);
    uint32_t frac = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        f = (sign << 31) | (frac << 13);
    } else if (exp == 0x1F) {
        f = (sign << 31) | (0xFF << 23) | (frac << 13);
    } else {
        f = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
    }
    float result;
    __builtin_memcpy(&result, &f, 4);
    return result;
}

static inline uint16_t f32_to_fp16(float f) {
    uint32_t fbits;
    __builtin_memcpy(&fbits, &f, 4);
    uint32_t sign = (fbits >> 31) & 0x1;
    int32_t exp   = ((fbits >> 23) & 0xFF) - 127;
    uint32_t frac = fbits & 0x7FFFFF;
    uint16_t h;
    if (exp <= -15) {
        h = static_cast<uint16_t>(sign << 15);
    } else if (exp >= 16) {
        h = static_cast<uint16_t>((sign << 15) | (0x1F << 10) | (frac ? 0x200 : 0));
    } else {
        h = static_cast<uint16_t>((sign << 15) | ((exp + 15) << 10) | (frac >> 13));
    }
    return h;
}

}}  // namespace hesa::simd

#endif  // HESA_SIMD_HPP

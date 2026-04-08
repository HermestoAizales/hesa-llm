#include "hesa/simd.hpp"

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>
#endif
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>

namespace hesa { namespace simd {

// ─── Global SIMD level ────────────────────────────────────────

static Level g_simd_level = Level::Scalar;

Level detect_level() {
#if HESA_SIMD == SCALAR
    return Level::Scalar;
#elif HESA_SIMD == NEON
    return Level::NEON;
#elif HESA_SIMD == AVX512
    return Level::AVX512;
#elif HESA_SIMD == AVX2_FMA
    return Level::AVX2_FMA;
#elif HESA_SIMD == AVX2
    return Level::AVX2;
#else
    return Level::Scalar;
#endif
}

Level current_level() { return g_simd_level; }
void set_level(Level level) { g_simd_level = level; }

#ifndef HESA_SIMD_LEVEL
#define HESA_SIMD_LEVEL SCALAR
#endif

// ─── DOT PRODUCT ──────────────────────────────────────────────

float dot_f32(const float* a, const float* b, size_t n) {
#if HESA_SIMD_LEVEL == NEON
    float32x4_t acc = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vfmaq_f32(acc, va, vb);
    }
    float sum = vaddvq_f32(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;

#elif HESA_SIMD_LEVEL == AVX512
    __m512 acc = _mm512_setzero_ps();
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        acc = _mm512_fmadd_ps(va, vb, acc);
    }
    float sum = _mm512_reduce_add_ps(acc);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;

#elif HESA_SIMD_LEVEL == AVX2_FMA
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_fmadd_ps(va, vb, acc);
    }
        __m128 hi_ = _mm256_extractf128_ps(acc, 1);
    __m128 lo_ = _mm256_castps256_ps128(acc);
    lo_ = _mm_add_ps(lo_, hi_);
    lo_ = _mm_add_ps(lo_, _mm_movehl_ps(lo_, lo_));
    lo_ = _mm_add_ss(lo_, _mm_movehl_ps(lo_, lo_));
    float sum = _mm_cvtss_f32(lo_);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;

#elif HESA_SIMD_LEVEL == AVX2
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 lo = _mm256_castps256_ps128(acc);
    lo = _mm_add_ps(lo, hi);
    hi = _mm_movehl_ps(lo, lo);
    lo = _mm_add_ps(lo, hi);
    lo = _mm_add_ss(lo, _mm_movehl_ps(lo, lo));
    float sum = _mm_cvtss_f32(lo);
    for (; i < n; ++i) sum += a[i] * b[i];
    return sum;

#else
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
#endif
}

// ─── FMA ──────────────────────────────────────────────────────

void fma_f32(const float* a, const float* b, const float* c, float* out, size_t n) {
#if HESA_SIMD_LEVEL == NEON
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t vc = vld1q_f32(c + i);
        vst1q_f32(out + i, vfmaq_f32(vc, va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i] + c[i];

#elif HESA_SIMD_LEVEL == AVX512
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        __m512 vc = _mm512_loadu_ps(c + i);
        _mm512_storeu_ps(out + i, _mm512_fmadd_ps(va, vb, vc));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i] + c[i];

#elif HESA_SIMD_LEVEL == AVX2_FMA
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(va, vb, vc));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i] + c[i];

#elif HESA_SIMD_LEVEL == AVX2
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 vc = _mm256_loadu_ps(c + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(vc, _mm256_mul_ps(va, vb)));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i] + c[i];

#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i] + c[i];
#endif
}

// ─── MUL ──────────────────────────────────────────────────────

void mul_f32(const float* a, const float* b, float* out, size_t n) {
#if HESA_SIMD_LEVEL == NEON
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vmulq_f32(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i];

#elif HESA_SIMD_LEVEL == AVX512
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(out + i, _mm512_mul_ps(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i];

#elif HESA_SIMD_LEVEL == AVX2 || HESA_SIMD_LEVEL == AVX2_FMA
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i];

#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i];
#endif
}

// ─── ADD ──────────────────────────────────────────────────────

void add_f32(const float* a, const float* b, float* out, size_t n) {
#if HESA_SIMD_LEVEL == NEON
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vaddq_f32(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] + b[i];

#elif HESA_SIMD_LEVEL == AVX512
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(out + i, _mm512_add_ps(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] + b[i];

#elif HESA_SIMD_LEVEL == AVX2 || HESA_SIMD_LEVEL == AVX2_FMA
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] + b[i];

#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + b[i];
#endif
}

// ─── ADD SCALAR ───────────────────────────────────────────────

void add_scalar_f32(const float* a, float s, float* out, size_t n) {
#if HESA_SIMD_LEVEL == NEON
    float32x4_t vs = vdupq_n_f32(s);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        vst1q_f32(out + i, vaddq_f32(va, vs));
    }
    for (; i < n; ++i) out[i] = a[i] + s;

#elif HESA_SIMD_LEVEL == AVX512
    __m512 vs = _mm512_set1_ps(s);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        _mm512_storeu_ps(out + i, _mm512_add_ps(va, vs));
    }
    for (; i < n; ++i) out[i] = a[i] + s;

#elif HESA_SIMD_LEVEL == AVX2 || HESA_SIMD_LEVEL == AVX2_FMA
    __m256 vs = _mm256_set1_ps(s);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        _mm256_storeu_ps(out + i, _mm256_add_ps(va, vs));
    }
    for (; i < n; ++i) out[i] = a[i] + s;

#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] + s;
#endif
}

// ─── RMSNorm ──────────────────────────────────────────────────

void rms_norm_f32(const float* in, const float* weight, float* out,
                  size_t n_rows, size_t row_size, float eps) {
    for (size_t r = 0; r < n_rows; ++r) {
        const float* row = in + r * row_size;
        float* o = out + r * row_size;

        // Compute sum of squares
        float ss = 0.0f;
        for (size_t j = 0; j < row_size; ++j) ss += row[j] * row[j];
        float inv_rms = 1.0f / std::sqrt(ss / row_size + eps);

#if HESA_SIMD_LEVEL == NEON
        float32x4_t v_inv = vdupq_n_f32(inv_rms);
        size_t j = 0;
        for (; j + 4 <= row_size; j += 4) {
            float32x4_t vi = vld1q_f32(row + j);
            float32x4_t vw = vld1q_f32(weight + j);
            vst1q_f32(o + j, vmulq_f32(vi, vmulq_f32(vw, v_inv)));
        }
        for (; j < row_size; ++j) o[j] = row[j] * inv_rms * weight[j];

#elif HESA_SIMD_LEVEL >= AVX2
        __m256 v_inv = _mm256_set1_ps(inv_rms);
        size_t j = 0;
        for (; j + 8 <= row_size; j += 8) {
            __m256 vi = _mm256_loadu_ps(row + j);
            __m256 vw = _mm256_loadu_ps(weight + j);
            _mm256_storeu_ps(o + j, _mm256_mul_ps(vi, _mm256_mul_ps(vw, v_inv)));
        }
        for (; j < row_size; ++j) o[j] = row[j] * inv_rms * weight[j];

#else
        for (size_t j = 0; j < row_size; ++j)
            o[j] = row[j] * inv_rms * weight[j];
#endif
    }
}

// ─── Softmax ──────────────────────────────────────────────────

void softmax_f32(const float* in, float* out, size_t n_rows, size_t row_size) {
    for (size_t r = 0; r < n_rows; ++r) {
        const float* row = in + r * row_size;
        float* o = out + r * row_size;

        float max_val = row[0];
        for (size_t j = 1; j < row_size; ++j)
            if (row[j] > max_val) max_val = row[j];

        float sum = 0.0f;
        for (size_t j = 0; j < row_size; ++j) {
            o[j] = std::exp(row[j] - max_val);
            sum += o[j];
        }
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (size_t j = 0; j < row_size; ++j) o[j] *= inv;
        }
    }
}

// ─── SiLU ─────────────────────────────────────────────────────

void silu_f32(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float x = in[i];
        out[i] = x / (1.0f + std::exp(-x));
    }
}

// ─── GELU ─────────────────────────────────────────────────────

void gelu_f32(const float* in, float* out, size_t n) {
    const float k = static_cast<float>(std::sqrt(2.0 / 3.14159265358979323846));
    const float c = 0.044715f;
    for (size_t i = 0; i < n; ++i) {
        float x = in[i];
        float t = std::tanh(k * (x + c * x * x * x));
        out[i] = 0.5f * x * (1.0f + t);
    }
}

// ─── RoPE ─────────────────────────────────────────────────────

void rope_apply(float* vec, int32_t position, int n_dims,
                const float* cos_table, const float* sin_table,
                int freq_stride) {
    int pairs = n_dims / 2;
    int base = position * freq_stride;
    for (int i = 0; i < pairs; ++i) {
        float q0 = vec[2 * i];
        float q1 = vec[2 * i + 1];
        float c = cos_table[base + i];
        float s = sin_table[base + i];
        vec[2 * i]     = q0 * c - q1 * s;
        vec[2 * i + 1] = q0 * s + q1 * c;
    }
}

// ─── FMA with scalar ──────────────────────────────────────────

void fma_scalar_f32(const float* a, const float* b, float c, float* out, size_t n) {
#if HESA_SIMD_LEVEL == NEON
    float32x4_t vs = vdupq_n_f32(c);
    size_t i = 0;
    for (; i + 4 <= n; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        vst1q_f32(out + i, vfmaq_f32(vs, va, vb));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i] + c;

#elif HESA_SIMD_LEVEL == AVX512
    __m512 vs = _mm512_set1_ps(c);
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        __m512 va = _mm512_loadu_ps(a + i);
        __m512 vb = _mm512_loadu_ps(b + i);
        _mm512_storeu_ps(out + i, _mm512_fmadd_ps(va, vb, vs));
    }
    for (; i < n; ++i) out[i] = a[i] * b[i] + c;

#elif HESA_SIMD_LEVEL == AVX2 || HESA_SIMD_LEVEL == AVX2_FMA
    __m256 vs = _mm256_set1_ps(c);
    size_t i = 0;
    for (; i + 8 <= n; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
#if HESA_SIMD_LEVEL == AVX2_FMA
        _mm256_storeu_ps(out + i, _mm256_fmadd_ps(va, vb, vs));
#else
        _mm256_storeu_ps(out + i, _mm256_add_ps(vs, _mm256_mul_ps(va, vb)));
#endif
    }
    for (; i < n; ++i) out[i] = a[i] * b[i] + c;

#else
    for (size_t i = 0; i < n; ++i) out[i] = a[i] * b[i] + c;
#endif
}

// ─── Dequantization: Q4_0 ─────────────────────────────────────

void dequantize_q4_0(const uint8_t* block, float* out, int n_blocks) {
    // Block: [d:2][m:2][4-bit weights:16] = 18 bytes, 32 elements
    for (int b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = block + b * 18;
        uint16_t dh, mh;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&mh, blk + 2, 2);
        float d = fp16_to_f32(dh);
        float m = fp16_to_f32(mh);

        float* ob = out + b * 32;
        for (int i = 0; i < 16; ++i) {
            uint8_t q = blk[4 + i];
            int8_t lo = static_cast<int8_t>(q & 0xF) - 8;
            int8_t hi  = static_cast<int8_t>((q >> 4) & 0xF) - 8;
            ob[2 * i]     = d * lo + m;
            ob[2 * i + 1] = d * hi + m;
        }
    }
}

// ─── Dequantization: Q8_0 ─────────────────────────────────────

void dequantize_q8_0(const uint8_t* block, float* out, int n_blocks) {
    // Block: [d:2][int8 weights:32] = 34 bytes, 32 elements
    for (int b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = block + b * 34;
        uint16_t dh;
        std::memcpy(&dh, blk, 2);
        float d = fp16_to_f32(dh);

        const int8_t* w = reinterpret_cast<const int8_t*>(blk + 2);
        float* ob = out + b * 32;
        for (int i = 0; i < 32; ++i) {
            ob[i] = d * static_cast<float>(w[i]);
        }
    }
}

// ─── Dequantization: Q4_K ─────────────────────────────────────

// Extract 6-bit scale/min from the 12-byte K-scale block (llama.cpp get_scale_min_k4).
// The 12-byte layout: bytes 0-3 = scale low 6 bits (s0-s3),
// bytes 4-7 = min low 6 bits (m0-m3), bytes 8-9 = upper 2 bits for s4-s7 / m4-m7,
// bytes 10-11 = upper 2 bits for s4-s7 / m4-m7.
static inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63u;        // s0-s3 or m0-m3: low 6 bits
        *m = q[j + 4] & 63u;    // mins are at +4 offset
    } else {
        *d = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6u) << 4u);  // upper 2 bits from q[j-4] >> 6
        *m = (q[j + 4] >>  4u) | ((q[j + 0] >> 6u) << 4u);  // high nibble of same byte
    }
}

void dequantize_q4_k(const uint8_t* block, float* out, int n_blocks) {
    // Q4_K: [d:2][dmin:2][scales_k:12][qs:128] = 144 bytes, 256 elements
    // 4 super-blocks of 64 elements each (2x32 sub-groups).
    // Each super-block uses 2 scale/min pairs and 32 weight bytes.
    // llama.cpp reference: ggml/src/ggml-quants.c dequantize_row_q4_K
    for (int b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = block + b * 144;
        uint16_t dh, dmh;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&dmh, blk + 2, 2);
        float d    = fp16_to_f32(dh);
        float dmin = fp16_to_f32(dmh);

        const uint8_t* scales = blk + 4;   // 12 bytes packed scale+min
        const uint8_t* q      = blk + 16;  // 128 bytes packed weights
        float* ob = out + b * 256;

        for (int sb = 0; sb < 4; ++sb) {  // 4 super-blocks, 64 elements each
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(2 * sb,     scales, &sc0, &m0);
            get_scale_min_k4(2 * sb + 1, scales, &sc1, &m1);

            float d1 = d * sc0;
            float dm1 = dmin * m0;
            float d2 = d * sc1;
            float dm2 = dmin * m1;

            for (int l = 0; l < 32; ++l) {
                ob[0]  = d1 * static_cast<float>(q[l] & 0xFu) - dm1;  // low nibble → group 0
                ob[32] = d2 * static_cast<float>(q[l] >>  4u) - dm2;  // high nibble → group 1
                ++ob;
            }
            ob  += 32;
            q   += 32;
        }
    }
}

// ─── Dequantization: Q5_K ─────────────────────────────────────

void dequantize_q5_k(const uint8_t* block, float* out, int n_blocks) {
    // Q5_K: [d:2][dmin:2][scales_k:12][qs:128][qh:32] = 176 bytes, 256 elements
    // 5-bit weights = 4 low bits in qs + 1 high bit in qh (bit-per-element).
    // llama.cpp reference: ggml/src/ggml-quants.c dequantize_row_q5_K
    for (int b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = block + b * 176;
        uint16_t dh, dmh;
        std::memcpy(&dh, blk, 2);
        std::memcpy(&dmh, blk + 2, 2);
        float d    = fp16_to_f32(dh);
        float dmin = fp16_to_f32(dmh);

        const uint8_t* scales = blk + 4;   // 12 bytes packed scale+min
        const uint8_t* ql     = blk + 16;  // 128 bytes 4-bit low nibbles
        const uint8_t* qh     = blk + 144; // 32 bytes high bit per element

        float* ob = out + b * 256;

        uint8_t u1 = 1, u2 = 2;  // qh bit masks, shift by 2 per super-block
        for (int sb = 0; sb < 4; ++sb) {  // 4 super-blocks, 64 elements each
            uint8_t sc0, m0, sc1, m1;
            get_scale_min_k4(2 * sb,     scales, &sc0, &m0);
            get_scale_min_k4(2 * sb + 1, scales, &sc1, &m1);

            float d1 = d * sc0;
            float dm1 = dmin * m0;
            float d2 = d * sc1;
            float dm2 = dmin * m1;

            for (int l = 0; l < 32; ++l) {
                uint8_t w1 = (ql[l] & 0xFu) + (qh[l] & u1 ? 16u : 0u);
                uint8_t w2 = (ql[l]  >>  4u) + (qh[l] & u2 ? 16u : 0u);
                ob[0]  = d1 * static_cast<float>(w1) - dm1;
                ob[32] = d2 * static_cast<float>(w2) - dm2;
                ++ob;
            }
            ob  += 32;
            ql  += 32;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

// ─── Dequantization: Q6_K ─────────────────────────────────────

void dequantize_q6_k(const uint8_t* block, float* out, int n_blocks) {
    // Q6_K: [ql:128][qh:64][scales:16][d:2] = 210 bytes, 256 elements
    // llama.cpp: ggml/src/ggml-quants.c dequantize_row_q6_K
    // Block struct order: ql[128], qh[64], scales[16], d[2]
    // 6-bit values: q = (low4 | (high2 << 4)) - 32,  value = d * scale * q

    for (int b = 0; b < n_blocks; ++b) {
        const uint8_t* blk = block + b * 210;
        const uint8_t* ql  = blk;            // 128 bytes, low 4 bits
        const uint8_t* qh  = blk + 128;      // 64 bytes, high 2 bits
        const int8_t*  sc  = reinterpret_cast<const int8_t*>(blk + 192); // 16 bytes

        uint16_t dh;
        std::memcpy(&dh, blk + 208, 2);     // d at offset 208
        float d = fp16_to_f32(dh);

        float* y = out + b * 256;

        for (int n = 0; n < 2; ++n) {    // 2 chunks of 128 elements
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;         // index within 0..1 for each pair

                // Reconstruct 6-bit values from ql + qh, then subtract 32
                const int8_t q1 = static_cast<int8_t>(
                    (ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3u) << 4)) - 32;
                const int8_t q2 = static_cast<int8_t>(
                    (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3u) << 4)) - 32;
                const int8_t q3 = static_cast<int8_t>(
                    (ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3u) << 4)) - 32;
                const int8_t q4 = static_cast<int8_t>(
                    (ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3u) << 4)) - 32;

                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// ─── Quantization: F32 → Q4_0 ─────────────────────────────────

void quantize_q4_0(const float* src, uint8_t* out, int n_blocks) {
    // Q4_0: [d: fp16][m: fp16][16 bytes packed] = 18 bytes, 32 elements per block
    for (int b = 0; b < n_blocks; ++b) {
        const float* s = src + b * 32;
        uint8_t* blk  = out + b * 18;

        // Find max abs value and compute scale
        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float abs_v = std::abs(s[i]);
            if (abs_v > amax) amax = abs_v;
        }
        float d = amax / 7.0f;  // 4-bit signed: [-8, 7]
        uint16_t dh = f32_to_fp16(d);
        std::memcpy(blk, &dh, 2);
        // Q4_0 has no independent offset (m = -d * 0 effectively)
        // So we set offset to -8*d which means m = 0
        float m = 0.0f;
        uint16_t mh = f32_to_fp16(m);
        std::memcpy(blk + 2, &mh, 2);

        // Pack 4-bit values
        if (d > 0.0f) {
            float id = 1.0f / d;
            for (int i = 0; i < 16; ++i) {
                int8_t lo = static_cast<int8_t>(nearbyintf(s[2 * i]     * id) + 8);
                int8_t hi = static_cast<int8_t>(nearbyintf(s[2 * i + 1] * id) + 8);
                lo = std::max(int8_t(0), std::min(int8_t(15), lo));
                hi = std::max(int8_t(0), std::min(int8_t(15), hi));
                blk[4 + i] = static_cast<uint8_t>(lo) | (static_cast<uint8_t>(hi) << 4);
            }
        } else {
            std::memset(blk + 4, 0x88, 16);
        }
    }
}

// ─── Quantization: F32 → Q8_0 ─────────────────────────────────

void quantize_q8_0(const float* src, uint8_t* out, int n_blocks) {
    // Q8_0: [d: fp16][32 bytes int8 weights] = 34 bytes, 32 elements per block
    for (int b = 0; b < n_blocks; ++b) {
        const float* s = src + b * 32;
        uint8_t* blk  = out + b * 34;

        float amax = 0.0f;
        for (int i = 0; i < 32; ++i) {
            float abs_v = std::abs(s[i]);
            if (abs_v > amax) amax = abs_v;
        }
        float d = amax / 127.0f;  // 8-bit signed: [-128, 127]
        uint16_t dh = f32_to_fp16(d);
        std::memcpy(blk, &dh, 2);

        if (d > 0.0f) {
            float id = 1.0f / d;
            int8_t* w = reinterpret_cast<int8_t*>(blk + 2);
            for (int i = 0; i < 32; ++i) {
                w[i] = static_cast<int8_t>(std::max(-128.0f, std::min(127.0f, nearbyintf(s[i] * id))));
            }
        } else {
            std::memset(blk + 2, 0, 32);
        }
    }
}

// Block byte sizes for quantization types
static inline int quant_block_bytes(int qtype) {
    switch (qtype) {
        case 2:  return 18;   // Q4_0
        case 3:  return 20;   // Q4_1
        case 6:  return 22;   // Q5_0
        case 7:  return 24;   // Q5_1
        case 8:  return 34;   // Q8_0
        case 12: return 144;  // Q4_K
        case 13: return 176;  // Q5_K
        case 14: return 210;  // Q6_K
        default: return 32 * 4; // F32 fallback
    }
}

static inline int quant_block_size(int qtype) {
    switch (qtype) {
        case 2: case 3: case 6: case 7: case 8: return 32;
        case 12: case 13: case 14: return 256;
        default: return 1;
    }
}

void matvec_dequant(const uint8_t* A_q, const float* x, float* y,
                    int M, int N, int block_size, int qtype) {
    // Fused dequantize + matrix-vector multiply.
    // A_q: quantized weight matrix [M, N] laid out as blocks.
    // x:   input vector [N]
    // y:   output vector [M], accumulated (y += A @ x).
    int blck = block_size > 0 ? block_size : quant_block_size(qtype);
    int bpb = quant_block_bytes(qtype);
    int n_blocks_per_row = (N + blck - 1) / blck;

    for (int m = 0; m < M; ++m) {
        float dot = 0.0f;
        for (int bi = 0; bi < n_blocks_per_row; ++bi) {
            const uint8_t* blk = A_q + (m * n_blocks_per_row + bi) * bpb;
            int x_start = bi * blck;
            int x_len = std::min(blck, N - x_start);

            // Dequantize single block into vals[]
            if (qtype == 2) { // Q4_0: [d:2][m:2][w:16] = 18 bytes, 32 elements
                uint16_t dh, mh;
                std::memcpy(&dh, blk, 2);
                std::memcpy(&mh, blk + 2, 2);
                float d = fp16_to_f32(dh);
                float msc = fp16_to_f32(mh);
                for (int k = 0; k < 16; ++k) {
                    uint8_t q = blk[4 + k];
                    float lo = d * (static_cast<int8_t>(q & 0xF) - 8) + msc;
                    float hi = d * (static_cast<int8_t>(q >> 4) - 8) + msc;
                    if (2*k     < x_len) dot += lo * x[x_start + 2*k];
                    if (2*k + 1 < x_len) dot += hi * x[x_start + 2*k + 1];
                }
            } else if (qtype == 8) { // Q8_0: [d:2][w:32] = 34 bytes, 32 elements
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                float d = fp16_to_f32(dh);
                for (int k = 0; k < std::min(32, x_len); ++k) {
                    int8_t w = blk[2 + k];
                    dot += (d * static_cast<float>(w)) * x[x_start + k];
                }
            } else if (qtype == 6) { // Q5_0: [d:2][ql:16][qh:4] = 22 bytes, 32 elements
                uint16_t dh;
                std::memcpy(&dh, blk, 2);
                float d = fp16_to_f32(dh);
                uint32_t qh_val;
                std::memcpy(&qh_val, blk + 18, 4);
                for (int k = 0; k < 16; ++k) {
                    uint8_t q = blk[2 + k];
                    int8_t lo = static_cast<int8_t>((q & 0xF) | (((qh_val >> k)     & 1) << 4)) - 16;
                    int8_t hi = static_cast<int8_t>((q >>  4) | (((qh_val >> (k+16)) & 1) << 4)) - 16;
                    if (2*k     < x_len) dot += (d * lo) * x[x_start + 2*k];
                    if (2*k + 1 < x_len) dot += (d * hi) * x[x_start + 2*k + 1];
                }
            } else if (qtype == 7) { // Q5_1: [d:2][dmin:2][ql:16][qh:4] = 24 bytes, 32 elements
                uint16_t dh, dmh;
                std::memcpy(&dh, blk, 2);
                std::memcpy(&dmh, blk + 2, 2);
                float d = fp16_to_f32(dh);
                float dmin = fp16_to_f32(dmh);
                uint32_t qh_val;
                std::memcpy(&qh_val, blk + 20, 4);
                for (int k = 0; k < 16; ++k) {
                    uint8_t q = blk[4 + k];
                    int8_t lo = static_cast<int8_t>((q & 0xF) | (((qh_val >> k) & 1) << 4)) - 16;
                    int8_t hi = static_cast<int8_t>((q >> 4) | (((qh_val >> (k+16)) & 1) << 4)) - 16;
                    if (2*k < x_len) dot += (d * lo + dmin) * x[x_start + 2*k];
                    if (2*k + 1 < x_len) dot += (d * hi + dmin) * x[x_start + 2*k + 1];
                }
            } else if (qtype == 12) { // Q4_K: SIMD fused dequant+dot
                uint16_t _dh, _dmh;
                std::memcpy(&_dh, blk, 2);
                std::memcpy(&_dmh, blk + 2, 2);
                float kv_d = fp16_to_f32(_dh);
                float kv_dm = fp16_to_f32(_dmh);
                const uint8_t* scales = blk + 4;
                const uint8_t* q      = blk + 16;
                for (int sb = 0; sb < 4; ++sb) {
                    uint8_t sc0, m0, sc1, m1;
                    get_scale_min_k4(2 * sb,     scales, &sc0, &m0);
                    get_scale_min_k4(2 * sb + 1, scales, &sc1, &m1);
                    float d1 = kv_d * sc0, dm1 = kv_dm * m0;
                    float d2 = kv_d * sc1, dm2 = kv_dm * m1;
                    int base = sb * 64;  // x offset for this super-block
                    for (int l = 0; l < 32; ++l) {
                        int xi = base + l;
                        if (xi >= x_start && xi < x_start + x_len) {
                            dot += (d1 * static_cast<float>(q[l] & 0xF) - dm1) * x[xi];
                        }
                    }
                    for (int l = 0; l < 32; ++l) {
                        int xi = base + 32 + l;
                        if (xi >= x_start && xi < x_start + x_len) {
                            dot += (d2 * static_cast<float>(q[l] >> 4) - dm2) * x[xi];
                        }
                    }
                    q += 32;
                }
            } else if (qtype == 13) { // Q5_K: SIMD fused dequant+dot
                uint16_t _dh5, _dmh5;
                std::memcpy(&_dh5, blk, 2);
                std::memcpy(&_dmh5, blk + 2, 2);
                float kv_d5  = fp16_to_f32(_dh5);
                float kv_dm5 = fp16_to_f32(_dmh5);
                const uint8_t* scales = blk + 4;
                const uint8_t* ql     = blk + 16;
                const uint8_t* qh     = blk + 144;
                uint8_t u1 = 1, u2 = 2;
                for (int sb = 0; sb < 4; ++sb) {
                    uint8_t sc0, m0, sc1, m1;
                    get_scale_min_k4(2 * sb,     scales, &sc0, &m0);
                    get_scale_min_k4(2 * sb + 1, scales, &sc1, &m1);
                    float d1 = kv_d5 * sc0, dm1 = kv_dm5 * m0;
                    float d2 = kv_d5 * sc1, dm2 = kv_dm5 * m1;
                    int base = sb * 64;
                    for (int l = 0; l < 32; ++l) {
                        int xi = base + l;
                        if (xi >= x_start && xi < x_start + x_len) {
                            uint8_t w = (ql[l] & 0xF) + (qh[l] & u1 ? 16u : 0u);
                            dot += (d1 * static_cast<float>(w) - dm1) * x[xi];
                        }
                    }
                    for (int l = 0; l < 32; ++l) {
                        int xi = base + 32 + l;
                        if (xi >= x_start && xi < x_start + x_len) {
                            uint8_t w = (ql[l] >> 4) + (qh[l] & u2 ? 16u : 0u);
                            dot += (d2 * static_cast<float>(w) - dm2) * x[xi];
                        }
                    }
                    ql += 32;
                    u1 <<= 2;
                    u2 <<= 2;
                }
            } else if (qtype == 14) { // Q6_K: SIMD fused dequant+dot
                uint16_t _dh6;
                std::memcpy(&_dh6, blk + 208, 2);
                float kv_d6 = fp16_to_f32(_dh6);
                const uint8_t* bql = blk;
                const uint8_t* bqh = blk + 128;
                const int8_t*  bsc = reinterpret_cast<const int8_t*>(blk + 192);
                for (int n = 0; n < 2; ++n) {
                    for (int l = 0; l < 32; ++l) {
                        int is = l / 16;
                        const int8_t q1 = static_cast<int8_t>(
                            (bql[l +  0] & 0xF) | (((bqh[l] >> 0) & 3u) << 4)) - 32;
                        const int8_t q2 = static_cast<int8_t>(
                            (bql[l + 32] & 0xF) | (((bqh[l] >> 2) & 3u) << 4)) - 32;
                        const int8_t q3 = static_cast<int8_t>(
                            (bql[l +  0] >> 4)  | (((bqh[l] >> 4) & 3u) << 4)) - 32;
                        const int8_t q4 = static_cast<int8_t>(
                            (bql[l + 32] >> 4)  | (((bqh[l] >> 6) & 3u) << 4)) - 32;
                        int xs = n * 128;
                        { int xi = xs + l +  0; if (xi >= x_start && xi < x_start + x_len) dot += kv_d6 * bsc[is + 0] * q1 * x[xi]; }
                        { int xi = xs + l + 32; if (xi >= x_start && xi < x_start + x_len) dot += kv_d6 * bsc[is + 2] * q2 * x[xi]; }
                        { int xi = xs + l + 64; if (xi >= x_start && xi < x_start + x_len) dot += kv_d6 * bsc[is + 4] * q3 * x[xi]; }
                        { int xi = xs + l + 96; if (xi >= x_start && xi < x_start + x_len) dot += kv_d6 * bsc[is + 6] * q4 * x[xi]; }
                    }
                    bql += 64;
                    bqh += 32;
                    bsc += 8;
                }
            } else {
                // Fallback: dequantize via dedicated functions or zeros
                if (qtype == 3) { // Q4_1 (fallback only)
                    float qk[32];
                    dequantize_q4_0(blk, qk, 1);  // approximate
                    for (int k = 0; k < x_len; ++k) dot += qk[k] * x[x_start + k];
                }
            }
        }
        y[m] += dot;
    }
}

}}  // namespace hesa::simd

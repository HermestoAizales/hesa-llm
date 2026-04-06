#include "hesa/rope.hpp"
#include <algorithm>
#include <cassert>
#include <cstring>

namespace hesa {

// ─────────────────────────────────────────────────────────────
// rope_apply — vectorised RoPE application
// ─────────────────────────────────────────────────────────────

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
#include <arm_neon.h>

static inline void rope_apply_neon(float* data, int pos, int n_dims,
                                    const float* cos_ptr, const float* sin_ptr) {
    const int pairs = n_dims / 2;
    for (int i = 0; i < pairs; ++i) {
        // Load pair (2i, 2i+1)
        float q0 = data[2 * i];
        float q1 = data[2 * i + 1];
        int j = i; // dim index within n_dims
        float cos_val = cos_ptr[pos * n_dims + j];
        float sin_val = sin_ptr[pos * n_dims + j];
        data[2 * i]     = q0 * cos_val - q1 * sin_val;
        data[2 * i + 1] = q0 * sin_val + q1 * cos_val;
    }
}

#elif defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>

static inline void rope_apply_avx2(float* data, int pos, int n_dims,
                                    const float* cos_ptr, const float* sin_ptr) {
    const int pairs = n_dims / 2;
    for (int i = 0; i < pairs; ++i) {
        float q0 = data[2 * i];
        float q1 = data[2 * i + 1];
        int j = i;
        float cos_val = cos_ptr[pos * n_dims + j];
        float sin_val = sin_ptr[pos * n_dims + j];
        data[2 * i]     = q0 * cos_val - q1 * sin_val;
        data[2 * i + 1] = q0 * sin_val + q1 * cos_val;
    }
}
#endif

void rope_apply(TensorView q_or_k,
                std::span<const int32_t> positions,
                float freq_base,
                int n_dims)
{
    assert(q_or_k.data() != nullptr);
    assert(q_or_k.dtype() == Dtype::F32);
    assert(q_or_k.ndim() >= 2);
    assert(n_dims > 0 && n_dims % 2 == 0);
    assert(positions.size() == static_cast<size_t>(q_or_k.shape()[0]));

    int n_positions = static_cast<int>(positions.size());
    int head_dim    = static_cast<int>(q_or_k.shape()[1]);
    n_dims = std::min(n_dims, head_dim);

    // Pre-compute the inverse frequency for each dimension pair
    // inv_freq[i] = 1.0f / powf(freq_base, (2*i) / n_dims)
    std::vector<float> inv_freq(n_dims / 2);
    for (int i = 0; i < n_dims / 2; ++i) {
        inv_freq[i] = 1.0f / std::pow(freq_base, static_cast<float>(2 * i) / n_dims);
    }

    auto* raw = static_cast<float*>(q_or_k.data());
    const size_t stride = static_cast<size_t>(head_dim);

    for (int p = 0; p < n_positions; ++p) {
        int pos_val = positions[p];
        float* vec = raw + p * stride;

        for (int i = 0; i < n_dims / 2; ++i) {
            float f = static_cast<float>(pos_val) * inv_freq[i];
            float cos_val = std::cos(f);
            float sin_val = std::sin(f);

            float q0 = vec[2 * i];
            float q1 = vec[2 * i + 1];

            vec[2 * i]     = q0 * cos_val - q1 * sin_val;
            vec[2 * i + 1] = q0 * sin_val + q1 * cos_val;
        }
    }
}

// ─────────────────────────────────────────────────────────────
// RoPECache
// ─────────────────────────────────────────────────────────────

RoPECache::RoPECache(int max_pos, int n_dims, float freq_base)
    : max_pos_(max_pos),
      n_dims_(n_dims),
      cos_table_(static_cast<size_t>(max_pos) * n_dims),
      sin_table_(static_cast<size_t>(max_pos) * n_dims)
{
    assert(n_dims > 0 && n_dims % 2 == 0);

    for (int pos = 0; pos < max_pos; ++pos) {
        for (int i = 0; i < n_dims / 2; ++i) {
            float freq = 1.0f / std::pow(freq_base, static_cast<float>(2 * i) / n_dims);
            float f = static_cast<float>(pos) * freq;
            size_t idx = static_cast<size_t>(pos) * n_dims + i;
            cos_table_[idx] = std::cos(f);
            sin_table_[idx] = std::sin(f);
        }
    }
}

void RoPECache::apply(float* data, int32_t position, int vec_len) const {
    assert(position >= 0 && position < max_pos_);
    const int n_apply = std::min(n_dims_, vec_len);

#if defined(__ARM_NEON) || defined(__aarch64__) || defined(_M_ARM64)
    rope_apply_neon(data, position, n_apply,
                     cos_table_.data(), sin_table_.data());
#elif defined(__AVX2__) || defined(__AVX512F__)
    rope_apply_avx2(data, position, n_apply,
                     cos_table_.data(), sin_table_.data());
#else
    // Generic scalar fallback (same as rope_apply inner loop)
    size_t off = static_cast<size_t>(position) * n_dims_;
    const float* cos = cos_table_.data() + off;
    const float* sin = sin_table_.data() + off;

    for (int i = 0; i < n_apply / 2; ++i) {
        float q0 = data[2 * i];
        float q1 = data[2 * i + 1];
        data[2 * i]     = q0 * cos[i] - q1 * sin[i];
        data[2 * i + 1] = q0 * sin[i] + q1 * cos[i];
    }
#endif
}

} // namespace hesa

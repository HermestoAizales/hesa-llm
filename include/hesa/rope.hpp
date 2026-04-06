#ifndef HESA_ROPE_HPP
#define HESA_ROPE_HPP

#include "hesa/tensor.hpp"
#include <cstdlib>
#include <cstdint>
#include <span>
#include <vector>
#include <cmath>

namespace hesa {

/**
 * rope_apply — Apply Rotary Position Embedding to a Q or K tensor in-place.
 *
 * For each position p and each dimension pair (2i, 2i+1):
 *   theta_i = 1 / (freq_base^(2i / n_dims))
 *   f = p * theta_i
 *   q[2i]   = q[2i] * cos(f) - q[2i+1] * sin(f)
 *   q[2i+1] = q[2i] * sin(f) + q[2i+1] * cos(f)
 *
 * @param q_or_k      TensorView of shape [n_positions, head_dim] (F32)
 *                    Modified in-place.
 * @param positions   Array of logical positions (size = n_positions)
 * @param freq_base   Base frequency for RoPE (typically 10000.0f)
 * @param n_dims      Number of dimensions to apply RoPE to (<= head_dim,
 *                    typically head_dim; must be even)
 */
void rope_apply(TensorView q_or_k,
                std::span<const int32_t> positions,
                float freq_base,
                int n_dims);

/**
 * RoPEFreqTable — Pre-computed cosine/sin table for repeated use.
 *
 * Useful when applying RoPE position-by-position during incremental decoding.
 * Stores cos/sin values for positions [0, max_pos) and dimensions [0, n_dims).
 */
class RoPECache {
public:
    /**
     * Precompute frequency table.
     * @param max_pos     Maximum position index (+1)
     * @param n_dims      Number of RoPE dimensions (must be even)
     * @param freq_base   Base frequency (e.g., 10000.0f)
     */
    RoPECache(int max_pos, int n_dims, float freq_base = 10000.0f);

    /** Apply RoPE to a single position's data in-place. */
    void apply(float* data, int32_t position, int vec_len) const;

    const float* cos_table() const { return cos_table_.data(); }
    const float* sin_table() const { return sin_table_.data(); }
    int max_pos() const { return max_pos_; }
    int n_dims() const { return n_dims_; }

private:
    int max_pos_;
    int n_dims_;
    std::vector<float> cos_table_;  // [max_pos][n_dims]
    std::vector<float> sin_table_;  // [max_pos][n_dims]
};

} // namespace hesa

#endif // HESA_ROPE_HPP

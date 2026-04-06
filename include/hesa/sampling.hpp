#ifndef HESA_SAMPLING_HPP
#define HESA_SAMPLING_HPP

#include "hesa/result.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace hesa {

struct GenerationConfig {
    float   temperature        = 0.8f;
    float   top_p              = 0.95f;
    int32_t top_k              = 40;
    float   repetition_penalty = 1.1f;
    int32_t max_tokens         = 2048;
    int32_t seed               = -1; // -1 = random
};

/**
 * Select a single token from logits using temperature + top-k + top-p + rep-penalty.
 * Modifies logits in-place (applies softmax).
 */
int32_t sample_token(float* logits_in_out, int32_t vocab_size,
                     std::span<const int32_t> prev_tokens,
                     const GenerationConfig& config,
                     int32_t rng_seed = -1);

} // namespace hesa

#endif // HESA_SAMPLING_HPP

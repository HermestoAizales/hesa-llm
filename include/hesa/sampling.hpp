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
 * Standalone per-token sampling functions (no heap allocation in hot path).
 * All operate on float logits arrays (CPU, F32).
 */

/**
 * Greedy sampling — returns argmax(logits).
 */
int32_t sample_greedy(std::span<const float> logits);

/**
 * Temperature sampling — apply softmax with temperature, then sample.
 */
int32_t sample_temperature(std::span<const float> logits, float temp);

/**
 * Top-K sampling — keep top-k logits, apply temperature softmax, sample.
 */
int32_t sample_top_k(std::span<const float> logits, int32_t k, float temp);

/**
 * Top-P (nucleus) sampling — cumulative probability filtering.
 */
int32_t sample_top_p(std::span<const float> logits, float top_p, float temp);

/**
 * Repetition-penalty sampling — reduce logit of previously generated tokens.
 * penalty > 1.0f: penalise repetition.
 * penalty == 1.0f: identity.
 *
 * @param logits       Logits array [vocab_size], modified in-place
 * @param last_tokens  Previously generated token IDs
 * @param penalty      Repetition penalty factor
 * @return Sampled token ID
 */
int32_t sample_repetition_penalty(std::span<float> logits,
                                   std::span<const int32_t> last_tokens,
                                   float penalty);

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

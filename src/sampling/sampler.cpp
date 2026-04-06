#include "hesa/sampling.hpp"

#include <algorithm>
#include <random>
#include <cmath>

namespace hesa {

int32_t sample_token(float* logits, int32_t vocab_size,
                     std::span<const int32_t> prev_tokens,
                     const GenerationConfig& config,
                     int32_t rng_seed) {
    if (vocab_size <= 0) return 0;

    // Apply repetition penalty
    if (config.repetition_penalty != 1.0f && !prev_tokens.empty()) {
        for (int32_t id : prev_tokens) {
            if (id >= 0 && id < vocab_size) {
                if (logits[id] > 0.0f)
                    logits[id] /= config.repetition_penalty;
                else
                    logits[id] *= config.repetition_penalty;
            }
        }
    }

    // Temperature
    float temp = config.temperature > 0.0f ? config.temperature : 1.0f;
    if (temp != 1.0f) {
        for (int32_t i = 0; i < vocab_size; ++i)
            logits[i] /= temp;
    }

    // Top-K filtering: zero out logits outside top-k
    if (config.top_k > 0 && config.top_k < vocab_size) {
        std::vector<float> sorted_logits(logits, logits + vocab_size);
        std::sort(sorted_logits.begin(), sorted_logits.end(), std::greater<float>());
        float threshold = sorted_logits[config.top_k - 1];
        for (int32_t i = 0; i < vocab_size; ++i)
            if (logits[i] < threshold) logits[i] = -1e9f;
    }

    // Top-P (nucleus) filtering
    if (config.top_p < 1.0f && config.top_p > 0.0f) {
        // Compute softmax first
        float max_val = logits[0];
        for (int32_t i = 1; i < vocab_size; ++i)
            if (logits[i] > max_val) max_val = logits[i];
        float sum = 0.0f;
        for (int32_t i = 0; i < vocab_size; ++i) {
            logits[i] = std::exp(logits[i] - max_val);
            sum += logits[i];
        }
        float inv_sum = 1.0f / sum;
        for (int32_t i = 0; i < vocab_size; ++i) logits[i] *= inv_sum;

        // Nucleus filter
        std::vector<std::pair<float, int32_t>> sorted;
        sorted.reserve(vocab_size);
        for (int32_t i = 0; i < vocab_size; ++i)
            sorted.emplace_back(logits[i], i);
        std::sort(sorted.begin(), sorted.end(), std::greater<>());

        float cumsum = 0.0f;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cumsum += sorted[i].first;
            if (cumsum > config.top_p) {
                for (size_t j = i + 1; j < sorted.size(); ++j)
                    logits[sorted[j].second] = 0.0f;
                break;
            }
        }
        // Re-normalize
        sum = 0.0f;
        for (int32_t i = 0; i < vocab_size; ++i) sum += logits[i];
        if (sum > 0.0f) {
            inv_sum = 1.0f / sum;
            for (int32_t i = 0; i < vocab_size; ++i) logits[i] *= inv_sum;
        }
    } else {
        // Simple softmax
        float max_val = logits[0];
        for (int32_t i = 1; i < vocab_size; ++i)
            if (logits[i] > max_val) max_val = logits[i];
        float sum = 0.0f;
        for (int32_t i = 0; i < vocab_size; ++i) {
            logits[i] = std::exp(logits[i] - max_val);
            sum += logits[i];
        }
        float inv_sum = 1.0f / sum;
        for (int32_t i = 0; i < vocab_size; ++i) logits[i] *= inv_sum;
    }

    // Sample
    std::mt19937 rng(rng_seed >= 0 ? static_cast<unsigned>(rng_seed) :
                     static_cast<unsigned>(std::random_device{}()));
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);

    float cumsum = 0.0f;
    for (int32_t i = 0; i < vocab_size; ++i) {
        cumsum += logits[i];
        if (r <= cumsum) return i;
    }
    return vocab_size - 1;
}

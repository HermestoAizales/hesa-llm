#include "hesa/sampling.hpp"

#include <algorithm>
#include <random>
#include <cmath>
#include <cassert>
#include <vector>

namespace hesa {

static inline void softmax_inplace(float* logits, size_t n) {
    if (n == 0) return;
    float max_val = logits[0];
    for (size_t i = 1; i < n; ++i)
        if (logits[i] > max_val) max_val = logits[i];
    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        logits[i] = std::exp(logits[i] - max_val);
        sum += logits[i];
    }
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (size_t i = 0; i < n; ++i) logits[i] *= inv;
    }
}

static inline int32_t weighted_sample(const float* probs, size_t n,
                                       std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    float r = dist(rng);
    float cumsum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        cumsum += probs[i];
        if (r <= cumsum) return static_cast<int32_t>(i);
    }
    return static_cast<int32_t>(n - 1);
}

int32_t sample_greedy(std::span<const float> logits) {
    assert(!logits.empty());
    int32_t best = 0;
    float max_val = logits[0];
    for (size_t i = 1; i < logits.size(); ++i) {
        if (logits[i] > max_val) {
            max_val = logits[i];
            best = static_cast<int32_t>(i);
        }
    }
    return best;
}

int32_t sample_temperature(std::span<const float> logits, float temp) {
    assert(!logits.empty());
    assert(temp > 0.0f);
    const size_t n = logits.size();
    std::vector<float> buf(logits.begin(), logits.end());
    if (temp != 1.0f) {
        float inv_temp = 1.0f / temp;
        for (size_t i = 0; i < n; ++i) buf[i] *= inv_temp;
    }
    softmax_inplace(buf.data(), n);
    std::random_device rd;
    std::mt19937 rng(rd());
    return weighted_sample(buf.data(), n, rng);
}

int32_t sample_top_k(std::span<const float> logits, int32_t k, float temp) {
    assert(!logits.empty());
    assert(k > 0);
    assert(temp > 0.0f);
    const size_t n = logits.size();
    if (k >= static_cast<int32_t>(n))
        return sample_temperature(logits, temp);

    std::vector<float> buf(logits.begin(), logits.end());
    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(n);
    for (int32_t i = 0; i < static_cast<int32_t>(n); ++i)
        scored.emplace_back(buf[i], i);
    std::partial_sort(scored.begin(), scored.begin() + k, scored.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });
    float threshold = scored[k - 1].first;

    if (temp != 1.0f) {
        float inv_temp = 1.0f / temp;
        for (size_t i = 0; i < n; ++i) buf[i] *= inv_temp;
    }
    for (size_t i = 0; i < n; ++i)
        if (buf[i] < threshold) buf[i] = -1e9f;
    softmax_inplace(buf.data(), n);

    std::random_device rd;
    std::mt19937 rng(rd());
    return weighted_sample(buf.data(), n, rng);
}

int32_t sample_top_p(std::span<const float> logits, float top_p, float temp) {
    assert(!logits.empty());
    assert(top_p > 0.0f && top_p <= 1.0f);
    assert(temp > 0.0f);
    const size_t n = logits.size();
    std::vector<float> buf(logits.begin(), logits.end());

    if (temp != 1.0f) {
        float inv_temp = 1.0f / temp;
        for (size_t i = 0; i < n; ++i) buf[i] *= inv_temp;
    }
    softmax_inplace(buf.data(), n);

    std::vector<std::pair<float, int32_t>> scored;
    scored.reserve(n);
    for (int32_t i = 0; i < static_cast<int32_t>(n); ++i)
        scored.emplace_back(buf[i], i);
    std::sort(scored.begin(), scored.end(),
              [](const auto& a, const auto& b) { return a.first > b.first; });

    float cumsum = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        cumsum += scored[i].first;
        if (cumsum > top_p) {
            for (size_t j = i + 1; j < n; ++j)
                buf[scored[j].second] = 0.0f;
            break;
        }
    }

    float sum = 0.0f;
    for (size_t i = 0; i < n; ++i) sum += buf[i];
    if (sum > 0.0f) {
        float inv = 1.0f / sum;
        for (size_t i = 0; i < n; ++i) buf[i] *= inv;
    }

    std::random_device rd;
    std::mt19937 rng(rd());
    return weighted_sample(buf.data(), n, rng);
}

int32_t sample_repetition_penalty(std::span<float> logits,
                                   std::span<const int32_t> last_tokens,
                                   float penalty) {
    assert(!logits.empty());
    assert(penalty > 0.0f);
    const int32_t vocab_size = static_cast<int32_t>(logits.size());
    if (penalty != 1.0f && !last_tokens.empty()) {
        for (int32_t id : last_tokens) {
            if (id >= 0 && id < vocab_size) {
                if (logits[id] > 0.0f) logits[id] /= penalty;
                else logits[id] *= penalty;
            }
        }
    }
    return sample_greedy(std::span<const float>(logits.data(), logits.size()));
}

int32_t sample_token(float* logits_ptr, int32_t vocab_size,
                     std::span<const int32_t> prev_tokens,
                     const GenerationConfig& config,
                     int32_t rng_seed) {
    if (vocab_size <= 0) return 0;

    if (config.repetition_penalty != 1.0f && !prev_tokens.empty()) {
        for (int32_t id : prev_tokens) {
            if (id >= 0 && id < vocab_size) {
                if (logits_ptr[id] > 0.0f)
                    logits_ptr[id] /= config.repetition_penalty;
                else
                    logits_ptr[id] *= config.repetition_penalty;
            }
        }
    }

    float temp = config.temperature > 0.0f ? config.temperature : 1.0f;
    if (temp != 1.0f) {
        float inv_temp = 1.0f / temp;
        for (int32_t i = 0; i < vocab_size; ++i)
            logits_ptr[i] *= inv_temp;
    }

    if (config.top_k > 0 && config.top_k < vocab_size) {
        std::vector<std::pair<float, int32_t>> scored;
        scored.reserve(vocab_size);
        for (int32_t i = 0; i < vocab_size; ++i)
            scored.emplace_back(logits_ptr[i], i);
        std::partial_sort(scored.begin(), scored.begin() + config.top_k, scored.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });
        float threshold = scored[config.top_k - 1].first;
        for (int32_t i = 0; i < vocab_size; ++i)
            if (logits_ptr[i] < threshold) logits_ptr[i] = -1e9f;
    }

    if (config.top_p < 1.0f && config.top_p > 0.0f) {
        softmax_inplace(logits_ptr, vocab_size);
        std::vector<std::pair<float, int32_t>> sorted;
        sorted.reserve(vocab_size);
        for (int32_t i = 0; i < vocab_size; ++i)
            sorted.emplace_back(logits_ptr[i], i);
        std::sort(sorted.begin(), sorted.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        float cumsum = 0.0f;
        bool cut = false;
        for (size_t i = 0; i < sorted.size(); ++i) {
            cumsum += sorted[i].first;
            if (cumsum > config.top_p) cut = true;
            if (cut) logits_ptr[sorted[i].second] = 0.0f;
        }
        float sum = 0.0f;
        for (int32_t i = 0; i < vocab_size; ++i) sum += logits_ptr[i];
        if (sum > 0.0f) {
            float inv = 1.0f / sum;
            for (int32_t i = 0; i < vocab_size; ++i) logits_ptr[i] *= inv;
        }
    } else {
        softmax_inplace(logits_ptr, vocab_size);
    }

    std::mt19937 rng(rng_seed >= 0
        ? static_cast<unsigned>(rng_seed)
        : static_cast<unsigned>(std::random_device{}()));
    return weighted_sample(logits_ptr, vocab_size, rng);
}

} // namespace hesa

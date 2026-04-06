#ifndef HESA_ENGINE_HPP
#define HESA_ENGINE_HPP

#include "hesa/backend.hpp"
#include "hesa/model.hpp"
#include "hesa/result.hpp"
#include "hesa/sampling.hpp"
#include "hesa/tokenizer.hpp"

#include <atomic>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace hesa {

// ─── Forward pass state ──────────────────────────────────────────
struct InferenceState {
    std::vector<float> logits;       // [vocab_size]
    std::vector<int32_t> generated;  // tokens generated so far
    int32_t last_token = -1;
    bool    done       = false;
};

// ─── Engine ───────────────────────────────────────────────────────
class Engine {
public:
    struct Config {
        BackendType      backend       = BackendType::AUTO;
        int32_t          n_threads     = 0;
        size_t           kv_cache_size = 4096;
        GenerationConfig sampling;
    };

    /** Load model, create backend, and initialise an engine. */
    static Result<std::unique_ptr<Engine>> create(const std::string& model_path,
                                                  const Config& cfg);

    ~Engine();

    /** Generate tokens from a tokenised prompt. Returns the full output sequence. */
    Result<std::vector<int32_t>> generate(std::span<const int32_t> prompt,
                                          size_t max_tokens,
                                          float temperature,
                                          float top_p);

    /** Stop an in-progress generation (thread-safe). */
    Result<void> stop();

    /** Access the loaded model metadata. */
    const ModelMetadata& metadata() const { return model_->metadata(); }

    /** Access the model (for advanced usage). */
    const Model& model() const { return *model_; }

    // Public constructor for make_unique access (only create() should call this).
    Engine() = default;

    // --- Phase-1 forward pass (embedding → simple layer → LM head) ---
    Result<void> forward_single_token(int32_t token_id,
                                      int32_t position,
                                      std::vector<float>& out_hidden);

    // --- Logits from hidden state via output embedding ---
    Result<void> compute_logits(const std::vector<float>& hidden,
                                std::vector<float>& logits_out);

    std::unique_ptr<Model>     model_;
    std::unique_ptr<Backend>   backend_;
    std::unique_ptr<Tokenizer> tokenizer_;

    std::atomic<bool> stop_requested_{false};

    // Working buffers (re-used each step to avoid allocations)
    std::vector<float> hidden_buf_;
    std::vector<float> logits_buf_;
};

} // namespace hesa

#endif // HESA_ENGINE_HPP

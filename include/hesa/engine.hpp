#ifndef HESA_ENGINE_HPP
#define HESA_ENGINE_HPP

#include "hesa/backend.hpp"
#include "hesa/kv_cache.hpp"
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

struct LayerBuffers;

struct InferenceState {
    std::vector<float> logits;
    std::vector<int32_t> generated;
    int32_t last_token = -1;
    bool    done       = false;
};

class Engine {
public:
    struct Config {
        BackendType      backend       = BackendType::AUTO;
        int32_t          n_threads     = 0;
        size_t           kv_cache_size = 4096;
        GenerationConfig sampling;
    };

    static Result<std::unique_ptr<Engine>> create(const std::string& model_path,
                                                  const Config& cfg);

    ~Engine();

    Result<std::vector<int32_t>> generate(std::span<const int32_t> prompt,
                                          size_t max_tokens,
                                          float temperature,
                                          float top_p);

    Result<void> stop();

    const ModelMetadata& metadata() const { return model_->metadata(); }
    const Model& model() const { return *model_; }
    Tokenizer* tokenizer() const { return tokenizer_.get(); }

    Result<void> forward_single_token(int32_t token_id,
                                      int32_t position,
                                      std::vector<float>& out_hidden);

    Result<void> compute_logits(const std::vector<float>& hidden,
                                std::vector<float>& logits_out);

    std::unique_ptr<Model>     model_;
    std::unique_ptr<Backend>   backend_;
    std::unique_ptr<Tokenizer> tokenizer_;

    std::unique_ptr<KVCache>   kv_cache_;
    std::vector<LayerBuffers>  layer_bufs_;
    int32_t                    position_counter_ = 0;
    std::atomic<bool>          stop_requested_{false};
    std::vector<float>         hidden_buf_;
    std::vector<float>         logits_buf_;

private:
    Engine();
};

} // namespace hesa
#endif

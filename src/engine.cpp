#include "hesa/engine.hpp"
#include "backend/cpu_backend.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>
#include <cstring>
#include <span>

namespace hesa {

// Helper macros for functions returning unique_ptr<Engine>
#define ENG_CHECK(expr) \
    do { \
        auto _r = (expr); \
        if (!_r) return std::unexpected{_r.error()}; \
        (void)_r; \
    } while(0)

// Engine factory
Result<std::unique_ptr<Engine>> Engine::create(const std::string& model_path,
                                                const Config& cfg) {
    fprintf(stderr, "[Engine] Loading model: %s\n", model_path.c_str());

    DeviceConfig dev_cfg;
    dev_cfg.n_threads = cfg.n_threads;

    auto backend = create_backend(cfg.backend, dev_cfg);
    if (!backend) return std::unexpected{backend.error()};

    auto model = Model::load(model_path, backend->get());
    if (!model) return std::unexpected{model.error()};

    auto tokenizer = create_tokenizer_from_model(**model);
    if (!tokenizer) return std::unexpected{tokenizer.error()};

    const auto& meta = (*model)->metadata();
    fprintf(stderr, "[Engine] Model loaded: arch=%s, vocab=%u, layers=%u, hidden=%u\n",
            meta.architecture.empty() ? "?" : meta.architecture.c_str(),
            meta.vocab_size, meta.block_count, meta.embedding_length);

    auto engine = std::make_unique<Engine>();
    engine->backend_   = std::move(*backend);
    engine->model_     = std::move(*model);
    engine->tokenizer_ = std::move(*tokenizer);
    engine->stop_requested_.store(false);

    size_t hidden_dim = engine->model_->metadata().embedding_length;
    if (hidden_dim == 0) hidden_dim = 256;
    engine->hidden_buf_.resize(hidden_dim);
    engine->logits_buf_.resize(engine->model_->metadata().vocab_size > 0
                               ? engine->model_->metadata().vocab_size : 32000);

    fprintf(stderr, "[Engine] Engine ready (hidden=%zu, vocab=%zu)\n",
            hidden_dim, engine->logits_buf_.size());

    return engine;
}

Engine::~Engine() = default;

Result<void> Engine::stop() {
    stop_requested_.store(true);
    return ok();
}

// Forward pass - simplified for Phase 1
Result<void> Engine::forward_single_token(int32_t token_id, int32_t /*pos*/,
                                           std::vector<float>& out_hidden) {
    const auto& meta = model_->metadata();
    size_t hidden = meta.embedding_length;
    if (hidden == 0) hidden = hidden_buf_.size();
    std::fill(out_hidden.begin(), out_hidden.end(), 0.0f);
    if (hidden == 0) return ok();

    Tensor const* emb = nullptr;
    for (const char* name : {
        "token_embd.weight",
        "tok_embeddings.weight",
        "model.embed_tokens.weight",
    }) {
        auto r = model_->get_tensor(std::string(name));
        if (r) {
            emb = *r;
            break;
        }
    }

    if (!emb || emb->dtype() != Dtype::F32 || emb->ndim() < 2) return ok();

    // GGUF/Llama: token_embd is [hidden, vocab], so shape[0]=hidden, shape[1]=vocab
    size_t emb_hidden = static_cast<size_t>(emb->shape()[0]);
    size_t emb_vocab  = static_cast<size_t>(emb->shape()[1]);

    if (emb_hidden != hidden && emb_vocab == hidden) {
        // Shape is [vocab, hidden] - need to handle differently
        std::swap(emb_hidden, emb_vocab);
    }

    if (emb_hidden < hidden) hidden = emb_hidden;
    std::fill(out_hidden.begin(), out_hidden.begin() + hidden, 0.0f);

    if (static_cast<size_t>(token_id) < emb_vocab) {
        const float* edata = static_cast<const float*>(emb->data());
        // If shape is [hidden, vocab], then token_id indexes into column
        // Column-major: offset = token_id * hidden
        size_t offset = static_cast<size_t>(token_id);
        // GGUF reverse: original [hidden, vocab] stored as [vocab, hidden]
        // So shape[0]=vocab, shape[1]=hidden in our storage
        // We need row=token_id, then take the full row
        std::copy(edata + static_cast<size_t>(token_id) * emb_vocab,
                  edata + (static_cast<size_t>(token_id) + 1) * emb_vocab,
                  out_hidden.data());
    }

    return ok();
}

Result<void> Engine::compute_logits(const std::vector<float>& hidden,
                                     std::vector<float>& logits_out) {
    const auto& meta = model_->metadata();
    size_t vocab = meta.vocab_size;
    size_t hidden_dim = hidden.size();
    if (vocab == 0) vocab = logits_out.size();
    if (hidden_dim == 0) {
        std::fill(logits_out.begin(), logits_out.end(), 0.0f);
        return ok();
    }

    Tensor const* ow = nullptr;
    for (const char* name : {
        "output.weight",
        "lm_head.weight",
    }) {
        auto r = model_->get_tensor(std::string(name));
        if (r) { ow = *r; break; }
    }

    if (ow && ow->dtype() == Dtype::F32 && ow->ndim() >= 2) {
        size_t ow_vocab  = static_cast<size_t>(ow->shape()[0]);
        size_t ow_hidden = static_cast<size_t>(ow->shape()[1]);
        const float* wd = static_cast<const float*>(ow->data());

        for (size_t v = 0; v < std::min(vocab, logits_out.size()) && v < ow_vocab; ++v) {
            float sum = 0.0f;
            for (size_t h = 0; h < std::min(hidden_dim, ow_hidden); ++h) {
                sum += wd[v * ow_hidden + h] * hidden[h];
            }
            logits_out[v] = sum / std::sqrt(static_cast<float>(hidden_dim));
        }
    } else {
        // Tied weights: use transposed embedding
        Tensor const* emb = nullptr;
        for (const char* name : {"token_embd.weight", "tok_embeddings.weight"}) {
            auto r = model_->get_tensor(std::string(name));
            if (r) { emb = *r; break; }
        }
        if (emb && emb->dtype() == Dtype::F32 && emb->ndim() >= 2) {
            size_t e_rows = static_cast<size_t>(emb->shape()[0]);
            size_t e_cols = static_cast<size_t>(emb->shape()[1]);
            const float* ed = static_cast<const float*>(emb->data());

            for (size_t v = 0; v < std::min(vocab, logits_out.size()) && v < e_rows; ++v) {
                float sum = 0.0f;
                for (size_t h = 0; h < std::min(hidden_dim, e_cols); ++h) {
                    sum += ed[v * e_cols + h] * hidden[h];
                }
                logits_out[v] = sum / std::sqrt(static_cast<float>(hidden_dim));
            }
        }
    }

    if (vocab < logits_out.size()) {
        std::fill(logits_out.begin() + static_cast<ptrdiff_t>(vocab), logits_out.end(), 0.0f);
    }

    return ok();
}

// Generate
Result<std::vector<int32_t>> Engine::generate(std::span<const int32_t> prompt,
                                               size_t max_tokens,
                                               float temperature,
                                               float top_p) {
    stop_requested_.store(false);

    const auto& meta = model_->metadata();
    size_t vocab = meta.vocab_size;
    if (vocab == 0) vocab = 32000;

    std::vector<int32_t> output{prompt.begin(), prompt.end()};
    output.reserve(prompt.size() + max_tokens);

    GenerationConfig gen_cfg;
    gen_cfg.temperature = temperature;
    gen_cfg.top_p = top_p;
    gen_cfg.max_tokens = static_cast<int32_t>(max_tokens);
    gen_cfg.top_k = 40;
    gen_cfg.repetition_penalty = 1.0f;

    std::vector<int32_t> generated;
    generated.reserve(max_tokens);

    auto t0 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < max_tokens; ++i) {
        if (stop_requested_.load()) break;

        int32_t input_token = output.back();

        HESA_CHECK(forward_single_token(input_token,
                                         static_cast<int32_t>(output.size()) - 1,
                                         hidden_buf_));

        logits_buf_.resize(std::max(logits_buf_.size(), vocab));
        HESA_CHECK(compute_logits(hidden_buf_, logits_buf_));

        int32_t next_token = sample_token(logits_buf_.data(),
                                           static_cast<int32_t>(vocab),
                                           output, gen_cfg, -1);

        generated.push_back(next_token);
        output.push_back(next_token);

        if ((i + 1) % 16 == 0 || i == max_tokens - 1) {
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            double tps = (i + 1) / elapsed;
            fprintf(stderr, "\r[Engine] Generated %zu/%zu (%.1f tok/s) ...",
                    i + 1, max_tokens, tps);
            fflush(stderr);
        }
    }

    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();
    fprintf(stderr, "\n[Engine] %zu tokens in %.3fs (%.1f tok/s)\n",
            generated.size(), elapsed,
            elapsed > 0 ? generated.size() / elapsed : 0.0);

    return generated;
}

} // namespace hesa

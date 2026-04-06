#include "hesa/engine.hpp"
#include "backend/cpu_backend.hpp"
#include "transformer_block.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>
#include <cstring>
#include <span>
#include <vector>

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

    // Get hyper-parameters
    size_t hidden_dim = meta.embedding_length;
    if (hidden_dim == 0) hidden_dim = 256;
    size_t n_layers = meta.block_count;
    size_t n_heads = meta.attention_head_count > 0 ? meta.attention_head_count : 32;
    size_t n_kv_heads = meta.attention_head_count_kv > 0
                       ? meta.attention_head_count_kv : n_heads;
    size_t head_dim = (n_heads > 0 && hidden_dim > 0)
                    ? hidden_dim / n_heads : 128;
    size_t ffn_dim = meta.feed_forward_length;
    if (ffn_dim == 0) ffn_dim = n_heads * head_dim * 8 / 3;
    size_t ctx_len = cfg.kv_cache_size > 0 ? cfg.kv_cache_size : 4096;

    // Create KV cache if we have valid hyper-params
    std::unique_ptr<KVCache> kv_cache;
    if (n_layers > 0 && n_heads > 0 && head_dim > 0) {
        KVCacheConfig kv_cfg;
        kv_cfg.max_seq_len = ctx_len;
        kv_cfg.n_layers    = n_layers;
        kv_cfg.n_heads     = n_heads;
        kv_cfg.head_dim    = head_dim;
        kv_cfg.n_kv_heads  = n_kv_heads;
        kv_cache = std::make_unique<KVCache>(kv_cfg);
        fprintf(stderr, "[Engine] KV cache: layers=%zu, heads=%zu, kv_heads=%zu, "
                        "head_dim=%zu, ctx=%zu\n",
                n_layers, n_heads, n_kv_heads, head_dim, ctx_len);
    }

    // Allocate per-layer buffers
    std::vector<LayerBuffers> layer_bufs;
    if (n_layers > 0 && ffn_dim > 0) {
        layer_bufs.resize(n_layers);
        for (size_t l = 0; l < n_layers; ++l) {
            bool ok = allocate_layer_buffers(layer_bufs[l],
                                            **backend, hidden_dim, ffn_dim,
                                            n_heads, n_kv_heads, head_dim);
            if (!ok) {
                fprintf(stderr, "[Engine] Warning: failed to allocate buffers for layer %zu\n", l);
            }
        }
        fprintf(stderr, "[Engine] Layer buffers allocated: %zu layers, ffn_dim=%zu\n",
                n_layers, ffn_dim);
    }

    // Construct the engine using a raw pointer since constructor is deleted
    auto* engine = new Engine();
    engine->backend_    = std::move(*backend);
    engine->model_      = std::move(*model);
    engine->tokenizer_  = std::move(*tokenizer);
    engine->kv_cache_   = kv_cache.get();
    // Keep the unique_ptr alive by releasing it
    (void)kv_cache.release(); // engine owns the pointer via raw pointer
    engine->layer_bufs_  = std::move(layer_bufs);
    engine->position_counter_ = 0;
    engine->stop_requested_.store(false);

    engine->hidden_buf_.resize(hidden_dim);
    engine->logits_buf_.resize(meta.vocab_size > 0 ? meta.vocab_size : 32000);

    fprintf(stderr, "[Engine] Engine ready (hidden=%zu, vocab=%zu)\n",
            hidden_dim, engine->logits_buf_.size());

    return std::unique_ptr<Engine>(engine);
}

Engine::Engine() = default;

Engine::~Engine() {
    // Free KV Cache (we own it via raw pointer)
    if (kv_cache_) {
        delete kv_cache_;
        kv_cache_ = nullptr;
    }
}

Result<void> Engine::stop() {
    stop_requested_.store(true);
    return ok();
}

// Forward pass - full transformer
Result<void> Engine::forward_single_token(int32_t token_id,
                                           int32_t position,
                                           std::vector<float>& out_hidden) {
    const auto& meta = model_->metadata();
    size_t hidden = meta.embedding_length;
    if (hidden == 0) hidden = hidden_buf_.size();
    size_t n_layers = meta.block_count;
    if (n_layers == 0) {
        // No layers — just do embedding lookup
        return ok();
    }

    float rms_eps = 1e-6f;
    if (meta.attention_layer_norm_rms_epsilon > 0) {
        rms_eps = static_cast<float>(meta.attention_layer_norm_rms_epsilon) * 1e-7f;
    }

    size_t n_heads = meta.attention_head_count > 0 ? meta.attention_head_count : 32;
    size_t n_kv_heads = meta.attention_head_count_kv > 0
                       ? meta.attention_head_count_kv : n_heads;
    size_t head_dim = (n_heads > 0 && hidden > 0) ? hidden / n_heads : 128;
    float rope_freq = meta.rope_freq_base;
    int rope_dims = meta.rope_dimension_count > 0
                  ? static_cast<int>(meta.rope_dimension_count)
                  : static_cast<int>(head_dim);

    // ─── Embedding lookup ──────────────────────────────────────────
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

    if (!emb || emb->dtype() != Dtype::F32 || emb->ndim() < 2) {
        return make_error<void>(Error::TENSOR_NOT_FOUND);
    }

    size_t emb_vocab  = static_cast<size_t>(emb->shape()[0]);
    size_t emb_hidden = static_cast<size_t>(emb->shape()[1]);

    if (static_cast<size_t>(token_id) >= emb_vocab) {
        return make_error<void>(Error::INVALID_ARGUMENT);
    }

    const float* edata = static_cast<const float*>(emb->data());
    if (out_hidden.size() < emb_hidden)
        return make_error<void>(Error::INVALID_ARGUMENT);

    std::memcpy(out_hidden.data(),
                edata + static_cast<size_t>(token_id) * emb_hidden,
                emb_hidden * sizeof(float));
    for (size_t i = emb_hidden; i < hidden; ++i)
        out_hidden[i] = 0.0f;

    // ─── Transformer layers ───────────────────────────────────────
    for (size_t l = 0; l < n_layers; ++l) {
        if (l < layer_bufs_.size() && kv_cache_) {
            HESA_CHECK(transformer_layer_forward(
                out_hidden,
                static_cast<int>(l),
                static_cast<int>(position),
                *model_,
                *backend_,
                *kv_cache_,
                n_heads,
                n_kv_heads,
                head_dim,
                rope_freq,
                rope_dims,
                rms_eps,
                layer_bufs_[l]
            ));
        } else {
            fprintf(stderr, "[Engine] Warning: skipping layer %zu (no buffers or cache)\n", l);
        }
    }

    // ─── Final RMSNorm ────────────────────────────────────────────
    Tensor const* final_norm_w = nullptr;
    for (const char* name : {
        "output_norm.weight",
        "norm.weight",
        "model.norm.weight",
    }) {
        auto r = model_->get_tensor(std::string(name));
        if (r) {
            final_norm_w = *r;
            break;
        }
    }

    if (final_norm_w && final_norm_w->dtype() == Dtype::F32) {
        HESA_CHECK(final_rms_norm(out_hidden, *final_norm_w, *backend_, rms_eps));
    }

    return ok();
}

// Logits computation
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
            for (size_t h = 0; h < std::min(hidden_dim, ow_hidden); ++h)
                sum += wd[v * ow_hidden + h] * hidden[h];
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
                for (size_t h = 0; h < std::min(hidden_dim, e_cols); ++h)
                    sum += ed[v * e_cols + h] * hidden[h];
                logits_out[v] = sum / std::sqrt(static_cast<float>(hidden_dim));
            }
        } else {
            fprintf(stderr, "[Engine] No output weight found; using hidden as logits\n");
            size_t copy_len = std::min(hidden_dim, logits_out.size());
            std::memcpy(logits_out.data(), hidden.data(), copy_len * sizeof(float));
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

    // Reset KV cache and position counter
    if (kv_cache_) kv_cache_->clear();
    position_counter_ = 0;

    GenerationConfig gen_cfg;
    gen_cfg.temperature = temperature;
    gen_cfg.top_p = top_p;
    gen_cfg.max_tokens = static_cast<int32_t>(max_tokens);
    gen_cfg.top_k = 40;
    gen_cfg.repetition_penalty = 1.0f;

    std::vector<int32_t> generated;
    generated.reserve(max_tokens);

    // ─── Prompt processing (skip if empty) ────────────────────────
    int32_t start_token = 0;

    if (!prompt.empty()) {
        // Process all but last prompt token
        for (size_t i = 0; i < prompt.size() - 1; ++i) {
            HESA_CHECK(forward_single_token(prompt[i],
                                            static_cast<int32_t>(i),
                                            hidden_buf_));
        }

        start_token = prompt.back();
        position_counter_ = static_cast<int32_t>(prompt.size());
    }

    // ─── First forward pass to produce initial logits ─────────────
    int32_t first_pos = position_counter_ > 0 ? position_counter_ - 1 : 0;
    HESA_CHECK(forward_single_token(start_token, first_pos, hidden_buf_));

    logits_buf_.resize(std::max(logits_buf_.size(), vocab));
    HESA_CHECK(compute_logits(hidden_buf_, logits_buf_));

    // Sample the first generated token
    std::vector<int32_t> full_ctx{prompt.begin(), prompt.end()};
    int32_t next_token = sample_token(logits_buf_.data(),
                                       static_cast<int32_t>(vocab),
                                       full_ctx, gen_cfg, -1);

    auto t0 = std::chrono::steady_clock::now();

    for (size_t i = 0; i < max_tokens; ++i) {
        if (stop_requested_.load()) break;

        generated.push_back(next_token);

        int32_t pos = position_counter_++;
        HESA_CHECK(forward_single_token(next_token, pos, hidden_buf_));

        logits_buf_.resize(std::max(logits_buf_.size(), vocab));
        HESA_CHECK(compute_logits(hidden_buf_, logits_buf_));

        full_ctx.push_back(next_token);
        next_token = sample_token(logits_buf_.data(),
                                   static_cast<int32_t>(vocab),
                                   full_ctx, gen_cfg, -1);

        if ((i + 1) % 16 == 0 || i == max_tokens - 1) {
            auto t1 = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(t1 - t0).count();
            double tps = (i + 1) / (elapsed > 0 ? elapsed : 0.001);
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

    // Return full sequence: prompt + generated
    std::vector<int32_t> output{prompt.begin(), prompt.end()};
    output.insert(output.end(), generated.begin(), generated.end());

    return output;
}

} // namespace hesa

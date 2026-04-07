#include "transformer_block.hpp"
#include "hesa/backend.hpp"
#include "hesa/kv_cache.hpp"
#include "hesa/model.hpp"
#include "hesa/result.hpp"
#include "hesa/tensor.hpp"
#include "hesa/simd.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace hesa {

/* ==========================================================================
 *  Helper: build layer tensor name prefix
 * ========================================================================== */
std::string layer_prefix(int layer, const std::string& arch) {
    (void)arch;
    return "blk." + std::to_string(layer) + ".";
}

/* ==========================================================================
 *  Helper: safe tensor lookup
 * ========================================================================== */
static const Tensor* get_tensor_safe(const Model& model, const std::string& name) {
    auto r = model.get_tensor(name);
    return r ? *r : nullptr;
}

/* ==========================================================================
 *  Helper: map hesa::Dtype to GGML qtype code used by simd::matvec_dequant
 * ========================================================================== */
static int dtype_to_qtype(Dtype dt) {
    switch (dt) {
        case Dtype::Q4_0:  return 2;
        case Dtype::Q4_1:  return 3;
        case Dtype::Q5_0:  return 6;
        case Dtype::Q5_1:  return 7;
        case Dtype::Q8_0:  return 8;
        case Dtype::Q2_K:  return 10;
        case Dtype::Q3_K:  return 11;
        case Dtype::Q4_K:  return 12;
        case Dtype::Q5_K:  return 13;
        case Dtype::Q6_K:  return 14;
        default:           return -1; // non-quantized
    }
}

/* ==========================================================================
 *  quantized_matvec
 *
 *  Matrix-vector multiply: out[m] += sum_n(w[m, n] * x[n])
 *
 *  The weight tensor is [M, N] stored row-major in quantized format.
 *  x is [N] (float).  out is [M] (float), accumulated.
 *
 *  Dispatcher:
 *    - Q4_0 / Q8_0 → simd::matvec_dequant
 *    - Q4_K / Q5_K / Q6_K → simd::matvec_dequant
 *    - F16 / BF16 → on-the-fly fp16 dequant
 *    - F32 → SIMD dot-product / naive loop
 *  ========================================================================== */

static void quantized_matvec_2d(const Tensor& weight, const float* x, float* out,
                                 int M, int N) {
    // weight shape is [M, N] in C-contiguous (row-major) layout
    Dtype dt = weight.dtype();
    const void* wdata = weight.data();

    // ── Q4_0, Q5_0, Q5_1, Q8_0 → matvec_dequant ──────────────────
    int qtype = dtype_to_qtype(dt);
    if (qtype >= 0) {
        const uint8_t* wu8 = static_cast<const uint8_t*>(wdata);
        simd::matvec_dequant(wu8, x, out, M, N, 0, qtype);
        return;
    }

    // ── F16 → on-the-fly dequant ──────────────────────────────────
    if (dt == Dtype::F16) {
        const uint16_t* wh = static_cast<const uint16_t*>(wdata);
        for (int m = 0; m < M; ++m) {
            float dot = 0.0f;
            for (int n = 0; n < N; ++n) {
                float wf = simd::fp16_to_f32(wh[m * N + n]);
                dot += wf * x[n];
            }
            out[m] += dot;
        }
        return;
    }

    // ── BF16 → on-the-fly dequant ─────────────────────────────────
    if (dt == Dtype::BF16) {
        const uint16_t* wh = static_cast<const uint16_t*>(wdata);
        for (int m = 0; m < M; ++m) {
            float dot = 0.0f;
            for (int n = 0; n < N; ++n) {
                // BF16: just zero-extend the lower 16 bits
                uint32_t bits = static_cast<uint32_t>(wh[m * N + n]) << 16;
                float wf;
                std::memcpy(&wf, &bits, 4);
                dot += wf * x[n];
            }
            out[m] += dot;
        }
        return;
    }

    // ── F32 → SIMD matvec ─────────────────────────────────────────
    if (dt == Dtype::F32) {
        const float* wf = static_cast<const float*>(wdata);
        for (int m = 0; m < M; ++m) {
            // Use SIMD dot product
            float dot = simd::dot_f32(wf + m * N, x, N);
            out[m] += dot;
        }
        return;
    }

    // ── I8 (linear) ───────────────────────────────────────────────
    if (dt == Dtype::I8) {
        const int8_t* wi8 = static_cast<const int8_t*>(wdata);
        for (int m = 0; m < M; ++m) {
            float dot = 0.0f;
            for (int n = 0; n < N; ++n) {
                dot += static_cast<float>(wi8[m * N + n]) * x[n];
            }
            out[m] += dot;
        }
        return;
    }

    // ── Fallback: treat as F32 (with warning) ─────────────────────
    fprintf(stderr, "[quantized_matvec] Unknown dtype %s, treating as F32\n",
            dtype_name(dt));
    const float* wf = static_cast<const float*>(wdata);
    if (wf) {
        for (int m = 0; m < M; ++m) {
            float dot = simd::dot_f32(wf + m * N, x, N);
            out[m] += dot;
        }
    }
}

/* ==========================================================================
 *  Allocate layer buffers
 * ========================================================================== */
bool allocate_layer_buffers(LayerBuffers& buf,
                            Backend& backend,
                            size_t hidden_dim,
                            size_t ffn_inter_dim,
                            size_t n_query_heads,
                            size_t n_kv_heads,
                            size_t head_dim) {
    auto alloc = [&](Tensor& t, std::initializer_list<int64_t> shape) -> bool {
        auto r = backend.alloc_tensor(Dtype::F32, shape);
        if (!r) return false;
        t = std::move(*r);
        return true;
    };

    if (!alloc(buf.attn_norm,   {static_cast<int64_t>(hidden_dim)}))     return false;
    if (!alloc(buf.attn_out,    {static_cast<int64_t>(hidden_dim)}))     return false;
    if (!alloc(buf.ff_norm,     {static_cast<int64_t>(hidden_dim)}))     return false;
    if (!alloc(buf.ff_gate,     {static_cast<int64_t>(ffn_inter_dim)})) return false;
    if (!alloc(buf.ff_up,       {static_cast<int64_t>(ffn_inter_dim)})) return false;
    if (!alloc(buf.ff_down,     {static_cast<int64_t>(hidden_dim)}))    return false;
    if (!alloc(buf.silu_gate,   {static_cast<int64_t>(ffn_inter_dim)})) return false;
    if (!alloc(buf.gate_up,     {static_cast<int64_t>(ffn_inter_dim)})) return false;
    if (!alloc(buf.q_buf,       {static_cast<int64_t>(n_query_heads),
                                 static_cast<int64_t>(head_dim)}))       return false;
    if (!alloc(buf.k_buf,       {static_cast<int64_t>(n_kv_heads),
                                 static_cast<int64_t>(head_dim)}))       return false;
    if (!alloc(buf.v_buf,       {static_cast<int64_t>(n_kv_heads),
                                 static_cast<int64_t>(head_dim)}))       return false;

    return true;
}

/* ==========================================================================
 *  Transformer layer forward pass
 * ========================================================================== */
Result<void> transformer_layer_forward(
    std::vector<float>& hidden,
    int layer,
    int position,
    const Model& model,
    Backend& backend,
    KVCache& kv_cache,
    size_t n_query_heads,
    size_t n_kv_heads,
    size_t head_dim,
    float rope_freq_base,
    int rope_dims,
    float rms_eps,
    LayerBuffers& buf)
{
    const std::string prefix = layer_prefix(layer, model.metadata().architecture);
    size_t hidden_dim = hidden.size();
    float* hn = hidden.data();

    // ─── 1. Attention RMSNorm ──────────────────────────────────────────────
    const Tensor* attn_norm_w = get_tensor_safe(model, prefix + "attn_norm.weight");
    if (!attn_norm_w) {
        return make_error<void>(Error::TENSOR_NOT_FOUND);
    }

    TensorView hidden_view(hn, Dtype::F32, Shape{static_cast<int64_t>(hidden_dim)});
    HESA_CHECK(backend.rms_norm(hidden_view, attn_norm_w->view(), buf.attn_norm.view(), rms_eps));

    hn = hidden.data();  // re-read (same pointer)
    const float* attn_in = static_cast<const float*>(buf.attn_norm.data());

    // ─── 2. Q, K, V projections ───────────────────────────────────────────
    const Tensor* wq = get_tensor_safe(model, prefix + "attn_q.weight");
    const Tensor* wk = get_tensor_safe(model, prefix + "attn_k.weight");
    const Tensor* wv = get_tensor_safe(model, prefix + "attn_v.weight");
    if (!wq || !wk || !wv) {
        fprintf(stderr, "[Layer %d] Missing attention weights\n", layer);
        return make_error<void>(Error::TENSOR_NOT_FOUND);
    }

    size_t q_proj_dim = n_query_heads * head_dim;
    size_t k_proj_dim = n_kv_heads * head_dim;

    // Q = Wq @ attn_in   — uses quantized_matvec if weights are quantized
    std::vector<float> q_vec(q_proj_dim, 0.0f);
    quantized_matvec_2d(*wq, attn_in, q_vec.data(),
                        static_cast<int>(q_proj_dim), static_cast<int>(hidden_dim));

    // K = Wk @ attn_in
    std::vector<float> k_vec(k_proj_dim, 0.0f);
    quantized_matvec_2d(*wk, attn_in, k_vec.data(),
                        static_cast<int>(k_proj_dim), static_cast<int>(hidden_dim));

    // V = Wv @ attn_in
    std::vector<float> v_vec(k_proj_dim, 0.0f);
    quantized_matvec_2d(*wv, attn_in, v_vec.data(),
                        static_cast<int>(k_proj_dim), static_cast<int>(hidden_dim));

    // ─── 3. Split Q into heads and apply RoPE ──────────────────────────────
    float* q_data = static_cast<float*>(buf.q_buf.data());
    float* k_data = static_cast<float*>(buf.k_buf.data());
    float* v_data = static_cast<float*>(buf.v_buf.data());

    for (size_t h = 0; h < n_query_heads; ++h)
        std::memcpy(q_data + h * head_dim, q_vec.data() + h * head_dim,
                    head_dim * sizeof(float));

    // Apply RoPE to Q [1, n_query_heads, head_dim]
    std::vector<float> q_rope_buf(n_query_heads * head_dim);
    std::memcpy(q_rope_buf.data(), q_data, q_proj_dim * sizeof(float));

    int32_t pos_int = static_cast<int32_t>(position);
    TensorView q_rope_view(q_rope_buf.data(), Dtype::F32,
                           Shape{1, static_cast<int64_t>(n_query_heads),
                                 static_cast<int64_t>(head_dim)});
    HESA_CHECK(backend.rope(q_rope_view, {&pos_int, 1}, rope_freq_base,
                            rope_dims > 0 ? rope_dims : static_cast<int>(head_dim)));
    std::memcpy(q_data, q_rope_buf.data(), q_proj_dim * sizeof(float));

    // ─── 4. Split K, apply RoPE, write to KV cache ─────────────────────────
    for (size_t h = 0; h < n_kv_heads; ++h)
        std::memcpy(k_data + h * head_dim, k_vec.data() + h * head_dim,
                    head_dim * sizeof(float));

    for (size_t h = 0; h < n_kv_heads; ++h)
        std::memcpy(v_data + h * head_dim, v_vec.data() + h * head_dim,
                    head_dim * sizeof(float));

    // Write K, V to KV cache
    for (size_t h = 0; h < n_kv_heads; ++h) {
        kv_cache.write(static_cast<size_t>(layer), h,
                       static_cast<size_t>(position),
                       k_data + h * head_dim,
                       v_data + h * head_dim);
    }
    kv_cache.advance(1);

    // ─── 5. Read full context from KV cache ─────────────────────────────────
    size_t ctx_len = kv_cache.used();
    if (ctx_len == 0) ctx_len = 1;

    std::vector<float> k_cached(ctx_len * n_kv_heads * head_dim);
    std::vector<float> v_cached(ctx_len * n_kv_heads * head_dim);

    for (size_t t = 0; t < ctx_len; ++t) {
        for (size_t h = 0; h < n_kv_heads; ++h) {
            size_t phys = kv_cache.physical_pos(t);
            const auto& kt = kv_cache.key_tensor(static_cast<size_t>(layer), h);
            const auto& vt = kv_cache.value_tensor(static_cast<size_t>(layer), h);
            const float* k_full = static_cast<const float*>(kt.data());
            const float* v_full = static_cast<const float*>(vt.data());

            std::memcpy(k_cached.data() + (t * n_kv_heads + h) * head_dim,
                        k_full + phys * head_dim, head_dim * sizeof(float));
            std::memcpy(v_cached.data() + (t * n_kv_heads + h) * head_dim,
                        v_full + phys * head_dim, head_dim * sizeof(float));
        }
    }

    // ─── 6. Scaled Dot-Product Attention ───────────────────────────────────
    std::vector<float> attn_out(n_query_heads * head_dim, 0.0f);

    if (n_query_heads == n_kv_heads) {
        for (size_t h = 0; h < n_query_heads; ++h) {
            const float* qh = q_data + h * head_dim;
            const float* kh = k_cached.data() + h * head_dim;
            const float* vh = v_cached.data() + h * head_dim;
            float* oh = attn_out.data() + h * head_dim;

            std::vector<float> scores(ctx_len);
            float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));

            for (size_t t = 0; t < ctx_len; ++t) {
                float dot = 0.0f;
                const float* kt = kh + t * head_dim;
                for (size_t d = 0; d < head_dim; ++d)
                    dot += qh[d] * kt[d];
                scores[t] = dot * inv_sqrt_d;
            }

            float max_s = scores[0];
            for (size_t t = 1; t < ctx_len; ++t)
                if (scores[t] > max_s) max_s = scores[t];
            float sum_s = 0.0f;
            for (size_t t = 0; t < ctx_len; ++t) {
                scores[t] = std::exp(scores[t] - max_s);
                sum_s += scores[t];
            }
            float inv_sum = 1.0f / std::max(sum_s, 1e-30f);
            for (size_t t = 0; t < ctx_len; ++t)
                scores[t] *= inv_sum;

            for (size_t d = 0; d < head_dim; ++d) {
                float val = 0.0f;
                for (size_t t = 0; t < ctx_len; ++t)
                    val += scores[t] * (vh + t * head_dim)[d];
                oh[d] = val;
            }
        }
    } else {
        size_t q_per_kv = n_query_heads / n_kv_heads;

        for (size_t kv_h = 0; kv_h < n_kv_heads; ++kv_h) {
            const float* kh = k_cached.data() + kv_h * head_dim;
            const float* vh = v_cached.data() + kv_h * head_dim;

            std::vector<float> scores(ctx_len);

            for (size_t qi = 0; qi < q_per_kv; ++qi) {
                size_t qh = kv_h * q_per_kv + qi;
                const float* qh_ptr = q_data + qh * head_dim;
                float* oh_ptr = attn_out.data() + qh * head_dim;

                float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(head_dim));
                for (size_t t = 0; t < ctx_len; ++t) {
                    float dot = 0.0f;
                    const float* kt = kh + t * head_dim;
                    for (size_t d = 0; d < head_dim; ++d)
                        dot += qh_ptr[d] * kt[d];
                    scores[t] = dot * inv_sqrt_d;
                }

                float max_s = scores[0];
                for (size_t t = 1; t < ctx_len; ++t)
                    if (scores[t] > max_s) max_s = scores[t];
                float sum_s = 0.0f;
                for (size_t t = 0; t < ctx_len; ++t) {
                    scores[t] = std::exp(scores[t] - max_s);
                    sum_s += scores[t];
                }
                for (size_t t = 0; t < ctx_len; ++t)
                    scores[t] *= (1.0f / std::max(sum_s, 1e-30f));

                for (size_t d = 0; d < head_dim; ++d) {
                    float val = 0.0f;
                    for (size_t t = 0; t < ctx_len; ++t)
                        val += scores[t] * (vh + t * head_dim)[d];
                    oh_ptr[d] = val;
                }
            }
        }
    }

    // ─── 7. Attention output projection ────────────────────────────────────
    const Tensor* wo = get_tensor_safe(model, prefix + "attn_output.weight");
    if (!wo) {
        fprintf(stderr, "[Layer %d] Missing attn_output.weight\n", layer);
        return make_error<void>(Error::TENSOR_NOT_FOUND);
    }

    std::vector<float> attn_proj(hidden_dim, 0.0f);
    quantized_matvec_2d(*wo, attn_out.data(), attn_proj.data(),
                        static_cast<int>(hidden_dim),
                        static_cast<int>(n_query_heads * head_dim));

    // ─── 8. Residual connection (post-attention) ───────────────────────────
    for (size_t i = 0; i < hidden_dim; ++i)
        hn[i] += attn_proj[i];

    // ─── 9. FFN RMSNorm ───────────────────────────────────────────────────
    const Tensor* ffn_norm_w = get_tensor_safe(model, prefix + "ffn_norm.weight");
    if (!ffn_norm_w) {
        return make_error<void>(Error::TENSOR_NOT_FOUND);
    }

    TensorView hidden_view2(hn, Dtype::F32, Shape{static_cast<int64_t>(hidden_dim)});
    HESA_CHECK(backend.rms_norm(hidden_view2, ffn_norm_w->view(), buf.ff_norm.view(), rms_eps));

    const float* fn = static_cast<const float*>(buf.ff_norm.data());

    // ─── 10. SwiGLU FFN ──────────────────────────────────────────────────
    size_t ffn_inter = model.metadata().feed_forward_length;
    if (ffn_inter == 0) {
        ffn_inter = static_cast<size_t>(model.metadata().attention_head_count) *
                    head_dim * 8 / 3;
    }

    const Tensor* w_gate = get_tensor_safe(model, prefix + "ffn_gate.weight");
    const Tensor* w_up   = get_tensor_safe(model, prefix + "ffn_up.weight");
    const Tensor* w_down = get_tensor_safe(model, prefix + "ffn_down.weight");

    if (!w_gate || !w_up || !w_down) {
        fprintf(stderr, "[Layer %d] Missing FFN weights\n", layer);
        return make_error<void>(Error::TENSOR_NOT_FOUND);
    }

    /// gate = fn @ W_gate^T  [ffn_inter]
    std::vector<float> gate_v(ffn_inter, 0.0f);
    quantized_matvec_2d(*w_gate, fn, gate_v.data(),
                        static_cast<int>(ffn_inter), static_cast<int>(hidden_dim));

    /// up = fn @ W_up^T  [ffn_inter]
    std::vector<float> up_v(ffn_inter, 0.0f);
    quantized_matvec_2d(*w_up, fn, up_v.data(),
                        static_cast<int>(ffn_inter), static_cast<int>(hidden_dim));

    /// SiLU(gate) * up
    for (size_t d = 0; d < ffn_inter; ++d) {
        float x = gate_v[d];
        gate_v[d] = (x / (1.0f + std::exp(-x))) * up_v[d];
    }

    /// down = gate_v @ W_down^T  [hidden]
    std::vector<float> down_proj(hidden_dim, 0.0f);
    quantized_matvec_2d(*w_down, gate_v.data(), down_proj.data(),
                        static_cast<int>(hidden_dim), static_cast<int>(ffn_inter));
    for (size_t o = 0; o < hidden_dim; ++o)
        hn[o] += down_proj[o];  // residual

    return ok();
}

/* ==========================================================================
 *  Final RMSNorm
 * ========================================================================== */
Result<void> final_rms_norm(std::vector<float>& hidden,
                            const Tensor& norm_weight,
                            Backend& backend,
                            float eps)
{
    size_t dim = hidden.size();
    TensorView in_view(hidden.data(), Dtype::F32, Shape{static_cast<int64_t>(dim)});
    // Write in-place (backend.rms_norm writes to out, reading from in)
    // For in-place: we read from in and write to out, but they point to same data.
    // Need a temp buffer to avoid race.
    std::vector<float> temp(dim);
    TensorView out_view(temp.data(), Dtype::F32, Shape{static_cast<int64_t>(dim)});
    HESA_CHECK(backend.rms_norm(in_view, norm_weight.view(), out_view, eps));
    std::memcpy(hidden.data(), temp.data(), dim * sizeof(float));
    return ok();
}

} // namespace hesa

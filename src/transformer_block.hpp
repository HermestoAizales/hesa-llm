#ifndef HESA_TRANSFORMER_BLOCK_HPP
#define HESA_TRANSFORMER_BLOCK_HPP

#include "hesa/backend.hpp"
#include "hesa/kv_cache.hpp"
#include "hesa/model.hpp"
#include "hesa/result.hpp"
#include "hesa/tensor.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace hesa {

/**
 * Temporary buffers used by a single transformer layer forward pass.
 * Allocated once per layer and reused across tokens.
 */
struct LayerBuffers {
    Tensor attn_norm;   // [hidden] — after attention RMSNorm
    Tensor attn_out;    // [hidden] — output of attention projection
    Tensor ff_norm;     // [hidden] — after FFN RMSNorm
    Tensor ff_gate;     // [ffn_inter] — gate projection result
    Tensor ff_up;       // [ffn_inter] — up projection result
    Tensor ff_down;     // [hidden] — down projection result
    Tensor silu_gate;   // [ffn_inter] — SiLU(gate)
    Tensor gate_up;     // [ffn_inter] — silu_gate * up

    // Q, K, V per-head buffers for attention
    Tensor q_buf;       // [n_heads, head_dim]
    Tensor k_buf;       // [n_kv_heads, head_dim]
    Tensor v_buf;       // [n_kv_heads, head_dim]

    // Q expanded to query heads for GQA (if n_kv_heads < n_heads)
    Tensor q_expanded;  // [n_kv_heads, q_heads_per_kv, head_dim]
};

/**
 * Compute the GGUF/GGML layer name prefix for a given block layer.
 */
std::string layer_prefix(int layer, const std::string& arch);

/**
 * Allocate temporary buffers for one transformer layer.
 * Returns false if allocation failed.
 */
bool allocate_layer_buffers(LayerBuffers& buf,
                            Backend& backend,
                            size_t hidden_dim,
                            size_t ffn_inter_dim,
                            size_t n_query_heads,
                            size_t n_kv_heads,
                            size_t head_dim);

/**
 * Forward pass for a single transformer layer.
 *
 * Dataflow:
 *   x_norm = RMSNorm(x) * attn_norm.weight
 *   q = x_norm @ Q_weight.T    [n_query_heads, head_dim]
 *   k = x_norm @ K_weight.T    [n_kv_heads, head_dim]
 *   v = x_norm @ V_weight.T    [n_kv_heads, head_dim]
 *   Apply RoPE to q and k (in-place)
 *   Write k, v to KV cache
 *   Read K, V for positions [0..pos] from cache
 *   Reshape Q to [1, n_kv_heads, q_per_kv, head_dim] for GQA
 *   SDPA(Q, K_cached, V_cached) → attn_out_per_head
 *   Reshape & project: attn_out = attn_heads @ O_weight.T
 *   x = x + attn_out     (residual)
 *
 *   x_norm2 = RMSNorm(x) * ffn_norm.weight
 *   gate = x_norm2 @ Gate_weight.T    [ffn_inter]
 *   up   = x_norm2 @ Up_weight.T      [ffn_inter]
 *   silu(gate) * up → gate_up
 *   x = x + (gate_up @ Down_weight.T)
 *
 * @param hidden           Input/output hidden state [hidden_dim]. Modified in-place.
 * @param layer            Layer index (0-based).
 * @param position         Current token position in sequence.
 * @param model            Loaded model with all GGUF weights.
 * @param backend          Compute backend.
 * @param kv_cache         KV cache for attention.
 * @param n_query_heads    Number of query heads.
 * @param n_kv_heads       Number of KV heads (GQA).
 * @param head_dim         Per-head dimension.
 * @param rope_freq_base   RoPE frequency base (e.g., 10000.0).
 * @param rope_dims        Number of RoPE dimensions.
 * @param rms_eps          RMSNorm epsilon.
 * @param buf              Pre-allocated temporary buffers for this layer.
 * @return ok() on success, error otherwise.
 */
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
    LayerBuffers& buf
);

/**
 * Apply final RMSNorm to hidden state before LM head.
 */
Result<void> final_rms_norm(std::vector<float>& hidden,
                            const Tensor& norm_weight,
                            Backend& backend,
                            float eps = 1e-6f);

} // namespace hesa

#endif // HESA_TRANSFORMER_BLOCK_HPP

#include "backend/cpu_backend.hpp"
#include "hesa/tensor.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <thread>

// -- SIMD headers --
#if defined(__AVX2__)
#include <immintrin.h>
#endif

// -- Fallback: detect SIMD at compile time on x86 --
#if !defined(__AVX__) && !defined(__AVX2__)
#define HESA_CPU_SCALAR 1
#endif

namespace hesa {

// ============================================================================
//  CPU Reference Kernels (scalar & SIMD)
// ============================================================================

void cpu_matmul_f32(const float* a, const float* b, float* out,
                    int M, int N, int K, float scale) {
#if HESA_CPU_SCALAR
    // Scalar reference:  C = scale * A @ B
    // A: [M x K], B: [K x N], C: [M x N]
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            out[i * N + j] = sum * scale;
        }
    }
#else
    (void)a; (void)b; (void)out; (void)M; (void)N; (void)K; (void)scale;
#endif
}

void cpu_add_f32(const float* a, const float* b, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i)
        out[i] = a[i] + b[i];
}

void cpu_softmax_f32(const float* in, float* out, size_t n_rows, size_t row_size) {
    for (size_t i = 0; i < n_rows; ++i) {
        const float* row = in + i * row_size;
        float* o = out + i * row_size;

        // Max for numerical stability
        float max_val = row[0];
        for (size_t j = 1; j < row_size; ++j)
            if (row[j] > max_val) max_val = row[j];

        // Exp and sum
        float sum = 0.0f;
        for (size_t j = 0; j < row_size; ++j) {
            o[j] = std::exp(row[j] - max_val);
            sum += o[j];
        }

        // Normalize
        float inv_sum = 1.0f / sum;
        for (size_t j = 0; j < row_size; ++j)
            o[j] *= inv_sum;
    }
}

void cpu_rms_norm_f32(const float* in, const float* weight, float* out,
                      size_t n_rows, size_t row_size, float eps) {
    for (size_t i = 0; i < n_rows; ++i) {
        const float* row = in + i * row_size;
        float* o = out + i * row_size;
        const float* w = weight;

        // Compute RMS
        float sum_sq = 0.0f;
        for (size_t j = 0; j < row_size; ++j)
            sum_sq += row[j] * row[j];
        float inv_rms = 1.0f / std::sqrt(sum_sq / row_size + eps);

        // Normalize and scale
        for (size_t j = 0; j < row_size; ++j)
            o[j] = row[j] * inv_rms * w[j];
    }
}

void cpu_silu_f32(const float* in, float* out, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        float x = in[i];
        out[i] = x / (1.0f + std::exp(-x)); // x * sigmoid(x)
    }
}

void cpu_rope_f32(float* data, const int32_t* positions,
                  size_t n_tokens, size_t n_heads, size_t head_dim,
                  float freq_base) {
    // data shape: [n_tokens, n_heads, head_dim]
    // Apply RoPE to the first half of head_dim
    size_t half = head_dim / 2;
    for (size_t t = 0; t < n_tokens; ++t) {
        int32_t pos = positions ? positions[t] : static_cast<int32_t>(t);
        for (size_t h = 0; h < n_heads; ++h) {
            float* row = data + (t * n_heads + h) * head_dim;
            for (size_t d = 0; d < half; ++d) {
                float freq = 1.0f / std::pow(freq_base, 2.0f * d / head_dim);
                float theta = pos * freq;
                float cos_t = std::cos(theta);
                float sin_t = std::sin(theta);
                float x0 = row[d];
                float x1 = row[d + half];
                row[d]       = x0 * cos_t - x1 * sin_t;
                row[d + half] = x0 * sin_t + x1 * cos_t;
            }
        }
    }
}

// ============================================================================
//  CPU_Backend
// ============================================================================

CPU_Backend::CPU_Backend(DeviceConfig cfg) {
    n_threads_ = cfg.n_threads > 0 ? cfg.n_threads :
                 static_cast<int>(std::thread::hardware_concurrency());
    if (n_threads_ <= 0) n_threads_ = 1;
}

CPU_Backend::~CPU_Backend() = default;

Result<Tensor> CPU_Backend::alloc_tensor(Dtype dtype, std::span<const int64_t> shape) {
    Tensor t(dtype, shape, this);
    if (t.nelements() == 0)
        return make_error<Tensor>(Error::INVALID_ARGUMENT);
    return t;
}

Result<void> CPU_Backend::free_tensor(Tensor& tensor) {
    tensor = Tensor{};
    return ok();
}

Result<void> CPU_Backend::matmul(TensorView a, TensorView b, TensorView out, float scale) {
    // Simple matmul via reference kernel
    // Views may be strided — for now require contiguous 2D
    // Full strided support later
    const float* fa = static_cast<const float*>(a.data());
    const float* fb = static_cast<const float*>(b.data());
    float* fo = static_cast<float*>(out.data());

    int M = a.shape()[0];
    int N = b.shape()[1];
    int K = a.shape()[1];

    cpu_matmul_f32(fa, fb, fo, M, N, K, scale);
    return ok();
}

Result<void> CPU_Backend::add(TensorView a, TensorView b, TensorView out) {
    const float* fa = static_cast<const float*>(a.data());
    const float* fb = static_cast<const float*>(b.data());
    float* fo = static_cast<float*>(out.data());
    cpu_add_f32(fa, fb, fo, a.nelements());
    return ok();
}

Result<void> CPU_Backend::mul(TensorView a, TensorView b, TensorView out) {
    const float* fa = static_cast<const float*>(a.data());
    const float* fb = static_cast<const float*>(b.data());
    float* fo = static_cast<float*>(out.data());
    for (size_t i = 0; i < a.nelements(); ++i)
        fo[i] = fa[i] * fb[i];
    return ok();
}

Result<void> CPU_Backend::softmax(TensorView in, TensorView out, int axis) {
    (void)axis; // Only supports last axis for now
    const float* fi = static_cast<const float*>(in.data());
    float* fo = static_cast<float*>(out.data());
    size_t n_rows = 1;
    size_t row_size = in.shape().nelements();
    if (in.ndim() >= 2) {
        // Flatten all but last dim
        row_size = in.shape()[in.ndim() - 1];
        n_rows = in.nelements() / row_size;
    }
    cpu_softmax_f32(fi, fo, n_rows, row_size);
    return ok();
}

Result<void> CPU_Backend::rms_norm(TensorView in, TensorView weight,
                                     TensorView out, float eps) {
    const float* fi = static_cast<const float*>(in.data());
    const float* fw = static_cast<const float*>(weight.data());
    float* fo = static_cast<float*>(out.data());
    size_t row_size = in.shape()[in.ndim() - 1];
    size_t n_rows = in.nelements() / row_size;
    cpu_rms_norm_f32(fi, fw, fo, n_rows, row_size, eps);
    return ok();
}

Result<void> CPU_Backend::rope(TensorView in_out, std::span<const int32_t> positions,
                                float freq_base, int n_dims) {
    float* data = static_cast<float*>(in_out.data());
    size_t n_tokens = 1;
    size_t n_heads = 1;
    if (in_out.ndim() >= 3) {
        n_tokens = in_out.shape()[0];
        n_heads = in_out.shape()[1];
    }
    cpu_rope_f32(data, positions.data(), n_tokens, n_heads, n_dims, freq_base);
    return ok();
}

Result<void> CPU_Backend::silu(TensorView in, TensorView out) {
    const float* fi = static_cast<const float*>(in.data());
    float* fo = static_cast<float*>(out.data());
    cpu_silu_f32(fi, fo, in.nelements());
    return ok();
}

Result<void> CPU_Backend::scaled_dot_product_attention(
    TensorView q, TensorView k, TensorView v, TensorView out,
    TensorView mask, float scale) {
    (void)mask;
    (void)scale;
    // Naive SDPA: out = softmax(Q @ K^T / sqrt(d)) @ V
    // For now: just implement as a fused matmul + softmax + matmul
    // Q: [n_tokens_q, n_heads, head_dim]
    // K: [n_tokens_k, n_heads, head_dim]
    // V: [n_tokens_k, n_heads, head_dim]
    // out: [n_tokens_q, n_heads, head_dim]

    int n_q = q.shape()[0];
    int n_k = k.shape()[0];
    int n_heads = q.shape()[1];
    int head_dim = q.shape()[2];

    const float* fq = static_cast<const float*>(q.data());
    const float* fk = static_cast<const float*>(k.data());
    const float* fv = static_cast<const float*>(v.data());
    float* fo = static_cast<float*>(out.data());

    // Temporary: scores [n_heads, n_q, n_k]
    std::vector<float> scores(n_heads * n_q * n_k);

    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < n_q; ++i) {
            for (int j = 0; j < n_k; ++j) {
                float sum = 0.0f;
                const float* qi = fq + (i * n_heads + h) * head_dim;
                const float* kj = fk + (j * n_heads + h) * head_dim;
                for (int d = 0; d < head_dim; ++d)
                    sum += qi[d] * kj[d];
                if (scale > 0.0f) sum *= scale;
                else sum *= 1.0f / std::sqrt(static_cast<float>(head_dim));
                scores[h * n_q * n_k + i * n_k + j] = sum;
            }
        }
    }

    // Softmax over last dim (n_k)
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < n_q; ++i) {
            float* row = &scores[h * n_q * n_k + i * n_k];
            float max_val = row[0];
            for (int j = 1; j < n_k; ++j) if (row[j] > max_val) max_val = row[j];
            float sum = 0.0f;
            for (int j = 0; j < n_k; ++j) { row[j] = std::exp(row[j] - max_val); sum += row[j]; }
            float inv = 1.0f / sum;
            for (int j = 0; j < n_k; ++j) row[j] *= inv;
        }
    }

    // out = scores @ V
    float inv_scale = (scale > 0.0f) ? 1.0f : 1.0f; // already applied above
    (void)inv_scale;
    for (int h = 0; h < n_heads; ++h) {
        for (int i = 0; i < n_q; ++i) {
            for (int j = 0; j < n_k; ++j) {
                for (int d = 0; d < head_dim; ++d) {
                    float* oi_ptr = &fo[(i * n_heads + h) * head_dim + d];
                    const float* vj_ptr = &fv[(j * n_heads + h) * head_dim + d];
                    *oi_ptr += scores[h * n_q * n_k + i * n_k + j] * *vj_ptr;
                }
            }
        }
    }

    return ok();
}

Result<Tensor> CPU_Backend::quantize(TensorView src, Dtype target_dtype) {
    // Q4_0 placeholder: simply round and pack
    // Full implementation in Phase 2
    (void)src;
    (void)target_dtype;
    return make_error<Tensor>(Error::BACKEND_NOT_AVAILABLE);
}

Result<void> CPU_Backend::copy_tensor(const Tensor& src, Tensor& dst) {
    std::memcpy(dst.data(), src.data(), std::min(src.nbytes(), dst.nbytes()));
    return ok();
}

} // namespace hesa

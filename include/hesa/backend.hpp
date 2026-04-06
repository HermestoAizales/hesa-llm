#ifndef HESA_BACKEND_HPP
#define HESA_BACKEND_HPP

#include "hesa/result.hpp"
#include "hesa/tensor.hpp"

#include <cstdint>
#include <functional>
#include <span>
#include <string>

namespace hesa {

enum class BackendType : uint8_t {
    AUTO,
    CPU_X86,
    CPU_ARM,
    CUDA,
    METAL
};

struct DeviceConfig {
    int device_id = 0;
    size_t memory_limit = 0; // 0 = no limit (use all available)
    int n_threads = 0;       // 0 = auto-detect
};

class Backend {
public:
    virtual ~Backend() = default;

    virtual BackendType type() const = 0;
    virtual const char* name() const = 0;
    virtual bool supports(Dtype dt) const = 0;
    virtual size_t device_memory() const = 0;
    virtual size_t device_memory_used() const = 0;

    // -- Tensor allocation / free --
    virtual Result<Tensor> alloc_tensor(Dtype dtype, std::span<const int64_t> shape) = 0;
    virtual Result<void> free_tensor(Tensor& tensor) = 0;

    // -- Core operations --
    virtual Result<void> matmul(TensorView a, TensorView b, TensorView out,
                                float scale = 1.0f) = 0;
    virtual Result<void> add(TensorView a, TensorView b, TensorView out) = 0;
    virtual Result<void> mul(TensorView a, TensorView b, TensorView out) = 0;
    virtual Result<void> softmax(TensorView in, TensorView out, int axis = -1) = 0;
    virtual Result<void> rms_norm(TensorView in, TensorView weight,
                                  TensorView out, float eps = 1e-6f) = 0;
    virtual Result<void> rope(TensorView in_out, std::span<const int32_t> positions,
                              float freq_base, int n_dims) = 0;
    virtual Result<void> silu(TensorView in, TensorView out) = 0;

    // -- Attention --
    virtual Result<void> scaled_dot_product_attention(
        TensorView q, TensorView k, TensorView v, TensorView out,
        TensorView mask = {}, float scale = 0.0f) = 0;

    // -- Quantization --
    virtual Result<Tensor> quantize(TensorView src, Dtype target_dtype) = 0;

    // -- Memory management --
    virtual Result<void> synchronize() = 0;
    virtual Result<void> copy_tensor(const Tensor& src, Tensor& dst) = 0;
};

// Backend factory
Result<std::unique_ptr<Backend>> create_backend(BackendType type,
                                                 DeviceConfig cfg = {});
BackendType auto_detect_backend();

// -- Inline CPU kernels (header-only for SIMD dispatch) --
void cpu_matmul_f32(const float* a, const float* b, float* out,
                    int M, int N, int K, float scale = 1.0f);
void cpu_add_f32(const float* a, const float* b, float* out, size_t n);
void cpu_softmax_f32(const float* in, float* out, size_t n_rows, size_t row_size);
void cpu_rms_norm_f32(const float* in, const float* weight, float* out,
                      size_t n_rows, size_t row_size, float eps = 1e-6f);
void cpu_silu_f32(const float* in, float* out, size_t n);
void cpu_rope_f32(float* data, const int32_t* positions,
                  size_t n_tokens, size_t n_heads, size_t head_dim,
                  float freq_base);

} // namespace hesa

#endif // HESA_BACKEND_HPP

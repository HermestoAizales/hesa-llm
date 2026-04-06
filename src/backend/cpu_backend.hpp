#ifndef HESA_CPU_BACKEND_HPP
#define HESA_CPU_BACKEND_HPP

#include "hesa/backend.hpp"

#include <mutex>
#include <thread>
#include <vector>

namespace hesa {

class CPU_Backend final : public Backend {
public:
    explicit CPU_Backend(DeviceConfig cfg);
    ~CPU_Backend() override;

    BackendType type() const override {
#if defined(HESA_ARCH_x86_64)
        return BackendType::CPU_X86;
#else
        return BackendType::CPU_ARM;
#endif
    }
    const char* name() const override { return "CPU"; }
    bool supports(Dtype dt) const override {
        return dt == Dtype::F32 || dt == Dtype::F16 || dt == Dtype::Q4_0 ||
               dt == Dtype::Q8_0 || dt == Dtype::Q4_K || dt == Dtype::Q5_K ||
               dt == Dtype::Q6_K;
    }
    size_t device_memory() const override { return 0; } // not applicable for CPU
    size_t device_memory_used() const override { return 0; }

    Result<Tensor> alloc_tensor(Dtype dtype, std::span<const int64_t> shape) override;
    Result<void> free_tensor(Tensor& tensor) override;

    Result<void> matmul(TensorView a, TensorView b, TensorView out,
                        float scale = 1.0f) override;
    Result<void> add(TensorView a, TensorView b, TensorView out) override;
    Result<void> mul(TensorView a, TensorView b, TensorView out) override;
    Result<void> softmax(TensorView in, TensorView out, int axis = -1) override;
    Result<void> rms_norm(TensorView in, TensorView weight,
                          TensorView out, float eps = 1e-6f) override;
    Result<void> rope(TensorView in_out, std::span<const int32_t> positions,
                      float freq_base, int n_dims) override;
    Result<void> silu(TensorView in, TensorView out) override;

    Result<void> scaled_dot_product_attention(
        TensorView q, TensorView k, TensorView v, TensorView out,
        TensorView mask = {}, float scale = 0.0f) override;

    Result<Tensor> quantize(TensorView src, Dtype target_dtype) override;

    Result<void> synchronize() override { return ok(); }
    Result<void> copy_tensor(const Tensor& src, Tensor& dst) override;

    int n_threads() const { return n_threads_; }

private:
    int n_threads_;
};

} // namespace hesa

#endif // HESA_CPU_BACKEND_HPP

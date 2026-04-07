#include "hesa/tensor.hpp"
#include "hesa/backend.hpp"

#include <cmath>
#include <cstring>

namespace hesa {

TensorView make_const_f32(std::initializer_list<int64_t> shape, const float* data);

// -- Tensor operation implementations --
// These are thin wrappers over backend calls for now.
// When no backend is available, falls back to CPU reference implementations.

Result<void> tensor_matmul(const Tensor& a, const Tensor& b, Tensor& out,
                           float scale = 1.0f) {
    if (a.backend()) {
        return a.backend()->matmul(a.view(), b.view(), out.view(), scale);
    }
    cpu_matmul_f32(static_cast<const float*>(a.data()),
                   static_cast<const float*>(b.data()),
                   static_cast<float*>(out.data()),
                   static_cast<int>(a.shape()[0]),
                   static_cast<int>(b.shape()[1]),
                   static_cast<int>(a.shape()[1]), scale);
    return ok();
}

Result<void> tensor_add(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.backend()) {
        return a.backend()->add(a.view(), b.view(), out.view());
    }
    size_t n = a.nelements();
    cpu_add_f32(static_cast<const float*>(a.data()),
                static_cast<const float*>(b.data()),
                static_cast<float*>(out.data()), n);
    return ok();
}

Result<void> tensor_mul(const Tensor& a, const Tensor& b, Tensor& out) {
    if (a.backend()) {
        return a.backend()->mul(a.view(), b.view(), out.view());
    }
    const float* fa = static_cast<const float*>(a.data());
    const float* fb = static_cast<const float*>(b.data());
    float* fo = static_cast<float*>(out.data());
    for (size_t i = 0; i < static_cast<size_t>(a.nelements()); ++i)
        fo[i] = fa[i] * fb[i];
    return ok();
}

Result<void> tensor_softmax(const Tensor& in, Tensor& out, int axis = -1) {
    if (in.backend()) {
        return in.backend()->softmax(in.view(), out.view(), axis);
    }
    (void)axis;
    /* size_t n = in.shape()[0] * in.shape()[1]; */
    cpu_softmax_f32(static_cast<const float*>(in.data()),
                    static_cast<float*>(out.data()),
                    in.shape()[0], in.shape()[1]);
    return ok();
}

Result<void> tensor_rms_norm(const Tensor& in, const Tensor& weight, Tensor& out,
                             float eps = 1e-6f) {
    if (in.backend()) {
        return in.backend()->rms_norm(in.view(), weight.view(), out.view(), eps);
    }
    size_t row_size = in.shape()[in.ndim() - 1];
    size_t n_rows = in.nelements() / row_size;
    cpu_rms_norm_f32(static_cast<const float*>(in.data()),
                     static_cast<const float*>(weight.data()),
                     static_cast<float*>(out.data()),
                     n_rows, row_size, eps);
    return ok();
}

Result<void> tensor_silu(const Tensor& in, Tensor& out) {
    if (in.backend()) {
        return in.backend()->silu(in.view(), out.view());
    }
    cpu_silu_f32(static_cast<const float*>(in.data()),
                 static_cast<float*>(out.data()),
                 in.nelements());
    return ok();
}

Result<void> tensor_rope(Tensor& in_out, std::span<const int32_t> positions,
                         float freq_base, int n_dims) {
    if (in_out.backend()) {
        return in_out.backend()->rope(in_out.view(), positions, freq_base, n_dims);
    }
    // Reference implementation: 1D tensor [n_tokens * head_dim]
    cpu_rope_f32(static_cast<float*>(in_out.data()),
                 positions.data(),
                 1, in_out.shape().nelements() / n_dims,
                 n_dims, freq_base);
    return ok();
}

} // namespace hesa

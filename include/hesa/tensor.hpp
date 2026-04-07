#ifndef HESA_TENSOR_HPP
#define HESA_TENSOR_HPP

#include "hesa/result.hpp"

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace hesa {

class Backend;

enum class Dtype : uint8_t {
    F32,
    F16,
    BF16,
    I8,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q2_K,
    Q3_K,
    Q4_K,
    Q5_K,
    Q6_K
};

// Shape type: small fixed-size array with dynamic semantics
struct Shape {
    static constexpr size_t MAX_DIMS = 4;
    int64_t data[MAX_DIMS] = {1, 1, 1, 1};
    size_t ndim = 0;

    Shape() = default;
    Shape(std::initializer_list<int64_t> dims) {
        ndim = std::min(dims.size(), MAX_DIMS);
        size_t i = 0;
        for (auto d : dims) {
            if (i < MAX_DIMS) data[i++] = d;
        }
        for (; i < MAX_DIMS; ++i) data[i] = 1;
    }
    explicit Shape(std::span<const int64_t> dims) {
        ndim = std::min(dims.size(), MAX_DIMS);
        for (size_t i = 0; i < ndim; ++i) data[i] = dims[i];
        for (size_t i = ndim; i < MAX_DIMS; ++i) data[i] = 1;
    }

    int64_t operator[](size_t i) const { return i < ndim ? data[i] : 1; }
    int64_t& operator[](size_t i) { return data[i]; }

    int64_t nelements() const {
        int64_t n = 1;
        for (size_t i = 0; i < ndim; ++i) n *= data[i];
        return n;
    }

    std::string to_string() const;
    bool operator==(const Shape& o) const {
        if (ndim != o.ndim) return false;
        for (size_t i = 0; i < ndim; ++i)
            if (data[i] != o.data[i]) return false;
        return true;
    }
};

size_t dtype_size(Dtype dt);
const char* dtype_name(Dtype dt);

/**
 * TensorView — zero-copy view into a Tensor's data.
 */
class TensorView {
public:
    TensorView() = default;
    TensorView(void* data, Dtype dtype, Shape shape,
               std::span<const size_t> byte_strides = {});
    TensorView(const void* data, Dtype dtype, Shape shape,
               std::span<const size_t> byte_strides = {});

    void* data() { return data_; }
    const void* data() const { return data_; }
    Dtype dtype() const { return dtype_; }
    const Shape& shape() const { return shape_; }
    std::span<const size_t> byte_strides() const { return {strides_, shape_.ndim}; }
    size_t ndim() const { return shape_.ndim; }
    size_t nelements() const { return shape_.nelements(); }
    size_t nbytes() const;

private:
    void* data_ = nullptr;
    Dtype dtype_ = Dtype::F32;
    Shape shape_;
    size_t strides_[Shape::MAX_DIMS] = {0, 0, 0, 0};
};

/**
 * Tensor — owned tensor data with device/host memory management.
 * Uses PIMPL for backend-specific storage details.
 */
class Tensor {
public:
    Tensor();
    Tensor(Dtype dtype, std::span<const int64_t> shape, Backend* backend = nullptr);
    Tensor(Dtype dtype, Shape shape, Backend* backend = nullptr);
    ~Tensor();

    Tensor(Tensor&&) noexcept;
    Tensor& operator=(Tensor&&) noexcept;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    void* data();
    const void* data() const;

    Dtype dtype() const { return dtype_; }
    const Shape& shape() const { return shape_; }
    size_t ndim() const { return shape_.ndim; }
    int64_t nelements() const { return shape_.nelements(); }
    size_t nbytes() const { return nelements() * dtype_size(dtype_); }
    size_t byte_stride(int axis) const;

    // Views (zero-copy)
    TensorView view() const;
    TensorView reshape(std::span<const int64_t> new_shape) const;
    TensorView transpose(int ax0, int ax1) const;
    TensorView select(int axis, int64_t index) const;

    // Memory: host vs device
    bool is_on_device() const;
    Backend* backend() const { return backend_; }

    // Set tensor name (for model loading / debugging)
    void set_name(std::string name) { name_ = std::move(name); }
    const std::string& name() const { return name_; }

private:
    struct Impl;
    Dtype dtype_ = Dtype::F32;
    Shape shape_;
    std::unique_ptr<Impl> impl_;
    Backend* backend_ = nullptr;
    std::string name_;
};

// Convenience constructors
inline Tensor make_tensor_f32(std::initializer_list<int64_t> shape, Backend* b = nullptr) {
    return Tensor(Dtype::F32, Shape{shape}, b);
}

} // namespace hesa

#endif // HESA_TENSOR_HPP

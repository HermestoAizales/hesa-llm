#include "hesa/tensor.hpp"
#include "hesa/backend.hpp"

#include <cstring>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace hesa {

// -- Shape --
std::string Shape::to_string() const {
    if (ndim == 0) return "[]";
    std::string s = "[" + std::to_string(data[0]);
    for (size_t i = 1; i < ndim; ++i)
        s += ", " + std::to_string(data[i]);
    s += "]";
    return s;
}

// -- Dtype utilities --
size_t dtype_size(Dtype dt) {
    switch (dt) {
        case Dtype::F32:  return 4;
        case Dtype::F16:  return 2;
        case Dtype::BF16: return 2;
        case Dtype::I8:   return 1;
        case Dtype::Q4_0: return 1;  // 4-bit packed: 0.5 bytes per weight
        case Dtype::Q4_1: return 1;
        case Dtype::Q5_0: return 1;
        case Dtype::Q5_1: return 1;
        case Dtype::Q8_0: return 1;
        case Dtype::Q2_K: return 1;
        case Dtype::Q3_K: return 1;
        case Dtype::Q4_K: return 1;
        case Dtype::Q5_K: return 1;
        case Dtype::Q6_K: return 1;
        default:          return 4;
    }
}

const char* dtype_name(Dtype dt) {
    switch (dt) {
        case Dtype::F32:  return "F32";
        case Dtype::F16:  return "F16";
        case Dtype::BF16: return "BF16";
        case Dtype::I8:   return "I8";
        case Dtype::Q4_0: return "Q4_0";
        case Dtype::Q4_1: return "Q4_1";
        case Dtype::Q5_0: return "Q5_0";
        case Dtype::Q5_1: return "Q5_1";
        case Dtype::Q8_0: return "Q8_0";
        case Dtype::Q2_K: return "Q2_K";
        case Dtype::Q3_K: return "Q3_K";
        case Dtype::Q4_K: return "Q4_K";
        case Dtype::Q5_K: return "Q5_K";
        case Dtype::Q6_K: return "Q6_K";
        default:          return "unknown";
    }
}

// -- Tensor --
struct Tensor::Impl {
    std::vector<uint8_t> host_data; // CPU memory
    size_t byte_strides[Shape::MAX_DIMS] = {0, 0, 0, 0};
    bool owns_data = true;

    // For backend-managed tensors
    void* device_ptr = nullptr;

    void compute_strides(const Shape& s, Dtype dt) {
        byte_strides[s.ndim - 1] = dtype_size(dt);
        for (int i = static_cast<int>(s.ndim) - 2; i >= 0; --i) {
            byte_strides[i] = byte_strides[i + 1] * s.data[i + 1];
        }
    }
};

Tensor::Tensor() : impl_(std::make_unique<Impl>()) {}

Tensor::Tensor(Dtype dtype, std::span<const int64_t> shape, Backend* backend)
    : dtype_(dtype), shape_(shape), impl_(std::make_unique<Impl>()),
      backend_(backend)
{
    size_t n = shape_.nelements();
    impl_->host_data.resize(n * dtype_size(dtype_));
    impl_->compute_strides(shape_, dtype_);
}

Tensor::Tensor(Dtype dtype, Shape shape, Backend* backend)
    : Tensor(dtype, std::span<const int64_t>(shape.data, shape.ndim), backend) {}

Tensor::~Tensor() = default;

Tensor::Tensor(Tensor&& other) noexcept
    : dtype_(other.dtype_), shape_(other.shape_),
      impl_(std::move(other.impl_)), backend_(other.backend_),
      name_(std::move(other.name_)) {
    other.backend_ = nullptr;
}

Tensor& Tensor::operator=(Tensor&& other) noexcept {
    if (this != &other) {
        dtype_ = other.dtype_;
        shape_ = other.shape_;
        backend_ = other.backend_;
        impl_ = std::move(other.impl_);
        name_ = std::move(other.name_);
        other.backend_ = nullptr;
    }
    return *this;
}

void* Tensor::data() { return impl_->host_data.data(); }
const void* Tensor::data() const { return impl_->host_data.data(); }

size_t Tensor::byte_stride(int axis) const {
    if (axis < 0 || axis >= static_cast<int>(shape_.ndim)) return 0;
    return impl_->byte_strides[axis];
}

bool Tensor::is_on_device() const {
    return backend_ != nullptr && impl_->device_ptr != nullptr;
}

// -- TensorView --
TensorView::TensorView(void* data, Dtype dtype, Shape shape,
                       std::span<const size_t> byte_strides)
    : data_(data), dtype_(dtype), shape_(shape)
{
    if (!byte_strides.empty()) {
        for (size_t i = 0; i < shape.ndim && i < byte_strides.size(); ++i)
            strides_[i] = byte_strides[i];
    } else {
        strides_[shape.ndim - 1] = dtype_size(dtype);
        for (int i = static_cast<int>(shape.ndim) - 2; i >= 0; --i)
            strides_[i] = strides_[i + 1] * shape[i + 1];
    }
}

TensorView::TensorView(const void* data, Dtype dtype, Shape shape,
                       std::span<const size_t> byte_strides)
    : TensorView(const_cast<void*>(data), dtype, shape, byte_strides) {}

size_t TensorView::nbytes() const {
    if (ndim() == 0) return 0;
    return nelements() * dtype_size(dtype_);
}

TensorView Tensor::view() const {
    return TensorView(impl_->host_data.data(), dtype_, shape_,
                      std::span<const size_t>(impl_->byte_strides, shape_.ndim));
}

TensorView Tensor::reshape(std::span<const int64_t> new_shape) const {
    Shape ns(new_shape);
    // Validate element count
    if (ns.nelements() != shape_.nelements()) {
        // Return invalid view — validation happens at op level
        return TensorView();
    }
    // Compute new strides
    size_t strides[Shape::MAX_DIMS]{};
    strides[ns.ndim - 1] = dtype_size(dtype_);
    for (int i = static_cast<int>(ns.ndim) - 2; i >= 0; --i)
        strides[i] = strides[i + 1] * ns[i];
    return TensorView(impl_->host_data.data(), dtype_, ns,
                      std::span<const size_t>(strides, ns.ndim));
}

TensorView Tensor::transpose(int ax0, int ax1) const {
    Shape ns = shape_;
    std::swap(ns.data[ax0], ns.data[ax1]);
    size_t strides[Shape::MAX_DIMS];
    std::memcpy(strides, impl_->byte_strides, sizeof(strides));
    std::swap(strides[ax0], strides[ax1]);
    return TensorView(impl_->host_data.data(), dtype_, ns,
                      std::span<const size_t>(strides, shape_.ndim));
}

TensorView Tensor::select(int axis, int64_t index) const {
    if (axis < 0 || axis >= static_cast<int>(shape_.ndim))
        return TensorView();
    Shape ns = shape_;
    // Remove the selected dimension
    for (int i = axis; i < static_cast<int>(ns.ndim) - 1; ++i)
        ns.data[i] = ns.data[i + 1];
    ns.ndim--;
    // Advance data pointer
    uint8_t* ptr = static_cast<uint8_t*>(impl_->host_data.data());
    ptr += impl_->byte_strides[axis] * index;
    return TensorView(ptr, dtype_, ns);
}

} // namespace hesa

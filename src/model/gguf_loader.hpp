#ifndef HESA_GGUF_LOADER_HPP
#define HESA_GGUF_LOADER_HPP

#include "hesa/model.hpp"
#include "hesa/result.hpp"
#include "hesa/tensor.hpp"

namespace hesa {

// GGUF format constants
static constexpr uint32_t GGUF_MAGIC = 0x46554747; // "GGUF"
static constexpr uint32_t GGUF_VERSION = 3;

// GGUF tensor data types (match llama.cpp ggml_type)
enum class GGUFType : int32_t {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    IQ2_XXS = 16,
    IQ2_XS  = 17,
    IQ3_XXS = 18,
    IQ1_S   = 19,
    IQ4_NL  = 20,
    IQ3_S   = 21,
    IQ2_S   = 22,
    IQ4_XS  = 23,
    I8     = 24,
    I16    = 25,
    I32    = 26,
    I64    = 27,
    F64    = 28,
    IQ1_M  = 29,
    BF16   = 30,
};

size_t gguf_type_size(GGUFType t, size_t nelements);
Dtype gguf_to_hesa_dtype(GGUFType t);

// -- Low level reader --

class GGUFReader {
public:
    explicit GGUFReader(const std::string& path);
    ~GGUFReader();

    Result<void> parse();

    // Metadata
    const std::unordered_map<std::string, ModelMetadata>& metadata() const { return meta_; }
    uint64_t tensor_count() const { return n_tensors_; }
    uint64_t kv_count() const { return n_kv_; }

    // Tensor info
    struct TensorInfo {
        std::string name;
        GGUFType dtype;
        std::vector<int64_t> dims;
        uint64_t offset; // Offset within the tensor data region
    };

    const std::vector<TensorInfo>& tensors() const { return tensor_infos_; }

    // Memory mapped file
    const uint8_t* data_ptr() const { return data_; }
    uint64_t file_size() const { return file_size_; }

    // Offset where tensor data begins
    uint64_t data_offset() const { return data_offset_; }

private:
    Result<void> read_header();
    Result<void> read_metadata();
    Result<void> read_tensor_infos();

    std::string path_;
    int fd_ = -1;
    const uint8_t* data_ = nullptr;
    uint64_t file_size_ = 0;
    uint64_t data_offset_ = 0;

    uint64_t n_tensors_ = 0;
    uint64_t n_kv_ = 0;

    ModelMetadata meta_;
    std::vector<TensorInfo> tensor_infos_;
};

} // namespace hesa

#endif // HESA_GGUF_LOADER_HPP

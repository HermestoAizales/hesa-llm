#ifndef HESA_GGUF_LOADER_HPP
#define HESA_GGUF_LOADER_HPP

#include "hesa/model.hpp"
#include "hesa/result.hpp"
#include "hesa/tensor.hpp"

#include <cinttypes>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace hesa {

static constexpr uint32_t GGUF_MAGIC = 0x46554747;
static constexpr uint32_t GGUF_VERSION = 3;

enum class GGUFType : int32_t {
    F32  = 0, F16  = 1, Q4_0 = 2, Q4_1 = 3,
    Q5_0 = 6, Q5_1 = 7, Q8_0 = 8, Q8_1 = 9,
    Q2_K = 10, Q3_K = 11, Q4_K = 12, Q5_K = 13, Q6_K = 14,
    I8   = 24, I16  = 25, I32  = 26, I64  = 27, F64  = 28,
    BF16 = 30,
};

size_t gguf_type_size(GGUFType t, size_t nelements);
Dtype gguf_to_hesa_dtype(GGUFType t);

class GGUFReader {
public:
    explicit GGUFReader(const std::string& path);
    ~GGUFReader();

    Result<void> parse();

    const std::unordered_map<std::string, std::variant<std::string, int64_t, double, bool>>& metadata() const { return meta_custom_; }
    uint64_t tensor_count() const { return n_tensors_; }
    uint64_t kv_count() const { return n_kv_; }

    struct TensorInfo {
        std::string name;
        GGUFType dtype;
        std::vector<int64_t> dims;
        uint64_t offset;
    };

    const std::vector<TensorInfo>& tensors() const { return tensor_infos_; }

    const uint8_t* data_ptr() const { return data_; }
    uint64_t file_size() const { return file_size_; }
    uint64_t data_offset() const { return data_offset_; }

    // Array-type metadata accessors (for tokenizer data)
    const std::unordered_map<std::string, std::vector<std::string>>& string_arrays() const { return string_arrays_; }
    const std::unordered_map<std::string, std::vector<float>>& float_arrays() const { return float_arrays_; }
    const std::unordered_map<std::string, std::vector<int32_t>>& int_arrays() const { return int_arrays_; }

    // Release ownership of the mmap (used when transferring to Model)
    void release();

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

    // Raw metadata (before extraction into ModelMetadata)
    std::unordered_map<std::string, std::variant<std::string, int64_t, double, bool>> meta_custom_;
    std::unordered_map<std::string, std::vector<std::string>> string_arrays_;
    std::unordered_map<std::string, std::vector<float>> float_arrays_;
    std::unordered_map<std::string, std::vector<int32_t>> int_arrays_;
    std::vector<TensorInfo> tensor_infos_;
};

} // namespace hesa

#endif // HESA_GGUF_LOADER_HPP

#include "model/gguf_loader.hpp"
#include "hesa/backend.hpp"
#include "hesa/model.hpp"
#include "hesa/result.hpp"
#include "hesa/tensor.hpp"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstring>
#include <stdexcept>
#include <system_error>

namespace hesa {

// -- GGUF type size (including block packing for quantized types) --
size_t gguf_type_size(GGUFType t, size_t nelements) {
    // For quantized types, compute packed size based on block structure
    switch (t) {
        case GGUFType::F32:   return nelements * 4;
        case GGUFType::F16:   return nelements * 2;
        case GGUFType::BF16:  return nelements * 2;
        case GGUFType::I8:    return nelements * 1;
        case GGUFType::Q4_0:  return (nelements / 32) * 18;
        case GGUFType::Q4_1:  return (nelements / 32) * 20;
        case GGUFType::Q5_0:  return (nelements / 32) * 22;
        case GGUFType::Q5_1:  return (nelements / 32) * 24;
        case GGUFType::Q8_0:  return (nelements / 32) * 34;
        case GGUFType::Q2_K:  return (nelements / 256) * 84;
        case GGUFType::Q3_K:  return (nelements / 256) * 112;
        case GGUFType::Q4_K:  return (nelements / 256) * 144;
        case GGUFType::Q5_K:  return (nelements / 256) * 176;
        case GGUFType::Q6_K:  return (nelements / 256) * 210;
        default:              return nelements * 4; // conservative fallback
    }
}

Dtype gguf_to_hesa_dtype(GGUFType t) {
    switch (t) {
        case GGUFType::F32:   return Dtype::F32;
        case GGUFType::F16:   return Dtype::F16;
        case GGUFType::BF16:  return Dtype::BF16;
        case GGUFType::I8:    return Dtype::I8;
        case GGUFType::Q4_0:  return Dtype::Q4_0;
        case GGUFType::Q4_1:  return Dtype::Q4_1;
        case GGUFType::Q5_0:  return Dtype::Q5_0;
        case GGUFType::Q5_1:  return Dtype::Q5_1;
        case GGUFType::Q8_0:  return Dtype::Q8_0;
        case GGUFType::Q2_K:  return Dtype::Q2_K;
        case GGUFType::Q3_K:  return Dtype::Q3_K;
        case GGUFType::Q4_K:  return Dtype::Q4_K;
        case GGUFType::Q5_K:  return Dtype::Q5_K;
        case GGUFType::Q6_K:  return Dtype::Q6_K;
        default:              return Dtype::F32;
    }
}

// -- Helpers for reading raw values --

template<typename T>
static T read_val(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

static std::string read_string(const uint8_t* p, uint64_t& offset_out) {
    uint64_t len = read_val<uint64_t>(p);
    offset_out += 8;
    std::string s(reinterpret_cast<const char*>(p + 8), len);
    offset_out += len;
    return s;
}

// -- GGUFReader implementation --

struct Model::MappedFile {
    int fd = -1;
    uint8_t* data = nullptr;
    uint64_t size = 0;

    ~MappedFile() {
        if (data && data != MAP_FAILED) munmap(data, size);
        if (fd >= 0) close(fd);
    }
};

GGUFReader::GGUFReader(const std::string& path) : path_(path) {}

GGUFReader::~GGUFReader() = default;

Result<void> GGUFReader::parse() {
    HESA_CHECK(read_header());
    HESA_CHECK(read_metadata());
    HESA_CHECK(read_tensor_infos());
    return ok();
}

Result<void> GGUFReader::read_header() {
    // Open and mmap file
    fd_ = open(path_.c_str(), O_RDONLY);
    if (fd_ < 0) return error(Error::FILE_NOT_FOUND);

    struct stat st;
    if (fstat(fd_, &st) < 0) {
        close(fd_);
        fd_ = -1;
        return error(Error::FILE_NOT_FOUND);
    }
    file_size_ = static_cast<uint64_t>(st.st_size);

    data_ = static_cast<const uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (!data_ || data_ == MAP_FAILED) {
        close(fd_);
        fd_ = -1;
        data_ = nullptr;
        return error(Error::FILE_NOT_FOUND);
    }

    // Read magic
    uint32_t magic = read_val<uint32_t>(data_);
    if (magic != GGUF_MAGIC) return error(Error::INVALID_FORMAT);

    // Read version
    uint32_t version = read_val<uint32_t>(data_ + 4);
    if (version != GGUF_VERSION) return error(Error::INVALID_FORMAT);

    uint64_t off = 8;
    n_tensors_ = read_val<uint64_t>(data_ + off); off += 8;
    n_kv_      = read_val<uint64_t>(data_ + off); off += 8;
    data_offset_ = off;

    return ok();
}

Result<void> GGUFReader::read_metadata() {
    uint64_t off = data_offset_;

    for (uint64_t i = 0; i < n_kv_; ++i) {
        std::string key = read_string(data_ + off, off);
        int32_t type_byte = read_val<int32_t>(data_ + off); off += 4;

        // GGUF type enum: 0=UINT8, 1=INT8, 2=UINT16, 3=INT16,
        // 4=UINT32, 5=INT32, 6=FLOAT32, 7=BOOL, 8=STRING,
        // 9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64
        switch (type_byte) {
            case 4: case 5: { // UINT32/INT32
                uint32_t v = read_val<uint32_t>(data_ + off); off += 4;
                meta_.custom[key] = static_cast<int64_t>(v);
                break;
            }
            case 6: { // FLOAT32
                float v = read_val<float>(data_ + off); off += 4;
                meta_.custom[key] = static_cast<double>(v);
                break;
            }
            case 7: { // BOOL
                uint8_t v = data_[off]; off += 1;
                meta_.custom[key] = (v != 0);
                break;
            }
            case 8: { // STRING
                std::string v = read_string(data_ + off, off);
                meta_.custom[key] = v;
                break;
            }
            case 10: { // UINT64
                uint64_t v = read_val<uint64_t>(data_ + off); off += 8;
                meta_.custom[key] = static_cast<int64_t>(v);
                break;
            }
            case 11: { // INT64
                int64_t v = read_val<int64_t>(data_ + off); off += 8;
                meta_.custom[key] = v;
                break;
            }
            case 12: { // FLOAT64
                double v = read_val<double>(data_ + off); off += 8;
                meta_.custom[key] = v;
                break;
            }
            default:
                // Skip unknown types conservatively
                off += 4; // skip at least some
                break;
        }
    }

    // Extract well-known metadata keys
    auto str_val = [&](const std::string& key) -> std::string {
        auto it = meta_.custom.find(key);
        if (it != meta_.custom.end() && std::holds_alternative<std::string>(it->second))
            return std::get<std::string>(it->second);
        return {};
    };
    auto int_val = [&](const std::string& key, int def = 0) -> int {
        auto it = meta_.custom.find(key);
        if (it != meta_.custom.end() && std::holds_alternative<int64_t>(it->second))
            return static_cast<int>(std::get<int64_t>(it->second));
        return def;
    };
    auto float_val = [&](const std::string& key, float def = 0.0f) -> float {
        auto it = meta_.custom.find(key);
        if (it != meta_.custom.end() && (std::holds_alternative<double>(it->second) || std::holds_alternative<int64_t>(it->second))) {
            if (std::holds_alternative<double>(it->second)) return static_cast<float>(std::get<double>(it->second));
            return static_cast<float>(std::get<int64_t>(it->second));
        }
        return def;
    };
    auto bool_val = [&](const std::string& key, bool def = false) -> bool {
        auto it = meta_.custom.find(key);
        if (it != meta_.custom.end() && std::holds_alternative<bool>(it->second))
            return std::get<bool>(it->second);
        return def;
    };

    meta_.architecture = str_val("general.architecture");
    meta_.vocab_size = int_val("general.vocab_size");
    meta_.block_count = int_val("general.block_count");
    meta_.context_length = int_val("general.context_length");

    std::string arch = meta_.architecture;
    if (!arch.empty()) {
        meta_.embedding_length = int_val(arch + ".embedding_length");
        meta_.feed_forward_length = int_val(arch + ".feed_forward_length");
        meta_.attention_head_count = int_val(arch + ".attention.head_count");
        meta_.attention_head_count_kv = int_val(arch + ".attention.head_count_kv", meta_.attention_head_count);
        meta_.attention_layer_norm_rms_epsilon = int_val(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
        meta_.rope_dimension_count = int_val(arch + ".rope.dimension_count");
        meta_.rope_freq_base = float_val(arch + ".rope.freq_base", 10000.0f);
    }

    // HESA-specific keys
    meta_.use_neural_memory = bool_val("hesa.use_neural_memory");
    meta_.use_ttt = bool_val("hesa.use_ttt");
    meta_.use_hyper_connections = bool_val("hesa.use_hyper_connections");
    meta_.engram_enabled = bool_val("hesa.engram_enabled");

    data_offset_ = off;
    return ok();
}

Result<void> GGUFReader::read_tensor_infos() {
    uint64_t off = data_offset_;

    // Align to 32 bytes
    off = (off + 31) & ~static_cast<uint64_t>(31);

    for (uint64_t i = 0; i < n_tensors_; ++i) {
        TensorInfo info;
        info.name = read_string(data_ + off, off);
        info.dims.resize(read_val<uint32_t>(data_ + off)); off += 4;
        info.dtype = static_cast<GGUFType>(read_val<int32_t>(data_ + off)); off += 4;

        size_t nelements = 1;
        for (size_t d = 0; d < info.dims.size(); ++d) {
            info.dims[d] = read_val<int64_t>(data_ + off); off += 8;
            nelements *= info.dims[d];
        }

        info.offset = read_val<uint64_t>(data_ + off); off += 8;

        tensor_infos_.push_back(std::move(info));
        (void)nelements;
    }

    // Align tensor data offset
    data_offset_ = (off + 31) & ~static_cast<uint64_t>(31);
    return ok();
}

// -- Model::load --

Result<std::unique_ptr<Model>> Model::load(const std::string& path, Backend* backend) {
    GGUFReader reader(path);
    HESA_CHECK(reader.parse());

    auto model = std::make_unique<Model>();
    model->metadata_ = reader.meta_;
    model->path_ = path;
    model->file_size_ = reader.file_size();
    model->mapped_file_ = std::make_unique<Model::MappedFile>();
    model->mapped_file_->fd = -1; // GGUFReader already opened; we open fresh
    model->mapped_file_->size = reader.file_size();
    model->mapped_file_->data = const_cast<uint8_t*>(reader.data_ptr());

    const uint8_t* data_start = reader.data_ptr() + reader.data_offset();

    for (const auto& ti : reader.tensors()) {
        // Compute element count from dims (GGUF stores in reverse order for BLAS)
        std::vector<int64_t> shape_reversed(ti.dims.rbegin(), ti.dims.rend());

        uint8_t* weight_data = const_cast<uint8_t*>(data_start + ti.offset);
        size_t nelem = 1;
        for (auto d : ti.dims) nelem *= d;

        Dtype dtype = gguf_to_hesa_dtype(ti.dtype);

        Tensor t;
        if (dtype == Dtype::F32 || dtype == Dtype::F16 || dtype == Dtype::BF16) {
            // For F32: directly map (or copy to host)
            t = Tensor(dtype, shape_reversed, backend);
            std::memcpy(t.data(), weight_data, t.nbytes());
        } else {
            // For quantized: store raw packed data
            size_t packed_size = gguf_type_size(ti.dtype, nelem);
            t = Tensor(dtype, shape_reversed, backend);
            // For now, just copy packed bytes
            std::memcpy(t.data(), weight_data, std::min(packed_size, t.nbytes()));
        }
        t.set_name(ti.name);
        model->tensors_[ti.name] = std::move(t);
        model->tensor_names_.push_back(ti.name);
    }

    return model;
}

Model::~Model() = default;

Result<const Tensor*> Model::get_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return error(Error::TENSOR_NOT_FOUND);
    return &it->second;
}

} // namespace hesa

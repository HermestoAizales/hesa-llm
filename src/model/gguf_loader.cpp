#include "model/gguf_loader.hpp"
#include "hesa/backend.hpp"
#include "hesa/model.hpp"
#include "hesa/result.hpp"
#include "hesa/tensor.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sys/mman.h>

namespace hesa {

size_t gguf_type_size(GGUFType t, size_t nelements) {
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
        default:              return nelements * 4;
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

template<typename T>
static T read_val(const uint8_t* p) {
    T v;
    std::memcpy(&v, p, sizeof(T));
    return v;
}

static std::string read_string(const uint8_t* p, uint64_t& offset_out) {
    uint64_t len = read_val<uint64_t>(p);
    offset_out += 8;
    std::string s(reinterpret_cast<const char*>(p + 8), static_cast<size_t>(len));
    offset_out += len;
    return s;
}

struct Model::MappedFile {
    int fd = -1;
    uint8_t* data = nullptr;
    uint64_t size = 0;
    ~MappedFile() {
        if (data && data != MAP_FAILED) munmap(data, size);
        if (fd >= 0) ::close(fd);
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
    fd_ = open(path_.c_str(), O_RDONLY);
    if (fd_ < 0) return make_error<void>(Error::FILE_NOT_FOUND);

    struct stat st;
    if (fstat(fd_, &st) < 0) {
        ::close(fd_);
        fd_ = -1;
        return make_error<void>(Error::FILE_NOT_FOUND);
    }
    file_size_ = static_cast<uint64_t>(st.st_size);

    data_ = static_cast<const uint8_t*>(mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0));
    if (!data_ || data_ == MAP_FAILED) {
        ::close(fd_);
        fd_ = -1;
        data_ = nullptr;
        return make_error<void>(Error::FILE_NOT_FOUND);
    }

    uint32_t magic = read_val<uint32_t>(data_);
    if (magic != GGUF_MAGIC) {
        munmap(const_cast<uint8_t*>(data_), file_size_);
        data_ = nullptr;
        return make_error<void>(Error::INVALID_FORMAT);
    }

    uint32_t version = read_val<uint32_t>(data_ + 4);
    if (version != GGUF_VERSION) {
        munmap(const_cast<uint8_t*>(data_), file_size_);
        data_ = nullptr;
        return make_error<void>(Error::INVALID_FORMAT);
    }

    uint64_t off = 8;
    n_tensors_ = read_val<uint64_t>(data_ + off); off += 8;
    n_kv_      = read_val<uint64_t>(data_ + off); off += 8;
    data_offset_ = off;

    return ok();
}

Result<void> GGUFReader::read_metadata() {
    uint64_t off = data_offset_;

    for (uint64_t i = 0; i < n_kv_; ++i) {
        if (off >= file_size_) break;

        std::string key = read_string(data_ + off, off);
        if (off >= file_size_) break;

        int32_t type_byte = read_val<int32_t>(data_ + off); off += 4;

        switch (type_byte) {
            case 4: { // UINT32
                uint32_t v = read_val<uint32_t>(data_ + off); off += 4;
                meta_custom_[key] = static_cast<int64_t>(v);
                break;
            }
            case 5: { // INT32
                int32_t v = read_val<int32_t>(data_ + off); off += 4;
                meta_custom_[key] = static_cast<int64_t>(v);
                break;
            }
            case 6: { // FLOAT32
                float v = read_val<float>(data_ + off); off += 4;
                meta_custom_[key] = static_cast<double>(v);
                break;
            }
            case 7: { // BOOL
                uint8_t v = data_[off]; off += 1;
                meta_custom_[key] = (v != 0);
                break;
            }
            case 8: { // STRING
                std::string v = read_string(data_ + off, off);
                meta_custom_[key] = std::move(v);
                break;
            }
            case 10: { // UINT64
                uint64_t v = read_val<uint64_t>(data_ + off); off += 8;
                meta_custom_[key] = static_cast<int64_t>(v);
                break;
            }
            case 11: { // INT64
                int64_t v = read_val<int64_t>(data_ + off); off += 8;
                meta_custom_[key] = v;
                break;
            }
            case 12: { // FLOAT64
                double v = read_val<double>(data_ + off); off += 8;
                meta_custom_[key] = v;
                break;
            }
            case 9: { // ARRAY
                int32_t arr_type = read_val<int32_t>(data_ + off); off += 4;
                uint64_t arr_len = read_val<uint64_t>(data_ + off); off += 8;
                if (arr_type == 8) { // STRING array
                    auto& arr = string_arrays_[key];
                    arr.reserve(static_cast<size_t>(arr_len));
                    for (uint64_t j = 0; j < arr_len; ++j)
                        arr.push_back(read_string(data_ + off, off));
                } else if (arr_type == 6) { // FLOAT32 array
                    auto& arr = float_arrays_[key];
                    arr.reserve(static_cast<size_t>(arr_len));
                    for (uint64_t j = 0; j < arr_len; ++j) {
                        arr.push_back(read_val<float>(data_ + off));
                        off += 4;
                    }
                } else if (arr_type == 12) { // FLOAT64 array
                    auto& arr = float_arrays_[key];
                    arr.reserve(static_cast<size_t>(arr_len));
                    for (uint64_t j = 0; j < arr_len; ++j) {
                        arr.push_back(static_cast<float>(read_val<double>(data_ + off)));
                        off += 8;
                    }
                } else if (arr_type == 5) { // INT32 array
                    auto& arr = int_arrays_[key];
                    arr.reserve(static_cast<size_t>(arr_len));
                    for (uint64_t j = 0; j < arr_len; ++j) {
                        arr.push_back(read_val<int32_t>(data_ + off));
                        off += 4;
                    }
                } else if (arr_type == 10) { // UINT64 array
                    auto& arr = int_arrays_[key];
                    arr.reserve(static_cast<size_t>(arr_len));
                    for (uint64_t j = 0; j < arr_len; ++j) {
                        uint64_t v = read_val<uint64_t>(data_ + off); off += 8;
                        arr.push_back(v > static_cast<uint64_t>(INT32_MAX) ? INT32_MAX : static_cast<int32_t>(v));
                    }
                } else if (arr_type == 11) { // INT64 array
                    auto& arr = int_arrays_[key];
                    arr.reserve(static_cast<size_t>(arr_len));
                    for (uint64_t j = 0; j < arr_len; ++j) {
                        int64_t v = read_val<int64_t>(data_ + off); off += 8;
                        if (v > INT32_MAX) v = INT32_MAX;
                        else if (v < INT32_MIN) v = INT32_MIN;
                        arr.push_back(static_cast<int32_t>(v));
                    }
                } else {
                    // Fallback: skip remaining bytes
                    if (arr_type <= 7) off += arr_len * 4;
                    else off += arr_len * 8;
                }
                break;
            }
            default:
                break;
        }
    }

    data_offset_ = off;
    return ok();
}

    Result<void> GGUFReader::read_tensor_infos() {
        uint64_t off = data_offset_;

        for (uint64_t i = 0; i < n_tensors_; ++i) {
            if (off >= file_size_) break;

            TensorInfo info;
            info.name = read_string(data_ + off, off);
            if (off >= file_size_) break;

            uint32_t n_dims = read_val<uint32_t>(data_ + off); off += 4;
            info.dims.resize(n_dims);
            for (uint32_t d = 0; d < n_dims && off + 8 <= file_size_; ++d) {
                info.dims[d] = read_val<int64_t>(data_ + off); off += 8;
            }
            // Reverse to match engine expectations.
            std::reverse(info.dims.begin(), info.dims.end());

            if (off + 8 > file_size_) break;
            int32_t dtype_raw = read_val<int32_t>(data_ + off); off += 4;
            info.dtype = static_cast<GGUFType>(dtype_raw);
            info.offset = read_val<uint64_t>(data_ + off); off += 8;

            tensor_infos_.push_back(std::move(info));
        }

        data_offset_ = (off + 31) & ~static_cast<uint64_t>(31);
        return ok();
    }

    // Count transformer layers from tensor names (e.g. "model.layers.0.attn" or "blk.0.")
    int GGUFReader::count_layers() const {
        int max_idx = -1;
        for (const auto& t : tensor_infos_) {
            const std::string& name = t.name;
            size_t pos = name.find("layers.");
            if (pos == std::string::npos) pos = name.find("blk.");
            if (pos == std::string::npos) continue;
            size_t dot = name.find('.', pos + (pos == name.find("layers.") ? 7 : 4));
            if (dot == std::string::npos) continue;
            try {
                int idx = std::stoi(name.substr(pos + (pos == name.find("layers.") ? 7 : 4),
                                                dot - (pos + (pos == name.find("layers.") ? 7 : 4))));
                if (idx > max_idx) max_idx = idx;
            } catch (...) {}
        }
        return max_idx >= 0 ? max_idx + 1 : 0;
    }

void GGUFReader::release() {
    if (data_ && data_ != MAP_FAILED) munmap(const_cast<uint8_t*>(data_), file_size_);
    if (fd_ >= 0) ::close(fd_);
    fd_ = -1;
    data_ = nullptr;
    path_.clear();
    file_size_ = 0;
}

// Transfer the mmap from the reader to the caller.
// After this, the reader no longer owns the mmap.
struct GGUFReader::MmapHandle {
    int fd = -1;
    const uint8_t* data = nullptr;
    uint64_t size = 0;
};

std::unique_ptr<GGUFReader::MmapHandle> GGUFReader::detach_mmap() {
    auto handle = std::make_unique<MmapHandle>();
    handle->fd = fd_;
    handle->data = data_;
    handle->size = file_size_;
    fd_ = -1;
    data_ = nullptr;
    file_size_ = 0;
    return handle;
}

// ─── Model::load ───
Result<std::unique_ptr<Model>> Model::load(const std::string& path, Backend* backend) {
    GGUFReader reader(path);
    HESA_CHECK(reader.parse());

    auto model = std::make_unique<Model>();
    model->path_ = path;
    model->file_size_ = reader.file_size();

    // Auto-detect layer count from tensor names if not present in metadata.
    int detected_layers = reader.count_layers();
    if (detected_layers > 0) {
        model->metadata_.block_count = detected_layers;
    }
    // If still 0, we may get block_count from metadata below (general.block_count / arch.block_count).

    // Extract metadata into ModelMetadata
    auto meta = model->metadata_;
    auto str_val = [&](const std::string& key) -> std::string {
        auto it = reader.metadata().find(key);
        if (it != reader.metadata().end() && std::holds_alternative<std::string>(it->second))
            return std::get<std::string>(it->second);
        return {};
    };
    auto int_val = [&](const std::string& key, int def = 0) -> int {
        auto it = reader.metadata().find(key);
        if (it != reader.metadata().end() && std::holds_alternative<int64_t>(it->second))
            return static_cast<int>(std::get<int64_t>(it->second));
        return def;
    };
    auto float_val = [&](const std::string& key, float def = 0.0f) -> float {
        auto it = reader.metadata().find(key);
        if (it != reader.metadata().end()) {
            if (std::holds_alternative<double>(it->second))
                return static_cast<float>(std::get<double>(it->second));
            if (std::holds_alternative<int64_t>(it->second))
                return static_cast<float>(std::get<int64_t>(it->second));
        }
        return def;
    };
    auto bool_val = [&](const std::string& key, bool def = false) -> bool {
        auto it = reader.metadata().find(key);
        if (it != reader.metadata().end() && std::holds_alternative<bool>(it->second))
            return std::get<bool>(it->second);
        return def;
    };

    meta.architecture = str_val("general.architecture");
    meta.vocab_size = int_val("general.vocab_size");
    meta.block_count = int_val("general.block_count");
    meta.context_length = int_val("general.context_length", 2048);

    std::string arch = meta.architecture;
    if (!arch.empty()) {
        meta.embedding_length = int_val(arch + ".embedding_length");
        meta.feed_forward_length = int_val(arch + ".feed_forward_length");
        meta.attention_head_count = int_val(arch + ".attention.head_count");
        meta.attention_head_count_kv = int_val(arch + ".attention.head_count_kv", meta.attention_head_count);
        float rms_eps = float_val(arch + ".attention.layer_norm_rms_epsilon", 1e-5f);
        meta.attention_layer_norm_rms_epsilon = static_cast<uint32_t>(rms_eps * 1e7f);
        meta.rope_dimension_count = int_val(arch + ".rope.dimension_count");
        meta.rope_freq_base = float_val(arch + ".rope.freq_base", 10000.0f);
        // Architecture-specific fallbacks if general.* were not found
        if (meta.vocab_size == 0) {
            meta.vocab_size = int_val(arch + ".vocab_size");
        }
        if (meta.block_count == 0) {
            meta.block_count = int_val(arch + ".block_count");
        }
        if (meta.block_count == 0) {
            meta.block_count = int_val(arch + ".num_hidden_layers");
        }
    }

    meta.use_neural_memory = bool_val("hesa.use_neural_memory");
    meta.use_ttt = bool_val("hesa.use_ttt");
    meta.use_hyper_connections = bool_val("hesa.use_hyper_connections");
    meta.engram_enabled = bool_val("hesa.engram_enabled");

    // ─── Extract tokenizer arrays from GGUF (now stored by the loader!) ────
    auto sa_it = reader.string_arrays().find("tokenizer.ggml.tokens");
    if (sa_it != reader.string_arrays().end()) {
        meta.vocab = sa_it->second;
    }
    auto sa_merges = reader.string_arrays().find("tokenizer.ggml.merges");
    if (sa_merges != reader.string_arrays().end()) {
        meta.merges = sa_merges->second;
    }
    auto fa_scores = reader.float_arrays().find("tokenizer.ggml.scores");
    if (fa_scores != reader.float_arrays().end()) {
        meta.vocab_scores = fa_scores->second;
    }
    auto ia_types = reader.int_arrays().find("tokenizer.ggml.token_type");
    if (ia_types != reader.int_arrays().end()) {
        meta.token_types = ia_types->second;
    }

    // If vocab_size not explicitly provided, derive from tokenizer tokens
    if (meta.vocab_size == 0 && !meta.vocab.empty()) {
        meta.vocab_size = static_cast<int>(meta.vocab.size());
    }

    model->metadata_ = meta;

    // Detach the reader's mmap and transfer ownership to the Model.
    // This avoids a second mmap of the same file.
    auto mmap_handle = reader.detach_mmap();
    if (mmap_handle->data && mmap_handle->data != MAP_FAILED) {
        model->mapped_file_ = std::make_unique<Model::MappedFile>();
        model->mapped_file_->fd = mmap_handle->fd;
        model->mapped_file_->data = const_cast<uint8_t*>(mmap_handle->data);
        model->mapped_file_->size = mmap_handle->size;
    }

    const uint8_t* base = const_cast<uint8_t*>(mmap_handle->data);

    for (const auto& ti : reader.tensors()) {
        // Keep shape as stored in GGUF (e.g., [vocab, hidden] for embeddings)
        std::vector<int64_t> shape = ti.dims;

        uint64_t nelem = 1;
        for (auto d : shape) nelem *= d;

        Dtype dtype = gguf_to_hesa_dtype(ti.dtype);
        // FIX: ti.offset is an absolute file offset. Previously we added reader.data_offset()
        // again via data_start, causing double offset -> SIGSEGV on first tensor access.
        const uint8_t* weight_data = base + ti.offset;

        Tensor t;
        if (dtype == Dtype::F32) {
            // Zero-copy: tensor data points directly into the mmap'd region
            t = Tensor::make_from_external(const_cast<uint8_t*>(weight_data), dtype, shape);
        } else if (dtype == Dtype::F16 || dtype == Dtype::BF16) {
            // Need to convert F16/BF16 -> F32, so allocate new memory
            t = Tensor(Dtype::F32, shape, backend);
            const uint16_t* src = reinterpret_cast<const uint16_t*>(weight_data);
            float* dst = static_cast<float*>(t.data());
            for (uint64_t i = 0; i < nelem; ++i) {
                uint32_t u = src[i];
                uint32_t sign = (u >> 15) & 0x1;
                int32_t exp   = ((u >> 10) & 0x1f);
                uint32_t frac = u & 0x3ff;
                uint32_t f32_bits;
                if (exp == 0) {
                    f32_bits = (sign << 31) | (frac << 13);
                } else if (exp == 0x1f) {
                    f32_bits = (sign << 31) | (0xff << 23) | (frac << 13);
                } else {
                    f32_bits = (sign << 31) | ((exp + 127 - 15) << 23) | (frac << 13);
                }
                std::memcpy(&dst[i], &f32_bits, sizeof(float));
            }
        } else {
            // Zero-copy for quantized types: reference mmap directly
            (void)nelem; // shape already encodes element count
            t = Tensor::make_from_external(const_cast<uint8_t*>(weight_data), dtype, shape);
        }
        t.set_name(ti.name);
        model->tensors_[ti.name] = std::move(t);
        model->tensor_names_.push_back(ti.name);
    }

    // block_count already set via metadata or detected via reader.count_layers() above.

    return model;
}

Model::~Model() = default;

Result<const Tensor*> Model::get_tensor(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end()) return make_error<Tensor*>(Error::TENSOR_NOT_FOUND);
    return &it->second;
}

} // namespace hesa

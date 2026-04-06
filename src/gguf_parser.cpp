#include "hesa/gguf.hpp"
#include "hesa/result.hpp"

#include <cstring>
#include <cstdio>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace hesa {

// ─── MmapRegion ───────────────────────────────────────────────────
MmapRegion::~MmapRegion() {
    if (data_ && data_ != MAP_FAILED) munmap(data_, size_);
    if (fd_ >= 0) close(fd_);
}

MmapRegion::MmapRegion(MmapRegion&& o) noexcept
    : path_(std::move(o.path_)), fd_(o.fd_), data_(o.data_), size_(o.size_) {
    o.fd_ = -1; o.data_ = nullptr; o.size_ = 0;
}

MmapRegion& MmapRegion::operator=(MmapRegion&& o) noexcept {
    if (this != &o) {
        if (data_ && data_ != MAP_FAILED) munmap(data_, size_);
        if (fd_ >= 0) close(fd_);
        path_ = std::move(o.path_); fd_ = o.fd_; data_ = o.data_; size_ = o.size_;
        o.fd_ = -1; o.data_ = nullptr; o.size_ = 0;
    }
    return *this;
}

Result<MmapRegion> MmapRegion::open(const std::string& path) {
    MmapRegion r;
    r.path_ = path;
    r.fd_ = ::open(path.c_str(), O_RDONLY);
    if (r.fd_ < 0) return std::unexpected{make_error_code(Error::FILE_NOT_FOUND)};

    struct stat st{};
    if (::fstat(r.fd_, &st) < 0) return std::unexpected{make_error_code(Error::FILE_NOT_FOUND)};
    r.size_ = static_cast<size_t>(st.st_size);
    if (r.size_ == 0) return std::unexpected{make_error_code(Error::INVALID_FORMAT)};

    r.data_ = static_cast<uint8_t*>(
        mmap(nullptr, r.size_, PROT_READ, MAP_PRIVATE | MAP_NORESERVE, r.fd_, 0));
    if (!r.data_ || r.data_ == MAP_FAILED) {
        r.data_ = nullptr;
        ::close(r.fd_); r.fd_ = -1;
        return std::unexpected{make_error_code(Error::FILE_NOT_FOUND)};
    }
    return r;
}

// ─── Read helpers ─────────────────────────────────────────────────
template<typename T>
static inline T read_val(const uint8_t* p) {
    T v; std::memcpy(&v, p, sizeof(T)); return v;
}

static std::string read_gguf_str(const uint8_t* p, uint64_t& off) {
    uint64_t len = read_val<uint64_t>(p);
    off += 8;
    std::string s(reinterpret_cast<const char*>(p + 8), static_cast<size_t>(len));
    off += len;
    return s;
}

// ─── GGUFHeader helpers ──────────────────────────────────────────
const GGUFKV* GGUFHeader::find_kv(const std::string& key) const {
    for (auto& kv : kv_pairs) if (kv.key == key) return &kv;
    return nullptr;
}
int64_t GGUFHeader::get_i64(const std::string& key, int64_t def) const {
    if (auto* kv = find_kv(key)) {
        if (kv->value_type == GGUFValueType::UINT32)  return static_cast<int64_t>(kv->val_u32);
        if (kv->value_type == GGUFValueType::INT32)   return kv->val_i32;
        if (kv->value_type == GGUFValueType::UINT64)  return static_cast<int64_t>(kv->val_u64);
        if (kv->value_type == GGUFValueType::INT64)   return kv->val_i64;
    }
    return def;
}
float GGUFHeader::get_f32(const std::string& key, float def) const {
    if (auto* kv = find_kv(key)) {
        if (kv->value_type == GGUFValueType::FLOAT32) return kv->val_f32;
        if (kv->value_type == GGUFValueType::FLOAT64) return static_cast<float>(kv->val_f64);
    }
    return def;
}
std::string GGUFHeader::get_str(const std::string& key, const std::string& def) const {
    if (auto* kv = find_kv(key)) return kv->val_str.empty() ? def : kv->val_str;
    return def;
}
bool GGUFHeader::get_bool(const std::string& key, bool def) const {
    if (auto* kv = find_kv(key)) return kv->val_bool;
    return def;
}

// ─── GGML type utility ───────────────────────────────────────────
static int ggml_blck_size(GGMLType t) {
    switch (t) {
        case GGMLType::Q4_0: case GGMLType::Q4_1:
        case GGMLType::Q5_0: case GGMLType::Q5_1:
        case GGMLType::Q8_0: case GGMLType::Q8_1:
            return 32;
        case GGMLType::Q2_K: case GGMLType::Q3_K: case GGMLType::Q4_K:
        case GGMLType::Q5_K: case GGMLType::Q6_K: case GGMLType::Q8_K:
        case GGMLType::IQ2_XXS: case GGMLType::IQ2_XS: case GGMLType::IQ3_XXS:
        case GGMLType::IQ3_S: case GGMLType::IQ1_S: case GGMLType::IQ1_M:
        case GGMLType::IQ2_S: case GGMLType::IQ4_NL: case GGMLType::IQ4_XS:
            return 256;
        default: return 1;
    }
}

static size_t ggml_type_size_bytes(GGMLType t) {
    switch (t) {
        case GGMLType::F32:   return 4;  case GGMLType::F16: return 2;
        case GGMLType::BF16:  return 2;  case GGMLType::I8:  return 1;
        case GGMLType::Q4_0:  return 18; case GGMLType::Q4_1: return 20;
        case GGMLType::Q5_0:  return 22; case GGMLType::Q5_1: return 24;
        case GGMLType::Q8_0:  return 34; case GGMLType::Q8_1: return 36;
        case GGMLType::Q2_K:  return 84; case GGMLType::Q3_K: return 112;
        case GGMLType::Q4_K:  return 144; case GGMLType::Q5_K: return 176;
        case GGMLType::Q6_K:  return 210; case GGMLType::I16: return 2;
        case GGMLType::I32:   return 4;  case GGMLType::I64: return 8;
        case GGMLType::F64:   return 8;  default: return 4;
    }
}

size_t ggml_type_size(GGMLType t) { return ggml_type_size_bytes(t); }

Dtype ggml_to_hesa_dtype(GGMLType t) {
    switch (t) {
        case GGMLType::F32:  return Dtype::F32;  case GGMLType::F16: return Dtype::F16;
        case GGMLType::BF16: return Dtype::BF16; case GGMLType::I8:  return Dtype::I8;
        case GGMLType::Q4_0: return Dtype::Q4_0; case GGMLType::Q4_1: return Dtype::Q4_1;
        case GGMLType::Q5_0: return Dtype::Q5_0; case GGMLType::Q5_1: return Dtype::Q5_1;
        case GGMLType::Q8_0: return Dtype::Q8_0; case GGMLType::Q2_K: return Dtype::Q2_K;
        case GGMLType::Q3_K: return Dtype::Q3_K; case GGMLType::Q4_K: return Dtype::Q4_K;
        case GGMLType::Q5_K: return Dtype::Q5_K; case GGMLType::Q6_K: return Dtype::Q6_K;
        default: return Dtype::F32;
    }
}

uint64_t tensor_padded_size(int64_t nelements, GGMLType t) {
    int blck = ggml_blck_size(t);
    size_t per_block = ggml_type_size_bytes(t);
    uint64_t n_blocks = static_cast<uint64_t>((nelements + blck - 1) / blck);
    return n_blocks * per_block;
}

// ─── Parser ──────────────────────────────────────────────────────
Result<GGUFHeader> parse_gguf(MmapRegion& file, uint64_t& tensor_data_start, uint64_t& header_end) {
    if (file.data() == nullptr || file.size() < 32)
        return std::unexpected{make_error_code(Error::INVALID_FORMAT)};

    GGUFHeader hdr;
    const uint8_t* p = file.data();
    size_t file_size = file.size();

    hdr.magic = read_val<uint32_t>(p);
    if (hdr.magic != GGUF_MAGIC) {
        fprintf(stderr, "GGUF: bad magic 0x%x\n", hdr.magic);
        return std::unexpected{make_error_code(Error::INVALID_FORMAT)};
    }

    hdr.version = read_val<uint32_t>(p + 4);
    if (hdr.version != GGUF_VERSION) {
        fprintf(stderr, "GGUF: unsupported version %u\n", hdr.version);
        return std::unexpected{make_error_code(Error::INVALID_FORMAT)};
    }

    uint64_t off = 8;
    hdr.n_tensors = read_val<uint64_t>(p + off); off += 8;
    hdr.n_kv      = read_val<uint64_t>(p + off); off += 8;

    fprintf(stderr, "GGUF v%u: %lu KV pairs, %lu tensors\n",
            hdr.version, (unsigned long)hdr.n_kv, (unsigned long)hdr.n_tensors);

    // KV pairs
    hdr.kv_pairs.reserve(hdr.n_kv);
    for (uint64_t i = 0; i < hdr.n_kv && off < file_size; ++i) {
        GGUFKV kv;
        kv.key = read_gguf_str(p + off, off);
        if (off >= file_size) return std::unexpected{make_error_code(Error::INVALID_FORMAT)};

        int32_t vtype = read_val<int32_t>(p + off); off += 4;
        kv.value_type = static_cast<GGUFValueType>(vtype);

        switch (kv.value_type) {
            case GGUFValueType::UINT8: { uint8_t v = p[off++];
                kv.val_u32=v;kv.val_i32=v;kv.val_u64=v;kv.val_i64=v; break; }
            case GGUFValueType::INT8: { int8_t v = static_cast<int8_t>(p[off++]);
                kv.val_i32=v;kv.val_i64=v; break; }
            case GGUFValueType::UINT16: { uint16_t v = read_val<uint16_t>(p+off); off+=2;
                kv.val_u32=v;kv.val_i32=(int32_t)v;kv.val_u64=v;kv.val_i64=v; break; }
            case GGUFValueType::INT16: { int16_t v = read_val<int16_t>(p+off); off+=2;
                kv.val_i32=v;kv.val_i64=v; break; }
            case GGUFValueType::UINT32: { kv.val_u32 = read_val<uint32_t>(p+off); off+=4;
                kv.val_i32=(int32_t)kv.val_u32; kv.val_u64=kv.val_u32; kv.val_i64=kv.val_u32; break; }
            case GGUFValueType::INT32: { kv.val_i32 = read_val<int32_t>(p+off); off+=4;
                kv.val_u32=(uint32_t)kv.val_i32; kv.val_u64=kv.val_i32; kv.val_i64=kv.val_i32; break; }
            case GGUFValueType::FLOAT32: { kv.val_f32 = read_val<float>(p+off); off+=4; break; }
            case GGUFValueType::BOOL: { kv.val_bool=(p[off++]!=0); break; }
            case GGUFValueType::STRING: { kv.val_str = read_gguf_str(p+off, off); break; }
            case GGUFValueType::UINT64: { kv.val_u64 = read_val<uint64_t>(p+off); off+=8;
                kv.val_i64=(int64_t)kv.val_u64; break; }
            case GGUFValueType::INT64: { kv.val_i64 = read_val<int64_t>(p+off); off+=8;
                kv.val_u64=(uint64_t)kv.val_i64; break; }
            case GGUFValueType::FLOAT64: { kv.val_f64 = read_val<double>(p+off); off+=8; break; }
            case GGUFValueType::ARRAY: {
                int32_t atype = read_val<int32_t>(p+off); off+=4;
                uint64_t alen = read_val<uint64_t>(p+off); off+=8;
                auto agt = static_cast<GGUFValueType>(atype);
                kv.val_array.reserve(alen);
                for (uint64_t j = 0; j < alen && off < file_size; ++j) {
                    if (agt == GGUFValueType::STRING)
                        kv.val_array.emplace_back(read_gguf_str(p+off, off));
                    else if (agt == GGUFValueType::FLOAT32 || agt == GGUFValueType::FLOAT64)
                        { double v = read_val<double>(p+off); off+=8; kv.val_array.emplace_back(v); }
                    else
                        { uint64_t v = read_val<uint64_t>(p+off); off+=8; kv.val_array.emplace_back((int64_t)v); }
                }
                break;
            }
            default: break;
        }
        hdr.kv_pairs.push_back(std::move(kv));
    }

    off = (off + 31) & ~uint64_t(31);

    // Tensor info
    hdr.tensor_infos.reserve(hdr.n_tensors);
    for (uint64_t i = 0; i < hdr.n_tensors && off < file_size; ++i) {
        GGUFTensorInfo ti;
        ti.name = read_gguf_str(p + off, off);
        if (off >= file_size) return std::unexpected{make_error_code(Error::INVALID_FORMAT)};

        ti.n_dims = read_val<uint32_t>(p + off); off += 4;
        if (ti.n_dims > 4) ti.n_dims = 4;
        ti.type = static_cast<GGMLType>(read_val<int32_t>(p + off)); off += 4;
        for (uint32_t d = 0; d < ti.n_dims; ++d) { ti.dims[d] = read_val<int64_t>(p + off); off += 8; }
        ti.offset = read_val<uint64_t>(p + off); off += 8;
        hdr.tensor_infos.push_back(std::move(ti));
    }

    tensor_data_start = (off + 31) & ~uint64_t(31);
    header_end = tensor_data_start;
    if (tensor_data_start > file_size)
        return std::unexpected{make_error_code(Error::INVALID_FORMAT)};

    fprintf(stderr, "GGUF: parsed OK, %lu tensors, data @ offset %lu\n",
            (unsigned long)hdr.tensor_infos.size(), (unsigned long)tensor_data_start);
    return hdr;
}

Result<GGUFFile> load_gguf(const std::string& filepath) {
    GGUFFile result;
    auto mm = MmapRegion::open(filepath);
    if (!mm) return std::unexpected{mm.error()};
    result.mmap = std::move(*mm);

    uint64_t tds = 0, hend = 0;
    auto hdr = parse_gguf(result.mmap, tds, hend);
    if (!hdr) return std::unexpected{hdr.error()};
    result.header = std::move(*hdr);
    result.tensor_data_start = tds;
    result.header_end = hend;
    return result;
}

} // namespace hesa

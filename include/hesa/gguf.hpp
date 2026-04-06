#ifndef HESA_GGUF_HPP
#define HESA_GGUF_HPP

#include "hesa/result.hpp"
#include "hesa/tensor.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace hesa {

// ─── GGUF magic & version ─────────────────────────────────────────
static constexpr uint32_t GGUF_MAGIC   = 0x46554747u; // "GGUF"
static constexpr uint32_t GGUF_VERSION = 3u;

// ─── GGUF value-type enum (from spec) ─────────────────────────────
enum class GGUFValueType : int32_t {
    UINT8   = 0,
    INT8    = 1,
    UINT16  = 2,
    INT16   = 3,
    UINT32  = 4,
    INT32   = 5,
    FLOAT32 = 6,
    BOOL    = 7,
    STRING  = 8,
    ARRAY   = 9,
    UINT64  = 10,
    INT64   = 11,
    FLOAT64 = 12,
};

// ─── GGFF tensor type (ggml_type) ────────────────────────────────
enum class GGMLType : int32_t {
    F32    =  0,
    F16    =  1,
    Q4_0   =  2,
    Q4_1   =  3,
    // 4-5 reserved
    Q5_0   =  6,
    Q5_1   =  7,
    Q8_0   =  8,
    Q8_1   =  9,
    Q2_K   = 10,
    Q3_K   = 11,
    Q4_K   = 12,
    Q5_K   = 13,
    Q6_K   = 14,
    Q8_K   = 15,
    IQ2_XXS= 16,
    IQ2_XS = 17,
    IQ3_XXS= 18,
    IQ1_S  = 19,
    IQ4_NL = 20,
    IQ3_S  = 21,
    IQ2_S  = 22,
    IQ4_XS = 23,
    I8     = 24,
    I16    = 25,
    I32    = 26,
    I64    = 27,
    F64    = 28,
    IQ1_M  = 29,
    BF16   = 30,
};

// ─── KV value union ──────────────────────────────────────────────
struct GGUFKV {
    std::string key;
    GGUFValueType value_type;

    // Scalar values
    uint32_t     val_u32   = 0;
    int32_t      val_i32   = 0;
    uint64_t     val_u64   = 0;
    int64_t      val_i64   = 0;
    float        val_f32   = 0.0f;
    double       val_f64   = 0.0f;
    bool         val_bool  = false;
    std::string  val_str;

    // Array: each element is a variant
    std::vector<std::variant<int64_t, double, std::string>> val_array;
};

// ─── Tensor info ──────────────────────────────────────────────────
struct GGUFTensorInfo {
    std::string  name;
    uint32_t     n_dims  = 0;
    int64_t      dims[4] = {1, 1, 1, 1}; // GGUF stores dims in reverse (C-order) order
    GGMLType     type    = GGMLType::F32;
    uint64_t     offset  = 0;            // byte offset from tensor-data region start

    int64_t nelements() const {
        int64_t n = 1;
        for (uint32_t i = 0; i < n_dims; ++i) n *= dims[i];
        return n;
    }
};

// ─── Header ───────────────────────────────────────────────────────
struct GGUFHeader {
    uint32_t                      magic      = 0;
    uint32_t                      version    = 0;
    uint64_t                      n_tensors  = 0;
    uint64_t                      n_kv       = 0;
    std::vector<GGUFKV>           kv_pairs;
    std::vector<GGUFTensorInfo>   tensor_infos;

    // Convenience: look up a key
    const GGUFKV* find_kv(const std::string& key) const;
    int64_t       get_i64(const std::string& key, int64_t def = 0) const;
    float         get_f32(const std::string& key, float def = 0.0f) const;
    std::string   get_str(const std::string& key, const std::string& def = "") const;
    bool          get_bool(const std::string& key, bool def = false) const;
};

// ─── Mmap'd data region (RAII) ────────────────────────────────────
class MmapRegion {
public:
    MmapRegion() = default;
    ~MmapRegion();

    MmapRegion(MmapRegion&&) noexcept;
    MmapRegion& operator=(MmapRegion&&) noexcept;
    MmapRegion(const MmapRegion&) = delete;
    MmapRegion& operator=(const MmapRegion&) = delete;

    static Result<MmapRegion> open(const std::string& path);

    const uint8_t*  data() const { return data_; }
    uint8_t*        data()       { return data_; }
    size_t          size() const { return size_; }
    const std::string& path() const { return path_; }
    int             fd()   const { return fd_; }

private:
    std::string path_;
    int         fd_   = -1;
    uint8_t*    data_ = nullptr;
    size_t      size_ = 0;
};

// ─── Load / parse ─────────────────────────────────────────────────

/**
 * Parse a GGUF file header from an mmap'd region.
 * Returns the header; tensor data lives inside [mmap + header size]
 * and can be accessed via tensor_infos[].offset relative to the
 * tensor-data region start (returned as *tensor_data_start).
 */
Result<GGUFHeader> parse_gguf(MmapRegion& mmap,
                              uint64_t& tensor_data_start,
                              uint64_t& header_end);

/**
 * Convenience: open file, parse header, and return everything in one
 * call.  The MmapRegion is moved into the return tuple so it stays
 * alive.
 */
struct GGUFFile {
    MmapRegion   mmap;
    GGUFHeader   header;
    uint64_t     tensor_data_start = 0;  // absolute file offset
    uint64_t     header_end        = 0;  // absolute file offset (after tensor-info)
};

Result<GGUFFile> load_gguf(const std::string& filepath);

// ─── Utility ──────────────────────────────────────────────────────
size_t       ggml_type_size(GGMLType t);
Dtype        ggml_to_hesa_dtype(GGMLType t);
uint64_t     tensor_padded_size(int64_t nelements, GGMLType t);

} // namespace hesa

#endif // HESA_GGUF_HPP

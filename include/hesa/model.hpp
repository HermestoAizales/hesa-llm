#ifndef HESA_MODEL_HPP
#define HESA_MODEL_HPP

#include "hesa/result.hpp"
#include "hesa/tensor.hpp"
#include "hesa/backend.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace hesa {

/**
 * Model metadata key-value pairs as loaded from GGUF/HESF files.
 */
struct ModelMetadata {
    std::string architecture;       // e.g. "llama", "gemma"
    uint32_t vocab_size = 0;
    uint32_t embedding_length = 0;
    uint32_t block_count = 0;
    uint32_t feed_forward_length = 0;
    uint32_t attention_head_count = 0;
    uint32_t attention_head_count_kv = 0;
    uint32_t attention_layer_norm_rms_epsilon = 0;
    uint32_t rope_dimension_count = 0;
    float    rope_freq_base = 10000.0f;
    uint32_t context_length = 0;

    // HESA-specific metadata keys
    bool use_neural_memory = false;
    bool use_ttt = false;
    bool use_hyper_connections = false;
    bool engram_enabled = false;

    std::unordered_map<std::string, std::variant<std::string, int64_t, double, bool>> custom;
};

/**
 * Loaded model — weights and metadata from a GGUF/HESF file.
 * Weights are mmapped for zero-copy access.
 */
class Model {
public:
    static Result<std::unique_ptr<Model>> load(const std::string& path,
                                                Backend* backend = nullptr);
    ~Model();

    const ModelMetadata& metadata() const { return metadata_; }

    // Lookup tensor by name (e.g. "blk.0.attn_q.weight")
    Result<const Tensor*> get_tensor(const std::string& name) const;

    // All loaded tensor names
    const std::vector<std::string>& tensor_names() const { return tensor_names_; }

    // Direct access to raw tensors (owned by Model)
    const std::unordered_map<std::string, Tensor>& tensors() const { return tensors_; }

    // File size in bytes
    size_t file_size() const { return file_size_; }
    const std::string& path() const { return path_; }

private:
    struct MappedFile;
    std::unique_ptr<MappedFile> mapped_file_;
    ModelMetadata metadata_;
    std::unordered_map<std::string, Tensor> tensors_;
    std::vector<std::string> tensor_names_;
    std::string path_;
    size_t file_size_ = 0;
};

} // namespace hesa

#endif // HESA_MODEL_HPP

#ifndef HESA_TOKENIZER_HPP
#define HESA_TOKENIZER_HPP

#include "hesa/result.hpp"
namespace hesa { class Model; }

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace hesa {

/**
 * Abstract tokenizer interface.
 */
class Tokenizer {
public:
    virtual ~Tokenizer() = default;

    virtual std::vector<int32_t> encode(const std::string& text) const = 0;
    virtual std::string decode(std::span<const int32_t> tokens) const = 0;

    virtual std::vector<std::vector<int32_t>>
        encode_batch(std::span<const std::string> texts) const;

    virtual int32_t bos_token_id() const = 0;
    virtual int32_t eos_token_id() const = 0;
    virtual int32_t vocab_size() const = 0;
    virtual size_t max_token_length() const { return 256; }
};

// -- Factory --

enum class TokenizerKind : uint8_t {
    BPE,        // OpenAI-style BPE (tiktoken-compatible)
    SENTENCEPIECE  // BPE or Unigram model from SentencePiece
};

/**
 * Create a tokenizer from a GGUF model file.
 * Reads tokenizer data embedded in the model (pre-tokenizer rules,
 * vocab, merges, added tokens).
 */
Result<std::unique_ptr<Tokenizer>> create_tokenizer_from_model(const Model& model);

/**
 * Create a BPE tokenizer from raw vocab and merge data.
 * @param vocab vector of (token_string, token_id) pairs
 * @param merges vector of (left, right) merge rules
 */
Result<std::unique_ptr<Tokenizer>> create_bpe_tokenizer(
    std::vector<std::pair<std::string, int32_t>> vocab,
    std::vector<std::pair<std::string, std::string>> merges,
    int32_t bos_id, int32_t eos_id);

} // namespace hesa

#endif // HESA_TOKENIZER_HPP

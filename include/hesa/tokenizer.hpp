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

/**
 * Create a SentencePiece tokenizer from GGUF-extracted vocab data.
 * @param tokens array of token strings (tokenizer.ggml.tokens)
 * @param scores log-probabilities for unigram, or merge ranks for BPE
 * @param token_types GGUF token types (1=normal, 2=unknown, 3=control, 4=unused, 5=byte, 6=byte)
 * @param merges BPE merge rules (tokenizer.ggml.merges), empty for unigram
 * @param model_type "unigram" or "bpe"
 * @param bos_id bos token id
 * @param eos_id eos token id
 * @param unk_id unknown token id (-1 if none)
 */
Result<std::unique_ptr<Tokenizer>> create_sp_tokenizer(
    std::vector<std::string> tokens,
    std::vector<float> scores,
    std::vector<int32_t> token_types,
    std::vector<std::string> merges,
    const std::string& model_type,
    int32_t bos_id, int32_t eos_id, int32_t unk_id = -1);

} // namespace hesa

#endif // HESA_TOKENIZER_HPP

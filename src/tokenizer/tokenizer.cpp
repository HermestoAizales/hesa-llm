#include "hesa/tokenizer.hpp"
#include "hesa/model.hpp"
#include "hesa/result.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace hesa {

// ============================================================================
//  BPE Tokenizer
// ============================================================================

class BPE_Tokenizer final : public Tokenizer {
public:
    BPE_Tokenizer(int32_t bos_id, int32_t eos_id,
                  std::vector<std::pair<std::string, int32_t>> vocab,
                  std::vector<std::pair<std::string, std::string>> merges)
        : bos_id_(bos_id), eos_id_(eos_id),
          vocab_(std::move(vocab)), merges_(std::move(merges))
    {
        // Build reverse map: string -> id
        for (const auto& [s, id] : vocab_)
            token_to_id_[s] = id;
        max_token_length_ = 0;
        for (const auto& [s, _] : vocab_)
            if (s.size() > max_token_length_) max_token_length_ = s.size();
    }

    int32_t bos_token_id() const override { return bos_id_; }
    int32_t eos_token_id() const override { return eos_id_; }
    int32_t vocab_size() const override { return static_cast<int32_t>(vocab_.size()); }
    size_t max_token_length() const override { return max_token_length_; }

    std::vector<int32_t> encode(const std::string& text) const override {
        // Step 1: byte-level BPE pre-tokenization
        // Map each byte to initial tokens
        std::vector<std::string> tokens;
        for (unsigned char c : text) {
            // Use byte value as token (hex representation or direct)
            // Simplified: treat each character/byte as a token
            tokens.push_back(std::string(1, c));
        }

        // Step 2: Apply merge rules
        auto merged = apply_merges(tokens);

        // Step 3: Convert to IDs
        std::vector<int32_t> ids;
        ids.reserve(merged.size());
        for (const auto& t : merged) {
            auto it = token_to_id_.find(t);
            if (it != token_to_id_.end())
                ids.push_back(it->second);
            else
                ids.push_back(unk_id()); // fallback
        }
        return ids;
    }

    std::string decode(std::span<const int32_t> tokens) const override {
        // Build reverse map: id -> string
        std::unordered_map<int32_t, std::string> id_to_token;
        for (const auto& [s, id] : vocab_)
            id_to_token[id] = s;

        std::string result;
        for (int32_t id : tokens) {
            auto it = id_to_token.find(id);
            if (it != id_to_token.end())
                result += it->second;
        }
        return result;
    }

private:
    int32_t bos_id_, eos_id_;
    std::vector<std::pair<std::string, int32_t>> vocab_;
    std::vector<std::pair<std::string, std::string>> merges_;
    std::unordered_map<std::string, int32_t> token_to_id_;
    size_t max_token_length_ = 0;

    int32_t unk_id() const {
        auto it = token_to_id_.find("<unk>");
        if (it != token_to_id_.end()) return it->second;
        return vocab_.empty() ? 0 : vocab_[0].second;
    }

    std::vector<std::string> apply_merges(const std::vector<std::string>& initial) const {
        std::vector<std::string> tokens = initial;
        // Apply merge rules in order (greedy)
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto it = merges_.begin(); it != merges_.end() && !changed; ++it) {
                for (size_t i = 0; i + 1 < tokens.size(); ++i) {
                    if (tokens[i] == it->first && tokens[i + 1] == it->second) {
                        tokens[i] = it->first + it->second;
                        tokens.erase(tokens.begin() + i + 1);
                        changed = true;
                        break;
                    }
                }
            }
        }
        return tokens;
    }
};

// ============================================================================
//  Stub SentencePiece Tokenizer
// ============================================================================

class SP_Tokenizer final : public Tokenizer {
public:
    SP_Tokenizer(int32_t bos_id, int32_t eos_id)
        : bos_id_(bos_id), eos_id_(eos_id) {}

    int32_t bos_token_id() const override { return bos_id_; }
    int32_t eos_token_id() const override { return eos_id_; }
    int32_t vocab_size() const override { return 0; }

    std::vector<int32_t> encode(const std::string&) const override {
        return {}; // Not implemented — needs SentencePiece model
    }
    std::string decode(std::span<const int32_t>) const override { return {}; }

private:
    int32_t bos_id_, eos_id_;
};

// ============================================================================
//  Factory
// ============================================================================

std::vector<std::vector<int32_t>>
Tokenizer::encode_batch(std::span<const std::string> texts) const {
    std::vector<std::vector<int32_t>> result;
    result.reserve(texts.size());
    for (const auto& t : texts)
        result.push_back(encode(t));
    return result;
}

Result<std::unique_ptr<Tokenizer>> create_bpe_tokenizer(
    std::vector<std::pair<std::string, int32_t>> vocab,
    std::vector<std::pair<std::string, std::string>> merges,
    int32_t bos_id, int32_t eos_id)
{
    return std::make_unique<BPE_Tokenizer>(
        bos_id, eos_id, std::move(vocab), std::move(merges));
}

Result<std::unique_ptr<Tokenizer>> create_tokenizer_from_model(const Model& model) {
    const auto& meta = model.metadata();

    // Try to extract vocab from token_embd.weight tensor
    auto embd = model.get_tensor("token_embd.weight");
    if (!embd) {
        // No embedding tensor — create minimal fallback tokenizer
        auto tok = std::make_unique<BPE_Tokenizer>(
            1, 2,
            std::vector<std::pair<std::string, int32_t>>{},
            std::vector<std::pair<std::string, std::string>>{}
        );
        return tok;
    }

    // For now, create a BPE tokenizer with empty vocab.
    // Full extraction from GGUF tokenizer fields needs more parsing.
    return std::make_unique<BPE_Tokenizer>(
        1, 2,
        std::vector<std::pair<std::string, int32_t>>{},
        std::vector<std::pair<std::string, std::string>>{}
    );
}

} // namespace hesa

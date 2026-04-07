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
//  SentencePiece Tokenizer (Unigram / BPE)
// ============================================================================

enum class SPModelType { UNIGRAM, BPE };

enum class SPTokenType : int32_t {
    NORMAL   = 1,
    UNKNOWN  = 2,
    CONTROL  = 3,
    UNUSED   = 4,
    BYTE     = 5,
    USER_DEF = 6,
};

class SP_Tokenizer final : public Tokenizer {
public:
    SP_Tokenizer(SPModelType model_type,
                 std::vector<std::string> tokens,
                 std::vector<float> scores,
                 std::vector<SPTokenType> token_types,
                 std::vector<std::pair<std::string, std::string>> merge_rules,
                 int32_t bos_id, int32_t eos_id, int32_t unk_id)
        : model_type_(model_type),
          tokens_(std::move(tokens)),
          scores_(std::move(scores)),
          token_types_(std::move(token_types)),
          merge_rules_(std::move(merge_rules)),
          bos_id_(bos_id), eos_id_(eos_id), unk_id_(unk_id)
    {
        // Build string -> id and id -> string maps
        for (int32_t i = 0; i < static_cast<int32_t>(tokens_.size()); ++i) {
            str_to_id_[tokens_[i]] = i;
        }

        max_token_length_ = 0;
        for (const auto& t : tokens_)
            if (t.size() > max_token_length_) max_token_length_ = t.size();

        // Extract byte tokens for fallback (type 5 = byte)
        for (int32_t i = 0; i < static_cast<int32_t>(token_types_.size()); ++i) {
            if (token_types_[i] == SPTokenType::BYTE && !tokens_[i].empty()) {
                byte_token_map_.push_back({static_cast<uint8_t>(i / 256), static_cast<uint8_t>(i % 256)});
            }
        }

        // Build merge rule index for BPE: map "left right" -> rank
        for (size_t i = 0; i < merge_rules_.size(); ++i) {
            std::string key = merge_rules_[i].first + " " + merge_rules_[i].second;
            merge_rank_[std::move(key)] = static_cast<int32_t>(i);
        }
    }

    int32_t bos_token_id() const override { return bos_id_; }
    int32_t eos_token_id() const override { return eos_id_; }
    int32_t vocab_size() const override { return static_cast<int32_t>(tokens_.size()); }
    size_t max_token_length() const override { return max_token_length_; }

    std::vector<int32_t> encode(const std::string& text) const override {
        if (model_type_ == SPModelType::BPE)
            return encode_bpe(text);
        else
            return encode_unigram(text);
    }

    std::string decode(std::span<const int32_t> ids) const override {
        return decode_impl(ids);
    }

private:
    SPModelType model_type_;
    std::vector<std::string> tokens_;
    std::vector<float> scores_;
    std::vector<SPTokenType> token_types_;
    std::vector<std::pair<std::string, std::string>> merge_rules_;
    std::unordered_map<std::string, int32_t> str_to_id_;
    std::unordered_map<std::string, int32_t> merge_rank_;
    int32_t bos_id_, eos_id_, unk_id_;
    size_t max_token_length_ = 0;
    std::vector<std::pair<uint8_t, uint8_t>> byte_token_map_;

    // ── Decode ──────────────────────────────────────────────────────────

    std::string decode_impl(std::span<const int32_t> ids) const {
        std::string result;
        for (int32_t id : ids) {
            if (id < 0 || id >= static_cast<int32_t>(tokens_.size())) continue;
            if (token_types_[id] == SPTokenType::CONTROL) continue;
            result += detokenize(tokens_[id]);
        }
        return result;
    }

    /** Convert a SentencePiece token string back to its raw text form.
     *  SentencePiece encodes:
     *  - spaces as '\xe2\x96\x81' (U+2581 "lower block")  
     *  - non-printable / byte tokens as <0xHH>
     */
    std::string detokenize(const std::string& tok) const {
        // Handle <0xHH> encoded bytes (SentencePiece style)
        std::string out;
        out.reserve(tok.size());

        size_t i = 0;
        while (i < tok.size()) {
            if (i + 6 <= tok.size() && tok[i] == '<' && tok[i+1] == '0' &&
                tok[i+2] == 'x' && tok[i+5] == '>')
            {
                // Decode hex byte
                char hb = tok[i+3], lb = tok[i+4];
                uint8_t val = static_cast<uint8_t>(hex_byte(hb) * 16 + hex_byte(lb));
                out += static_cast<char>(val);
                i += 6;
            }
            else {
                out += tok[i];
                i++;
            }
        }
        return out;
    }

    static constexpr uint8_t hex_byte(char c) {
        if (c >= '0' && c <= '9') return c - '0';
        if (c >= 'a' && c <= 'f') return c - 'a' + 10;
        if (c >= 'A' && c <= 'F') return c - 'A' + 10;
        return 0;
    }

    // ── BPE Encoding ────────────────────────────────────────────────────

    std::vector<int32_t> encode_bpe(std::string_view text) const {
        // Split input into initial pieces (characters/bytes)
        std::vector<std::string> pieces;
        pieces.reserve(text.size());
        for (size_t i = 0; i < text.size();) {
            // Try to decode UTF-8 sequence
            unsigned char c = static_cast<unsigned char>(text[i]);
            size_t len = utf8_char_len(c);
            if (len == 0 || i + len > text.size()) len = 1;
            pieces.emplace_back(text.substr(i, len));
            i += len;
        }

        if (pieces.empty()) return {};

        // Apply merge rules iteratively (greedy lowest rank first)
        apply_bpe_merges(pieces);

        // Convert to IDs
        std::vector<int32_t> ids;
        ids.reserve(pieces.size());
        for (const auto& p : pieces) {
            auto it = str_to_id_.find(p);
            if (it != str_to_id_.end())
                ids.push_back(it->second);
            else if (unk_id_ >= 0)
                ids.push_back(unk_id_);
        }
        return ids;
    }

    static size_t utf8_char_len(unsigned char c) {
        if (c < 0x80) return 1;
        if ((c & 0xE0) == 0xC0) return 2;
        if ((c & 0xF0) == 0xE0) return 3;
        if ((c & 0xF8) == 0xF0) return 4;
        return 0;
    }

    void apply_bpe_merges(std::vector<std::string>& pieces) const {
        while (pieces.size() > 1) {
            // Find the best merge (lowest rank = earliest in merge list)
            int32_t best_rank = INT32_MAX;
            size_t best_pos = 0;

            for (size_t i = 0; i + 1 < pieces.size(); ++i) {
                std::string key = pieces[i] + " " + pieces[i + 1];
                auto it = merge_rank_.find(key);
                if (it != merge_rank_.end() && it->second < best_rank) {
                    best_rank = it->second;
                    best_pos = i;
                }
            }

            if (best_rank == INT32_MAX) break; // No more merges

            // Merge
            pieces[best_pos] = pieces[best_pos] + pieces[best_pos + 1];
            pieces.erase(pieces.begin() + static_cast<ptrdiff_t>(best_pos) + 1);
        }
    }

    // ── Unigram Encoding (Viterbi segmentation) ─────────────────────────

    std::vector<int32_t> encode_unigram(std::string_view text) const {
        if (text.empty()) return {};

        // Build all possible token matches at each byte position
        size_t n = text.size();

        // Viterbi: best_score[i] = best log-prob to cover text[0..i)
        // best_token_id[i] = token id used to reach position i
        // prev_pos[i] = previous position
        std::vector<float> best_score(n + 1, -INFINITY);
        std::vector<int32_t> best_token_id(n + 1, unk_id_);
        std::vector<int32_t> prev_pos(n + 1, -1);

        best_score[0] = 0.0f;

        for (size_t i = 0; i < n; ++i) {
            if (best_score[i] == -INFINITY) continue;

            // Try all token lengths starting at position i
            size_t max_len = std::min(max_token_length_, n - i);
            for (size_t len = 1; len <= max_len; ++len) {
                std::string_view candidate = text.substr(i, len);
                auto it = str_to_id_.find(std::string(candidate));
                if (it != str_to_id_.end()) {
                    int32_t tid = it->second;
                    if (token_types_[tid] == SPTokenType::CONTROL ||
                        token_types_[tid] == SPTokenType::UNUSED)
                        continue;

                    // Skip <UNK> token (we use explicit unk_id_ for fallback)
                    if (token_types_[tid] == SPTokenType::UNKNOWN) continue;

                    float new_score = best_score[i] + scores_[tid];
                    if (new_score > best_score[i + len]) {
                        best_score[i + len] = new_score;
                        best_token_id[i + len] = tid;
                        prev_pos[i + len] = static_cast<int32_t>(i);
                    }
                }
            }
        }

        // Handle unknown bytes: fill gaps where no token matched
        for (size_t i = 1; i <= n; ++i) {
            if (best_score[i] == -INFINITY) {
                // Try byte-level fallback
                for (size_t len = 1; len <= 4 && i >= len; ++len) {
                    size_t start = i - len;
                    if (best_score[start] != -INFINITY) {
                        // Use <0xHH> tokens or unk_id_
                        for (size_t j = start; j < i; ++j) {
                            uint8_t byte = static_cast<uint8_t>(text[j]);
                            std::string byte_tok = byte_to_spm_token(byte);
                            auto it = str_to_id_.find(byte_tok);
                            if (it != str_to_id_.end() && token_types_[it->second] == SPTokenType::BYTE) {
                                float s = best_score[j] + scores_[it->second];
                                if (s > best_score[j + 1]) {
                                    best_score[j + 1] = s;
                                    best_token_id[j + 1] = it->second;
                                    prev_pos[j + 1] = static_cast<int32_t>(j);
                                }
                            }
                        }
                        break;
                    }
                }
                // Final fallback: use unk token
                if (best_score[i] == -INFINITY && unk_id_ >= 0 && static_cast<size_t>(unk_id_) < tokens_.size()) {
                    // Find the nearest reachable position
                    for (int32_t j = static_cast<int32_t>(i) - 1; j >= 0; --j) {
                        if (best_score[j] != -INFINITY) {
                            best_score[i] = best_score[j] + (unk_id_ >= 0 && static_cast<size_t>(unk_id_) < scores_.size() ? scores_[unk_id_] : 0.0f);
                            best_token_id[i] = unk_id_;
                            prev_pos[i] = j;
                            break;
                        }
                    }
                }
            }
        }

        // Backtrack to collect token IDs
        std::vector<int32_t> ids;
        int32_t pos = static_cast<int32_t>(n);
        while (pos > 0) {
            int32_t pp = prev_pos[pos];
            if (pp < 0) break;
            ids.push_back(best_token_id[pos]);
            pos = pp;
        }
        std::reverse(ids.begin(), ids.end());
        return ids;
    }

    /** Encode a single byte as a SentencePiece <0xHH> token string. */
    std::string byte_to_spm_token(uint8_t byte) const {
        const char hex[] = "0123456789abcdef";
        std::string tok;
        tok.reserve(6);
        tok += '<';
        tok += '0';
        tok += 'x';
        tok += hex[(byte >> 4) & 0xF];
        tok += hex[byte & 0xF];
        tok += '>';
        return tok;
    }
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

// ============================================================================
//  Factory
// ============================================================================

/** Extract GGUF tokenizer data from a model's metadata.
 *
 *  The GGUF loader stores all KV pairs in meta_custom_. We need:
 *  - tokenizer.ggml.tokens       (ARRAY of STRING)
 *  - tokenizer.ggml.scores       (ARRAY of FLOAT64)
 *  - tokenizer.ggml.token_type   (ARRAY of INT)
 *  - tokenizer.ggml.merges       (ARRAY of STRING)
 *  - tokenizer.ggml.bos_token_id (INT)
 *  - tokenizer.ggml.eos_token_id (INT)
 */
struct GGUFTokenizerData {
    std::vector<std::string> tokens;
    std::vector<float> scores;
    std::vector<int32_t> token_types;
    std::vector<std::string> merges;
    uint32_t vocab_size = 0;
    std::string model_type; // "llama" (BPE), "bert" (unigram), etc.
    int32_t bos_id = -1;
    int32_t eos_id = -1;
};

static Result<GGUFTokenizerData> extract_tokenizer_from_metadata(
    const std::unordered_map<std::string, std::variant<std::string, int64_t, double, bool>>& meta)
{
    GGUFTokenizerData td;

    // Determine model type from architecture
    auto arch_it = meta.find("general.architecture");
    if (arch_it != meta.end() && std::holds_alternative<std::string>(arch_it->second)) {
        td.model_type = std::get<std::string>(arch_it->second);
    }

    // BOS / EOS token IDs
    auto bos_it = meta.find("tokenizer.ggml.bos_token_id");
    if (bos_it != meta.end() && std::holds_alternative<int64_t>(bos_it->second))
        td.bos_id = static_cast<int32_t>(std::get<int64_t>(bos_it->second));

    auto eos_it = meta.find("tokenizer.ggml.eos_token_id");
    if (eos_it != meta.end() && std::holds_alternative<int64_t>(eos_it->second))
        td.eos_id = static_cast<int32_t>(std::get<int64_t>(eos_it->second));

    // Vocab size
    auto vs_it = meta.find("general.vocab_size");
    if (vs_it != meta.end() && std::holds_alternative<int64_t>(vs_it->second))
        td.vocab_size = static_cast<uint32_t>(std::get<int64_t>(vs_it->second));

    // tokenizer.ggml.tokens — stored as an ARRAY in the GGUF loader
    // The loader flattens arrays by consuming them in read_metadata().
    // We need to look at the keys: tokenizer.ggml.tokens is an ARRAY of STRING.
    // In the loader's read_metadata, it skips arrays (reads but doesn't store).
    // So we need to re-read them from the GGUF file. For now, check if they exist
    // in a flattened format (tokenizer.ggml.tokens.0, etc.) or as a special key.
    //
    // Since the GGUFReader currently consumes but doesn't store array contents
    // for non-string/non-numeric types, we need to handle this differently.
    // The GGUF spec uses array keys like "tokenizer.ggml.tokens" containing
    // an array value. Our simplified loader stores only scalar metadata.
    //
    // Workaround: check for individual indexed keys if array wasn't captured.
    for (auto& [key, val] : meta) {
        if (key.find("tokenizer.ggml.tokens.") == 0 && std::holds_alternative<std::string>(val)) {
            size_t idx = 0;
            if (sscanf(key.c_str() + strlen("tokenizer.ggml.tokens."), "%zu", &idx) == 1) {
                if (idx >= td.tokens.size()) td.tokens.resize(idx + 1);
                td.tokens[idx] = std::get<std::string>(val);
            }
        }
        if (key.find("tokenizer.ggml.scores.") == 0) {
            if (std::holds_alternative<double>(val)) {
                size_t idx = 0;
                if (sscanf(key.c_str() + strlen("tokenizer.ggml.scores."), "%zu", &idx) == 1) {
                    if (idx >= td.scores.size()) td.scores.resize(idx + 1);
                    td.scores[idx] = static_cast<float>(std::get<double>(val));
                }
            }
        }
        if (key.find("tokenizer.ggml.token_type.") == 0) {
            if (std::holds_alternative<int64_t>(val)) {
                size_t idx = 0;
                if (sscanf(key.c_str() + strlen("tokenizer.ggml.token_type."), "%zu", &idx) == 1) {
                    if (idx >= td.token_types.size()) td.token_types.resize(idx + 1);
                    td.token_types[idx] = static_cast<int32_t>(std::get<int64_t>(val));
                }
            }
        }
        if (key.find("tokenizer.ggml.merges.") == 0 && std::holds_alternative<std::string>(val)) {
            size_t idx = 0;
            if (sscanf(key.c_str() + strlen("tokenizer.ggml.merges."), "%zu", &idx) == 1) {
                if (idx >= td.merges.size()) td.merges.resize(idx + 1);
                td.merges[idx] = std::get<std::string>(val);
            }
        }
    }

    return td;
}

/** Build SP_Tokenizer model type from GGUF architecture name. */
[[maybe_unused]] static SPModelType sp_model_from_arch(const std::string& arch) {
    // llama, qwen2, gemma, mistral, etc. use SentencePiece BPE
    // bert uses unigram
    // t5 uses unigram
    if (arch == "bert" || arch == "nomic-bert" || arch == "t5" || arch == "qwen3moe")
        return SPModelType::UNIGRAM;
    // Default to BPE for most modern models
    return SPModelType::BPE;
}

Result<std::unique_ptr<Tokenizer>> create_sp_tokenizer(
    std::vector<std::string> tokens,
    std::vector<float> scores,
    std::vector<int32_t> token_types_raw,
    std::vector<std::string> merges,
    const std::string& model_type,
    int32_t bos_id, int32_t eos_id, int32_t unk_id)
{
    SPModelType mt = (model_type == "unigram") ? SPModelType::UNIGRAM : SPModelType::BPE;

    // Parse merge rules from BPE format: "left right" -> (left, right)
    std::vector<std::pair<std::string, std::string>> merge_rules;
    if (!merges.empty()) {
        for (const auto& m : merges) {
            // Find the last space that separates left and right parts
            size_t last_space = m.find(' ');
            if (last_space != std::string::npos && last_space < m.size() - 1) {
                merge_rules.emplace_back(m.substr(0, last_space), m.substr(last_space + 1));
            }
        }
    }

    // Convert raw int32_t token types to SPTokenType enum
    std::vector<SPTokenType> token_types;
    token_types.reserve(token_types_raw.size());
    for (int32_t t : token_types_raw) {
        token_types.push_back(static_cast<SPTokenType>(t));
    }

    // Fill in defaults for bos/eos/unk from token_types
    if (bos_id < 0 || eos_id < 0 || unk_id < 0) {
        for (int32_t i = 0; i < static_cast<int32_t>(token_types.size()); ++i) {
            if (token_types[i] == SPTokenType::CONTROL) {
                // Check if this looks like a BOS, EOS, or UNK token
                const auto& tok = tokens[static_cast<size_t>(i)];
                if (tok == "<s>" && bos_id < 0) bos_id = i;
                else if ((tok == "</s>" || tok == "<eos>" || tok == "<|end|>") && eos_id < 0) eos_id = i;
                else if ((tok == "<unk>" || tok == "<UNK>" || tok == "<|unk|>") && unk_id < 0) unk_id = i;
                else if (bos_id < 0) bos_id = i; // fallback: first control tok is BOS
                else if (eos_id < 0) eos_id = i; // fallback: second control tok is EOS
                else if (unk_id < 0) unk_id = i; // fallback: third control tok is UNK
            }
        }
    }
    if (unk_id < 0 && !tokens.empty()) unk_id = 0; // ultimate fallback

    return std::make_unique<SP_Tokenizer>(
        mt, std::move(tokens), std::move(scores),
        std::move(token_types), std::move(merge_rules),
        bos_id, eos_id, unk_id);
}

Result<std::unique_ptr<Tokenizer>> create_tokenizer_from_model(const Model& model) {
    const auto& md = model.metadata();
    const auto& meta = md.custom;

    // Phase 2: Use tokenizer arrays directly from ModelMetadata (populated by GGUF loader)
    // This replaces the old indexed-key workaround that required keys like
    // "tokenizer.ggml.tokens.0", "tokenizer.ggml.tokens.1", etc.
    if (!md.vocab.empty()) {
        std::vector<float> scores;
        if (!md.vocab_scores.empty() && md.vocab_scores.size() == md.vocab.size()) {
            scores = md.vocab_scores;
        } else {
            scores.assign(md.vocab.size(), 0.0f);
        }

        std::vector<int32_t> types;
        if (!md.token_types.empty() && md.token_types.size() == md.vocab.size()) {
            types = md.token_types;
        } else {
            types.assign(md.vocab.size(), static_cast<int32_t>(SPTokenType::NORMAL));
        }

        // Extract BOS/EOS from metadata
        int32_t bos_id = -1, eos_id = -1, unk_id = -1;
        auto bos_it = meta.find("tokenizer.ggml.bos_token_id");
        if (bos_it != meta.end() && std::holds_alternative<int64_t>(bos_it->second))
            bos_id = static_cast<int32_t>(std::get<int64_t>(bos_it->second));
        auto eos_it = meta.find("tokenizer.ggml.eos_token_id");
        if (eos_it != meta.end() && std::holds_alternative<int64_t>(eos_it->second))
            eos_id = static_cast<int32_t>(std::get<int64_t>(eos_it->second));

        // Determine model type for BPE vs unigram
        std::string model_type;
        auto arch_it2 = meta.find("general.architecture");
        if (arch_it2 != meta.end() && std::holds_alternative<std::string>(arch_it2->second))
            model_type = std::get<std::string>(arch_it2->second);
        std::string tokenizer_mode = (model_type == "bert" || model_type == "t5")
            ? "unigram" : "bpe";

        auto tok = create_sp_tokenizer(
            std::vector<std::string>(md.vocab),
            scores,
            types,
            std::vector<std::string>(md.merges),
            tokenizer_mode,
            bos_id, eos_id, unk_id);
        return tok;
    }

    // Legacy fallback: try indexed-key extraction from meta_custom_
    // (used when GGUF array data wasn't stored in ModelMetadata, e.g. older code paths)
    auto td = extract_tokenizer_from_metadata(meta);
    if (td && !td->tokens.empty()) {
        auto tok = create_sp_tokenizer(
            std::move(td->tokens),
            std::move(td->scores),
            std::move(td->token_types),
            std::move(td->merges),
            td->model_type,
            td->bos_id, td->eos_id, -1);
        return tok;
    }

    // Fallback: create minimal BPE tokenizer
    if (td) {
        return std::make_unique<BPE_Tokenizer>(
            td->bos_id >= 0 ? td->bos_id : 1,
            td->eos_id >= 0 ? td->eos_id : 2,
            std::vector<std::pair<std::string, int32_t>>{},
            std::vector<std::pair<std::string, std::string>>{}
        );
    }

    return std::make_unique<BPE_Tokenizer>(
        1, 2,
        std::vector<std::pair<std::string, int32_t>>{},
        std::vector<std::pair<std::string, std::string>>{}
    );
}

} // namespace hesa

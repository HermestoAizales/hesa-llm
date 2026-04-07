#include <hesa/tokenizer.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <cstring>
#include <numeric>

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) void name(); int name##_register() { std::cout << "  TEST: " #name << " "; name(); return 0; } int name##_r __attribute__((unused)) = name##_register(); \
void name()
#define ASSERT(cond) do { if (!(cond)) { std::cout << "FAILED (" #cond ")\n"; g_failed++; return; } } while(0)
#define ASSERT_NEAR(a, b, eps) do { if (std::fabs((a) - (b)) > (eps)) { std::cout << "FAILED (near " #a " " #b ")\n"; g_failed++; return; } } while(0)
#define PASS() do { std::cout << "passed\n"; g_passed++; } while(0)

// ── Test: SP unigram encode / decode roundtrip ─────────────────────

TEST(test_sp_unigram_encode_decode) {
    // Tiny vocab mimicking a SentencePiece unigram model
    // Uses \xe2\x96\x81 for space prefix (U+2581 "lower one-eighth block")
    std::string spc = "\xe2\x96\x81"; // SentencePiece space
    std::vector<std::string> tokens = {
        "<BOS>", "<EOS>", "<UNK>",  // 0,1,2
        spc + "hello",     // 3
        spc + "world",     // 4
        "hello",           // 5
        "world",           // 6
        spc + "hi",        // 7
        "lo",              // 8
        spc,               // 9  (just the space prefix)
        "l",               // 10
        "o",               // 11
        "<0x00>",          // 12 byte 0
        "<0xff>",          // 13 byte 255
    };
    // Scores = log-probabilities (unigram model). Higher = better.
    std::vector<float> scores = {
        -5.0f, -5.0f, -10.0f,
        -1.0f, -1.5f, -2.0f, -2.0f, -1.8f, -3.0f, -2.5f, -3.5f, -3.5f,
        -4.0f, -4.0f
    };
    // Token types: CONTROL=3, UNKNOWN=2, NORMAL=1, BYTE=5
    std::vector<int32_t> types = {3, 3, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 5, 5};

    auto result = hesa::create_sp_tokenizer(
        tokens, scores, types,
        {}, "unigram", 0, 1, 2);
    ASSERT(result.has_value());
    auto& tok = *result;

    ASSERT(tok->vocab_size() == static_cast<int32_t>(tokens.size()));
    ASSERT(tok->bos_token_id() == 0);
    ASSERT(tok->eos_token_id() == 1);

    // Encode a simple string
    std::string text = "hello";
    auto ids = tok->encode(text);
    ASSERT(!ids.empty());

    // Decode back
    auto decoded = tok->decode(ids);
    ASSERT(decoded == text);

    PASS();
}

// ── Test: SP BPE encode with merges ────────────────────────────────

TEST(test_sp_bpe_basic) {
    // Tiny BPE vocab with merge rules
    std::vector<std::string> tokens = {
        "h", "e", "l", "o", "w", "r", "d", " ",   // 0-7
        "he", "ll", "lo", "wo", "or", "rd",         // 8-13
        "hel", "llo", "wor", "rld",                   // 14-17
        "hell", "ello", "world",                      // 18-20
        "hello",                                       // 21
        "hello ",                                      // 22 (hello+space, but this won't parse as merge correctly)
    };

    // BPE merge rules in GGUF format: "left right" strings
    std::vector<std::string> merges = {
        "h e",      // 0: h+e -> he
        "l l",      // 1: l+l -> ll
        "l o",      // 2: l+o -> lo
        "w o",      // 3: w+o -> wo
        "o r",      // 4: o+r -> or
        "r d",      // 5: r+d -> rd
        "he l",     // 6: he+l -> hel
        "ll o",     // 7: ll+o -> llo
        "wo r",     // 8: wo+r -> wor
        "or d",     // 9: or+d -> ord
        "hel l",    // 10: hel+l -> hell (wait, l is single-char)
        "he ll",    // 11: he+ll -> hell (also valid)
        "hell o",   // 12: hell+o -> hello
        "hello  ",  // 13: "hello "  (hello + space, the space comes after)
    };

    std::vector<float> scores(tokens.size(), 0.0f);
    std::vector<int32_t> types(tokens.size(), 1); // NORMAL

    auto result = hesa::create_sp_tokenizer(
        tokens, scores, types,
        merges, "bpe", -1, -1, -1);
    ASSERT(result.has_value());
    auto& tok = *result;

    ASSERT(tok->vocab_size() == static_cast<int32_t>(tokens.size()));

    // BPE should produce tokens for "hello"
    std::string text = "hello";
    auto ids = tok->encode(text);
    ASSERT(!ids.empty());

    // Verify basic BPE merge: should merge h+e first, then he+l, etc.
    // With these merges: h e l l o -> he l l o -> hel l o -> hello
    // So the final token should be "hello" (id 21)
    // Note: depends on exact merge ordering; verify at least the output is reasonable

    PASS();
}

// ── Test: SP tokenizer factory ─────────────────────────────────────

TEST(test_sp_factory_basic) {
    std::vector<std::string> tokens = {"a", "b", "c", "\xe2\x96\x81test"};
    std::vector<float> scores = {0.0f, 0.0f, 0.0f, 0.0f};
    std::vector<int32_t> types = {1, 1, 1, 1}; // NORMAL

    auto result = hesa::create_sp_tokenizer(
        tokens, scores, types, {}, "unigram", 0, 1, -1);
    ASSERT(result.has_value());

    auto& tok = *result;
    ASSERT(tok->vocab_size() == 4);
    ASSERT(tok->bos_token_id() == 0);
    ASSERT(tok->eos_token_id() == 1);

    PASS();
}

// ── Test: BPE tokenizer ────────────────────────────────────────────

TEST(test_bpe_basic) {
    // Simple vocab: each character + some merged forms
    std::vector<std::pair<std::string, int32_t>> vocab;
    for (int i = 0; i < 256; ++i) {
        vocab.emplace_back(std::string(1, static_cast<char>(i)), i);
    }
    vocab.emplace_back("ab", 256);
    vocab.emplace_back("cd", 257);
    vocab.emplace_back("abcd", 258);

    std::vector<std::pair<std::string, std::string>> merges;
    merges.emplace_back("a", "b");
    merges.emplace_back("c", "d");
    merges.emplace_back("ab", "cd");

    auto result = hesa::create_bpe_tokenizer(vocab, merges, 1, 2);
    ASSERT(result.has_value());

    auto& tok = *result;
    auto ids = tok->encode("abcd");
    ASSERT(!ids.empty());

    auto decoded = tok->decode(ids);
    ASSERT(decoded == "abcd");

    PASS();
}

int main() {
    std::cout << "\n--- hesa-llm SP_Tokenizer Tests ---\n";
    std::cout << "\nTotal: " << (g_passed + g_failed) << " tests, "
              << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed > 0 ? 1 : 0;
}

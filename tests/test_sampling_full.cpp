// test_sampling_full.cpp — Comprehensive sampling tests
#include <hesa/sampling.hpp>
#include <cmath>
#include <cstring>
#include <iostream>
#include <cassert>
#include <set>

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) void name(); int name##_register() { std::cout << "  TEST: " #name << " "; name(); return 0; } int name##_r __attribute__((unused)) = name##_register(); \
void name()
#define ASSERT(cond) do { if (!(cond)) { std::cout << "FAILED (" #cond ")\n"; g_failed++; return; } } while(0)
#define ASSERT_NEAR(a, b, eps) do { if (std::fabs((a) - (b)) > (eps)) { std::cout << "FAILED (near " #a " " #b ")\n"; g_failed++; return; } } while(0)
#define PASS() do { std::cout << "passed\n"; g_passed++; } while(0)

TEST(test_greedy) {
    float logits[] = {0.1f, 5.0f, -1.0f, 3.0f, 2.0f};
    int32_t tok = hesa::sample_greedy(logits);
    ASSERT(tok == 1);
    PASS();
}

TEST(test_greedy_single) {
    float logits[] = {42.0f};
    int32_t tok = hesa::sample_greedy(logits);
    ASSERT(tok == 0);
    PASS();
}

TEST(test_temperature_valid) {
    float logits[] = {1.0f, 2.0f, 3.0f, 4.0f};
    for (float temp : {0.1f, 0.5f, 1.0f, 2.0f}) {
        int32_t tok = hesa::sample_temperature(logits, temp);
        ASSERT(tok >= 0 && tok < 4);
    }
    PASS();
}

TEST(test_temperature_low_temp) {
    float logits[] = {-10.0f, -10.0f, 100.0f, -10.0f};
    int32_t tok = hesa::sample_temperature(logits, 0.001f);
    ASSERT(tok == 2);
    PASS();
}

TEST(test_top_k_filtering) {
    float logits[] = {1.0f, 2.0f, 100.0f, 3.0f};
    for (int seed = 0; seed < 20; ++seed) {
        int32_t tok = hesa::sample_top_k(logits, 1, 1.0f);
        ASSERT(tok == 2);
    }
    PASS();
}

TEST(test_top_k_distribution) {
    float logits[] = {0.0f, 0.0f, -1e9f, 0.0f};

    std::set<int32_t> seen;
    for (int seed = 0; seed < 100; ++seed) {
        int32_t tok = hesa::sample_top_k(logits, 2, 1.0f);
        ASSERT(tok >= 0 && tok < 4);
        seen.insert(tok);
    }
    // With logits {0,0,-1e9,0} and softmax, tokens 0,1,3 have equal prob.
    // k=2 threshold is 0.0f (same for all 3), so we expect 0,1,3 not 2.
    ASSERT(seen.count(2) == 0);  // -1e9 token never appears
    PASS();
}

TEST(test_top_p_filtering) {
    float logits[] = {10.0f, 1.0f, 1.0f, 1.0f, 1.0f};

    std::set<int32_t> seen;
    for (int seed = 0; seed < 50; ++seed) {
        int32_t tok = hesa::sample_top_p(logits, 0.5f, 1.0f);
        seen.insert(tok);
    }
    ASSERT(seen.size() == 1);
    ASSERT(*seen.begin() == 0);
    PASS();
}

TEST(test_top_p_wide) {
    float logits[] = {1.0f, 1.0f, 1.0f};

    std::set<int32_t> seen;
    for (int seed = 0; seed < 50; ++seed) {
        int32_t tok = hesa::sample_top_p(logits, 1.0f, 1.0f);
        ASSERT(tok >= 0 && tok < 3);
        seen.insert(tok);
    }
    PASS();
}

TEST(test_repetition_penalty) {
    float logits[] = {5.0f, 5.0f, 5.0f};
    int32_t last_tokens[] = {0};

    int32_t tok_no = hesa::sample_greedy(logits);
    ASSERT(tok_no == 0);

    float logits2[] = {5.0f, 5.0f, 5.0f};
    int32_t tok_pen = hesa::sample_repetition_penalty(
        {logits2, 3}, {last_tokens, 1}, 2.0f);

    ASSERT(tok_pen != 0);
    PASS();
}

TEST(test_repetition_penalty_identity) {
    float logits[] = {1.0f, 5.0f, 3.0f};
    int32_t last_tokens[] = {1};

    float logits_copy[] = {1.0f, 5.0f, 3.0f};
    int32_t tok = hesa::sample_repetition_penalty(
        {logits_copy, 3}, {last_tokens, 1}, 1.0f);

    ASSERT(tok == 1);
    PASS();
}

TEST(test_sample_token_combined) {
    float logits[] = {0.0f, 10.0f, 5.0f, 3.0f, 1.0f};

    hesa::GenerationConfig cfg;
    cfg.temperature = 0.5f;
    cfg.top_k = 3;
    cfg.top_p = 0.9f;
    cfg.repetition_penalty = 1.5f;

    int32_t prev[] = {1};
    int32_t tok = hesa::sample_token(logits, 5, prev, cfg, 42);
    ASSERT(tok >= 0 && tok < 5);

    PASS();
}

TEST(test_sample_token_low_temp) {
    float logits[] = {-10.0f, 100.0f, 50.0f, -5.0f, 0.0f};
    hesa::GenerationConfig cfg;
    cfg.temperature = 0.001f;
    cfg.top_k = 0;
    cfg.top_p = 1.0f;
    cfg.repetition_penalty = 1.0f;

    int32_t tok = hesa::sample_token(logits, 5, {}, cfg, 42);
    ASSERT(tok == 1);
    PASS();
}

int main() {
    std::cout << "\n--- hesa-llm Sampling Tests (Full) ---\n";
    std::cout << "\nTotal: " << (g_passed + g_failed) << " tests, "
              << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed > 0 ? 1 : 0;
}

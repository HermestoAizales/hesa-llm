// test_rope.cpp — Test RoPE correctness with known values
#include <hesa/rope.hpp>
#include <hesa/tensor.hpp>
#include <cmath>
#include <cstring>
#include <iostream>
#include <cassert>

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) void name(); int name##_register() { std::cout << "  TEST: " #name << " "; name(); return 0; } int name##_r __attribute__((unused)) = name##_register(); \
void name()
#define ASSERT(cond) do { if (!(cond)) { std::cout << "FAILED (" #cond ")\n"; g_failed++; return; } } while(0)
#define ASSERT_NEAR(a, b, eps) do { if (std::fabs((a) - (b)) > (eps)) { std::cout << "FAILED (near " #a " " #b ")\n"; g_failed++; return; } } while(0)
#define PASS() do { std::cout << "passed\n"; g_passed++; } while(0)

TEST(test_rope_pos0_identity) {
    float data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    int32_t positions[] = {0};

    hesa::TensorView tv(data, hesa::Dtype::F32, hesa::Shape{1, 4});
    hesa::rope_apply(tv, positions, 10000.0f, 4);

    ASSERT_NEAR(data[0], 1.0f, 1e-6f);
    ASSERT_NEAR(data[1], 2.0f, 1e-6f);
    ASSERT_NEAR(data[2], 3.0f, 1e-6f);
    ASSERT_NEAR(data[3], 4.0f, 1e-6f);
    PASS();
}

TEST(test_rope_pos1) {
    float data[] = {1.0f, 0.0f};
    int32_t positions[] = {1};

    hesa::TensorView tv(data, hesa::Dtype::F32, hesa::Shape{1, 2});
    hesa::rope_apply(tv, positions, 10000.0f, 2);

    ASSERT_NEAR(data[0], std::cos(1.0f), 1e-4f);
    ASSERT_NEAR(data[1], std::sin(1.0f), 1e-4f);
    PASS();
}

TEST(test_rope_two_pairs) {
    float data[] = {1.0f, 0.0f, 0.0f, 1.0f};
    int32_t positions[] = {1};

    hesa::TensorView tv(data, hesa::Dtype::F32, hesa::Shape{1, 4});
    hesa::rope_apply(tv, positions, 10000.0f, 4);

    float cos1_0 = std::cos(1.0f);
    float sin1_0 = std::sin(1.0f);
    float cos1_1 = std::cos(0.01f);
    float sin1_1 = std::sin(0.01f);

    ASSERT_NEAR(data[0], 1.0f * cos1_0 - 0.0f * sin1_0, 1e-4f);
    ASSERT_NEAR(data[1], 1.0f * sin1_0 + 0.0f * cos1_0, 1e-4f);
    ASSERT_NEAR(data[2], 0.0f * cos1_1 - 1.0f * sin1_1, 1e-4f);
    ASSERT_NEAR(data[3], 0.0f * sin1_1 + 1.0f * cos1_1, 1e-4f);
    PASS();
}

TEST(test_rope_multi_position) {
    float data[6] = {1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f};
    int32_t positions[] = {0, 1, 10};

    hesa::TensorView tv(data, hesa::Dtype::F32, hesa::Shape{3, 2});
    hesa::rope_apply(tv, positions, 10000.0f, 2);

    ASSERT_NEAR(data[0], 1.0f, 1e-5f);
    ASSERT_NEAR(data[1], 0.0f, 1e-5f);

    ASSERT_NEAR(data[2], -std::sin(1.0f), 1e-4f);
    ASSERT_NEAR(data[3], std::cos(1.0f), 1e-4f);

    float c = std::cos(10.0f), s = std::sin(10.0f);
    ASSERT_NEAR(data[4], c - s, 1e-4f);
    ASSERT_NEAR(data[5], s + c, 1e-4f);
    PASS();
}

TEST(test_rope_cache_precompute) {
    hesa::RoPECache cache(5, 4, 10000.0f);

    ASSERT(cache.max_pos() == 5);
    ASSERT(cache.n_dims() == 4);

    float data[] = {1.0f, 0.0f, 0.0f, 1.0f};
    cache.apply(data, 1, 4);

    float cos1 = cache.cos_table()[1 * 4 + 0];
    float sin1 = cache.sin_table()[1 * 4 + 0];

    ASSERT_NEAR(data[0], cos1, 1e-5f);
    ASSERT_NEAR(data[1], sin1, 1e-5f);
    PASS();
}

TEST(test_rope_rotation_preserves_norm) {
    float data[] = {3.0f, 4.0f, 5.0f, 12.0f};
    float norm_before = std::sqrt(data[0]*data[0] + data[1]*data[1] +
                                  data[2]*data[2] + data[3]*data[3]);

    int32_t positions[] = {42};
    hesa::TensorView tv(data, hesa::Dtype::F32, hesa::Shape{1, 4});
    hesa::rope_apply(tv, positions, 10000.0f, 4);

    float norm_after = std::sqrt(data[0]*data[0] + data[1]*data[1] +
                                 data[2]*data[2] + data[3]*data[3]);

    ASSERT_NEAR(norm_before, norm_after, 1e-4f);
    PASS();
}

int main() {
    std::cout << "\n--- hesa-llm RoPE Tests ---\n";
    std::cout << "\nTotal: " << (g_passed + g_failed) << " tests, "
              << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed > 0 ? 1 : 0;
}

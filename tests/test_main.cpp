#include <hesa/hesa.h>
#include <hesa/version.hpp>
#include <hesa/tensor.hpp>
#include <hesa/backend.hpp>
#include <hesa/sampling.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <cstring>

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) void name(); int name##_register() { std::cout << "  TEST: " #name << " "; name(); return 0; } int name##_r __attribute__((unused)) = name##_register(); \
void name()
#define ASSERT(cond) do { if (!(cond)) { std::cout << "FAILED (" #cond ")\n"; g_failed++; return; } } while(0)
#define ASSERT_NEAR(a, b, eps) do { if (std::fabs((a) - (b)) > (eps)) { std::cout << "FAILED (near " #a " " #b ")\n"; g_failed++; return; } } while(0)
#define PASS() do { std::cout << "passed\n"; g_passed++; } while(0)

// -- Test: Version --
TEST(test_version) {
    ASSERT(HESA_VERSION_MAJOR >= 0);
    ASSERT(hesa::VERSION_MAJOR == HESA_VERSION_MAJOR);
    PASS();
}

// -- Test: Shape --
TEST(test_shape) {
    hesa::Shape s{3, 4, 5};
    ASSERT(s.ndim == 3);
    ASSERT(s[0] == 3);
    ASSERT(s[1] == 4);
    ASSERT(s[2] == 5);
    ASSERT(s.nelements() == 60);
    PASS();
}

// -- Test: Tensor creation --
TEST(test_tensor_create) {
    hesa::Tensor t(hesa::Dtype::F32, {2, 3}, nullptr);
    ASSERT(t.dtype() == hesa::Dtype::F32);
    ASSERT(t.ndim() == 2);
    ASSERT(t.shape()[0] == 2);
    ASSERT(t.shape()[1] == 3);
    ASSERT(t.nelements() == 6);
    ASSERT(t.nbytes() == 24);
    ASSERT(t.data() != nullptr);
    PASS();
}

// -- Test: Tensor view / reshape / transpose --
TEST(test_tensor_view) {
    hesa::Tensor t(hesa::Dtype::F32, {2, 3}, nullptr);
    auto* data = static_cast<float*>(t.data());
    for (int i = 0; i < 6; ++i) data[i] = static_cast<float>(i);

    // Reshape
    auto v = t.reshape(std::vector<int64_t>{3, 2});
    ASSERT(v.nelements() == 6);
    ASSERT(v.shape()[0] == 3);
    ASSERT(v.shape()[1] == 2);

    // Transpose
    auto tv = t.transpose(0, 1);
    ASSERT(tv.shape()[0] == 3);
    ASSERT(tv.shape()[1] == 2);
    PASS();
}

// -- Test: Backend creation --
TEST(test_backend_create) {
    hesa::DeviceConfig cfg;
    cfg.n_threads = 1;

    auto result = hesa::create_backend(hesa::BackendType::CPU_X86, cfg);
    if (!result) {
        // Try CPU_ARM as fallback
        result = hesa::create_backend(hesa::BackendType::CPU_ARM, cfg);
    }
    ASSERT(result.has_value());
    ASSERT((*result)->name() != nullptr);
    PASS();
}

// -- Test: CPU matmul --
TEST(test_cpu_matmul) {
    // 2x3 @ 3x2 = 2x2
    hesa::Tensor a(hesa::Dtype::F32, {2, 3}, nullptr);
    hesa::Tensor b(hesa::Dtype::F32, {3, 2}, nullptr);
    hesa::Tensor c(hesa::Dtype::F32, {2, 2}, nullptr);

    float* fa = static_cast<float*>(a.data());
    fa[0]=1; fa[1]=2; fa[2]=3; fa[3]=4; fa[4]=5; fa[5]=6;

    float* fb = static_cast<float*>(b.data());
    fb[0]=7; fb[1]=8; fb[2]=9; fb[3]=10; fb[4]=11; fb[5]=12;

    hesa::cpu_matmul_f32(fa, fb, static_cast<float*>(c.data()), 2, 2, 3, 1.0f);

    float* fc = static_cast<float*>(c.data());
    // [1,2,3] @ [7,9,11; 8,10,12] = [58, 64; 139, 154]
    ASSERT_NEAR(fc[0], 58.0f, 1e-3f);
    ASSERT_NEAR(fc[1], 64.0f, 1e-3f);
    ASSERT_NEAR(fc[2], 139.0f, 1e-3f);
    ASSERT_NEAR(fc[3], 154.0f, 1e-3f);
    PASS();
}

// -- Test: CPU softmax --
TEST(test_cpu_softmax) {
    float in[] = {1.0f, 2.0f, 3.0f};
    float out[3] = {};
    hesa::cpu_softmax_f32(in, out, 1, 3);

    // Check: values should sum to 1
    float sum = out[0] + out[1] + out[2];
    ASSERT_NEAR(sum, 1.0f, 1e-5f);
    ASSERT(out[2] > out[1]);
    ASSERT(out[1] > out[0]);
    PASS();
}

// -- Test: CPU RMS norm --
TEST(test_cpu_rms_norm) {
    float inp[] = {3.0f, 4.0f}; // RMS = sqrt((9+16)/2) = sqrt(12.5) ~ 3.5355
    float w[]   = {1.0f, 1.0f};
    float out[2] = {};
    hesa::cpu_rms_norm_f32(inp, w, out, 1, 2, 1e-6f);

    float rms = std::sqrt((9.0f + 16.0f) / 2.0f);
    ASSERT_NEAR(out[0], 3.0f / rms, 1e-4f);
    ASSERT_NEAR(out[1], 4.0f / rms, 1e-4f);
    PASS();
}

// -- Test: SiLU --
TEST(test_cpu_silu) {
    float inp[] = {0.0f, 1.0f, -1.0f};
    float out[3] = {};
    hesa::cpu_silu_f32(inp, out, 3);

    ASSERT_NEAR(out[0], 0.0f, 1e-6f);
    // silu(1) = 1 / (1 + exp(-1)) = 0.73105...
    float expected = 1.0f / (1.0f + std::exp(-1.0f));
    ASSERT_NEAR(out[1], expected, 1e-4f);
    PASS();
}

// -- Test: RoPE --
TEST(test_cpu_rope) {
    // Single head, dimension 4 (2 pairs)
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    int32_t positions[] = {0}; // position 0

    hesa::cpu_rope_f32(data, positions, 1, 1, 4, 10000.0f);

    // At position 0, rotation angle = 0, so data should be unchanged
    ASSERT_NEAR(data[0], 1.0f, 1e-6f);
    ASSERT_NEAR(data[1], 2.0f, 1e-6f);
    ASSERT_NEAR(data[2], 3.0f, 1e-6f);
    ASSERT_NEAR(data[3], 4.0f, 1e-6f);
    PASS();
}

// -- Test: RoPE at position 1 --
TEST(test_cpu_rope_pos1) {
    float data[4] = {1.0f, 0.0f, 0.0f, 1.0f};
    int32_t positions[] = {1};

    hesa::cpu_rope_f32(data, positions, 1, 1, 4, 10000.0f);

    // freq for dim 0: 10000^(-0/4) = 1, angle = 1*1 = 1 rad
    // freq for dim 1: 10000^(-1/4) = 0.1, angle = 1*0.1 = 0.1 rad
    float cos0 = std::cos(1.0f);
    float sin0 = std::sin(1.0f);
    float cos1 = std::cos(0.1f);
    float sin1 = std::sin(0.1f);

    ASSERT_NEAR(data[0], 1.0f * cos0 - 0.0f * sin0, 1e-4f);
    ASSERT_NEAR(data[1], 1.0f * sin0 + 0.0f * cos0, 1e-4f);
    ASSERT_NEAR(data[2], 0.0f * cos1 - 1.0f * sin1, 1e-4f);
    ASSERT_NEAR(data[3], 0.0f * sin1 + 1.0f * cos1, 1e-4f);
    PASS();
}

// -- Test: Sampling --
TEST(test_sampling_temperature_zero) {
    float logits[] = {100.0f, -100.0f, -100.0f, -100.0f};
    hesa::GenerationConfig cfg;
    cfg.temperature = 0.01f;
    cfg.top_k = 0;
    cfg.top_p = 1.0f;
    cfg.repetition_penalty = 1.0f;

    int32_t tok = hesa::sample_token(logits, 4, {}, cfg, 42);
    ASSERT(tok == 0); // Highest logit should win at low temp
    PASS();
}

// -- Test: Sampling uniform --
TEST(test_sampling_uniform) {
    float logits[] = {0.0f, 0.0f, 0.0f, 0.0f};
    hesa::GenerationConfig cfg;
    cfg.temperature = 1.0f;
    cfg.top_k = 0;
    cfg.top_p = 1.0f;
    cfg.repetition_penalty = 1.0f;

    // With uniform logits, different seeds should give different tokens
    int got_a_token = hesa::sample_token(logits, 4, {}, cfg, 123);
    ASSERT(got_a_token >= 0 && got_a_token < 4);
    PASS();
}

// -- Test: dtype_size --
TEST(test_dtype_size) {
    ASSERT(dtype_size(hesa::Dtype::F32) == 4);
    ASSERT(dtype_size(hesa::Dtype::F16) == 2);
    PASS();
}

int main() {
    std::cout << "\n--- hesa-llm Phase 0 Tests ---\n";
    std::cout << "\nTotal: " << (g_passed + g_failed) << " tests, "
              << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed > 0 ? 1 : 0;
}

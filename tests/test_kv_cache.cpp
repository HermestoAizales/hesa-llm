// test_kv_cache.cpp — Test ring buffer KV cache wrap-around and eviction
#include <hesa/kv_cache.hpp>
#include <cstring>
#include <cmath>
#include <iostream>
#include <cassert>

static int g_passed = 0;
static int g_failed = 0;

#define TEST(name) void name(); int name##_register() { std::cout << "  TEST: " #name << " "; name(); return 0; } int name##_r __attribute__((unused)) = name##_register(); \
void name()
#define ASSERT(cond) do { if (!(cond)) { std::cout << "FAILED (" #cond ")\n"; g_failed++; return; } } while(0)
#define ASSERT_NEAR(a, b, eps) do { if (std::fabs((a) - (b)) > (eps)) { std::cout << "FAILED (near " #a " " #b ")\n"; g_failed++; return; } } while(0)
#define PASS() do { std::cout << "passed\n"; g_passed++; } while(0)

TEST(test_kv_basic_write_read) {
    hesa::KVCacheConfig cfg;
    cfg.max_seq_len = 8;
    cfg.n_layers = 1;
    cfg.n_heads = 2;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 4;

    hesa::KVCache cache(cfg);
    ASSERT(cache.capacity() == 8);
    ASSERT(cache.used() == 0);

    float k_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v_data[] = {5.0f, 6.0f, 7.0f, 8.0f};

    cache.write(0, 0, 0, k_data, v_data);
    cache.advance(1);
    ASSERT(cache.used() == 1);

    size_t phys = cache.physical_pos(0);
    float* k_ptr = static_cast<float*>(cache.key_tensor(0, 0).data());
    float* v_ptr = static_cast<float*>(cache.value_tensor(0, 0).data());

    for (int i = 0; i < 4; ++i) {
        ASSERT_NEAR(k_ptr[phys * 4 + i], k_data[i], 1e-6f);
        ASSERT_NEAR(v_ptr[phys * 4 + i], v_data[i], 1e-6f);
    }
    PASS();
}

TEST(test_kv_ring_wrap) {
    hesa::KVCacheConfig cfg;
    cfg.max_seq_len = 4;
    cfg.n_layers = 1;
    cfg.n_kv_heads = 1;
    cfg.head_dim = 4;

    hesa::KVCache cache(cfg);

    for (size_t i = 0; i < 4; ++i) {
        float k[4] = {static_cast<float>(i), static_cast<float>(i+1),
                       static_cast<float>(i+2), static_cast<float>(i+3)};
        float v[4] = {-static_cast<float>(i), -static_cast<float>(i+1),
                       -static_cast<float>(i+2), -static_cast<float>(i+3)};
        cache.write(0, 0, i, k, v);
    }
    cache.advance(4);
    ASSERT(cache.used() == 4);

    for (size_t i = 0; i < 4; ++i) {
        size_t phys = cache.physical_pos(i);
        ASSERT(phys == i);
    }

    float k4[4] = {100.0f, 101.0f, 102.0f, 103.0f};
    float v4[4] = {-100.0f, -101.0f, -102.0f, -103.0f};
    cache.write(0, 0, 4, k4, v4);
    cache.advance(4);

    float* k_ptr = static_cast<float*>(cache.key_tensor(0, 0).data());
    size_t phys0 = cache.physical_pos(4);
    ASSERT_NEAR(k_ptr[phys0 * 4 + 0], 100.0f, 1e-6f);
    ASSERT_NEAR(k_ptr[phys0 * 4 + 3], 103.0f, 1e-6f);

    PASS();
}

TEST(test_kv_clear) {
    hesa::KVCacheConfig cfg;
    cfg.max_seq_len = 4;
    cfg.n_layers = 1;
    cfg.n_kv_heads = 1;
    cfg.head_dim = 4;

    hesa::KVCache cache(cfg);

    float k[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float v[4] = {5.0f, 6.0f, 7.0f, 8.0f};
    cache.write(0, 0, 0, k, v);
    cache.advance(4);

    ASSERT(cache.used() == 4);
    cache.clear();
    ASSERT(cache.used() == 0);

    float* k_ptr = static_cast<float*>(cache.key_tensor(0, 0).data());
    for (size_t i = 0; i < 16; ++i) {
        ASSERT_NEAR(k_ptr[i], 0.0f, 1e-6f);
    }
    PASS();
}

TEST(test_kv_sliding_window_eviction) {
    hesa::KVCacheConfig cfg;
    cfg.max_seq_len = 8;
    cfg.n_layers = 1;
    cfg.n_kv_heads = 1;
    cfg.head_dim = 4;

    hesa::KVCache cache(cfg);

    for (size_t i = 0; i < 6; ++i) {
        float k[4] = {static_cast<float>(i), 0, 0, 0};
        float v[4] = {0, 0, 0, 0};
        cache.write(0, 0, i, k, v);
    }
    cache.advance(6);
    ASSERT(cache.used() == 6);

    cache.evict_sliding_window(3);
    ASSERT(cache.used() == 3);

    float* k_ptr = static_cast<float*>(cache.key_tensor(0, 0).data());
    for (size_t i = 0; i < 3; ++i) {
        size_t phys = cache.physical_pos(i);
        ASSERT_NEAR(k_ptr[phys * 4 + 0], 0.0f, 1e-6f);
    }

    for (int v = 3; v <= 5; ++v) {
        size_t phys = cache.physical_pos(v);
        ASSERT_NEAR(k_ptr[phys * 4 + 0], static_cast<float>(v), 1e-6f);
    }

    PASS();
}

TEST(test_kv_ring_wrap_eviction) {
    hesa::KVCacheConfig cfg;
    cfg.max_seq_len = 4;
    cfg.n_layers = 1;
    cfg.n_kv_heads = 1;
    cfg.head_dim = 2;

    hesa::KVCache cache(cfg);

    for (size_t i = 0; i < 6; ++i) {
        float k[2] = {static_cast<float>(100 + i), static_cast<float>(200 + i)};
        float v[2] = {0, 0};
        cache.write(0, 0, i, k, v);
        cache.advance(1);
    }
    // seq_len_ = 4 (capped). Ring: physical[0]=pos4(104,204), [1]=pos5(105,205),
    // [2]=pos2(102,202), [3]=pos3(103,203)
    // oldest data wraps to physical_pos(4%4)=0

    cache.evict_sliding_window(2);
    // Remove oldest 2 items (pos4@phys0, pos5@phys1), keeping pos2@phys2, pos3@phys3
    // But we want to keep newest, so evict oldest (pos2, pos3 @ phys 2,3)
    // Wait - with ring buffer: after 6 writes with size=4, newest are at phys 0,1
    // oldest are at phys 2,3. Evict oldest 2 → clear phys 2,3. seq_len_=2.
    ASSERT(cache.used() == 2);

    float* k_ptr = static_cast<float*>(cache.key_tensor(0, 0).data());
    // After eviction, seq_len=2. physical_pos(0)=0, physical_pos(1)=1.
    // phys0 has pos4 data (104,204), phys1 has pos5 data (105,205)
    ASSERT_NEAR(k_ptr[0], 104.0f, 1e-6f);
    ASSERT_NEAR(k_ptr[1], 204.0f, 1e-6f);

    PASS();
}

TEST(test_kv_multi_layer_head) {
    hesa::KVCacheConfig cfg;
    cfg.max_seq_len = 4;
    cfg.n_layers = 3;
    cfg.n_heads = 4;
    cfg.n_kv_heads = 2;
    cfg.head_dim = 4;

    hesa::KVCache cache(cfg);

    for (size_t l = 0; l < cfg.n_layers; ++l) {
        for (size_t h = 0; h < cfg.n_kv_heads; ++h) {
            float k[4] = {1.0f, 2.0f, 3.0f, 4.0f};
            float v[4] = {5.0f, 6.0f, 7.0f, 8.0f};
            cache.write(l, h, 0, k, v);

            float* kp = static_cast<float*>(cache.key_tensor(l, h).data());
            float* vp = static_cast<float*>(cache.value_tensor(l, h).data());

            for (int i = 0; i < 4; ++i) {
                ASSERT_NEAR(kp[i], k[i], 1e-6f);
                ASSERT_NEAR(vp[i], v[i], 1e-6f);
            }
        }
    }

    PASS();
}

int main() {
    std::cout << "\n--- hesa-llm KV Cache Tests ---\n";
    std::cout << "\nTotal: " << (g_passed + g_failed) << " tests, "
              << g_passed << " passed, " << g_failed << " failed\n";
    return g_failed > 0 ? 1 : 0;
}

# tests/CMakeLists.txt

# Phase 0: Minimal test executable
add_executable(hesa_test
    test_main.cpp
    test_tensor.cpp
    test_backend.cpp
    test_sampling.cpp
)
target_link_libraries(hesa_test PRIVATE hesa_core)

add_test(NAME hesa_test_basic COMMAND hesa_test)

# Phase 1: KV Cache tests
add_executable(test_kv_cache
    test_kv_cache.cpp
)
target_link_libraries(test_kv_cache PRIVATE hesa_core)
add_test(NAME hesa_kv_cache COMMAND test_kv_cache)

# Phase 1: RoPE tests
add_executable(test_rope
    test_rope.cpp
)
target_link_libraries(test_rope PRIVATE hesa_core)
add_test(NAME hesa_rope COMMAND test_rope)

# Phase 1: Sampling tests (full suite)
add_executable(test_sampling_full
    test_sampling_full.cpp
)
target_link_libraries(test_sampling_full PRIVATE hesa_core)
add_test(NAME hesa_sampling_full COMMAND test_sampling_full)

# Tokenizer tests
add_executable(test_sp_tokenizer
    test_sp_tokenizer.cpp
)
target_link_libraries(test_sp_tokenizer PRIVATE hesa_core)
add_test(NAME hesa_sp_tokenizer COMMAND test_sp_tokenizer)

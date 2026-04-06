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

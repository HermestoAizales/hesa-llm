# tests/CMakeLists.txt

# Simple manual test placeholder
# Replace with Catch2, GoogleTest, or other framework via FetchContent

add_executable(hesa_test test_main.cpp)
target_link_libraries(hesa_test PRIVATE hesa_core)

add_test(NAME hesa_test_basic COMMAND hesa_test)

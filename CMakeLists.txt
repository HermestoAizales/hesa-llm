cmake_minimum_required(VERSION 3.20)
project(hesa-llm
    VERSION     0.1.0
    DESCRIPTION "hesa-llm: A modern, portable LLM inference engine"
    LANGUAGES   CXX
)

# -- C++ Standard --
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -- Cross-platform defaults --
include(GNUInstallDirs)

if(MSVC)
    add_compile_options(/W4 /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# -- FetchContent (declare deps lazily) --
include(FetchContent)

# Example: uncomment and configure as needed
# FetchContent_Declare(
#     spdlog
#     GIT_REPOSITORY https://github.com/gabime/spdlog.git
#     GIT_TAG        v1.14.1
# )
# FetchContent_MakeAvailable(spdlog)

# -- Core library --
add_library(hesa_core
    src/hesa_core.cpp
)
target_include_directories(hesa_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# -- Main executable --
add_executable(hesa-llm src/main.cpp)
target_link_libraries(hesa-llm PRIVATE hesa_core)

# -- Testing --
option(HESA_ENABLE_TESTS "Build unit tests" ON)
if(HESA_ENABLE_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# -- Install rules --
install(TARGETS hesa-llm hesa_core
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

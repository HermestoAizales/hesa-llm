cmake_minimum_required(VERSION 3.20)
project(hesa-llm
    VERSION     0.1.0
    DESCRIPTION "hesa-llm: A modern, portable LLM inference engine"
    LANGUAGES   CXX
)

# -- C++20 (coroutines, concepts, std::span, std::expected) --
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# -- Cross-platform defaults --
include(GNUInstallDirs)
include(CheckCXXCompilerFlag)

if(MSVC)
    add_compile_options(/W4 /utf-8)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
    check_cxx_compiler_flag("-mavx2" COMPILER_SUPPORTS_AVX2)
    check_cxx_compiler_flag("-mavx512f" COMPILER_SUPPORTS_AVX512F)
endif()

# -- Platform detection --
set(HESA_PLATFORM "unknown")
if(APPLE)
    set(HESA_PLATFORM "macos")
elseif(WIN32)
    set(HESA_PLATFORM "windows")
elseif(UNIX)
    set(HESA_PLATFORM "linux")
endif()

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(HESA_ARCH "arm64")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    set(HESA_ARCH "x86_64")
else()
    set(HESA_ARCH "${CMAKE_SYSTEM_PROCESSOR}")
endif()

message(STATUS "HESA Platform: ${HESA_PLATFORM} Arch: ${HESA_ARCH}")

# -- Backend options --
option(HESA_ENABLE_CUDA  "Enable CUDA backend"     OFF)
option(HESA_ENABLE_METAL "Enable Metal backend"    OFF)
option(HESA_ENABLE_TESTS "Build unit tests"        ON)

# Auto-enable Metal on macOS
if(APPLE)
    list(APPEND CMAKE_PREFIX_PATH /Applications)
    check_cxx_compiler_flag("-fobjc-arc" HAS_OBJC_ARC)
    find_library(METAL_FRAMEWORK Metal)
    if(METAL_FRAMEWORK AND HAS_OBJC_ARC)
        set(HESA_ENABLE_METAL ON CACHE BOOL "Metal available on macOS" FORCE)
        message(STATUS "Metal available on macOS")
    endif()
endif()

# Auto-detect CUDA
if(NOT DEFINED HESA_ENABLE_CUDA OR NOT HESA_ENABLE_CUDA)
    include(CheckLanguage)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(HESA_ENABLE_CUDA ON CACHE BOOL "CUDA auto-detected" FORCE)
        message(STATUS "CUDA auto-detected")
    endif()
endif()

# -- FetchContent (lazy deps) --
include(FetchContent)

# nlohmann/json for GGUF metadata & config
FetchContent_Declare(
    nlohmann_json
    URL      https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
    URL_HASH SHA256=40f2e335995684229052c0c8e4b03bf1769fa52c7095976c7c4e55b3f3870e2b
)
FetchContent_MakeAvailable(nlohmann_json)

# -- Source files --
set(HESA_CORE_SRCS
    src/hesa_core.cpp
    src/tensor/tensor.cpp
    src/tensor/tensor_ops.cpp
    src/backend/backend.cpp
    src/backend/cpu_backend.cpp
    src/model/gguf_loader.cpp
    src/tokenizer/tokenizer.cpp
    src/sampling/sampler.cpp
)

if(HESA_ENABLE_CUDA)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    list(APPEND HESA_CORE_SRCS
        src/backend/cuda_backend.cpp
        src/backend/cuda_kernels.cu
    )
    message(STATUS "CUDA backend enabled")
endif()

if(HESA_ENABLE_METAL)
    enable_language(OBJCXX)
    list(APPEND HESA_CORE_SRCS src/backend/metal_backend.mm)
    message(STATUS "Metal backend enabled")
endif()

# -- Core library --
add_library(hesa_core ${HESA_CORE_SRCS})
set_target_properties(hesa_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(hesa_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(hesa_core PUBLIC nlohmann_json::nlohmann_json)

# Platform libraries
if(APPLE)
    target_link_libraries(hesa_core PRIVATE "-framework Foundation" "-framework Metal" "-framework MetalPerformanceShaders")
endif()

# SIMD flags
if(COMPILER_SUPPORTS_AVX512F)
    target_compile_options(hesa_core PRIVATE -mavx512f -mfma)
elseif(COMPILER_SUPPORTS_AVX2)
    target_compile_options(hesa_core PRIVATE -mavx2 -mfma)
endif()

target_compile_definitions(hesa_core PRIVATE
    HESA_PLATFORM_${HESA_PLATFORM}=1
    HESA_ARCH_${HESA_ARCH}=1
)
if(HESA_ENABLE_CUDA)
    target_compile_definitions(hesa_core PUBLIC HESA_HAS_CUDA=1)
endif()
if(HESA_ENABLE_METAL)
    target_compile_definitions(hesa_core PUBLIC HESA_HAS_METAL=1)
endif()

# -- Main executable --
add_executable(hesa-llm src/main.cpp)
target_link_libraries(hesa-llm PRIVATE hesa_core)

# -- Testing --
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

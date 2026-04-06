cmake_minimum_required(VERSION 3.20)
project(hesa-llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(BUILD_SHARED_LIBS "Build shared libraries" OFF)

# Cross-platform support
if(WIN32)
    add_compile_definitions(_WINDOWS)
elseif(APPLE)
    add_compile_definitions(__APPLE__)
elseif(UNIX)
    add_compile_definitions(__LINUX__)
endif()

# Executable
add_executable(hesa-llm src/main.cpp)
target_include_directories(hesa-llm PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

# Install targets
include(GNUInstallDirs)

install(TARGETS hesa-llm
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

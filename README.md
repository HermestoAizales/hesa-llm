# hesa-llm

**hesa-llm** is a modern, portable LLM inference engine aiming for Gemma-4 level performance in a smaller and faster form factor.

## Goals

- **Efficiency**: Deliver high-quality inference with minimal resource consumption
- **Portability**: Cross-platform support for Linux, macOS, and Windows
- **Simplicity**: Clean, maintainable C++17 codebase with modern CMake
- **Extensibility**: Modular architecture for adding new models and kernels

## Features

- C++17 codebase with zero external dependencies at build-time (FetchContent for optional deps)
- Cross-platform build system (CMake 3.20+)
- Modular library + CLI executable structure
- Unit testing infrastructure

## Building

```bash
# Clone
git clone https://github.com/HermestoAizales/hesa-llm.git
cd hesa-llm

# Configure
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(nproc)

# Run
./build/hesa-llm
```

## License

MIT

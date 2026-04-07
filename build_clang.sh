#!/bin/bash
# Build hesa-llm directly with clang++ (no cmake needed)

set -e
cd "$(dirname "$0")"
OUT_DIR="${1:-build}"
mkdir -p "$OUT_DIR"

CXX="clang++"
CXXFLAGS="-std=c++20 -O2 -Wall -DHESA_ARCH_arm64=1"
LDFLAGS=""

echo "=== Building hesa-llm ==="
$CXX $CXXFLAGS $LDFLAGS   -I include   src/tensor/tensor.cpp   src/tensor/tensor_ops.cpp   src/backend/backend.cpp   src/backend/cpu_backend.cpp   src/sampling/sampler.cpp   src/model/gguf_loader.cpp   src/tokenizer/tokenizer.cpp   src/engine.cpp   src/gguf_parser.cpp   src/memory/kv_cache.cpp   src/layers/rope.cpp   src/transformer_block.cpp   src/main.cpp   -o "$OUT_DIR/hesa-llm"

echo "=== Build complete: $OUT_DIR/hesa-llm ==="

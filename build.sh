#!/bin/bash
# Build hesa-llm directly with clang++ (no cmake needed for this host)

cd "$(dirname "$0")"
if [ -d build ]; then
  cd build
else
  mkdir -p build && cd build
fi

HESA_DIR=".."
BUILD_DIR=$(pwd)
CXX="clang++"
CXXFLAGS="-std=c++20 -O2 -DHESA_ARCH_arm64=1 -I${HESA_DIR}/include"
LDFLAGS=""

SRCS=(
  "${HESA_DIR}/src/tensor/tensor.cpp"
  "${HESA_DIR}/src/tensor/tensor_ops.cpp"
  "${HESA_DIR}/src/backend/backend.cpp"
  "${HESA_DIR}/src/backend/cpu_backend.cpp"
  "${HESA_DIR}/src/sampling/sampler.cpp"
  "${HESA_DIR}/src/model/gguf_loader.cpp"
  "${HESA_DIR}/src/tokenizer/tokenizer.cpp"
  "${HESA_DIR}/src/engine.cpp"
  "${HESA_DIR}/src/gguf_parser.cpp"
  "${HESA_DIR}/src/memory/kv_cache.cpp"
  "${HESA_DIR}/src/layers/rope.cpp"
  "${HESA_DIR}/src/transformer_block.cpp"
  "${HESA_DIR}/src/main.cpp"
)

echo "=== Building hesa-llm ==="
OBJS=""
for src in "${SRCS[@]}"; do
  obj="${BUILD_DIR}/$(basename ${src%.cpp}).o"
  echo "  CC ${src}"
  ${CXX} ${CXXFLAGS} -c ${src} -o ${obj} || exit 1
  OBJS="${OBJS} ${obj}"
done

echo "  LINK hesa-llm"
${CXX} ${OBJS} ${LDFLAGS} -o hesa-llm
echo "=== Build complete: ${BUILD_DIR}/hesa-llm ==="

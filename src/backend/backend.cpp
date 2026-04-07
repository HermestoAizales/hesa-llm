#include "hesa/backend.hpp"
#include "backend/cpu_backend.hpp"

#include <thread>

namespace hesa {

BackendType auto_detect_backend() {
#ifdef HESA_HAS_CUDA
    return BackendType::CUDA;
#elif defined(HESA_HAS_METAL)
    return BackendType::METAL;
#elif defined(HESA_ARCH_x86_64)
    return BackendType::CPU_X86;
#elif defined(HESA_ARCH_arm64)
    return BackendType::CPU_ARM;
#else
    return BackendType::CPU_X86;
#endif
}

Result<std::unique_ptr<Backend>> create_backend(BackendType type, DeviceConfig cfg) {
    if (type == BackendType::AUTO) {
        type = auto_detect_backend();
    }

    switch (type) {
        case BackendType::CPU_X86:
        case BackendType::CPU_ARM: {
            auto be = std::make_unique<class CPU_Backend>(cfg);
            return be;
        }
#ifdef HESA_HAS_CUDA
        case BackendType::CUDA: {
            auto be = std::make_unique<class CudaBackend>(cfg);
            return be;
        }
#endif
#ifdef HESA_HAS_METAL
        case BackendType::METAL: {
            auto be = std::make_unique<class MetalBackend>(cfg);
            return be;
        }
#endif
        default:
            return make_error<std::unique_ptr<Backend>>(Error::BACKEND_NOT_AVAILABLE);
    }
}

} // namespace hesa

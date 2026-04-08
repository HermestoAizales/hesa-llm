// Metal backend stub — not yet implemented, but provides symbols for build
// This file is compiled only on macOS when HESA_ENABLE_METAL=ON.
// All functions are no-ops; actual Metal kernels will be added later.

#include <cstdint>
#include <cstdlib>

namespace hesa::metal {

// Placeholder: compile without Metal, all CPU fallbacks in other backends
void init_metal_device() {}
void shutdown_metal_device() {}

// Dummy alloc — returns nullptr; hesa core will fall back to system alloc
void* metal_alloc(size_t bytes) {
    (void)bytes;
    return nullptr;
}

void metal_free(void* ptr) {
    (void)ptr;
}

// No-op kernels
void metal_rms_norm([[maybe_unused]] const float* x,
                    [[maybe_unused]] const float* weight,
                    [[maybe_unused]] float* out,
                    [[maybe_unused]] size_t n,
                    [[maybe_unused]] float eps) {}

void metal_rope([[maybe_unused]] float* q,
                [[maybe_unused]] const int32_t* pos,
                [[maybe_unused]] size_t n_pos,
                [[maybe_unused]] float freq_base,
                [[maybe_unused]] int dim_count) {}

} // namespace hesa::metal

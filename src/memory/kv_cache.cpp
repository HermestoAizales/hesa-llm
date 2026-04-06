#include "hesa/kv_cache.hpp"
#include <cstring>
#include <algorithm>
#include <cassert>

namespace hesa {

KVCache::KVCache(const KVCacheConfig& cfg)
    : cfg_(cfg),
      key_cache_(),
      value_cache_()
{
    const size_t n_kv_total = cfg_.n_layers * cfg_.n_kv_heads;
    key_cache_.reserve(n_kv_total);
    value_cache_.reserve(n_kv_total);

    // Each key/value tensor: [max_seq_len, head_dim] (contiguous F32)
    for (size_t l = 0; l < cfg_.n_layers; ++l) {
        for (size_t h = 0; h < cfg_.n_kv_heads; ++h) {
            key_cache_.emplace_back(
                Dtype::F32,
                Shape{static_cast<int64_t>(cfg_.max_seq_len),
                      static_cast<int64_t>(cfg_.head_dim)});
            value_cache_.emplace_back(
                Dtype::F32,
                Shape{static_cast<int64_t>(cfg_.max_seq_len),
                      static_cast<int64_t>(cfg_.head_dim)});
        }
    }
}

KVCache::~KVCache() = default;

TensorView KVCache::alloc(size_t layer, size_t head, size_t pos) {
    assert(layer < cfg_.n_layers);
    assert(head < cfg_.n_kv_heads);

    auto& kt = key_cache_[layer * cfg_.n_kv_heads + head];
    const size_t phys = physical_pos(pos);
    return kt.select(0, static_cast<int64_t>(phys));
}

void KVCache::write(size_t layer, size_t head, size_t pos,
                     const void* key_data, const void* value_data) {
    assert(layer < cfg_.n_layers);
    assert(head < cfg_.n_kv_heads);

    const size_t phys = physical_pos(pos);
    const size_t stride = cfg_.head_dim * sizeof(float);

    auto& kt = key_cache_[layer * cfg_.n_kv_heads + head];
    auto& vt = value_cache_[layer * cfg_.n_kv_heads + head];

    auto kv = kt.select(0, static_cast<int64_t>(phys));
    auto vv = vt.select(0, static_cast<int64_t>(phys));

    std::memcpy(kv.data(), key_data, stride);
    std::memcpy(vv.data(), value_data, stride);
}

std::pair<TensorView, TensorView>
KVCache::read(size_t layer, size_t head,
              size_t pos_begin, size_t pos_end) const {
    assert(layer < cfg_.n_layers);
    assert(head < cfg_.n_kv_heads);

    size_t n_positions = pos_end - pos_begin;
    if (n_positions == 0) {
        return {{}, {}};
    }

    auto& kt = key_cache_[layer * cfg_.n_kv_heads + head];
    auto& vt = value_cache_[layer * cfg_.n_kv_heads + head];

    // If contiguous and non-wrapping, return sub-views
    if (!wraps(pos_begin, n_positions)) {
        size_t phys_begin = physical_pos(pos_begin);
        if (n_positions == 1) {
            return {kt.select(0, static_cast<int64_t>(phys_begin)),
                    vt.select(0, static_cast<int64_t>(phys_begin))};
        }
    }

    // General / wrapped case: return full tensors (caller handles selection)
    return {kt.view(), vt.view()};
}

std::pair<TensorView, TensorView>
KVCache::read_positions(size_t layer, size_t head,
                         std::span<const size_t> positions) const {
    assert(layer < cfg_.n_layers);
    assert(head < cfg_.n_kv_heads);

    if (positions.empty()) {
        return {{}, {}};
    }

    auto& kt = key_cache_[layer * cfg_.n_kv_heads + head];
    auto& vt = value_cache_[layer * cfg_.n_kv_heads + head];

    // Check contiguity
    bool contiguous = true;
    for (size_t i = 1; i < positions.size(); ++i) {
        if (positions[i] != positions[i - 1] + 1) {
            contiguous = false;
            break;
        }
    }

    if (contiguous) {
        size_t pos_begin = positions.front();
        if (!wraps(pos_begin, positions.size())) {
            size_t phys_begin = physical_pos(pos_begin);
            return {kt.select(0, static_cast<int64_t>(phys_begin)),
                    vt.select(0, static_cast<int64_t>(phys_begin))};
        }
    }

    // Non-contiguous or wrapping
    return {kt.view(), vt.view()};
}

void KVCache::clear() {
    seq_len_ = 0;
    const size_t element_size = sizeof(float);
    const size_t head_bytes = cfg_.head_dim * element_size;

    for (size_t l = 0; l < cfg_.n_layers; ++l) {
        for (size_t h = 0; h < cfg_.n_kv_heads; ++h) {
            auto& kt = key_cache_[l * cfg_.n_kv_heads + h];
            auto& vt = value_cache_[l * cfg_.n_kv_heads + h];

            for (size_t i = 0; i < cfg_.max_seq_len; ++i) {
                auto kv = kt.select(0, static_cast<int64_t>(i));
                auto vv = vt.select(0, static_cast<int64_t>(i));
                std::memset(kv.data(), 0, head_bytes);
                std::memset(vv.data(), 0, head_bytes);
            }
        }
    }
}

void KVCache::evict_sliding_window(size_t window_size) {
    if (seq_len_ <= window_size || window_size == 0) {
        return;
    }

    size_t to_remove = seq_len_ - window_size;
    const size_t head_bytes = cfg_.head_dim * sizeof(float);

    for (size_t l = 0; l < cfg_.n_layers; ++l) {
        for (size_t h = 0; h < cfg_.n_kv_heads; ++h) {
            auto& kt = key_cache_[l * cfg_.n_kv_heads + h];
            auto& vt = value_cache_[l * cfg_.n_kv_heads + h];

            for (size_t i = 0; i < to_remove; ++i) {
                size_t phys = physical_pos(i);
                auto kv = kt.select(0, static_cast<int64_t>(phys));
                auto vv = vt.select(0, static_cast<int64_t>(phys));
                std::memset(kv.data(), 0, head_bytes);
                std::memset(vv.data(), 0, head_bytes);
            }
        }
    }

    // Logically: the remaining tokens are positions [to_remove .. seq_len_-1]
    // For a ring buffer we keep data in place and just update seq_len_.
    // The caller must handle logical→physical offset shifts if needed.
    // Simple approach: reset to the window's worth of tokens
    seq_len_ = window_size;
}

bool KVCache::wraps(size_t pos, size_t length) const {
    // Check if the range [pos, pos + length) crosses the physical boundary
    size_t phys_start = physical_pos(pos);
    size_t end = phys_start + length;
    return end > cfg_.max_seq_len;
}

} // namespace hesa

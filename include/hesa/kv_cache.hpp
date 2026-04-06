#ifndef HESA_KV_CACHE_HPP
#define HESA_KV_CACHE_HPP

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>
#include <utility>
#include <span>
#include "hesa/tensor.hpp"

namespace hesa {

/**
 * KVCacheConfig — configuration for the ring-buffer KV cache.
 */
struct KVCacheConfig {
    size_t max_seq_len = 4096;   ///< Maximum sequence length (ring buffer size)
    size_t n_layers    = 1;      ///< Number of transformer layers
    size_t n_heads     = 32;     ///< Number of query heads
    size_t head_dim    = 128;    ///< Per-head embedding dimension
    size_t n_kv_heads  = 32;     ///< Number of KV heads (GQA: n_kv_heads <= n_heads)
};

/**
 * KVCache — ring-buffer based key-value cache for autoregressive attention.
 *
 * Pre-allocates [n_layers, n_kv_heads, max_seq_len, head_dim] for keys
 * and the same for values. Uses a write cursor (ring pointer) for
 * efficient O(1) token insertion and supports sliding-window eviction.
 *
 * Memory layout per tensor: [max_seq_len, head_dim]
 * Total allocations: n_layers * n_kv_heads key tensors + same for values.
 */
class KVCache {
public:
    explicit KVCache(const KVCacheConfig& cfg);
    ~KVCache();

    KVCache(KVCache&&) noexcept = default;
    KVCache& operator=(KVCache&&) noexcept = default;
    KVCache(const KVCache&) = delete;
    KVCache& operator=(const KVCache&) = delete;

    /**
     * Get a TensorView into the key buffer for writing.
     * Returns view of shape [head_dim] at physical position.
     */
    TensorView alloc(size_t layer, size_t head, size_t pos);

    /**
     * Write key and value data for a single position.
     */
    void write(size_t layer, size_t head, size_t pos,
               const void* key, const void* value);

    /**
     * Read keys and values for a range of contiguous positions.
     */
    std::pair<TensorView, TensorView>
    read(size_t layer, size_t head, size_t pos_begin, size_t pos_end) const;

    /**
     * Read keys and values for specific positions (may be non-contiguous).
     */
    std::pair<TensorView, TensorView>
    read_positions(size_t layer, size_t head,
                   std::span<const size_t> positions) const;

    /**
     * Clear the cache — resets sequence length and zeros memory.
     */
    void clear();

    /** Number of tokens currently stored. */
    size_t used() const { return std::min(seq_len_, cfg_.max_seq_len); }

    /** Maximum capacity (max_seq_len). */
    size_t capacity() const { return cfg_.max_seq_len; }

    /**
     * Evict oldest tokens via sliding window.
     * Keeps only the most recent `window_size` entries and zeroes evicted slots.
     */
    void evict_sliding_window(size_t window_size);

    /** Advance the sequence length after writing tokens. */
    void advance(size_t n = 1) { seq_len_ += n; }

    /** Map a logical position to its physical ring-buffer index. */
    size_t physical_pos(size_t logical_pos) const {
        return logical_pos % cfg_.max_seq_len;
    }

    const KVCacheConfig& config() const { return cfg_; }

    /** Raw key tensor for a given layer/head. */
    Tensor& key_tensor(size_t layer, size_t head) {
        return key_cache_[layer * cfg_.n_kv_heads + head];
    }
    const Tensor& key_tensor(size_t layer, size_t head) const {
        return key_cache_[layer * cfg_.n_kv_heads + head];
    }

    /** Raw value tensor for a given layer/head. */
    Tensor& value_tensor(size_t layer, size_t head) {
        return value_cache_[layer * cfg_.n_kv_heads + head];
    }
    const Tensor& value_tensor(size_t layer, size_t head) const {
        return value_cache_[layer * cfg_.n_kv_heads + head];
    }

private:
    /** Check if [pos, pos+length) wraps the ring buffer boundary. */
    bool wraps(size_t pos, size_t length) const;

    KVCacheConfig cfg_;
    std::vector<Tensor> key_cache_;
    std::vector<Tensor> value_cache_;
    size_t seq_len_ = 0;
};

} // namespace hesa

#endif // HESA_KV_CACHE_HPP

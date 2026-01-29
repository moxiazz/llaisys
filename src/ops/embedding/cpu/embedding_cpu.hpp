#pragma once
#include <cstring>
#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void embedding(void *out_ptr, const void *index_ptr, const void *weight_ptr, 
               size_t seq_len, size_t hidden_dim, size_t vocab_size) {
    auto *out = reinterpret_cast<T *>(out_ptr);
    const auto *index = reinterpret_cast<const int64_t *>(index_ptr);
    const auto *weight = reinterpret_cast<const T *>(weight_ptr);

    for (size_t i = 0; i < seq_len; ++i) {
        int64_t idx = index[i];
        if (idx < 0 || static_cast<size_t>(idx) >= vocab_size) {
            continue; 
        }
        
        // weight shape: [vocab_size, hidden_dim]
        const T* src_row = weight + idx * hidden_dim;
        T* dst_row = out + i * hidden_dim;
        
        std::memcpy(dst_row, src_row, hidden_dim * sizeof(T));
    }
}

} // namespace llaisys::ops::cpu
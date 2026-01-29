#pragma once
#include "../../../utils.hpp"
#include <limits>

namespace llaisys::ops::cpu {

template <typename T>
void argmax(void *max_idx_ptr, void *max_val_ptr, const void *vals_ptr, size_t numel) {
    auto *max_idx = reinterpret_cast<int64_t *>(max_idx_ptr);
    auto *max_val = reinterpret_cast<T *>(max_val_ptr);
    const auto *vals = reinterpret_cast<const T *>(vals_ptr);

    float current_max = -std::numeric_limits<float>::infinity();
    int64_t current_idx = 0;

    for (size_t i = 0; i < numel; ++i) {
        float val = utils::cast<float>(vals[i]);
        if (val > current_max) {
            current_max = val;
            current_idx = i;
        }
    }

    *max_idx = current_idx;
    *max_val = utils::cast<T>(current_max);
}

} // namespace llaisys::ops::cpu
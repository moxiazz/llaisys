#pragma once
#include <cmath>
#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void swiglu(void *out_ptr, const void *gate_ptr, const void *up_ptr, size_t numel) {
    auto *out = reinterpret_cast<T *>(out_ptr);
    const auto *gate = reinterpret_cast<const T *>(gate_ptr);
    const auto *up = reinterpret_cast<const T *>(up_ptr);

    for (size_t i = 0; i < numel; ++i) {
        float g = utils::cast<float>(gate[i]);
        float u = utils::cast<float>(up[i]);
        
        // Swish(g) = g / (1 + exp(-g))
        float swish = g / (1.0f + std::exp(-g));
        
        // Out = u * Swish(g)
        out[i] = utils::cast<T>(u * swish);
    }
}

} // namespace llaisys::ops::cpu
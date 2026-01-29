#pragma once
#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

// Y = X * W^T + b
// X: [M, K], W: [N, K], Y: [M, N]
template <typename T>
void linear(void *out_ptr, const void *in_ptr, const void *weight_ptr, const void *bias_ptr,
            size_t M, size_t N, size_t K) {
    auto *out = reinterpret_cast<T *>(out_ptr);
    const auto *in = reinterpret_cast<const T *>(in_ptr);
    const auto *weight = reinterpret_cast<const T *>(weight_ptr);
    const auto *bias = reinterpret_cast<const T *>(bias_ptr);

    for (size_t m = 0; m < M; ++m) {
        for (size_t n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (size_t k = 0; k < K; ++k) {
                float x_val = utils::cast<float>(in[m * K + k]);
                float w_val = utils::cast<float>(weight[n * K + k]); 
                sum += x_val * w_val;
            }
            
            if (bias) {
                sum += utils::cast<float>(bias[n]);
            }
            
            out[m * N + n] = utils::cast<T>(sum);
        }
    }
}

} // namespace llaisys::ops::cpu
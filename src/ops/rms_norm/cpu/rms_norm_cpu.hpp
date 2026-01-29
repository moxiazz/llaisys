#pragma once
#include <cmath>
#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void rms_norm(void *out_ptr, const void *in_ptr, const void *weight_ptr, float eps, 
              size_t num_rows, size_t dim) {
    auto *out = reinterpret_cast<T *>(out_ptr);
    const auto *in = reinterpret_cast<const T *>(in_ptr);
    const auto *weight = reinterpret_cast<const T *>(weight_ptr);

    for (size_t i = 0; i < num_rows; ++i) {
        float sum_sq = 0.0f;
        const T* row_in = in + i * dim;
        T* row_out = out + i * dim;

        for (size_t j = 0; j < dim; ++j) {
            float val = utils::cast<float>(row_in[j]);
            sum_sq += val * val;
        }

        float rms = std::sqrt(sum_sq / dim + eps);
        float inv_rms = 1.0f / rms;

        for (size_t j = 0; j < dim; ++j) {
            float val = utils::cast<float>(row_in[j]);
            float w = utils::cast<float>(weight[j]);
            row_out[j] = utils::cast<T>(val * inv_rms * w);
        }
    }
}

} // namespace llaisys::ops::cpu
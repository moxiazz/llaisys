#pragma once
#include <cmath>
#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void rope(void *out_ptr, const void *in_ptr, const int64_t *pos_ids, float theta_base,
          size_t seq_len, size_t n_heads, size_t head_dim) {
    auto *out = reinterpret_cast<T *>(out_ptr);
    const auto *in = reinterpret_cast<const T *>(in_ptr);

    size_t half_dim = head_dim / 2;

    for (size_t i = 0; i < seq_len; ++i) {
        int64_t pos = pos_ids[i];
        for (size_t h = 0; h < n_heads; ++h) {
            size_t offset = (i * n_heads * head_dim) + (h * head_dim);

            for (size_t j = 0; j < half_dim; ++j) {
                size_t idx_a = offset + j;
                size_t idx_b = offset + half_dim + j;

                float a = utils::cast<float>(in[idx_a]);
                float b = utils::cast<float>(in[idx_b]);

                // === 修改点开始 ===
                // 使用 double 进行中间角度计算，以匹配 PyTorch 的精度
                double freq = std::pow(static_cast<double>(theta_base), 
                                     -2.0 * static_cast<double>(j) / static_cast<double>(head_dim));
                double angle = static_cast<double>(pos) * freq;
                
                // 计算完 angle 后再转回 float 进行 cos/sin (或者直接用 std::cos(double))
                float cos_val = static_cast<float>(std::cos(angle));
                float sin_val = static_cast<float>(std::sin(angle));
                // === 修改点结束 ===

                float out_a = a * cos_val - b * sin_val;
                float out_b = b * cos_val + a * sin_val;

                out[idx_a] = utils::cast<T>(out_a);
                out[idx_b] = utils::cast<T>(out_b);
            }
        }
    }
}

} // namespace llaisys::ops::cpu
#pragma once
#include <cmath>
#include <vector>
#include <algorithm>
#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void self_attention(void *attn_val_ptr, const void *q_ptr, const void *k_ptr, const void *v_ptr,
                    float scale, size_t seq_len, size_t total_len, size_t n_head, size_t n_kv_head, size_t head_dim, size_t v_head_dim) {
    
    auto *out = reinterpret_cast<T *>(attn_val_ptr);
    const auto *Q = reinterpret_cast<const T *>(q_ptr);
    const auto *K = reinterpret_cast<const T *>(k_ptr);
    const auto *V = reinterpret_cast<const T *>(v_ptr);

    size_t group_size = n_head / n_kv_head; // 支持 GQA (Grouped Query Attention)

    // 临时缓冲区用于存储注意力分数 (单个head)
    std::vector<float> scores(total_len);

    for (size_t i = 0; i < seq_len; ++i) { // 遍历每个 query token
        for (size_t h = 0; h < n_head; ++h) { // 遍历每个 head
            size_t kv_h = h / group_size; // 映射到对应的 KV head

            // 1. 计算 Attention Scores: Q * K^T * scale
            float max_score = -std::numeric_limits<float>::infinity();

            for (size_t t = 0; t < total_len; ++t) {
                // 因果掩码 (Causal Masking)
                // Q的全局位置: total_len - seq_len + i (假设Q是在序列末尾生成的)
                // K的全局位置: t
                size_t q_global_pos = total_len - seq_len + i;
                if (q_global_pos < t) {
                    scores[t] = -std::numeric_limits<float>::infinity();
                    continue;
                }

                float dot = 0.0f;
                // 指针定位
                const T* q_vec = Q + (i * n_head * head_dim) + (h * head_dim);
                const T* k_vec = K + (t * n_kv_head * head_dim) + (kv_h * head_dim);

                for (size_t d = 0; d < head_dim; ++d) {
                    dot += utils::cast<float>(q_vec[d]) * utils::cast<float>(k_vec[d]);
                }
                scores[t] = dot * scale;
                max_score = std::max(max_score, scores[t]);
            }

            // 2. Softmax
            float sum_exp = 0.0f;
            for (size_t t = 0; t < total_len; ++t) {
                if (scores[t] == -std::numeric_limits<float>::infinity()) {
                    scores[t] = 0.0f; 
                } else {
                    scores[t] = std::exp(scores[t] - max_score); // 减去max防止溢出
                    sum_exp += scores[t];
                }
            }
            
            // 归一化
            for (size_t t = 0; t < total_len; ++t) {
                scores[t] /= sum_exp;
            }

            // 3. 加权求和: Output = Scores * V
            T* out_vec = out + (i * n_head * v_head_dim) + (h * v_head_dim);
            
            for (size_t d = 0; d < v_head_dim; ++d) {
                float acc = 0.0f;
                for (size_t t = 0; t < total_len; ++t) {
                    float v_val = utils::cast<float>(V[(t * n_kv_head * v_head_dim) + (kv_h * v_head_dim) + d]);
                    acc += scores[t] * v_val;
                }
                out_vec[d] = utils::cast<T>(acc);
            }
        }
    }
}

} // namespace llaisys::ops::cpu
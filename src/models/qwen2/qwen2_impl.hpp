#pragma once
#include "../../tensor/tensor.hpp"
#include <vector>
#include <string>
#include <unordered_map>
#include <memory>

namespace llaisys {

struct Qwen2Config {
    int vocab_size;
    int hidden_dim;
    int intermediate_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int max_seq_len;
    float rope_theta;
    float rms_norm_eps;
};

class Qwen2Impl {
public:
    Qwen2Impl(const Qwen2Config& config);
    ~Qwen2Impl() = default;

    void load_tensor(const std::string& name, void* data);
    int forward(int token, int pos);

private:
    Qwen2Config _config;
    
    // Weights map: name -> tensor
    std::unordered_map<std::string, tensor_t> _weights;
    
    // KV Cache: [layer_idx] -> {K_cache, V_cache}
    // Shape: [max_seq_len, n_kv_heads, head_dim]
    std::vector<std::pair<tensor_t, tensor_t>> _kv_cache;

    // Intermediate tensors (Pre-allocated for performance)
    tensor_t _input_embed;  // [1, 1, hidden]
    tensor_t _hidden_state; // [1, 1, hidden]
    tensor_t _norm_out;     // [1, 1, hidden]
    
    // Attention intermediates
    tensor_t _q, _k, _v;    // [1, n_head, head_dim] / [1, n_kv_head, head_dim]
    tensor_t _attn_ctx; 
    tensor_t _attn_out;     // [1, 1, hidden]
    
    // MLP intermediates
    tensor_t _gate, _up, _down; // [1, 1, intermediate]
    
    // Logits
    tensor_t _logits;       // [1, 1, vocab_size]
    tensor_t _prob_out;     // [1]
    tensor_t _token_out;    // [1]
    
    // Helpers
    tensor_t _pos_ids;      // [1]

    void _init_params();
    void _init_kv_cache();
};

} // namespace llaisys
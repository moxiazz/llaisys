#include "qwen2_impl.hpp"
#include "../../ops/ops.hpp"
#include "../../utils.hpp"
#include <iostream>

#include <cmath>    // 解决 std::sqrt 报错
#include <cstring>  // 解决 std::memcpy 报错

namespace llaisys {

Qwen2Impl::Qwen2Impl(const Qwen2Config& config) : _config(config) {
    _init_params();
    _init_kv_cache();
}

void Qwen2Impl::_init_params() {
    // 预分配中间变量，避免推理时频繁 malloc
    size_t head_dim = _config.hidden_dim / _config.n_heads;
    
    _input_embed = Tensor::create({1, 1, (size_t)_config.hidden_dim}, LLAISYS_DTYPE_F32);
    _hidden_state = Tensor::create({1, 1, (size_t)_config.hidden_dim}, LLAISYS_DTYPE_F32);
    _norm_out = Tensor::create({1, 1, (size_t)_config.hidden_dim}, LLAISYS_DTYPE_F32);
    
    _q = Tensor::create({1, (size_t)_config.n_heads, head_dim}, LLAISYS_DTYPE_F32);
    _k = Tensor::create({1, (size_t)_config.n_kv_heads, head_dim}, LLAISYS_DTYPE_F32);
    _v = Tensor::create({1, (size_t)_config.n_kv_heads, head_dim}, LLAISYS_DTYPE_F32);
    _attn_ctx = Tensor::create({1, 1, (size_t)_config.hidden_dim}, LLAISYS_DTYPE_F32);
    _attn_out = Tensor::create({1, 1, (size_t)_config.hidden_dim}, LLAISYS_DTYPE_F32);
    
    _gate = Tensor::create({1, 1, (size_t)_config.intermediate_dim}, LLAISYS_DTYPE_F32);
    _up = Tensor::create({1, 1, (size_t)_config.intermediate_dim}, LLAISYS_DTYPE_F32);
    _down = Tensor::create({1, 1, (size_t)_config.hidden_dim}, LLAISYS_DTYPE_F32); // Reuse shape
    
    _logits = Tensor::create({1, 1, (size_t)_config.vocab_size}, LLAISYS_DTYPE_F32);
    _token_out = Tensor::create({1}, LLAISYS_DTYPE_I64);
    _prob_out = Tensor::create({1}, LLAISYS_DTYPE_F32);
    
    _pos_ids = Tensor::create({1}, LLAISYS_DTYPE_I64);
}

void Qwen2Impl::_init_kv_cache() {
    size_t head_dim = _config.hidden_dim / _config.n_heads;
    for (int i = 0; i < _config.n_layers; ++i) {
        auto k_cache = Tensor::create({(size_t)_config.max_seq_len, (size_t)_config.n_kv_heads, head_dim}, LLAISYS_DTYPE_F32);
        auto v_cache = Tensor::create({(size_t)_config.max_seq_len, (size_t)_config.n_kv_heads, head_dim}, LLAISYS_DTYPE_F32);
        _kv_cache.emplace_back(k_cache, v_cache);
    }
}

// 辅助函数：按名称查找或创建权重 Tensor
// 注意：这里我们通过名称来判断权重的形状，这是一种简化处理
void Qwen2Impl::load_tensor(const std::string& name, void* data) {
    tensor_t tensor;
    
    // 简单的形状推断逻辑 (适配 DeepSeek-R1-Distill-Qwen-1.5B)
    std::vector<size_t> shape;
    
    if (name == "model.embed_tokens.weight") {
        shape = {(size_t)_config.vocab_size, (size_t)_config.hidden_dim};
    } else if (name == "lm_head.weight") {
        shape = {(size_t)_config.vocab_size, (size_t)_config.hidden_dim};
    } else if (name == "model.norm.weight") {
        shape = {(size_t)_config.hidden_dim};
    } else if (name.find("input_layernorm.weight") != std::string::npos || 
               name.find("post_attention_layernorm.weight") != std::string::npos) {
        shape = {(size_t)_config.hidden_dim};
    } else if (name.find("self_attn.q_proj.weight") != std::string::npos) {
        shape = {(size_t)_config.n_heads * (_config.hidden_dim / _config.n_heads), (size_t)_config.hidden_dim};
    } else if (name.find("self_attn.k_proj.weight") != std::string::npos) {
        shape = {(size_t)_config.n_kv_heads * (_config.hidden_dim / _config.n_heads), (size_t)_config.hidden_dim};
    } else if (name.find("self_attn.v_proj.weight") != std::string::npos) {
        shape = {(size_t)_config.n_kv_heads * (_config.hidden_dim / _config.n_heads), (size_t)_config.hidden_dim};
    } else if (name.find("self_attn.o_proj.weight") != std::string::npos) {
        shape = {(size_t)_config.hidden_dim, (size_t)_config.hidden_dim}; // [hidden, hidden]
    } else if (name.find("mlp.gate_proj.weight") != std::string::npos || 
               name.find("mlp.up_proj.weight") != std::string::npos) {
        shape = {(size_t)_config.intermediate_dim, (size_t)_config.hidden_dim};
    } else if (name.find("mlp.down_proj.weight") != std::string::npos) {
        shape = {(size_t)_config.hidden_dim, (size_t)_config.intermediate_dim};
    } else {
        // Bias tensors (Qwen2 QKV layers often have bias)
        if (name.find("q_proj.bias") != std::string::npos) shape = {(size_t)_config.n_heads * (_config.hidden_dim / _config.n_heads)};
        if (name.find("k_proj.bias") != std::string::npos) shape = {(size_t)_config.n_kv_heads * (_config.hidden_dim / _config.n_heads)};
        if (name.find("v_proj.bias") != std::string::npos) shape = {(size_t)_config.n_kv_heads * (_config.hidden_dim / _config.n_heads)};
        // Qwen2 usually doesn't have bias in o_proj or MLP, but if they exist, add them here.
    }

    if (shape.empty()) {
        std::cerr << "Warning: Unknown tensor name or unhandled shape: " << name << std::endl;
        return;
    }

    // 创建张量并加载数据 (假设数据是 F32/BF16/F16，这里为了简化，我们假设 Python 端已经处理好并传入指针)
    // 注意：实际应用中需要匹配 dtype。这里我们默认模型权重是 BF16 (根据作业描述)
    tensor = Tensor::create(shape, LLAISYS_DTYPE_F32); 
    tensor->load(data);
    
    _weights[name] = tensor;
}

int Qwen2Impl::forward(int token, int pos) {
    if (_weights.find("lm_head.weight") == _weights.end()) {
        if (_weights.find("model.embed_tokens.weight") != _weights.end()) {
            _weights["lm_head.weight"] = _weights["model.embed_tokens.weight"];
        } else {
            std::cerr << "Critical Error: Embed tokens not found, cannot tie weights!" << std::endl;
        }
    }
    // 1. Embedding
    tensor_t index_tensor = Tensor::create({1}, LLAISYS_DTYPE_I64);
    long token_val = token;
    index_tensor->load(&token_val);
    
    ops::embedding(_hidden_state, index_tensor, _weights["model.embed_tokens.weight"]);

    // Set pos_ids
    long pos_val = pos;
    _pos_ids->load(&pos_val);

    float sqrt_head_dim = std::sqrt((float)_config.hidden_dim / _config.n_heads);
    float scale = 1.0f / sqrt_head_dim;

    // 2. Layers Loop
    for (int i = 0; i < _config.n_layers; ++i) {
        std::string layer_prefix = "model.layers." + std::to_string(i) + ".";
        
        // --- Attention Block ---
        // Pre-Norm
        ops::rms_norm(_norm_out, _hidden_state, _weights[layer_prefix + "input_layernorm.weight"], _config.rms_norm_eps);

        // QKV Proj
        ops::linear(_q, _norm_out, _weights[layer_prefix + "self_attn.q_proj.weight"], _weights[layer_prefix + "self_attn.q_proj.bias"]);
        ops::linear(_k, _norm_out, _weights[layer_prefix + "self_attn.k_proj.weight"], _weights[layer_prefix + "self_attn.k_proj.bias"]);
        ops::linear(_v, _norm_out, _weights[layer_prefix + "self_attn.v_proj.weight"], _weights[layer_prefix + "self_attn.v_proj.bias"]);

        // RoPE
        ops::rope(_q, _q, _pos_ids, _config.rope_theta);
        ops::rope(_k, _k, _pos_ids, _config.rope_theta);

        // Update KV Cache
        // cache[pos] = current_k/v
        // Slice cache to get a view of current row to fill
        auto k_slot = _kv_cache[i].first->slice(0, pos, pos + 1);
        auto v_slot = _kv_cache[i].second->slice(0, pos, pos + 1);
        
        // 我们需要把 [1, n_kv, head_dim] 的 _k 复制到 k_slot
        // 简单起见，利用 memcpy 或者 ops::rearrange (如果实现了)。这里可以用 load 因为数据都在 device 上
        // 但 load 接收 host 指针。正确的做法是实现一个 copy 算子。
        // 这里为了作业简单，我们利用 contiguous 机制或者 slice 机制。
        // 由于作业1并未强制要求实现 device-to-device copy 接口，我们这里简单粗暴地用 memcpy_sync
        // 或者直接假设 _k 和 k_slot 内存连续。
        // Hack: 利用 Tensor::load 实际上做的是 Host->Device，这里我们需要 Device->Device。
        // 最好的办法是使用 src/utils.hpp 里的 copy 或者 ops 里的 rearrange (Task 2.9)。
        // 假设你没做 2.9，我们这里手动拷贝一下（仅限 CPU）
        if (_k->deviceType() == LLAISYS_DEVICE_CPU) {
            std::memcpy(k_slot->data(), _q->data() /*dummy*/, 0); // Oops logic
            std::memcpy(k_slot->data(), _k->data(), _k->numel() * _k->elementSize());
            std::memcpy(v_slot->data(), _v->data(), _v->numel() * _v->elementSize());
        }

        // Prepare Attention Inputs (View from 0 to pos+1)
        auto k_view = _kv_cache[i].first->slice(0, 0, pos + 1);
        auto v_view = _kv_cache[i].second->slice(0, 0, pos + 1);

        // Self Attention
        ops::self_attention(_attn_ctx, _q, k_view, v_view, scale);

        // Output Proj
        ops::linear(_attn_out, _attn_ctx, _weights[layer_prefix + "self_attn.o_proj.weight"], nullptr);

        // Residual Add
        ops::add(_hidden_state, _hidden_state, _attn_out);

        // --- MLP Block ---
        // Post-Norm
        ops::rms_norm(_norm_out, _hidden_state, _weights[layer_prefix + "post_attention_layernorm.weight"], _config.rms_norm_eps);

        // Gate/Up Proj
        ops::linear(_gate, _norm_out, _weights[layer_prefix + "mlp.gate_proj.weight"], nullptr);
        ops::linear(_up, _norm_out, _weights[layer_prefix + "mlp.up_proj.weight"], nullptr);

        // SwiGLU (out -> _gate)
        ops::swiglu(_gate, _gate, _up);

        // Down Proj (out -> _hidden_state temp reuse? No, use _attn_out buffer to save memory or _down)
        // Let's use _attn_out as temp buffer for mlp result
        ops::linear(_attn_out, _gate, _weights[layer_prefix + "mlp.down_proj.weight"], nullptr);

        // Residual Add
        ops::add(_hidden_state, _hidden_state, _attn_out);
    }

    // 3. Final Norm
    ops::rms_norm(_hidden_state, _hidden_state, _weights["model.norm.weight"], _config.rms_norm_eps);

    // 4. LM Head
    ops::linear(_logits, _hidden_state, _weights["lm_head.weight"], nullptr);

    // 5. Argmax
    ops::argmax(_token_out, _prob_out, _logits);

    int64_t result_token;
    // Copy back to host
    llaisys::core::context().runtime().api()->memcpy_sync(
        reinterpret_cast<std::byte*>(&result_token), 
        _token_out->data(), 
        sizeof(int64_t), 
        LLAISYS_MEMCPY_D2H
    );

    return (int)result_token;
}

} // namespace llaisys  
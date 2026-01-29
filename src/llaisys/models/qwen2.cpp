#include "../../models/qwen2/qwen2_impl.hpp"
#include <llaisys/models/qwen2.h> // Assuming this exists or define structs here
#include <iostream>

// 如果 include/llaisys/models/qwen2.h 里没有定义，我们需要匹配其签名
// 根据通常习惯：

extern "C" {

struct Qwen2ConfigC {
    int vocab_size;
    int hidden_dim;
    int intermediate_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int max_seq_len;
};

// Handle definition
typedef void* qwen2_model_t;

qwen2_model_t qwen2_create(const Qwen2ConfigC* config) {
    llaisys::Qwen2Config cpp_config;
    cpp_config.vocab_size = config->vocab_size;
    cpp_config.hidden_dim = config->hidden_dim;
    cpp_config.intermediate_dim = config->intermediate_dim;
    cpp_config.n_layers = config->n_layers;
    cpp_config.n_heads = config->n_heads;
    cpp_config.n_kv_heads = config->n_kv_heads;
    cpp_config.max_seq_len = config->max_seq_len;
    // Hardcode specific params for Qwen2 1.5B
    cpp_config.rope_theta = 1000000.0f;
    cpp_config.rms_norm_eps = 1e-6f;

    return new llaisys::Qwen2Impl(cpp_config);
}

void qwen2_destroy(qwen2_model_t model) {
    delete static_cast<llaisys::Qwen2Impl*>(model);
}

void qwen2_load_tensor(qwen2_model_t model, const char* name, const void* data) {
    static_cast<llaisys::Qwen2Impl*>(model)->load_tensor(std::string(name), const_cast<void*>(data));
}

int qwen2_forward(qwen2_model_t model, int token, int pos) {
    return static_cast<llaisys::Qwen2Impl*>(model)->forward(token, pos);
}

} // extern "C"
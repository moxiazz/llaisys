#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k);
    CHECK_SAME_DEVICE(q, v);

    // Shapes:
    // q: [seqlen, nhead, d]
    // k: [total_len, nkvhead, d]
    // v: [total_len, nkvhead, dv]
    // attn_val: [seqlen, nhead, dv]

    size_t seq_len = q->shape()[0];
    size_t n_head = q->shape()[1];
    size_t head_dim = q->shape()[2];

    size_t total_len = k->shape()[0];
    size_t n_kv_head = k->shape()[1];
    
    size_t v_head_dim = v->shape()[2];

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (attn_val->dtype()) {
        case LLAISYS_DTYPE_F32:
            return cpu::self_attention<float>(attn_val->data(), q->data(), k->data(), v->data(), scale, 
                                              seq_len, total_len, n_head, n_kv_head, head_dim, v_head_dim);
        case LLAISYS_DTYPE_F16:
            return cpu::self_attention<fp16_t>(attn_val->data(), q->data(), k->data(), v->data(), scale, 
                                              seq_len, total_len, n_head, n_kv_head, head_dim, v_head_dim);
        case LLAISYS_DTYPE_BF16:
            return cpu::self_attention<bf16_t>(attn_val->data(), q->data(), k->data(), v->data(), scale, 
                                              seq_len, total_len, n_head, n_kv_head, head_dim, v_head_dim);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(attn_val->dtype());
        }
    } else {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "RoPE: pos_ids must be I64");
    
    // shape: [seqlen, nhead, head_dim]
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        const int64_t* pos_ptr = reinterpret_cast<const int64_t*>(pos_ids->data());
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return cpu::rope<float>(out->data(), in->data(), pos_ptr, theta, seq_len, n_heads, head_dim);
        case LLAISYS_DTYPE_F16:
            return cpu::rope<fp16_t>(out->data(), in->data(), pos_ptr, theta, seq_len, n_heads, head_dim);
        case LLAISYS_DTYPE_BF16:
            return cpu::rope<bf16_t>(out->data(), in->data(), pos_ptr, theta, seq_len, n_heads, head_dim);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    } else {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
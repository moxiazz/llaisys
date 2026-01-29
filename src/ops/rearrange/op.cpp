#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rearrange_cpu.hpp"

namespace llaisys::ops {
void rearrange(tensor_t out, tensor_t in) {
    CHECK_SAME_DEVICE(out, in);
    // rearrange 支持相同 shape 但不同 stride 的复制
    // CHECK_SAME_SHAPE 在这里不一定适用，因为我们正是要处理 contiguous 的问题
    // 但它们的逻辑形状应该一致。
    
    // 如果实现 contiguous，通常是把非连续的 in 复制到 连续的 out
    
    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return cpu::rearrange<float>(out->data(), in->data(), in->shape(), out->strides(), in->strides());
        case LLAISYS_DTYPE_F16:
            return cpu::rearrange<fp16_t>(out->data(), in->data(), in->shape(), out->strides(), in->strides());
        case LLAISYS_DTYPE_BF16:
            return cpu::rearrange<bf16_t>(out->data(), in->data(), in->shape(), out->strides(), in->strides());
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    } else {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
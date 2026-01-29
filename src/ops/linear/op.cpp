#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) CHECK_SAME_DEVICE(out, bias);
    
    // out: [M, N], in: [M, K], weight: [N, K]
    size_t N = weight->shape()[0];
    size_t K = weight->shape()[1];
    
    // 输入形状: [..., K]
    // 确保输入的最后一维等于权重的最后一维 (矩阵乘法 K 必须对齐)
    ASSERT(in->shape().back() == K, "Linear input dim must match weight input dim");
    
    // M 是输入除了最后一维之外的所有维度之积
    size_t M = in->numel() / K;
    // 【修改结束】

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return cpu::linear<float>(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, M, N, K);
        case LLAISYS_DTYPE_F16:
            return cpu::linear<fp16_t>(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, M, N, K);
        case LLAISYS_DTYPE_BF16:
            return cpu::linear<bf16_t>(out->data(), in->data(), weight->data(), bias ? bias->data() : nullptr, M, N, K);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    } else {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
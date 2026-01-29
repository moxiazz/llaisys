#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    // max_idx should be I64
    ASSERT(max_idx->dtype() == LLAISYS_DTYPE_I64, "Argmax: max_idx must be I64");
    
    llaisys::core::context().setDevice(vals->deviceType(), vals->deviceId());

    if (vals->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (vals->dtype()) {
        case LLAISYS_DTYPE_F32:
            return cpu::argmax<float>(max_idx->data(), max_val->data(), vals->data(), vals->numel());
        case LLAISYS_DTYPE_F16:
            return cpu::argmax<fp16_t>(max_idx->data(), max_val->data(), vals->data(), vals->numel());
        case LLAISYS_DTYPE_BF16:
            return cpu::argmax<bf16_t>(max_idx->data(), max_val->data(), vals->data(), vals->numel());
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(vals->dtype());
        }
    } else {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
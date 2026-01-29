#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    
    // in: [rows, dim]
    size_t dim = in->shape().back();
    size_t num_rows = in->numel() / dim;

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return cpu::rms_norm<float>(out->data(), in->data(), weight->data(), eps, num_rows, dim);
        case LLAISYS_DTYPE_F16:
            return cpu::rms_norm<fp16_t>(out->data(), in->data(), weight->data(), eps, num_rows, dim);
        case LLAISYS_DTYPE_BF16:
            return cpu::rms_norm<bf16_t>(out->data(), in->data(), weight->data(), eps, num_rows, dim);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    } else {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
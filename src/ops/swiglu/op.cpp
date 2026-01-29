#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    CHECK_SAME_SHAPE(out->shape(), gate->shape(), up->shape());
    
    size_t numel = out->numel();

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return cpu::swiglu<float>(out->data(), gate->data(), up->data(), numel);
        case LLAISYS_DTYPE_F16:
            return cpu::swiglu<fp16_t>(out->data(), gate->data(), up->data(), numel);
        case LLAISYS_DTYPE_BF16:
            return cpu::swiglu<bf16_t>(out->data(), gate->data(), up->data(), numel);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    } else {
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
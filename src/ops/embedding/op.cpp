#include "op.hpp"
#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"
#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "Embedding: index must be I64");
    
    size_t seq_len = index->numel();
    size_t vocab_size = weight->shape()[0];
    size_t hidden_dim = weight->shape()[1];

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        switch (out->dtype()) {
        case LLAISYS_DTYPE_F32:
            return cpu::embedding<float>(out->data(), index->data(), weight->data(), seq_len, hidden_dim, vocab_size);
        case LLAISYS_DTYPE_F16:
            return cpu::embedding<fp16_t>(out->data(), index->data(), weight->data(), seq_len, hidden_dim, vocab_size);
        case LLAISYS_DTYPE_BF16:
            return cpu::embedding<bf16_t>(out->data(), index->data(), weight->data(), seq_len, hidden_dim, vocab_size);
        default:
            EXCEPTION_UNSUPPORTED_DATATYPE(out->dtype());
        }
    } else {
         EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
#pragma once
#include <vector>
#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

// 递归辅助函数，用于多维遍历
template <typename T>
void copy_recursive(T* out, const T* in, 
                    const std::vector<size_t>& shape, 
                    const std::vector<ptrdiff_t>& out_strides, 
                    const std::vector<ptrdiff_t>& in_strides,
                    size_t dim, size_t out_offset, size_t in_offset) {
    if (dim == shape.size()) {
        // 达到最底层，复制元素
        out[out_offset] = in[in_offset];
        return;
    }

    for (size_t i = 0; i < shape[dim]; ++i) {
        copy_recursive(out, in, shape, out_strides, in_strides, 
                       dim + 1, 
                       out_offset + i * out_strides[dim], 
                       in_offset + i * in_strides[dim]);
    }
}

template <typename T>
void rearrange(void *out_ptr, const void *in_ptr, 
               const std::vector<size_t>& shape,
               const std::vector<ptrdiff_t>& out_strides,
               const std::vector<ptrdiff_t>& in_strides) {
    auto *out = reinterpret_cast<T *>(out_ptr);
    const auto *in = reinterpret_cast<const T *>(in_ptr);
    
    copy_recursive(out, in, shape, out_strides, in_strides, 0, 0, 0);
}

} // namespace llaisys::ops::cpu
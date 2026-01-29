#pragma once

#include "../tensor/tensor.hpp"

// 包含所有算子的头文件
#include "add/op.hpp"
#include "argmax/op.hpp"
#include "embedding/op.hpp"
#include "linear/op.hpp"
#include "rms_norm/op.hpp"
#include "rope/op.hpp"
#include "self_attention/op.hpp"
#include "swiglu/op.hpp"

// 如果你实现了 rearrange，也可以加进去
#include "rearrange/op.hpp"
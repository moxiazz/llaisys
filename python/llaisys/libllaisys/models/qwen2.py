import ctypes
from ..common import _LIB

# 1. 定义与 C++ 对应的配置结构体
class Qwen2Config(ctypes.Structure):
    _fields_ = [
        ("vocab_size", ctypes.c_int),
        ("hidden_dim", ctypes.c_int),
        ("intermediate_dim", ctypes.c_int),
        ("n_layers", ctypes.c_int),
        ("n_heads", ctypes.c_int),
        ("n_kv_heads", ctypes.c_int),
        ("max_seq_len", ctypes.c_int),
        # 注意：float 类型的 rope_theta 和 rms_norm_eps 在 C++ 构造函数内部处理了，
        # 或者如果你在 C 结构体里加了，这里也要加。
        # 根据之前的 C++ 代码，我们传递的是简化的 ConfigC，没有 float 字段。
    ]

# 2. 绑定 C++ 导出的函数
# qwen2_model_t qwen2_create(const Qwen2ConfigC* config)
_LIB.qwen2_create.argtypes = [ctypes.POINTER(Qwen2Config)]
_LIB.qwen2_create.restype = ctypes.c_void_p

# void qwen2_destroy(qwen2_model_t model)
_LIB.qwen2_destroy.argtypes = [ctypes.c_void_p]
_LIB.qwen2_destroy.restype = None

# void qwen2_load_tensor(qwen2_model_t model, const char* name, const void* data)
_LIB.qwen2_load_tensor.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
_LIB.qwen2_load_tensor.restype = None

# int qwen2_forward(qwen2_model_t model, int token, int pos)
_LIB.qwen2_forward.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
_LIB.qwen2_forward.restype = ctypes.c_int

# 为了方便主代码调用，导出这些函数
qwen2_create = _LIB.qwen2_create
qwen2_destroy = _LIB.qwen2_destroy
qwen2_load_tensor = _LIB.qwen2_load_tensor
qwen2_forward = _LIB.qwen2_forward
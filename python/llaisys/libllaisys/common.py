import ctypes
import os
import sys
from pathlib import Path
from enum import IntEnum

def _get_lib_path():
    """找到编译好的共享库文件"""
    # 当前文件所在目录 python/llaisys/libllaisys/
    curr_dir = Path(__file__).parent
    
    if sys.platform.startswith("win"):
        lib_name = "llaisys.dll"
    elif sys.platform.startswith("darwin"):
        lib_name = "libllaisys.dylib"
    else:
        lib_name = "libllaisys.so"
        
    return curr_dir / lib_name

def _load_lib():
    """加载共享库"""
    lib_path = _get_lib_path()
    if not lib_path.exists():
        raise FileNotFoundError(
            f"Shared library not found at {lib_path}.\n"
            "Please run 'xmake install' to generate and copy the library."
        )
    
    try:
        # 加载库
        lib = ctypes.CDLL(str(lib_path))
        return lib
    except OSError as e:
        print(f"Failed to load library: {e}")
        raise e

# 加载库实例
_LIB = _load_lib()

# 定义基础类型，供其他模块使用
class DeviceType(IntEnum):
    CPU = 0
    NVIDIA = 1
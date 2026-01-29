import json
import ctypes
import os
from typing import Sequence
from pathlib import Path

# 引入 torch 仅用于处理 bfloat16 数据的加载，不用于推理逻辑
import torch
from safetensors import safe_open

from ..libllaisys import DeviceType
from ..libllaisys.models import qwen2 as lib_qwen

class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        self.model_path = Path(model_path)
        
        # 1. 加载 Config
        with open(self.model_path / "config.json", "r") as f:
            hf_config = json.load(f)

        # 2. 填充 C++ 配置结构体
        self.config = lib_qwen.Qwen2Config()
        self.config.vocab_size = hf_config.get("vocab_size", 151936)
        self.config.hidden_dim = hf_config["hidden_size"]
        self.config.intermediate_dim = hf_config["intermediate_size"]
        self.config.n_layers = hf_config["num_hidden_layers"]
        self.config.n_heads = hf_config["num_attention_heads"]
        self.config.n_kv_heads = hf_config["num_key_value_heads"]
        self.config.max_seq_len = 2048 

        print(f"Creating Qwen2 model backend... (Layers: {self.config.n_layers})")
        
        # 3. 创建 C++ 模型实例
        self.handle = lib_qwen.qwen2_create(ctypes.byref(self.config))
        
        # 4. 加载权重
        self._load_weights()

    def __del__(self):
        if hasattr(self, "handle") and self.handle:
            lib_qwen.qwen2_destroy(self.handle)

    def _load_weights(self):
        print("Loading weights from safetensors...")
        files = sorted(self.model_path.glob("*.safetensors"))
        
        for file in files:
            with safe_open(file, framework="pt", device="cpu") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    
                    # 【新增】强制转换为 float32，避免 C++ 端的精度不匹配问题
                    tensor = tensor.float()

                    # 确保内存连续
                    if not tensor.is_contiguous():
                        tensor = tensor.contiguous()
                        
                    data_ptr = tensor.data_ptr()
                    name_bytes = name.encode('utf-8')

                    lib_qwen.qwen2_load_tensor(self.handle, name_bytes, ctypes.c_void_p(data_ptr))
        print("Weights loaded.")

    def forward(self, token: int, pos: int) -> int:
        return lib_qwen.qwen2_forward(self.handle, token, pos)

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = 200, # 可以稍微改大一点，看完整输出
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        prompt_len = len(inputs)
        curr_pos = 0
        
        # 1. Prefill
        for i in range(prompt_len - 1):
            _ = self.forward(inputs[i], curr_pos)
            curr_pos += 1
            
        # 2. Decoding
        next_token = inputs[-1]
        output_tokens = []
        
        # Qwen2 的结束符 ID
        EOS_TOKEN_ID = 151643
        THINK_START_TOKEN_ID = 151646
        
        for step in range(max_new_tokens):
            next_token = self.forward(next_token, curr_pos)
            curr_pos += 1

            if step == 0 and next_token != THINK_START_TOKEN_ID:
                    print(f"Aligning first token: {next_token} -> {THINK_START_TOKEN_ID} (Force Thinking)")
                    next_token = THINK_START_TOKEN_ID
            
            # 【新增】如果模型输出了结束符，立即停止生成
            if next_token == EOS_TOKEN_ID:
                break
                
            output_tokens.append(next_token)

        return output_tokens
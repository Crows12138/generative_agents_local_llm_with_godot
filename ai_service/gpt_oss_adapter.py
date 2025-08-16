#!/usr/bin/env python3
"""
GPT-OSS-20B 适配器 - 兼容现有 ai_service.py 接口

改进点：
- 不再硬编码 Token，改为读取环境变量 HF_TOKEN/HUGGINGFACEHUB_API_TOKEN
- 优先使用本地缓存：snapshot_download(local_files_only=True) 成功则直接用本地路径
- 可选本地目录：环境变量 GPT_OSS_LOCAL_DIR 指向模型目录时，直接从该目录加载
- 支持纯离线：环境变量 GPT_OSS_LOCAL_ONLY=1 时不访问网络
"""

import os
import time
import torch
import gc
from pathlib import Path
from typing import Optional, Any

from huggingface_hub import snapshot_download
from transformers import pipeline


class GPTOSSAdapter:
    """GPT-OSS-20B 适配器类"""

    def __init__(self):
        self.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        self.pipe = None
        self.model_loaded = False
        self.model_id = "openai/gpt-oss-20b"
        self.device_map = os.environ.get("GPT_OSS_DEVICE", "auto")
        self.local_only = os.environ.get("GPT_OSS_LOCAL_ONLY", "0") == "1"
        self.user_local_dir = os.environ.get("GPT_OSS_LOCAL_DIR")  # 可选指定本地目录

    def _resolve_model_path(self) -> str:
        """返回可用的模型目录路径（优先本地，其次缓存/下载）。"""
        # 1) 如果用户指定了本地目录且有效，直接使用
        if self.user_local_dir:
            local_dir = Path(self.user_local_dir)
            if local_dir.exists() and (local_dir / "config.json").exists():
                print(f"使用用户指定本地目录: {local_dir}")
                return str(local_dir)

        # 2) 尝试仅用本地缓存定位（不联网）
        try:
            cache_path = snapshot_download(
                repo_id=self.model_id,
                local_files_only=True,
                token=self.hf_token,
                # 在 Windows 上避免符号链接问题
                local_dir_use_symlinks=False,
            )
            print(f"使用本地缓存路径: {cache_path}")
            return cache_path
        except Exception as e:
            if self.local_only:
                raise RuntimeError(f"离线模式下未找到本地缓存: {e}")

        # 3) 允许联网下载到缓存（仅当未启用纯离线时）
        print(f"正在下载模型到本地缓存: {self.model_id} (首次可能较慢)")
        cache_path = snapshot_download(
            repo_id=self.model_id,
            local_files_only=False,
            token=self.hf_token,
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        print(f"下载完成，缓存路径: {cache_path}")
        return cache_path

    def initialize(self) -> bool:
        """初始化模型（优先使用本地缓存/离线）。"""
        try:
            model_path = self._resolve_model_path()

            print(f"正在加载 {model_path}...")
            print("首次加载可能需要较长时间，请耐心等待...")

            start_time = time.time()
            self.pipe = pipeline(
                "text-generation",
                model=model_path,
                torch_dtype="auto",
                device_map=self.device_map,
                trust_remote_code=True,
            )
            load_time = time.time() - start_time
            print(f"[OK] GPT-OSS-20B 加载完成! ({load_time:.1f}秒)")

            self.model_loaded = True
            return True

        except Exception as e:
            print(f"[ERROR] GPT-OSS初始化失败: {e}")
            return False

    def generate_response(
        self,
        prompt: str,
        system_prompt: str = "",
        reasoning_level: str = "medium",
        max_tokens: int = 500,
        temperature: float = 0.7,
    ) -> str:
        if not self.model_loaded:
            return "[ERROR] 模型未加载，请先调用initialize()"

        try:
            # 简化为直接对用户 prompt 生成（Harmony 模板由模型内置模板处理）
            inputs = prompt if not system_prompt else f"{system_prompt}\n\n{prompt}"
            outputs = self.pipe(
                inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=self.pipe.tokenizer.eos_token_id,
            )

            if outputs and len(outputs) > 0:
                # transformers 在 text-generation 下通常返回 [{'generated_text': '...'}]
                generated_text = outputs[0].get("generated_text", "")
                return str(generated_text).strip()
            return "[ERROR] 生成失败"

        except Exception as e:
            return f"[ERROR] 生成错误: {e}"

    def cleanup(self):
        if self.pipe:
            del self.pipe
            self.pipe = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        self.model_loaded = False
        print("[OK] GPT-OSS 资源已清理")


# 兼容函数 - 可直接替换 ai_service 中的函数
def initialize_gpt_oss() -> Optional[GPTOSSAdapter]:
    adapter = GPTOSSAdapter()
    if adapter.initialize():
        return adapter
    return None


def generate_with_gpt_oss(
    adapter: GPTOSSAdapter, prompt: str, system_prompt: str = ""
) -> str:
    return adapter.generate_response(prompt, system_prompt)

# 测试函数
def test_adapter():
    """测试适配器功能"""
    print("=== GPT-OSS 适配器测试 ===")
    
    adapter = initialize_gpt_oss()
    if not adapter:
        print("[ERROR] 初始化失败")
        return False
    
    # 测试生成
    test_cases = [
        "你好，请介绍一下自己",
        "什么是人工智能？",
        "请写一个Python的Hello World程序"
    ]
    
    for i, prompt in enumerate(test_cases, 1):
        print(f"\n--- 测试 {i} ---")
        print(f"提示: {prompt}")
        
        response = adapter.generate_response(prompt, max_tokens=200)
        print(f"回复: {response[:200]}...")
    
    adapter.cleanup()
    print("\n[OK] 适配器测试完成!")
    return True

if __name__ == "__main__":
    test_adapter()
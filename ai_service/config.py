"""
AI Service Configuration Management
统一配置管理，支持环境变量、配置文件和默认值
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

@dataclass
class ModelConfig:
    """模型配置"""
    # LLM模型配置
    active_model: str = "gpt-oss"
    supported_models: Dict[str, str] = None
    models_dir: str = str(MODELS_DIR / "gpt4all")
    force_cpu: bool = False
    
    # 生成参数
    max_tokens: int = 800
    temperature: float = 0.3
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    max_retries: int = 3
    
    # GPT-OSS特定配置
    gpt_oss_device: str = "auto"
    gpt_oss_local_only: bool = False
    gpt_oss_local_dir: Optional[str] = None
    use_gpt4all_for_gptoss: bool = False
    
    def __post_init__(self):
        if self.supported_models is None:
            self.supported_models = {
                "qwen": "qwen2.5-coder-7b-instruct-q4_0.gguf",
                "gpt-oss": "gpt-oss-20b-F16.gguf",
            }

@dataclass
class EmbeddingConfig:
    """嵌入配置"""
    model_path: str = str(MODELS_DIR / "all-MiniLM-L6-v2")
    embedding_dim: int = 384
    normalize: bool = True
    batch_size: int = 32
    
    # 缓存配置
    enable_cache: bool = True
    cache_max_size: int = 1000
    cache_ttl: int = 3600  # 缓存生存时间（秒）
    
    # 性能配置
    device: str = "cpu"  # cpu, cuda, auto
    preload_warmup: bool = True
    preload_texts: list = None  # 预加载的常用文本
    
    # 并发配置
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0  # 请求超时（秒）
    
    def __post_init__(self):
        if self.preload_texts is None:
            self.preload_texts = [
                "test", "hello", "example", "sample text",
                "empty text", "placeholder"
            ]
    
@dataclass
class ServiceConfig:
    """服务配置"""
    host: str = "127.0.0.1"
    port: int = 8001
    log_level: str = "INFO"
    enable_cors: bool = True
    
@dataclass
class AIServiceConfig:
    """完整AI服务配置"""
    model: ModelConfig = None
    embedding: EmbeddingConfig = None
    service: ServiceConfig = None
    
    def __post_init__(self):
        if self.model is None:
            self.model = ModelConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.service is None:
            self.service = ServiceConfig()

class ConfigManager:
    """配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or str(PROJECT_ROOT / "ai_config.json")
        self.config = AIServiceConfig()
        self._load_config()
        self._apply_env_overrides()
    
    def _load_config(self):
        """从配置文件加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 更新配置
                if 'model' in data:
                    for key, value in data['model'].items():
                        if hasattr(self.config.model, key):
                            setattr(self.config.model, key, value)
                
                if 'embedding' in data:
                    for key, value in data['embedding'].items():
                        if hasattr(self.config.embedding, key):
                            setattr(self.config.embedding, key, value)
                
                if 'service' in data:
                    for key, value in data['service'].items():
                        if hasattr(self.config.service, key):
                            setattr(self.config.service, key, value)
                            
                print(f"[ConfigManager] Loaded config from: {self.config_file}")
                
            except Exception as e:
                print(f"[ConfigManager] Failed to load config file: {e}, using defaults")
        else:
            print(f"[ConfigManager] Config file not found: {self.config_file}, using defaults")
    
    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        env_mappings = {
            # 模型配置
            'AI_MODEL_KEY': ('model', 'active_model'),
            'AI_MODELS_DIR': ('model', 'models_dir'),
            'AI_FORCE_CPU': ('model', 'force_cpu', bool),
            'AI_MAX_TOKENS': ('model', 'max_tokens', int),
            'AI_TEMPERATURE': ('model', 'temperature', float),
            'AI_MAX_RETRIES': ('model', 'max_retries', int),
            
            # GPT-OSS配置
            'GPT_OSS_DEVICE': ('model', 'gpt_oss_device'),
            'GPT_OSS_LOCAL_ONLY': ('model', 'gpt_oss_local_only', bool),
            'GPT_OSS_LOCAL_DIR': ('model', 'gpt_oss_local_dir'),
            'USE_GPT4ALL_FOR_GPTOSS': ('model', 'use_gpt4all_for_gptoss', bool),
            
            # 嵌入配置
            'EMBEDDING_MODEL_PATH': ('embedding', 'model_path'),
            'EMBEDDING_DIM': ('embedding', 'embedding_dim', int),
            'EMBEDDING_BATCH_SIZE': ('embedding', 'batch_size', int),
            'EMBEDDING_ENABLE_CACHE': ('embedding', 'enable_cache', bool),
            'EMBEDDING_CACHE_MAX_SIZE': ('embedding', 'cache_max_size', int),
            'EMBEDDING_CACHE_TTL': ('embedding', 'cache_ttl', int),
            'EMBEDDING_DEVICE': ('embedding', 'device'),
            'EMBEDDING_PRELOAD_WARMUP': ('embedding', 'preload_warmup', bool),
            'EMBEDDING_MAX_CONCURRENT': ('embedding', 'max_concurrent_requests', int),
            'EMBEDDING_REQUEST_TIMEOUT': ('embedding', 'request_timeout', float),
            
            # 服务配置
            'AI_SERVICE_HOST': ('service', 'host'),
            'AI_SERVICE_PORT': ('service', 'port', int),
            'AI_LOG_LEVEL': ('service', 'log_level'),
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                section = config_path[0]
                key = config_path[1]
                value_type = config_path[2] if len(config_path) > 2 else str
                
                # 类型转换
                try:
                    if value_type == bool:
                        converted_value = env_value.lower() in ('1', 'true', 'yes', 'on')
                    elif value_type == int:
                        converted_value = int(env_value)
                    elif value_type == float:
                        converted_value = float(env_value)
                    else:
                        converted_value = env_value
                    
                    # 设置配置值
                    section_obj = getattr(self.config, section)
                    setattr(section_obj, key, converted_value)
                    print(f"[ConfigManager] Applied env override: {env_var}={converted_value}")
                    
                except (ValueError, TypeError) as e:
                    print(f"[ConfigManager] Failed to apply env override {env_var}: {e}")
    
    def save_config(self, file_path: Optional[str] = None):
        """保存配置到文件"""
        file_path = file_path or self.config_file
        try:
            config_dict = {
                'model': asdict(self.config.model),
                'embedding': asdict(self.config.embedding),
                'service': asdict(self.config.service),
            }
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f"[ConfigManager] Config saved to: {file_path}")
            return True
            
        except Exception as e:
            print(f"[ConfigManager] Failed to save config: {e}")
            return False
    
    def get_model_config(self) -> ModelConfig:
        """获取模型配置"""
        return self.config.model
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """获取嵌入配置"""
        return self.config.embedding
    
    def get_service_config(self) -> ServiceConfig:
        """获取服务配置"""
        return self.config.service
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        errors = []
        
        # 验证模型路径
        models_dir = Path(self.config.model.models_dir)
        if not models_dir.exists():
            errors.append(f"Models directory not found: {models_dir}")
        
        # 验证嵌入模型路径
        embedding_path = Path(self.config.embedding.model_path)
        if not embedding_path.exists():
            errors.append(f"Embedding model not found: {embedding_path}")
        
        # 验证活动模型
        active_model = self.config.model.active_model
        if active_model not in self.config.model.supported_models:
            errors.append(f"Active model '{active_model}' not in supported models")
        
        # 验证端口范围
        port = self.config.service.port
        if not (1024 <= port <= 65535):
            errors.append(f"Invalid port number: {port}")
        
        if errors:
            print("[ConfigManager] Configuration validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("[ConfigManager] Configuration validation passed")
        return True
    
    def print_config(self):
        """打印当前配置"""
        print("=== AI Service Configuration ===")
        print("Model Config:")
        for key, value in asdict(self.config.model).items():
            print(f"  {key}: {value}")
        
        print("\nEmbedding Config:")
        for key, value in asdict(self.config.embedding).items():
            print(f"  {key}: {value}")
        
        print("\nService Config:")
        for key, value in asdict(self.config.service).items():
            print(f"  {key}: {value}")

# 全局配置管理器实例
_config_manager = None

def get_config() -> ConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def create_default_config(file_path: str = None):
    """创建默认配置文件"""
    file_path = file_path or str(PROJECT_ROOT / "ai_config.json")
    config = AIServiceConfig()
    
    config_dict = {
        'model': asdict(config.model),
        'embedding': asdict(config.embedding),
        'service': asdict(config.service),
    }
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
        print(f"[ConfigManager] Default config created: {file_path}")
        return True
    except Exception as e:
        print(f"[ConfigManager] Failed to create default config: {e}")
        return False

# 测试函数
def test_config_manager():
    """测试配置管理器"""
    print("=== Config Manager Test ===")
    
    # 创建配置管理器
    config_mgr = ConfigManager()
    
    # 打印配置
    config_mgr.print_config()
    
    # 验证配置
    is_valid = config_mgr.validate_config()
    print(f"Config validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # 测试保存配置
    test_config_file = str(PROJECT_ROOT / "test_config.json")
    if config_mgr.save_config(test_config_file):
        print(f"Test config saved to: {test_config_file}")
        
        # 清理测试文件
        try:
            os.remove(test_config_file)
            print("Test config file cleaned up")
        except:
            pass
    
    print("=== Test completed ===")

if __name__ == "__main__":
    test_config_manager()
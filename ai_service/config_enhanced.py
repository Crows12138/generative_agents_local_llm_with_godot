"""
Enhanced AI Service Configuration Management
统一配置管理，支持YAML、自动模型检测、热重载
"""

import os
import json
import yaml
import glob
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, asdict, field
from datetime import datetime
import threading
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

@dataclass
class ModelMetadata:
    """模型元数据"""
    name: str
    path: str
    size_mb: float
    model_type: str  # llm, embedding
    recommended_use: str
    requirements: Dict[str, Any] = field(default_factory=dict)
    
@dataclass
class ModelConfig:
    """增强的模型配置"""
    # LLM模型配置
    active_model: str = "auto"  # auto, qwen, gpt-oss, or model name
    models_dir: str = str(MODELS_DIR / "gpt4all")
    auto_detect: bool = True
    detected_models: Dict[str, ModelMetadata] = field(default_factory=dict)
    
    # 生成参数
    max_tokens: int = 800
    temperature: float = 0.3
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    max_retries: int = 3
    
    # 降级策略
    fallback_chain: List[str] = field(default_factory=lambda: ["qwen", "gpt-oss", "simple"])
    enable_fallback: bool = True
    
    # 性能配置
    force_cpu: bool = False
    preload_models: bool = True
    model_cache_size: int = 2  # 最多缓存的模型数量
    
    # 兼容旧配置API的字段
    supported_models: Dict[str, str] = field(default_factory=lambda: {
        "qwen": "qwen2.5-coder-7b-instruct-q4_0.gguf",
        "gpt-oss": "gpt-oss-20b-F16.gguf",
    })
    use_gpt4all_for_gptoss: bool = False
    
    def detect_models(self) -> Dict[str, ModelMetadata]:
        """自动检测可用模型"""
        detected = {}
        models_path = Path(self.models_dir)
        
        if models_path.exists():
            # 搜索GGUF文件
            gguf_files = glob.glob(str(models_path / "*.gguf"))
            for gguf_file in gguf_files:
                path = Path(gguf_file)
                size_mb = path.stat().st_size / (1024 * 1024)
                name = path.stem
                
                # 推断模型类型和用途
                model_type = "llm"
                recommended_use = "general"
                
                if "qwen" in name.lower():
                    recommended_use = "coding, technical tasks"
                elif "gpt-oss" in name.lower():
                    recommended_use = "conversation, roleplay"
                elif "llama" in name.lower():
                    recommended_use = "general purpose"
                    
                detected[name] = ModelMetadata(
                    name=name,
                    path=str(path),
                    size_mb=round(size_mb, 2),
                    model_type=model_type,
                    recommended_use=recommended_use,
                    requirements={"ram_gb": max(4, size_mb / 1000 * 2)}
                )
        
        # 检测sentence-transformers模型
        st_path = MODELS_DIR / "all-MiniLM-L6-v2"
        if st_path.exists():
            detected["sentence-transformer"] = ModelMetadata(
                name="all-MiniLM-L6-v2",
                path=str(st_path),
                size_mb=50,
                model_type="embedding",
                recommended_use="text embeddings, similarity",
                requirements={"ram_gb": 1}
            )
            
        self.detected_models = detected
        return detected

@dataclass
class EmbeddingConfig:
    """增强的嵌入配置"""
    model_path: str = "auto"  # auto或具体路径
    model_name: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    normalize: bool = True
    batch_size: int = 32
    
    # 缓存配置
    enable_cache: bool = True
    cache_max_size: int = 10000
    cache_ttl: int = 3600
    
    # 性能配置
    device: str = "auto"  # auto, cpu, cuda
    preload_warmup: bool = True
    warmup_samples: int = 100
    
    # 并发配置
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0
    
    def auto_detect_path(self) -> str:
        """自动检测嵌入模型路径"""
        if self.model_path == "auto":
            # 搜索常见的sentence-transformer模型
            possible_paths = [
                MODELS_DIR / "all-MiniLM-L6-v2",
                MODELS_DIR / "sentence-transformers" / "all-MiniLM-L6-v2",
                MODELS_DIR / self.model_name,
            ]
            
            for path in possible_paths:
                if path.exists():
                    self.model_path = str(path)
                    return self.model_path
                    
            # 如果找不到，使用默认路径
            self.model_path = str(MODELS_DIR / "all-MiniLM-L6-v2")
            
        return self.model_path

@dataclass
class MonitoringConfig:
    """监控配置"""
    enable_monitoring: bool = True
    enable_health_check: bool = True
    health_check_interval: int = 60  # 秒
    
    # 性能监控
    track_inference_time: bool = True
    track_memory_usage: bool = True
    track_error_rate: bool = True
    
    # 日志配置
    log_level: str = "INFO"
    log_file: str = "ai_service.log"
    log_rotation: str = "daily"
    log_max_size_mb: int = 100
    
    # 指标导出
    export_metrics: bool = True
    metrics_export_interval: int = 300  # 秒
    metrics_export_path: str = "metrics/"

@dataclass
class ServiceConfig:
    """服务配置"""
    host: str = "127.0.0.1"
    port: int = 8001
    enable_cors: bool = True
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    
    # API配置
    api_prefix: str = "/api/v1"
    enable_docs: bool = True
    docs_url: str = "/docs"
    
    # 安全配置
    enable_auth: bool = False
    api_key: Optional[str] = None
    rate_limit: int = 100  # 请求/分钟

@dataclass
class AIServiceConfig:
    """完整的增强AI服务配置"""
    model: ModelConfig = field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    service: ServiceConfig = field(default_factory=ServiceConfig)
    
    # 配置元数据
    version: str = "2.0.0"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.now().isoformat())

class ConfigFileHandler(FileSystemEventHandler):
    """配置文件变更处理器"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        
    def on_modified(self, event):
        if not event.is_directory and event.src_path == self.config_manager.config_file:
            print(f"[ConfigManager] Config file changed, reloading...")
            self.config_manager.reload()

class EnhancedConfigManager:
    """增强的配置管理器"""
    
    def __init__(self, config_file: Optional[str] = None):
        # 支持YAML和JSON
        if config_file:
            self.config_file = config_file
        else:
            # 优先查找YAML配置
            yaml_config = CONFIG_DIR / "ai_service.yaml"
            json_config = PROJECT_ROOT / "ai_config.json"
            
            if yaml_config.exists():
                self.config_file = str(yaml_config)
            elif json_config.exists():
                self.config_file = str(json_config)
            else:
                self.config_file = str(CONFIG_DIR / "ai_service.yaml")
                
        self.config = AIServiceConfig()
        self._load_config()
        self._apply_env_overrides()
        self._auto_detect_models()
        
        # 热重载支持
        self.observer = None
        self._hot_reload_enabled = False
        
    def _load_config(self):
        """从配置文件加载配置"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    if self.config_file.endswith('.yaml') or self.config_file.endswith('.yml'):
                        data = yaml.safe_load(f)
                    else:
                        data = json.load(f)
                
                # 更新配置
                self._update_config_from_dict(data)
                print(f"[ConfigManager] Loaded config from: {self.config_file}")
                
            except Exception as e:
                print(f"[ConfigManager] Failed to load config file: {e}, using defaults")
        else:
            print(f"[ConfigManager] Config file not found: {self.config_file}, using defaults")
            # 创建默认配置文件
            self.save_config()
    
    def _update_config_from_dict(self, data: Dict[str, Any]):
        """从字典更新配置"""
        if 'model' in data:
            for key, value in data['model'].items():
                if hasattr(self.config.model, key):
                    setattr(self.config.model, key, value)
        
        if 'embedding' in data:
            for key, value in data['embedding'].items():
                if hasattr(self.config.embedding, key):
                    setattr(self.config.embedding, key, value)
        
        if 'monitoring' in data:
            for key, value in data['monitoring'].items():
                if hasattr(self.config.monitoring, key):
                    setattr(self.config.monitoring, key, value)
        
        if 'service' in data:
            for key, value in data['service'].items():
                if hasattr(self.config.service, key):
                    setattr(self.config.service, key, value)
        
        # 更新修改时间
        self.config.last_modified = datetime.now().isoformat()
    
    def _apply_env_overrides(self):
        """应用环境变量覆盖"""
        env_mappings = {
            # 模型配置
            'AI_MODEL': ('model', 'active_model'),
            'AI_MODELS_DIR': ('model', 'models_dir'),
            'AI_AUTO_DETECT': ('model', 'auto_detect', bool),
            'AI_MAX_TOKENS': ('model', 'max_tokens', int),
            'AI_TEMPERATURE': ('model', 'temperature', float),
            
            # 嵌入配置
            'EMBEDDING_MODEL': ('embedding', 'model_name'),
            'EMBEDDING_PATH': ('embedding', 'model_path'),
            'EMBEDDING_DEVICE': ('embedding', 'device'),
            
            # 监控配置
            'AI_MONITORING': ('monitoring', 'enable_monitoring', bool),
            'AI_LOG_LEVEL': ('monitoring', 'log_level'),
            
            # 服务配置
            'AI_HOST': ('service', 'host'),
            'AI_PORT': ('service', 'port', int),
            'AI_API_KEY': ('service', 'api_key'),
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.environ.get(env_var)
            if env_value is not None:
                section = config_path[0]
                key = config_path[1]
                value_type = config_path[2] if len(config_path) > 2 else str
                
                try:
                    if value_type == bool:
                        converted_value = env_value.lower() in ('1', 'true', 'yes', 'on')
                    elif value_type == int:
                        converted_value = int(env_value)
                    elif value_type == float:
                        converted_value = float(env_value)
                    else:
                        converted_value = env_value
                    
                    section_obj = getattr(self.config, section)
                    setattr(section_obj, key, converted_value)
                    print(f"[ConfigManager] Env override: {env_var}={converted_value}")
                    
                except (ValueError, TypeError) as e:
                    print(f"[ConfigManager] Failed to apply env {env_var}: {e}")
    
    def _auto_detect_models(self):
        """自动检测可用模型"""
        if self.config.model.auto_detect:
            detected = self.config.model.detect_models()
            print(f"[ConfigManager] Detected {len(detected)} models:")
            for name, meta in detected.items():
                print(f"  - {name}: {meta.size_mb}MB, {meta.recommended_use}")
            
            # 如果active_model是auto，选择第一个检测到的模型
            if self.config.model.active_model == "auto" and detected:
                # 优先选择qwen或gpt-oss
                for preferred in ["qwen", "gpt-oss"]:
                    for model_name in detected.keys():
                        if preferred in model_name.lower():
                            self.config.model.active_model = model_name
                            print(f"[ConfigManager] Auto-selected model: {model_name}")
                            return
                
                # 如果没有首选模型，选择第一个
                self.config.model.active_model = list(detected.keys())[0]
                print(f"[ConfigManager] Auto-selected model: {self.config.model.active_model}")
        
        # 自动检测嵌入模型路径
        if self.config.embedding.model_path == "auto":
            path = self.config.embedding.auto_detect_path()
            print(f"[ConfigManager] Auto-detected embedding model: {path}")
    
    def enable_hot_reload(self, enable: bool = True):
        """启用/禁用热重载"""
        self._hot_reload_enabled = enable
        
        if enable and self.observer is None:
            self.observer = Observer()
            event_handler = ConfigFileHandler(self)
            self.observer.schedule(event_handler, 
                                  path=os.path.dirname(self.config_file),
                                  recursive=False)
            self.observer.start()
            print("[ConfigManager] Hot reload enabled")
        elif not enable and self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            print("[ConfigManager] Hot reload disabled")
    
    def reload(self):
        """重新加载配置"""
        self._load_config()
        self._apply_env_overrides()
        self._auto_detect_models()
        print("[ConfigManager] Configuration reloaded")
    
    def save_config(self, file_path: Optional[str] = None):
        """保存配置到文件"""
        file_path = file_path or self.config_file
        
        # 确保目录存在
        Path(file_path).parent.mkdir(parents=True, exist_ok=True)
        
        try:
            config_dict = {
                'version': self.config.version,
                'model': asdict(self.config.model),
                'embedding': asdict(self.config.embedding),
                'monitoring': asdict(self.config.monitoring),
                'service': asdict(self.config.service),
                'metadata': {
                    'created_at': self.config.created_at,
                    'last_modified': datetime.now().isoformat()
                }
            }
            
            # 移除detected_models（运行时数据）
            if 'detected_models' in config_dict['model']:
                del config_dict['model']['detected_models']
            
            with open(file_path, 'w', encoding='utf-8') as f:
                if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                    yaml.dump(config_dict, f, default_flow_style=False, 
                             allow_unicode=True, sort_keys=False)
                else:
                    json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f"[ConfigManager] Config saved to: {file_path}")
            return True
            
        except Exception as e:
            print(f"[ConfigManager] Failed to save config: {e}")
            return False
    
    def get_model_metadata(self, model_name: str) -> Optional[ModelMetadata]:
        """获取模型元数据"""
        return self.config.model.detected_models.get(model_name)
    
    def list_available_models(self) -> List[str]:
        """列出所有可用模型"""
        return list(self.config.model.detected_models.keys())
    
    def validate_config(self) -> bool:
        """验证配置有效性"""
        errors = []
        
        # 验证模型路径
        models_dir = Path(self.config.model.models_dir)
        if not models_dir.exists():
            errors.append(f"Models directory not found: {models_dir}")
        
        # 验证活动模型
        if self.config.model.active_model not in ["auto", "simple"] + self.list_available_models():
            errors.append(f"Active model '{self.config.model.active_model}' not available")
        
        # 验证端口
        port = self.config.service.port
        if not (1024 <= port <= 65535):
            errors.append(f"Invalid port: {port}")
        
        if errors:
            print("[ConfigManager] Validation failed:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        print("[ConfigManager] Validation passed")
        return True
    
    def print_config(self):
        """打印当前配置"""
        print("=== AI Service Configuration ===")
        print(f"Version: {self.config.version}")
        print(f"Config file: {self.config_file}")
        
        print("\nModel Config:")
        model_dict = asdict(self.config.model)
        model_dict.pop('detected_models', None)  # 不打印检测到的模型详情
        for key, value in model_dict.items():
            print(f"  {key}: {value}")
        
        print(f"\nDetected Models: {len(self.config.model.detected_models)}")
        for name in self.config.model.detected_models.keys():
            print(f"  - {name}")
        
        print("\nEmbedding Config:")
        for key, value in asdict(self.config.embedding).items():
            print(f"  {key}: {value}")
        
        print("\nMonitoring Config:")
        for key, value in asdict(self.config.monitoring).items():
            print(f"  {key}: {value}")
        
        print("\nService Config:")
        for key, value in asdict(self.config.service).items():
            if key == "api_key" and value:
                value = "***hidden***"
            print(f"  {key}: {value}")
    
    def __del__(self):
        """清理资源"""
        if self.observer:
            self.observer.stop()
            self.observer.join()

# 全局配置管理器实例
_config_manager = None

def get_config() -> EnhancedConfigManager:
    """获取全局配置管理器"""
    global _config_manager
    if _config_manager is None:
        _config_manager = EnhancedConfigManager()
    return _config_manager

def create_default_config(file_path: str = None, format: str = "yaml"):
    """创建默认配置文件"""
    if file_path is None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        file_path = str(CONFIG_DIR / f"ai_service.{format}")
    
    config_mgr = EnhancedConfigManager(file_path)
    return config_mgr.save_config()

# 测试函数
def test_enhanced_config():
    """测试增强配置管理器"""
    print("=== Enhanced Config Manager Test ===")
    
    # 创建配置管理器
    config_mgr = EnhancedConfigManager()
    
    # 打印配置
    config_mgr.print_config()
    
    # 验证配置
    is_valid = config_mgr.validate_config()
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    
    # 测试热重载
    print("\nTesting hot reload...")
    config_mgr.enable_hot_reload(True)
    time.sleep(1)
    config_mgr.enable_hot_reload(False)
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_enhanced_config()



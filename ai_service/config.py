"""
AI Service Configuration Management
Unified configuration management, supporting environment variables, config files and default values
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

@dataclass
class ModelConfig:
    """Model configuration"""
    # LLM model configuration
    active_model: str = "qwen"
    supported_models: Dict[str, str] = None
    models_dir: str = str(MODELS_DIR / "gpt4all")
    force_cpu: bool = False
    
    # Generation parameters
    max_tokens: int = 800
    temperature: float = 0.3
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    max_retries: int = 3
    
    
    def __post_init__(self):
        if self.supported_models is None:
            self.supported_models = {
                "qwen3": "Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf"
            }

@dataclass
class EmbeddingConfig:
    """Embedding configuration"""
    model_path: str = str(MODELS_DIR / "all-MiniLM-L6-v2")
    embedding_dim: int = 384
    normalize: bool = True
    batch_size: int = 32
    
    # Cache configuration
    enable_cache: bool = True
    cache_max_size: int = 1000
    cache_ttl: int = 3600  # Cache TTL (seconds)
    
    # Performance configuration
    device: str = "cpu"  # cpu, cuda, auto
    preload_warmup: bool = True
    preload_texts: list = None  # Preload common texts
    
    # Concurrency configuration
    max_concurrent_requests: int = 10
    request_timeout: float = 30.0  # Request timeout (seconds)
    
    def __post_init__(self):
        if self.preload_texts is None:
            self.preload_texts = [
                "test", "hello", "example", "sample text",
                "empty text", "placeholder"
            ]
    
@dataclass
class ServiceConfig:
    """Service configuration"""
    host: str = "127.0.0.1"
    port: int = 8001
    log_level: str = "INFO"
    enable_cors: bool = True
    
@dataclass
class AIServiceConfig:
    """Complete AI service configuration"""
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
    """Configuration manager"""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or str(PROJECT_ROOT / "ai_config.json")
        self.config = AIServiceConfig()
        self._load_config()
        self._apply_env_overrides()
    
    def _load_config(self):
        """Load configuration from config file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Update configuration
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
        """Apply environment variable overrides"""
        env_mappings = {
            # Model configuration
            'AI_MODEL_KEY': ('model', 'active_model'),
            'AI_MODELS_DIR': ('model', 'models_dir'),
            'AI_FORCE_CPU': ('model', 'force_cpu', bool),
            'AI_MAX_TOKENS': ('model', 'max_tokens', int),
            'AI_TEMPERATURE': ('model', 'temperature', float),
            'AI_MAX_RETRIES': ('model', 'max_retries', int),
            
            # Embedding configuration
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
            
            # Service configuration
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
                
                # Type conversion
                try:
                    if value_type == bool:
                        converted_value = env_value.lower() in ('1', 'true', 'yes', 'on')
                    elif value_type == int:
                        converted_value = int(env_value)
                    elif value_type == float:
                        converted_value = float(env_value)
                    else:
                        converted_value = env_value
                    
                    # Set configuration value
                    section_obj = getattr(self.config, section)
                    setattr(section_obj, key, converted_value)
                    print(f"[ConfigManager] Applied env override: {env_var}={converted_value}")
                    
                except (ValueError, TypeError) as e:
                    print(f"[ConfigManager] Failed to apply env override {env_var}: {e}")
    
    def save_config(self, file_path: Optional[str] = None):
        """Save configuration to file"""
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
        """Get model configuration"""
        return self.config.model
    
    def get_embedding_config(self) -> EmbeddingConfig:
        """Get embedding configuration"""
        return self.config.embedding
    
    def get_service_config(self) -> ServiceConfig:
        """Get service configuration"""
        return self.config.service
    
    def validate_config(self) -> bool:
        """Validate configuration validity"""
        errors = []
        
        # Validate model path
        models_dir = Path(self.config.model.models_dir)
        if not models_dir.exists():
            errors.append(f"Models directory not found: {models_dir}")
        
        # Validate embedding model path
        embedding_path = Path(self.config.embedding.model_path)
        if not embedding_path.exists():
            errors.append(f"Embedding model not found: {embedding_path}")
        
        # Validate active model
        active_model = self.config.model.active_model
        if active_model not in self.config.model.supported_models:
            errors.append(f"Active model '{active_model}' not in supported models")
        
        # Validate port range
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
        """Print current configuration"""
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

# Global configuration manager instance
_config_manager = None

def get_config() -> ConfigManager:
    """Get global configuration manager"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def create_default_config(file_path: str = None):
    """Create default configuration file"""
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

# Test function
def test_config_manager():
    """Test configuration manager"""
    print("=== Config Manager Test ===")
    
    # Create configuration manager
    config_mgr = ConfigManager()
    
    # Print configuration
    config_mgr.print_config()
    
    # Validate configuration
    is_valid = config_mgr.validate_config()
    print(f"Config validation: {'PASSED' if is_valid else 'FAILED'}")
    
    # Test save configuration
    test_config_file = str(PROJECT_ROOT / "test_config.json")
    if config_mgr.save_config(test_config_file):
        print(f"Test config saved to: {test_config_file}")
        
        # Clean up test file
        try:
            os.remove(test_config_file)
            print("Test config file cleaned up")
        except:
            pass
    
    print("=== Test completed ===")

if __name__ == "__main__":
    test_config_manager()
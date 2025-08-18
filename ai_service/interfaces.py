"""
AI Service Core Interfaces
定义AI服务的核心接口和抽象类，确保模块边界清晰
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass


class ModelType(Enum):
    """支持的模型类型"""
    QWEN = "qwen"
    GPT_OSS = "gpt-oss" 
    LLAMA = "llama"


class ServiceStatus(Enum):
    """服务状态"""
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    LOADING = "loading"


@dataclass
class GenerationRequest:
    """生成请求"""
    prompt: str
    model_key: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    system_prompt: Optional[str] = None


@dataclass
class GenerationResponse:
    """生成响应"""
    text: str
    model_used: str
    tokens_generated: int
    generation_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class EmbeddingRequest:
    """嵌入请求"""
    text: Union[str, List[str]]
    normalize: bool = True


@dataclass
class EmbeddingResponse:
    """嵌入响应"""
    embeddings: Union[List[float], List[List[float]]]
    dimensions: int
    model_used: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class ILanguageModel(ABC):
    """语言模型接口"""
    
    @abstractmethod
    def load_model(self) -> bool:
        """加载模型"""
        pass
    
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """生成文本"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass
    
    @abstractmethod
    def unload_model(self) -> bool:
        """卸载模型"""
        pass


class IEmbeddingModel(ABC):
    """嵌入模型接口"""
    
    @abstractmethod
    def load_model(self) -> bool:
        """加载嵌入模型"""
        pass
    
    @abstractmethod
    def get_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """获取嵌入向量"""
        pass
    
    @abstractmethod
    def compute_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """计算相似度"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        pass


class IConfigManager(ABC):
    """配置管理接口"""
    
    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """获取模型配置"""
        pass
    
    @abstractmethod
    def get_service_config(self) -> Dict[str, Any]:
        """获取服务配置"""
        pass
    
    @abstractmethod
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """更新配置"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """验证配置"""
        pass


class IErrorHandler(ABC):
    """错误处理接口"""
    
    @abstractmethod
    def handle_error(self, error: Exception, context: str) -> Optional[str]:
        """处理错误"""
        pass
    
    @abstractmethod
    def should_retry(self, error: Exception) -> bool:
        """判断是否应该重试"""
        pass
    
    @abstractmethod
    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计"""
        pass


class IMonitor(ABC):
    """监控接口"""
    
    @abstractmethod
    def record_generation(self, request: GenerationRequest, response: GenerationResponse) -> None:
        """记录生成指标"""
        pass
    
    @abstractmethod
    def record_embedding(self, request: EmbeddingRequest, response: EmbeddingResponse) -> None:
        """记录嵌入指标"""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        pass


class IAIService(ABC):
    """AI服务主接口"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """初始化服务"""
        pass
    
    @abstractmethod
    def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """生成文本"""
        pass
    
    @abstractmethod
    def get_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """获取嵌入"""
        pass
    
    @abstractmethod
    def set_active_model(self, model_key: str) -> bool:
        """设置活动模型"""
        pass
    
    @abstractmethod
    def get_active_model(self) -> str:
        """获取活动模型"""
        pass
    
    @abstractmethod
    def get_service_status(self) -> ServiceStatus:
        """获取服务状态"""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """关闭服务"""
        pass


# 工厂函数类型定义
ModelFactory = Dict[str, type]
ServiceFactory = Dict[str, type]

# 常量定义
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPEAT_PENALTY = 1.1

# 错误类型
class AIServiceError(Exception):
    """AI服务基础错误"""
    pass


class ModelLoadError(AIServiceError):
    """模型加载错误"""
    pass


class GenerationError(AIServiceError):
    """生成错误"""
    pass


class EmbeddingError(AIServiceError):
    """嵌入错误"""
    pass


class ConfigurationError(AIServiceError):
    """配置错误"""
    pass
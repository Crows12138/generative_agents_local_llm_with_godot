"""
AI Service Core Interfaces
Define core interfaces and abstract classes for AI services, ensuring clear module boundaries
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass


class ModelType(Enum):
    """Supported model types"""
    QWEN = "qwen"
 
    LLAMA = "llama"


class ServiceStatus(Enum):
    """Service status"""
    INITIALIZING = "initializing"
    READY = "ready"
    ERROR = "error"
    LOADING = "loading"


@dataclass
class GenerationRequest:
    """Generation request"""
    prompt: str
    model_key: Optional[str] = None
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    repeat_penalty: Optional[float] = None
    system_prompt: Optional[str] = None


@dataclass
class GenerationResponse:
    """Generation response"""
    text: str
    model_used: str
    tokens_generated: int
    generation_time: float
    success: bool
    error_message: Optional[str] = None


@dataclass
class EmbeddingRequest:
    """Embedding request"""
    text: Union[str, List[str]]
    normalize: bool = True


@dataclass
class EmbeddingResponse:
    """Embedding response"""
    embeddings: Union[List[float], List[List[float]]]
    dimensions: int
    model_used: str
    processing_time: float
    success: bool
    error_message: Optional[str] = None


class ILanguageModel(ABC):
    """Language model interface"""
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load model"""
        pass
    
    @abstractmethod
    def generate(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass
    
    @abstractmethod
    def unload_model(self) -> bool:
        """Unload model"""
        pass


class IEmbeddingModel(ABC):
    """Embedding model interface"""
    
    @abstractmethod
    def load_model(self) -> bool:
        """Load embedding model"""
        pass
    
    @abstractmethod
    def get_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Get embedding vector"""
        pass
    
    @abstractmethod
    def compute_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Compute similarity"""
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """Check if model is loaded"""
        pass


class IConfigManager(ABC):
    """Configuration management interface"""
    
    @abstractmethod
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        pass
    
    @abstractmethod
    def get_service_config(self) -> Dict[str, Any]:
        """Get service configuration"""
        pass
    
    @abstractmethod
    def update_config(self, section: str, key: str, value: Any) -> bool:
        """Update configuration"""
        pass
    
    @abstractmethod
    def validate_config(self) -> bool:
        """Validate configuration"""
        pass


class IErrorHandler(ABC):
    """Error handling interface"""
    
    @abstractmethod
    def handle_error(self, error: Exception, context: str) -> Optional[str]:
        """Handle error"""
        pass
    
    @abstractmethod
    def should_retry(self, error: Exception) -> bool:
        """Determine if should retry"""
        pass
    
    @abstractmethod
    def get_error_stats(self) -> Dict[str, Any]:
        """Get error statistics"""
        pass


class IMonitor(ABC):
    """Monitoring interface"""
    
    @abstractmethod
    def record_generation(self, request: GenerationRequest, response: GenerationResponse) -> None:
        """Record generation metrics"""
        pass
    
    @abstractmethod
    def record_embedding(self, request: EmbeddingRequest, response: EmbeddingResponse) -> None:
        """Record embedding metrics"""
        pass
    
    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        pass
    
    @abstractmethod
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        pass


class IAIService(ABC):
    """Main AI service interface"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize service"""
        pass
    
    @abstractmethod
    def generate_text(self, request: GenerationRequest) -> GenerationResponse:
        """Generate text"""
        pass
    
    @abstractmethod
    def get_embedding(self, request: EmbeddingRequest) -> EmbeddingResponse:
        """Get embedding"""
        pass
    
    @abstractmethod
    def set_active_model(self, model_key: str) -> bool:
        """Set active model"""
        pass
    
    @abstractmethod
    def get_active_model(self) -> str:
        """Get active model"""
        pass
    
    @abstractmethod
    def get_service_status(self) -> ServiceStatus:
        """Get service status"""
        pass
    
    @abstractmethod
    def shutdown(self) -> bool:
        """Shutdown service"""
        pass


# Factory function type definitions
ModelFactory = Dict[str, type]
ServiceFactory = Dict[str, type]

# Constant definitions
DEFAULT_MAX_TOKENS = 500
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_REPEAT_PENALTY = 1.1

# Error types
class AIServiceError(Exception):
    """Base AI service error"""
    pass


class ModelLoadError(AIServiceError):
    """Model loading error"""
    pass


class GenerationError(AIServiceError):
    """Generation error"""
    pass


class EmbeddingError(AIServiceError):
    """Embedding error"""
    pass


class ConfigurationError(AIServiceError):
    """Configuration error"""
    pass
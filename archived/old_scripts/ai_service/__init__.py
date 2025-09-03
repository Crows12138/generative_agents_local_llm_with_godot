# AI Service module for local LLM integration with optimizations
from .ai_service import (
    AIService, get_ai_service, local_llm_generate, 
    set_active_model, get_active_model, warmup_model, 
    cleanup_memory, get_model_health, get_memory_usage, 
    check_memory_pressure
)

__all__ = [
    'AIService', 'get_ai_service', 'local_llm_generate',
    'set_active_model', 'get_active_model', 'warmup_model',
    'cleanup_memory', 'get_model_health', 'get_memory_usage',
    'check_memory_pressure'
]

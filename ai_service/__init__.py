# AI Service module for local LLM integration
from .ai_service import local_llm_generate, set_active_model, get_active_model

__all__ = ['local_llm_generate', 'set_active_model', 'get_active_model']

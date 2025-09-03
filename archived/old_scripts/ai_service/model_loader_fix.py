"""Fixed model loader based on monitoring insights"""

import os
import logging
from pathlib import Path
import torch

class FixedModelLoader:
    """Model loader with fixes from monitoring insights"""
    
    def __init__(self):
        self.logger = logging.getLogger("model_loader")
        self.models = {}
        self.model_configs = {
            "qwen3": {
                "model_name": "llms/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf",  # Updated path
                "n_ctx": 4096,  # Match actual context limit
                "n_batch": 512,
                "device": "cpu",  # Or "cuda" if available
                "n_threads": 8
            },
            "llama": {
                "model_name": "Meta-Llama-3-8B-Instruct.Q4_0.gguf",
                "n_ctx": 4096,
                "n_batch": 512,
                "device": "cpu",
                "n_threads": 8
            }
        }
    
    def load_model(self, model_key: str):
        """Load model with proper configuration"""
        
        if model_key in self.models:
            self.logger.info(f"Model {model_key} already loaded")
            return self.models[model_key]
        
        config = self.model_configs.get(model_key)
        if not config:
            raise ValueError(f"Unknown model: {model_key}")
        
        # Check model file exists - use actual path structure
        model_path = Path("models") / config["model_name"]
        if not model_path.exists():
            self.logger.error(f"Model file not found: {model_path}")
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        try:
            # Use existing AI service infrastructure
            from ai_service.ai_service import get_ai_service
            
            self.logger.info(f"Loading {model_key} using existing AI service...")
            
            # Get the AI service and ensure model is loaded
            ai_service = get_ai_service()
            
            # Test the service
            from ai_service.ai_service import local_llm_generate
            test_response = local_llm_generate("Hello", model_key=model_key)
            
            if test_response and "LocalLLM ERROR" not in test_response:
                self.models[model_key] = ai_service
                self.logger.info(f"✅ Model {model_key} loaded successfully")
                return ai_service
            else:
                raise Exception(f"Model test failed: {test_response}")
            
        except Exception as e:
            self.logger.error(f"Failed to load {model_key}: {e}")
            raise
    
    def ensure_model_ready(self, model_key: str) -> bool:
        """Ensure model is loaded and ready"""
        try:
            model = self.load_model(model_key)
            # Quick health check using existing infrastructure
            from ai_service.ai_service import local_llm_generate
            response = local_llm_generate("test", model_key=model_key)
            return response and len(response) > 0 and "LocalLLM ERROR" not in response
        except Exception as e:
            self.logger.error(f"Model health check failed: {e}")
            return False

# Global loader instance
model_loader = FixedModelLoader()
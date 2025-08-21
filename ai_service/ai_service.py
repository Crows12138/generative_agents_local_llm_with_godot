"""
Unified AI Service - Complete AI functionality with optimizations
Combines original functions with optimization features in one self-contained file
"""

from __future__ import annotations

import os
import time
import random
import threading
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime

import requests
from fastapi import FastAPI
from pydantic import BaseModel

# LLM backend
try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    print("[ai_service] llama-cpp-python not available, falling back to simple responses")

# Import dependencies
try:
    from .config_enhanced import get_config
    from .monitoring import get_performance_monitor, timing_context, get_ai_logger
    from .error_handling import handle_generation_errors, RetryHandler, GENERATION_RETRY_CONFIG
except ImportError:
    from config_enhanced import get_config
    from monitoring import get_performance_monitor, timing_context, get_ai_logger
    from error_handling import handle_generation_errors, RetryHandler, GENERATION_RETRY_CONFIG

# Import optimization components
import sys
# Add both relative and absolute paths for debug_system
debug_system_path = os.path.join(os.path.dirname(__file__), '..', 'debug_system')
debug_system_abs_path = os.path.abspath(debug_system_path)
sys.path.insert(0, debug_system_abs_path)
sys.path.insert(0, debug_system_path)

try:
    # Try direct imports from debug_system directory
    import importlib.util
    
    # Import llm_optimizer
    llm_optimizer_path = os.path.join(debug_system_abs_path, 'llm_optimizer.py')
    spec = importlib.util.spec_from_file_location("llm_optimizer", llm_optimizer_path)
    llm_optimizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(llm_optimizer)
    SmartLLMClient = llm_optimizer.SmartLLMClient
    LLMCache = llm_optimizer.LLMCache
    PromptOptimizer = llm_optimizer.PromptOptimizer
    
    # Import flow_tracer
    flow_tracer_path = os.path.join(debug_system_abs_path, 'flow_tracer.py')
    spec = importlib.util.spec_from_file_location("flow_tracer", flow_tracer_path)
    flow_tracer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(flow_tracer)
    get_tracer = flow_tracer.get_tracer
    
    # Import performance_analyzer
    perf_analyzer_path = os.path.join(debug_system_abs_path, 'performance_analyzer.py')
    spec = importlib.util.spec_from_file_location("performance_analyzer", perf_analyzer_path)
    perf_analyzer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(perf_analyzer)
    get_performance_analyzer = perf_analyzer.get_performance_analyzer
    
    OPTIMIZATION_AVAILABLE = True
    print("[ai_service] Optimization components loaded via direct import")
except Exception as e:
    OPTIMIZATION_AVAILABLE = False
    print(f"[ai_service] Optimization components not available: {e}")
    print(f"[ai_service] Debug system path: {debug_system_abs_path}")
    print(f"[ai_service] Path exists: {os.path.exists(debug_system_abs_path)}")


# =============================================================================
# ORIGINAL AI SERVICE FUNCTIONS (from ai_service_original_backup.py)
# =============================================================================

# Global state
_model_instances = {}
_active_model = None
_model_config = None
_model_health = {}

def _initialize_config():
    """Initialize model configuration"""
    global _model_config, _active_model
    if _model_config is None:
        config = get_config()
        # Get available models
        available_models = config.list_available_models()
        if available_models:
            first_model_key = available_models[0]
            _active_model = first_model_key
            # Create a simple config object
            _model_config = type('Config', (), {
                'model_path': f'models/{first_model_key}.gguf',
                'context_length': 4096,
                'threads': 4
            })()
        else:
            # Fallback configuration
            _active_model = "qwen3"
            _model_config = type('Config', (), {
                'model_path': 'models/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf',
                'context_length': 4096,
                'threads': 4
            })()

def set_active_model(model_key: str) -> bool:
    """Set the active model"""
    global _active_model
    _initialize_config()
    config = get_config()
    
    available_models = config.list_available_models()
    if model_key.lower() in [k.lower() for k in available_models]:
        _active_model = model_key.lower()
        return True
    return False

def get_active_model() -> str:
    """Get the active model"""
    global _active_model
    _initialize_config()
    return _active_model or "qwen3"

def _get_model_instance(model_key: str) -> Llama:
    """Get or create model instance"""
    global _model_instances
    _initialize_config()
    
    if model_key not in _model_instances:
        config = get_config()
        
        # Check if model is available
        available_models = config.list_available_models()
        model_found = False
        for key in available_models:
            if key.lower() == model_key.lower():
                model_found = True
                break
        
        if not model_found:
            raise ValueError(f"Model {model_key} not found in configuration")
        
        # Create model config
        model_config = type('Config', (), {
            'model_path': f'models/{model_key}.gguf',
            'context_length': 4096,
            'threads': 4
        })()
        
        if not LLAMACPP_AVAILABLE:
            raise ImportError("llama-cpp-python not available")
        
        # Create model instance
        _model_instances[model_key] = Llama(
            model_path=str(model_config.model_path),
            n_ctx=getattr(model_config, 'context_length', 4096),
            n_threads=getattr(model_config, 'threads', 4),
            verbose=False
        )
    
    return _model_instances[model_key]

def _format_prompt(prompt: str, model_key: str) -> str:
    """Format prompt for specific model"""
    if model_key in ("qwen", "qwen3"):
        return f"<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n"
    else:
        return f"User: {prompt}\\nAssistant: "

def _clean_response(raw: str, model_key: str) -> str:
    """Clean model response"""
    cleaned = raw.strip()
    
    # Remove common artifacts
    if model_key in ("qwen", "qwen3"):
        cleaned = cleaned.replace("<|im_end|>", "").replace("<|im_start|>", "")
    
    # Remove extra whitespace
    cleaned = " ".join(cleaned.split())
    
    return cleaned

def _core_llm_generate(prompt: str, model_key: Optional[str] = None) -> str:
    """Core LLM generation function"""
    mk = (model_key or get_active_model()).lower()
    
    try:
        ai = _get_model_instance(mk)
        formatted_prompt = _format_prompt(prompt, mk)
        
        # Generate response
        output = ai(
            formatted_prompt,
            max_tokens=512,
            temperature=0.7,
            top_p=0.9,
            repeat_penalty=1.1,
            stop=["<|im_end|>", "<|im_start|>"] if mk in ("qwen", "qwen3") else ["<|end|>"],
            echo=False
        )
        
        tokens = output['choices'][0]['text']
        cleaned = _clean_response(tokens, mk)
        
        return cleaned if cleaned else "I need more information to help you."
        
    except Exception as e:
        print(f"[ai_service] Generation error: {e}")
        return _get_simple_response(prompt)

def _get_simple_response(prompt: str) -> str:
    """Fallback simple response"""
    responses = [
        "I understand. Let me help you with that.",
        "That's a good question. Here's what I think...",
        "I can help you with that task.",
        "Let me consider the best approach for this.",
        "That sounds like something I can assist with."
    ]
    return random.choice(responses)

def warmup_model(model_key: str) -> bool:
    """Warm up a model"""
    try:
        _get_model_instance(model_key)
        return True
    except Exception as e:
        print(f"[ai_service] Warmup failed for {model_key}: {e}")
        return False

def get_model_health(model_key: str) -> Dict[str, Any]:
    """Get model health status"""
    global _model_health
    return _model_health.get(model_key, {"status": "unknown", "last_check": None})

def get_memory_usage() -> Dict[str, Any]:
    """Get memory usage information"""
    import psutil
    process = psutil.Process()
    memory_info = process.memory_info()
    
    return {
        "rss_mb": memory_info.rss / 1024 / 1024,
        "vms_mb": memory_info.vms / 1024 / 1024,
        "total_mb": memory_info.rss / 1024 / 1024,
        "percent": process.memory_percent()
    }

def check_memory_pressure() -> Dict[str, Any]:
    """Check memory pressure"""
    import psutil
    memory = psutil.virtual_memory()
    
    return {
        "total_gb": memory.total / 1024**3,
        "available_gb": memory.available / 1024**3,
        "used_percent": memory.percent,
        "pressure_level": "high" if memory.percent > 80 else "normal"
    }

def cleanup_memory():
    """Clean up memory"""
    import gc
    gc.collect()
    print("[ai_service] Memory cleanup performed")


# =============================================================================
# UNIFIED AI SERVICE CLASS WITH OPTIMIZATIONS
# =============================================================================

class AIService:
    """
    Unified AI Service combining original functionality with optimizations
    """
    
    def __init__(self, enable_optimizations: bool = True):
        """Initialize the AI service with optional optimizations"""
        
        # Initialize core configuration
        _initialize_config()
        
        # Core service setup
        self.config = get_config()
        self.performance_monitor = get_performance_monitor()
        
        # Optimization setup
        self.optimizations_enabled = enable_optimizations and OPTIMIZATION_AVAILABLE
        
        if self.optimizations_enabled:
            self._setup_optimizations()
        
        # Service state
        self._initialized = True
        
        print(f"[AIService] Initialized with optimizations: {self.optimizations_enabled}")
    
    def _setup_optimizations(self) -> None:
        """Setup optimization components"""
        
        # Initialize smart LLM client
        self.smart_llm = SmartLLMClient(
            llm_function=self._core_generate,
            cache_size=10000
        )
        
        # Setup flow tracing
        self.flow_tracer = get_tracer()
        
        # Setup performance monitoring
        self.perf_analyzer = get_performance_analyzer()
        self.perf_analyzer.start_monitoring(interval=2.0)
        
        # Optimization settings
        self.use_llm_cache = True
        self.use_prompt_optimization = True
        self.use_flow_tracing = True
        self.use_performance_monitoring = True
        
        print("[AIService] Optimizations initialized")
    
    def _core_generate(self, prompt: str, model_key: Optional[str] = None) -> str:
        """Core generation using original functions"""
        return _core_llm_generate(prompt, model_key)
    
    def generate(self, prompt: str, use_optimizations: bool = None, **kwargs) -> str:
        """
        Generate response with optional optimizations
        """
        
        if use_optimizations is None:
            use_optimizations = self.optimizations_enabled
        
        # Use optimized path if available
        if use_optimizations and self.optimizations_enabled:
            return self._optimized_generate(prompt, **kwargs)
        else:
            return self._core_generate(prompt, kwargs.get('model_key'))
    
    def _optimized_generate(self, prompt: str, **kwargs) -> str:
        """Optimized generation with caching and monitoring"""
        
        # Start flow tracing
        if self.use_flow_tracing:
            agent_name = kwargs.get('agent_name', 'unknown')
            self.flow_tracer.trace_llm_prompt(prompt)
        
        # Start performance monitoring
        if self.use_performance_monitoring:
            self.perf_analyzer.start_measurement("llm_generation")
        
        try:
            # Use smart LLM client with optimizations
            response = self.smart_llm.call(
                prompt,
                use_cache=self.use_llm_cache,
                optimize_prompt=self.use_prompt_optimization
            )
            
            # Complete flow tracing
            if self.use_flow_tracing:
                self.flow_tracer.trace_llm_response(response)
            
            return response
            
        except Exception as e:
            print(f"[AIService] Optimized generation error: {e}")
            # Fallback to core method
            return self._core_generate(prompt, kwargs.get('model_key'))
        
        finally:
            # End performance monitoring
            if self.use_performance_monitoring:
                self.perf_analyzer.end_measurement("llm_generation")
    
    def batch_generate(self, prompts: List[str], callbacks: Optional[List[Callable]] = None) -> List[str]:
        """Batch generation with optimization"""
        
        if not self.optimizations_enabled:
            # Fallback: process sequentially
            return [self._core_generate(prompt) for prompt in prompts]
        
        results = []
        
        def collect_result(response: str, metadata: Dict[str, Any]) -> None:
            results.append(response)
        
        # Use async batch processing
        for i, prompt in enumerate(prompts):
            callback = callbacks[i] if callbacks and i < len(callbacks) else collect_result
            self.smart_llm.call_async(prompt, callback)
        
        # Wait for all results
        self.smart_llm.flush_batches()
        time.sleep(0.1)  # Allow async processing to complete
        
        return results
    
    # Expose original functions as methods
    def set_active_model(self, model_key: str) -> bool:
        """Set the active model"""
        return set_active_model(model_key)
    
    def get_active_model(self) -> str:
        """Get the active model"""
        return get_active_model()
    
    def warmup_model(self, model_key: str) -> bool:
        """Warm up a model"""
        return warmup_model(model_key)
    
    def cleanup_memory(self) -> None:
        """Clean up memory"""
        cleanup_memory()
    
    def get_model_health(self, model_key: str) -> Dict[str, Any]:
        """Get model health status"""
        return get_model_health(model_key)
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage information"""
        return get_memory_usage()
    
    def check_memory_pressure(self) -> Dict[str, Any]:
        """Check memory pressure"""
        return check_memory_pressure()
    
    # Optimization-specific methods
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        
        if not self.optimizations_enabled:
            return {"error": "Optimizations not available"}
        
        stats = {
            "llm_stats": self.smart_llm.get_stats(),
            "performance_stats": self.perf_analyzer.get_summary(),
            "flow_stats": self.flow_tracer.get_summary()
        }
        
        return stats
    
    def optimize_for_scenario(self, scenario: str) -> None:
        """Optimize settings for specific scenarios"""
        
        if not self.optimizations_enabled:
            return
        
        if scenario == "high_throughput":
            self.use_llm_cache = True
            self.use_prompt_optimization = True
            self.use_flow_tracing = False  # Reduce overhead
            
        elif scenario == "low_latency":
            self.use_llm_cache = True
            self.use_prompt_optimization = True
            self.use_performance_monitoring = False
            
        elif scenario == "debugging":
            self.use_flow_tracing = True
            self.use_performance_monitoring = True
            self.use_llm_cache = False  # Avoid cache for debugging
            
        elif scenario == "production":
            self.use_llm_cache = True
            self.use_prompt_optimization = True
            self.use_flow_tracing = False
            self.use_performance_monitoring = True
        
        print(f"[AIService] Optimized for {scenario} scenario")
    
    def clear_optimizations(self) -> None:
        """Clear optimization caches and reset"""
        
        if not self.optimizations_enabled:
            return
        
        self.smart_llm.clear_cache()
        self.flow_tracer.clear_trace()
        self.perf_analyzer.clear_data()
        
        print("[AIService] Optimization data cleared")
    
    def __del__(self):
        """Cleanup on destruction"""
        if self.optimizations_enabled and hasattr(self, 'perf_analyzer'):
            self.perf_analyzer.stop_monitoring()


# =============================================================================
# GLOBAL SERVICE AND COMPATIBILITY FUNCTIONS
# =============================================================================

# Global service instance
_ai_service = None

def get_ai_service(enable_optimizations: bool = True) -> AIService:
    """Get global AI service instance"""
    global _ai_service
    
    if _ai_service is None:
        _ai_service = AIService(enable_optimizations=enable_optimizations)
    
    return _ai_service

# Legacy compatibility functions
def local_llm_generate(prompt: str, model_key: Optional[str] = None) -> str:
    """Legacy compatibility function - delegates to service"""
    return _core_llm_generate(prompt, model_key)

# Export everything
__all__ = [
    'AIService',
    'get_ai_service', 
    'local_llm_generate',
    'set_active_model',
    'get_active_model',
    'warmup_model',
    'cleanup_memory',
    'get_model_health',
    'get_memory_usage',
    'check_memory_pressure'
]
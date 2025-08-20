"""Local LLM service wrapper used by planners and other modules.

Adds switchable backends for local models and proper prompt formatting:
 - qwen3: Qwen3-30B chat template

Use local GGUF files via GPT4All and Transformers for generative agents.
"""

from __future__ import annotations

import os
import time
import random
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from gpt4all import GPT4All
from importlib import import_module

# Import configuration manager, error handling and monitoring
try:
    from .config_enhanced import get_config
    from .error_handling import (
        handle_model_errors, handle_generation_errors, 
        RetryHandler, GENERATION_RETRY_CONFIG, MODEL_RETRY_CONFIG,
        get_error_monitor, ErrorType, RetryableError, NonRetryableError
    )
    from .monitoring import (
        get_performance_monitor, get_ai_logger, timing_context,
        monitor_performance, get_health_status, get_metrics_json
    )
except ImportError:
    from config_enhanced import get_config
    from error_handling import (
        handle_model_errors, handle_generation_errors,
        RetryHandler, GENERATION_RETRY_CONFIG, MODEL_RETRY_CONFIG,
        get_error_monitor, ErrorType, RetryableError, NonRetryableError
    )
    from monitoring import (
        get_performance_monitor, get_ai_logger, timing_context,
        monitor_performance, get_health_status, get_metrics_json
    )

# Get configuration
_config_manager = get_config()
_config = _config_manager.config
_model_config = _config.model
_service_config = _config.service

app = FastAPI(
    title="AI Service API",
    description="Enhanced AI service with model fallback and memory management",
    version="1.0.0"
)
_BASE_URL = f"http://{_service_config.host}:{_service_config.port}"

# API version prefix
API_V1_PREFIX = "/v1"

# General error handling decorator
def handle_api_errors(func):
    """General API error handling decorator"""
    import functools
    
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        import uuid
        from fastapi import HTTPException
        
        request_id = str(uuid.uuid4())
        start_time = datetime.now()
        logger = get_ai_logger()
        
        try:
            return await func(*args, **kwargs)
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            logger.error(f"API error [{request_id}]: {str(e)}")
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            error_response = ErrorResponse(
                request_id=request_id,
                timestamp=datetime.now().isoformat(),
                processing_time_ms=processing_time,
                error_code="INTERNAL_ERROR",
                error_message=str(e),
                error_details={"function": func.__name__}
            )
            
            raise HTTPException(status_code=500, detail=error_response.dict())
    
    return wrapper

# Use paths and models from configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(_model_config.models_dir)

# Supported models from configuration
SUPPORTED_GGUF: Dict[str, str] = _model_config.supported_models

# Active model from configuration
_ACTIVE_MODEL_KEY: str = _model_config.active_model.lower()

# Cache of GPT4All instances by model key
_MODEL_INSTANCES: Dict[str, GPT4All] = {}


def set_active_model(model_key: str) -> bool:
    """Set the globally active model key (qwen)."""
    global _ACTIVE_MODEL_KEY
    if model_key not in SUPPORTED_GGUF:
        print(f"[ai_service] Unknown model key '{model_key}'. Supported: {list(SUPPORTED_GGUF.keys())}")
        return False
    _ACTIVE_MODEL_KEY = model_key
    print(f"[ai_service] Active model set to: {_ACTIVE_MODEL_KEY}")
    return True


def get_active_model() -> str:
    """Return the current active model key."""
    return _ACTIVE_MODEL_KEY


@handle_model_errors
def _get_model_instance(model_key: str) -> GPT4All:
    """Get or lazily create a GPT4All instance for the given key."""
    model_key = model_key.lower()
    if model_key not in SUPPORTED_GGUF:
        print(f"[ai_service] Unsupported model key '{model_key}', falling back to 'qwen'")
        model_key = "qwen"

    # Model loading setup

    if model_key in _MODEL_INSTANCES:
        return _MODEL_INSTANCES[model_key]

    # Check memory pressure and clean up if necessary
    pressure = check_memory_pressure()
    if pressure["pressure_level"] in ["warning", "critical"]:
        logger = get_ai_logger()
        logger.warning(f"Memory pressure detected before loading {model_key}: {pressure['pressure_level']}")
        auto_memory_management()

    model_file = SUPPORTED_GGUF[model_key]
    model_path = MODELS_DIR / model_file
    print(f"[ai_service] Loading model '{model_key}' from: {model_path}")
    print(f"[ai_service] Model file exists: {model_path.exists()}")
    
    if not model_path.exists():
        raise NonRetryableError(
            f"Model file not found: {model_path}",
            ErrorType.CONFIGURATION_ERROR
        )

    # Resolve desired device. Force CPU for llama to avoid Kompute/Vulkan crashes.
    force_cpu_env = _model_config.force_cpu
    desired_device = "cpu" if (force_cpu_env or model_key in ("llama",)) else "gpu"
    print(f"[ai_service] Desired device for '{model_key}': {desired_device}{' (forced by env)' if force_cpu_env else ''}")

    # Create instance; prefer desired device but fallback to CPU on failure
    try:
        ai = GPT4All(
            model_name=model_file,
            model_path=str(MODELS_DIR),
            allow_download=False,
            device=desired_device,
        )
    except Exception as dev_error:
        if desired_device != "cpu":
            print(f"[ai_service] Device '{desired_device}' init failed for '{model_key}': {dev_error}; falling back to CPU")
            ai = GPT4All(
                model_name=model_file,
                model_path=str(MODELS_DIR),
                allow_download=False,
                device="cpu",
            )
        else:
            raise
    _MODEL_INSTANCES[model_key] = ai
    return ai

# Unified API request/response models
class BaseRequest(BaseModel):
    """Base request model"""
    request_id: Optional[str] = None
    timestamp: Optional[str] = None
    version: str = "v1"

class BaseResponse(BaseModel):
    """Base response model"""
    request_id: Optional[str] = None
    timestamp: str
    version: str = "v1"
    processing_time_ms: Optional[float] = None
    status: str = "success"

class ChatRequest(BaseRequest):
    model_key: Optional[str] = None
    prompt: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseResponse):
    response: str
    model_used: str
    fallback_used: bool = False
    metadata: Optional[Dict[str, Any]] = None

class ErrorResponse(BaseResponse):
    """Error response model"""
    status: str = "error"
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None

# Backward compatible endpoints
@app.post("/chat", response_model=ChatResponse)
@handle_api_errors
async def chat_legacy(req: ChatRequest):
    """Legacy chat endpoint for backward compatibility."""
    return await chat_v1(req)

# V1 API endpoints
@app.post(f"{API_V1_PREFIX}/chat", response_model=ChatResponse)
@handle_api_errors
async def chat_v1(req: ChatRequest):
    """Expose a /chat endpoint using the requested model (or active default)."""
    import uuid
    
    # Generate request ID and timestamp
    request_id = req.request_id or str(uuid.uuid4())
    start_time = datetime.now()
    
    logger = get_ai_logger()
    logger.info(f"Chat request [{request_id}]: model={req.model_key}, prompt_length={len(req.prompt)}")
    
    try:
        # Save original fallback setting status
        original_fallback_enabled = _model_config.enable_fallback
        fallback_used = False
        
        # If model parameter is specified, use it temporarily
        original_active_model = get_active_model()
        if req.model_key and req.model_key != original_active_model:
            set_active_model(req.model_key)
        
        # Call generation service
        response_text = local_llm_generate(
            req.prompt, 
            model_key=req.model_key,
            max_retries=_model_config.max_retries
        )
        
        # Check if fallback was used
        if not original_fallback_enabled and _model_config.enable_fallback:
            fallback_used = True
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Determine the actual model used
        model_used = req.model_key or get_active_model()
        
        return ChatResponse(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
            response=response_text,
            model_used=model_used,
            fallback_used=fallback_used,
            metadata={
                "prompt_length": len(req.prompt),
                "response_length": len(response_text),
                "max_tokens": req.max_tokens or _model_config.max_tokens,
                "temperature": req.temperature or _model_config.temperature
            }
        )
        
    except Exception as e:
        logger.error(f"Chat request [{request_id}] failed: {str(e)}")
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Return error response
        error_response = ErrorResponse(
            request_id=request_id,
            timestamp=datetime.now().isoformat(),
            processing_time_ms=processing_time,
            error_code="GENERATION_FAILED",
            error_message=str(e),
            error_details={
                "model_key": req.model_key,
                "prompt_length": len(req.prompt)
            }
        )
        
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=error_response.dict())

# Health and Metrics endpoints (compatible version)
@app.get("/health")
@handle_api_errors
async def health():
    """Health check endpoint with performance metrics."""
    return get_health_status()

@app.get("/metrics")
@handle_api_errors
async def metrics():
    """Performance metrics endpoint."""
    return {"metrics": get_metrics_json()}

@app.get("/metrics/json")
@handle_api_errors
async def metrics_json():
    """Raw JSON metrics endpoint."""
    from fastapi.responses import PlainTextResponse
    return PlainTextResponse(get_metrics_json(), media_type="application/json")

# V1 Health and Metrics endpoints
@app.get(f"{API_V1_PREFIX}/health")
@handle_api_errors
async def health_v1():
    """V1 Health check endpoint with enhanced metrics."""
    health_data = get_health_status()
    memory_data = get_memory_usage()
    
    return {
        **health_data,
        "memory": memory_data,
        "models": {
            "loaded": list(_MODEL_INSTANCES.keys()),
            "health": _MODEL_HEALTH,
            "active": get_active_model()
        },
        "api_version": "v1",
        "timestamp": datetime.now().isoformat()
    }

@app.get(f"{API_V1_PREFIX}/models/health")
@handle_api_errors
async def models_health_v1():
    """Get comprehensive health status of all models."""
    return {
        "models": _MODEL_HEALTH,
        "active_model": get_active_model(),
        "fallback_enabled": _model_config.enable_fallback,
        "fallback_chain": _model_config.fallback_chain,
        "memory_usage": get_memory_usage(),
        "api_version": "v1",
        "timestamp": datetime.now().isoformat()
    }

# Backward compatibility
@app.get("/models/health")
@handle_api_errors
async def models_health():
    """Legacy models health endpoint."""
    return await models_health_v1()

@app.post(f"{API_V1_PREFIX}/models/warmup")
@handle_api_errors
async def trigger_warmup_v1(model_key: Optional[str] = None):
    """V1 Trigger model warmup."""
    return await _warmup_logic(model_key)

@app.post("/models/warmup")
@handle_api_errors
async def trigger_warmup(model_key: Optional[str] = None):
    """Legacy warmup endpoint."""
    return await _warmup_logic(model_key)

async def _warmup_logic(model_key: Optional[str] = None):
    """Trigger model warmup."""
    logger = get_ai_logger()
    
    if model_key:
        # Warm up specified model
        logger.info(f"Triggering warmup for model: {model_key}")
        success = warmup_model(model_key)
        return {
            "model": model_key,
            "success": success,
            "message": f"Model {model_key} warmup {'succeeded' if success else 'failed'}",
            "timestamp": datetime.now().isoformat()
        }
    else:
        # Warm up all models
        logger.info("Triggering warmup for all models")
        results = warmup_all_models()
        successful_count = sum(1 for success in results.values() if success)
        return {
            "results": results,
            "total_models": len(results),
            "successful": successful_count,
            "message": f"Warmup completed: {successful_count}/{len(results)} models ready",
            "timestamp": datetime.now().isoformat()
        }

# Memory management endpoints (V1)
@app.get(f"{API_V1_PREFIX}/memory")
@handle_api_errors
async def memory_status_v1():
    """V1 Get current memory usage and status."""
    memory_data = get_memory_usage()
    pressure_data = check_memory_pressure()
    
    return {
        **memory_data,
        "pressure": pressure_data,
        "api_version": "v1",
        "timestamp": datetime.now().isoformat()
    }

@app.get(f"{API_V1_PREFIX}/memory/pressure")
@handle_api_errors
async def memory_pressure_v1():
    """V1 Check memory pressure and get recommendations."""
    return check_memory_pressure()

@app.post(f"{API_V1_PREFIX}/memory/cleanup")
@handle_api_errors
async def trigger_memory_cleanup_v1():
    """V1 Trigger memory cleanup."""
    logger = get_ai_logger()
    logger.info("Manual memory cleanup triggered")
    
    memory_before = get_memory_usage()
    memory_after = cleanup_memory()
    
    return {
        "status": "completed",
        "memory_before": memory_before,
        "memory_after": memory_after,
        "freed_mb": round(memory_before["process"]["rss_mb"] - memory_after["process"]["rss_mb"], 2),
        "api_version": "v1",
        "timestamp": datetime.now().isoformat()
    }

@app.delete(f"{API_V1_PREFIX}/models/{{model_key}}")
@handle_api_errors
async def unload_model_v1(model_key: str):
    """V1 Unload a specific model from memory."""
    logger = get_ai_logger()
    
    if model_key in _MODEL_INSTANCES:
        unload_model(model_key)
        return {
            "status": "success",
            "message": f"Model {model_key} unloaded successfully",
            "remaining_models": list(_MODEL_INSTANCES.keys()),
            "api_version": "v1",
            "timestamp": datetime.now().isoformat()
        }
    else:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail={
                "status": "error",
                "message": f"Model {model_key} not found in cache",
                "available_models": list(_MODEL_INSTANCES.keys()),
                "api_version": "v1",
                "timestamp": datetime.now().isoformat()
            }
        )

# Backward compatible endpoints
@app.get("/memory")
@handle_api_errors
async def memory_status():
    """Legacy memory status endpoint."""
    return get_memory_usage()

@app.get("/memory/pressure")
@handle_api_errors
async def memory_pressure():
    """Legacy memory pressure endpoint."""
    return check_memory_pressure()

@app.post("/memory/cleanup")
@handle_api_errors
async def trigger_memory_cleanup():
    """Legacy memory cleanup endpoint."""
    logger = get_ai_logger()
    logger.info("Manual memory cleanup triggered")
    
    memory_before = get_memory_usage()
    memory_after = cleanup_memory()
    
    return {
        "status": "completed",
        "memory_before": memory_before,
        "memory_after": memory_after,
        "freed_mb": round(memory_before["process"]["rss_mb"] - memory_after["process"]["rss_mb"], 2)
    }

@app.delete("/models/{model_key}")
@handle_api_errors
async def unload_model_endpoint(model_key: str):
    """Legacy model unload endpoint."""
    return await unload_model_v1(model_key)

def _format_prompt(prompt: str, model_key: str) -> str:
    """Format prompt based on model template."""
    mk = model_key.lower()
    if mk == "qwen":
        return (
            f"<|im_start|>system\nYou are a helpful AI assistant. Respond concisely and accurately.<|im_end|>\n"
            f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        )
    if mk == "llama":
        return (
            f"<|system|>You are a helpful AI assistant. Respond concisely and accurately.<|end|>\n"
            f"<|user|>{prompt}<|end|>\n<|assistant|>"
        )
    # Default generic
    return (
        f"### System: You are a helpful AI assistant. Respond concisely and accurately.\n\n"
        f"### User: {prompt}\n\n### Assistant:"
    )


def _clean_response(raw: str, model_key: str) -> str:
    """Clean model artifacts according to template."""
    text = raw.strip()
    mk = model_key.lower()
    if mk == "qwen" and "<|im_end|>" in text:
        text = text.split("<|im_end|>")[0].strip()
    elif mk == "llama" and "<|end|>" in text:
        text = text.split("<|end|>")[0].strip()
    # Strip fenced json if present
    if text.startswith("```json"):
        import re
        m = re.search(r"```json\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```", text)
        if m:
            text = m.group(1)
    elif text.startswith("```"):
        import re
        m = re.search(r"```\s*([\s\S]*?)\s*```", text)
        if m:
            text = m.group(1)
    return text


@handle_generation_errors
@monitor_performance(get_performance_monitor())
def local_llm_generate(prompt: str, model_key: Optional[str] = None, max_retries: Optional[int] = None) -> str:
    """Generate a response using the selected local model.

    If model_key is None, uses the globally active model key.
    If max_retries is None, uses the configured default.
    """
    mk = (model_key or get_active_model()).lower()
    max_retries = max_retries or _model_config.max_retries
    
    # Get monitor and logger
    error_monitor = get_error_monitor()
    logger = get_ai_logger()
    
    logger.info(f"Starting generation: model={mk}, prompt_length={len(prompt)}")
    # Standard GPT4All generation path
    ai = _get_model_instance(mk)
    formatted_prompt = _format_prompt(prompt, mk)
    print(f"[ai_service] Using model '{mk}' with prompt length {len(formatted_prompt)}")
    
    # Use retry handler
    retry_handler = RetryHandler(GENERATION_RETRY_CONFIG)
    
    def _generate_with_model():
        tokens = ""
        token_count = 0
        params = {
            "prompt": formatted_prompt,
            "max_tokens": _model_config.max_tokens,
            "temp": _model_config.temperature,
            "top_p": _model_config.top_p,
            "repeat_penalty": _model_config.repeat_penalty,
            "streaming": False,
        }
        
        try:
            for tok in ai.generate(**params):
                tokens += tok
                token_count += 1
                if token_count % 100 == 0:
                    print(f"[ai_service] Generated {token_count} tokens...")
        except Exception as gen_error:
            print(f"[ai_service] Primary generation failed: {gen_error}")
            error_monitor.record_error(gen_error, ErrorType.GENERATION_ERROR)
            
            # Try fallback mode
            print(f"[ai_service] Trying fallback generation...")
            try:
                for tok in ai.generate(prompt, max_tokens=400, temp=0.5):
                    tokens += tok
            except Exception as fallback_error:
                error_monitor.record_error(fallback_error, ErrorType.GENERATION_ERROR)
                raise RetryableError(f"Both primary and fallback generation failed: {fallback_error}")

        cleaned = _clean_response(tokens, mk)
        if not cleaned:
            # Perform sanity check
            try:
                sanity = "".join(ai.generate("Say hello", max_tokens=10))
                if not sanity.strip():
                    raise RetryableError("Model sanity check failed - no output produced")
            except Exception as sanity_error:
                error_monitor.record_error(sanity_error, ErrorType.MODEL_LOAD_ERROR)
                raise RetryableError(f"Model sanity check failed: {sanity_error}")
            
            raise RetryableError("Empty response after cleaning")
        
        return cleaned
    
    try:
        result = retry_handler(_generate_with_model)()
        # Record success
        update_model_health(mk, success=True)
        logger.info(f"Model '{mk}' generation succeeded")
        return result
    except Exception as e:
        # Record failure
        update_model_health(mk, success=False)
        error_monitor.record_error(e, ErrorType.GENERATION_ERROR)
        print(f"[ai_service] Model '{mk}' failed: {e}")
        
        # Enable fallback strategy
        if _model_config.enable_fallback and mk in _model_config.fallback_chain:
            return _try_fallback_models(prompt, mk, error_monitor, logger)
        else:
            print(f"[ai_service] No fallback available for model '{mk}'")
            return _get_simple_response(prompt)


def _try_fallback_models(prompt: str, failed_model: str, error_monitor, logger) -> str:
    """Try the next model in the fallback chain"""
    fallback_chain = _model_config.fallback_chain.copy()
    
    # Find the position of the failed model in the chain
    try:
        failed_index = fallback_chain.index(failed_model)
        # Try subsequent models in the chain
        remaining_models = fallback_chain[failed_index + 1:]
    except ValueError:
        # If the failed model is not in the chain, try the entire chain
        remaining_models = fallback_chain
    
    logger.info(f"Trying fallback models: {remaining_models}")
    
    for fallback_model in remaining_models:
        if fallback_model == "simple":
            logger.info("Using simple response fallback")
            return _get_simple_response(prompt)
            
        try:
            logger.info(f"Attempting fallback to model: {fallback_model}")
            
            # Recursive call, but disable further fallback to avoid infinite loops
            original_enable_fallback = _model_config.enable_fallback
            _model_config.enable_fallback = False
            
            try:
                result = local_llm_generate(prompt, model_key=fallback_model, max_retries=1)
                update_model_health(fallback_model, success=True)
                logger.info(f"Fallback to {fallback_model} succeeded")
                return result
            finally:
                _model_config.enable_fallback = original_enable_fallback
                
        except Exception as fallback_error:
            error_monitor.record_error(fallback_error, ErrorType.GENERATION_ERROR)
            logger.warning(f"Fallback model {fallback_model} also failed: {fallback_error}")
            continue
    
    # All fallback models failed
    logger.error("All fallback models failed, using simple response")
    return _get_simple_response(prompt)


def _get_simple_response(prompt: str) -> str:
    """Generate simple default response without relying on any model"""
    prompt_lower = prompt.lower()
    
    # Simple response based on prompt content
    if "hello" in prompt_lower or "hi" in prompt_lower:
        return "Hello! How can I help you?"
    elif "how are you" in prompt_lower:
        return "I'm doing well, thank you for asking!"
    elif "what" in prompt_lower and "time" in prompt_lower:
        from datetime import datetime
        return f"The current time is {datetime.now().strftime('%H:%M:%S')}"
    elif "weather" in prompt_lower:
        return "I don't have access to current weather information."
    elif "name" in prompt_lower:
        return "I'm an AI assistant."
    elif "help" in prompt_lower:
        return "I'm here to help! Please let me know what you need assistance with."
    elif "?" in prompt:
        return "I understand you have a question, but I'm having trouble processing it right now. Could you please rephrase?"
    elif any(word in prompt_lower for word in ["action", "move", "go", "walk"]):
        return '{"action": "wander", "reason": "default action when uncertain"}'
    elif any(word in prompt_lower for word in ["decide", "choose", "option"]):
        return "I would recommend taking some time to consider your options carefully."
    else:
        return "I apologize, but I'm experiencing technical difficulties. Please try again later."


# Model health status tracking
_MODEL_HEALTH = {}


def get_model_health(model_key: str) -> Dict[str, Any]:
    """Get model health status"""
    if model_key not in _MODEL_HEALTH:
        _MODEL_HEALTH[model_key] = {
            "status": "unknown",
            "last_success": None,
            "last_failure": None,
            "consecutive_failures": 0,
            "total_requests": 0,
            "success_rate": 0.0
        }
    return _MODEL_HEALTH[model_key]


def update_model_health(model_key: str, success: bool):
    """Update model health status"""
    health = get_model_health(model_key)
    health["total_requests"] += 1
    
    if success:
        health["status"] = "healthy"
        health["last_success"] = datetime.now()
        health["consecutive_failures"] = 0
    else:
        health["last_failure"] = datetime.now()
        health["consecutive_failures"] += 1
        
        # Mark as unhealthy if consecutive failures exceed threshold
        if health["consecutive_failures"] >= 3:
            health["status"] = "unhealthy"
    
    # Calculate success rate
    if health["total_requests"] > 0:
        success_count = health["total_requests"] - health["consecutive_failures"]
        health["success_rate"] = success_count / health["total_requests"]


def is_model_healthy(model_key: str) -> bool:
    """Check if model is healthy"""
    health = get_model_health(model_key)
    return health["status"] != "unhealthy"


# Model warmup functionality
def warmup_model(model_key: str) -> bool:
    """Warm up specified model"""
    logger = get_ai_logger()
    logger.info(f"Starting warmup for model: {model_key}")
    
    try:
        # Use simple test prompts for warmup
        test_prompts = [
            "Hello",
            "Test response",
            "How are you?"
        ]
        
        for prompt in test_prompts:
            try:
                response = local_llm_generate(prompt, model_key=model_key, max_retries=1)
                if response and len(response.strip()) > 0:
                    logger.info(f"Model {model_key} warmup successful")
                    update_model_health(model_key, success=True)
                    return True
            except Exception as e:
                logger.warning(f"Warmup attempt failed for {model_key}: {e}")
                continue
        
        logger.error(f"Model {model_key} warmup failed after all attempts")
        update_model_health(model_key, success=False)
        return False
        
    except Exception as e:
        logger.error(f"Model {model_key} warmup failed: {e}")
        update_model_health(model_key, success=False)
        return False


def warmup_all_models() -> Dict[str, bool]:
    """Warm up all configured models"""
    logger = get_ai_logger()
    logger.info("Starting model warmup process")
    
    warmup_results = {}
    
    # Warm up main models
    if _model_config.preload_models:
        # First warm up active model
        active_model = get_active_model()
        if active_model and active_model != "auto":
            warmup_results[active_model] = warmup_model(active_model)
        
        # Then warm up models in fallback chain
        for model_key in _model_config.fallback_chain:
            if model_key != "simple" and model_key not in warmup_results:
                warmup_results[model_key] = warmup_model(model_key)
    
    successful_warmups = sum(1 for success in warmup_results.values() if success)
    logger.info(f"Warmup completed: {successful_warmups}/{len(warmup_results)} models ready")
    
    return warmup_results


def warmup_model_async(model_key: str):
    """Asynchronously warm up model (runs in background thread)"""
    import threading
    
    def _warmup_thread():
        try:
            warmup_model(model_key)
        except Exception as e:
            logger = get_ai_logger()
            logger.error(f"Async warmup failed for {model_key}: {e}")
    
    thread = threading.Thread(target=_warmup_thread, daemon=True)
    thread.start()
    return thread


# Memory monitoring and management
def get_memory_usage() -> Dict[str, Any]:
    """Get memory usage information"""
    import psutil
    import gc
    
    # System memory
    memory = psutil.virtual_memory()
    
    # Process memory
    process = psutil.Process()
    process_memory = process.memory_info()
    
    # Python object count
    gc.collect()  # Force garbage collection
    
    return {
        "system": {
            "total_mb": round(memory.total / 1024 / 1024, 2),
            "available_mb": round(memory.available / 1024 / 1024, 2),
            "used_mb": round(memory.used / 1024 / 1024, 2),
            "percent": memory.percent
        },
        "process": {
            "rss_mb": round(process_memory.rss / 1024 / 1024, 2),
            "vms_mb": round(process_memory.vms / 1024 / 1024, 2),
            "percent": process.memory_percent()
        },
        "models": {
            "loaded_count": len(_MODEL_INSTANCES),
            "loaded_models": list(_MODEL_INSTANCES.keys()),
            "cache_limit": _model_config.model_cache_size
        }
    }


def check_memory_pressure() -> Dict[str, Any]:
    """Check memory pressure situation"""
    memory_info = get_memory_usage()
    
    # Define thresholds
    SYSTEM_MEMORY_WARNING = 85.0  # System memory usage exceeds 85%
    SYSTEM_MEMORY_CRITICAL = 95.0  # System memory usage exceeds 95%
    PROCESS_MEMORY_WARNING = 2048  # Process memory usage exceeds 2GB
    PROCESS_MEMORY_CRITICAL = 4096  # Process memory usage exceeds 4GB
    
    pressure_level = "normal"
    warnings = []
    recommendations = []
    
    # Check system memory
    system_percent = memory_info["system"]["percent"]
    if system_percent >= SYSTEM_MEMORY_CRITICAL:
        pressure_level = "critical"
        warnings.append(f"System memory usage critical: {system_percent:.1f}%")
        recommendations.append("Immediately unload unused models")
    elif system_percent >= SYSTEM_MEMORY_WARNING:
        pressure_level = "warning" if pressure_level == "normal" else pressure_level
        warnings.append(f"System memory usage high: {system_percent:.1f}%")
        recommendations.append("Consider unloading some models")
    
    # Check process memory
    process_mb = memory_info["process"]["rss_mb"]
    if process_mb >= PROCESS_MEMORY_CRITICAL:
        pressure_level = "critical"
        warnings.append(f"Process memory usage critical: {process_mb:.1f}MB")
        recommendations.append("Force garbage collection and model cleanup")
    elif process_mb >= PROCESS_MEMORY_WARNING:
        pressure_level = "warning" if pressure_level == "normal" else pressure_level
        warnings.append(f"Process memory usage high: {process_mb:.1f}MB")
        recommendations.append("Run garbage collection")
    
    # Check model cache
    loaded_models = memory_info["models"]["loaded_count"]
    cache_limit = memory_info["models"]["cache_limit"]
    if loaded_models > cache_limit:
        warnings.append(f"Model cache overflow: {loaded_models}/{cache_limit}")
        recommendations.append("Unload least recently used models")
    
    return {
        "pressure_level": pressure_level,
        "warnings": warnings,
        "recommendations": recommendations,
        "memory_info": memory_info,
        "timestamp": datetime.now().isoformat()
    }


def cleanup_memory():
    """Clean up memory, unload unnecessary models"""
    import gc
    
    logger = get_ai_logger()
    logger.info("Starting memory cleanup")
    
    # Force garbage collection
    gc.collect()
    
    # If model cache exceeds limit, unload some models
    if len(_MODEL_INSTANCES) > _model_config.model_cache_size:
        # Keep active model and healthy models, unload others
        active_model = get_active_model()
        models_to_keep = {active_model} if active_model != "auto" else set()
        
        # Keep healthy models
        for model_key in list(_MODEL_INSTANCES.keys()):
            if is_model_healthy(model_key):
                models_to_keep.add(model_key)
            
            # If enough already, stop adding
            if len(models_to_keep) >= _model_config.model_cache_size:
                break
        
        # Unload unnecessary models
        models_to_unload = set(_MODEL_INSTANCES.keys()) - models_to_keep
        for model_key in models_to_unload:
            unload_model(model_key)
    
    # Garbage collection again
    gc.collect()
    
    memory_after = get_memory_usage()
    logger.info(f"Memory cleanup completed. Process memory: {memory_after['process']['rss_mb']:.1f}MB")
    
    return memory_after


def unload_model(model_key: str):
    """Unload specified model"""
    logger = get_ai_logger()
    
    if model_key in _MODEL_INSTANCES:
        logger.info(f"Unloading model: {model_key}")
        del _MODEL_INSTANCES[model_key]
        
        # Force garbage collection
        import gc
        gc.collect()
        
        logger.info(f"Model {model_key} unloaded successfully")
    else:
        logger.warning(f"Model {model_key} not found in cache")


def auto_memory_management():
    """Automatic memory management"""
    pressure = check_memory_pressure()
    
    if pressure["pressure_level"] == "critical":
        logger = get_ai_logger()
        logger.warning("Critical memory pressure detected, starting aggressive cleanup")
        cleanup_memory()
    elif pressure["pressure_level"] == "warning":
        logger = get_ai_logger()
        logger.info("Memory pressure warning, starting gentle cleanup")
        import gc
        gc.collect()
    
    return pressure



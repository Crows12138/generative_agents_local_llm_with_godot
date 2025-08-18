"""Local LLM service wrapper used by planners and other modules.

Adds switchable backends for two local models and proper prompt formatting:
 - qwen: qwen2.5 chat template
 - gpt-oss: Harmony chat format (special)

Use local GGUF files via GPT4All and Transformers for generative agents.
"""

from __future__ import annotations

import os
import time
import random
from pathlib import Path
from typing import Dict, Optional

import requests
from fastapi import FastAPI
from pydantic import BaseModel
from gpt4all import GPT4All
from importlib import import_module

# 导入配置管理器、错误处理和监控
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

# 获取配置
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

# API版本前缀
API_V1_PREFIX = "/v1"

# 通用错误处理装饰器
def handle_api_errors(func):
    """通用API错误处理装饰器"""
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
            # 重新抛出HTTP异常
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

# 使用配置中的路径和模型
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = Path(_model_config.models_dir)

# 支持的模型从配置获取
SUPPORTED_GGUF: Dict[str, str] = _model_config.supported_models

# 活动模型从配置获取
_ACTIVE_MODEL_KEY: str = _model_config.active_model.lower()

# Cache of GPT4All instances by model key
_MODEL_INSTANCES: Dict[str, GPT4All] = {}
_GPT_OSS_ADAPTER = None  # Lazy-initialized transformers adapter


def set_active_model(model_key: str) -> bool:
    """Set the globally active model key (qwen|gpt-oss)."""
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

    # GPT-OSS uses transformers adapter by default; skip GPT4All path
    use_gpt4all_for_gptoss = _model_config.use_gpt4all_for_gptoss
    if model_key == "gpt-oss" and not use_gpt4all_for_gptoss:
        raise RuntimeError("GPT-OSS is handled by transformers adapter, not GPT4All instance")

    if model_key in _MODEL_INSTANCES:
        return _MODEL_INSTANCES[model_key]

    # 检查内存压力，如果必要则清理
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

    # Resolve desired device. Force CPU for llama/gpt-oss to avoid Kompute/Vulkan crashes.
    force_cpu_env = _model_config.force_cpu
    desired_device = "cpu" if (force_cpu_env or model_key in ("llama", "gpt-oss")) else "gpu"
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

# 统一API请求/响应模型
class BaseRequest(BaseModel):
    """基础请求模型"""
    request_id: Optional[str] = None
    timestamp: Optional[str] = None
    version: str = "v1"

class BaseResponse(BaseModel):
    """基础响应模型"""
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
    """错误响应模型"""
    status: str = "error"
    error_code: str
    error_message: str
    error_details: Optional[Dict[str, Any]] = None

# 向后兼容的端点
@app.post("/chat", response_model=ChatResponse)
@handle_api_errors
async def chat_legacy(req: ChatRequest):
    """Legacy chat endpoint for backward compatibility."""
    return await chat_v1(req)

# V1 API端点
@app.post(f"{API_V1_PREFIX}/chat", response_model=ChatResponse)
@handle_api_errors
async def chat_v1(req: ChatRequest):
    """Expose a /chat endpoint using the requested model (or active default)."""
    import uuid
    
    # 生成请求ID和时间戳
    request_id = req.request_id or str(uuid.uuid4())
    start_time = datetime.now()
    
    logger = get_ai_logger()
    logger.info(f"Chat request [{request_id}]: model={req.model_key}, prompt_length={len(req.prompt)}")
    
    try:
        # 保存原始降级设置状态
        original_fallback_enabled = _model_config.enable_fallback
        fallback_used = False
        
        # 如果指定了模型参数，临时使用
        original_active_model = get_active_model()
        if req.model_key and req.model_key != original_active_model:
            set_active_model(req.model_key)
        
        # 调用生成服务
        response_text = local_llm_generate(
            req.prompt, 
            model_key=req.model_key,
            max_retries=_model_config.max_retries
        )
        
        # 检查是否使用了降级
        if not original_fallback_enabled and _model_config.enable_fallback:
            fallback_used = True
        
        # 计算处理时间
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # 确定实际使用的模型
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
        
        # 返回错误响应
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

# Health和Metrics端点 (兼容版本)
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

# V1 Health和Metrics端点
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

# 向后兼容
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
        # 预热指定模型
        logger.info(f"Triggering warmup for model: {model_key}")
        success = warmup_model(model_key)
        return {
            "model": model_key,
            "success": success,
            "message": f"Model {model_key} warmup {'succeeded' if success else 'failed'}",
            "timestamp": datetime.now().isoformat()
        }
    else:
        # 预热所有模型
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

# Memory管理端点 (V1)
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

@app.delete(f"{API_V1_PREFIX}/models/{model_key}")
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

# 向后兼容的端点
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
    if mk == "gpt-oss":
        # When using transformers adapter, no special formatting needed.
        return prompt
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
    elif mk == "gpt-oss":
        # Adapter already returns plain text
        pass
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
    
    # 获取监控器和日志器
    error_monitor = get_error_monitor()
    logger = get_ai_logger()
    
    logger.info(f"Starting generation: model={mk}, prompt_length={len(prompt)}")
    # Special path for GPT-OSS via transformers adapter (default)
    if mk == "gpt-oss" and not _model_config.use_gpt4all_for_gptoss:
        global _GPT_OSS_ADAPTER
        if _GPT_OSS_ADAPTER is None:
            try:
                module = import_module("gpt_oss_adapter")
                init_fn = getattr(module, "initialize_gpt_oss")
                _GPT_OSS_ADAPTER = init_fn()
            except Exception as e:
                return f"[ERROR] Failed to initialize GPT-OSS adapter: {e}"
            if _GPT_OSS_ADAPTER is None:
                return "[ERROR] GPT-OSS adapter initialization returned None"

        try:
            # Let adapter handle system prompt and formatting internally
            gen_fn = getattr(import_module("gpt_oss_adapter"), "generate_with_gpt_oss")
            response = gen_fn(_GPT_OSS_ADAPTER, prompt, system_prompt="")
            return response.strip()
        except Exception as e:
            return f"[ERROR] GPT-OSS adapter generation failed: {e}"

    # Default GPT4All path
    ai = _get_model_instance(mk)
    formatted_prompt = _format_prompt(prompt, mk)
    print(f"[ai_service] Using model '{mk}' with prompt length {len(formatted_prompt)}")
    
    # 使用重试处理器
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
            
            # 尝试fallback模式
            print(f"[ai_service] Trying fallback generation...")
            try:
                for tok in ai.generate(prompt, max_tokens=400, temp=0.5):
                    tokens += tok
            except Exception as fallback_error:
                error_monitor.record_error(fallback_error, ErrorType.GENERATION_ERROR)
                raise RetryableError(f"Both primary and fallback generation failed: {fallback_error}")

        cleaned = _clean_response(tokens, mk)
        if not cleaned:
            # 进行健全性检查
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
        # 记录成功
        update_model_health(mk, success=True)
        logger.info(f"Model '{mk}' generation succeeded")
        return result
    except Exception as e:
        # 记录失败
        update_model_health(mk, success=False)
        error_monitor.record_error(e, ErrorType.GENERATION_ERROR)
        print(f"[ai_service] Model '{mk}' failed: {e}")
        
        # 启用降级策略
        if _model_config.enable_fallback and mk in _model_config.fallback_chain:
            return _try_fallback_models(prompt, mk, error_monitor, logger)
        else:
            print(f"[ai_service] No fallback available for model '{mk}'")
            return _get_simple_response(prompt)


def _try_fallback_models(prompt: str, failed_model: str, error_monitor, logger) -> str:
    """尝试降级模型链中的下一个模型"""
    fallback_chain = _model_config.fallback_chain.copy()
    
    # 找到失败模型在链中的位置
    try:
        failed_index = fallback_chain.index(failed_model)
        # 尝试链中后续的模型
        remaining_models = fallback_chain[failed_index + 1:]
    except ValueError:
        # 如果失败的模型不在链中，尝试整个链
        remaining_models = fallback_chain
    
    logger.info(f"Trying fallback models: {remaining_models}")
    
    for fallback_model in remaining_models:
        if fallback_model == "simple":
            logger.info("Using simple response fallback")
            return _get_simple_response(prompt)
            
        try:
            logger.info(f"Attempting fallback to model: {fallback_model}")
            
            # 递归调用，但禁用进一步的降级以避免无限循环
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
    
    # 所有降级模型都失败了
    logger.error("All fallback models failed, using simple response")
    return _get_simple_response(prompt)


def _get_simple_response(prompt: str) -> str:
    """生成简单的默认响应，不依赖任何模型"""
    prompt_lower = prompt.lower()
    
    # 基于提示内容的简单响应
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


# 模型健康状态跟踪
_MODEL_HEALTH = {}


def get_model_health(model_key: str) -> Dict[str, Any]:
    """获取模型健康状态"""
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
    """更新模型健康状态"""
    health = get_model_health(model_key)
    health["total_requests"] += 1
    
    if success:
        health["status"] = "healthy"
        health["last_success"] = datetime.now()
        health["consecutive_failures"] = 0
    else:
        health["last_failure"] = datetime.now()
        health["consecutive_failures"] += 1
        
        # 连续失败超过阈值则标记为不健康
        if health["consecutive_failures"] >= 3:
            health["status"] = "unhealthy"
    
    # 计算成功率
    if health["total_requests"] > 0:
        success_count = health["total_requests"] - health["consecutive_failures"]
        health["success_rate"] = success_count / health["total_requests"]


def is_model_healthy(model_key: str) -> bool:
    """检查模型是否健康"""
    health = get_model_health(model_key)
    return health["status"] != "unhealthy"


# 模型预热功能
def warmup_model(model_key: str) -> bool:
    """预热指定模型"""
    logger = get_ai_logger()
    logger.info(f"Starting warmup for model: {model_key}")
    
    try:
        # 使用简单的测试提示进行预热
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
    """预热所有配置的模型"""
    logger = get_ai_logger()
    logger.info("Starting model warmup process")
    
    warmup_results = {}
    
    # 预热主要模型
    if _model_config.preload_models:
        # 首先预热活动模型
        active_model = get_active_model()
        if active_model and active_model != "auto":
            warmup_results[active_model] = warmup_model(active_model)
        
        # 然后预热降级链中的模型
        for model_key in _model_config.fallback_chain:
            if model_key != "simple" and model_key not in warmup_results:
                warmup_results[model_key] = warmup_model(model_key)
    
    successful_warmups = sum(1 for success in warmup_results.values() if success)
    logger.info(f"Warmup completed: {successful_warmups}/{len(warmup_results)} models ready")
    
    return warmup_results


def warmup_model_async(model_key: str):
    """异步预热模型（在后台线程中运行）"""
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


# 内存监控和管理
def get_memory_usage() -> Dict[str, Any]:
    """获取内存使用情况"""
    import psutil
    import gc
    
    # 系统内存
    memory = psutil.virtual_memory()
    
    # 进程内存
    process = psutil.Process()
    process_memory = process.memory_info()
    
    # Python对象计数
    gc.collect()  # 强制垃圾回收
    
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
    """检查内存压力情况"""
    memory_info = get_memory_usage()
    
    # 定义阈值
    SYSTEM_MEMORY_WARNING = 85.0  # 系统内存使用超过85%
    SYSTEM_MEMORY_CRITICAL = 95.0  # 系统内存使用超过95%
    PROCESS_MEMORY_WARNING = 2048  # 进程内存使用超过2GB
    PROCESS_MEMORY_CRITICAL = 4096  # 进程内存使用超过4GB
    
    pressure_level = "normal"
    warnings = []
    recommendations = []
    
    # 检查系统内存
    system_percent = memory_info["system"]["percent"]
    if system_percent >= SYSTEM_MEMORY_CRITICAL:
        pressure_level = "critical"
        warnings.append(f"System memory usage critical: {system_percent:.1f}%")
        recommendations.append("Immediately unload unused models")
    elif system_percent >= SYSTEM_MEMORY_WARNING:
        pressure_level = "warning" if pressure_level == "normal" else pressure_level
        warnings.append(f"System memory usage high: {system_percent:.1f}%")
        recommendations.append("Consider unloading some models")
    
    # 检查进程内存
    process_mb = memory_info["process"]["rss_mb"]
    if process_mb >= PROCESS_MEMORY_CRITICAL:
        pressure_level = "critical"
        warnings.append(f"Process memory usage critical: {process_mb:.1f}MB")
        recommendations.append("Force garbage collection and model cleanup")
    elif process_mb >= PROCESS_MEMORY_WARNING:
        pressure_level = "warning" if pressure_level == "normal" else pressure_level
        warnings.append(f"Process memory usage high: {process_mb:.1f}MB")
        recommendations.append("Run garbage collection")
    
    # 检查模型缓存
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
    """清理内存，卸载不必要的模型"""
    import gc
    
    logger = get_ai_logger()
    logger.info("Starting memory cleanup")
    
    # 强制垃圾回收
    gc.collect()
    
    # 如果模型缓存超过限制，卸载一些模型
    if len(_MODEL_INSTANCES) > _model_config.model_cache_size:
        # 保留活动模型和健康的模型，卸载其他的
        active_model = get_active_model()
        models_to_keep = {active_model} if active_model != "auto" else set()
        
        # 保留健康的模型
        for model_key in list(_MODEL_INSTANCES.keys()):
            if is_model_healthy(model_key):
                models_to_keep.add(model_key)
            
            # 如果已经足够了，停止添加
            if len(models_to_keep) >= _model_config.model_cache_size:
                break
        
        # 卸载不需要的模型
        models_to_unload = set(_MODEL_INSTANCES.keys()) - models_to_keep
        for model_key in models_to_unload:
            unload_model(model_key)
    
    # 再次垃圾回收
    gc.collect()
    
    memory_after = get_memory_usage()
    logger.info(f"Memory cleanup completed. Process memory: {memory_after['process']['rss_mb']:.1f}MB")
    
    return memory_after


def unload_model(model_key: str):
    """卸载指定模型"""
    logger = get_ai_logger()
    
    if model_key in _MODEL_INSTANCES:
        logger.info(f"Unloading model: {model_key}")
        del _MODEL_INSTANCES[model_key]
        
        # 强制垃圾回收
        import gc
        gc.collect()
        
        logger.info(f"Model {model_key} unloaded successfully")
    else:
        logger.warning(f"Model {model_key} not found in cache")


def auto_memory_management():
    """自动内存管理"""
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



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

app = FastAPI()
_BASE_URL = "http://127.0.0.1:8001"

# Since this file is in reverie/backend_server/ai_service, go up 3 levels to get to generative_agents-main root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models" / "gpt4all"

# Supported local models (GGUF files)
SUPPORTED_GGUF: Dict[str, str] = {
    "qwen": "qwen2.5-coder-7b-instruct-q4_0.gguf",
    "gpt-oss": "gpt-oss-20b-F16.gguf",  # transformers adapter by default
}

# Active model key; can be overridden via environment variable AI_MODEL_KEY
_ACTIVE_MODEL_KEY: str = os.environ.get("AI_MODEL_KEY", "qwen").lower()

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


def _get_model_instance(model_key: str) -> GPT4All:
    """Get or lazily create a GPT4All instance for the given key."""
    model_key = model_key.lower()
    if model_key not in SUPPORTED_GGUF:
        print(f"[ai_service] Unsupported model key '{model_key}', falling back to 'qwen'")
        model_key = "qwen"

    # GPT-OSS uses transformers adapter by default; skip GPT4All path
    use_gpt4all_for_gptoss = os.environ.get("USE_GPT4ALL_FOR_GPTOSS", "0") == "1"
    if model_key == "gpt-oss" and not use_gpt4all_for_gptoss:
        raise RuntimeError("GPT-OSS is handled by transformers adapter, not GPT4All instance")

    if model_key in _MODEL_INSTANCES:
        return _MODEL_INSTANCES[model_key]

    model_file = SUPPORTED_GGUF[model_key]
    model_path = MODELS_DIR / model_file
    print(f"[ai_service] Loading model '{model_key}' from: {model_path}")
    print(f"[ai_service] Model file exists: {model_path.exists()}")

    # Resolve desired device. Force CPU for llama/gpt-oss to avoid Kompute/Vulkan crashes.
    force_cpu_env = os.environ.get("AI_FORCE_CPU", "0") == "1"
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

class ChatRequest(BaseModel):
    model_key: Optional[str] = None
    prompt: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Expose a /chat endpoint using the requested model (or active default)."""
    response = local_llm_generate(req.prompt, model_key=req.model_key)
    return ChatResponse(response=response)

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


def local_llm_generate(prompt: str, model_key: Optional[str] = None, max_retries: int = 3) -> str:
    """Generate a response using the selected local model.

    If model_key is None, uses the globally active model key.
    """
    mk = (model_key or get_active_model()).lower()
    # Special path for GPT-OSS via transformers adapter (default)
    if mk == "gpt-oss" and os.environ.get("USE_GPT4ALL_FOR_GPTOSS", "0") != "1":
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
    
    for attempt in range(max_retries):
        try:
            tokens = ""
            token_count = 0
            params = {
                "prompt": formatted_prompt,
                "max_tokens": 800,
                "temp": 0.3,
                "top_p": 0.9,
                "repeat_penalty": 1.1,
                "streaming": False,
            }
            
            try:
                for tok in ai.generate(**params):
                    tokens += tok
                    token_count += 1
                    if token_count % 100 == 0:
                        print(f"[ai_service] Generated {token_count} tokens...")
            except Exception as gen_error:
                print(f"[ai_service] Generation error: {gen_error}; trying fallback mode")
                for tok in ai.generate(prompt, max_tokens=400, temp=0.5):
                    tokens += tok

            cleaned = _clean_response(tokens, mk)
            if cleaned:
                return cleaned

            # If empty, do a tiny test to validate the backend
            sanity = "".join(ai.generate("Say hello", max_tokens=10))
            if not sanity.strip():
                raise RuntimeError("Model produced no output in sanity check")

                if attempt == max_retries - 1:
                    return '{"action": "wander"}'
            raise RuntimeError("Empty response; retrying")
            
        except Exception as e:
            wait_time = (2 ** attempt) + random.uniform(0, 1)
            print(f"[ai_service] Attempt {attempt+1}/{max_retries} failed: {e}; retrying in {wait_time:.2f}s")
            if attempt < max_retries - 1:
                time.sleep(wait_time)
            else:
                return '{"action": "wander"}'
    
    return '{"action": "wander"}'


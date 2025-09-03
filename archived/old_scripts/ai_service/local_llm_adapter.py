"""
Local LLM Adapter for Generative Agents

This module replaces OpenAI API calls with local model calls,
maintaining compatibility with the existing generative agents architecture.
"""

import sys
import os
from pathlib import Path

# Add the ai_service directory to Python path
sys.path.append(str(Path(__file__).parent))

try:
    from .ai_service import local_llm_generate, set_active_model, get_active_model
except ImportError:
    # Fallback for direct execution
    from ai_service import local_llm_generate, set_active_model, get_active_model


def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False,
                           model_key=None):
    """
    Drop-in replacement for the original safe_generate_response function.
    Uses local LLM instead of OpenAI API.
    
    Args:
        prompt: The prompt string
        gpt_parameter: Original GPT parameters (mostly ignored for local models)
        repeat: Number of retry attempts
        fail_safe_response: Fallback response if all attempts fail
        func_validate: Validation function for the response
        func_clean_up: Cleanup function for the response
        verbose: Whether to print debug info
        model_key: Optional model key override (qwen)
    
    Returns:
        Generated response string
    """
    if verbose:
        print(f"[LocalLLM] Using model: {model_key or get_active_model()}")
        print(f"[LocalLLM] Prompt: {prompt[:200]}...")
    
    for i in range(repeat):
        try:
            # Generate response using local LLM
            curr_response = local_llm_generate(prompt, model_key=model_key)
            
            if func_validate and not func_validate(curr_response, prompt=prompt):
                if verbose:
                    print(f"[LocalLLM] Validation failed on attempt {i+1}")
                continue
            
            if func_clean_up:
                curr_response = func_clean_up(curr_response, prompt=prompt)
            
            return curr_response
            
        except Exception as e:
            if verbose:
                print(f"[LocalLLM] Error on attempt {i+1}: {e}")
            continue
    
    if verbose:
        print(f"[LocalLLM] All attempts failed, returning fail_safe: {fail_safe_response}")
    return fail_safe_response


def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False,
                                   model_key=None):
    """
    Drop-in replacement for ChatGPT_safe_generate_response.
    Formats the prompt to match the expected JSON output format.
    """
    # Reconstruct the prompt format that the original function expects
    formatted_prompt = '"""\n' + prompt + '\n"""\n'
    formatted_prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    formatted_prompt += "Example output json:\n"
    formatted_prompt += '{"output": "' + str(example_output) + '"}'
    
    if verbose:
        print("[LocalLLM] ChatGPT-style prompt:")
        print(formatted_prompt)
    
    for i in range(repeat):
        try:
            # Generate response using local LLM
            curr_response = local_llm_generate(formatted_prompt, model_key=model_key)
            
            # Try to extract JSON output
            try:
                import json
                end_index = curr_response.rfind('}') + 1
                if end_index > 0:
                    json_part = curr_response[:end_index]
                    parsed = json.loads(json_part)
                    if "output" in parsed:
                        curr_response = parsed["output"]
            except:
                # If JSON parsing fails, use the raw response
                pass
            
            if func_validate and not func_validate(curr_response, prompt=formatted_prompt):
                if verbose:
                    print(f"[LocalLLM] ChatGPT validation failed on attempt {i+1}")
                continue
            
            if func_clean_up:
                curr_response = func_clean_up(curr_response, prompt=formatted_prompt)
            
            return curr_response
            
        except Exception as e:
            if verbose:
                print(f"[LocalLLM] ChatGPT error on attempt {i+1}: {e}")
            continue
    
    return False


def GPT4_safe_generate_response(prompt, 
                                example_output,
                                special_instruction,
                                repeat=3,
                                fail_safe_response="error",
                                func_validate=None,
                                func_clean_up=None,
                                verbose=False,
                                model_key=None):
    """
    Drop-in replacement for GPT4_safe_generate_response.
    Uses the same logic as ChatGPT version but with GPT-4 prompt format.
    """
    preferred_model = model_key or "qwen"
    
    formatted_prompt = 'GPT-3 Prompt:\n"""\n' + prompt + '\n"""\n'
    formatted_prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    formatted_prompt += "Example output json:\n"
    formatted_prompt += '{"output": "' + str(example_output) + '"}'
    
    return ChatGPT_safe_generate_response(
        prompt, example_output, special_instruction, repeat, 
        fail_safe_response, func_validate, func_clean_up, verbose, preferred_model
    )


def get_embedding(text, model="text-embedding-ada-002"):
    """
    Enhanced embedding using sentence-transformers model.
    Completely replaces hash-based embedding with proper sentence-transformers.
    """
    try:
        from .embedding_service import get_embedding_service
        service = get_embedding_service()
        return service.get_embedding(text)
    except ImportError:
        # Fallback for direct execution
        from embedding_service import get_embedding_service
        service = get_embedding_service()
        return service.get_embedding(text)
    except Exception as e:
        print(f"[ERROR] Embedding service completely failed: {e}")
        # Create a minimal fallback service if everything fails
        return _create_emergency_embedding(text)


def _create_emergency_embedding(text, target_size=384):
    """
    Emergency fallback embedding when all services fail.
    Only used as absolute last resort.
    """
    import hashlib
    import math
    
    if not text or not text.strip():
        text = "empty_text_placeholder"
    
    text = text.replace("\n", " ").strip()
    
    # Use multiple hash functions for better distribution
    hash_funcs = [hashlib.md5, hashlib.sha1, hashlib.sha256]
    raw_values = []
    
    for hash_func in hash_funcs:
        hash_obj = hash_func(text.encode('utf-8'))
        hash_hex = hash_obj.hexdigest()
        
        # Convert hex to normalized values
        for i in range(0, len(hash_hex), 2):
            if len(raw_values) >= target_size:
                break
            val = int(hash_hex[i:i+2], 16) / 255.0 * 2.0 - 1.0
            raw_values.append(val)
        
        if len(raw_values) >= target_size:
            break
    
    # Pad if needed
    while len(raw_values) < target_size:
        raw_values.extend(raw_values[:min(len(raw_values), target_size - len(raw_values))])
    
    return raw_values[:target_size]


# Legacy function aliases for backward compatibility
def ChatGPT_request(prompt, model_key=None):
    """Simple ChatGPT request replacement."""
    return local_llm_generate(prompt, model_key=model_key)


def GPT4_request(prompt, model_key=None):
    """Simple GPT-4 request replacement."""
    preferred_model = model_key or "qwen"
    return local_llm_generate(prompt, model_key=preferred_model)


def GPT_request(prompt, gpt_parameter, model_key=None):
    """Legacy GPT-3 request replacement."""
    return local_llm_generate(prompt, model_key=model_key)

"""LLM Adapter for Reverie Integration

This module replaces all OpenAI API calls in Reverie with local LLM calls
using our DirectLLMService that provides complete, non-truncated outputs.
"""

import sys
import os
import time
import re
from typing import Optional, Dict, Any, List, Callable

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from ai_service.direct_llm_service import direct_llm_service

class ReverieLocalLLMAdapter:
    """Adapter that makes local LLM compatible with all Reverie function signatures"""
    
    def __init__(self):
        self.service = direct_llm_service
        self.request_count = 0
        self.total_tokens_used = 0
        
        # Ensure model is loaded
        if not self.service.ensure_model_loaded():
            raise RuntimeError("Failed to load local LLM for Reverie integration")
    
    def ChatGPT_request(self, prompt: str) -> str:
        """Direct replacement for Reverie's ChatGPT_request function"""
        self.request_count += 1
        
        # Determine appropriate type based on prompt content
        expected_type = self._infer_prompt_type(prompt)
        
        response = self.service.generate_complete(
            prompt=prompt,
            expected_type=expected_type,
            skip_preamble=True
        )
        
        # Track usage
        self.total_tokens_used += len(response.split())
        
        return response.strip()
    
    def GPT4_request(self, prompt: str) -> str:
        """Direct replacement for Reverie's GPT4_request function"""
        # Use same implementation as ChatGPT for consistency
        return self.ChatGPT_request(prompt)
    
    def ChatGPT_safe_generate_response(self, 
                                     prompt: str, 
                                     example_output: str = "",
                                     special_instruction: str = "", 
                                     repeat: int = 3,
                                     fail_safe_response: str = "error",
                                     func_validate: Optional[Callable] = None,
                                     func_clean_up: Optional[Callable] = None,
                                     verbose: bool = False) -> str:
        """
        Safe generation with validation and retry logic
        Matches Reverie's ChatGPT_safe_generate_response signature exactly
        """
        
        # Enhance prompt with example and special instruction
        enhanced_prompt = prompt
        if example_output:
            enhanced_prompt += f"\n\nExample output format:\n{example_output}"
        if special_instruction:
            enhanced_prompt += f"\n\nSpecial instruction: {special_instruction}"
        
        for attempt in range(repeat):
            try:
                if verbose:
                    print(f"[LLM Adapter] Generation attempt {attempt + 1}/{repeat}")
                
                # Generate response
                response = self.ChatGPT_request(enhanced_prompt)
                
                # Apply cleanup function if provided
                if func_clean_up:
                    response = func_clean_up(response)
                
                # Validate if function provided
                if func_validate:
                    if func_validate(response):
                        if verbose:
                            print(f"[LLM Adapter] Validation passed on attempt {attempt + 1}")
                        return response
                    else:
                        if verbose:
                            print(f"[LLM Adapter] Validation failed on attempt {attempt + 1}")
                        continue
                else:
                    # No validation function, return if non-empty
                    if response and len(response.strip()) > 5:
                        return response
                
            except Exception as e:
                if verbose:
                    print(f"[LLM Adapter] Error on attempt {attempt + 1}: {e}")
                continue
        
        # All attempts failed, return fail-safe
        if verbose:
            print(f"[LLM Adapter] All attempts failed, returning fail-safe: {fail_safe_response}")
        return fail_safe_response
    
    def safe_generate_response(self,
                             prompt: str,
                             gpt_parameter: Dict[str, Any] = None,
                             repeat: int = 5,
                             fail_safe_response: str = "error",
                             func_validate: Optional[Callable] = None,
                             func_clean_up: Optional[Callable] = None) -> str:
        """
        Another safe generation method used by Reverie
        """
        
        # Extract relevant parameters
        special_instruction = ""
        example_output = ""
        
        if gpt_parameter:
            special_instruction = gpt_parameter.get("special_instruction", "")
            example_output = gpt_parameter.get("example_output", "")
        
        return self.ChatGPT_safe_generate_response(
            prompt=prompt,
            example_output=example_output,
            special_instruction=special_instruction,
            repeat=repeat,
            fail_safe_response=fail_safe_response,
            func_validate=func_validate,
            func_clean_up=func_clean_up,
            verbose=False
        )
    
    def _infer_prompt_type(self, prompt: str) -> str:
        """Infer the expected response type from prompt content"""
        
        prompt_lower = prompt.lower()
        
        # Daily/hourly planning
        if any(word in prompt_lower for word in ["daily", "schedule", "plan for today", "wake up"]):
            return "daily_plan"
        
        # Poignancy/rating
        if any(word in prompt_lower for word in ["rate", "scale", "1 to 10", "poignancy", "score"]):
            return "number"
        
        # Conversation/dialogue
        if any(word in prompt_lower for word in ["said:", "response", "what would", "conversation", "talking"]):
            return "conversation"
        
        # Memory retrieval/listing
        if any(word in prompt_lower for word in ["list", "remember", "recall", "memories", "events"]):
            return "list"
        
        # Decision making
        if any(word in prompt_lower for word in ["should", "action", "decide", "choose", "what to do"]):
            return "action"
        
        # Reflection/analysis
        if any(word in prompt_lower for word in ["reflect", "analyze", "think about", "consider"]):
            return "reflection"
        
        return "default"
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return {
            "total_requests": self.request_count,
            "total_tokens_used": self.total_tokens_used,
            "avg_tokens_per_request": self.total_tokens_used / max(1, self.request_count)
        }

# Global adapter instance
reverie_llm_adapter = ReverieLocalLLMAdapter()

# Functions to replace Reverie's OpenAI calls
def ChatGPT_request(prompt: str) -> str:
    """Global function replacement for reverie.backend_server.utils.ChatGPT_request"""
    return reverie_llm_adapter.ChatGPT_request(prompt)

def GPT4_request(prompt: str) -> str:
    """Global function replacement for reverie.backend_server.utils.GPT4_request"""
    return reverie_llm_adapter.GPT4_request(prompt)

def ChatGPT_safe_generate_response(prompt: str, 
                                 example_output: str = "",
                                 special_instruction: str = "", 
                                 repeat: int = 3,
                                 fail_safe_response: str = "error",
                                 func_validate: Optional[Callable] = None,
                                 func_clean_up: Optional[Callable] = None,
                                 verbose: bool = False) -> str:
    """Global function replacement for reverie ChatGPT_safe_generate_response"""
    return reverie_llm_adapter.ChatGPT_safe_generate_response(
        prompt, example_output, special_instruction, repeat, 
        fail_safe_response, func_validate, func_clean_up, verbose
    )

def safe_generate_response(prompt: str,
                         gpt_parameter: Dict[str, Any] = None,
                         repeat: int = 5,
                         fail_safe_response: str = "error",
                         func_validate: Optional[Callable] = None,
                         func_clean_up: Optional[Callable] = None) -> str:
    """Global function replacement for reverie safe_generate_response"""
    return reverie_llm_adapter.safe_generate_response(
        prompt, gpt_parameter, repeat, fail_safe_response, func_validate, func_clean_up
    )

def patch_reverie_llm_functions():
    """
    Monkey patch Reverie's LLM functions with our local implementations
    Call this function before importing any Reverie modules
    """
    
    try:
        # Try to patch utils module if it exists
        import reverie.backend_server.utils as utils
        utils.ChatGPT_request = ChatGPT_request
        utils.GPT4_request = GPT4_request
        utils.ChatGPT_safe_generate_response = ChatGPT_safe_generate_response
        utils.safe_generate_response = safe_generate_response
        print("[Reverie Adapter] Successfully patched utils module")
    except ImportError:
        print("[Reverie Adapter] utils module not found, will patch on demand")
    
    try:
        # Try to patch run_gpt_prompt module if it exists
        import reverie.backend_server.persona.prompt_template.run_gpt_prompt as gpt_prompt
        
        # Patch all run_gpt_prompt functions to use our adapter
        for attr_name in dir(gpt_prompt):
            if attr_name.startswith('run_gpt_prompt'):
                original_func = getattr(gpt_prompt, attr_name)
                patched_func = create_patched_gpt_function(original_func, attr_name)
                setattr(gpt_prompt, attr_name, patched_func)
        
        print("[Reverie Adapter] Successfully patched run_gpt_prompt module")
    except ImportError:
        print("[Reverie Adapter] run_gpt_prompt module not found, will patch on demand")

def create_patched_gpt_function(original_func, func_name: str):
    """Create a patched version of a run_gpt_prompt function"""
    
    def patched_function(*args, **kwargs):
        # Extract prompt from arguments
        prompt = None
        
        # Most run_gpt_prompt functions have persona as first arg, prompt as second
        if len(args) > 1 and isinstance(args[1], str):
            prompt = args[1]
        elif 'prompt' in kwargs:
            prompt = kwargs['prompt']
        
        if prompt:
            # Use our adapter instead of OpenAI
            response = reverie_llm_adapter.ChatGPT_request(prompt)
            
            # Return in format expected by Reverie (usually includes metadata)
            if func_name in ["run_gpt_prompt_event_poignancy", "run_gpt_prompt_poignancy_event_v1"]:
                # Poignancy functions expect just the number
                import re
                numbers = re.findall(r'\b([1-9]|10)\b', response)
                if numbers:
                    return int(numbers[0]), [response, prompt, {}, "", ""]
                else:
                    return 5, [response, prompt, {}, "", ""]  # Default poignancy
            else:
                # Most functions expect (response, metadata) tuple
                return response, [response, prompt, {}, "", ""]
        
        # Fallback to original function if we can't extract prompt
        return original_func(*args, **kwargs)
    
    return patched_function
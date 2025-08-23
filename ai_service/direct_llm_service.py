"""Direct LLM service that bypasses configuration issues and provides complete outputs"""

import os
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import re

# Import llama_cpp directly
try:
    from llama_cpp import Llama
    LLAMACPP_AVAILABLE = True
except ImportError:
    LLAMACPP_AVAILABLE = False
    print("[direct_llm] llama-cpp-python not available")

class DirectLLMService:
    """Direct LLM service with complete output generation"""
    
    def __init__(self):
        self.logger = logging.getLogger("direct_llm")
        self.model = None
        self.model_loaded = False
        
        # Model path
        self.models_dir = Path("models")
        self.model_file = self.models_dir / "Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf"
        
        # Enhanced generation settings
        self.generation_settings = {
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "n_ctx": 4096,
            "n_batch": 512,
            "n_threads": 4
        }
        
        # Preamble patterns to detect and skip
        self.preamble_patterns = [
            r"^I can help you with that\.?\s*",
            r"^Let me .*?for you\.?\s*",
            r"^Based on .*?:\s*",
            r"^Here's? .*?:\s*",
            r"^Sure,?\s*",
            r"^I'll .*?:\s*",
            r"^To answer your .*?:\s*",
            r"^Certainly!?\s*",
            r"^I understand\.?\s*",
            r"^That's? .*?question\.?\s*",
            r"^That sounds? .*?assist.*?\s*"
        ]
    
    def ensure_model_loaded(self) -> bool:
        """Ensure model is loaded and ready"""
        
        if self.model_loaded and self.model:
            return True
        
        if not LLAMACPP_AVAILABLE:
            self.logger.error("llama-cpp-python not available")
            return False
        
        if not self.model_file.exists():
            self.logger.error(f"Model file not found: {self.model_file}")
            return False
        
        try:
            self.logger.info(f"Loading model: {self.model_file}")
            self.model = Llama(
                model_path=str(self.model_file),
                n_ctx=self.generation_settings["n_ctx"],
                n_batch=self.generation_settings["n_batch"],
                n_threads=self.generation_settings["n_threads"],
                verbose=False
            )
            
            self.model_loaded = True
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False
    
    def generate_complete(self, 
                         prompt: str, 
                         max_tokens: Optional[int] = None,
                         expected_type: str = "default",
                         skip_preamble: bool = True) -> str:
        """Generate complete response with proper token allocation"""
        
        if not self.ensure_model_loaded():
            return "Model not available"
        
        # Determine optimal token count
        if max_tokens is None:
            max_tokens = self._determine_tokens(prompt, expected_type)
        
        # Enhance prompt for better output
        enhanced_prompt = self._enhance_prompt(prompt, expected_type)
        
        # Generate with retry logic
        for attempt in range(2):
            try:
                response = self._generate_raw(enhanced_prompt, max_tokens)
                
                if not response or len(response.strip()) < 10:
                    # Retry with more tokens
                    max_tokens = int(max_tokens * 1.5)
                    continue
                
                # Post-process response
                if skip_preamble:
                    processed = self._skip_preamble(response)
                else:
                    processed = response
                
                # Verify completeness
                if self._is_complete(processed, expected_type):
                    return processed
                else:
                    # Retry with more tokens if incomplete
                    max_tokens = int(max_tokens * 2)
                    continue
                    
            except Exception as e:
                self.logger.error(f"Generation error (attempt {attempt + 1}): {e}")
                
        return "Generation failed after retries"
    
    def _determine_tokens(self, prompt: str, expected_type: str) -> int:
        """Determine optimal token count based on task analysis"""
        
        # Base tokens by type
        type_tokens = {
            "list": 350,
            "daily_plan": 500,
            "json": 300,
            "action": 150,
            "conversation": 200,
            "reflection": 400,
            "number": 80,
            "default": 300  # Increased default
        }
        
        base = type_tokens.get(expected_type, 300)
        
        # Adjust based on prompt keywords
        prompt_lower = prompt.lower()
        
        if any(word in prompt_lower for word in ["list", "enumerate", "steps"]):
            base = max(base, 350)
        if any(word in prompt_lower for word in ["schedule", "plan", "routine"]):
            base = max(base, 450)
        if any(word in prompt_lower for word in ["describe", "explain", "detail"]):
            base = max(base, 400)
        if "daily" in prompt_lower or "hourly" in prompt_lower:
            base = max(base, 500)
        
        # Length adjustment
        if len(prompt) > 200:
            base += 100
        elif len(prompt) > 100:
            base += 50
        
        return min(base, 800)  # Cap at reasonable limit
    
    def _enhance_prompt(self, prompt: str, expected_type: str) -> str:
        """Enhance prompt to get direct, complete answers"""
        
        # Format-specific enhancements
        enhancements = {
            "list": "Provide a complete numbered list. Include all items:",
            "daily_plan": "Create a full schedule with specific times:",
            "action": "State the specific action to take:",
            "conversation": "Provide the complete conversation response:",
            "json": "Output complete valid JSON:",
            "default": "Provide a complete detailed answer:"
        }
        
        enhancement = enhancements.get(expected_type, enhancements["default"])
        
        # Add instructions to prevent truncation
        instructions = [
            f"\n{enhancement}",
            "\nDo not use phrases like 'I can help' or 'Let me' at the start.",
            "\nGive a complete answer, not just the beginning.",
            "\nStart directly with the substantive content."
        ]
        
        return prompt + "".join(instructions)
    
    def _generate_raw(self, prompt: str, max_tokens: int) -> str:
        """Generate raw response from model"""
        
        try:
            output = self.model(
                prompt,
                max_tokens=max_tokens,
                temperature=self.generation_settings["temperature"],
                top_p=self.generation_settings["top_p"],
                top_k=self.generation_settings["top_k"],
                repeat_penalty=self.generation_settings["repeat_penalty"],
                stop=["<|im_end|>", "<|im_start|>", "\n\n\n"],
                echo=False
            )
            
            return output['choices'][0]['text'].strip()
            
        except Exception as e:
            self.logger.error(f"Raw generation error: {e}")
            return ""
    
    def _skip_preamble(self, text: str) -> str:
        """Remove common preamble phrases that add no value"""
        
        if not text:
            return text
        
        original_text = text
        
        # Apply preamble patterns
        for pattern in self.preamble_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # If we removed too much, keep original
        if len(text.strip()) < len(original_text.strip()) * 0.3:
            return original_text.strip()
        
        return text.strip()
    
    def _is_complete(self, text: str, expected_type: str) -> bool:
        """Check if output appears complete for the expected type"""
        
        if not text or len(text) < 15:
            return False
        
        # Type-specific completeness checks
        if expected_type == "list":
            # Should have multiple items or clear structure
            lines = [l.strip() for l in text.split('\n') if l.strip()]
            return len(lines) >= 2 and any(char in text for char in ['1.', '2.', '-', '•'])
        
        elif expected_type == "daily_plan":
            # Should have time references
            return any(time in text.lower() for time in ['am', 'pm', ':', 'morning', 'afternoon', 'evening'])
        
        elif expected_type == "action":
            # Should have action verbs
            return any(verb in text.lower() for verb in ['go', 'do', 'make', 'take', 'get', 'say', 'serve', 'clean', 'greet'])
        
        # General completeness checks
        incomplete_indicators = [
            text.endswith('...'),
            text.endswith(':'),
            text.endswith(','),
            len(text.split()) < 5,
            text.count('(') != text.count(')'),
            text.count('[') != text.count(']')
        ]
        
        return not any(incomplete_indicators)

# Global instance
direct_llm_service = DirectLLMService()
"""Enhanced LLM service with better output handling"""

from typing import Optional, Dict, Any
import re
import logging

class EnhancedLLMService:
    """LLM service with output optimization"""
    
    def __init__(self, model_key="qwen3"):
        self.model_key = model_key
        self.logger = logging.getLogger("enhanced_llm")
        
        # Optimized token settings
        self.token_settings = {
            "min_tokens": 50,      # Minimum to ensure content
            "default_tokens": 200,  # Default for most tasks
            "max_tokens": 500,     # Maximum for complex tasks
            "buffer_tokens": 50    # Extra buffer for completion
        }
        
        # Common preamble patterns to skip
        self.preamble_patterns = [
            r"^I can help you with that\.?\s*",
            r"^Let me .*?for you\.?\s*",
            r"^Based on .*?:\s*",
            r"^Here's? .*?:\s*",
            r"^Sure,?\s*",
            r"^I'll .*?:\s*",
            r"^To answer your .*?:\s*",
            r"^Certainly!?\s*",
        ]
    
    def generate_optimized(self, 
                          prompt: str, 
                          expected_type: str = "default",
                          skip_preamble: bool = True) -> str:
        """Generate with optimized settings"""
        
        # 1. Determine token count based on task type
        token_count = self._determine_token_count(prompt, expected_type)
        
        # 2. Enhance prompt to reduce preamble
        enhanced_prompt = self._enhance_prompt(prompt, expected_type)
        
        # 3. Generate with proper token limit
        raw_output = self._generate_with_retries(
            enhanced_prompt, 
            max_tokens=token_count
        )
        
        # 4. Process output to skip preamble if needed
        if skip_preamble:
            processed_output = self._skip_preamble(raw_output)
        else:
            processed_output = raw_output
        
        # 5. Verify output completeness
        if self._is_incomplete(processed_output):
            self.logger.warning("Output appears incomplete, retrying with more tokens")
            # Retry with more tokens
            raw_output = self._generate_with_retries(
                enhanced_prompt,
                max_tokens=token_count * 2  # Double the tokens
            )
            processed_output = self._skip_preamble(raw_output) if skip_preamble else raw_output
        
        return processed_output
    
    def _determine_token_count(self, prompt: str, expected_type: str) -> int:
        """Determine optimal token count based on task"""
        
        token_requirements = {
            "list": 300,        # Lists need more tokens
            "json": 250,        # JSON structures
            "daily_plan": 400,  # Daily plans are long
            "reflection": 350,  # Reflections need space
            "conversation": 150, # Conversations are shorter
            "action": 100,      # Single actions are brief
            "number": 50,       # Numbers are very brief
            "default": 200      # Default case
        }
        
        base_tokens = token_requirements.get(expected_type, 200)
        
        # Adjust based on prompt complexity
        if "list" in prompt.lower() or "enumerate" in prompt.lower():
            base_tokens = max(base_tokens, 300)
        if "describe" in prompt.lower() or "explain" in prompt.lower():
            base_tokens = max(base_tokens, 250)
        if "plan" in prompt.lower() or "schedule" in prompt.lower():
            base_tokens = max(base_tokens, 350)
        
        return base_tokens + self.token_settings["buffer_tokens"]
    
    def _enhance_prompt(self, prompt: str, expected_type: str) -> str:
        """Enhance prompt to reduce preamble and get direct answers"""
        
        # Add format-specific instructions
        format_instructions = {
            "list": "\nProvide ONLY the list items, no introduction:",
            "json": "\nOutput ONLY valid JSON, no explanation:",
            "action": "\nState ONLY the action to take:",
            "number": "\nProvide ONLY the number:",
            "daily_plan": "\nList ONLY the schedule items with times:",
            "default": "\nProvide a direct answer:"
        }
        
        instruction = format_instructions.get(expected_type, format_instructions["default"])
        
        # Add anti-preamble instruction
        anti_preamble = "\nDo not include phrases like 'I can help' or 'Let me'. Start directly with the answer."
        
        # Combine prompt with instructions
        enhanced = f"{prompt}{instruction}{anti_preamble}"
        
        return enhanced
    
    def _generate_with_retries(self, prompt: str, max_tokens: int, retries: int = 2) -> str:
        """Generate with retry logic"""
        
        from ai_service.ai_service import local_llm_generate
        
        for attempt in range(retries):
            try:
                # Log the generation attempt
                self.logger.debug(f"Generation attempt {attempt + 1} with {max_tokens} tokens")
                
                # Generate with specified tokens
                response = local_llm_generate(
                    prompt,
                    model_key=self.model_key,
                    max_tokens=max_tokens,
                    temperature=0.7,
                    top_p=0.9
                )
                
                if response and len(response.strip()) > 10:  # Minimal valid response
                    return response
                    
            except Exception as e:
                self.logger.error(f"Generation error: {e}")
        
        # Fallback
        return "Unable to generate response"
    
    def _skip_preamble(self, text: str) -> str:
        """Skip common preamble phrases"""
        
        if not text:
            return text
        
        # Try each preamble pattern
        for pattern in self.preamble_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE)
        
        # Also skip first sentence if it's generic
        lines = text.strip().split('\n')
        if len(lines) > 1:
            first_line = lines[0].strip()
            # Check if first line is generic
            generic_indicators = [
                "help", "assist", "let me", "sure", "certainly",
                "i can", "i'll", "based on", "here"
            ]
            
            if any(indicator in first_line.lower() for indicator in generic_indicators):
                # Skip first line
                return '\n'.join(lines[1:]).strip()
        
        return text.strip()
    
    def _is_incomplete(self, text: str) -> bool:
        """Check if output appears incomplete"""
        
        if not text or len(text) < 20:
            return True
        
        # Check for incomplete patterns
        incomplete_indicators = [
            text.endswith("..."),
            text.endswith(":"),
            text.endswith(","),
            text.count('(') != text.count(')'),
            text.count('[') != text.count(']'),
            text.count('{') != text.count('}'),
            text.count('"') % 2 != 0,  # Unmatched quotes
        ]
        
        return any(incomplete_indicators)

# Global instance
enhanced_llm_service = EnhancedLLMService()
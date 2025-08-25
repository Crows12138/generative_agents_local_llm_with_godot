#!/usr/bin/env python
"""
Quick LLM No Cache - Always calls real LLM, no caching
Usage: python quick_llm_nocache.py "Your message here"
"""

import sys
import os
from pathlib import Path

# Ensure we can find ai_service module
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

def get_response(message: str) -> str:
    """Get response from LLM - no caching, always fresh"""
    
    try:
        # Import and use LLM directly
        from ai_service.direct_llm_service import DirectLLMService
        
        # Create or get instance
        llm = DirectLLMService()
        
        # Ensure model is loaded
        if not llm.ensure_model_loaded():
            return "Model loading failed"
        
        # Generate fresh response every time
        prompt = f"User: {message}\nAssistant:"
        response = llm.generate_complete(
            prompt, 
            max_tokens=50,  # Adjust for response length
            expected_type="conversation"
        )
        
        # Clean response
        import re
        response = re.sub(r'[^\x00-\x7F]+', '', response)
        
        # Extract just the first sentence/line
        if "\n" in response:
            response = response.split("\n")[0]
        
        return response.strip()
        
    except Exception as e:
        # If LLM fails, return error message
        return f"LLM Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        
        # Always generate fresh response
        import time
        start = time.time()
        response = get_response(message)
        elapsed = time.time() - start
        
        # Print response
        print(response)
        
        # Optional: print timing to stderr for debugging
        import sys
        print(f"[Generated in {elapsed:.2f}s]", file=sys.stderr)
    else:
        print("Usage: python quick_llm_nocache.py 'Your message here'")
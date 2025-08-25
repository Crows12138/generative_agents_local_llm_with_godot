#!/usr/bin/env python
"""
Quick LLM - Minimal script for fastest possible LLM calls
Usage: python quick_llm.py "Your message here"
"""

import sys
import os
import json
import hashlib
from pathlib import Path

# Simple file-based cache for instant responses
CACHE_FILE = Path("llm_cache.json")
cache = {}

# Load cache if exists
if CACHE_FILE.exists():
    try:
        with open(CACHE_FILE, 'r') as f:
            cache = json.load(f)
    except:
        cache = {}

def get_response(message: str) -> str:
    """Get response with caching"""
    
    # Create hash of message for cache key
    msg_hash = hashlib.md5(message.encode()).hexdigest()
    
    # Check cache first (instant response)
    if msg_hash in cache:
        return cache[msg_hash]
    
    # Try to load model (only if not in cache)
    try:
        # Lazy import to speed up cached responses
        from ai_service.direct_llm_service import DirectLLMService
        
        # Use singleton if available
        if hasattr(DirectLLMService, '_instance'):
            llm = DirectLLMService._instance
        else:
            llm = DirectLLMService()
            DirectLLMService._instance = llm
        
        # Generate response
        prompt = f"User: {message}\nAssistant:"
        response = llm.generate_complete(
            prompt, 
            max_tokens=30,
            expected_type="conversation"
        )
        
        # Remove emojis and special characters
        import re
        response = re.sub(r'[^\x00-\x7F]+', '', response)
        
        # Cache it
        cache[msg_hash] = response
        
        # Save cache
        with open(CACHE_FILE, 'w') as f:
            json.dump(cache, f)
        
        return response
        
    except Exception as e:
        # Fallback for any errors
        simple_responses = {
            "hello": "Hello! How are you today?",
            "how are you": "I'm doing great, thanks for asking!",
            "bye": "Goodbye! See you later!",
            "help": "What can I help you with?",
            "default": "That's interesting! Tell me more."
        }
        
        # Simple keyword matching
        msg_lower = message.lower()
        for key in simple_responses:
            if key in msg_lower:
                return simple_responses[key]
        
        return simple_responses["default"]

if __name__ == "__main__":
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        response = get_response(message)
        # Clean response for safe printing
        import re
        clean_response = re.sub(r'[^\x00-\x7F]+', '', response)
        print(clean_response)
    else:
        print("Usage: python quick_llm.py 'Your message here'")
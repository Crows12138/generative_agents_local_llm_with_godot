"""
LLM Singleton - Keep model loaded in memory for instant responses
Shortest path for Godot to call LLM
"""

import os
import sys
from pathlib import Path

# Singleton instance
_instance = None
_model_loaded = False

class QuickLLM:
    """Ultra-fast LLM wrapper"""
    
    def __init__(self):
        self.model = None
        self.responses_cache = {}
        
    def load_model_lazy(self):
        """Load model only when needed"""
        global _model_loaded
        if _model_loaded:
            return
            
        try:
            from llama_cpp import Llama
            model_path = Path("models/llms/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf")
            
            if model_path.exists():
                print("[QuickLLM] Loading model...")
                self.model = Llama(
                    model_path=str(model_path),
                    n_ctx=512,  # Small context for speed
                    n_batch=8,  # Small batch
                    n_threads=2,  # Few threads for speed
                    verbose=False
                )
                _model_loaded = True
                print("[QuickLLM] Model loaded!")
            else:
                print("[QuickLLM] Model not found, using fallback")
        except:
            print("[QuickLLM] Using simple fallback")
    
    def quick_generate(self, message: str) -> str:
        """Generate response as fast as possible"""
        
        # Check cache first
        if message in self.responses_cache:
            return self.responses_cache[message]
        
        # Try to use model
        if self.model:
            try:
                output = self.model(
                    f"User: {message}\nAssistant:",
                    max_tokens=30,
                    temperature=0.7,
                    stop=["\n"],
                    echo=False
                )
                response = output['choices'][0]['text'].strip()
                
                # Cache it
                self.responses_cache[message] = response
                return response
            except:
                pass
        
        # Fallback responses
        fallbacks = [
            "Hello there!",
            "How can I help you?",
            "Nice to see you!",
            "What brings you here?",
            "Welcome!"
        ]
        
        import random
        return random.choice(fallbacks)

def get_instance():
    """Get or create singleton instance"""
    global _instance
    if _instance is None:
        _instance = QuickLLM()
        _instance.load_model_lazy()
    return _instance

# Direct function for Godot
def quick_response(message: str) -> str:
    """Direct function call for Godot"""
    llm = get_instance()
    return llm.quick_generate(message)

# Command line interface
if __name__ == "__main__":
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        print(quick_response(message))
    else:
        # Interactive mode
        print("QuickLLM Ready! Type 'exit' to quit.")
        while True:
            user_input = input("> ")
            if user_input.lower() == 'exit':
                break
            response = quick_response(user_input)
            print(f"< {response}")
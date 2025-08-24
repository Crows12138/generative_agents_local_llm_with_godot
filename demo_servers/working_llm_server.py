"""
Working LLM Server with Proper Character Responses
"""

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import sys
import os
import time
import threading
import logging
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("working_server")

# Add project path
# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

app = FastAPI()

# Request model
class InteractionRequest(BaseModel):
    message: str

# Server state
class ServerState:
    def __init__(self):
        self.llm = None
        self.loading = False
        self.ready = False
        self.cache = {}

state = ServerState()

# Character data
CHARACTERS = {
    "Bob": {
        "job": "bartender",
        "personality": "friendly and professional",
        "background": "I've been tending this bar for 20 years",
        "fallback": [
            "What can I get you?",
            "Coming right up!",
            "The usual?"
        ]
    },
    "Alice": {
        "job": "regular customer",
        "personality": "thoughtful and relaxed",
        "background": "I'm an artist who comes here to unwind",
        "fallback": [
            "This place is so peaceful.",
            "I love the atmosphere here.",
            "Perfect evening for a drink."
        ]
    },
    "Sam": {
        "job": "jazz musician",
        "personality": "passionate and energetic",
        "background": "Music is my life, especially jazz",
        "fallback": [
            "Jazz is the soul of music!",
            "Ready for some tunes?",
            "The stage is calling!"
        ]
    }
}

def load_llm():
    """Load LLM in background"""
    if state.loading or state.ready:
        return
    
    state.loading = True
    logger.info("[LLM] Starting load...")
    
    try:
        # Import directly from llama_cpp
        from llama_cpp import Llama
        
        model_path = "models/Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf"
        
        if not os.path.exists(model_path):
            logger.error(f"[LLM] Model not found: {model_path}")
            state.loading = False
            return
        
        # Load with minimal settings for speed
        state.llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Smaller context for speed
            n_batch=256,
            n_threads=4,
            verbose=False
        )
        
        # Quick test
        test = state.llm("Hello", max_tokens=5, echo=False)
        if test:
            state.ready = True
            logger.info("[LLM] Ready!")
        else:
            logger.error("[LLM] Test failed")
            
    except Exception as e:
        logger.error(f"[LLM] Load failed: {e}")
    finally:
        state.loading = False

def generate_response(character: str, message: str) -> str:
    """Generate character response using LLM"""
    
    if not state.ready or not state.llm:
        return None
    
    # Check cache
    cache_key = f"{character}:{message[:30]}"
    if cache_key in state.cache:
        return state.cache[cache_key]
    
    char_info = CHARACTERS[character]
    
    # Direct, simple prompt that works with Qwen
    prompt = f"""Human: {message}
{character} ({char_info['job']}): """
    
    try:
        # Generate with minimal processing
        output = state.llm(
            prompt,
            max_tokens=40,
            temperature=0.7,
            top_p=0.9,
            stop=["\n", "Human:", f"{character}:"],
            echo=False
        )
        
        if output and 'choices' in output:
            text = output['choices'][0]['text'].strip()
            
            # Basic cleanup
            if text and len(text) > 3:
                # Remove any remaining meta text
                for bad in ["I'll", "Let me", "I can help", "Based on", "Here's"]:
                    if text.startswith(bad):
                        return None
                
                # Take first sentence
                text = text.split('.')[0] + '.'
                text = text[:60]  # Limit length
                
                # Cache successful response
                state.cache[cache_key] = text
                return text
                
    except Exception as e:
        logger.error(f"[LLM] Generation error: {e}")
    
    return None

@app.on_event("startup")
async def startup():
    """Start LLM loading"""
    thread = threading.Thread(target=load_llm, daemon=True)
    thread.start()

@app.get("/npcs")
async def get_npcs():
    """Get NPC states"""
    return {
        "Bob": {
            "x": 400 + random.randint(-10, 10),
            "y": 80,
            "thought": "Polishing glasses",
            "action": "working"
        },
        "Alice": {
            "x": 200,
            "y": 300,
            "thought": "Enjoying my drink",
            "action": "sitting"
        },
        "Sam": {
            "x": 600 + random.randint(-20, 20),
            "y": 400,
            "thought": "Tuning my guitar",
            "action": "preparing"
        }
    }

@app.post("/interact/{npc_name}")
async def interact(npc_name: str, request: InteractionRequest):
    """Handle interaction"""
    
    if npc_name not in CHARACTERS:
        return {"error": "Unknown NPC"}
    
    start = time.time()
    message = request.message
    
    logger.info(f"[{npc_name}] <- '{message}'")
    
    # Try LLM
    response = None
    using_llm = False
    
    if state.ready:
        response = generate_response(npc_name, message)
        if response:
            using_llm = True
            logger.info(f"[{npc_name}] LLM -> '{response}'")
    
    # Fallback
    if not response:
        response = random.choice(CHARACTERS[npc_name]["fallback"])
        logger.info(f"[{npc_name}] Fallback -> '{response}'")
    
    return {
        "npc": npc_name,
        "response": response,
        "emotion": "friendly",
        "time": time.time() - start,
        "using_llm": using_llm
    }

@app.get("/test_llm")
async def test_llm():
    """Test LLM with each character"""
    
    if not state.ready:
        return {"error": "LLM not ready"}
    
    results = {}
    
    test_messages = [
        "What's your story?",
        "How long have you been here?",
        "What do you enjoy most?"
    ]
    
    for char in CHARACTERS.keys():
        msg = random.choice(test_messages)
        response = generate_response(char, msg)
        results[char] = {
            "question": msg,
            "response": response if response else "Failed"
        }
    
    return {
        "llm_ready": state.ready,
        "results": results
    }

@app.get("/status")
async def status():
    """Server status"""
    return {
        "status": "running",
        "llm_ready": state.ready,
        "llm_loading": state.loading,
        "cache_size": len(state.cache),
        "npcs": list(CHARACTERS.keys())
    }

if __name__ == "__main__":
    port = 8000  # Standard port for Godot integration
    
    print("\n" + "="*50)
    print("WORKING LLM SERVER")
    print("="*50)
    print(f"URL: http://127.0.0.1:{port}")
    print(f"Status: http://127.0.0.1:{port}/status")
    print(f"Test: http://127.0.0.1:{port}/test_llm")
    print("LLM loads in background...")
    print("="*50 + "\n")
    
    uvicorn.run(app, host="127.0.0.1", port=port)
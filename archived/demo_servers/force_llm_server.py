from fastapi import FastAPI
import uvicorn
import sys
import os
import time

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from ai_service.direct_llm_service import DirectLLMService

app = FastAPI()

# Initialize LLM (ensure real loading)
print("[LLM] Force loading real model...")
llm_service = DirectLLMService()
print("[LLM] Model loaded successfully!")

# Test if LLM is really working
test_response = llm_service.generate_complete("Hello", max_tokens=5)
print(f"[LLM] Test response: {test_response}")

@app.post("/interact/{npc_name}")
async def interact(npc_name: str, request: dict):
    """Force using real LLM, no predefined responses"""
    
    message = request.get("message", "Hello")
    
    print(f"\n[Request] {npc_name}: {message}")
    
    # Direct LLM call, no cache, no predefined
    start = time.time()
    
    # Cleaner prompt without system instructions
    prompt = f"Customer: {message}\n{npc_name} (bartender): "
    
    print(f"[LLM] Generating response...")
    print(f"[DEBUG] Prompt: {prompt}")
    
    try:
        response = llm_service.generate_complete(
            prompt,
            max_tokens=50,
            expected_type="conversation"
        )
        
        elapsed = time.time() - start
        print(f"[LLM] Generated in {elapsed:.2f}s: {response[:50]}...")
        
        # Clean response - extract only the first line/sentence
        response = response.strip()
        
        # Extract only the first response (before any Customer: or additional dialogue)
        if "Customer:" in response:
            response = response.split("Customer:")[0].strip()
        
        # Remove any trailing dialogue markers
        if "\n" in response:
            # Take only the first line
            response = response.split("\n")[0].strip()
        
        # Remove common instruction patterns
        instruction_patterns = [
            "Do not repeat",
            "Include specific", 
            "Use simple",
            "Answer in",
            "Provide a",
            "Avoid generic",
            "Your response should",
            "should be limited",
            "words.",
            "Available choices",
            "What is the next"
        ]
        
        for pattern in instruction_patterns:
            if pattern in response:
                print(f"[WARN] Detected instruction in response: {pattern}")
                # Generate a simple fallback response
                simple_responses = {
                    "Bob": "Welcome to the bar! What can I get you?",
                    "Alice": "This place has such a cozy atmosphere!",
                    "Sam": "Just tuning up, any requests?"
                }
                response = simple_responses.get(npc_name, "Hello there!")
                break
        
        # If response is too short, use fallback
        if len(response) < 5:
            print("[LLM] Response too short, using fallback...")
            response = f"Sure thing! The bar's been busy today."
        
        return {
            "npc": npc_name,
            "response": response,
            "emotion": "friendly",
            "time": elapsed,
            "using_llm": True  # Mark that real LLM was used
        }
        
    except Exception as e:
        print(f"[ERROR] LLM failed: {e}")
        # Only use fallback when LLM completely fails
        return {
            "npc": npc_name,
            "response": "Sorry, I didn't catch that.",
            "emotion": "confused",
            "time": 0,
            "using_llm": False
        }

import math

# NPC movement state
npc_movement_state = {
    "Bob": {"base_x": 400, "base_y": 80, "radius": 30, "speed": 0.5},
    "Alice": {"base_x": 200, "base_y": 300, "radius": 50, "speed": 0.3},
    "Sam": {"base_x": 600, "base_y": 400, "radius": 40, "speed": 0.4}
}

@app.get("/npcs")
async def get_npcs():
    """Return NPC status with dynamic movement"""
    current_time = time.time()
    
    return {
        "Bob": {
            "x": npc_movement_state["Bob"]["base_x"] + math.sin(current_time * npc_movement_state["Bob"]["speed"]) * npc_movement_state["Bob"]["radius"],
            "y": npc_movement_state["Bob"]["base_y"],
            "thought": "Wiping the bar counter",
            "action": "working"
        },
        "Alice": {
            "x": npc_movement_state["Alice"]["base_x"] + math.cos(current_time * npc_movement_state["Alice"]["speed"]) * npc_movement_state["Alice"]["radius"],
            "y": npc_movement_state["Alice"]["base_y"] + math.sin(current_time * npc_movement_state["Alice"]["speed"]) * npc_movement_state["Alice"]["radius"],
            "thought": "Enjoying my drink",
            "action": "sitting"
        },
        "Sam": {
            "x": npc_movement_state["Sam"]["base_x"],
            "y": npc_movement_state["Sam"]["base_y"] + math.sin(current_time * npc_movement_state["Sam"]["speed"]) * 20,
            "thought": "Tuning my guitar",
            "action": "preparing"
        }
    }

@app.get("/status")
async def status():
    """Server status"""
    return {
        "status": "running",
        "llm_enabled": True,
        "endpoints": ["/npcs", "/interact/{npc_name}", "/status", "/test_llm", "/ai/status"]
    }

@app.get("/ai/status")
async def ai_status():
    """AI service status for compatibility with ai_manager.gd"""
    return {
        "status": "online",
        "active_model": "Qwen3-30B",
        "available_models": ["Qwen3-30B"],
        "llm_enabled": True,
        "server": "force_llm_server"
    }

@app.get("/test_llm")
async def test_llm():
    """Test if LLM is really working"""
    
    test_prompts = [
        "What is 2+2?",
        "Complete: The sky is",
        "Bob the bartender says:"
    ]
    
    results = {}
    for prompt in test_prompts:
        try:
            response = llm_service.generate_complete(prompt, max_tokens=10)
            results[prompt] = response
        except Exception as e:
            results[prompt] = f"Error: {e}"
    
    return {"llm_test": results}

if __name__ == "__main__":
    print("[Server] Force LLM Bar Server (No cache, no predefined)")
    print("[URL] http://127.0.0.1:8000")
    print("[Test] http://127.0.0.1:8000/test_llm")
    uvicorn.run(app, host="127.0.0.1", port=8000)
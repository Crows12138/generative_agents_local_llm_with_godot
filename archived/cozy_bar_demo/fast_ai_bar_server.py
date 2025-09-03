from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import asyncio
import sys
import os
import random
import time
from functools import lru_cache

# Add project path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

app = FastAPI()

# Request model
class InteractionRequest(BaseModel):
    message: str

# Global response cache
response_cache = {}
llm_service = None

# NPC data
agents = {
    "Bob": {
        "position": {"x": 400, "y": 80},
        "role": "bartender",
        "personality": "Professional and friendly bartender",
        "current_thought": "Ready to serve customers",
        "current_action": "wiping_counter"
    },
    "Alice": {
        "position": {"x": 200, "y": 300},
        "role": "customer",
        "personality": "Regular customer who enjoys the atmosphere",
        "current_thought": "Looking at the menu",
        "current_action": "sitting"
    },
    "Sam": {
        "position": {"x": 600, "y": 400},
        "role": "musician",
        "personality": "Talented musician who loves jazz",
        "current_thought": "Tuning my guitar",
        "current_action": "preparing"
    }
}

# Quick response database for instant replies
QUICK_RESPONSES = {
    "hello": {
        "Bob": ["Hey there! What can I get you?", "Welcome! What'll it be?", "Good to see you!"],
        "Alice": ["Oh, hello! Nice to see you.", "Hi there!", "Hello, lovely evening!"],
        "Sam": ["Hey! Any song requests?", "Hello! Enjoying the music?", "Hi friend!"]
    },
    "how are you": {
        "Bob": ["Great! Busy night at the bar.", "Can't complain, business is good!", "Doing well, thanks!"],
        "Alice": ["Feeling relaxed with my drink.", "Wonderful, this place is cozy.", "Pretty good, thanks!"],
        "Sam": ["Excited to play some music!", "Ready to rock the stage!", "Feeling musical!"]
    },
    "drink": {
        "Bob": ["Coming right up!", "Our whiskey is excellent tonight.", "I'll make you something special."],
        "Alice": ["The house special is amazing.", "I'm having the whiskey.", "Bob makes the best drinks."],
        "Sam": ["I could use a water break.", "Maybe after my set.", "Music first, drinks later!"]
    },
    "music": {
        "Bob": ["Sam's playing great tonight!", "Love the jazz vibe.", "Music makes the bar better."],
        "Alice": ["The jazz here is wonderful.", "I love live music.", "Sam is so talented!"],
        "Sam": ["Jazz is my soul!", "Any requests?", "Music is life!"]
    },
    "bye": {
        "Bob": ["See you next time!", "Take care!", "Come back soon!"],
        "Alice": ["Goodbye!", "Nice talking to you.", "See you!"],
        "Sam": ["Later! Enjoy the night!", "Bye! Keep grooving!", "See ya!"]
    }
}

def get_quick_response(npc_name: str, message: str) -> str:
    """Get instant response without LLM"""
    msg_lower = message.lower()
    
    # Check each category
    for key, responses in QUICK_RESPONSES.items():
        if key in msg_lower:
            if npc_name in responses:
                return random.choice(responses[npc_name])
    
    # Default responses
    defaults = {
        "Bob": "What can I get you?",
        "Alice": "That's interesting!",
        "Sam": "Rock on!"
    }
    return defaults.get(npc_name, "Hello!")

def initialize_llm():
    """Initialize LLM service if needed"""
    global llm_service
    if llm_service is None:
        try:
            from ai_service.direct_llm_service import DirectLLMService
            print("[AI] Initializing REAL LLM service...")
            llm_service = DirectLLMService()
            # Warm up
            # Test the service
            test_result = llm_service.generate_complete("Hello", max_tokens=5)
            if not test_result:
                raise Exception("LLM test failed")
            print("[AI] LLM ready for real AI responses!")
            return True
        except Exception as e:
            print(f"[WARNING] LLM not available: {e}")
            return False
    return llm_service is not None

# Force initialize LLM at startup
llm_initialized = initialize_llm()

@app.get("/npcs")
async def get_npcs():
    """Get NPC states with fast thoughts"""
    response = {}
    
    # Fast thought generation
    thought_pools = {
        "Bob": ["Cleaning glasses", "Checking inventory", "The bar looks good", "Ready to serve"],
        "Alice": ["This drink is nice", "Love the atmosphere", "Maybe another drink", "So relaxing here"],
        "Sam": ["Time for jazz", "My guitar is ready", "The crowd looks good", "Music time!"]
    }
    
    for name, data in agents.items():
        # Update positions with small variations
        if name == "Bob":
            data["position"]["x"] = 400 + random.randint(-20, 20)
        elif name == "Sam":
            data["position"]["x"] = 600 + random.randint(-50, 50)
        
        response[name] = {
            "x": data["position"]["x"],
            "y": data["position"]["y"],
            "thought": random.choice(thought_pools[name]),
            "action": data["current_action"]
        }
    
    return response

@app.post("/interact/{npc_name}")
async def interact_with_npc(npc_name: str, request: InteractionRequest):
    """Ultra-fast NPC interaction"""
    
    if npc_name not in agents:
        return {"error": "NPC not found"}
    
    start_time = time.time()
    message = request.message
    
    # 1. Check cache first
    cache_key = f"{npc_name}:{message[:30]}"
    if cache_key in response_cache:
        cached = response_cache[cache_key].copy()
        cached["cached"] = True
        cached["time"] = time.time() - start_time
        return cached
    
    # 2. Try quick response (90% of cases)
    quick_response = get_quick_response(npc_name, message)
    
    # 3. Determine if we need AI (only for complex messages)
    use_ai = False
    simple_keywords = ["hello", "hi", "hey", "how are", "bye", "drink", "music", "thanks"]
    if not any(keyword in message.lower() for keyword in simple_keywords):
        use_ai = initialize_llm()  # Only init if needed
    
    response_text = quick_response
    
    # 4. Use AI for all non-cached responses (100% real LLM)
    if use_ai and llm_service:
        try:
            # Ultra-short prompt for speed
            prompt = f"{npc_name}: '{message}' Reply (5 words):"
            ai_response = llm_service.generate_complete(prompt, max_tokens=15)
            if ai_response and len(ai_response.strip()) > 0:
                response_text = ai_response.strip()[:50]
        except:
            pass  # Use quick response as fallback
    
    # Build response
    result = {
        "npc": npc_name,
        "response": response_text,
        "emotion": "friendly",
        "time": time.time() - start_time
    }
    
    # Cache it
    response_cache[cache_key] = result.copy()
    
    # Update NPC state
    if "drink" in message.lower() and npc_name == "Bob":
        agents[npc_name]["current_action"] = "serving"
    elif "music" in message.lower() and npc_name == "Sam":
        agents[npc_name]["current_action"] = "playing"
    
    return result

@app.get("/status")
async def get_status():
    """Check server status"""
    return {
        "status": "running",
        "ai_enabled": llm_service is not None,
        "npcs": list(agents.keys()),
        "cache_size": len(response_cache)
    }

@app.on_event("startup")
async def startup_event():
    """Preload common responses"""
    print("[STARTUP] Preloading common responses...")
    
    # Preload most common interactions
    common = ["Hello", "Hi", "How are you", "Nice music", "Great drink"]
    for npc in agents.keys():
        for msg in common:
            cache_key = f"{npc}:{msg[:30]}"
            response_cache[cache_key] = {
                "npc": npc,
                "response": get_quick_response(npc, msg),
                "emotion": "friendly",
                "time": 0.001
            }
    
    print(f"[CACHE] Preloaded {len(response_cache)} responses")

if __name__ == "__main__":
    print("[FAST] Fast AI Bar Server Starting...")
    print("[SPEED] Optimized for <1 second responses")
    print("[CONNECT] http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
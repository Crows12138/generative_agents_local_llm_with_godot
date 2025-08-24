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
    
    # Simple direct prompt
    prompt = f"""Character: {npc_name} (bartender at a cozy bar)
Customer asks: {message}
{npc_name} responds:"""
    
    print(f"[LLM] Generating response...")
    
    try:
        response = llm_service.generate_complete(
            prompt,
            max_tokens=50
        )
        
        elapsed = time.time() - start
        print(f"[LLM] Generated in {elapsed:.2f}s: {response[:50]}...")
        
        # Clean response
        response = response.strip()
        
        # If response is too short or contains errors, retry
        if len(response) < 5:
            print("[LLM] Response too short, retrying...")
            response = llm_service.generate_complete(
                f"{npc_name} says: ",
                max_tokens=30
            )
        
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
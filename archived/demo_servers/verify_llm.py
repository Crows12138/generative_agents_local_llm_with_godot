"""Verify if LLM is really generating responses"""

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from ai_service.direct_llm_service import DirectLLMService
import time

print("Testing Direct LLM Service...")

# Initialize
llm = DirectLLMService()

# Test various prompts
test_cases = [
    ("Simple math", "What is 25 + 37?"),
    ("Roleplay", "You are Bob the bartender. A customer asks about your childhood. You respond:"),
    ("Creative", "Write a haiku about whiskey:"),
    ("Conversation", "Customer: Tell me about yourself. Bartender:"),
]

for name, prompt in test_cases:
    print(f"\n[{name}]")
    print(f"Prompt: {prompt}")
    
    start = time.time()
    response = llm.generate_complete(prompt, max_tokens=50)
    elapsed = time.time() - start
    
    print(f"Response ({elapsed:.2f}s): {response}")
    print("-" * 50)

print("\nLLM is working!" if len(response) > 0 else "LLM not working!")
"""
Dialogue Generation Load Test
Continuously sends dialogue requests to measure GPU utilization
"""
import requests
import time
import threading
import random

DIALOGUE_PROMPTS = [
    "Tell me about your childhood memories",
    "What's your opinion on the weather today?",
    "Describe your favorite meal in detail",
    "What are your plans for tomorrow?",
    "Tell me a story about an adventure",
    "How do you feel about living in this town?",
    "What's the most interesting thing that happened this week?",
    "Describe your ideal vacation",
    "What skills are you most proud of?",
    "Tell me about your friends and family",
    "What makes you happy?",
    "Describe a typical day in your life",
    "What are your biggest fears?",
    "Tell me about your hobbies",
    "What would you change about your life?"
]

def send_dialogue_request(prompt, npc_name="TestNPC"):
    """Send a single dialogue request"""
    try:
        response = requests.post(
            "http://localhost:8020/v1/chat/completions",
            json={
                "model": "Llama-3.2-3B-Instruct.Q4_0.gguf",
                "messages": [
                    {"role": "system", "content": f"You are {npc_name}, a resident of a small town."},
                    {"role": "user", "content": prompt}
                ],
                "stream": True,
                "max_tokens": 150
            },
            stream=True,
            timeout=30
        )
        
        # Consume the stream to ensure GPU is working
        tokens = 0
        for line in response.iter_lines():
            if line:
                tokens += 1
        
        print(f"[{npc_name}] Completed response ({tokens} chunks)")
        return True
    except Exception as e:
        print(f"[{npc_name}] Request failed: {e}")
        return False

def continuous_dialogue_test(duration=30, concurrent_npcs=1):
    """Run continuous dialogue test with specified concurrency"""
    print(f"\nStarting dialogue load test...")
    print(f"Duration: {duration} seconds")
    print(f"Concurrent NPCs: {concurrent_npcs}")
    print("-" * 40)
    
    start_time = time.time()
    request_count = 0
    success_count = 0
    
    def worker(npc_id):
        nonlocal request_count, success_count
        while time.time() - start_time < duration:
            prompt = random.choice(DIALOGUE_PROMPTS)
            request_count += 1
            if send_dialogue_request(prompt, f"NPC_{npc_id}"):
                success_count += 1
            time.sleep(1)  # Small delay between requests
    
    # Start worker threads
    threads = []
    for i in range(concurrent_npcs):
        t = threading.Thread(target=worker, args=(i,))
        t.start()
        threads.append(t)
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Print summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 40)
    print(f"Test completed in {elapsed:.1f} seconds")
    print(f"Total requests: {request_count}")
    print(f"Successful: {success_count}")
    print(f"Failed: {request_count - success_count}")
    print(f"Avg requests/sec: {request_count/elapsed:.2f}")
    print("=" * 40)

if __name__ == "__main__":
    import sys
    
    # Check if server is running
    try:
        response = requests.get("http://localhost:8020/health", timeout=2)
        print("Server is running!")
    except:
        print("ERROR: gpt4all_server.py is not running on port 8020")
        print("Please start it first: python gpt4all_server.py")
        sys.exit(1)
    
    # Parse arguments
    duration = 30
    concurrent = 1
    
    if len(sys.argv) > 1:
        duration = int(sys.argv[1])
    if len(sys.argv) > 2:
        concurrent = int(sys.argv[2])
    
    continuous_dialogue_test(duration, concurrent)
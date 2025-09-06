"""
Decision Making Load Test
Continuously sends decision requests to measure GPU utilization
"""
import requests
import time
import threading
import random
import json

# Different scenarios for decision making
LOCATIONS = ["bar", "home", "street", "park", "store", "restaurant", "office"]
TIMES = ["morning", "afternoon", "evening", "night"]
ACTIONS = ["idle", "walking", "talking", "working", "eating", "sleeping", "shopping"]
NPC_NAMES = ["John", "Maria", "Bob", "Alice", "Tom", "Sarah", "Mike", "Lisa"]

def generate_state():
    """Generate a random state for decision making"""
    return {
        "npc_name": random.choice(NPC_NAMES),
        "location": random.choice(LOCATIONS),
        "time": random.choice(TIMES),
        "nearby_npcs": random.sample(NPC_NAMES, random.randint(0, 3)),
        "player_nearby": random.choice([True, False]),
        "current_action": random.choice(ACTIONS),
        "energy": random.randint(20, 100),
        "mood": random.choice(["happy", "neutral", "sad", "angry", "excited"])
    }

def send_decision_request(state):
    """Send a single decision request"""
    try:
        response = requests.post(
            "http://localhost:8021/state_int",
            json=state,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            print(f"[{state['npc_name']}] Decision: {result.get('action', 'unknown')}")
            return True
        else:
            print(f"[{state['npc_name']}] Request failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[{state['npc_name']}] Request error: {e}")
        return False

def continuous_decision_test(duration=30, concurrent_npcs=1):
    """Run continuous decision test with specified concurrency"""
    print(f"\nStarting decision load test...")
    print(f"Duration: {duration} seconds")
    print(f"Concurrent NPCs: {concurrent_npcs}")
    print("-" * 40)
    
    start_time = time.time()
    request_count = 0
    success_count = 0
    
    def worker(worker_id):
        nonlocal request_count, success_count
        while time.time() - start_time < duration:
            state = generate_state()
            state["npc_name"] = f"Worker{worker_id}_{state['npc_name']}"
            request_count += 1
            if send_decision_request(state):
                success_count += 1
            time.sleep(0.5)  # Faster than dialogue since decisions are quicker
    
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

def test_batch_decisions(batch_size=10):
    """Test sending a batch of decision requests quickly"""
    print(f"\nSending batch of {batch_size} decision requests...")
    
    start_time = time.time()
    success_count = 0
    
    for i in range(batch_size):
        state = generate_state()
        state["npc_name"] = f"Batch_{i}_{state['npc_name']}"
        if send_decision_request(state):
            success_count += 1
    
    elapsed = time.time() - start_time
    print(f"\nBatch completed in {elapsed:.2f} seconds")
    print(f"Success rate: {success_count}/{batch_size}")
    print(f"Avg time per decision: {elapsed/batch_size:.3f} seconds")

if __name__ == "__main__":
    import sys
    
    # Check if server is running
    try:
        test_state = generate_state()
        response = requests.post("http://localhost:8021/state_int", json=test_state, timeout=5)
        print("Decision server is running!")
    except:
        print("ERROR: decision_server.py is not running on port 8021")
        print("Please start it first: python decision_server.py")
        sys.exit(1)
    
    # Parse arguments
    mode = "continuous"
    duration = 30
    concurrent = 1
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "batch":
            mode = "batch"
            batch_size = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        else:
            duration = int(sys.argv[1])
            if len(sys.argv) > 2:
                concurrent = int(sys.argv[2])
    
    if mode == "batch":
        test_batch_decisions(batch_size)
    else:
        continuous_decision_test(duration, concurrent)
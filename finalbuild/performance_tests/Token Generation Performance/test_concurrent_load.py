"""
Concurrent Workload Test
Simulates realistic game scenario with multiple NPCs using both dialogue and decision systems
"""
import requests
import time
import threading
import random
import json
from datetime import datetime

# Configuration
DIALOGUE_PORT = 8020
DECISION_PORT = 8021

# Test data
DIALOGUE_PROMPTS = [
    "What are you doing right now?",
    "How do you feel about the player?",
    "Tell me about this place",
    "What happened earlier today?",
    "What are your plans?",
]

LOCATIONS = ["bar", "home", "street", "park", "store"]
TIMES = ["morning", "afternoon", "evening", "night"]

class NPC:
    """Simulates an NPC with both dialogue and decision capabilities"""
    
    def __init__(self, name, location="bar"):
        self.name = name
        self.location = location
        self.current_action = "idle"
        self.energy = 80
        self.dialogue_count = 0
        self.decision_count = 0
        self.active = True
        
    def make_decision(self):
        """NPC makes a decision about next action"""
        try:
            state = {
                "npc_name": self.name,
                "location": self.location,
                "time": random.choice(TIMES),
                "nearby_npcs": [],
                "player_nearby": random.choice([True, False]),
                "current_action": self.current_action,
                "energy": self.energy
            }
            
            response = requests.post(
                f"http://localhost:{DECISION_PORT}/state_int",
                json=state,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                self.current_action = result.get("action", "idle")
                self.decision_count += 1
                print(f"[{self.name}] Decision: {self.current_action}")
                return True
        except Exception as e:
            print(f"[{self.name}] Decision failed: {e}")
        return False
    
    def generate_dialogue(self):
        """NPC generates dialogue"""
        try:
            prompt = random.choice(DIALOGUE_PROMPTS)
            response = requests.post(
                f"http://localhost:{DIALOGUE_PORT}/v1/chat/completions",
                json={
                    "model": "Llama-3.2-3B-Instruct.Q4_0.gguf",
                    "messages": [
                        {"role": "system", "content": f"You are {self.name}, currently {self.current_action} at {self.location}."},
                        {"role": "user", "content": prompt}
                    ],
                    "stream": True,
                    "max_tokens": 100
                },
                stream=True,
                timeout=30
            )
            
            # Consume stream
            chunks = 0
            for line in response.iter_lines():
                if line:
                    chunks += 1
            
            self.dialogue_count += 1
            print(f"[{self.name}] Dialogue completed ({chunks} chunks)")
            return True
        except Exception as e:
            print(f"[{self.name}] Dialogue failed: {e}")
        return False
    
    def simulate_behavior(self, duration):
        """Simulate realistic NPC behavior"""
        start_time = time.time()
        
        while self.active and (time.time() - start_time < duration):
            # 70% chance to make a decision
            if random.random() < 0.7:
                self.make_decision()
                time.sleep(random.uniform(0.5, 1.5))
            
            # 30% chance to generate dialogue (if player nearby)
            if random.random() < 0.3:
                self.generate_dialogue()
                time.sleep(random.uniform(1, 3))
            
            # Update energy
            self.energy = max(10, self.energy - random.randint(1, 5))
            
            # Small random delay
            time.sleep(random.uniform(0.5, 2))
        
        return {
            "name": self.name,
            "decisions": self.decision_count,
            "dialogues": self.dialogue_count
        }

def simulate_game_scenario(num_npcs=3, duration=30):
    """Simulate a realistic game scenario with multiple NPCs"""
    print(f"\n{'='*60}")
    print(f"CONCURRENT WORKLOAD TEST - GAME SCENARIO")
    print(f"{'='*60}")
    print(f"NPCs: {num_npcs}")
    print(f"Duration: {duration} seconds")
    print(f"Start time: {datetime.now()}")
    print(f"{'='*60}\n")
    
    # Create NPCs
    npcs = [
        NPC(f"NPC_{i}", random.choice(LOCATIONS))
        for i in range(num_npcs)
    ]
    
    # Start NPC threads
    threads = []
    results = []
    
    def run_npc(npc):
        result = npc.simulate_behavior(duration)
        results.append(result)
    
    for npc in npcs:
        t = threading.Thread(target=run_npc, args=(npc,))
        t.start()
        threads.append(t)
        time.sleep(0.5)  # Stagger starts slightly
    
    # Monitor progress
    start = time.time()
    while time.time() - start < duration:
        elapsed = time.time() - start
        print(f"\rProgress: {elapsed:.1f}/{duration}s", end="")
        time.sleep(1)
    
    # Wait for completion
    for t in threads:
        t.join()
    
    # Print summary
    print(f"\n\n{'='*60}")
    print("TEST RESULTS")
    print(f"{'='*60}")
    
    total_decisions = sum(r["decisions"] for r in results)
    total_dialogues = sum(r["dialogues"] for r in results)
    
    for result in results:
        print(f"{result['name']}: {result['decisions']} decisions, {result['dialogues']} dialogues")
    
    print(f"\nTOTALS:")
    print(f"  Decisions: {total_decisions} ({total_decisions/duration:.2f}/sec)")
    print(f"  Dialogues: {total_dialogues} ({total_dialogues/duration:.2f}/sec)")
    print(f"  Total operations: {total_decisions + total_dialogues}")
    print(f"{'='*60}")

def stress_test(num_npcs=5, duration=60):
    """Run a stress test with many NPCs"""
    print(f"\n{'='*60}")
    print(f"STRESS TEST - MAXIMUM LOAD")
    print(f"{'='*60}")
    
    simulate_game_scenario(num_npcs, duration)

if __name__ == "__main__":
    import sys
    
    # Check if both servers are running
    servers_ok = True
    
    try:
        requests.get(f"http://localhost:{DIALOGUE_PORT}/health", timeout=2)
        print(f"Dialogue server (port {DIALOGUE_PORT}) is running")
    except:
        print(f"ERROR: Dialogue server not running on port {DIALOGUE_PORT}")
        servers_ok = False
    
    try:
        test_state = {"npc_name": "test", "location": "test", "time": "test"}
        requests.post(f"http://localhost:{DECISION_PORT}/state_int", json=test_state, timeout=2)
        print(f"Decision server (port {DECISION_PORT}) is running")
    except:
        print(f"ERROR: Decision server not running on port {DECISION_PORT}")
        servers_ok = False
    
    if not servers_ok:
        print("\nPlease start both servers:")
        print("  python gpt4all_server.py")
        print("  python decision_server.py")
        sys.exit(1)
    
    print("\nServers ready!\n")
    
    # Parse arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "stress":
            npcs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
            duration = int(sys.argv[3]) if len(sys.argv) > 3 else 60
            stress_test(npcs, duration)
        else:
            npcs = int(sys.argv[1]) if len(sys.argv) > 1 else 3
            duration = int(sys.argv[2]) if len(sys.argv) > 2 else 30
            simulate_game_scenario(npcs, duration)
    else:
        # Default: 3 NPCs for 30 seconds
        simulate_game_scenario(3, 30)
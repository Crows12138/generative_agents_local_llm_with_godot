"""
GPU Benchmark Suite for Generative Agent System
Tests different scenarios for paper reporting
"""
import subprocess
import time
import statistics
import threading
import requests
import json
from datetime import datetime
import sys

class GPUMonitor:
    def __init__(self):
        self.monitoring = False
        self.utilizations = []
        self.memories = []
        self.temps = []
        
    def start_monitoring(self):
        """Start monitoring in background thread"""
        self.monitoring = True
        self.utilizations = []
        self.memories = []
        self.temps = []
        self.thread = threading.Thread(target=self._monitor_loop)
        self.thread.start()
        
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,name',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                line = result.stdout.strip()
                parts = line.split(', ')
                if len(parts) >= 5:
                    util = int(parts[0])
                    mem_used = int(parts[1])
                    mem_total = int(parts[2])
                    temp = int(parts[3])
                    gpu_name = parts[4]
                    
                    self.utilizations.append(util)
                    self.memories.append(mem_used)
                    self.temps.append(temp)
                    
                    mem_percent = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                    print(f"\r[{gpu_name}] GPU: {util:3}% | Mem: {mem_used:5}MB ({mem_percent:.1f}%) | Temp: {temp}C", end='')
            
            time.sleep(0.5)
    
    def stop_monitoring(self):
        """Stop monitoring and return results"""
        self.monitoring = False
        self.thread.join()
        return self.get_stats()
    
    def get_stats(self):
        """Calculate and return statistics"""
        if not self.utilizations:
            return None
            
        stats = {
            'gpu_util_min': min(self.utilizations),
            'gpu_util_max': max(self.utilizations),
            'gpu_util_avg': statistics.mean(self.utilizations),
            'gpu_util_std': statistics.stdev(self.utilizations) if len(self.utilizations) > 1 else 0,
            'mem_min': min(self.memories),
            'mem_max': max(self.memories),
            'mem_avg': statistics.mean(self.memories),
            'temp_avg': statistics.mean(self.temps),
            'samples': len(self.utilizations)
        }
        return stats

def print_results(scenario, stats):
    """Pretty print test results"""
    print(f"\n\n{'='*60}")
    print(f"SCENARIO: {scenario}")
    print(f"{'='*60}")
    print(f"GPU Utilization:")
    print(f"  Range: {stats['gpu_util_min']}-{stats['gpu_util_max']}%")
    print(f"  Average: {stats['gpu_util_avg']:.1f}%")
    print(f"  StdDev: {stats['gpu_util_std']:.1f}%")
    print(f"\nMemory Usage:")
    print(f"  Range: {stats['mem_min']}-{stats['mem_max']}MB")
    print(f"  Average: {stats['mem_avg']:.1f}MB")
    print(f"\nTemperature: {stats['temp_avg']:.1f}C")
    print(f"Samples: {stats['samples']}")
    print(f"{'='*60}")
    
    # Generate LaTeX table row
    print(f"\nLaTeX Table Row:")
    print(f"{scenario} & {stats['gpu_util_min']}-{stats['gpu_util_max']}\\% & {stats['gpu_util_avg']:.1f}\\% & {stats['mem_avg']:.1f}MB \\\\")
    
    return stats

def test_idle(duration=10):
    """Test 1: Idle state with models loaded"""
    print("\n[TEST 1] Measuring idle state (models loaded but inactive)...")
    print("Make sure servers are running but NOT processing requests")
    
    monitor = GPUMonitor()
    monitor.start_monitoring()
    time.sleep(duration)
    stats = monitor.stop_monitoring()
    
    return print_results("Idle (Models Loaded)", stats)

def test_dialogue_generation(duration=30):
    """Test 2: Dialogue generation with Llama 3.2"""
    print("\n[TEST 2] Testing dialogue generation (Llama 3.2)...")
    print("Make sure gpt4all_server.py is running on port 8020")
    
    monitor = GPUMonitor()
    monitor.start_monitoring()
    
    # Send multiple dialogue requests
    def send_dialogue_requests():
        queries = [
            "Tell me about your day at the bar",
            "What's your favorite drink to make?",
            "Describe the atmosphere in this place",
            "What kind of customers do you usually get?",
            "Tell me an interesting story from last night"
        ]
        
        for query in queries:
            try:
                response = requests.post(
                    "http://localhost:8020/v1/chat/completions",
                    json={
                        "model": "Llama-3.2-3B-Instruct.Q4_0.gguf",
                        "messages": [
                            {"role": "system", "content": "You are a friendly bartender."},
                            {"role": "user", "content": query}
                        ],
                        "stream": True
                    },
                    stream=True
                )
                # Consume the stream
                for line in response.iter_lines():
                    if line:
                        pass
                time.sleep(2)  # Brief pause between requests
            except Exception as e:
                print(f"Request failed: {e}")
    
    # Start dialogue requests in background
    request_thread = threading.Thread(target=send_dialogue_requests)
    request_thread.start()
    
    time.sleep(duration)
    stats = monitor.stop_monitoring()
    request_thread.join()
    
    return print_results("Dialogue Generation", stats)

def test_decision_making(duration=30):
    """Test 3: Decision making with Phi-3"""
    print("\n[TEST 3] Testing decision making (Phi-3)...")
    print("Make sure decision_server.py is running on port 8021")
    
    monitor = GPUMonitor()
    monitor.start_monitoring()
    
    # Send multiple decision requests
    def send_decision_requests():
        for i in range(15):
            try:
                # Test state_int endpoint
                response = requests.post(
                    "http://localhost:8021/state_int",
                    json={
                        "npc_name": f"TestNPC_{i}",
                        "location": "bar",
                        "time": "evening",
                        "nearby_npcs": ["John", "Maria"],
                        "player_nearby": True,
                        "current_action": "idle"
                    }
                )
                time.sleep(2)  # Pause between requests
            except Exception as e:
                print(f"Decision request failed: {e}")
    
    # Start decision requests in background
    request_thread = threading.Thread(target=send_decision_requests)
    request_thread.start()
    
    time.sleep(duration)
    stats = monitor.stop_monitoring()
    request_thread.join()
    
    return print_results("Decision Making", stats)

def test_concurrent_workload(duration=30):
    """Test 4: Both models running concurrently (3 NPCs)"""
    print("\n[TEST 4] Testing concurrent workload (3 NPCs active)...")
    print("Make sure both servers are running")
    
    monitor = GPUMonitor()
    monitor.start_monitoring()
    
    # Simulate 3 NPCs with mixed workload
    def simulate_npc_activity():
        for round in range(5):
            # Each round simulates 3 NPCs
            threads = []
            
            # NPC 1: Dialogue
            def npc1():
                try:
                    requests.post(
                        "http://localhost:8020/v1/chat/completions",
                        json={
                            "model": "Llama-3.2-3B-Instruct.Q4_0.gguf",
                            "messages": [
                                {"role": "user", "content": f"NPC1 action {round}"}
                            ],
                            "stream": True
                        },
                        stream=True
                    )
                except:
                    pass
            
            # NPC 2: Decision
            def npc2():
                try:
                    requests.post(
                        "http://localhost:8021/state_int",
                        json={
                            "npc_name": "NPC2",
                            "location": "street",
                            "time": "night"
                        }
                    )
                except:
                    pass
            
            # NPC 3: Both
            def npc3():
                try:
                    # Decision first
                    requests.post(
                        "http://localhost:8021/state_int",
                        json={
                            "npc_name": "NPC3",
                            "location": "home",
                            "time": "morning"
                        }
                    )
                    # Then dialogue
                    requests.post(
                        "http://localhost:8020/v1/chat/completions",
                        json={
                            "model": "Llama-3.2-3B-Instruct.Q4_0.gguf",
                            "messages": [
                                {"role": "user", "content": f"NPC3 speaks {round}"}
                            ],
                            "stream": True
                        },
                        stream=True
                    )
                except:
                    pass
            
            # Run all NPCs concurrently
            t1 = threading.Thread(target=npc1)
            t2 = threading.Thread(target=npc2)
            t3 = threading.Thread(target=npc3)
            
            t1.start()
            t2.start()
            t3.start()
            
            t1.join()
            t2.join()
            t3.join()
            
            time.sleep(3)  # Pause between rounds
    
    # Start NPC simulation
    sim_thread = threading.Thread(target=simulate_npc_activity)
    sim_thread.start()
    
    time.sleep(duration)
    stats = monitor.stop_monitoring()
    sim_thread.join()
    
    return print_results("Full System (3 NPCs)", stats)

def run_full_benchmark():
    """Run complete benchmark suite"""
    print("="*60)
    print("GPU BENCHMARK SUITE FOR GENERATIVE AGENTS")
    print("="*60)
    print(f"Start Time: {datetime.now()}")
    
    results = {}
    
    # Test 1: Idle
    print("\nStarting IDLE test...")
    time.sleep(2)
    results['idle'] = test_idle(10)
    
    # Test 2: Dialogue
    print("\nStarting DIALOGUE test...")
    time.sleep(2)
    results['dialogue'] = test_dialogue_generation(20)
    
    # Test 3: Decision
    print("\nStarting DECISION test...")
    time.sleep(2)
    results['decision'] = test_decision_making(20)
    
    # Test 4: Concurrent
    print("\nStarting CONCURRENT test...")
    time.sleep(2)
    results['concurrent'] = test_concurrent_workload(30)
    
    # Summary
    print("\n\n" + "="*60)
    print("BENCHMARK SUMMARY - LaTeX Table Format")
    print("="*60)
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{GPU Utilization Under Different Workloads}")
    print("\\begin{tabular}{lccc}")
    print("\\hline")
    print("\\textbf{Scenario} & \\textbf{Min-Max} & \\textbf{Average} & \\textbf{Memory} \\\\")
    print("\\hline")
    
    for scenario, stats in results.items():
        if stats:
            scenario_name = {
                'idle': 'Idle (Models Loaded)',
                'dialogue': 'Dialogue Generation',
                'decision': 'Decision Making',
                'concurrent': 'Full System (3 NPCs)'
            }.get(scenario, scenario)
            
            print(f"{scenario_name} & {stats['gpu_util_min']}-{stats['gpu_util_max']}\\% & "
                  f"{stats['gpu_util_avg']:.1f}\\% & {stats['mem_avg']:.0f}MB \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")
    
    # Save results to file
    with open("gpu_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2, default=float)
    print(f"\nResults saved to gpu_benchmark_results.json")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "idle":
            test_idle()
        elif sys.argv[1] == "dialogue":
            test_dialogue_generation()
        elif sys.argv[1] == "decision":
            test_decision_making()
        elif sys.argv[1] == "concurrent":
            test_concurrent_workload()
        elif sys.argv[1] == "full":
            run_full_benchmark()
        else:
            print("Usage: python gpu_benchmark_suite.py [idle|dialogue|decision|concurrent|full]")
    else:
        run_full_benchmark()
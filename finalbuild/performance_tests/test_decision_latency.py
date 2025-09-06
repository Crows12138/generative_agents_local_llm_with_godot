"""
Decision Server Latency Test
Tests actual response times for the lightweight decision model
"""
import time
import asyncio
import websockets
import json
import statistics
import requests

async def test_decision_websocket():
    """Test decision server via WebSocket"""
    uri = "ws://127.0.0.1:9998"  # Decision server port
    
    print("Testing Decision Server WebSocket Latency")
    print("="*50)
    
    # Test different state complexities
    test_states = [0, 1, 15, 63, 127, 255]  # Different complexity levels
    
    all_results = {}
    
    for state in test_states:
        times = []
        errors = 0
        
        print(f"\nTesting State {state} (Binary: {bin(state)[2:].zfill(8)}):")
        
        for i in range(10):
            try:
                start = time.time()
                
                async with websockets.connect(uri) as websocket:
                    # Send decision request
                    request = {
                        "npc": "Bob",
                        "state_int": state,
                        "location": "bar",
                        "time": "evening",
                        "nearby_npcs": ["Alice", "Sam"],
                        "player_nearby": state > 100
                    }
                    
                    await websocket.send(json.dumps(request))
                    
                    # Wait for response
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    latency = (time.time() - start) * 1000  # Convert to ms
                    times.append(latency)
                    
                    print(f"  Test {i+1}: {latency:.1f}ms - Action: {data.get('action', 'unknown')}")
                    
            except Exception as e:
                errors += 1
                print(f"  Test {i+1}: ERROR - {str(e)}")
        
        if times:
            avg = statistics.mean(times)
            median = statistics.median(times)
            min_time = min(times)
            max_time = max(times)
            
            all_results[state] = {
                "avg": avg,
                "median": median,
                "min": min_time,
                "max": max_time,
                "errors": errors
            }
            
            print(f"\n  Results for State {state}:")
            print(f"    Average: {avg:.1f}ms")
            print(f"    Median:  {median:.1f}ms")
            print(f"    Min:     {min_time:.1f}ms")
            print(f"    Max:     {max_time:.1f}ms")
            print(f"    Success: {10-errors}/10")
    
    return all_results

def test_decision_http():
    """Test decision server via HTTP endpoint"""
    base_url = "http://localhost:8021/state_int"
    
    print("\nTesting Decision Server HTTP Latency")
    print("="*50)
    
    # Test different state complexities
    test_states = [0, 1, 15, 63, 127, 255]
    
    all_results = {}
    
    for state in test_states:
        times = []
        errors = 0
        
        print(f"\nTesting State {state}:")
        
        for i in range(10):
            try:
                start = time.time()
                
                # Send HTTP request
                response = requests.post(
                    base_url,
                    json={
                        "npc_name": "Bob",
                        "state_int": state,
                        "location": "bar",
                        "time": "evening",
                        "nearby_npcs": ["Alice", "Sam"],
                        "player_nearby": state > 100,
                        "current_action": "idle",
                        "energy": 80
                    },
                    timeout=5
                )
                
                if response.status_code == 200:
                    latency = (time.time() - start) * 1000
                    times.append(latency)
                    data = response.json()
                    print(f"  Test {i+1}: {latency:.1f}ms - Action: {data.get('action', 'unknown')}")
                else:
                    errors += 1
                    print(f"  Test {i+1}: HTTP {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                print(f"  Test {i+1}: Connection failed - server not running on port 8021")
                errors += 1
                break
            except Exception as e:
                errors += 1
                print(f"  Test {i+1}: ERROR - {str(e)}")
        
        if times:
            avg = statistics.mean(times)
            median = statistics.median(times)
            
            all_results[state] = {
                "avg": avg,
                "median": median,
                "min": min(times),
                "max": max(times),
                "errors": errors
            }
            
            print(f"\n  Results for State {state}:")
            print(f"    Average: {avg:.1f}ms")
            print(f"    Median:  {median:.1f}ms")
            print(f"    Range:   {min(times):.1f}-{max(times):.1f}ms")
    
    return all_results

async def main():
    """Run comprehensive decision server latency tests"""
    
    # First check if decision server is running
    print("Decision Server Latency Test Suite")
    print("="*50)
    print("Checking server availability...\n")
    
    # Try WebSocket first
    try:
        async with websockets.connect("ws://127.0.0.1:9998") as ws:
            await ws.close()
        print("WebSocket server found on port 9998")
        ws_results = await test_decision_websocket()
    except:
        print("WebSocket server not found on port 9998")
        ws_results = None
    
    # Try HTTP
    try:
        response = requests.get("http://localhost:8021/health", timeout=1)
        print("HTTP server found on port 8021")
        http_results = test_decision_http()
    except:
        print("HTTP server not found on port 8021")
        http_results = None
    
    # Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    if ws_results:
        print("\nWebSocket Performance:")
        overall_times = []
        for state, stats in ws_results.items():
            overall_times.append(stats["avg"])
            print(f"  State {state:3d}: {stats['avg']:6.1f}ms avg, {stats['median']:6.1f}ms median")
        
        print(f"\n  Overall Average: {statistics.mean(overall_times):.1f}ms")
        print(f"  Overall Range: {min(overall_times):.1f}-{max(overall_times):.1f}ms")
        
        # Check <100ms claim
        if statistics.mean(overall_times) < 100:
            print("  ✓ VERIFIED: <100ms average latency")
        else:
            print(f"  ✗ NOT VERIFIED: Average {statistics.mean(overall_times):.1f}ms > 100ms")
    
    if http_results:
        print("\nHTTP Performance:")
        overall_times = []
        for state, stats in http_results.items():
            overall_times.append(stats["avg"])
            print(f"  State {state:3d}: {stats['avg']:6.1f}ms avg")
        
        print(f"\n  Overall Average: {statistics.mean(overall_times):.1f}ms")
    
    # Save results
    import json
    with open("decision_latency_results.json", "w") as f:
        json.dump({
            "websocket": ws_results,
            "http": http_results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"\nResults saved to decision_latency_results.json")

if __name__ == "__main__":
    asyncio.run(main())
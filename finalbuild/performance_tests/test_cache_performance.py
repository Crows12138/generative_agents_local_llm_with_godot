"""
Quick test to measure cache performance impact
"""
import time
import requests
import statistics

def test_response_time(use_cache=True):
    """Test response time with/without cache"""
    
    base_url = "http://localhost:8020/v1/chat/completions"
    
    # Test queries
    queries = [
        "Hello, how are you?",
        "What's the weather like?", 
        "Tell me a joke",
        "What's your name?",
        "How's your day going?"
    ]
    
    times = []
    
    for query in queries:
        start = time.time()
        
        try:
            response = requests.post(
                base_url,
                json={
                    "model": "Llama-3.2-3B-Instruct.Q4_0.gguf",
                    "messages": [
                        {"role": "system", "content": "You are Bob, a friendly bartender."},
                        {"role": "user", "content": query}
                    ],
                    "max_tokens": 50,
                    "stream": False
                },
                timeout=30
            )
            
            if response.status_code == 200:
                elapsed = (time.time() - start) * 1000  # Convert to ms
                times.append(elapsed)
                print(f"  Query {len(times)}: {elapsed:.0f}ms")
            else:
                print(f"  Error: {response.status_code}")
                
        except Exception as e:
            print(f"  Failed: {str(e)}")
    
    if times:
        return {
            "min": min(times),
            "max": max(times),
            "avg": statistics.mean(times),
            "median": statistics.median(times)
        }
    return None

def main():
    print("Performance Optimization Impact Test")
    print("="*40)
    
    # Check server
    try:
        response = requests.get("http://localhost:8020/health", timeout=2)
        print("Server is running\n")
    except:
        # Try WebSocket server
        print("Testing with WebSocket server on port 9999...")
        import asyncio
        import websockets
        import json
        
        async def test_ws():
            times = []
            queries = ["Hello", "How are you?", "What's up?", "Tell me something", "Good day"]
            
            for query in queries:
                try:
                    start = time.time()
                    async with websockets.connect("ws://127.0.0.1:9999") as ws:
                        await ws.send(json.dumps({
                            "npc": "Bob",
                            "message": f"Bob|{query}"
                        }))
                        
                        # Wait for first token
                        first_token_time = None
                        while True:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            if data.get("type") == "token" and first_token_time is None:
                                first_token_time = (time.time() - start) * 1000
                            elif data.get("type") == "complete":
                                total_time = (time.time() - start) * 1000
                                times.append(first_token_time)
                                print(f"  Query {len(times)}: First token {first_token_time:.0f}ms, Total {total_time:.0f}ms")
                                break
                except Exception as e:
                    print(f"  Error: {e}")
            
            if times:
                print(f"\nResults:")
                print(f"  First Token Average: {statistics.mean(times):.0f}ms")
                print(f"  First Token Median: {statistics.median(times):.0f}ms")
                print(f"  Min: {min(times):.0f}ms")
                print(f"  Max: {max(times):.0f}ms")
        
        asyncio.run(test_ws())
        return
    
    # Test with cache (warm)
    print("Testing with cache (warm responses):")
    print("-"*40)
    results = test_response_time(use_cache=True)
    
    if results:
        print(f"\nCache Active Results:")
        print(f"  Average: {results['avg']:.0f}ms")
        print(f"  Median: {results['median']:.0f}ms")
        print(f"  Min: {results['min']:.0f}ms")
        print(f"  Max: {results['max']:.0f}ms")

if __name__ == "__main__":
    main()
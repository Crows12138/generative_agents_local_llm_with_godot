# websocket_performance_test.py
import asyncio
import websockets
import json
import time
import statistics
from typing import Dict, List
from datetime import datetime

class WebSocketPerformanceTest:
    """Performance test for GPT4All WebSocket server"""
    
    def __init__(self):
        self.server_url = "ws://localhost:9999"
        self.test_cases = {
            "simple": ["Hello", "Hi", "Good day"],
            "context": ["What drinks?", "Tell me about the bar"],
            "complex": ["What's your life story?", "Philosophy of bartending?"]
        }
        
    async def send_message_and_measure(self, npc_name: str, message: str) -> float:
        """Send message via WebSocket and measure response time"""
        try:
            async with websockets.connect(self.server_url) as websocket:
                start_time = time.time()
                
                # Send message
                await websocket.send(json.dumps({
                    "npc": npc_name,
                    "message": message
                }))
                
                # Collect all tokens until complete
                full_response = ""
                while True:
                    response = await websocket.recv()
                    data = json.loads(response)
                    
                    if data["type"] == "complete":
                        full_response = data["content"]
                        break
                    elif data["type"] == "token":
                        full_response += data["content"]
                    elif data["type"] == "error":
                        raise Exception(f"Server error: {data['content']}")
                
                response_time = (time.time() - start_time) * 1000  # Convert to ms
                return response_time
                
        except Exception as e:
            print(f"    ERROR: {str(e)}")
            return -1
    
    async def run_category_tests(self, npc_name: str, category: str, queries: List[str]) -> Dict:
        """Run tests for a category of queries"""
        times = []
        print(f"Testing {category} queries...")
        
        for query in queries:
            response_time = await self.send_message_and_measure(npc_name, query)
            if response_time > 0:
                times.append(response_time)
                print(f"  - '{query[:30]}...': {response_time:.2f}ms")
            else:
                print(f"  - '{query[:30]}...': FAILED")
        
        if times:
            return {
                "mean": statistics.mean(times),
                "median": statistics.median(times),
                "stdev": statistics.stdev(times) if len(times) > 1 else 0,
                "min": min(times),
                "max": max(times),
                "count": len(times),
                "total_queries": len(queries)
            }
        return None
    
    async def run_all_tests(self):
        """Run all performance tests"""
        print(f"\n{'='*60}")
        print(f"WebSocket Performance Test - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Server: {self.server_url}")
        print(f"{'='*60}\n")
        
        results = {}
        npc_name = "Bob"
        
        for category, queries in self.test_cases.items():
            result = await self.run_category_tests(npc_name, category, queries)
            if result:
                results[category] = result
                
                print(f"\n{category.capitalize()} Results:")
                print(f"  Mean: {result['mean']:.2f}ms")
                print(f"  Median: {result['median']:.2f}ms")
                print(f"  StdDev: {result['stdev']:.2f}ms")
                print(f"  Min: {result['min']:.2f}ms")
                print(f"  Max: {result['max']:.2f}ms")
                print(f"  Success: {result['count']}/{result['total_queries']}\n")
        
        # Save results
        with open('websocket_performance_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"{'='*60}")
        print("Results saved to websocket_performance_results.json")
        print(f"{'='*60}\n")
        
        return results
    
    async def test_concurrent_connections(self):
        """Test concurrent WebSocket connections"""
        print("\n" + "="*60)
        print("Concurrent Connection Test")
        print("="*60 + "\n")
        
        async def send_concurrent_message(message: str) -> float:
            return await self.send_message_and_measure("Bob", message)
        
        concurrent_counts = [1, 3, 5]
        message = "Hello"
        
        for count in concurrent_counts:
            print(f"Testing with {count} concurrent connections...")
            
            tasks = [send_concurrent_message(message) for _ in range(count)]
            start = time.time()
            times = await asyncio.gather(*tasks)
            total_time = (time.time() - start) * 1000
            
            successful = [t for t in times if t > 0]
            if successful:
                print(f"  Success rate: {len(successful)}/{count}")
                print(f"  Avg response time: {statistics.mean(successful):.2f}ms")
                print(f"  Total time: {total_time:.2f}ms\n")

async def main():
    """Main test runner"""
    # Check server is running
    try:
        async with websockets.connect("ws://localhost:9999") as ws:
            print("Server is running. Starting tests...\n")
    except:
        print("ERROR: WebSocket server is not responding at ws://localhost:9999")
        print("Please start the server first")
        return
    
    tester = WebSocketPerformanceTest()
    await tester.run_all_tests()
    await tester.test_concurrent_connections()

if __name__ == "__main__":
    asyncio.run(main())
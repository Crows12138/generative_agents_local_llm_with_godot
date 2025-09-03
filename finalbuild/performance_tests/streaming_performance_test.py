"""
Streaming Performance Test
Tests Time to First Token (TTFT) and streaming metrics for GPT4All WebSocket server
"""

import asyncio
import websockets
import json
import time
from typing import Dict, List
from datetime import datetime

class StreamingPerformanceTester:
    def __init__(self):
        self.results = {
            "simple": [],
            "context": [],
            "complex": []
        }
        self.uri = "ws://127.0.0.1:9999"
        
    async def test_streaming_response(self, npc: str, query: str, query_type: str):
        """Test streaming response with detailed timing metrics"""
        
        timings = {
            "start_time": None,
            "first_token_time": None,
            "last_token_time": None,
            "tokens_received": [],
            "full_response": ""
        }
        
        try:
            async with websockets.connect(self.uri) as websocket:
                # Send request
                request = {
                    "npc": npc,
                    "message": f"{npc}|{query}"  # Use clean protocol format
                }
                
                timings["start_time"] = time.time()
                await websocket.send(json.dumps(request))
                
                # Receive streaming response
                first_token_received = False
                
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    current_time = time.time()
                    
                    if data.get("type") == "token":
                        token = data.get("content", "")
                        if not first_token_received:
                            timings["first_token_time"] = current_time
                            first_token_received = True
                            ttft = (current_time - timings["start_time"]) * 1000
                            print(f"  ⚡ First token in {ttft:.2f}ms: '{token[:20]}...'")
                        
                        timings["tokens_received"].append({
                            "token": token,
                            "time": current_time - timings["start_time"]
                        })
                        timings["full_response"] += token
                        
                    elif data.get("type") == "complete":
                        timings["last_token_time"] = current_time
                        total_time = (current_time - timings["start_time"]) * 1000
                        print(f"  ✓ Complete in {total_time:.2f}ms ({len(timings['tokens_received'])} tokens)")
                        break
                    
                    elif data.get("type") == "error":
                        print(f"  ❌ Error: {data.get('content')}")
                        break
                
                # Calculate metrics
                if timings["first_token_time"] and timings["last_token_time"]:
                    streaming_duration = timings["last_token_time"] - timings["first_token_time"]
                    
                    result = {
                        "query": query[:50],
                        "ttft_ms": (timings["first_token_time"] - timings["start_time"]) * 1000,
                        "total_time_ms": (timings["last_token_time"] - timings["start_time"]) * 1000,
                        "tokens_count": len(timings["tokens_received"]),
                        "tokens_per_second": len(timings["tokens_received"]) / streaming_duration if streaming_duration > 0 else 0,
                        "response_length": len(timings["full_response"])
                    }
                    
                    self.results[query_type].append(result)
                    return result
                    
        except Exception as e:
            print(f"  ❌ Connection error: {e}")
            return None
    
    async def run_comprehensive_test(self):
        """Run complete streaming performance test suite"""
        print("\n" + "="*60)
        print("STREAMING PERFORMANCE TEST - GPT4All WebSocket")
        print("="*60)
        print(f"Server: {self.uri}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        test_queries = {
            "simple": [
                "Hello!",
                "How are you?",
                "What's your name?"
            ],
            "context": [
                "What drinks do you serve?",
                "Tell me about this place.",
                "What's the special today?"
            ],
            "complex": [
                "What's your philosophy on bartending and customer service?",
                "Tell me about your most memorable experience here.",
                "How would you describe the perfect evening at this bar?"
            ]
        }
        
        # Test each category
        for query_type, queries in test_queries.items():
            print(f"\n📊 Testing {query_type.upper()} queries...")
            print("-" * 40)
            
            for query in queries:
                print(f"\nQuery: '{query[:50]}...'")
                result = await self.test_streaming_response("Bob", query, query_type)
                if result:
                    print(f"  Tokens/sec: {result['tokens_per_second']:.1f}")
                await asyncio.sleep(1)  # Avoid overloading
        
        # Generate report
        self.generate_report()
        self.save_results()
    
    def generate_report(self):
        """Generate detailed performance report"""
        print("\n" + "="*60)
        print("STREAMING METRICS SUMMARY")
        print("="*60)
        
        all_ttfts = []
        all_total_times = []
        
        for query_type, results in self.results.items():
            if results:
                ttfts = [r["ttft_ms"] for r in results if r.get("ttft_ms")]
                total_times = [r["total_time_ms"] for r in results if r.get("total_time_ms")]
                tokens_counts = [r["tokens_count"] for r in results if r.get("tokens_count")]
                
                if ttfts:
                    all_ttfts.extend(ttfts)
                    all_total_times.extend(total_times)
                    
                    print(f"\n{query_type.upper()} Queries ({len(results)} samples):")
                    print(f"  Time to First Token (TTFT):")
                    print(f"    • Average: {sum(ttfts)/len(ttfts):.2f}ms")
                    print(f"    • Min: {min(ttfts):.2f}ms")
                    print(f"    • Max: {max(ttfts):.2f}ms")
                    
                    print(f"  Total Response Time:")
                    print(f"    • Average: {sum(total_times)/len(total_times):.2f}ms")
                    print(f"    • Min: {min(total_times):.2f}ms")
                    print(f"    • Max: {max(total_times):.2f}ms")
                    
                    print(f"  Tokens Generated:")
                    print(f"    • Average: {sum(tokens_counts)/len(tokens_counts):.1f} tokens")
                    
                    # Calculate perceived latency reduction
                    avg_ttft = sum(ttfts)/len(ttfts)
                    avg_total = sum(total_times)/len(total_times)
                    improvement = ((avg_total - avg_ttft) / avg_total) * 100
                    print(f"  Perceived Latency Reduction: {improvement:.1f}%")
        
        # Overall summary
        if all_ttfts:
            print("\n" + "="*60)
            print("OVERALL PERFORMANCE")
            print("="*60)
            print(f"\n🎯 Key Metrics:")
            print(f"  • Average TTFT: {sum(all_ttfts)/len(all_ttfts):.2f}ms")
            print(f"  • Average Total Time: {sum(all_total_times)/len(all_total_times):.2f}ms")
            print(f"  • Samples Tested: {len(all_ttfts)}")
            
            # User experience impact
            avg_ttft = sum(all_ttfts)/len(all_ttfts)
            avg_total = sum(all_total_times)/len(all_total_times)
            
            print(f"\n📈 User Experience Impact:")
            print(f"  • Users start reading in ~{avg_ttft:.0f}ms (vs waiting {avg_total:.0f}ms)")
            print(f"  • Perceived responsiveness improvement: {(avg_total - avg_ttft):.0f}ms faster")
            print(f"  • Reduction in perceived latency: {((avg_total - avg_ttft) / avg_total * 100):.1f}%")
    
    def save_results(self):
        """Save results to JSON file"""
        results_data = {
            "test_time": datetime.now().isoformat(),
            "server": self.uri,
            "results": self.results,
            "summary": {
                "total_queries": sum(len(r) for r in self.results.values()),
                "categories_tested": list(self.results.keys())
            }
        }
        
        filename = f"streaming_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

async def compare_with_non_streaming():
    """Compare streaming vs non-streaming performance"""
    print("\n" + "="*60)
    print("STREAMING vs NON-STREAMING COMPARISON")
    print("="*60)
    
    # Simulated comparison based on typical results
    print("\n📊 Typical Performance Comparison:")
    print("-" * 40)
    
    streaming_ttft = 350  # ms
    streaming_total = 2500  # ms
    non_streaming_total = 2500  # ms (same generation time)
    
    print(f"\nWITH STREAMING:")
    print(f"  • First token visible: {streaming_ttft}ms")
    print(f"  • User starts reading immediately")
    print(f"  • Total time: {streaming_total}ms")
    print(f"  • User experience: Responsive, engaging")
    
    print(f"\nWITHOUT STREAMING:")
    print(f"  • First token visible: {non_streaming_total}ms")
    print(f"  • User waits for entire response")
    print(f"  • Total time: {non_streaming_total}ms")
    print(f"  • User experience: Delayed, static")
    
    improvement = ((non_streaming_total - streaming_ttft) / non_streaming_total) * 100
    
    print(f"\n🚀 IMPROVEMENT:")
    print(f"  • {improvement:.1f}% reduction in perceived latency")
    print(f"  • {non_streaming_total - streaming_ttft}ms faster initial response")
    print(f"  • Eliminates 'dead time' waiting for response")

async def main():
    """Main test execution"""
    print("\n🚀 GPT4All Streaming Performance Test Suite")
    print("=" * 60)
    
    # Check server connection first
    try:
        async with websockets.connect("ws://127.0.0.1:9999") as ws:
            print("✅ Server connected successfully")
    except:
        print("❌ Cannot connect to server at ws://127.0.0.1:9999")
        print("Please start the server with: python finalbuild/server/gpt4all_server.py")
        return
    
    # Run streaming tests
    tester = StreamingPerformanceTester()
    await tester.run_comprehensive_test()
    
    # Show comparison
    await compare_with_non_streaming()
    
    print("\n✅ All tests completed!")

if __name__ == "__main__":
    asyncio.run(main())
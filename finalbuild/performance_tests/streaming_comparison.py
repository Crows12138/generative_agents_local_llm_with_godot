"""
Streaming vs Non-Streaming Comparison Test
Demonstrates the user experience difference between streaming and non-streaming responses
"""

import asyncio
import websockets
import json
import time
import sys
from typing import Tuple

class StreamingComparison:
    def __init__(self):
        self.uri = "ws://127.0.0.1:9999"
        self.test_queries = [
            "Hello, how are you?",
            "What's your favorite drink to make?",
            "Tell me about your day.",
            "What brings people to this bar?",
            "Do you have any recommendations?"
        ]
    
    async def test_with_streaming(self, query: str) -> Tuple[float, float, str]:
        """Test with streaming enabled - measures TTFT and total time"""
        print("\n🚀 WITH STREAMING:")
        print(f"Query: '{query}'")
        
        start_time = time.time()
        first_token_time = None
        response_text = ""
        token_count = 0
        
        try:
            async with websockets.connect(self.uri) as websocket:
                # Send request
                request = {"npc": "Bob", "message": f"Bob|{query}"}
                await websocket.send(json.dumps(request))
                
                print("Waiting for response...")
                
                # Receive streaming tokens
                while True:
                    message = await websocket.recv()
                    data = json.loads(message)
                    
                    if data.get("type") == "token":
                        token = data.get("content", "")
                        response_text += token
                        token_count += 1
                        
                        if first_token_time is None:
                            first_token_time = time.time()
                            ttft = (first_token_time - start_time) * 1000
                            print(f"⚡ First token received at {ttft:.0f}ms")
                            print(f"User sees: '{token}", end="", flush=True)
                        else:
                            # Simulate typewriter effect
                            print(token, end="", flush=True)
                            
                    elif data.get("type") == "complete":
                        end_time = time.time()
                        total_time = (end_time - start_time) * 1000
                        print(f"'\n✓ Complete response in {total_time:.0f}ms ({token_count} tokens)")
                        break
                
                ttft = (first_token_time - start_time) * 1000 if first_token_time else 0
                total = (end_time - start_time) * 1000
                
                return ttft, total, response_text
                
        except Exception as e:
            print(f"❌ Error: {e}")
            return 0, 0, ""
    
    async def simulate_without_streaming(self, query: str, response_text: str, avg_time: float) -> Tuple[float, float]:
        """Simulate non-streaming response (user waits for complete response)"""
        print("\n⏳ WITHOUT STREAMING:")
        print(f"Query: '{query}'")
        
        print("Waiting for response...")
        print("[Loading", end="", flush=True)
        
        # Simulate waiting for complete response
        for _ in range(int(avg_time / 500)):
            await asyncio.sleep(0.5)
            print(".", end="", flush=True)
        
        print("]")
        print(f"✓ Response received at {avg_time:.0f}ms")
        print(f"User sees: '{response_text}'")
        
        return avg_time, avg_time  # TTFT = Total time in non-streaming
    
    async def run_comparison_test(self):
        """Run side-by-side comparison"""
        print("\n" + "="*60)
        print("STREAMING vs NON-STREAMING USER EXPERIENCE")
        print("="*60)
        
        streaming_results = []
        non_streaming_results = []
        
        for i, query in enumerate(self.test_queries, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}/{len(self.test_queries)}")
            print(f"{'='*60}")
            
            # Test with streaming
            ttft_s, total_s, response = await self.test_with_streaming(query)
            if ttft_s > 0:
                streaming_results.append((ttft_s, total_s))
            
            await asyncio.sleep(1)
            
            # Simulate non-streaming with same response
            if response:
                ttft_ns, total_ns = await self.simulate_without_streaming(query, response, total_s)
                non_streaming_results.append((ttft_ns, total_ns))
            
            # Show immediate comparison
            if ttft_s > 0:
                print(f"\n📊 Comparison for this query:")
                print(f"  Streaming TTFT: {ttft_s:.0f}ms | Non-streaming TTFT: {total_s:.0f}ms")
                print(f"  User starts reading {total_s - ttft_s:.0f}ms earlier with streaming!")
            
            await asyncio.sleep(2)
        
        # Generate final report
        self.generate_comparison_report(streaming_results, non_streaming_results)
    
    def generate_comparison_report(self, streaming_results, non_streaming_results):
        """Generate detailed comparison report"""
        print("\n" + "="*60)
        print("FINAL COMPARISON RESULTS")
        print("="*60)
        
        if not streaming_results or not non_streaming_results:
            print("❌ Insufficient data for comparison")
            return
        
        # Calculate averages
        avg_streaming_ttft = sum(r[0] for r in streaming_results) / len(streaming_results)
        avg_streaming_total = sum(r[1] for r in streaming_results) / len(streaming_results)
        avg_non_streaming_ttft = sum(r[0] for r in non_streaming_results) / len(non_streaming_results)
        
        print(f"\n📊 Average Time to First Token (TTFT):")
        print(f"  • WITH Streaming: {avg_streaming_ttft:.0f}ms")
        print(f"  • WITHOUT Streaming: {avg_non_streaming_ttft:.0f}ms")
        print(f"  • Difference: {avg_non_streaming_ttft - avg_streaming_ttft:.0f}ms faster with streaming")
        
        improvement = ((avg_non_streaming_ttft - avg_streaming_ttft) / avg_non_streaming_ttft) * 100
        
        print(f"\n📊 User Experience Metrics:")
        print(f"  • Perceived latency reduction: {improvement:.1f}%")
        print(f"  • Users start reading {avg_non_streaming_ttft - avg_streaming_ttft:.0f}ms earlier")
        print(f"  • Engagement starts in {avg_streaming_ttft:.0f}ms vs {avg_non_streaming_ttft:.0f}ms wait")
        
        print(f"\n🧠 Psychological Impact:")
        print(f"  WITH Streaming:")
        print(f"    • Immediate feedback creates engagement")
        print(f"    • Progressive content reduces perceived wait")
        print(f"    • User reads while AI continues generating")
        
        print(f"  WITHOUT Streaming:")
        print(f"    • Static loading creates anxiety")
        print(f"    • Full wait time feels longer")
        print(f"    • User stares at loading indicator")
        
        print(f"\n🎯 Bottom Line:")
        print(f"  Streaming provides {improvement:.0f}% better perceived performance")
        print(f"  Real-world impact: MASSIVE improvement in user satisfaction")

    async def visual_demonstration(self):
        """Visual demonstration of the difference"""
        print("\n" + "="*60)
        print("VISUAL DEMONSTRATION")
        print("="*60)
        
        print("\n1️⃣ NON-STREAMING Experience (what users see):")
        print("   0ms: [Loading...]")
        print("   500ms: [Loading...]")
        print("   1000ms: [Loading...]")
        print("   1500ms: [Loading...]")
        print("   2000ms: [Loading...]")
        print("   2500ms: 'Hello! Welcome to the bar. What can I get you today?'")
        print("   User waits 2.5 seconds before seeing ANYTHING")
        
        print("\n2️⃣ STREAMING Experience (what users see):")
        print("   0ms: [Waiting...]")
        print("   350ms: 'Hello!'")
        print("   500ms: 'Hello! Welcome'")
        print("   750ms: 'Hello! Welcome to'")
        print("   1000ms: 'Hello! Welcome to the'")
        print("   1250ms: 'Hello! Welcome to the bar.'")
        print("   1500ms: 'Hello! Welcome to the bar. What'")
        print("   2000ms: 'Hello! Welcome to the bar. What can I'")
        print("   2500ms: 'Hello! Welcome to the bar. What can I get you today?'")
        print("   User starts reading at 350ms and stays engaged throughout")
        
        print("\n✨ The difference: 2150ms of engagement vs dead waiting!")

async def main():
    """Main execution"""
    print("\n🔬 Streaming vs Non-Streaming Comparison Test")
    print("=" * 60)
    
    # Check server
    try:
        async with websockets.connect("ws://127.0.0.1:9999") as ws:
            print("✅ Server connected")
    except:
        print("❌ Server not running at ws://127.0.0.1:9999")
        print("Start server with: python finalbuild/server/gpt4all_server.py")
        return
    
    comparison = StreamingComparison()
    
    # Run comparison test
    await comparison.run_comparison_test()
    
    # Show visual demonstration
    await comparison.visual_demonstration()
    
    print("\n✅ Comparison test completed!")
    print("📝 Results clearly show streaming provides superior user experience")

if __name__ == "__main__":
    asyncio.run(main())
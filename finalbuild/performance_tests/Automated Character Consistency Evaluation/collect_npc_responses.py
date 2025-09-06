"""
NPC Response Collector for Consistency Evaluation
Collects standardized responses from NPCs for ChatGPT evaluation
"""
import json
import asyncio
import websockets
from datetime import datetime
import sys
import os

class NPCResponseCollector:
    def __init__(self):
        self.uri = "ws://127.0.0.1:9999"
        
        # Standardized test prompts for consistency evaluation
        self.test_prompts = {
            "personality": [
                "How do you feel about your job?",
                "What makes you happy?",
                "Describe your personality in your own words",
                "What's your biggest fear?",
                "Tell me about your daily routine"
            ],
            "memory_context": [
                "What did we just talk about?",
                "Do you remember our last conversation?",
                "Have we met before?",
                "What do you know about me so far?",
                "What happened earlier today in the bar?"
            ],
            "role_expertise": [
                "What's your area of expertise?",
                "How long have you been working here?",
                "What professional skills do you have?",
                "Describe your main responsibilities",
                "What makes you good at your job?"
            ],
            "emotional_response": [
                "How do you feel right now?",
                "What upset you recently?",
                "Tell me about a time you were really happy",
                "What makes you angry?",
                "How do you handle stress?"
            ],
            "social_interaction": [
                "How do you get along with your coworkers?",
                "Describe your relationship with the regulars",
                "Who's your best friend here?",
                "What do you think about the other staff?",
                "How do you handle difficult customers?"
            ]
        }
        
        self.npcs = ["Bob", "Alice", "Sam"]
        
    async def get_npc_response(self, npc_name, prompt):
        """Get response from a single NPC"""
        try:
            async with websockets.connect(self.uri) as websocket:
                request = {
                    "npc": npc_name,
                    "message": f"{npc_name}|{prompt}"
                }
                
                await websocket.send(json.dumps(request))
                
                full_response = ""
                start_time = datetime.now()
                
                while True:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30.0)
                        data = json.loads(message)
                        
                        if data.get("type") == "token":
                            full_response += data.get("content", "")
                        elif data.get("type") == "complete":
                            response_time = (datetime.now() - start_time).total_seconds()
                            return {
                                "response": data.get("content", full_response),
                                "response_time": response_time
                            }
                        elif data.get("type") == "error":
                            return {
                                "response": f"[Error: {data.get('content')}]",
                                "response_time": 0,
                                "error": True
                            }
                    except asyncio.TimeoutError:
                        return {
                            "response": "[Timeout: No response within 30 seconds]",
                            "response_time": 30,
                            "error": True
                        }
                        
        except Exception as e:
            return {
                "response": f"[Connection error: {str(e)}]",
                "response_time": 0,
                "error": True
            }
    
    async def collect_all_responses(self):
        """Collect responses from all NPCs"""
        results = {
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": "Llama 3.2 3B Q4_0",
                "test_categories": list(self.test_prompts.keys()),
                "npcs_tested": self.npcs,
                "total_prompts": sum(len(prompts) for prompts in self.test_prompts.values())
            },
            "responses": {},
            "statistics": {}
        }
        
        total_prompts = len(self.npcs) * results["test_metadata"]["total_prompts"]
        current_prompt = 0
        
        for npc in self.npcs:
            print(f"\n{'='*50}")
            print(f"Testing NPC: {npc}")
            print(f"{'='*50}")
            
            results["responses"][npc] = {}
            npc_stats = {
                "total_responses": 0,
                "errors": 0,
                "avg_response_time": 0,
                "avg_word_count": 0
            }
            
            for category, prompts in self.test_prompts.items():
                print(f"\nCategory: {category}")
                results["responses"][npc][category] = []
                
                for i, prompt in enumerate(prompts, 1):
                    current_prompt += 1
                    progress = (current_prompt / total_prompts) * 100
                    
                    print(f"  [{i}/{len(prompts)}] {prompt[:40]}...", end=" ")
                    
                    response_data = await self.get_npc_response(npc, prompt)
                    response_text = response_data["response"]
                    
                    result = {
                        "prompt": prompt,
                        "response": response_text,
                        "word_count": len(response_text.split()) if not response_data.get("error") else 0,
                        "response_time": response_data["response_time"],
                        "character_count": len(response_text)
                    }
                    
                    results["responses"][npc][category].append(result)
                    
                    # Update statistics
                    npc_stats["total_responses"] += 1
                    if response_data.get("error"):
                        npc_stats["errors"] += 1
                        print("X Error")
                    else:
                        npc_stats["avg_response_time"] += response_data["response_time"]
                        npc_stats["avg_word_count"] += result["word_count"]
                        print(f"OK ({result['word_count']} words, {response_data['response_time']:.1f}s)")
                    
                    # Small delay between requests
                    await asyncio.sleep(1.5)
            
            # Calculate averages
            if npc_stats["total_responses"] > npc_stats["errors"]:
                valid_responses = npc_stats["total_responses"] - npc_stats["errors"]
                npc_stats["avg_response_time"] /= valid_responses
                npc_stats["avg_word_count"] /= valid_responses
            
            results["statistics"][npc] = npc_stats
            
            print(f"\n{npc} Statistics:")
            print(f"  - Total responses: {npc_stats['total_responses']}")
            print(f"  - Errors: {npc_stats['errors']}")
            print(f"  - Avg response time: {npc_stats['avg_response_time']:.2f}s")
            print(f"  - Avg word count: {npc_stats['avg_word_count']:.1f}")
        
        return results
    
    def save_results(self, results):
        """Save results to JSON file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"npc_consistency_data_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUCCESS] Results saved to: {filename}")
        return filename
    
    def generate_evaluation_report(self, results):
        """Generate evaluation prompt for ChatGPT"""
        prompt = """# NPC Character Consistency Evaluation Request

## Background
These responses are from NPCs in a bar simulation game using Llama 3.2 3B model.

## NPCs to Evaluate:
- **Bob**: Professional bartender, experienced, friendly
- **Alice**: Contemplative regular patron, philosophical  
- **Sam**: Energetic musician, creative, passionate about music

## Evaluation Criteria (Score 1-10):

### 1. Personality Consistency
- Are personality traits stable across responses?
- Does each character maintain their unique voice?
- Are emotional responses appropriate to character?

### 2. Memory & Context Awareness
- Do they maintain context within conversations?
- Are references to past events consistent?
- Do they show appropriate knowledge of their environment?

### 3. Role Expertise
- Do they demonstrate appropriate professional knowledge?
- Are their skills and expertise consistent with their role?
- Do they respond appropriately to role-specific questions?

### 4. Language & Communication Style
- Is their vocabulary consistent with their character?
- Do they maintain a consistent communication style?
- Are responses appropriately detailed?

## Required Output Format:
```
Character Consistency Scores:
================================
Bob:
  - Personality: X.X/10
  - Memory & Context: X.X/10
  - Role Expertise: X.X/10
  - Language Style: X.X/10
  - Overall: X.X/10

Alice:
  - Personality: X.X/10
  - Memory & Context: X.X/10
  - Role Expertise: X.X/10
  - Language Style: X.X/10
  - Overall: X.X/10

Sam:
  - Personality: X.X/10
  - Memory & Context: X.X/10
  - Role Expertise: X.X/10
  - Language Style: X.X/10
  - Overall: X.X/10

Overall System Score: X.X/10
================================

Detailed Analysis:
[Provide 2-3 sentences for each character highlighting strengths and areas for improvement]

Key Findings:
[List 3-4 main observations about the system's character consistency]
```

## JSON Data:
[Paste the contents of the JSON file below this line]

"""
        
        # Save evaluation prompt
        with open("chatgpt_evaluation_prompt.txt", 'w', encoding='utf-8') as f:
            f.write(prompt)
        
        # Also create a summary for quick review
        summary = {
            "test_date": results["test_metadata"]["timestamp"],
            "model": results["test_metadata"]["model"],
            "npcs_tested": len(results["test_metadata"]["npcs_tested"]),
            "total_prompts_per_npc": results["test_metadata"]["total_prompts"],
            "categories_tested": results["test_metadata"]["test_categories"]
        }
        
        with open("test_summary.json", 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        
        print("\n[INFO] Evaluation files created:")
        print("  - chatgpt_evaluation_prompt.txt (Instructions for ChatGPT)")
        print("  - test_summary.json (Quick test overview)")
        
        return summary

async def main():
    collector = NPCResponseCollector()
    
    print("NPC Character Consistency Data Collection")
    print("="*50)
    print("This will collect responses from all NPCs for consistency evaluation")
    print(f"NPCs to test: {', '.join(collector.npcs)}")
    print(f"Categories: {len(collector.test_prompts)}")
    print(f"Total prompts per NPC: {sum(len(p) for p in collector.test_prompts.values())}")
    print("="*50)
    
    # Check if server is running
    print("\nChecking server connection...", end=" ")
    test_response = await collector.get_npc_response("Bob", "Hello")
    if test_response.get("error"):
        print("[FAILED]")
        print("\nError: Cannot connect to server.")
        print("Please make sure the server is running:")
        print("  cd finalbuild/server")
        print("  python gpt4all_server.py")
        return
    else:
        print("[CONNECTED]")
    
    print("\nStarting data collection...")
    print("Estimated time: 10-15 minutes")
    print("-"*50)
    
    # Collect responses
    start_time = datetime.now()
    results = await collector.collect_all_responses()
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    # Save results
    filename = collector.save_results(results)
    
    # Generate evaluation materials
    summary = collector.generate_evaluation_report(results)
    
    print(f"\n{'='*50}")
    print(f"[COMPLETE] Data Collection Finished!")
    print(f"{'='*50}")
    print(f"Time taken: {elapsed_time/60:.1f} minutes")
    print(f"Data file: {filename}")
    print(f"\nNext steps:")
    print(f"1. Open {filename}")
    print(f"2. Copy all content (Ctrl+A, Ctrl+C)")
    print(f"3. Open ChatGPT")
    print(f"4. Paste the prompt from chatgpt_evaluation_prompt.txt")
    print(f"5. Replace [JSON data] section with copied content")
    print(f"6. Submit to get consistency scores")

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 7):
        print("Error: Python 3.7+ required")
        sys.exit(1)
    
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nData collection interrupted by user")
    except Exception as e:
        print(f"\n\nError during collection: {str(e)}")
        import traceback
        traceback.print_exc()
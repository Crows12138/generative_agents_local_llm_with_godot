#!/usr/bin/env python3
"""
LLM Monitor - Tracks AI model responses and performance
"""
import time
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class LLMCall:
    """Record of an LLM API call"""
    timestamp: datetime
    agent_name: str
    prompt_type: str
    prompt: str
    response: str
    response_time: float
    tokens_used: int = 0
    model_name: str = "local"
    success: bool = True
    error: Optional[str] = None

class LLMMonitor:
    """Monitor and analyze LLM interactions"""
    
    def __init__(self):
        self.calls: List[LLMCall] = []
        self.start_time = datetime.now()
        
    def log_call(self, 
                 agent_name: str,
                 prompt_type: str, 
                 prompt: str, 
                 response: str,
                 response_time: float,
                 tokens_used: int = 0,
                 model_name: str = "local",
                 success: bool = True,
                 error: Optional[str] = None):
        """Log an LLM call"""
        call = LLMCall(
            timestamp=datetime.now(),
            agent_name=agent_name,
            prompt_type=prompt_type,
            prompt=prompt,
            response=response,
            response_time=response_time,
            tokens_used=tokens_used,
            model_name=model_name,
            success=success,
            error=error
        )
        self.calls.append(call)
        
    def get_stats(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        if not self.calls:
            return {"message": "No LLM calls recorded yet"}
        
        total_calls = len(self.calls)
        successful_calls = len([c for c in self.calls if c.success])
        failed_calls = total_calls - successful_calls
        
        response_times = [c.response_time for c in self.calls if c.success]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        total_tokens = sum(c.tokens_used for c in self.calls)
        
        # Agent activity
        agent_stats = {}
        for call in self.calls:
            if call.agent_name not in agent_stats:
                agent_stats[call.agent_name] = {"calls": 0, "avg_response_time": 0}
            agent_stats[call.agent_name]["calls"] += 1
        
        # Calculate average response times per agent
        for agent_name in agent_stats:
            agent_calls = [c for c in self.calls if c.agent_name == agent_name and c.success]
            if agent_calls:
                agent_stats[agent_name]["avg_response_time"] = sum(c.response_time for c in agent_calls) / len(agent_calls)
        
        return {
            "session_duration": str(datetime.now() - self.start_time),
            "total_calls": total_calls,
            "successful_calls": successful_calls,
            "failed_calls": failed_calls,
            "success_rate": f"{(successful_calls/total_calls)*100:.1f}%",
            "average_response_time": f"{avg_response_time:.3f}s",
            "total_tokens": total_tokens,
            "agent_stats": agent_stats,
            "recent_errors": [c.error for c in self.calls[-5:] if c.error]
        }
    
    def print_detailed_log(self, last_n: int = 10):
        """Print detailed log of recent calls"""
        print(f"\n🤖 LLM MONITOR - Last {last_n} Calls")
        print("=" * 80)
        
        recent_calls = self.calls[-last_n:] if self.calls else []
        
        for i, call in enumerate(recent_calls, 1):
            status = "✅" if call.success else "❌"
            print(f"\n{status} Call #{len(self.calls) - len(recent_calls) + i}")
            print(f"  🕐 Time: {call.timestamp.strftime('%H:%M:%S.%f')[:-3]}")
            print(f"  👤 Agent: {call.agent_name}")
            print(f"  📝 Type: {call.prompt_type}")
            print(f"  ⚡ Response Time: {call.response_time:.3f}s")
            print(f"  📊 Tokens: {call.tokens_used}")
            
            if len(call.prompt) > 100:
                print(f"  💭 Prompt: {call.prompt[:97]}...")
            else:
                print(f"  💭 Prompt: {call.prompt}")
            
            if len(call.response) > 100:
                print(f"  💬 Response: {call.response[:97]}...")
            else:
                print(f"  💬 Response: {call.response}")
                
            if call.error:
                print(f"  ⚠️  Error: {call.error}")
    
    def save_to_file(self, filename: str = None):
        """Save monitoring data to JSON file"""
        if filename is None:
            filename = f"llm_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            "session_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_calls": len(self.calls)
            },
            "stats": self.get_stats(),
            "calls": [asdict(call) for call in self.calls]
        }
        
        # Convert datetime objects to strings for JSON serialization
        for call in data["calls"]:
            call["timestamp"] = call["timestamp"].isoformat()
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 LLM monitoring data saved to: {filename}")
    
    def analyze_patterns(self):
        """Analyze patterns in LLM usage"""
        if not self.calls:
            print("No data to analyze")
            return
        
        print("\n📊 LLM USAGE PATTERNS")
        print("=" * 50)
        
        # Response time patterns
        response_times = [c.response_time for c in self.calls if c.success]
        if response_times:
            print(f"Response Times:")
            print(f"  Min: {min(response_times):.3f}s")
            print(f"  Max: {max(response_times):.3f}s") 
            print(f"  Avg: {sum(response_times)/len(response_times):.3f}s")
        
        # Prompt type distribution
        prompt_types = {}
        for call in self.calls:
            prompt_types[call.prompt_type] = prompt_types.get(call.prompt_type, 0) + 1
        
        print(f"\nPrompt Type Distribution:")
        for prompt_type, count in sorted(prompt_types.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.calls)) * 100
            print(f"  {prompt_type}: {count} calls ({percentage:.1f}%)")
        
        # Agent activity over time
        print(f"\nAgent Activity:")
        agent_activity = {}
        for call in self.calls:
            agent_activity[call.agent_name] = agent_activity.get(call.agent_name, 0) + 1
        
        for agent, count in sorted(agent_activity.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(self.calls)) * 100
            print(f"  {agent}: {count} calls ({percentage:.1f}%)")

# Global monitor instance
monitor = LLMMonitor()

def track_llm_call(func):
    """Decorator to automatically track LLM calls"""
    def wrapper(*args, **kwargs):
        start_time = time.time()
        agent_name = "unknown"
        
        # Try to extract agent name from args
        if args and hasattr(args[0], 'name'):
            agent_name = args[0].name
        
        try:
            result = func(*args, **kwargs)
            response_time = time.time() - start_time
            
            # Log successful call
            monitor.log_call(
                agent_name=agent_name,
                prompt_type=func.__name__,
                prompt=str(kwargs.get('prompt', 'N/A')),
                response=str(result),
                response_time=response_time,
                success=True
            )
            
            return result
            
        except Exception as e:
            response_time = time.time() - start_time
            
            # Log failed call
            monitor.log_call(
                agent_name=agent_name,
                prompt_type=func.__name__,
                prompt=str(kwargs.get('prompt', 'N/A')),
                response="",
                response_time=response_time,
                success=False,
                error=str(e)
            )
            
            raise e
    
    return wrapper

if __name__ == "__main__":
    # Test the monitor
    print("🤖 LLM Monitor Test")
    
    # Simulate some calls
    monitor.log_call("Bob", "dialogue", "What can I get you?", "A whiskey, please.", 0.245)
    monitor.log_call("Charlie", "action", "What should I do?", "Sip whiskey slowly", 0.312) 
    monitor.log_call("Sam", "dialogue", "Any song requests?", "Play something melancholy", 0.189)
    
    monitor.print_detailed_log()
    print("\n" + "=" * 80)
    stats = monitor.get_stats()
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    monitor.analyze_patterns()
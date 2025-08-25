#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AI Service + LlamaCpp Cozy Bar - Integration with existing ai_service
"""
import sys
import os
import inspect
import time
import requests
from datetime import datetime
from typing import Dict, Any, Optional

# Set up UTF-8 encoding for Windows console
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cozy_bar_demo'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import debugging tools
from llm_monitor import monitor
from action_tracer import tracer

def debug_trace(message: str, data: Any = None):
    """Ultra-detailed debugging information"""
    frame = inspect.currentframe().f_back
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    print(f"""
{'='*60}
🔍 [{timestamp}] DEBUG: {message}
📍 Location: {os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}
📦 Function: {frame.f_code.co_name}
""")
    
    if data:
        if isinstance(data, dict):
            print(f"📊 Data:")
            for key, value in data.items():
                print(f"   {key}: {value}")
        else:
            print(f"📊 Data: {data}")
    print('='*60)

class AIServiceClient:
    """Client for calling existing ai_service with fallback to LlamaCpp"""
    
    def __init__(self, ai_service_url: str = "http://127.0.0.1:8058"):
        self.ai_service_url = ai_service_url
        self.llamacpp_client = None
        self._init_llamacpp_fallback()
        
    def _init_llamacpp_fallback(self):
        """Initialize LlamaCpp as fallback"""
        try:
            from llama_cpp import Llama
            model_path = os.path.join(os.path.dirname(__file__), "..", "models", "llms", "Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf")
            
            if os.path.exists(model_path):
                debug_trace("Initializing LlamaCpp fallback")
                self.llamacpp_client = Llama(
                    model_path=model_path,
                    n_ctx=4096,
                    n_threads=8,
                    n_gpu_layers=-1,
                    verbose=False,
                    n_batch=512,
                    seed=42,
                    f16_kv=True,
                    use_mlock=True,
                    use_mmap=True,
                )
                debug_trace("LlamaCpp fallback initialized successfully")
            else:
                debug_trace(f"LlamaCpp model file not found: {model_path}")
                
        except ImportError:
            debug_trace("llama-cpp-python not installed, fallback disabled")
        except Exception as e:
            debug_trace(f"Failed to initialize LlamaCpp fallback: {e}")
    
    def is_ai_service_available(self) -> bool:
        """Check if ai_service is available"""
        try:
            response = requests.get(f"{self.ai_service_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate response using ai_service with LlamaCpp fallback"""
        start_time = time.time()
        
        # Try ai_service first
        if self.is_ai_service_available():
            try:
                debug_trace("Calling ai_service")
                
                # Create system prompt based on context
                system_prompt = self._create_system_prompt(context)
                full_prompt = f"System: {system_prompt}\n\nUser: {prompt}"
                
                # Call ai_service
                payload = {
                    "model_key": "qwen3",
                    "prompt": full_prompt,
                    "max_tokens": 100,
                    "temperature": 0.7
                }
                
                response = requests.post(
                    f"{self.ai_service_url}/v1/chat",
                    json=payload,
                    timeout=15
                )
                
                if response.status_code == 200:
                    result = response.json()
                    ai_response = result.get("response", "").strip()
                    response_time = time.time() - start_time
                    
                    # Log to monitor
                    monitor.log_call(
                        agent_name="ai_service",
                        prompt_type="generate",
                        prompt=prompt,
                        response=ai_response,
                        response_time=response_time,
                        model_name="ai_service_qwen3",
                        success=True
                    )
                    
                    debug_trace(f"AI Service response successful", {
                        "response_length": len(ai_response),
                        "response_time": f"{response_time:.3f}s"
                    })
                    
                    return ai_response
                else:
                    debug_trace(f"AI Service returned error: {response.status_code}")
                    
            except Exception as e:
                debug_trace(f"AI Service call failed: {e}")
        
        # Fallback to LlamaCpp if ai_service unavailable
        if self.llamacpp_client:
            try:
                debug_trace("Using LlamaCpp fallback")
                
                # Create system prompt based on context
                system_prompt = self._create_system_prompt(context)
                
                # Format prompt for Qwen3
                formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
                
                # Generate response
                output = self.llamacpp_client(
                    formatted_prompt,
                    max_tokens=100,
                    temperature=0.7,
                    top_p=0.9,
                    repeat_penalty=1.1,
                    stop=["<|im_end|>", "<|im_start|>"],
                    echo=False
                )
                
                response = output['choices'][0]['text'].strip()
                response_time = time.time() - start_time
                
                # Log to monitor
                monitor.log_call(
                    agent_name="llamacpp_fallback",
                    prompt_type="generate",
                    prompt=prompt,
                    response=response,
                    response_time=response_time,
                    model_name="LlamaCpp_Qwen3_fallback",
                    success=True
                )
                
                debug_trace(f"LlamaCpp fallback response successful", {
                    "response_length": len(response),
                    "response_time": f"{response_time:.3f}s"
                })
                
                return response
                
            except Exception as e:
                debug_trace(f"LlamaCpp fallback failed: {e}")
        
        # Ultimate fallback
        response_time = time.time() - start_time
        fallback_response = self._get_fallback_response(prompt, context)
        
        monitor.log_call(
            agent_name="ultimate_fallback",
            prompt_type="generate",
            prompt=prompt,
            response=fallback_response,
            response_time=response_time,
            success=False,
            error="All generation methods failed"
        )
        
        return fallback_response
    
    def _create_system_prompt(self, context: Dict[str, Any]) -> str:
        """Create system prompt based on context"""
        if not context:
            return "You are a helpful assistant in a cozy bar setting. Keep responses natural and brief."
        
        role = context.get("role", "person")
        mood = context.get("mood", "neutral")
        setting = context.get("setting", "bar")
        
        if role == "bartender":
            return f"You are an experienced, friendly bartender in a cozy neighborhood bar. You're currently feeling {mood}. Keep responses natural and conversational, as if speaking to a customer. Respond in one sentence only."
        elif role == "regular customer":
            return f"You are a thoughtful regular customer at a cozy bar, known for meaningful conversations. You're feeling {mood}. Keep responses authentic and contemplative, as if talking to someone nearby. Respond in one sentence only."
        elif role == "musician":
            return f"You are a creative musician who performs at this cozy bar. You're feeling {mood}. Keep responses artistic and engaging, as if talking to audience members. Respond in one sentence only."
        else:
            return f"You are a {role} in a {setting}, currently feeling {mood}. Keep responses natural and in character. Respond in one sentence only."
    
    def _get_fallback_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Fallback response when all methods fail"""
        if not context:
            return "I'm taking a moment to think..."
        
        role = context.get("role", "person")
        
        if "action" in prompt.lower():
            if role == "bartender":
                return "polishing glasses while keeping an eye on the customers"
            elif role == "regular customer":
                return "sipping their drink thoughtfully"
            elif role == "musician":
                return "tuning their instrument quietly"
            else:
                return "pausing to observe the atmosphere"
        else:
            if role == "bartender":
                return "What can I get you tonight?"
            elif role == "regular customer":
                return "It's been quite an evening, hasn't it?"
            elif role == "musician":
                return "Any song requests for tonight?"
            else:
                return "This is quite a nice place."

class EnhancedBarAgent:
    """Enhanced bar agent with AI Service + LlamaCpp integration"""
    
    def __init__(self, name: str, role: str, position: tuple, ai_client: AIServiceClient):
        self.name = name
        self.role = role
        self.position = position
        self.ai_client = ai_client
        self.memories = []
        self.mood = "neutral"
        self.energy = 100
        self.drunk_level = 0
        self.social_battery = 100
        self.conversation_history = []
        
    def generate_action_with_ai(self) -> str:
        """Generate action using AI Service + LlamaCpp"""
        context = {
            "role": self.role,
            "mood": self.mood,
            "energy": self.energy,
            "setting": "cozy bar",
            "time_period": "evening"
        }
        
        recent_memories = ', '.join([m['content'] for m in self.memories[-2:]]) if self.memories else 'enjoying the peaceful atmosphere'
        
        prompt = f"""As {self.name}, a {self.role} in a cozy bar, what would you naturally do right now? 

Context:
- You're feeling {self.mood}
- Energy level: {self.energy}%
- Time: Evening at the bar
- Recent memories: {recent_memories}

Describe ONE simple action in a few words, starting with a verb (like 'mixing drinks', 'sipping whiskey', 'tuning guitar', etc.). Be brief and natural."""
        
        tracer.trace_action(
            agent_name=self.name,
            action_type="ai_service_action_generation",
            action="requesting AI Service action",
            context=context,
            decision_factors={"prompt_length": len(prompt)}
        )
        
        action = self.ai_client.generate_response(prompt, context=context)
        
        # Clean up the response
        action = action.strip().lower()
        # Remove common prefixes
        prefixes_to_remove = [self.name.lower(), "i would", "i am", "i'm", "is", "would be"]
        for prefix in prefixes_to_remove:
            if action.startswith(prefix):
                action = action[len(prefix):].strip()
        
        # Add to memories
        self.add_memory(f"I was {action}")
        
        tracer.trace_action(
            agent_name=self.name,
            action_type="action_execution",
            action=action,
            context=context,
            success=True,
            outcome="completed"
        )
        
        return action
    
    def generate_dialogue_with_ai(self, situation: str = "") -> str:
        """Generate dialogue using AI Service + LlamaCpp"""
        conversation_context = {
            "role": self.role,
            "mood": self.mood,
            "name": self.name,
            "setting": "cozy bar",
            "conversation_history": self.conversation_history[-2:] if self.conversation_history else []
        }
        
        recent_conversation = ', '.join(self.conversation_history[-2:]) if self.conversation_history else 'starting a new conversation'
        
        prompt = f"""You are {self.name}, a {self.role} at a cozy bar. You're feeling {self.mood}.

Situation: {situation if situation else 'casual evening conversation'}
Recent chat: {recent_conversation}

What would you say right now? Give me ONLY the dialogue - just what you would actually say out loud. Make it natural and in character. One sentence only."""
        
        tracer.trace_action(
            agent_name=self.name,
            action_type="ai_service_dialogue_generation",
            action="requesting AI Service dialogue",
            context=conversation_context,
            decision_factors={"prompt_length": len(prompt)}
        )
        
        dialogue = self.ai_client.generate_response(prompt, context=conversation_context)
        
        # Clean up the response
        dialogue = dialogue.strip()
        # Remove quotes if present
        if (dialogue.startswith('"') and dialogue.endswith('"')) or (dialogue.startswith("'") and dialogue.endswith("'")):
            dialogue = dialogue[1:-1]
        # Remove character name prefix if present
        if dialogue.lower().startswith(self.name.lower() + ":"):
            dialogue = dialogue[len(self.name)+1:].strip()
        if dialogue.lower().startswith("i say:") or dialogue.lower().startswith("i would say:"):
            dialogue = dialogue.split(":", 1)[1].strip()
        
        # Add to conversation history and memories
        self.conversation_history.append(dialogue)
        self.add_memory(f"I said: '{dialogue}'")
        
        tracer.trace_action(
            agent_name=self.name,
            action_type="dialogue_spoken",
            action=dialogue,
            context=conversation_context,
            success=True,
            outcome="dialogue_generated"
        )
        
        return dialogue
    
    def add_memory(self, content: str):
        """Add memory with timestamp"""
        self.memories.append({
            "content": content,
            "timestamp": datetime.now(),
            "mood": self.mood
        })
        
        # Keep only recent memories
        if len(self.memories) > 10:
            self.memories = self.memories[-10:]
    
    def get_status(self) -> Dict[str, Any]:
        """Get detailed agent status"""
        return {
            "name": self.name,
            "role": self.role,
            "position": self.position,
            "mood": self.mood,
            "energy": self.energy,
            "drunk_level": self.drunk_level,
            "social_battery": self.social_battery,
            "memory_count": len(self.memories),
            "recent_memories": [m["content"] for m in self.memories[-3:]]
        }

def run_ai_service_cozy_bar():
    """Run cozy bar with AI Service + LlamaCpp integration"""
    debug_trace("Starting AI Service + LlamaCpp Cozy Bar System")
    
    # Initialize AI Service client with LlamaCpp fallback
    ai_client = AIServiceClient()
    
    debug_trace("Checking AI Service availability")
    if ai_client.is_ai_service_available():
        print("✅ AI Service is available at http://127.0.0.1:8058")
        if ai_client.llamacpp_client:
            print("✅ LlamaCpp fallback is ready")
        else:
            print("⚠️ LlamaCpp fallback not available")
    else:
        print("❌ AI Service not available, checking fallback...")
        if ai_client.llamacpp_client:
            print("✅ Using LlamaCpp fallback mode")
        else:
            print("⚠️ No AI backends available, using simple fallbacks")
    
    # Create enhanced agents with AI Service integration
    agents = {}
    
    agent_configs = [
        ("Bob", "bartender", (6, 2)),
        ("Alice", "regular customer", (2, 6)),
        ("Sam", "musician", (9, 6))
    ]
    
    debug_trace("Creating AI Service-enhanced agents")
    for name, role, position in agent_configs:
        agent = EnhancedBarAgent(name, role, position, ai_client)
        agents[name] = agent
        debug_trace(f"Created enhanced agent: {name}", agent.get_status())
    
    debug_trace("Starting enhanced simulation with AI Service + LlamaCpp")
    
    # Run simulation with AI Service integration
    for step in range(3):
        debug_trace(f"=== AI SERVICE SIMULATION STEP {step+1} ===")
        print(f"\n🎭 STEP {step+1}: AI Service + LlamaCpp Enhanced Bar Simulation")
        print("=" * 60)
        
        for name, agent in agents.items():
            print(f"\n👤 {name} ({agent.role}):")
            
            # Capture pre-action state
            pre_state = agent.get_status()
            tracer.capture_state(name, pre_state)
            
            # Generate AI-powered action
            start_time = time.time()
            action = agent.generate_action_with_ai()
            action_time = time.time() - start_time
            
            print(f"   🎬 Action: {action}")
            debug_trace(f"{name} AI action generated", {
                "action": action,
                "response_time": f"{action_time:.3f}s"
            })
            
            # Generate AI-powered dialogue
            start_time = time.time()
            dialogue = agent.generate_dialogue_with_ai("evening bar atmosphere")
            dialogue_time = time.time() - start_time
            
            print(f"   💬 Says: \"{dialogue}\"")
            debug_trace(f"{name} AI dialogue generated", {
                "dialogue": dialogue,
                "response_time": f"{dialogue_time:.3f}s"
            })
            
            # Show state changes
            post_state = agent.get_status()
            state_changes = tracer.detect_state_changes(name, post_state)
            if state_changes:
                print(f"   📊 Changes: {len(state_changes)} attributes updated")
        
        # Show AI performance stats
        if step == 0:  # Show stats after first step
            print(f"\n🤖 AI PERFORMANCE:")
            stats = monitor.get_stats()
            if "total_calls" in stats:
                print(f"   Total AI calls: {stats['total_calls']}")
                print(f"   Success rate: {stats['success_rate']}")
                print(f"   Average response time: {stats['average_response_time']}")
        
        if step < 2:
            try:
                input(f"\n⏸️ Press Enter to continue to step {step+2}...")
            except EOFError:
                print(f"\n⏭️ Auto-continuing to step {step+2}...")
    
    debug_trace("AI Service simulation completed")
    
    # Show comprehensive reports
    print(f"\n📊 FINAL REPORTS")
    print("=" * 60)
    
    # AI monitoring report
    print(f"\n🤖 AI SERVICE + LLAMACPP MONITORING REPORT:")
    monitor.print_detailed_log(10)
    
    stats = monitor.get_stats()
    print(f"\nAI Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Action tracing report
    print(f"\n🎯 ACTION ANALYSIS REPORT:")
    tracer.print_action_timeline(last_n=15)
    
    for agent_name in agents.keys():
        print(f"\n📈 {agent_name} Pattern Analysis:")
        analysis = tracer.analyze_action_patterns(agent_name)
        for key, value in analysis.items():
            print(f"  {key}: {value}")
    
    # Save detailed logs
    monitor.save_to_file("ai_service_session.json")
    tracer.save_trace_data("ai_service_traces.json")
    
    print(f"\n✅ AI Service + LlamaCpp simulation completed!")
    print(f"📁 Detailed logs saved to files")
    print(f"🔧 System used: AI Service (primary) + LlamaCpp (fallback)")

if __name__ == "__main__":
    print("🍺🤖 AI Service + LlamaCpp Enhanced Cozy Bar System")
    print("=" * 60)
    print("This version integrates with existing ai_service and uses LlamaCpp as fallback.")
    print("Primary: AI Service at http://127.0.0.1:8058")
    print("Fallback: Local Qwen3-30B-A3B-Instruct GGUF model via LlamaCpp")
    print("=" * 60)
    
    run_ai_service_cozy_bar()
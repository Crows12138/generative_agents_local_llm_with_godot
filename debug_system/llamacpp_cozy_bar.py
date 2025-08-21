#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LlamaCpp Cozy Bar - Direct GGUF model integration using llama-cpp-python
"""
import sys
import os
import inspect
import time
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

class LlamaCppClient:
    """Client for running GGUF models directly via llama-cpp-python"""
    
    def __init__(self, model_path: str = None):
        self.model_path = model_path or os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf")
        self.model = None
        
    def load_model(self):
        """Load the GGUF model"""
        debug_trace(f"Loading GGUF model: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            debug_trace(f"Model file not found: {self.model_path}")
            return False
        
        try:
            from llama_cpp import Llama
            
            debug_trace("Initializing llama-cpp-python")
            
            # Load model with optimized settings
            self.model = Llama(
                model_path=self.model_path,
                n_ctx=4096,  # Context length
                n_threads=8,  # CPU threads
                n_gpu_layers=-1,  # Use all GPU layers if available
                verbose=False,
                n_batch=512,
                seed=42,
                f16_kv=True,
                use_mlock=True,
                use_mmap=True,
            )
            
            debug_trace("GGUF model loaded successfully")
            return True
            
        except ImportError:
            debug_trace("llama-cpp-python not installed")
            return False
        except Exception as e:
            debug_trace(f"Failed to load GGUF model: {e}")
            return False
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model is not None
    
    def generate_response(self, prompt: str, context: Dict[str, Any] = None) -> str:
        """Generate response using the loaded GGUF model"""
        if not self.is_model_loaded():
            return self._get_fallback_response(prompt, context)
        
        start_time = time.time()
        
        try:
            # Create system prompt based on context
            system_prompt = self._create_system_prompt(context)
            
            # Format prompt for Qwen3
            formatted_prompt = f"<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
            
            debug_trace(f"Generating response", {
                "prompt_length": len(prompt),
                "formatted_prompt_length": len(formatted_prompt)
            })
            
            # Generate response
            output = self.model(
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
                agent_name="llamacpp",
                prompt_type="generate",
                prompt=prompt,
                response=response,
                response_time=response_time,
                model_name="Qwen3-30B-GGUF",
                success=True
            )
            
            debug_trace(f"Response generated successfully", {
                "response_length": len(response),
                "response_time": f"{response_time:.3f}s"
            })
            
            return response
            
        except Exception as e:
            response_time = time.time() - start_time
            debug_trace(f"Generation failed: {e}")
            
            monitor.log_call(
                agent_name="llamacpp",
                prompt_type="generate",
                prompt=prompt,
                response="",
                response_time=response_time,
                success=False,
                error=str(e)
            )
            
            return self._get_fallback_response(prompt, context)
    
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
        """Fallback response when model is unavailable"""
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
    """Enhanced bar agent with llama-cpp-python integration"""
    
    def __init__(self, name: str, role: str, position: tuple, llamacpp_client: LlamaCppClient):
        self.name = name
        self.role = role
        self.position = position
        self.llamacpp_client = llamacpp_client
        self.memories = []
        self.mood = "neutral"
        self.energy = 100
        self.drunk_level = 0
        self.social_battery = 100
        self.conversation_history = []
        
    def generate_action_with_llamacpp(self) -> str:
        """Generate action using llama-cpp-python"""
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
            action_type="llamacpp_action_generation",
            action="requesting LlamaCpp action",
            context=context,
            decision_factors={"prompt_length": len(prompt)}
        )
        
        action = self.llamacpp_client.generate_response(prompt, context=context)
        
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
    
    def generate_dialogue_with_llamacpp(self, situation: str = "") -> str:
        """Generate dialogue using llama-cpp-python"""
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
            action_type="llamacpp_dialogue_generation",
            action="requesting LlamaCpp dialogue",
            context=conversation_context,
            decision_factors={"prompt_length": len(prompt)}
        )
        
        dialogue = self.llamacpp_client.generate_response(prompt, context=conversation_context)
        
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

def run_llamacpp_cozy_bar():
    """Run cozy bar with llama-cpp-python integration"""
    debug_trace("Starting LlamaCpp Cozy Bar System")
    
    # Initialize LlamaCpp client
    llamacpp_client = LlamaCppClient()
    
    debug_trace("Loading GGUF model via llama-cpp-python")
    print("🤖 Loading local GGUF model via llama-cpp-python...")
    print("This will use your local Qwen3 GGUF file directly...")
    
    model_loaded = llamacpp_client.load_model()
    
    if model_loaded:
        debug_trace("✅ GGUF model loaded successfully")
        print("✅ GGUF model loaded successfully!")
        print("🎯 Using local model: Qwen3-30B-A3B-Instruct GGUF")
    else:
        debug_trace("❌ Failed to load GGUF model, using fallback")
        print("⚠️ Failed to load GGUF model, using fallback responses")
        print("   Please install: pip install llama-cpp-python")
    
    # Create enhanced agents with LlamaCpp integration
    agents = {}
    
    agent_configs = [
        ("Bob", "bartender", (6, 2)),
        ("Alice", "regular customer", (2, 6)),
        ("Sam", "musician", (9, 6))
    ]
    
    debug_trace("Creating LlamaCpp-enhanced agents")
    for name, role, position in agent_configs:
        agent = EnhancedBarAgent(name, role, position, llamacpp_client)
        agents[name] = agent
        debug_trace(f"Created enhanced agent: {name}", agent.get_status())
    
    debug_trace("Starting enhanced simulation with LlamaCpp")
    
    # Run simulation with LlamaCpp integration
    for step in range(3):
        debug_trace(f"=== LLAMACPP SIMULATION STEP {step+1} ===")
        print(f"\n🎭 STEP {step+1}: LlamaCpp-Enhanced Bar Simulation")
        print("=" * 60)
        
        for name, agent in agents.items():
            print(f"\n👤 {name} ({agent.role}):")
            
            # Capture pre-action state
            pre_state = agent.get_status()
            tracer.capture_state(name, pre_state)
            
            # Generate LlamaCpp-powered action
            start_time = time.time()
            action = agent.generate_action_with_llamacpp()
            action_time = time.time() - start_time
            
            print(f"   🎬 Action: {action}")
            debug_trace(f"{name} LlamaCpp action generated", {
                "action": action,
                "response_time": f"{action_time:.3f}s"
            })
            
            # Generate LlamaCpp-powered dialogue
            start_time = time.time()
            dialogue = agent.generate_dialogue_with_llamacpp("evening bar atmosphere")
            dialogue_time = time.time() - start_time
            
            print(f"   💬 Says: \"{dialogue}\"")
            debug_trace(f"{name} LlamaCpp dialogue generated", {
                "dialogue": dialogue,
                "response_time": f"{dialogue_time:.3f}s"
            })
            
            # Show state changes
            post_state = agent.get_status()
            state_changes = tracer.detect_state_changes(name, post_state)
            if state_changes:
                print(f"   📊 Changes: {len(state_changes)} attributes updated")
        
        # Show LlamaCpp performance stats
        if step == 0:  # Show stats after first step
            print(f"\n🤖 LLAMACPP PERFORMANCE:")
            stats = monitor.get_stats()
            if "total_calls" in stats:
                print(f"   Total LlamaCpp calls: {stats['total_calls']}")
                print(f"   Success rate: {stats['success_rate']}")
                print(f"   Average response time: {stats['average_response_time']}")
        
        if step < 2:
            try:
                input(f"\n⏸️ Press Enter to continue to step {step+2}...")
            except EOFError:
                print(f"\n⏭️ Auto-continuing to step {step+2}...")
    
    debug_trace("LlamaCpp simulation completed")
    
    # Show comprehensive reports
    print(f"\n📊 FINAL REPORTS")
    print("=" * 60)
    
    # LLM monitoring report
    print(f"\n🤖 LLAMACPP MONITORING REPORT:")
    monitor.print_detailed_log(10)
    
    stats = monitor.get_stats()
    print(f"\nLlamaCpp Statistics:")
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
    monitor.save_to_file("llamacpp_session.json")
    tracer.save_trace_data("llamacpp_traces.json")
    
    print(f"\n✅ LlamaCpp simulation completed with local GGUF model!")
    print(f"📁 Detailed logs saved to files")

if __name__ == "__main__":
    print("🍺🦙 LlamaCpp Enhanced Cozy Bar System")
    print("=" * 60)
    print("This version uses llama-cpp-python to load local GGUF files.")
    print("Using your local Qwen3-30B-A3B-Instruct GGUF model directly.")
    print("=" * 60)
    
    run_llamacpp_cozy_bar()
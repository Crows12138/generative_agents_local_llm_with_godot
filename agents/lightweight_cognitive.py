"""
Lightweight Cognitive Module
Designed specifically for Godot game integration, bypassing Reverie complex architecture
"""

import time
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import threading
from contextlib import contextmanager

# Import our optimization components
from .llm_output_adapter import get_llm_adapter, adapt_llm_output
from .memory_optimizer import get_memory_optimizer, memory_managed

class CognitiveState(Enum):
    """Cognitive state enumeration"""
    IDLE = "idle"
    PERCEIVING = "perceiving"
    THINKING = "thinking"
    PLANNING = "planning"
    REFLECTING = "reflecting"
    ACTING = "acting"
    CONVERSING = "conversing"

@dataclass
class LightweightMemory:
    """Lightweight memory structure"""
    short_term: List[Dict[str, Any]] = field(default_factory=list)
    long_term: List[Dict[str, Any]] = field(default_factory=list)
    working_memory: Dict[str, Any] = field(default_factory=dict)
    max_short_term: int = 10
    max_long_term: int = 50
    
    def add_memory(self, content: str, memory_type: str = "observation", importance: int = 5):
        """Add memory"""
        memory_entry = {
            'content': content,
            'type': memory_type,
            'importance': importance,
            'timestamp': time.time(),
            'access_count': 0
        }
        
        self.short_term.append(memory_entry)
        
        # Limit short-term memory size
        if len(self.short_term) > self.max_short_term:
            # Move most important memories to long-term memory
            oldest = self.short_term.pop(0)
            if oldest['importance'] >= 7:
                self.long_term.append(oldest)
                
        # Limit long-term memory size
        if len(self.long_term) > self.max_long_term:
            self.long_term = sorted(self.long_term, key=lambda x: x['importance'], reverse=True)[:self.max_long_term]
    
    def get_relevant_memories(self, query: str = "", limit: int = 5) -> List[Dict[str, Any]]:
        """Get relevant memories"""
        all_memories = self.short_term + self.long_term
        
        if not query:
            # Return recent memories
            return sorted(all_memories, key=lambda x: x['timestamp'], reverse=True)[:limit]
        
        # Simple relevance matching
        relevant = []
        query_lower = query.lower()
        
        for memory in all_memories:
            if query_lower in memory['content'].lower():
                memory['access_count'] += 1
                relevant.append(memory)
        
        return sorted(relevant, key=lambda x: (x['importance'], x['access_count']), reverse=True)[:limit]

@dataclass
class CognitiveContext:
    """Cognitive context"""
    environment: Dict[str, Any] = field(default_factory=dict)
    recent_actions: List[str] = field(default_factory=list)
    current_goal: Optional[str] = None
    mood: str = "neutral"
    energy_level: float = 1.0
    focus_area: Optional[str] = None

class LightweightCognitive:
    """Lightweight cognitive module"""
    
    def __init__(self, agent_name: str, personality: str = "helpful assistant"):
        self.agent_name = agent_name
        self.personality = personality
        self.logger = logging.getLogger(f"cognitive.{agent_name}")
        
        # Core components
        self.memory = LightweightMemory()
        self.context = CognitiveContext()
        self.state = CognitiveState.IDLE
        self.llm_adapter = get_llm_adapter()
        self.memory_optimizer = get_memory_optimizer()
        
        # State management
        self._lock = threading.RLock()
        self._processing = False
        
        # Configuration
        self.config = {
            'max_response_time': 10.0,  # Maximum response time
            'enable_reflection': True,
            'enable_planning': True,
            'memory_consolidation_interval': 300,  # 5 minutes
            'debug_mode': False
        }
        
        # Performance statistics
        self.stats = {
            'total_requests': 0,
            'successful_responses': 0,
            'average_response_time': 0.0,
            'last_activity': time.time()
        }
        
        self.logger.info(f"Lightweight cognitive module initialized for {agent_name}")

    @contextmanager
    def _cognitive_operation(self, operation_name: str):
        """Cognitive operation context manager"""
        with self._lock:
            if self._processing:
                raise RuntimeError(f"Cognitive module busy with another operation")
            
            self._processing = True
            old_state = self.state
            start_time = time.time()
            
            try:
                with memory_managed(f"cognitive_{operation_name}"):
                    self.logger.debug(f"Starting {operation_name}")
                    yield
                    
            except Exception as e:
                self.logger.error(f"Cognitive operation {operation_name} failed: {e}")
                raise
            finally:
                self._processing = False
                self.state = old_state
                
                # Update statistics
                duration = time.time() - start_time
                self.stats['total_requests'] += 1
                self.stats['last_activity'] = time.time()
                
                if duration <= self.config['max_response_time']:
                    self.stats['successful_responses'] += 1
                
                # Update average response time
                old_avg = self.stats['average_response_time']
                total = self.stats['total_requests']
                self.stats['average_response_time'] = (old_avg * (total - 1) + duration) / total
                
                self.logger.debug(f"Completed {operation_name} in {duration:.3f}s")

    def perceive_environment(self, environment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Environment perception - lightweight implementation"""
        with self._cognitive_operation("perceive"):
            self.state = CognitiveState.PERCEIVING
            self.context.environment = environment_data
            
            # Build perception prompt
            prompt = self._build_perception_prompt(environment_data)
            
            # Get LLM response
            try:
                raw_response = self._get_llm_response(prompt, max_tokens=150)
                adapted_response = self.llm_adapter.adapt_output(
                    raw_response, 'perceive', self.agent_name
                )
                
                # Process perception results
                perceptions = self._process_perception_response(adapted_response, environment_data)
                
                # Store in memory
                for perception in perceptions:
                    self.memory.add_memory(
                        content=str(perception),
                        memory_type="perception",
                        importance=perception.get('importance', 5)
                    )
                
                return perceptions
                
            except Exception as e:
                self.logger.error(f"Perception failed: {e}")
                return self._fallback_perception(environment_data)

    def think_and_plan(self, goal: str = None) -> Dict[str, Any]:
        """Think and plan"""
        with self._cognitive_operation("plan"):
            self.state = CognitiveState.PLANNING
            
            if goal:
                self.context.current_goal = goal
            
            # Build planning prompt
            prompt = self._build_planning_prompt()
            
            try:
                raw_response = self._get_llm_response(prompt, max_tokens=200)
                adapted_response = self.llm_adapter.adapt_output(
                    raw_response, 'plan', self.agent_name
                )
                
                plan = self._process_planning_response(adapted_response)
                
                # Record plan
                self.memory.add_memory(
                    content=f"Planned: {plan.get('action', 'unknown')}",
                    memory_type="plan",
                    importance=6
                )
                
                return plan
                
            except Exception as e:
                self.logger.error(f"Planning failed: {e}")
                return self._fallback_plan()

    def reflect_on_experience(self, experience: str = None) -> Dict[str, Any]:
        """Reflect on experience"""
        if not self.config['enable_reflection']:
            return {'insight': 'Reflection disabled', 'mood': self.context.mood}
        
        with self._cognitive_operation("reflect"):
            self.state = CognitiveState.REFLECTING
            
            # Build reflection prompt
            prompt = self._build_reflection_prompt(experience)
            
            try:
                raw_response = self._get_llm_response(prompt, max_tokens=180)
                adapted_response = self.llm_adapter.adapt_output(
                    raw_response, 'reflect', self.agent_name
                )
                
                reflection = self._process_reflection_response(adapted_response)
                
                # Update mood and energy
                if 'mood' in reflection:
                    self.context.mood = reflection['mood']
                
                # Record reflection
                self.memory.add_memory(
                    content=f"Reflected: {reflection.get('insight', 'general_reflection')}",
                    memory_type="reflection",
                    importance=4
                )
                
                return reflection
                
            except Exception as e:
                self.logger.error(f"Reflection failed: {e}")
                return self._fallback_reflection()

    def converse(self, speaker: str, message: str, context: Dict[str, Any] = None) -> str:
        """Generate conversation"""
        with self._cognitive_operation("converse"):
            self.state = CognitiveState.CONVERSING
            
            # Build conversation prompt
            prompt = self._build_conversation_prompt(speaker, message, context)
            
            try:
                raw_response = self._get_llm_response(prompt, max_tokens=150)
                response = self.llm_adapter.adapt_output(
                    raw_response, 'converse', self.agent_name
                )
                
                # Record conversation
                self.memory.add_memory(
                    content=f"Conversation with {speaker}: {message[:50]}",
                    memory_type="conversation",
                    importance=5
                )
                
                self.memory.add_memory(
                    content=f"Responded to {speaker}: {response[:50]}",
                    memory_type="response",
                    importance=4
                )
                
                return str(response)
                
            except Exception as e:
                self.logger.error(f"Conversation failed: {e}")
                return self._fallback_conversation(speaker)

    def _build_perception_prompt(self, environment_data: Dict[str, Any]) -> str:
        """Build perception prompt"""
        events = environment_data.get('events', [])[:3]  # Max 3 events
        objects = environment_data.get('objects', [])[:5]  # Max 5 objects
        
        prompt = f"Character: {self.agent_name}\n"
        prompt += f"Personality: {self.personality}\n"
        prompt += "Task: Observe and identify the most important things in this environment.\n"
        
        if events:
            prompt += f"Events: {', '.join(str(e) for e in events)}\n"
        if objects:
            prompt += f"Objects: {', '.join(str(o) for o in objects)}\n"
            
        prompt += "List 1-3 most important observations:"
        return prompt

    def _build_planning_prompt(self) -> str:
        """Build planning prompt"""
        recent_memories = self.memory.get_relevant_memories(limit=3)
        goal = self.context.current_goal or "explore and interact appropriately"
        
        prompt = f"Character: {self.agent_name}\n"
        prompt += f"Personality: {self.personality}\n"
        prompt += f"Current goal: {goal}\n"
        
        if recent_memories:
            prompt += "Recent memories:\n"
            for mem in recent_memories:
                prompt += f"- {mem['content'][:50]}\n"
        
        prompt += "What should I do next? Choose one specific action:"
        return prompt

    def _build_reflection_prompt(self, experience: str = None) -> str:
        """Build reflection prompt"""
        recent_memories = self.memory.get_relevant_memories(limit=2)
        
        prompt = f"Character: {self.agent_name}\n"
        prompt += f"Current mood: {self.context.mood}\n"
        
        if experience:
            prompt += f"Recent experience: {experience}\n"
        elif recent_memories:
            prompt += "Recent experiences:\n"
            for mem in recent_memories:
                prompt += f"- {mem['content'][:40]}\n"
        
        prompt += "How do I feel about this? What insight can I gain?"
        return prompt

    def _build_conversation_prompt(self, speaker: str, message: str, context: Dict[str, Any] = None) -> str:
        """Build conversation prompt"""
        relevant_memories = self.memory.get_relevant_memories(message, limit=2)
        
        prompt = f"Character: {self.agent_name}\n"
        prompt += f"Personality: {self.personality}\n"
        prompt += f"Mood: {self.context.mood}\n"
        
        if relevant_memories:
            prompt += "Relevant memories:\n"
            for mem in relevant_memories:
                prompt += f"- {mem['content'][:40]}\n"
        
        prompt += f"\n{speaker} says: \"{message}\"\n"
        prompt += f"Reply as {self.agent_name} (keep it natural and brief):"
        return prompt

    def _get_llm_response(self, prompt: str, max_tokens: int = 200) -> str:
        """Get LLM response"""
        try:
            # This needs to call the actual LLM service
            # For now, return a simulated response for testing
            if hasattr(self, '_llm_service'):
                return self._llm_service.generate(prompt, max_tokens=max_tokens)
            else:
                # Simulated response - will be replaced with actual usage
                return f"Simulated response for: {prompt[:30]}..."
                
        except Exception as e:
            self.logger.error(f"LLM request failed: {e}")
            raise

    def _process_perception_response(self, response: Any, environment_data: Dict) -> List[Dict[str, Any]]:
        """Process perception response"""
        if isinstance(response, list):
            perceptions = []
            for i, item in enumerate(response[:3]):
                perceptions.append({
                    'description': str(item)[:100],
                    'importance': 5,
                    'type': 'observation',
                    'timestamp': time.time()
                })
            return perceptions
        else:
            return [{
                'description': str(response)[:100],
                'importance': 5,
                'type': 'general_observation',
                'timestamp': time.time()
            }]

    def _process_planning_response(self, response: Any) -> Dict[str, Any]:
        """Process planning response"""
        if isinstance(response, dict):
            return {
                'action': response.get('action', 'observe'),
                'reasoning': response.get('reasoning', 'default_reasoning'),
                'priority': response.get('priority', 'medium'),
                'estimated_duration': 'short'
            }
        else:
            return {
                'action': str(response)[:50] if response else 'observe',
                'reasoning': 'planning_complete',
                'priority': 'medium',
                'estimated_duration': 'short'
            }

    def _process_reflection_response(self, response: Any) -> Dict[str, Any]:
        """Process reflection response"""
        if isinstance(response, dict):
            return response
        else:
            return {
                'insight': str(response)[:100] if response else 'general_insight',
                'mood': self.context.mood,
                'confidence': 0.6
            }

    # Fallback methods
    def _fallback_perception(self, environment_data: Dict) -> List[Dict[str, Any]]:
        """Fallback for perception"""
        events = environment_data.get('events', [])
        if events:
            return [{'description': f'Noticed: {str(events[0])[:50]}', 'importance': 3, 'type': 'fallback'}]
        return [{'description': 'Environment seems normal', 'importance': 2, 'type': 'fallback'}]

    def _fallback_plan(self) -> Dict[str, Any]:
        """Fallback for planning"""
        return {
            'action': 'observe',
            'reasoning': 'Taking time to assess situation',
            'priority': 'low'
        }

    def _fallback_reflection(self) -> Dict[str, Any]:
        """Fallback for reflection"""
        return {
            'insight': 'Continuing with current approach',
            'mood': 'neutral'
        }

    def _fallback_conversation(self, speaker: str) -> str:
        """Fallback for conversation"""
        return f"Hello {speaker}, nice to meet you."

    def get_status(self) -> Dict[str, Any]:
        """Get cognitive module status"""
        return {
            'agent_name': self.agent_name,
            'state': self.state.value,
            'processing': self._processing,
            'mood': self.context.mood,
            'current_goal': self.context.current_goal,
            'memory_count': len(self.memory.short_term) + len(self.memory.long_term),
            'stats': self.stats.copy(),
            'last_activity': time.time() - self.stats['last_activity']
        }

    def cleanup(self):
        """Clean up resources"""
        with self._lock:
            self.memory.short_term.clear()
            self.memory.working_memory.clear()
            self.context.recent_actions.clear()
            
        self.logger.info(f"Cognitive module cleaned up for {self.agent_name}")

# Factory function
def create_lightweight_cognitive(agent_name: str, personality: str = "helpful assistant") -> LightweightCognitive:
    """Create lightweight cognitive module instance"""
    return LightweightCognitive(agent_name, personality)

# Test function
def test_lightweight_cognitive():
    """Test lightweight cognitive module"""
    print("Testing Lightweight Cognitive Module")
    
    # Create cognitive module
    cognitive = create_lightweight_cognitive("TestAgent", "curious explorer")
    
    # Test environment perception
    test_env = {
        'events': ['A bird is singing', 'Someone is walking'],
        'objects': ['tree', 'bench', 'lamp']
    }
    
    try:
        perceptions = cognitive.perceive_environment(test_env)
        print(f"Perceptions: {perceptions}")
        
        # Test planning
        plan = cognitive.think_and_plan("explore the area")
        print(f"Plan: {plan}")
        
        # Test reflection
        reflection = cognitive.reflect_on_experience("explored the park")
        print(f"Reflection: {reflection}")
        
        # Test conversation
        response = cognitive.converse("User", "Hello, how are you?")
        print(f"Conversation: {response}")
        
        # Status check
        status = cognitive.get_status()
        print(f"Status: {status}")
        
    except Exception as e:
        print(f"Test failed: {e}")
    finally:
        cognitive.cleanup()

if __name__ == "__main__":
    test_lightweight_cognitive()
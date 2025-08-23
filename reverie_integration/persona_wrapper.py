"""Persona Wrapper for Reverie Integration

This module creates a wrapper around Reverie's Persona class that:
1. Integrates with our existing SimpleAgent system
2. Uses local LLM via the adapter
3. Provides all cognitive functions (perceive, plan, reflect, etc.)
"""

import sys
import os
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

# Add project root to path
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from reverie_integration.llm_adapter import patch_reverie_llm_functions

# Patch Reverie functions before importing
patch_reverie_llm_functions()

class PersonaWrapper:
    """
    Wrapper that provides Reverie cognitive capabilities using local LLM
    Can work independently or integrate with existing SimpleAgent
    """
    
    def __init__(self, 
                 name: str,
                 age: int = 25,
                 personality: str = "friendly",
                 background: str = "bartender",
                 lifestyle: str = None):
        
        self.name = name
        self.age = age
        self.personality = personality
        self.background = background
        self.lifestyle = lifestyle or f"{name} is a {background} who works at a local establishment"
        
        # Initialize memory systems
        self.memories = {
            "events": [],           # Recent events
            "daily_plans": {},      # Daily schedules
            "reflections": [],      # Self-reflections
            "conversations": [],    # Conversation history
            "relationships": {},    # People and relationships
            "locations": {}         # Spatial memories
        }
        
        # Cognitive state
        self.current_plan = []
        self.current_activity = "idle"
        self.last_reflection_time = datetime.now()
        self.emotional_state = "neutral"
        
        # Try to initialize Reverie components if available
        self._init_reverie_components()
    
    def _init_reverie_components(self):
        """Initialize Reverie components if available"""
        
        self.reverie_available = False
        self.persona = None
        
        try:
            # Try to import and initialize Reverie Persona
            from reverie.backend_server.persona.persona import Persona
            from reverie.backend_server.persona.memory_structures.associative_memory import AssociativeMemory
            from reverie.backend_server.persona.memory_structures.scratch import Scratch
            
            # Create minimal Reverie persona
            self.persona = self._create_minimal_persona()
            self.reverie_available = True
            print(f"[PersonaWrapper] Reverie components initialized for {self.name}")
            
        except ImportError:
            print(f"[PersonaWrapper] Reverie not available, using lightweight cognitive modules for {self.name}")
        except Exception as e:
            print(f"[PersonaWrapper] Error initializing Reverie: {e}")
    
    def _create_minimal_persona(self):
        """Create a minimal Reverie persona for cognitive functions"""
        # This would create a minimal Reverie persona
        # For now, return None and use our own implementations
        return None
    
    def perceive(self, environment: Dict[str, Any]) -> List[str]:
        """
        Perceive and process environment information
        Returns list of observations
        """
        
        from reverie_integration.llm_adapter import reverie_llm_adapter
        
        # Build perception prompt
        prompt = f"""Character: {self.name}
Personality: {self.personality}
Background: {self.background}
Current location: {environment.get('location', 'unknown')}
Current time: {environment.get('time', datetime.now().strftime('%H:%M %p'))}

Environment description: {environment.get('description', '')}
Events happening: {', '.join(environment.get('events', []))}
Objects present: {', '.join(environment.get('objects', []))}
People present: {', '.join(environment.get('people', []))}

As {self.name}, list the most important things you observe in this environment and how they might affect your actions or mood."""
        
        response = reverie_llm_adapter.ChatGPT_request(prompt)
        
        # Parse response into individual observations
        observations = []
        for line in response.split('\n'):
            line = line.strip()
            if line and not line.startswith(('As ', 'Character:', 'The ', 'In ')):
                # Remove numbering and bullet points
                cleaned = line.lstrip('0123456789.- ')
                if len(cleaned) > 10:
                    observations.append(cleaned)
        
        # Store observations in memory
        self.memories["events"].append({
            "timestamp": datetime.now(),
            "type": "perception",
            "content": observations,
            "environment": environment
        })
        
        return observations
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieve relevant memories based on query
        Returns list of relevant memory items
        """
        
        relevant_memories = []
        
        # Search through all memory types for relevant items
        for memory_type, memories in self.memories.items():
            if isinstance(memories, list):
                for memory in memories[-20:]:  # Check recent 20 items
                    if isinstance(memory, dict) and 'content' in memory:
                        content_str = str(memory['content']).lower()
                        if query.lower() in content_str:
                            relevant_memories.append({
                                "type": memory_type,
                                "timestamp": memory.get("timestamp", "unknown"),
                                "content": memory['content'],
                                "relevance": self._calculate_relevance(query, content_str)
                            })
        
        # Sort by relevance and return top_k
        relevant_memories.sort(key=lambda x: x['relevance'], reverse=True)
        return relevant_memories[:top_k]
    
    def plan(self, time_horizon: str = "today") -> List[Dict[str, Any]]:
        """
        Create a plan based on personality, current situation, and memories
        Returns structured plan
        """
        
        from reverie_integration.llm_adapter import reverie_llm_adapter
        
        current_time = datetime.now()
        
        # Build planning prompt
        prompt = f"""Character: {self.name}
Age: {self.age}
Personality: {self.personality}
Background: {self.background}
Lifestyle: {self.lifestyle}

Current time: {current_time.strftime('%A, %B %d, %Y at %H:%M %p')}
Current activity: {self.current_activity}
Emotional state: {self.emotional_state}

Recent memories:
{self._format_recent_memories(5)}

Create a detailed plan for {time_horizon}. Include specific times and activities that match {self.name}'s personality and lifestyle.
Format each activity as: "HH:MM AM/PM - Activity description"
"""
        
        response = reverie_llm_adapter.ChatGPT_request(prompt)
        
        # Parse response into structured plan
        plan_items = []
        for line in response.split('\n'):
            line = line.strip()
            if ':' in line and ('AM' in line or 'PM' in line):
                try:
                    # Extract time and activity
                    time_part = line.split(' - ')[0].strip()
                    activity = ' - '.join(line.split(' - ')[1:]).strip()
                    
                    plan_items.append({
                        "time": time_part,
                        "activity": activity,
                        "planned_at": current_time
                    })
                except:
                    continue
        
        # Store plan
        self.memories["daily_plans"][time_horizon] = plan_items
        self.current_plan = plan_items
        
        return plan_items
    
    def reflect(self) -> List[str]:
        """
        Perform self-reflection based on recent experiences
        Returns list of insights/reflections
        """
        
        from reverie_integration.llm_adapter import reverie_llm_adapter
        
        current_time = datetime.now()
        time_since_reflection = current_time - self.last_reflection_time
        
        # Only reflect if enough time has passed or significant events occurred
        if time_since_reflection < timedelta(hours=1) and len(self.memories["events"]) < 5:
            return []
        
        # Build reflection prompt
        prompt = f"""Character: {self.name}
Personality: {self.personality}
Background: {self.background}

Recent experiences and events:
{self._format_recent_memories(10)}

Current emotional state: {self.emotional_state}
Time since last reflection: {time_since_reflection}

As {self.name}, reflect on your recent experiences. What have you learned? How do you feel? What insights do you have about yourself or others?
Provide 3-5 thoughtful reflections."""
        
        response = reverie_llm_adapter.ChatGPT_request(prompt)
        
        # Parse reflections
        reflections = []
        for line in response.split('\n'):
            line = line.strip()
            if line and len(line) > 20 and not line.startswith(('As ', 'Character:', 'Recent')):
                cleaned = line.lstrip('0123456789.- ')
                if len(cleaned) > 15:
                    reflections.append(cleaned)
        
        # Store reflections
        self.memories["reflections"].append({
            "timestamp": current_time,
            "insights": reflections
        })
        
        self.last_reflection_time = current_time
        return reflections
    
    def decide(self, observations: List[str], retrieved_memories: List[Dict] = None) -> Dict[str, Any]:
        """
        Make a decision based on observations and memories
        Returns decision with reasoning
        """
        
        from reverie_integration.llm_adapter import reverie_llm_adapter
        
        memories_text = ""
        if retrieved_memories:
            memories_text = f"Relevant memories:\n"
            for memory in retrieved_memories[:3]:
                memories_text += f"- {memory['content']}\n"
        
        prompt = f"""Character: {self.name}
Personality: {self.personality}
Background: {self.background}
Current activity: {self.current_activity}

Current observations:
{chr(10).join(f"- {obs}" for obs in observations)}

{memories_text}

Based on the current situation and {self.name}'s personality, what should {self.name} do next? 
Provide:
1. The specific action to take
2. Brief reasoning for this choice
3. Expected duration of the action

Format: "ACTION: [specific action] | REASON: [why] | DURATION: [how long]"
"""
        
        response = reverie_llm_adapter.ChatGPT_request(prompt)
        
        # Parse decision
        decision = {
            "action": "continue current activity",
            "reasoning": "maintaining status quo",
            "duration": "15 minutes",
            "timestamp": datetime.now()
        }
        
        if "|" in response:
            parts = response.split("|")
            for part in parts:
                part = part.strip()
                if part.startswith("ACTION:"):
                    decision["action"] = part.replace("ACTION:", "").strip()
                elif part.startswith("REASON:"):
                    decision["reasoning"] = part.replace("REASON:", "").strip()
                elif part.startswith("DURATION:"):
                    decision["duration"] = part.replace("DURATION:", "").strip()
        
        # Update current activity
        self.current_activity = decision["action"]
        
        return decision
    
    def converse(self, other_person: str, their_message: str, context: Dict[str, Any] = None) -> str:
        """
        Generate conversation response
        """
        
        from reverie_integration.llm_adapter import reverie_llm_adapter
        
        context_text = ""
        if context:
            context_text = f"Context: {context.get('description', '')}\n"
        
        # Check for conversation history
        conversation_history = ""
        recent_convs = [m for m in self.memories["conversations"] if other_person in str(m)][-3:]
        if recent_convs:
            conversation_history = "Recent conversation history:\n"
            for conv in recent_convs:
                conversation_history += f"- {conv.get('content', '')}\n"
        
        prompt = f"""Character: {self.name}
Personality: {self.personality}
Background: {self.background}
Emotional state: {self.emotional_state}

{context_text}Speaking with: {other_person}
{conversation_history}

{other_person} just said: "{their_message}"

How would {self.name} respond naturally and in character? Keep it conversational and authentic to {self.name}'s personality."""
        
        response = reverie_llm_adapter.ChatGPT_request(prompt)
        
        # Store conversation
        self.memories["conversations"].append({
            "timestamp": datetime.now(),
            "with": other_person,
            "their_message": their_message,
            "my_response": response,
            "content": f"{other_person}: {their_message} | {self.name}: {response}"
        })
        
        return response.strip()
    
    def _format_recent_memories(self, count: int = 5) -> str:
        """Format recent memories for prompt inclusion"""
        
        recent_events = []
        
        # Collect recent events from all memory types
        for memory_type, memories in self.memories.items():
            if isinstance(memories, list):
                for memory in memories[-count:]:
                    if isinstance(memory, dict) and 'timestamp' in memory:
                        recent_events.append((memory['timestamp'], memory_type, memory))
        
        # Sort by timestamp
        recent_events.sort(key=lambda x: x[0], reverse=True)
        
        formatted = ""
        for timestamp, mem_type, memory in recent_events[:count]:
            time_str = timestamp.strftime('%H:%M')
            content = str(memory.get('content', ''))[:100]
            formatted += f"- {time_str} ({mem_type}): {content}\n"
        
        return formatted or "No recent memories available"
    
    def _calculate_relevance(self, query: str, content: str) -> float:
        """Calculate relevance score between query and content"""
        
        query_words = set(query.lower().split())
        content_words = set(content.lower().split())
        
        if not query_words:
            return 0.0
        
        # Simple word overlap scoring
        overlap = len(query_words.intersection(content_words))
        return overlap / len(query_words)
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get summary of current memory state"""
        
        summary = {}
        for mem_type, memories in self.memories.items():
            if isinstance(memories, list):
                summary[mem_type] = len(memories)
            else:
                summary[mem_type] = len(memories) if memories else 0
        
        return {
            "memory_counts": summary,
            "current_activity": self.current_activity,
            "emotional_state": self.emotional_state,
            "last_reflection": self.last_reflection_time.strftime('%H:%M'),
            "plan_items": len(self.current_plan)
        }
    
    def cleanup(self):
        """Clean up resources and save state if needed"""
        
        # Limit memory size to prevent growth
        for mem_type, memories in self.memories.items():
            if isinstance(memories, list) and len(memories) > 50:
                # Keep only most recent 50 items
                self.memories[mem_type] = memories[-50:]
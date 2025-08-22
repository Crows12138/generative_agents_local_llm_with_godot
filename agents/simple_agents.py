"""
Simple AI Agents System
Lightweight AI character system for Godot integration
No complex dependencies, just essential features
"""

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from enum import Enum

# Import AI service (fixed import)
from ai_service.ai_service import local_llm_generate
from ai_service.unified_parser import get_unified_parser
from monitoring.parser_monitor import get_parser_monitor, ParserMonitorContext

# Import memory adapter for reverie integration
try:
    from agents.memory_adapter import ReverieMemoryAdapter
    REVERIE_MEMORY_AVAILABLE = True
except ImportError:
    REVERIE_MEMORY_AVAILABLE = False
    print("[simple_agents] Reverie memory adapter not available")

# Import cognitive wrapper for reverie cognitive modules
try:
    from agents.cognitive_wrapper import CognitiveModuleWrapper, EnhancedAgent
    REVERIE_COGNITIVE_AVAILABLE = True
except ImportError:
    REVERIE_COGNITIVE_AVAILABLE = False
    print("[simple_agents] Reverie cognitive wrapper not available")

class EmotionalState(Enum):
    """Character emotional states"""
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    NEUTRAL = "neutral"
    EXCITED = "excited"
    WORRIED = "worried"
    CONFUSED = "confused"
    RELAXED = "relaxed"

class ActivityType(Enum):
    """Character activity types"""
    IDLE = "idle"
    WALKING = "walking"
    TALKING = "talking"
    WORKING = "working"
    RESTING = "resting"
    EATING = "eating"
    THINKING = "thinking"
    SOCIALIZING = "socializing"

@dataclass
class Memory:
    """Single memory entry"""
    content: str
    timestamp: datetime
    importance: float = 0.5  # 0-1 scale
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "tags": self.tags
        }

@dataclass
class Location:
    """Location in the game world"""
    name: str
    x: float
    y: float
    zone: str = "default"
    
    def distance_to(self, other: 'Location') -> float:
        """Calculate distance to another location"""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5

class SimpleMemorySystem:
    """Simple memory management for AI characters"""
    
    def __init__(self, max_short_term: int = 10, max_long_term: int = 50):
        self.short_term: deque = deque(maxlen=max_short_term)
        self.long_term: List[Memory] = []
        self.max_long_term = max_long_term
        
    def add_memory(self, content: str, importance: float = 0.5, tags: List[str] = None):
        """Add a new memory"""
        memory = Memory(
            content=content,
            timestamp=datetime.now(),
            importance=importance,
            tags=tags or []
        )
        
        # Add to short-term memory
        self.short_term.append(memory)
        
        # If important enough, add to long-term memory
        if importance > 0.7:
            self.long_term.append(memory)
            # Keep only most important memories if over limit
            if len(self.long_term) > self.max_long_term:
                self.long_term.sort(key=lambda m: m.importance, reverse=True)
                self.long_term = self.long_term[:self.max_long_term]
    
    def get_recent_memories(self, count: int = 5) -> List[Memory]:
        """Get recent memories"""
        return list(self.short_term)[-count:]
    
    def get_relevant_memories(self, context: str, count: int = 3) -> List[Memory]:
        """Get memories relevant to context"""
        relevant = []
        
        # Check both short and long term memories
        all_memories = list(self.short_term) + self.long_term
        
        for memory in all_memories:
            # Simple relevance check
            if any(word in memory.content.lower() for word in context.lower().split()):
                relevant.append(memory)
        
        # Sort by importance and recency
        relevant.sort(key=lambda m: (m.importance, m.timestamp), reverse=True)
        return relevant[:count]
    
    def summarize_memories(self) -> str:
        """Get a summary of memories"""
        recent = self.get_recent_memories(3)
        if not recent:
            return "No recent memories."
        
        summary = "Recent memories:\\n"
        for mem in recent:
            summary += f"- {mem.content}\\n"
        return summary

class SimpleAgent:
    """Simple AI agent for game characters"""
    
    def __init__(
        self,
        name: str,
        personality: str,
        background: str = "",
        location: Optional[Location] = None,
        emotional_state: EmotionalState = EmotionalState.NEUTRAL,
        activity: ActivityType = ActivityType.IDLE
    ):
        self.name = name
        self.personality = personality
        self.background = background
        self.location = location or Location("Town Square", 0, 0)
        self.emotional_state = emotional_state
        self.activity = activity
        
        # Memory system - dual system approach
        self.memory = SimpleMemorySystem()
        
        # Enhanced memory with reverie integration
        if REVERIE_MEMORY_AVAILABLE:
            try:
                self.reverie_memory = ReverieMemoryAdapter(name)
                self.use_reverie_memory = True
                print(f"[SimpleAgent] Initialized reverie memory for {name}")
            except Exception as e:
                print(f"[SimpleAgent] Failed to initialize reverie memory for {name}: {e}")
                self.reverie_memory = None
                self.use_reverie_memory = False
        else:
            self.reverie_memory = None
            self.use_reverie_memory = False
        
        # Relationships
        self.relationships: Dict[str, float] = {}  # name -> affinity (-1 to 1)
        
        # Goals and motivations
        self.current_goal: Optional[str] = None
        self.motivations: List[str] = []
        
        # Stats
        self.energy: float = 1.0  # 0-1
        self.hunger: float = 0.0  # 0-1
        self.social_need: float = 0.5  # 0-1
        
        # Conversation state
        self.in_conversation: bool = False
        self.conversation_partner: Optional[str] = None
        
        # Initialize unified parser and monitoring
        self.parser = get_unified_parser()
        self.monitor = get_parser_monitor()
        
        # Initialize cognitive modules if available
        if REVERIE_COGNITIVE_AVAILABLE and self.use_reverie_memory:
            try:
                self.cognitive = CognitiveModuleWrapper(self)
                self.use_cognitive_modules = True
                print(f"[SimpleAgent] Initialized cognitive modules for {name}")
            except Exception as e:
                print(f"[SimpleAgent] Failed to initialize cognitive modules for {name}: {e}")
                self.cognitive = None
                self.use_cognitive_modules = False
        else:
            self.cognitive = None
            self.use_cognitive_modules = False
        
        # Initialize with background
        if background:
            self.memory.add_memory(f"Background: {background}", importance=0.9, tags=["background"])
            # Also add to reverie memory if available
            if self.use_reverie_memory:
                self.reverie_memory.add_event_memory(f"{name}'s background: {background}", importance=8)
        
        # Initialize spatial memory with current location
        if self.use_reverie_memory and location:
            location_data = {
                "world": {
                    location.zone: {
                        location.name: ["environment", "objects"]
                    }
                }
            }
            self.reverie_memory.add_spatial_memory(location_data)
    
    def perceive(self, observation: str, importance: float = 0.5):
        """Enhanced perception with reverie memory integration"""
        # Store in simple memory system
        self.memory.add_memory(observation, importance=importance, tags=["observation"])
        
        # Store in reverie memory if available
        if self.use_reverie_memory:
            # Convert importance scale (0-1 to 1-10)
            reverie_importance = int(importance * 10)
            self.reverie_memory.add_event_memory(
                f"{self.name} perceives: {observation}", 
                importance=reverie_importance
            )
            
            # Retrieve relevant memories for context
            relevant_memories = self.reverie_memory.retrieve_relevant_memories(observation, n=3)
            
            # Create enhanced perception context
            perception_context = {
                "current": observation,
                "memories": [mem.get("description", str(mem)) for mem in relevant_memories],
                "spatial_context": self.reverie_memory.get_spatial_context(self.location.name)
            }
        else:
            perception_context = {"current": observation, "memories": [], "spatial_context": ""}
        
        # Update emotional state based on observation
        if "danger" in observation.lower() or "threat" in observation.lower():
            self.emotional_state = EmotionalState.WORRIED
        elif "friend" in observation.lower() or "happy" in observation.lower():
            self.emotional_state = EmotionalState.HAPPY
            
        return perception_context
    
    def think(self, topic: Optional[str] = None) -> str:
        """Enhanced thinking with reverie memory integration"""
        if topic is None:
            topic = f"current {self.activity.value} activity"
        
        # Get relevant memories for enhanced context
        memory_context = self.memory.summarize_memories()
        if self.use_reverie_memory:
            relevant_memories = self.reverie_memory.retrieve_relevant_memories(topic, n=5)
            if relevant_memories:
                reverie_context = "\\n".join([f"- {mem.get('description', str(mem))}" for mem in relevant_memories])
                memory_context += f"\\n\\nRelevant memories:\\n{reverie_context}"
        
        # Build enhanced context
        context = f"""Character: {self.name}
Personality: {self.personality}
Current emotion: {self.emotional_state.value}
Current activity: {self.activity.value}
Location: {self.location.name}
Energy: {self.energy:.1f}
{memory_context}

What is {self.name} thinking about regarding {topic}?"""
        
        try:
            # Generate thought (fixed function call)
            thought = local_llm_generate(context, model_key=None)
            
            # Store as memory in both systems
            self.memory.add_memory(f"Thought: {thought}", importance=0.3, tags=["thought"])
            
            # Store thought in reverie memory
            if self.use_reverie_memory:
                self.reverie_memory.add_thought_memory(f"{self.name} thinks: {thought}", importance=4)
                self.reverie_memory.update_current_time(datetime.now())
            
            return thought
        except Exception as e:
            # Fallback response if AI service fails
            fallback_thought = f"I'm {self.emotional_state.value} and thinking about {topic}."
            self.memory.add_memory(f"Thought: {fallback_thought}", importance=0.2, tags=["thought", "fallback"])
            
            # Store fallback in reverie memory too
            if self.use_reverie_memory:
                self.reverie_memory.add_thought_memory(f"{self.name} thinks: {fallback_thought}", importance=2)
                
            return fallback_thought
    
    def decide_action(self, available_actions: List[str]) -> Tuple[str, str]:
        """Decide on an action from available options"""
        if not available_actions:
            return "idle", "No actions available"
            
        # Build context
        context = f"""Character: {self.name}
Personality: {self.personality}
Current state: {self.emotional_state.value}, {self.activity.value}
Energy: {self.energy:.1f}, Hunger: {self.hunger:.1f}, Social: {self.social_need:.1f}
Current goal: {self.current_goal or 'none'}
Location: {self.location.name}

Available actions:
{chr(10).join(f'- {action}' for action in available_actions)}

What action would {self.name} choose and why? Format: ACTION: [action] REASON: [reason]"""
        
        try:
            # Use monitoring context for decision parsing
            with ParserMonitorContext(self.monitor, "decision") as monitor_ctx:
                # Generate decision (fixed function call)
                response = local_llm_generate(context, model_key=None)
                
                # Use unified parser for decision parsing
                chosen_action, reason, confidence = self.parser.parse_decision(response, available_actions)
                
                # Record confidence for monitoring
                monitor_ctx.confidence = confidence
                
                # Update activity based on action
                self._update_activity_from_action(chosen_action)
                
                # Store decision in memory with enhanced metadata
                self.memory.add_memory(
                    f"Decided to: {chosen_action} (reason: {reason})",
                    importance=0.6,
                    tags=["decision", "action", self.emotional_state.value]
                )
                
                return chosen_action, reason
            
        except Exception as e:
            # Enhanced fallback with monitoring
            with ParserMonitorContext(self.monitor, "decision_fallback"):
                chosen_action = self._fallback_decision(available_actions)
                reason = f"Simple decision based on current needs (fallback due to: {str(e)[:50]})"
                self._update_activity_from_action(chosen_action)
                
                # Store fallback decision
                self.memory.add_memory(
                    f"Fallback decision: {chosen_action}",
                    importance=0.3,
                    tags=["decision", "fallback", "error"]
                )
                
                return chosen_action, reason
    
    def respond_to(self, speaker: str, message: str) -> str:
        """Generate response to another character"""
        # Update conversation state
        self.in_conversation = True
        self.conversation_partner = speaker
        
        # Get relationship context
        relationship = self.relationships.get(speaker, 0.0)
        rel_desc = "neutral"
        if relationship > 0.5:
            rel_desc = "friendly"
        elif relationship < -0.5:
            rel_desc = "unfriendly"
        
        # Build context
        context = f"""Character: {self.name}
Personality: {self.personality}
Emotional state: {self.emotional_state.value}
Relationship with {speaker}: {rel_desc}
Recent memories: {self.memory.summarize_memories()}

{speaker} says: "{message}"

How does {self.name} respond? (Stay in character, be natural and concise)"""
        
        try:
            # Use monitoring context for response generation
            with ParserMonitorContext(self.monitor, "chat_response") as monitor_ctx:
                # Generate response (fixed function call)
                response = local_llm_generate(context, model_key=None)
                
                # Parse emotion from response using unified parser
                detected_emotion = self.parser.parse_emotion(response)
                if detected_emotion:
                    # Update emotional state based on detected emotion
                    try:
                        new_emotional_state = EmotionalState(detected_emotion)
                        if new_emotional_state != self.emotional_state:
                            self.emotional_state = new_emotional_state
                            self.memory.add_memory(
                                f"Emotional state changed to {detected_emotion} during conversation with {speaker}",
                                importance=0.7,
                                tags=["emotion", "conversation", speaker]
                            )
                    except ValueError:
                        # Detected emotion not in our enum, just log it
                        self.memory.add_memory(
                            f"Expressed emotion: {detected_emotion}",
                            importance=0.4,
                            tags=["emotion", "conversation"]
                        )
                
                # Store interaction in memory with enhanced metadata
                self.memory.add_memory(
                    f"{speaker} said: {message}",
                    importance=0.6,
                    tags=["conversation", speaker, "received"]
                )
                self.memory.add_memory(
                    f"I responded: {response}",
                    importance=0.5,
                    tags=["conversation", "self", "sent", detected_emotion or "no_emotion"]
                )
                
                # Update relationship slightly
                self.update_relationship(speaker, 0.05)
                
                return response
            
        except Exception as e:
            # Enhanced fallback with monitoring
            with ParserMonitorContext(self.monitor, "chat_response_fallback"):
                fallback_response = self._generate_fallback_response(speaker, message)
                
                # Still store the interaction with fallback marker
                self.memory.add_memory(
                    f"{speaker} said: {message}",
                    importance=0.6,
                    tags=["conversation", speaker, "received"]
                )
                self.memory.add_memory(
                    f"I responded (fallback): {fallback_response}",
                    importance=0.3,
                    tags=["conversation", "self", "fallback", "error"]
                )
                
                return fallback_response
    
    def _fallback_decision(self, available_actions: List[str]) -> str:
        """Simple fallback decision logic when AI service fails"""
        # Prioritize based on needs
        if self.energy < 0.3 and any("rest" in action.lower() for action in available_actions):
            return next(action for action in available_actions if "rest" in action.lower())
        elif self.hunger > 0.7 and any("eat" in action.lower() for action in available_actions):
            return next(action for action in available_actions if "eat" in action.lower())
        elif self.social_need > 0.7 and any("talk" in action.lower() or "social" in action.lower() for action in available_actions):
            return next(action for action in available_actions if "talk" in action.lower() or "social" in action.lower())
        else:
            return available_actions[0]  # Default to first available action
    
    def _generate_fallback_response(self, speaker: str, message: str) -> str:
        """Generate simple fallback response when AI service fails"""
        # Simple response based on emotional state and relationship
        relationship = self.relationships.get(speaker, 0.0)
        
        if relationship > 0.5:
            responses = [
                f"Hello {speaker}!",
                "How are you today?",
                "Nice to see you!",
                "What can I do for you?"
            ]
        elif relationship < -0.5:
            responses = [
                "What do you want?",
                "I'm busy right now.",
                "Fine.",
                "Hmm."
            ]
        else:
            responses = [
                "Hello there.",
                "Yes?",
                "I see.",
                "Interesting."
            ]
        
        # Adjust based on emotional state
        if self.emotional_state == EmotionalState.HAPPY:
            return responses[0] if len(responses) > 0 else "Hello!"
        elif self.emotional_state == EmotionalState.SAD:
            return "I'm not feeling great right now."
        elif self.emotional_state == EmotionalState.ANGRY:
            return "I'm not in the mood to talk."
        else:
            return responses[0] if len(responses) > 0 else "Hello."
    
    def update_relationship(self, character_name: str, change: float):
        """Update relationship with another character"""
        if character_name not in self.relationships:
            self.relationships[character_name] = 0.0
        
        self.relationships[character_name] += change
        self.relationships[character_name] = max(-1.0, min(1.0, self.relationships[character_name]))
    
    def update_needs(self, time_delta: float):
        """Update character needs over time"""
        # time_delta in hours
        self.energy -= 0.1 * time_delta
        self.hunger += 0.15 * time_delta
        self.social_need += 0.05 * time_delta
        
        # Clamp values
        self.energy = max(0.0, min(1.0, self.energy))
        self.hunger = max(0.0, min(1.0, self.hunger))
        self.social_need = max(0.0, min(1.0, self.social_need))
        
        # Update emotional state based on needs
        if self.energy < 0.2:
            self.emotional_state = EmotionalState.SAD
        elif self.hunger > 0.8:
            self.emotional_state = EmotionalState.ANGRY
        elif self.social_need > 0.8:
            self.emotional_state = EmotionalState.WORRIED
    
    def move_to(self, location: Location):
        """Move to a new location"""
        old_location = self.location.name
        self.location = location
        self.activity = ActivityType.WALKING
        
        # Add memory of movement
        self.memory.add_memory(
            f"Moved from {old_location} to {location.name}",
            importance=0.3,
            tags=["movement", location.name]
        )
    
    def _update_activity_from_action(self, action: str):
        """Update activity based on chosen action"""
        action_lower = action.lower()
        
        if "walk" in action_lower or "move" in action_lower:
            self.activity = ActivityType.WALKING
        elif "talk" in action_lower or "chat" in action_lower:
            self.activity = ActivityType.TALKING
        elif "work" in action_lower:
            self.activity = ActivityType.WORKING
        elif "rest" in action_lower or "sleep" in action_lower:
            self.activity = ActivityType.RESTING
        elif "eat" in action_lower or "drink" in action_lower:
            self.activity = ActivityType.EATING
        elif "think" in action_lower:
            self.activity = ActivityType.THINKING
        elif "social" in action_lower or "meet" in action_lower:
            self.activity = ActivityType.SOCIALIZING
        else:
            self.activity = ActivityType.IDLE
    
    def save_memories(self):
        """Save reverie memories to disk"""
        if self.use_reverie_memory and self.reverie_memory:
            try:
                self.reverie_memory.save_memories()
                print(f"[SimpleAgent] Saved memories for {self.name}")
            except Exception as e:
                print(f"[SimpleAgent] Error saving memories for {self.name}: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get comprehensive memory summary"""
        summary = {
            "name": self.name,
            "simple_memory": {
                "short_term_count": len(self.memory.short_term),
                "long_term_count": len(self.memory.long_term)
            }
        }
        
        if self.use_reverie_memory and self.reverie_memory:
            try:
                reverie_summary = self.reverie_memory.get_memory_summary()
                summary["reverie_memory"] = reverie_summary
            except Exception as e:
                summary["reverie_memory"] = {"error": str(e)}
        else:
            summary["reverie_memory"] = "Not available"
        
        return summary
    
    def perceive_with_cognition(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced perception using cognitive modules"""
        if self.use_cognitive_modules and self.cognitive:
            try:
                # Use cognitive modules for enhanced perception
                cognitive_perception = self.cognitive.perceive_environment(environment_data)
                
                # Also run normal perception for comparison
                normal_perception = self.perceive(
                    environment_data.get("description", "Observing environment"), 
                    importance=0.6
                )
                
                return {
                    "cognitive_events": cognitive_perception,
                    "normal_perception": normal_perception,
                    "enhanced": True
                }
            except Exception as e:
                print(f"[SimpleAgent] Error in cognitive perception: {e}")
                # Fallback to normal perception
                return {"normal_perception": self.perceive(environment_data.get("description", ""), 0.5), "enhanced": False}
        else:
            # Use normal perception
            return {"normal_perception": self.perceive(environment_data.get("description", ""), 0.5), "enhanced": False}
    
    def think_with_cognition(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced thinking using cognitive modules"""
        if self.use_cognitive_modules and self.cognitive:
            try:
                # Get relevant memories using cognitive retrieval
                recent_thoughts = [mem.content for mem in self.memory.get_recent_memories(3)]
                if recent_thoughts:
                    memories = self.cognitive.retrieve_memories(recent_thoughts, n=5)
                else:
                    memories = {}
                
                # Generate reflection
                reflection = self.cognitive.reflect_on_experience()
                
                # Generate plan if needed
                plan = None
                if not self.current_goal or self.energy < 0.3:
                    plan = self.cognitive.plan_action(environment_data)
                
                return {
                    "memories": memories,
                    "reflection": reflection,
                    "plan": plan,
                    "enhanced": True
                }
            except Exception as e:
                print(f"[SimpleAgent] Error in cognitive thinking: {e}")
                # Fallback to normal thinking
                return {"thought": self.think(), "enhanced": False}
        else:
            # Use normal thinking
            return {"thought": self.think(), "enhanced": False}
    
    def decide_with_cognition(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced decision making using cognitive modules"""
        available_actions = environment_data.get("available_actions", ["idle", "move", "interact", "rest", "eat"])
        
        if self.use_cognitive_modules and self.cognitive:
            try:
                # Use cognitive execution module
                execution_result = self.cognitive.execute_action(environment_data)
                
                # Also get normal decision for comparison
                normal_action, normal_reason = self.decide_action(available_actions)
                
                return {
                    "cognitive_action": execution_result,
                    "normal_action": {"action": normal_action, "reason": normal_reason},
                    "enhanced": True
                }
            except Exception as e:
                print(f"[SimpleAgent] Error in cognitive decision: {e}")
                # Fallback to normal decision
                action, reason = self.decide_action(available_actions)
                return {"action": action, "reason": reason, "enhanced": False}
        else:
            # Use normal decision making
            action, reason = self.decide_action(available_actions)
            return {"action": action, "reason": reason, "enhanced": False}
    
    def converse_with_cognition(self, target_agent_name: str, message: str, environment_data: Dict[str, Any] = None) -> str:
        """Enhanced conversation using cognitive modules"""
        if self.use_cognitive_modules and self.cognitive and environment_data:
            try:
                # Create mock target agent for cognitive conversation
                target_agent = type('TargetAgent', (), {'name': target_agent_name})()
                
                # Use cognitive conversation
                cognitive_response = self.cognitive.converse_with_agent(target_agent, environment_data)
                
                # Store conversation in memory
                self.memory.add_memory(
                    f"Cognitive conversation with {target_agent_name}: {message} -> {cognitive_response}",
                    importance=0.7,
                    tags=["conversation", "cognitive", target_agent_name]
                )
                
                return cognitive_response
            except Exception as e:
                print(f"[SimpleAgent] Error in cognitive conversation: {e}")
                # Fallback to normal conversation
                return self.respond_to(target_agent_name, message)
        else:
            # Use normal conversation
            return self.respond_to(target_agent_name, message)
    
    def get_cognitive_summary(self) -> Dict[str, Any]:
        """Get summary of cognitive capabilities and status"""
        summary = {
            "agent_name": self.name,
            "cognitive_modules_available": self.use_cognitive_modules,
            "reverie_memory_available": self.use_reverie_memory
        }
        
        if self.use_cognitive_modules and self.cognitive:
            summary["cognitive_status"] = "Active"
            summary["reverie_integration"] = self.cognitive.use_reverie
        else:
            summary["cognitive_status"] = "Fallback mode"
            summary["reverie_integration"] = False
        
        return summary
    
    def cleanup(self):
        """Cleanup resources and save memories"""
        self.save_memories()
        if self.in_conversation:
            self.in_conversation = False
            self.conversation_partner = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert agent state to dictionary for Godot"""
        return {
            "name": self.name,
            "personality": self.personality,
            "location": {
                "name": self.location.name,
                "x": self.location.x,
                "y": self.location.y,
                "zone": self.location.zone
            },
            "emotional_state": self.emotional_state.value,
            "activity": self.activity.value,
            "needs": {
                "energy": self.energy,
                "hunger": self.hunger,
                "social": self.social_need
            },
            "current_goal": self.current_goal,
            "in_conversation": self.in_conversation,
            "conversation_partner": self.conversation_partner,
            "relationships": self.relationships
        }
    
    def from_dict(self, data: Dict[str, Any]):
        """Update agent state from dictionary"""
        if "location" in data:
            loc = data["location"]
            self.location = Location(loc["name"], loc["x"], loc["y"], loc.get("zone", "default"))
        
        if "emotional_state" in data:
            self.emotional_state = EmotionalState(data["emotional_state"])
        
        if "activity" in data:
            self.activity = ActivityType(data["activity"])
        
        if "needs" in data:
            needs = data["needs"]
            self.energy = needs.get("energy", self.energy)
            self.hunger = needs.get("hunger", self.hunger)
            self.social_need = needs.get("social", self.social_need)
        
        if "current_goal" in data:
            self.current_goal = data["current_goal"]
        
        if "relationships" in data:
            self.relationships = data["relationships"]

# Preset characters for demo
def create_demo_characters() -> List[SimpleAgent]:
    """Create preset demo characters"""
    characters = []
    
    # Character 1: Friendly shopkeeper
    alice = SimpleAgent(
        name="Alice",
        personality="Friendly, helpful, talkative shopkeeper who loves to gossip",
        background="Runs the general store in town. Knows everyone and everything that happens.",
        location=Location("General Store", 10, 5, "market"),
        emotional_state=EmotionalState.HAPPY
    )
    alice.motivations = ["help customers", "learn town gossip", "maintain store"]
    characters.append(alice)
    
    # Character 2: Grumpy blacksmith
    bob = SimpleAgent(
        name="Bob",
        personality="Grumpy but skilled blacksmith, secretly kind-hearted",
        background="Master blacksmith who takes pride in his work. Doesn't like small talk.",
        location=Location("Blacksmith Shop", -10, 5, "market"),
        emotional_state=EmotionalState.NEUTRAL
    )
    bob.motivations = ["create quality items", "maintain reputation", "avoid crowds"]
    characters.append(bob)
    
    # Character 3: Curious child
    charlie = SimpleAgent(
        name="Charlie",
        personality="Curious, energetic child who loves adventures",
        background="Young explorer always looking for fun and new discoveries.",
        location=Location("Town Square", 0, 0, "center"),
        emotional_state=EmotionalState.EXCITED
    )
    charlie.motivations = ["explore", "play games", "make friends"]
    characters.append(charlie)
    
    # Character 4: Wise elder
    diana = SimpleAgent(
        name="Diana",
        personality="Wise, patient elder with many stories to tell",
        background="Town elder who has seen it all. Gives advice to those who seek it.",
        location=Location("Park Bench", 5, -10, "park"),
        emotional_state=EmotionalState.RELAXED
    )
    diana.motivations = ["share wisdom", "help others", "enjoy peaceful moments"]
    characters.append(diana)
    
    # Character 5: Mysterious traveler
    erik = SimpleAgent(
        name="Erik",
        personality="Mysterious traveler with tales from distant lands",
        background="Recently arrived in town. Has many stories but keeps secrets.",
        location=Location("Inn", 15, 15, "residential"),
        emotional_state=EmotionalState.NEUTRAL
    )
    erik.motivations = ["gather information", "find opportunities", "stay mysterious"]
    characters.append(erik)
    
    return characters

# Test the system
def test_simple_agents():
    """Test the simple agents system"""
    print("=== Simple Agents Test ===\\n")
    
    # Create demo characters
    characters = create_demo_characters()
    alice = characters[0]
    bob = characters[1]
    
    # Test perception
    print("1. Testing perception:")
    alice.perceive("A customer enters the store looking for supplies")
    recent_memories = alice.memory.get_recent_memories(1)
    if recent_memories:
        print(f"Alice's recent memory: {recent_memories[0].content}\\n")
    else:
        print("No recent memories found\\n")
    
    # Test thinking
    print("2. Testing thinking:")
    thought = alice.think("the new customer")
    print(f"Alice thinks: {thought}\\n")
    
    # Test decision making
    print("3. Testing decision:")
    actions = ["Greet the customer", "Continue organizing shelves", "Check inventory", "Take a break"]
    action, reason = alice.decide_action(actions)
    print(f"Alice decides to: {action}")
    print(f"Reason: {reason}\\n")
    
    # Test conversation
    print("4. Testing conversation:")
    bob_message = "Do you have any iron ingots for sale?"
    alice_response = alice.respond_to("Bob", bob_message)
    print(f"Bob: {bob_message}")
    print(f"Alice: {alice_response}\\n")
    
    # Test needs update
    print("5. Testing needs update:")
    print(f"Before: Energy={alice.energy:.2f}, Hunger={alice.hunger:.2f}")
    alice.update_needs(2.0)  # 2 hours pass
    print(f"After 2 hours: Energy={alice.energy:.2f}, Hunger={alice.hunger:.2f}\\n")
    
    # Test serialization
    print("6. Testing serialization:")
    alice_dict = alice.to_dict()
    print(f"Alice as dict: {json.dumps(alice_dict, indent=2)[:200]}...")
    
    print("\\n=== Test completed ===")

if __name__ == "__main__":
    test_simple_agents()
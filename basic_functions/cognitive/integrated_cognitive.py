"""
Integrated Cognitive System
Combines existing decider functionality with enhanced cognitive features from generative_agents_main.
"""

import datetime
import time
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from basic_functions.persona import Persona
from basic_functions.maze import Maze
from basic_functions.memory.enhanced_memory_integrated import IntegratedEnhancedMemory, EnhancedMemoryType, EnhancedMemoryEntry
from basic_functions.decider.decider import ActionIntent
from basic_functions.decider.optimized_decider import OptimizedDecider
from ai_service.ai_service import local_llm_generate

class CognitiveState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    SOCIALIZING = "socializing"
    OBSERVING = "observing"

@dataclass
class PerceivedEvent:
    """Represents a perceived event in the environment."""
    event_type: str
    subject: str
    predicate: str
    object: str
    description: str
    location: Tuple[float, float, float]
    importance: float
    timestamp: datetime.datetime

@dataclass
class ActionPlan:
    """Represents a planned action."""
    action_type: str
    target: Optional[str]
    location: Optional[Tuple[float, float, float]]
    description: str
    priority: float
    estimated_duration: int  # minutes
    prerequisites: List[str]
    social_opportunity: bool = False
    distance: Optional[float] = None

class IntegratedCognitiveSystem:
    """
    Integrated cognitive system that combines existing decider functionality 
    with enhanced cognitive features from generative_agents_main.
    """
    
    def __init__(self, persona: Persona):
        self.persona = persona
        self.memory = IntegratedEnhancedMemory()
        self.current_state = CognitiveState.IDLE
        self.current_plan: Optional[ActionPlan] = None
        self.daily_schedule: List[Dict[str, Any]] = []
        
        # Enhanced parameters
        self.attention_bandwidth = 5  # Number of events to process at once
        self.retention_hours = 2  # How long to remember recent events
        
        # State tracking
        self.last_reflection_time = datetime.datetime.now()
        self.reflection_interval_hours = 4
        self.social_cooldown = {}  # Track social interaction cooldowns
        
        # Backward compatibility - existing decider
        self.optimized_decider = OptimizedDecider(max_action_history=10)
        
        # Action history for repetition avoidance
        self.action_history = []
        self.max_action_history = 10
    
    def perceive(self, maze: Maze, agents: List[Persona]) -> List[PerceivedEvent]:
        """Perceive events in the environment."""
        events = []
        x, y, z = self.persona.location
        
        # Get nearby entities - increase search radius to 10.0 to find other agents
        nearby_entities = maze.spatial.nearby(x, y, z, 10.0)
        
        for entity in nearby_entities:
            if hasattr(entity, 'name'):
                # Determine event type and importance
                if hasattr(entity, 'type'):
                    event_type = entity.type
                else:
                    event_type = "object"
                
                # Calculate importance based on distance and entity type
                distance = ((x - getattr(entity, 'x', x))**2 + 
                           (y - getattr(entity, 'y', y))**2)**0.5
                importance = max(0.1, 1.0 - distance / 10.0)
                
                event = PerceivedEvent(
                    event_type=event_type,
                    subject=self.persona.name,
                    predicate="perceives",
                    object=entity.name,
                    description=f"Noticed {entity.name} nearby",
                    location=(x, y, z),
                    importance=importance,
                    timestamp=datetime.datetime.now()
                )
                events.append(event)
        
        # Check for other personas nearby
        for agent in agents:
            if agent.name != self.persona.name:
                agent_x, agent_y, agent_z = agent.location
                distance = ((x - agent_x)**2 + (y - agent_y)**2)**0.5
                
                if distance <= 8.0:  # Within social range
                    importance = max(0.3, 1.0 - distance / 8.0)
                    
                    event = PerceivedEvent(
                        event_type="social",
                        subject=self.persona.name,
                        predicate="sees",
                        object=agent.name,
                        description=f"Saw {agent.name} nearby",
                        location=(x, y, z),
                        importance=importance,
                        timestamp=datetime.datetime.now()
                    )
                    events.append(event)
        
        # Sort by importance and limit by attention bandwidth
        events.sort(key=lambda e: e.importance, reverse=True)
        return events[:self.attention_bandwidth]
    
    def retrieve_relevant_memories(self, context: str = None) -> List[EnhancedMemoryEntry]:
        """Retrieve memories relevant to current context."""
        return self.memory.retrieve_relevant_memories(
            query=context,
            max_results=10,
            min_importance=0.2
        )
    
    def decide_action(self, maze: Maze, agents: List[Persona], perceived_events: List[PerceivedEvent]) -> ActionPlan:
        """
        Decide next action using integrated approach.
        Combines enhanced cognitive features with existing decider logic.
        """
        
        # Store perceived events in memory
        for event in perceived_events:
            self.memory.add(
                text=event.description,
                embedding=[],  # Will be filled by embedding system
                location=event.location,
                event_type=event.event_type,
                importance=event.importance,
                memory_type=self._get_memory_type_for_event(event.event_type),
                subject=event.subject,
                predicate=event.predicate,
                object=event.object,
                keywords=[event.event_type, event.object]
            )
        
        # Get relevant memories
        relevant_memories = self.retrieve_relevant_memories()
        
        # Check if it's time for reflection
        if self._should_reflect():
            return self._create_reflection_plan()
        
        # Check for social opportunities (enhanced)
        social_plan = self._check_social_opportunities(perceived_events)
        if social_plan:
            return social_plan
        
        # Check current schedule
        schedule_plan = self._check_schedule()
        if schedule_plan:
            return schedule_plan
        
        # Use existing optimized decider with enhanced context
        return self._use_optimized_decider(maze, agents, perceived_events, relevant_memories)
    
    def _get_memory_type_for_event(self, event_type: str) -> EnhancedMemoryType:
        """Map event type to memory type."""
        mapping = {
            "social": EnhancedMemoryType.SOCIAL,
            "action": EnhancedMemoryType.ACTION,
            "sight": EnhancedMemoryType.SIGHT,
            "interaction": EnhancedMemoryType.INTERACTION,
            "reflection": EnhancedMemoryType.REFLECTION,
            "chat": EnhancedMemoryType.CHAT,
            "thought": EnhancedMemoryType.THOUGHT,
            "planning": EnhancedMemoryType.PLANNING,
        }
        return mapping.get(event_type, EnhancedMemoryType.EVENT)
    
    def _should_reflect(self) -> bool:
        """Check if it's time for reflection."""
        time_since_reflection = datetime.datetime.now() - self.last_reflection_time
        return time_since_reflection.total_seconds() / 3600 >= self.reflection_interval_hours
    
    def _create_reflection_plan(self) -> ActionPlan:
        """Create a reflection action plan."""
        self.current_state = CognitiveState.REFLECTING
        
        return ActionPlan(
            action_type="reflect",
            target=None,
            location=self.persona.location,
            description="Reflecting on recent experiences",
            priority=0.8,
            estimated_duration=15,
            prerequisites=[]
        )
    
    def _check_social_opportunities(self, perceived_events: List[PerceivedEvent]) -> Optional[ActionPlan]:
        """Check for social interaction opportunities."""
        social_events = [e for e in perceived_events if e.event_type == "social"]
        
        for event in social_events:
            target_name = event.object
            current_time = datetime.datetime.now()
            
            # Check cooldown
            if target_name in self.social_cooldown:
                time_since_last = current_time - self.social_cooldown[target_name]
                if time_since_last.total_seconds() < 300:  # 5 minutes cooldown
                    continue
            
            # Check if target is close enough for interaction
            if event.importance > 0.5:
                self.current_state = CognitiveState.SOCIALIZING
                self.social_cooldown[target_name] = current_time
                
                # Calculate distance
                x, y, z = self.persona.location
                distance = ((x - event.location[0])**2 + (y - event.location[1])**2)**0.5
                
                return ActionPlan(
                    action_type="talk_to",
                    target=target_name,
                    location=self.persona.location,
                    description=f"Starting conversation with {target_name}",
                    priority=0.9,
                    estimated_duration=10,
                    prerequisites=[],
                    social_opportunity=True,
                    distance=distance
                )
        
        return None
    
    def _check_schedule(self) -> Optional[ActionPlan]:
        """Check current schedule for planned activities."""
        if not self.daily_schedule:
            return None
        
        current_time = datetime.datetime.now()
        current_hour = current_time.hour
        current_minute = current_time.minute
        
        for task in self.daily_schedule:
            start_time = datetime.datetime.strptime(task["start"], "%H:%M").time()
            end_time = datetime.datetime.strptime(task["end"], "%H:%M").time()
            current_time_obj = current_time.time()
            
            if start_time <= current_time_obj <= end_time:
                return ActionPlan(
                    action_type="work_on_task",
                    target=task["task"],
                    location=self.persona.location,
                    description=f"Working on: {task['task']}",
                    priority=0.7,
                    estimated_duration=60,
                    prerequisites=[]
                )
        
        return None
    
    def _use_optimized_decider(self, maze: Maze, agents: List[Persona], 
                              perceived_events: List[PerceivedEvent], 
                              relevant_memories: List[EnhancedMemoryEntry]) -> ActionPlan:
        """Use existing optimized decider with enhanced context."""
        
        # Set simulation context for the decider
        self.optimized_decider.set_simulation_context(maze, agents)
        
        # Build enhanced context
        context_parts = []
        
        # Add perceived events
        if perceived_events:
            event_descriptions = [f"- {e.description}" for e in perceived_events[:3]]
            context_parts.append(f"Recent observations:\n" + "\n".join(event_descriptions))
        
        # Add relevant memories
        if relevant_memories:
            memory_descriptions = [f"- {m.text}" for m in relevant_memories[:3]]
            context_parts.append(f"Relevant memories:\n" + "\n".join(memory_descriptions))
        
        # Add personality and goals
        context_parts.append(f"Personality: {self.persona.personality_description}")
        context_parts.append(f"Goals: {', '.join(self.persona.long_term_goals)}")
        
        # Use optimized decider
        try:
            # Convert memories to format expected by decider (MemoryEntry objects)
            from basic_functions.memory.memory import MemoryEntry
            memories_for_decider = []
            
            # Convert enhanced memories to MemoryEntry format
            for mem in relevant_memories:
                memory_entry = MemoryEntry(
                    timestamp=time.time(),
                    text=mem.text,
                    embedding=mem.embedding,
                    location=mem.location,
                    event_type=mem.event_type,
                    importance=mem.importance,
                    memory_type=mem.memory_type
                )
                memories_for_decider.append(memory_entry)
            
            # Also add base memory entries for backward compatibility
            if hasattr(self.persona, 'memory') and self.persona.memory:
                memories_for_decider.extend(self.persona.memory.entries[-5:])  # Last 5 memories
            
            # Get action from optimized decider
            action_data = self.optimized_decider.decide(
                persona_name=self.persona.name,
                location=self.persona.location,
                personality=self.persona.personality_description,
                surroundings="",  # Will be filled by decider
                memories=memories_for_decider,
                goals=", ".join(self.persona.long_term_goals),
                current_task="",
                previous_action=None
            )
            
            # Convert to ActionPlan
            return ActionPlan(
                action_type=action_data.get("action", "wander"),
                target=action_data.get("target"),
                location=self.persona.location,
                description=f"Performing {action_data.get('action', 'wander')}",
                priority=0.5,
                estimated_duration=5,
                prerequisites=[],
                social_opportunity=action_data.get("social_opportunity", False),
                distance=action_data.get("distance")
            )
            
        except Exception as e:
            print(f"Error in optimized decider: {e}")
            # Fallback to simple action
            return ActionPlan(
                action_type="wander",
                target=None,
                location=self.persona.location,
                description="Wandering around",
                priority=0.3,
                estimated_duration=5,
                prerequisites=[]
            )
    
    def execute_action(self, plan: ActionPlan, maze: Maze) -> bool:
        """Execute the planned action."""
        self.current_plan = plan
        self.current_state = CognitiveState.EXECUTING
        
        # Store the action in memory
        self.memory.add(
            text=plan.description,
            embedding=[],
            location=plan.location,
            event_type=plan.action_type,
            importance=plan.priority,
            memory_type=EnhancedMemoryType.ACTION,
            subject=self.persona.name,
            predicate="performs",
            object=plan.action_type,
            keywords=[plan.action_type]
        )
        
        # Add to action history for repetition avoidance
        self.action_history.append(plan.action_type)
        if len(self.action_history) > self.max_action_history:
            self.action_history.pop(0)
        
        return True
    
    def reflect(self) -> str:
        """Perform reflection on recent experiences."""
        self.current_state = CognitiveState.REFLECTING
        
        # Get recent memories
        recent_memories = self.memory.get_recent_memories(hours=6)
        
        if not recent_memories:
            return "No recent experiences to reflect on."
        
        # Generate reflection using LLM
        memory_summaries = [mem.text for mem in recent_memories[:5]]
        
        prompt = f"""
You are {self.persona.name}. Reflect on your recent experiences:

Recent experiences:
{chr(10).join(memory_summaries)}

Personality: {self.persona.personality_description}
Goals: {', '.join(self.persona.long_term_goals)}

Generate a brief reflection on what you've learned or how you feel about these experiences.
"""
        
        try:
            reflection = local_llm_generate(prompt)
            
            # Store reflection in memory
            self.memory.add(
                text=reflection,
                embedding=[],
                location=self.persona.location,
                event_type="reflection",
                importance=0.6,
                memory_type=EnhancedMemoryType.REFLECTION,
                subject=self.persona.name,
                predicate="reflects",
                object="recent_experiences",
                keywords=["reflection", "learning"]
            )
            
            self.last_reflection_time = datetime.datetime.now()
            return reflection
            
        except Exception as e:
            return f"Reflected on recent experiences and learned from them."
    
    def get_cognitive_summary(self) -> Dict[str, Any]:
        """Get a summary of the cognitive system's current state."""
        memory_summary = self.memory.get_memory_summary(hours=24)
        
        return {
            "current_state": self.current_state.value,
            "current_plan": self.current_plan.description if self.current_plan else None,
            "attention_bandwidth": self.attention_bandwidth,
            "memory_summary": memory_summary,
            "social_cooldowns": len(self.social_cooldown),
            "time_since_reflection": (datetime.datetime.now() - self.last_reflection_time).total_seconds() / 3600,
            "action_history": self.action_history[-5:],  # Last 5 actions
            "enhanced_memory_count": len(self.memory.entries)
        }
    
    # Backward compatibility methods
    def decide(self, maze: Maze, agents: List[Persona]) -> ActionIntent:
        """Backward compatibility method that returns ActionIntent."""
        perceived_events = self.perceive(maze, agents)
        action_plan = self.decide_action(maze, agents, perceived_events)
        
        return ActionIntent(action_plan.action_type, action_plan.target) 
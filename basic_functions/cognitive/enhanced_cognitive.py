"""
Enhanced Cognitive System inspired by Generative Agents
Provides better perception, planning, and execution capabilities.
"""

import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from basic_functions.memory.enhanced_memory import EnhancedMemory, MemoryType, MemoryNode
from basic_functions.persona import Persona
from basic_functions.maze import Maze
from ai_service.ai_service import local_llm_generate

class CognitiveState(Enum):
    IDLE = "idle"
    PLANNING = "planning"
    EXECUTING = "executing"
    REFLECTING = "reflecting"
    SOCIALIZING = "socializing"

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

class EnhancedCognitiveSystem:
    """Enhanced cognitive system with better perception, planning, and execution."""
    
    def __init__(self, persona: Persona):
        self.persona = persona
        self.memory = EnhancedMemory()
        self.current_state = CognitiveState.IDLE
        self.current_plan: Optional[ActionPlan] = None
        self.daily_schedule: List[Dict[str, Any]] = []
        self.attention_bandwidth = 5  # Number of events to process at once
        self.retention_hours = 2  # How long to remember recent events
        
        # State tracking
        self.last_reflection_time = datetime.datetime.now()
        self.reflection_interval_hours = 4
        self.social_cooldown = {}  # Track social interaction cooldowns
    
    def perceive(self, maze: Maze, agents: List[Persona]) -> List[PerceivedEvent]:
        """Perceive events in the environment."""
        events = []
        x, y, z = self.persona.location
        
        # Get nearby entities
        nearby_entities = maze.spatial.nearby(x, y, 5.0, 0.0)
        
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
    
    def retrieve_relevant_memories(self, context: str = None) -> List[MemoryNode]:
        """Retrieve memories relevant to current context."""
        return self.memory.retrieve_relevant_memories(
            query=context,
            max_results=10,
            min_importance=0.2
        )
    
    def plan_next_action(self, maze: Maze, perceived_events: List[PerceivedEvent]) -> ActionPlan:
        """Plan the next action based on current state and perceptions."""
        
        # Store perceived events in memory
        for event in perceived_events:
            self.memory.add_memory(
                memory_type=MemoryType.EVENT,
                subject=event.subject,
                predicate=event.predicate,
                object=event.object,
                description=event.description,
                importance=event.importance,
                keywords=[event.event_type, event.object],
                location=event.location
            )
        
        # Get relevant memories
        relevant_memories = self.retrieve_relevant_memories()
        
        # Check if it's time for reflection
        if self._should_reflect():
            return self._create_reflection_plan()
        
        # Check for social opportunities
        social_plan = self._check_social_opportunities(perceived_events)
        if social_plan:
            return social_plan
        
        # Check current schedule
        schedule_plan = self._check_schedule()
        if schedule_plan:
            return schedule_plan
        
        # Generate action based on context
        return self._generate_contextual_action(perceived_events, relevant_memories)
    
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
                
                return ActionPlan(
                    action_type="talk_to",
                    target=target_name,
                    location=self.persona.location,
                    description=f"Starting conversation with {target_name}",
                    priority=0.9,
                    estimated_duration=10,
                    prerequisites=[]
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
    
    def _generate_contextual_action(self, 
                                  perceived_events: List[PerceivedEvent],
                                  relevant_memories: List[MemoryNode]) -> ActionPlan:
        """Generate a contextual action based on current situation."""
        
        # Build context for LLM
        context_parts = []
        
        # Add perceived events
        if perceived_events:
            event_descriptions = [f"- {e.description}" for e in perceived_events[:3]]
            context_parts.append(f"Recent observations:\n" + "\n".join(event_descriptions))
        
        # Add relevant memories
        if relevant_memories:
            memory_descriptions = [f"- {m.description}" for m in relevant_memories[:3]]
            context_parts.append(f"Relevant memories:\n" + "\n".join(memory_descriptions))
        
        # Add personality and goals
        context_parts.append(f"Personality: {self.persona.personality_description}")
        context_parts.append(f"Goals: {', '.join(self.persona.long_term_goals)}")
        
        context = "\n\n".join(context_parts)
        
        # Generate action using LLM
        prompt = f"""
You are {self.persona.name}. Based on the following context, decide what action to take next.

Context:
{context}

Available actions: move_towards, move_away, search, observe, rest, wander, think

Respond with JSON: {{"action": "action_name", "target": "target_name", "description": "action_description", "priority": 0.5}}
"""
        
        try:
            response = local_llm_generate(prompt)
            # Parse response (simplified)
            if "move_towards" in response.lower():
                action_type = "move_towards"
                target = "nearby_object"
            elif "search" in response.lower():
                action_type = "search"
                target = "water_sources"
            elif "observe" in response.lower():
                action_type = "observe"
                target = "environment"
            else:
                action_type = "wander"
                target = None
            
            return ActionPlan(
                action_type=action_type,
                target=target,
                location=self.persona.location,
                description=f"Performing {action_type}",
                priority=0.5,
                estimated_duration=5,
                prerequisites=[]
            )
            
        except Exception as e:
            # Fallback action
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
        self.memory.add_memory(
            memory_type=MemoryType.EVENT,
            subject=self.persona.name,
            predicate="performs",
            object=plan.action_type,
            description=plan.description,
            importance=plan.priority,
            keywords=[plan.action_type],
            location=plan.location
        )
        
        # Execute the action (this would integrate with your existing executor)
        # For now, just return success
        return True
    
    def reflect(self) -> str:
        """Perform reflection on recent experiences."""
        self.current_state = CognitiveState.REFLECTING
        
        # Get recent memories
        recent_memories = self.memory.get_recent_memories(hours=6)
        
        if not recent_memories:
            return "No recent experiences to reflect on."
        
        # Generate reflection using LLM
        memory_summaries = [m.description for m in recent_memories[:5]]
        
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
            self.memory.add_memory(
                memory_type=MemoryType.REFLECTION,
                subject=self.persona.name,
                predicate="reflects",
                object="recent_experiences",
                description=reflection,
                importance=0.6,
                keywords=["reflection", "learning"],
                location=self.persona.location
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
            "time_since_reflection": (datetime.datetime.now() - self.last_reflection_time).total_seconds() / 3600
        } 
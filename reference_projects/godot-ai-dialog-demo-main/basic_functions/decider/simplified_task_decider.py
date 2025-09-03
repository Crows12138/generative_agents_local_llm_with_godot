"""
Simplified Task-Based Decision System

This module implements a more robust task-based approach with:
1. Simple string format instead of complex JSON
2. Predefined task sequences as fallback
3. Task interruption for urgent events
4. Maximum 3 actions per task for flexibility
"""

import time
from typing import Any, List, Optional, Dict, Tuple
from collections import deque
from dataclasses import dataclass
from enum import Enum
from basic_functions.memory.memory import MemoryEntry
from ai_service.ai_service import local_llm_generate
from performance_monitor import perf_monitor

class InterruptReason(Enum):
    """Reasons for interrupting current task."""
    PLAYER_INTERACTION = "player_interaction"
    SOCIAL_OPPORTUNITY = "social_opportunity"  
    URGENT_EVENT = "urgent_event"
    DANGER = "danger"
    TASK_FAILED = "task_failed"

# Predefined task sequences (3 actions each for flexibility)
TASK_SEQUENCES = {
    # Daily activities
    "explore": ["move_towards|tree", "observe|surroundings", "search|resources"],
    "gather": ["search|materials", "pickup|item", "observe|area"],
    "rest": ["move_towards|shelter", "rest|", "think|day"],
    "socialize": ["move_towards|person", "observe|person", "talk_to|person"],
    
    # Work tasks
    "build": ["move_towards|location", "work|construction", "observe|progress"],
    "maintain": ["observe|equipment", "work|maintenance", "rest|"],
    "organize": ["observe|inventory", "work|organizing", "think|efficiency"],
    
    # Survival tasks
    "hunt": ["search|prey", "move_towards|target", "observe|result"],
    "forage": ["search|food", "pickup|edible", "eat|food"],
    "patrol": ["wander|", "observe|surroundings", "think|security"],
    
    # Default/Emergency
    "wander": ["wander|", "observe|surroundings", "rest|"],
    "think": ["think|situation", "observe|surroundings", "wait|"],
}

@dataclass
class SimpleTask:
    """Simplified task representation."""
    name: str
    actions: List[Tuple[str, str]]  # (action, target) pairs
    current_index: int = 0
    interrupted: bool = False
    interrupt_reason: Optional[InterruptReason] = None
    
    def get_current_action(self) -> Optional[Dict[str, Any]]:
        """Get current action as dict."""
        if self.current_index < len(self.actions):
            action, target = self.actions[self.current_index]
            return {
                "action": action,
                "target": target if target else None,
                "task_name": self.name,
                "action_number": self.current_index + 1,
                "total_actions": len(self.actions)
            }
        return None
    
    def advance(self) -> bool:
        """Move to next action."""
        self.current_index += 1
        return self.current_index < len(self.actions)
    
    def is_complete(self) -> bool:
        """Check if task is complete."""
        return self.current_index >= len(self.actions)
    
    def mark_interrupted(self, reason: InterruptReason):
        """Mark task as interrupted."""
        self.interrupted = True
        self.interrupt_reason = reason


class SimplifiedTaskDecider:
    """
    Simplified task-based decision maker with robust parsing and interruption handling.
    """
    
    def __init__(self):
        self.current_task: Optional[SimpleTask] = None
        self.task_history = deque(maxlen=10)
        self.interrupted_tasks = deque(maxlen=5)  # Store interrupted tasks
        
        # Context
        self.maze = None
        self.all_agents = []
        
        # Statistics
        self.tasks_generated = 0
        self.actions_executed = 0
        self.ai_calls_made = 0
        self.tasks_interrupted = 0
        self.parse_failures = 0
    
    def set_simulation_context(self, maze, agents):
        """Set simulation context."""
        self.maze = maze
        self.all_agents = agents
    
    def get_next_action(
        self,
        persona_name: str,
        location: Tuple[float, float, float],
        personality: str,
        surroundings: str,
        memories: List[MemoryEntry],
        goals: str,
        current_schedule: str,
        previous_action: Any = None,
        force_new_task: bool = False
    ) -> Dict[str, Any]:
        """
        Get next action, handling interruptions and task generation.
        """
        
        # Check for interruption events
        interrupt_reason = self._check_interruption(persona_name, location, memories)
        if interrupt_reason:
            if self.current_task and not self.current_task.is_complete():
                print(f"[TASK] {persona_name}: Task interrupted by {interrupt_reason.value}")
                self.current_task.mark_interrupted(interrupt_reason)
                self.interrupted_tasks.append(self.current_task)
                self.tasks_interrupted += 1
            self.current_task = None
            force_new_task = True
        
        # Generate new task if needed
        if force_new_task or self.current_task is None or self.current_task.is_complete():
            # If task completed successfully, add to history
            if self.current_task and self.current_task.is_complete() and not self.current_task.interrupted:
                self.task_history.append(self.current_task)
            
            # Generate new task
            self._generate_new_task(
                persona_name, location, personality, surroundings,
                memories, goals, current_schedule, interrupt_reason
            )
            self.tasks_generated += 1
        
        # Get next action from current task
        if self.current_task:
            action = self.current_task.get_current_action()
            if action:
                self.actions_executed += 1
                print(f"[TASK] {persona_name}: {action['action']} {action.get('target', '')} "
                      f"({action['action_number']}/{action['total_actions']}) - Task: {action['task_name']}")
                
                # Advance to next action
                if not self.current_task.advance():
                    print(f"[TASK] {persona_name}: Completed task '{self.current_task.name}'")
                
                return action
        
        # Emergency fallback
        print(f"[TASK WARNING] {persona_name}: No action available, using emergency fallback")
        return {"action": "wait", "target": None, "emergency": True}
    
    def _generate_new_task(
        self,
        persona_name: str,
        location: Tuple[float, float, float],
        personality: str,
        surroundings: str,
        memories: List[MemoryEntry],
        goals: str,
        schedule: str,
        interrupt_reason: Optional[InterruptReason]
    ):
        """Generate new task using simplified format."""
        
        # Handle interruption-specific tasks
        if interrupt_reason:
            if interrupt_reason == InterruptReason.SOCIAL_OPPORTUNITY:
                # Find the person to talk to
                target = self._find_nearby_person(persona_name, location)
                if target:
                    self.current_task = SimpleTask(
                        name="social_interaction",
                        actions=[
                            ("move_towards", target),
                            ("observe", target),
                            ("talk_to", target)
                        ]
                    )
                    return
            elif interrupt_reason == InterruptReason.DANGER:
                self.current_task = SimpleTask(
                    name="escape_danger",
                    actions=[
                        ("move_away", "danger"),
                        ("observe", "surroundings"),
                        ("think", "safety")
                    ]
                )
                return
        
        # Build simple context
        context = self._build_simple_context(persona_name, personality, surroundings, memories, goals, schedule)
        
        # Simple prompt for task type
        prompt = f"""You are {persona_name}, {context['personality']}.
Location: {context['surroundings']}
Goal: {context['goal']}
Recent: {context['recent']}

Choose ONE task type from: explore, gather, rest, build, maintain, forage, patrol, think
Respond with just the task type word, nothing else.

Task type:"""
        
        # Get task type from AI
        perf_monitor.log_ai_call(f"[TASK_TYPE] {persona_name}", start=True)
        ai_start = time.time()
        self.ai_calls_made += 1
        
        try:
            response = local_llm_generate(prompt)
            ai_elapsed = time.time() - ai_start
            perf_monitor.log_ai_call("", start=False)
            perf_monitor.track_ai_call(ai_elapsed)
            
            # Parse task type (very simple)
            task_type = response.strip().lower().split()[0] if response else "wander"
            
            # Validate task type
            if task_type not in TASK_SEQUENCES:
                print(f"[TASK] Unknown task type '{task_type}', using 'wander'")
                task_type = "wander"
            
            # Get predefined sequence
            sequence = TASK_SEQUENCES[task_type]
            actions = []
            for action_str in sequence:
                parts = action_str.split("|")
                action = parts[0]
                target = parts[1] if len(parts) > 1 and parts[1] else None
                
                # Replace generic targets with specific ones from context
                if target == "person" and surroundings:
                    target = self._find_nearby_person(persona_name, location) or "someone"
                elif target == "tree" and "tree" in surroundings.lower():
                    target = "tree"
                elif target == "location":
                    target = "nearby area"
                
                actions.append((action, target))
            
            self.current_task = SimpleTask(name=task_type, actions=actions)
            print(f"[TASK] {persona_name}: Starting task '{task_type}' (AI: {ai_elapsed:.2f}s)")
            
        except Exception as e:
            print(f"[TASK ERROR] Failed to generate task: {e}")
            self.parse_failures += 1
            
            # Use fallback task
            self.current_task = SimpleTask(
                name="fallback_wander",
                actions=TASK_SEQUENCES["wander"]
            )
    
    def _build_simple_context(
        self, persona_name, personality, surroundings, 
        memories, goals, schedule
    ) -> Dict[str, str]:
        """Build minimal context for task generation."""
        
        # Very concise context
        personality_brief = personality.split(".")[0] if personality else "explorer"
        
        recent = "nothing"
        if memories and len(memories) > 0:
            recent = memories[-1].text[:30]
        
        goal = goals.split(";")[0] if goals else "survive"
        
        return {
            "personality": personality_brief[:50],
            "surroundings": surroundings[:50] if surroundings else "wilderness",
            "goal": goal[:50],
            "recent": recent
        }
    
    def _check_interruption(
        self, persona_name: str, 
        location: Tuple[float, float, float],
        memories: List[MemoryEntry]
    ) -> Optional[InterruptReason]:
        """
        Check if current task should be interrupted.
        Returns interruption reason if task should be interrupted.
        """
        
        if not self.current_task or self.current_task.is_complete():
            return None
        
        # Don't interrupt if almost done (last action)
        if self.current_task.current_index >= len(self.current_task.actions) - 1:
            return None
        
        # Check for player interaction (highest priority)
        # This would be triggered by Godot events
        
        # Check for nearby social opportunities
        if self.maze and self.all_agents:
            nearby_person = self._find_nearby_person(persona_name, location)
            if nearby_person:
                # Check if we recently talked
                recent_social = False
                for memory in memories[-3:]:
                    if "talk" in memory.text.lower() and nearby_person in memory.text:
                        recent_social = True
                        break
                
                if not recent_social and self.current_task.name not in ["socialize", "social_interaction"]:
                    return InterruptReason.SOCIAL_OPPORTUNITY
        
        # Check for danger (would be triggered by environment)
        # Check for urgent events (would be triggered by game events)
        
        return None
    
    def _find_nearby_person(self, persona_name: str, location: Tuple[float, float, float]) -> Optional[str]:
        """Find nearest person within social range."""
        if not self.all_agents:
            return None
        
        x, y, z = location
        nearest = None
        min_distance = float('inf')
        
        for agent in self.all_agents:
            if agent.name == persona_name:
                continue
            
            agent_x, agent_y, agent_z = agent.location
            distance = ((x - agent_x) ** 2 + (y - agent_y) ** 2) ** 0.5
            
            if distance <= 5.0 and distance < min_distance:
                min_distance = distance
                nearest = agent.name
        
        return nearest
    
    def handle_action_failure(self, action: Dict[str, Any], error: str):
        """
        Handle when an action fails to execute.
        """
        if self.current_task:
            print(f"[TASK] Action failed: {action['action']} - {error}")
            
            # Skip to next action if possible
            if self.current_task.advance():
                print(f"[TASK] Skipping to next action in task")
            else:
                # Task failed, mark as interrupted
                self.current_task.mark_interrupted(InterruptReason.TASK_FAILED)
                self.current_task = None
                print(f"[TASK] Task failed, will generate new task")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        
        efficiency = 0.0
        if self.ai_calls_made > 0:
            efficiency = self.actions_executed / self.ai_calls_made
        
        return {
            "tasks_generated": self.tasks_generated,
            "actions_executed": self.actions_executed,
            "ai_calls_made": self.ai_calls_made,
            "actions_per_ai_call": efficiency,
            "tasks_interrupted": self.tasks_interrupted,
            "parse_failures": self.parse_failures,
            "current_task": self.current_task.name if self.current_task else None
        }
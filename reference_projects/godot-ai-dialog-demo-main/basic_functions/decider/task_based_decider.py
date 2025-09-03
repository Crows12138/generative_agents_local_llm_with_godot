"""
Task-Based Decision System for AI Agents

This module implements a task-based approach where a single AI call generates
multiple related actions, drastically reducing the number of AI calls needed.
Instead of calling AI for every single action, we generate a complete task
with 3-5 related actions that are executed sequentially.
"""

import json
import re
import time
from typing import Any, List, Optional, Dict, Tuple
from collections import deque
from dataclasses import dataclass
from basic_functions.memory.memory import MemoryEntry
from ai_service.ai_service import local_llm_generate
from performance_monitor import perf_monitor

# Common task templates with their typical action sequences
TASK_TEMPLATES = {
    "explore": ["move_towards", "observe", "search"],
    "gather_resources": ["search", "move_towards", "pickup", "observe"],
    "social_interaction": ["move_towards", "observe", "talk_to"],
    "rest_and_reflect": ["move_towards", "rest", "think"],
    "work": ["move_towards", "work", "observe"],
    "maintenance": ["observe", "work", "organize"],
}

AVAILABLE_ACTIONS = [
    "move_towards", "move_away", "throw_towards", "pickup", "drop",
    "talk_to", "create", "eat", "sleep", "read", "write", 
    "search", "observe", "think", "rest", "exercise",
    "work", "play", "wander", "wait"
]

@dataclass
class Task:
    """Represents a task with multiple actions to execute."""
    name: str
    actions: List[Dict[str, Any]]
    priority: float
    estimated_duration: int  # minutes
    current_action_index: int = 0
    
    def get_current_action(self) -> Optional[Dict[str, Any]]:
        """Get the current action to execute."""
        if self.current_action_index < len(self.actions):
            return self.actions[self.current_action_index]
        return None
    
    def advance(self) -> bool:
        """Move to the next action. Returns True if there are more actions."""
        self.current_action_index += 1
        return self.current_action_index < len(self.actions)
    
    def is_complete(self) -> bool:
        """Check if all actions in the task are complete."""
        return self.current_action_index >= len(self.actions)
    
    def progress_percentage(self) -> float:
        """Get task completion percentage."""
        if len(self.actions) == 0:
            return 100.0
        return (self.current_action_index / len(self.actions)) * 100.0


class TaskBasedDecider:
    """
    Task-based decision maker that generates multiple actions from a single AI call.
    This reduces AI calls by ~80% compared to per-action decision making.
    """
    
    def __init__(self, actions_per_task: int = 4):
        self.current_task: Optional[Task] = None
        self.task_history = deque(maxlen=10)
        self.actions_per_task = actions_per_task
        self.interruption_threshold = 0.7  # Importance threshold for interrupting tasks
        
        # Context tracking
        self.maze = None
        self.all_agents = []
        
        # Performance tracking
        self.tasks_generated = 0
        self.actions_executed = 0
        self.ai_calls_made = 0
        
        # Memory-based reflection (from optimized_decider)
        self.memory_since_last_reflection = 0
        self.importance_since_reflection = 0.0
        self.time_since_reflection = time.time()
        self.reflection_memory_threshold = 8  # Higher threshold for task-based
        self.reflection_importance_threshold = 20.0
        self.reflection_time_threshold = 900  # 15 minutes
    
    def set_simulation_context(self, maze, agents):
        """Set the maze and agents for context-aware decision making."""
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
        current_schedule_task: str,
        previous_action: Any = None,
        force_new_task: bool = False
    ) -> Dict[str, Any]:
        """
        Get the next action to execute. If no task is active or force_new_task is True,
        generate a new task with multiple actions. Otherwise, return the next action
        from the current task.
        """
        
        # Check for interruption events (high-priority situations)
        interruption = self._check_for_interruption(persona_name, location, memories)
        if interruption:
            print(f"[TASK] {persona_name}: Interrupting current task for {interruption['reason']}")
            self.current_task = None  # Clear current task
            force_new_task = True
        
        # Generate new task if needed
        if force_new_task or self.current_task is None or self.current_task.is_complete():
            self._generate_new_task(
                persona_name, location, personality, surroundings, 
                memories, goals, current_schedule_task, previous_action
            )
            self.tasks_generated += 1
        
        # Get next action from current task
        if self.current_task:
            action = self.current_task.get_current_action()
            if action:
                self.actions_executed += 1
                
                # Add task context to action
                action['task_name'] = self.current_task.name
                action['task_progress'] = self.current_task.progress_percentage()
                action['is_task_based'] = True
                
                print(f"[TASK] {persona_name}: Executing {action['action']} "
                      f"({self.current_task.current_action_index + 1}/{len(self.current_task.actions)}) "
                      f"from task '{self.current_task.name}'")
                
                # Advance to next action for next call
                self.current_task.advance()
                
                return action
        
        # Fallback (should not happen)
        print(f"[TASK ERROR] {persona_name}: No action available, generating emergency task")
        return {"action": "wait", "target": None, "emergency": True}
    
    def _generate_new_task(
        self,
        persona_name: str,
        location: Tuple[float, float, float],
        personality: str,
        surroundings: str,
        memories: List[MemoryEntry],
        goals: str,
        current_schedule_task: str,
        previous_action: Any
    ):
        """Generate a new task with multiple actions using a single AI call."""
        
        # Build context
        context = self._build_task_context(
            persona_name, personality, surroundings, memories, 
            goals, current_schedule_task, previous_action
        )
        
        # Generate task using AI
        prompt = f"""You are {persona_name}, {context['personality_brief']}
Location: {context['location']}
Surroundings: {context['surroundings']}
Goals: {context['goals']}
Schedule: {context['schedule']}
Recent: {context['recent_memory']}

Create a coherent task with {self.actions_per_task} related actions.
Available actions: {', '.join(AVAILABLE_ACTIONS)}

CRITICAL: Output ONLY valid JSON, no other text.
Format:
{{
  "task_name": "brief_task_description",
  "actions": [
    {{"action": "action1", "target": "target1"}},
    {{"action": "action2", "target": "target2"}},
    {{"action": "action3", "target": "target3"}},
    {{"action": "action4", "target": "target4"}}
  ]
}}

Example:
{{
  "task_name": "explore_forest_area",
  "actions": [
    {{"action": "move_towards", "target": "tree"}},
    {{"action": "observe", "target": "surroundings"}},
    {{"action": "search", "target": "edible_plants"}},
    {{"action": "think", "target": "findings"}}
  ]
}}

Generate task:"""
        
        # Monitor AI call
        perf_monitor.log_ai_call(f"[TASK_GEN] {persona_name}", start=True)
        ai_start = time.time()
        self.ai_calls_made += 1
        
        try:
            raw_output = local_llm_generate(prompt)
            ai_elapsed = time.time() - ai_start
            perf_monitor.log_ai_call("", start=False)
            perf_monitor.track_ai_call(ai_elapsed)
            
            # Parse task
            task_data = self._parse_task_response(raw_output)
            
            # Create Task object
            self.current_task = Task(
                name=task_data['task_name'],
                actions=task_data['actions'],
                priority=0.5,
                estimated_duration=self.actions_per_task * 2  # Rough estimate
            )
            
            # Add to history
            self.task_history.append(self.current_task)
            
            print(f"[TASK] {persona_name}: New task '{self.current_task.name}' "
                  f"with {len(self.current_task.actions)} actions (AI took {ai_elapsed:.2f}s)")
            
        except Exception as e:
            print(f"[TASK ERROR] Failed to generate task: {e}")
            # Create emergency fallback task
            self.current_task = Task(
                name="emergency_routine",
                actions=[
                    {"action": "observe", "target": "surroundings"},
                    {"action": "think", "target": "situation"},
                    {"action": "wander", "target": None},
                    {"action": "rest", "target": None}
                ],
                priority=0.3,
                estimated_duration=8
            )
    
    def _build_task_context(
        self, persona_name, personality, surroundings, 
        memories, goals, schedule, previous_action
    ) -> Dict[str, str]:
        """Build concise context for task generation."""
        
        # Personality summary (first 80 chars)
        personality_brief = personality[:80] + "..." if len(personality) > 80 else personality
        
        # Recent memory summary
        recent_memory = "No recent memories"
        if memories:
            recent_texts = [m.text[:50] for m in memories[-2:]]  # Last 2 memories, truncated
            recent_memory = "; ".join(recent_texts)
        
        # Previous task summary
        prev_task = "None"
        if self.task_history:
            prev_task = self.task_history[-1].name
        
        return {
            "personality_brief": personality_brief,
            "location": f"wilderness area",
            "surroundings": surroundings[:100] if surroundings else "Open area",
            "goals": goals[:150] if goals else "Survive and explore",
            "schedule": schedule[:100] if schedule else "General activities",
            "recent_memory": recent_memory,
            "previous_task": prev_task
        }
    
    def _parse_task_response(self, raw_output: str) -> Dict[str, Any]:
        """Parse AI response to extract task data."""
        
        # Clean up response
        raw_output = raw_output.strip()
        for token in ['<|im_start|>', '<|im_end|>', '<|start|>', '<|end|>']:
            raw_output = raw_output.replace(token, '')
        
        # Try to extract JSON
        if '{' in raw_output and '}' in raw_output:
            # Find the first complete JSON object
            start = raw_output.index('{')
            
            # Count braces to find matching closing brace
            brace_count = 0
            end = start
            for i in range(start, len(raw_output)):
                if raw_output[i] == '{':
                    brace_count += 1
                elif raw_output[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            json_str = raw_output[start:end]
            
            try:
                data = json.loads(json_str)
                
                # Validate structure
                if 'task_name' not in data or 'actions' not in data:
                    raise ValueError("Missing required fields")
                
                # Validate actions
                valid_actions = []
                for action in data['actions']:
                    if isinstance(action, dict) and 'action' in action:
                        # Ensure action is valid
                        if action['action'] in AVAILABLE_ACTIONS:
                            valid_actions.append(action)
                
                if not valid_actions:
                    raise ValueError("No valid actions found")
                
                # Ensure we have the right number of actions
                while len(valid_actions) < self.actions_per_task:
                    valid_actions.append({"action": "observe", "target": "surroundings"})
                
                data['actions'] = valid_actions[:self.actions_per_task]
                return data
                
            except json.JSONDecodeError as e:
                print(f"[TASK PARSE ERROR] JSON decode failed: {e}")
                print(f"[TASK PARSE ERROR] Attempted to parse: {json_str[:200]}")
        
        raise ValueError(f"Could not parse task from response: {raw_output[:200]}")
    
    def _check_for_interruption(
        self, persona_name: str, location: Tuple[float, float, float], 
        memories: List[MemoryEntry]
    ) -> Optional[Dict[str, str]]:
        """
        Check if current task should be interrupted for high-priority events.
        Returns interruption info if task should be interrupted.
        """
        
        if not self.current_task:
            return None
        
        # Don't interrupt if task is almost done
        if self.current_task.progress_percentage() > 75:
            return None
        
        # Check for nearby characters (social opportunity)
        if self.maze and self.all_agents:
            x, y, z = location
            for agent in self.all_agents:
                if agent.name == persona_name:
                    continue
                
                agent_x, agent_y, agent_z = agent.location
                distance = ((x - agent_x) ** 2 + (y - agent_y) ** 2) ** 0.5
                
                if distance <= 3.0:  # Very close
                    # Check if we recently talked to them
                    recent_talk = False
                    for memory in memories[-5:]:
                        if "talk" in memory.text.lower() and agent.name in memory.text:
                            recent_talk = True
                            break
                    
                    if not recent_talk:
                        return {
                            "reason": f"social opportunity with {agent.name}",
                            "type": "social",
                            "target": agent.name
                        }
        
        # Could add more interruption conditions here (danger, important items, etc.)
        
        return None
    
    def should_think_after_task(self) -> bool:
        """
        Determine if character should reflect after completing a task.
        Less frequent than per-action reflection since tasks are coherent units.
        """
        
        # Always reflect after completing a task if enough time has passed
        if (time.time() - self.time_since_reflection) > self.reflection_time_threshold:
            self.time_since_reflection = time.time()
            return True
        
        # Reflect after every 2-3 tasks
        completed_tasks = len([t for t in self.task_history if t.is_complete()])
        if completed_tasks % 3 == 0 and completed_tasks > 0:
            self.time_since_reflection = time.time()
            return True
        
        return False
    
    def generate_reflection(self, persona_name: str, completed_task: Task) -> str:
        """Generate reflection on completed task."""
        
        # Build action summary
        action_summary = ", ".join([
            f"{a['action']} {a.get('target', '')}".strip() 
            for a in completed_task.actions[:3]  # First 3 actions
        ])
        
        prompt = f"""You are {persona_name}. You just completed the task: {completed_task.name}
Actions taken: {action_summary}

Briefly reflect on what you accomplished and learned. Keep it under 40 words.

Response:"""
        
        try:
            perf_monitor.log_ai_call(f"[TASK_REFLECT] {persona_name}", start=True)
            ai_start = time.time()
            reflection = local_llm_generate(prompt)
            ai_elapsed = time.time() - ai_start
            perf_monitor.log_ai_call("", start=False)
            perf_monitor.track_ai_call(ai_elapsed)
            return reflection.strip()
        except Exception as e:
            print(f"[TASK REFLECT ERROR] Failed: {e}")
            return f"Completed task: {completed_task.name}"
    
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
            "current_task": self.current_task.name if self.current_task else None,
            "task_progress": self.current_task.progress_percentage() if self.current_task else 0
        }
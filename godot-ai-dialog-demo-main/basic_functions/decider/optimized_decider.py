import json
import re
import time
from typing import Any, List, Optional, Dict
from collections import deque
from basic_functions.memory.memory import MemoryEntry
from ai_service.ai_service import local_llm_generate
from performance_monitor import perf_monitor

AVAILABLE_ACTIONS = [
    "move_towards <target>",
    "move_away <target>",
    "throw_towards <target1> <target2>",
    "pickup <target>",
    "drop <target>",
    "talk_to <target>",
    "create <target>",
    "eat <target>",
    "sleep <duration>",
    "read <target>",
    "write <target>",
    "search <target>",
    "observe <target>",
    "think <topic>",
    "rest",
    "exercise",
    "work <task>",
    "play <activity>",
    "wander",
    "wait",
]

class OptimizedDecider:
    def __init__(self, max_action_history: int = 5):
        self.action_history = deque(maxlen=max_action_history)
        self.repetition_penalty = {}  # Track repetition penalties
        self.enhanced_reflection = None
        self.maze = None  # Will be set by simulation
        self.all_agents = []  # Will be set by simulation
        
        # Memory-based reflection tracking
        self.memory_since_last_reflection = 0
        self.importance_since_reflection = 0.0
        self.time_since_reflection = time.time()
        self.reflection_memory_threshold = 5  # Reflect after 5 memories
        self.reflection_importance_threshold = 15.0  # Or after importance accumulates to 15
        self.reflection_time_threshold = 600  # Or after 10 minutes
    
    def set_simulation_context(self, maze, agents):
        """Set the maze and agents for context-aware decision making."""
        self.maze = maze
        self.all_agents = agents
    
    def decide(
        self,
        persona_name: str,
        location: Any,
        personality: str,
        surroundings: str,
        memories: List[MemoryEntry],
        goals: str,
        current_task: str,
        previous_action: Any = None,
    ) -> Dict[str, Any]:
        """
        Optimized decision making with concise prompts and repetition avoidance.
        """
        # Build concise context
        context = self._build_concise_context(
            persona_name, personality, surroundings, memories, goals, current_task, previous_action
        )
        
        # Check for social opportunities
        social_action = self._check_social_opportunities(persona_name, location, memories)
        if social_action:
            return social_action
        
        # Generate action
        action_data = self._generate_action(context)
        
        # Apply repetition avoidance
        action_data = self._apply_repetition_avoidance(action_data)
        
        # Add to history
        self._add_to_history(action_data)
        
        return action_data
    
    def _build_concise_context(self, persona_name, personality, surroundings, memories, goals, current_task, previous_action):
        """Build a concise context for decision making."""
        
        # Extract key personality traits (first 100 chars)
        personality_summary = personality[:100] + "..." if len(personality) > 100 else personality
        
        # Get most recent memory (if any)
        recent_memory = ""
        if memories:
            recent_memory = memories[-1].text[:80] + "..." if len(memories[-1].text) > 80 else memories[-1].text
        
        # Previous action
        prev_action = "None"
        if previous_action:
            if hasattr(previous_action, 'action_type'):
                prev_action = previous_action.action_type
                if hasattr(previous_action, 'target') and previous_action.target:
                    prev_action += f" {previous_action.target}"
        
        # Repetition warning
        repetition_warning = ""
        if len(self.action_history) >= 2:
            last_actions = list(self.action_history)[-2:]
            if len(set(last_actions)) == 1:  # Same action repeated
                repetition_warning = " AVOID_REPETITION"
        
        return {
            "name": persona_name,
            "personality": personality_summary,
            "location": str(persona_name),  # Use persona name as location identifier
            "surroundings": surroundings[:100] + "..." if len(surroundings) > 100 else surroundings,
            "goals": goals,
            "current_task": current_task,
            "recent_memory": recent_memory,
            "previous_action": prev_action,
            "repetition_warning": repetition_warning
        }
    
    def _generate_action(self, context):
        """Generate action using concise prompt."""
        
        prompt = f"""You are {context['name']} at {context['location']}.
Personality: {context['personality']}
Surroundings: {context['surroundings']}
Goals: {context['goals']}
Current task: {context['current_task']}
Recent memory: {context['recent_memory']}
Last action: {context['previous_action']}{context['repetition_warning']}

Available actions: {', '.join(AVAILABLE_ACTIONS)}

IMPORTANT: Respond with ONLY a single line of valid JSON, nothing else.
Output format: {{"action": "action_name", "target": "target_name"}}
Example: {{"action": "move_towards", "target": "tree"}}
Now output your decision as JSON:"""

        # Monitor AI call
        perf_monitor.log_ai_call(prompt, start=True)
        ai_start = time.time()
        raw_output = local_llm_generate(prompt)
        ai_elapsed = time.time() - ai_start
        perf_monitor.log_ai_call("", start=False)
        perf_monitor.track_ai_call(ai_elapsed)
        
        # Parse response
        try:
            # Clean up response
            raw_output = raw_output.strip()
            for token in ['<|im_start|>', '<|im_end|>', '<|start|>', '<|end|>']:
                raw_output = raw_output.replace(token, '')
            
            # Try to extract first JSON object (4B model often generates multiple)
            if '{' in raw_output and '}' in raw_output:
                # Split by newlines and try to parse the first line
                lines = raw_output.split('\n')
                for line in lines:
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        try:
                            data = json.loads(line)
                            break
                        except:
                            continue
                else:
                    # Fallback to original extraction
                    start = raw_output.index('{')
                    end = raw_output.index('}') + 1  # Use first closing brace, not last
                    data = json.loads(raw_output[start:end])
            else:
                raise ValueError("No JSON found")
            
            action = data.get("action")
            target = data.get("target")
            
            if not action:
                raise ValueError("AI did not provide an action")
            
            # Validate action
            valid_actions = [act.split()[0] for act in AVAILABLE_ACTIONS]
            if action not in valid_actions:
                print(f"[DECIDER ERROR] Invalid action '{action}' from AI")
                raise ValueError(f"AI returned invalid action: {action}")
            
            return {"action": action, "target": target, "raw_response": raw_output}
            
        except Exception as e:
            print(f"[DECIDER ERROR] Failed to parse AI response: {e}")
            print(f"[DECIDER ERROR] Raw output was: {raw_output}")
            raise RuntimeError(f"Decision generation failed: {e}")
    
    def _apply_repetition_avoidance(self, action_data):
        """Let AI handle repetition avoidance naturally through context."""
        # AI should handle repetition avoidance based on context
        # No hardcoded alternatives
        return action_data
    
    def _get_alternatives(self, current_action):
        """Deprecated - AI should handle action selection."""
        # AI should handle all action selection
        return []
    
    def _add_to_history(self, action_data):
        """Add action to history."""
        action_key = f"{action_data['action']}_{action_data['target']}"
        self.action_history.append(action_key)
    
    def should_think_after_action(self, action_data):
        """Intelligently determine if character should reflect based on memory accumulation."""
        # Track memory accumulation
        self.memory_since_last_reflection += 1
        
        # Estimate importance based on action type (avoid AI call)
        action_importance_map = {
            "talk_to": 6.0,
            "create": 5.0,
            "work": 4.0,
            "observe": 3.0,
            "search": 3.0,
            "eat": 2.0,
            "move_towards": 1.0,
            "move_away": 1.0,
            "wander": 0.5,
            "wait": 0.5,
            "think": 7.0  # Thinking itself is important
        }
        
        action_importance = action_importance_map.get(action_data['action'], 2.0)
        self.importance_since_reflection += action_importance
        
        # Check reflection conditions
        should_reflect = False
        reason = ""
        
        # Condition 1: Memory count threshold
        if self.memory_since_last_reflection >= self.reflection_memory_threshold:
            should_reflect = True
            reason = f"memory accumulation ({self.memory_since_last_reflection} memories)"
        
        # Condition 2: Importance accumulation threshold
        elif self.importance_since_reflection >= self.reflection_importance_threshold:
            should_reflect = True
            reason = f"importance accumulation ({self.importance_since_reflection:.1f} total)"
        
        # Condition 3: Time threshold
        elif (time.time() - self.time_since_reflection) > self.reflection_time_threshold:
            should_reflect = True
            reason = "time interval"
        
        # Condition 4: Critical actions always trigger reflection
        elif action_data['action'] in ['talk_to', 'create', 'work']:
            if self.memory_since_last_reflection >= 2:  # At least some context
                should_reflect = True
                reason = f"important action: {action_data['action']}"
        
        if should_reflect:
            print(f"[REFLECTION_TRIGGER] Triggered by {reason}")
            # Reset counters when reflection is triggered
            self.memory_since_last_reflection = 0
            self.importance_since_reflection = 0.0
            self.time_since_reflection = time.time()
        
        return should_reflect
    
    def generate_reflection(self, persona_name, action_data, result_description):
        """Generate a reflection after an action."""
        prompt = f"""You are {persona_name}. You just completed: {action_data['action']} {action_data.get('target', '')}
Result: {result_description}

Briefly reflect on this action and what you learned. Keep it under 50 words.

Response:"""

        try:
            # Monitor AI call
            perf_monitor.log_ai_call(prompt, start=True)
            ai_start = time.time()
            reflection = local_llm_generate(prompt)
            ai_elapsed = time.time() - ai_start
            perf_monitor.log_ai_call("", start=False)
            perf_monitor.track_ai_call(ai_elapsed)
            return reflection.strip()
        except Exception as e:
            print(f"[REFLECTION ERROR] Failed to generate reflection: {e}")
            raise RuntimeError(f"Reflection generation failed: {e}") 

    def _check_social_opportunities(self, persona_name: str, location: Any, memories: List[MemoryEntry]) -> Optional[Dict[str, Any]]:
        """Check if there are social opportunities and return appropriate action."""
        if not self.maze or not self.all_agents:
            return None
        
        x, y, z = location
        
        # Find nearby characters
        nearby_characters = []
        for agent in self.all_agents:
            if agent.name == persona_name:
                continue  # Skip self
            
            agent_x, agent_y, agent_z = agent.location
            distance = ((x - agent_x) ** 2 + (y - agent_y) ** 2) ** 0.5
            
            if distance <= 8.0:  # Within 8 units
                nearby_characters.append({
                    'name': agent.name,
                    'distance': distance,
                    'location': agent.location
                })
        
        if not nearby_characters:
            return None
        
        # Sort by distance
        nearby_characters.sort(key=lambda c: c['distance'])
        closest_character = nearby_characters[0]
        
        # Check recent memories for social interactions
        recent_social_interactions = 0
        for memory in memories[-5:]:  # Check last 5 memories
            if memory.event_type == "talk" or "talk" in memory.text.lower():
                recent_social_interactions += 1
        
        # If close enough to talk (within 5 units) and haven't talked recently
        if closest_character['distance'] <= 5.0 and recent_social_interactions < 2:
            return {
                "action": "talk_to",
                "target": closest_character['name'],
                "social_opportunity": True,
                "distance": closest_character['distance']
            }
        
        # Let AI decide if they want to move towards nearby characters
        # No hardcoded probability - AI makes the decision
        
        return None 
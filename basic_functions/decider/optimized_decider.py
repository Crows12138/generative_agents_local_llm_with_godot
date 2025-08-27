import json
import re
import random
from typing import Any, List, Optional, Dict
from collections import deque
from basic_functions.memory.memory import MemoryEntry
from ai_service.ai_service import local_llm_generate

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

Choose one action. Respond with JSON: {{"action": "action_name", "target": "target_name"}}"""

        raw_output = local_llm_generate(prompt)
        
        # Parse response
        try:
            # Clean up response
            raw_output = raw_output.strip()
            for token in ['<|im_start|>', '<|im_end|>', '<|start|>', '<|end|>']:
                raw_output = raw_output.replace(token, '')
            
            # Extract JSON
            if '{' in raw_output and '}' in raw_output:
                start = raw_output.index('{')
                end = raw_output.rindex('}') + 1
                data = json.loads(raw_output[start:end])
            else:
                raise ValueError("No JSON found")
            
            action = data.get("action", "wander")
            target = data.get("target")
            
            # Validate action
            valid_actions = [act.split()[0] for act in AVAILABLE_ACTIONS]
            if action not in valid_actions:
                action = "wander"
                target = None
            
            return {"action": action, "target": target, "raw_response": raw_output}
            
        except Exception as e:
            return {"action": "wander", "target": None, "raw_response": f"Error: {e}"}
    
    def _apply_repetition_avoidance(self, action_data):
        """Apply repetition avoidance logic."""
        action_key = f"{action_data['action']}_{action_data['target']}"
        
        # Check if this action was recently performed
        if action_key in self.action_history:
            # Get alternative actions
            alternatives = self._get_alternatives(action_data['action'])
            if alternatives:
                action_data['action'] = random.choice(alternatives)
                action_data['target'] = None
                action_data['avoided_repetition'] = True
        
        return action_data
    
    def _get_alternatives(self, current_action):
        """Get alternative actions to avoid repetition."""
        action_groups = {
            "movement": ["move_towards", "move_away", "wander"],
            "interaction": ["talk_to", "observe", "search"],
            "manipulation": ["pickup", "drop", "create", "throw_towards"],
            "activity": ["work", "play", "exercise", "rest"],
            "basic": ["eat", "sleep", "read", "write", "think", "wait"]
        }
        
        # Find which group the current action belongs to
        for group, actions in action_groups.items():
            if current_action in actions:
                # Return other actions from the same group
                return [a for a in actions if a != current_action]
        
        # Default alternatives
        return ["wander", "observe", "think"]
    
    def _add_to_history(self, action_data):
        """Add action to history."""
        action_key = f"{action_data['action']}_{action_data['target']}"
        self.action_history.append(action_key)
    
    def should_think_after_action(self, action_data):
        """Determine if character should think after this action."""
        # Actions that should trigger thinking
        thinking_triggers = [
            "observe", "search", "talk_to", "think", "work", "create"
        ]
        
        # Actions that shouldn't trigger thinking (too frequent)
        non_thinking_actions = [
            "wander", "move_towards", "move_away", "wait"
        ]
        
        if action_data['action'] in thinking_triggers:
            return True
        elif action_data['action'] in non_thinking_actions:
            return False
        else:
            # Random chance for other actions
            return random.random() < 0.3
    
    def generate_reflection(self, persona_name, action_data, result_description):
        """Generate a reflection after an action."""
        prompt = f"""You are {persona_name}. You just completed: {action_data['action']} {action_data.get('target', '')}
Result: {result_description}

Briefly reflect on this action and what you learned. Keep it under 50 words.

Response:"""

        try:
            reflection = local_llm_generate(prompt)
            return reflection.strip()
        except:
            return f"Completed {action_data['action']} and learned from the experience." 

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
        
        # If not close enough but within reasonable range, move towards
        elif closest_character['distance'] <= 8.0 and closest_character['distance'] > 5.0:
            # Random chance to move towards (30% probability)
            if random.random() < 0.3:
                return {
                    "action": "move_towards",
                    "target": closest_character['name'],
                    "social_opportunity": True,
                    "distance": closest_character['distance']
                }
        
        return None 
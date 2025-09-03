import json
import re
import random
import time
from typing import Any, List, Optional
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

actions_list = "\n".join(f"- {act}" for act in AVAILABLE_ACTIONS)


class ActionIntent:
    def __init__(
        self,
        action_type: str,
        target: Any = None,
        description: str = "",
        borrowed_action: str | None = None,
    ):
        self.action_type = action_type
        self.target = target
        self.description = description
        self.borrowed_action = borrowed_action

    def __repr__(self):
        return (
            f"ActionIntent(type={self.action_type}, target={self.target}, "
            f"description={self.description}, borrowed={self.borrowed_action})"
        )


class BottomDecider:
    def __init__(self):
        self.enhanced_reflection = None
        self.action_history = []  # Track recent actions to avoid repetition
        self.last_action_time = {}  # Track when actions were last performed
        
    def set_enhanced_reflection(self, enhanced_reflection):
        """Set reference to enhanced reflection system"""
        self.enhanced_reflection = enhanced_reflection
    
    def _add_action_to_history(self, action_type: str, target: str = None):
        """Add action to history for repetition avoidance"""
        action_key = f"{action_type}_{target}" if target else action_type
        self.action_history.append(action_key)
        self.last_action_time[action_key] = time.time()
        
        # Keep only last 10 actions
        if len(self.action_history) > 10:
            self.action_history.pop(0)
    
    def _should_avoid_repetition(self, action_type: str, target: str = None) -> bool:
        """Check if action should be avoided due to recent repetition"""
        action_key = f"{action_type}_{target}" if target else action_type
        
        # Count recent occurrences
        recent_count = self.action_history.count(action_key)
        
        # Avoid if action was performed 3+ times in last 10 actions
        if recent_count >= 3:
            return True
            
        # Avoid if same action was performed very recently (within last 2 actions)
        if len(self.action_history) >= 2 and self.action_history[-2:] == [action_key, action_key]:
            return True
            
        return False
    
    def _get_alternative_actions(self, current_task: str, surroundings: str) -> List[str]:
        """Get alternative actions based on current task and surroundings"""
        alternatives = []
        
        # Task-based alternatives
        if "exploration" in current_task.lower():
            alternatives.extend(["wander", "move_towards"])
        elif "survival" in current_task.lower():
            alternatives.extend(["pickup", "eat", "create"])
        elif "lunch" in current_task.lower() or "break" in current_task.lower():
            alternatives.extend(["eat", "wait", "talk_to"])
        elif "work" in current_task.lower():
            alternatives.extend(["pickup", "create", "move_towards"])
        
        # Surroundings-based alternatives
        if "empty" in surroundings.lower():
            alternatives.extend(["wander", "move_towards"])
        elif "kitchen" in surroundings.lower():
            alternatives.extend(["pickup", "eat", "create"])
        elif "forest" in surroundings.lower():
            alternatives.extend(["wander", "pickup", "create"])
        
        return list(set(alternatives))  # Remove duplicates
    
    def decide(
        self,
        persona_name: str,
        location: Any,
        self_identification: str,
        surroundings_desc: str,
        similar_memories: List[MemoryEntry],
        high_level_task: str,
        previous_action: ActionIntent | None = None,
        use_enhanced_memory: bool = True,
        daily_schedule: str = "",
        failed_actions: dict = None,
    ) -> ActionIntent:
        # If enhanced memory is enabled, use medium-term memory for recent memories
        if use_enhanced_memory and self.enhanced_reflection:
            # Get behavior context (from medium-term memory) - these are complete daily summaries
            behavior_context = self.enhanced_reflection.get_behavior_context_for_decider()
            
            # Get personality summary (from long-term memory)
            personality_summary = self.enhanced_reflection.get_personality_summary_for_decider()
            
            # Use medium-term memory as recent memories (complete sentences)
            enhanced_memory_section = f"Recent behavior patterns (from medium-term memory):\n{behavior_context if behavior_context and behavior_context != 'No recent behavior patterns available.' else 'No recent behavior patterns available.'}"
        else:
            # Fallback to basic memory fragments for backward compatibility
            memory_snippets = []
            for e in similar_memories:
                # Clean up memory text - replace Chinese punctuation and format consistently
                clean_text = e.text.replace('；', '; ').replace('，', ', ').replace('。', '. ')
                clean_text = clean_text.replace('、', ', ').replace('：', ': ')
                # Ensure proper spacing
                clean_text = ' '.join(clean_text.split())
                memory_snippets.append(clean_text)
            
            enhanced_memory_section = f"Recent memories (from short-term memory):\n" + "\n".join(memory_snippets) if memory_snippets else "Recent memories (from short-term memory):\nNo recent memories available."
            personality_summary = ""
        
        # Build enhanced self identification
        enhanced_self_identification = self_identification
        if personality_summary and personality_summary != "I am still learning about myself.":
            enhanced_self_identification += f"\n\n--- About Myself ---\n{personality_summary}"

        # Build previous action description
        if previous_action:
            if getattr(previous_action, "target", None) is not None:
                last_action_desc = (
                    f"{previous_action.action_type} -> {previous_action.target}"
                )
            else:
                last_action_desc = previous_action.action_type
        else:
            last_action_desc = "None"

        # Build enhanced prompt
        daily_schedule_section = f"\nYour current task: {daily_schedule}" if daily_schedule else ""
        
        # Only add failure history if there are significant failures
        failure_section = ""
        if failed_actions:
            # Only show failures that have been attempted multiple times
            significant_failures = []
            for action_key, failure_info in failed_actions.items():
                if failure_info["count"] >= 3:  # Higher threshold - only show after 3+ failures
                    action_type, target = action_key.split("_", 1)
                    reasons = ", ".join(set(failure_info["reasons"]))
                    significant_failures.append(f"- {action_type} {target} (failed {failure_info['count']} times: {reasons})")
            
            if significant_failures:
                failure_section = f"\nActions that have repeatedly failed (consider alternatives):\n" + "\n".join(significant_failures)
        
        # Clean up surroundings description
        clean_surroundings = surroundings_desc.strip() if surroundings_desc else "empty space"
        if not clean_surroundings or clean_surroundings == '""':
            clean_surroundings = "empty space"
        
        # Add repetition avoidance guidance
        repetition_guidance = ""
        if previous_action:
            prev_action_key = f"{previous_action.action_type}_{previous_action.target}" if getattr(previous_action, "target", None) else previous_action.action_type
            if self.action_history.count(prev_action_key) >= 2:
                repetition_guidance = "\n\nIMPORTANT: You have been repeating the same action. Consider trying something different to show behavioral variety."
        
        prompt = f"""
You are {persona_name} in {location}. You are {enhanced_self_identification}. You perceive "{clean_surroundings}".
Your current high-level task: {high_level_task}.{daily_schedule_section}

{enhanced_memory_section}{failure_section}

Last action: {last_action_desc}.{repetition_guidance}

Decide the next action based on your personality, recent behavior patterns, and current situation. Available actions:
{actions_list}

Consider:
1. Your established behavior patterns and preferences
2. Consistency with your past actions and personality
3. The current situation and high-level task
4. Your interactions with the environment
5. Behavioral variety - avoid repeating the same action too frequently{failure_section and "6. Consider alternatives to actions that have repeatedly failed" or ""}

Respond with single JSON, e.g. {{"action": "move_towards", "target": "Key"}} or {{"action": "wander"}} or {{"action": "throw_towards", "target1": "pillow", "target2": "John"}}.
"""

        print(f"[Enhanced Decider] Prompt:\n{prompt}")

        raw = local_llm_generate(prompt)
        action = "wander"
        target = None

        # Clean up LLM response tokens
        raw = raw.strip()
        # Remove common LLM tokens
        tokens_to_remove = ['<|im_start|>', '<|im_end|>', '<|start|>', '<|end|>']
        for token in tokens_to_remove:
            raw = raw.replace(token, '')
        raw = raw.strip()

        # Attempt JSON parse, with fallback extraction
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # Try to extract JSON from code blocks (```json ... ```)
            try:
                import re
                json_match = re.search(r'```json\s*(\{.*?\})\s*```', raw, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                else:
                    # Try to extract any {...} block
                    brace_matches = re.findall(r'\{[^{}]*\}', raw)
                    if brace_matches:
                        # Try each match until one works
                        for match in brace_matches:
                            try:
                                data = json.loads(match)
                                break
                            except json.JSONDecodeError:
                                continue
                        else:
                            # If no valid JSON found, try to fix common issues
                            fixed_raw = raw.replace("'", '"')  # Replace single quotes
                            fixed_raw = re.sub(r'([{,])\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', fixed_raw)  # Quote keys
                            try:
                                data = json.loads(fixed_raw)
                            except json.JSONDecodeError:
                                # Last resort: create a simple action
                                print(f"[Decider] Failed to parse JSON: {raw}")
                                return ActionIntent(action_type="wander", target=None, description=f"Parse failed: {raw}")
                    else:
                        # No JSON-like content found
                        print(f"[Decider] No JSON content found in: {raw}")
                        return ActionIntent(action_type="wander", target=None, description=f"No JSON: {raw}")
            except Exception as e:
                # parsing failed, default to wander
                print(f"[Decider] JSON extraction failed: {e}, raw: {raw}")
                return ActionIntent(action_type=action, target=target, description=f"Extraction failed: {raw}")

        # Validate action is in allowed list
        action = data.get("action", "wander")
        if action not in [act.split()[0] for act in AVAILABLE_ACTIONS]:
            print(f"[Decider] Invalid action: {action}, defaulting to wander")
            action = "wander"
        
        # Get target(s) based on action type
        if action in ["move_towards", "move_away", "pickup", "drop", "talk_to", "create", "eat"]:
            target = data.get("target")
        elif action == "throw_towards":
            target1 = data.get("target1")
            target2 = data.get("target2")
            target = f"{target1} -> {target2}" if target1 and target2 else None
        
        # Check for repetition and potentially override
        if self._should_avoid_repetition(action, target):
            print(f"[Decider] Avoiding repetition of {action} {target}")
            # Get alternative actions
            alternatives = self._get_alternative_actions(daily_schedule, clean_surroundings)
            if alternatives:
                # Choose a random alternative that's not the current action
                available_alternatives = [alt for alt in alternatives if alt != action]
                if available_alternatives:
                    action = random.choice(available_alternatives)
                    target = None  # Reset target for new action
                    print(f"[Decider] Chose alternative action: {action}")
        
        # Add action to history
        self._add_action_to_history(action, target)
        
        return ActionIntent(action_type=action, target=target, description=raw)

    def decide_simple(
        self,
        persona_name: str,
        location: Any,
        self_identification: str,
        surroundings_desc: str,
        similar_memories: List[MemoryEntry],
        high_level_task: str,
        previous_action: ActionIntent | None = None,
    ) -> ActionIntent:
        """
        Simple version of decide without enhanced memory system (backward compatible)
        """
        return self.decide(
            persona_name=persona_name,
            location=location,
            self_identification=self_identification,
            surroundings_desc=surroundings_desc,
            similar_memories=similar_memories,
            high_level_task=high_level_task,
            previous_action=previous_action,
            use_enhanced_memory=False
        )


_planner = BottomDecider()

def set_enhanced_reflection_for_decider(enhanced_reflection):
    """Set enhanced reflection system for decider"""
    _planner.set_enhanced_reflection(enhanced_reflection)

def low_level_decide(
    persona_name: str,
    location: Any,
    self_identification: str,
    surroundings_desc: str,
    similar_memories: List[MemoryEntry],
    high_level_task: str,
    previous_action: ActionIntent | None = None,
    use_enhanced_memory: bool = True,
) -> ActionIntent:
    """
    A thin wrapper to SimplePlanner.decide, so that you can:
       from basic_functions.decider.decider import low_level_decide
    """
    return _planner.decide(
        persona_name=persona_name,
        location=location,
        self_identification=self_identification,
        surroundings_desc=surroundings_desc,
        similar_memories=similar_memories,
        high_level_task=high_level_task,
        previous_action=previous_action,
        use_enhanced_memory=use_enhanced_memory,
    )

def low_level_decide_simple(
    persona_name: str,
    location: Any,
    self_identification: str,
    surroundings_desc: str,
    similar_memories: List[MemoryEntry],
    high_level_task: str,
    previous_action: ActionIntent | None = None,
) -> ActionIntent:
    """
    Simple version of decision function without enhanced memory system
    """
    return _planner.decide_simple(
        persona_name=persona_name,
        location=location,
        self_identification=self_identification,
        surroundings_desc=surroundings_desc,
        similar_memories=similar_memories,
        high_level_task=high_level_task,
        previous_action=previous_action,
    )

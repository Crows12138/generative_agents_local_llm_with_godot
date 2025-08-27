import json
import random
from typing import List, Dict, Any
from datetime import datetime

# local LLM wrapper
from ai_service.ai_service import local_llm_generate

def generate_daily_schedule(
    agent_name: str,
    personality_description: str,
    high_level_goals: List[str],
    medium_term_memories: str,
    current_date: str,
) -> List[Dict[str, Any]]:
    """
    Generate a daily schedule using personality, high-level goals, and recent behavior patterns.

    Args:
        agent_name: Name of the agent.
        personality_description: Agent's personality description.
        high_level_goals: A list of the agent's high-level goals.
        medium_term_memories: Recent behavior patterns from medium-term memory.
        current_date: Date string, e.g. "2025-06-07".

    Returns:
        A list of dicts, each with keys "start", "end", "task":
        [
            { "start": "08:00", "end": "12:00", "task": "Explore the forest" },
            { "start": "12:00", "end": "13:00", "task": "Prepare lunch" },
            ...
        ]
    """
    goals_text = "\n- ".join(high_level_goals) if high_level_goals else "No specific goals"
    dt = datetime.fromisoformat(current_date)
    weekday = dt.strftime("%A")
    
    # Build memory context
    memory_context = ""
    if medium_term_memories and medium_term_memories.strip() != "No recent behavior patterns available.":
        memory_context = f"\n\nYour recent behavior patterns:\n{medium_term_memories}"
    
    prompt = f"""
You are {agent_name}. {personality_description}
Today is {current_date} ({weekday}).

Your high-level goals:
- {goals_text}{memory_context}

Based on your personality, goals, and recent behavior patterns, plan your day by dividing 24 hours (00:00–24:00) into time blocks. Each block should have a specific task that aligns with your personality and goals.

Consider:
1. Your personality traits and preferences
2. Your high-level goals
3. Your recent behavior patterns
4. The current day of the week

Output a JSON array of objects with keys "start", "end", and "task", for example:
[
  {{ "start": "06:00", "end": "08:00", "task": "Morning exploration and planning" }},
  {{ "start": "08:00", "end": "12:00", "task": "Work towards survival goals" }},
  {{ "start": "12:00", "end": "13:00", "task": "Prepare and eat lunch" }},
  {{ "start": "13:00", "end": "17:00", "task": "Continue main activities" }},
  {{ "start": "17:00", "end": "19:00", "task": "Evening preparation" }},
  {{ "start": "19:00", "end": "22:00", "task": "Rest and reflection" }},
  {{ "start": "22:00", "end": "06:00", "task": "Sleep" }}
]
"""
    raw_output = local_llm_generate(prompt)
    
    # Clean up LLM response tokens
    raw_output = raw_output.strip()
    tokens_to_remove = ['<|im_start|>', '<|im_end|>', '<|start|>', '<|end|>']
    for token in tokens_to_remove:
        raw_output = raw_output.replace(token, '')
    raw_output = raw_output.strip()
    
    try:
        schedule = json.loads(raw_output)
        # Validate format
        if not all(
            isinstance(item, dict)
            and "start" in item
            and "end" in item
            and "task" in item
            for item in schedule
        ):
            raise ValueError("Invalid schedule format")
    except json.JSONDecodeError:
        # Try to extract JSON from code blocks
        try:
            import re
            json_match = re.search(r'```json\s*(\[.*?\])\s*```', raw_output, re.DOTALL)
            if json_match:
                schedule = json.loads(json_match.group(1))
            else:
                # Try to extract array directly
                start = raw_output.index("[")
                end = raw_output.rindex("]") + 1
                schedule = json.loads(raw_output[start:end])
        except Exception:
            # Fallback to default schedule if parsing fails
            print(f"[Plan] Failed to parse schedule for {agent_name}, using fallback")
            schedule = [
                {"start": "06:00", "end": "08:00", "task": "Morning activities"},
                {"start": "08:00", "end": "12:00", "task": "Work towards goals"},
                {"start": "12:00", "end": "13:00", "task": "Lunch break"},
                {"start": "13:00", "end": "17:00", "task": "Afternoon activities"},
                {"start": "17:00", "end": "22:00", "task": "Evening activities"},
                {"start": "22:00", "end": "06:00", "task": "Sleep"}
            ]
    except Exception:
        # Any other error, use fallback
        print(f"[Plan] Error generating schedule for {agent_name}, using fallback")
        schedule = [
            {"start": "06:00", "end": "08:00", "task": "Morning activities"},
            {"start": "08:00", "end": "12:00", "task": "Work towards goals"},
            {"start": "12:00", "end": "13:00", "task": "Lunch break"},
            {"start": "13:00", "end": "17:00", "task": "Afternoon activities"},
            {"start": "17:00", "end": "22:00", "task": "Evening activities"},
            {"start": "22:00", "end": "06:00", "task": "Sleep"}
        ]

    return schedule

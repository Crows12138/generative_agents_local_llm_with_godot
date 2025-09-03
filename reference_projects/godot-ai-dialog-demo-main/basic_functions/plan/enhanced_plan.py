"""
Enhanced planning module for AI agents.

This module provides enhanced daily schedule generation and task management
for AI agents in a wilderness survival simulation environment.
"""

import json
import random
import re
from typing import List, Dict, Any, Optional
from datetime import datetime
from ai_service.ai_service import local_llm_generate

def generate_enhanced_daily_schedule(
    agent_name: str,
    personality_description: str,
    high_level_goals: List[str],
    medium_term_memories: str,
    current_date: str,
) -> List[Dict[str, Any]]:
    """
    Generate a more specific and actionable daily schedule.
    
    Args:
        agent_name: Name of the agent
        personality_description: Description of the agent's personality
        high_level_goals: List of high-level goals for the agent
        medium_term_memories: Recent behavior patterns and memories
        current_date: Current date in ISO format
        
    Returns:
        List of schedule items with start time, end time, and task description
    """
    goals_text = "\n- ".join(high_level_goals) if high_level_goals else "No specific goals"
    try:
        dt = datetime.fromisoformat(current_date)
        weekday = dt.strftime("%A")
    except ValueError:
        weekday = "Unknown"
    
    # Build memory context
    memory_context = ""
    if medium_term_memories and medium_term_memories.strip() != "No recent behavior patterns available.":
        memory_context = f"\n\nYour recent behavior patterns:\n{medium_term_memories}"
    
    prompt = f"""
You are {agent_name}. {personality_description}
Today is {current_date} ({weekday}).

Your high-level goals:
- {goals_text}{memory_context}

Create a specific, actionable daily schedule. Each task should be concrete and executable, not vague descriptions.

Consider your environment: You're in a wilderness survival situation with access to:
- Natural resources (trees, rocks, water sources)
- Basic tools and objects
- Other characters to interact with

Generate specific tasks like:
- "Search for water sources near the stream"
- "Gather firewood from fallen branches"
- "Check the bridge for structural integrity"
- "Explore the forest area for edible plants"
- "Build a shelter using available materials"

CRITICAL: You must respond with ONLY a valid JSON array. No markdown formatting, no explanations, no additional text.

Required format:
[
  {{ "start": "06:00", "end": "08:00", "task": "Search for water sources and check stream quality" }},
  {{ "start": "08:00", "end": "10:00", "task": "Gather firewood and building materials" }},
  {{ "start": "10:00", "end": "12:00", "task": "Build or improve shelter structure" }},
  {{ "start": "12:00", "end": "13:00", "task": "Prepare and eat lunch using gathered resources" }},
  {{ "start": "13:00", "end": "15:00", "task": "Explore forest area for edible plants and herbs" }},
  {{ "start": "15:00", "end": "17:00", "task": "Check and maintain tools and equipment" }},
  {{ "start": "17:00", "end": "19:00", "task": "Prepare evening meal and secure camp" }},
  {{ "start": "19:00", "end": "22:00", "task": "Rest, reflect, and plan for tomorrow" }},
  {{ "start": "22:00", "end": "06:00", "task": "Sleep and rest" }}
]

Start your response with [ and end with ]. No other text.
"""
    
    try:
        raw_output = local_llm_generate(prompt)
    except Exception as e:
        print(f"[Enhanced Plan] Error generating schedule for {agent_name}: {e}")
        raw_output = ""
    
    # Clean up LLM response
    raw_output = raw_output.strip()
    tokens_to_remove = ['<|im_start|>', '<|im_end|>', '<|start|>', '<|end|>']
    for token in tokens_to_remove:
        raw_output = raw_output.replace(token, '')
    raw_output = raw_output.strip()
    
    # Debug: Show raw output
    print(f"[Enhanced Plan] Raw LLM output for {agent_name}: {raw_output[:200]}...")
    
    # Multiple parsing attempts
    schedule: Optional[List[Dict[str, Any]]] = None
    parsing_attempts = [
        # Attempt 1: Direct JSON parsing
        lambda x: json.loads(x),
        # Attempt 2: Extract from code blocks with json
        lambda x: (lambda match: json.loads(match.group(1)) if match else None)(re.search(r'```json\s*(\[.*?\])\s*```', x, re.DOTALL)),
        # Attempt 3: Extract from any code block
        lambda x: (lambda match: json.loads(match.group(1)) if match else None)(re.search(r'```\s*(\[.*?\])\s*```', x, re.DOTALL)),
        # Attempt 4: Find array pattern (more robust)
        lambda x: json.loads(x[x.index("["):x.rindex("]") + 1]) if "[" in x and "]" in x else None,
        # Attempt 5: Extract between first [ and last ] with regex
        lambda x: (lambda match: json.loads(match.group(0)) if match else None)(re.search(r'\[.*?\]', x, re.DOTALL)),
        # Attempt 6: Clean and try direct parsing again
        lambda x: json.loads(re.sub(r'[^\x00-\x7F]+', '', x).strip())
    ]
    
    for i, attempt in enumerate(parsing_attempts):
        try:
            if attempt is None:
                continue
            schedule = attempt(raw_output)
            # Validate format
            if (isinstance(schedule, list) and 
                all(isinstance(item, dict) and "start" in item and "end" in item and "task" in item 
                    for item in schedule)):
                print(f"[Enhanced Plan] Successfully parsed schedule for {agent_name} (attempt {i+1})")
                break
            else:
                schedule = None
        except Exception as e:
            print(f"[Enhanced Plan] Parsing attempt {i+1} failed for {agent_name}: {e}")
            continue
    
    # If all parsing attempts failed, no fallback
    if schedule is None:
        print(f"[Enhanced Plan] All parsing attempts failed for {agent_name}")
        raise RuntimeError(f"Failed to generate or parse schedule for {agent_name} after multiple attempts")
    
    return schedule

def get_current_task_from_schedule(schedule: List[Dict[str, Any]], current_time: str) -> str:
    """
    Get the current task based on the current time.
    
    Args:
        schedule: List of schedule items with start, end, and task
        current_time: Current time in "HH:MM" format
        
    Returns:
        Current task description or "General activities" if no match found
    """
    if not schedule:
        return "General activities"
    
    # Parse current time (format: "HH:MM")
    try:
        current_hour, current_minute = map(int, current_time.split(":"))
        current_minutes = current_hour * 60 + current_minute
    except (ValueError, AttributeError):
        return "General activities"
    
    for task in schedule:
        try:
            start_hour, start_minute = map(int, task["start"].split(":"))
            end_hour, end_minute = map(int, task["end"].split(":"))
            
            start_minutes = start_hour * 60 + start_minute
            end_minutes = end_hour * 60 + end_minute
            
            # Handle overnight tasks (e.g., 22:00-06:00)
            if end_minutes < start_minutes:
                if current_minutes >= start_minutes or current_minutes < end_minutes:
                    return task["task"]
            else:
                if start_minutes <= current_minutes < end_minutes:
                    return task["task"]
        except (ValueError, KeyError, AttributeError):
            continue
    
    return "General activities" 
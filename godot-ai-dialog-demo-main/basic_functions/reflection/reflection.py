# file: basic_functions/reflection/reflection.py
import json
import time
from typing import List, Dict, Any
from datetime import datetime, timedelta
from basic_functions.memory.memory import MemoryEntry, MemoryType
from basic_functions.memory.medium_term_memory import MediumTermMemory, MediumTermMemoryEntry
from basic_functions.memory.long_term_memory import LongTermMemory, LongTermMemoryEntry
from basic_functions.perception.embedding import get_embedding, get_single_embedding

try:  # heavy dependencies may be unavailable during testing
    from ai_service.ai_service import local_llm_generate
except Exception:  # pragma: no cover - provide fallback
    def local_llm_generate(prompt: str) -> str:
        return '{"insights": [], "adjustments": []}'

# Original reflection prompt
REFLECTION_PROMPT = """
You are {agent_name}. Today is {current_date}.
You have done these tasks: {today_tasks}.
You have these recent memories:
{recent_memories}

Answer the following reflection questions:
1. What patterns or insights did you observe in your interactions today?
2. How well did you follow your daily schedule? Any deviations?
3. Based on these, what should you adjust in your future behavior or goals?

Please output a JSON with keys:
- "insights": List[str]
- "adjustments": List[str]
"""

# Daily summary prompt for generating medium-term memory
SIMPLE_DAILY_SUMMARY_PROMPT = """
You are {agent_name}. Today is {current_date}.

Here are your memory fragments from today:
{sight_memories}
{action_memories}
{interaction_memories}

Please summarize your main behaviors and feelings of today in one paragraph.
"""

# Long-term memory generation prompt
LONG_TERM_REFLECTION_PROMPT = """
You are {agent_name}. You have been reflecting on your behavior over the past week.

Here are your daily behavior summaries:
{daily_summaries}

Based on these patterns, please analyze what you have learned about yourself. Output a JSON with:
- "personality_insights": List[Dict] (Each dict has "content": str, "confidence": float 0.0-1.0, "category": "personality")
- "values_discovered": List[Dict] (Each dict has "content": str, "confidence": float 0.0-1.0, "category": "values")
- "preferences_identified": List[Dict] (Each dict has "content": str, "confidence": float 0.0-1.0, "category": "preferences")
- "beliefs_formed": List[Dict] (Each dict has "content": str, "confidence": float 0.0-1.0, "category": "beliefs")
- "goals_refined": List[Dict] (Each dict has "content": str, "confidence": float 0.0-1.0, "category": "goals")

Focus on consistent patterns that reveal core aspects of your identity. Be honest about your confidence level for each insight.
"""

class EnhancedReflection:
    """
    Enhanced reflection system that supports a three-layer memory system.
    """
    def __init__(self):
        self.medium_term_memory = MediumTermMemory()
        self.long_term_memory = LongTermMemory()

    def daily_reflection(
        self,
        agent_name: str,
        current_date: str,
        short_term_memories: List[MemoryEntry],
        save_to_medium_term: bool = True,
    ) -> Dict[str, Any]:
        """
        Daily reflection: analyze short-term memories and generate medium-term memory.
        """
        # Group memories by type
        sight_memories = [m for m in short_term_memories if m.memory_type == MemoryType.SIGHT]
        action_memories = [m for m in short_term_memories if m.memory_type == MemoryType.ACTION]
        interaction_memories = [m for m in short_term_memories if m.memory_type == MemoryType.INTERACTION]
        # Format memory texts
        sight_texts = "\n".join(f"- {m.text}" for m in sight_memories)
        action_texts = "\n".join(f"- {m.text}" for m in action_memories)
        interaction_texts = "\n".join(f"- {m.text}" for m in interaction_memories)
        # Generate prompt
        prompt = SIMPLE_DAILY_SUMMARY_PROMPT.format(
            agent_name=agent_name,
            current_date=current_date,
            sight_memories=sight_texts or "",
            action_memories=action_texts or "",
            interaction_memories=interaction_texts or "",
        )
        print(f"[Daily Reflection] Generating summary for {current_date}")
        behavior_summary = local_llm_generate(prompt).strip()
        if not behavior_summary:
            behavior_summary = "No summary available."
        if save_to_medium_term:
            self.medium_term_memory.add_daily_summary(
                date=current_date,
                behavior_summary=behavior_summary
            )
        return {
            "behavior_summary": behavior_summary
        }

    def weekly_reflection(
        self,
        agent_name: str,
        current_date: str,
        days_back: int = 7,
        save_to_long_term: bool = True,
    ) -> Dict[str, Any]:
        """
        Weekly reflection: analyze medium-term memory and generate long-term memory.
        """
        # Get medium-term memories from the past week
        recent_entries = self.medium_term_memory.get_recent_entries(days_back)
        if not recent_entries:
            return {
                "personality_insights": [],
                "values_discovered": [],
                "preferences_identified": [],
                "beliefs_formed": [],
                "goals_refined": [],
                "message": "No recent medium-term memories to analyze"
            }
        # Format daily summaries
        daily_summaries = []
        for entry in recent_entries:
            summary = f"Date: {entry.date}\n"
            summary += f"Behavior: {entry.behavior_summary}\n"
            summary += "-" * 50
            daily_summaries.append(summary)
        # Generate prompt
        prompt = LONG_TERM_REFLECTION_PROMPT.format(
            agent_name=agent_name,
            daily_summaries="\n\n".join(daily_summaries)
        )
        print(f"[Weekly Reflection] Generating long-term insights for {agent_name}")
        raw_response = local_llm_generate(prompt)
        try:
            reflection_data = json.loads(raw_response)
            # Save to long-term memory
            if save_to_long_term:
                self._save_insights_to_long_term(reflection_data, f"Weekly reflection on {current_date}")
            return {
                "personality_insights": reflection_data.get("personality_insights", []),
                "values_discovered": reflection_data.get("values_discovered", []),
                "preferences_identified": reflection_data.get("preferences_identified", []),
                "beliefs_formed": reflection_data.get("beliefs_formed", []),
                "goals_refined": reflection_data.get("goals_refined", []),
                "raw_response": raw_response
            }
        except json.JSONDecodeError:
            print(f"[Weekly Reflection] JSON decode error: {raw_response}")
            return {
                "personality_insights": [],
                "values_discovered": [],
                "preferences_identified": [],
                "beliefs_formed": [],
                "goals_refined": [],
                "raw_response": raw_response
            }

    def _save_insights_to_long_term(self, reflection_data: Dict[str, Any], source_summary: str) -> None:
        """
        Save reflection results to long-term memory.
        """
        insight_categories = [
            "personality_insights",
            "values_discovered", 
            "preferences_identified",
            "beliefs_formed",
            "goals_refined"
        ]
        for category in insight_categories:
            insights = reflection_data.get(category, [])
            for insight in insights:
                if isinstance(insight, dict) and "content" in insight:
                    content = insight["content"]
                    confidence = insight.get("confidence", 0.5)
                    insight_category = insight.get("category", category.split("_")[0])
                    # Use get_single_embedding to get List[float]
                    embedding = get_single_embedding(content)
                    self.long_term_memory.add_or_update_belief(
                        category=insight_category,
                        content=content,
                        confidence=confidence,
                        source_summary=source_summary,
                        embedding=embedding
                    )

    def get_behavior_context_for_decider(self, days_back: int = 7) -> str:
        """
        Provide behavior context for decider (from medium-term memory).
        """
        return self.medium_term_memory.get_behavior_patterns_for_prompt(days_back)

    def get_personality_summary_for_decider(self) -> str:
        """
        Provide personality summary for decider (from long-term memory).
        """
        return self.long_term_memory.get_personality_summary()

    def cleanup_memories(self) -> Dict[str, int]:
        """
        Clean up expired memories.
        """
        medium_cleaned = self.medium_term_memory.cleanup_expired()
        long_cleaned = self.long_term_memory.remove_low_confidence_entries()
        
        return {
            "medium_term_cleaned": medium_cleaned,
            "long_term_cleaned": long_cleaned
        }

    def save_memories_to_disk(self, agent_name: str, base_path: str = "memories/") -> None:
        """
        Save memories to disk.
        """
        import os
        os.makedirs(base_path, exist_ok=True)
        
        medium_file = os.path.join(base_path, f"{agent_name}_medium_term.json")
        long_file = os.path.join(base_path, f"{agent_name}_long_term.json")
        
        self.medium_term_memory.save_to_file(medium_file)
        self.long_term_memory.save_to_file(long_file)

    def load_memories_from_disk(self, agent_name: str, base_path: str = "memories/") -> None:
        """
        Load memories from disk.
        """
        import os
        
        medium_file = os.path.join(base_path, f"{agent_name}_medium_term.json")
        long_file = os.path.join(base_path, f"{agent_name}_long_term.json")
        
        if os.path.exists(medium_file):
            self.medium_term_memory.load_from_file(medium_file)
        if os.path.exists(long_file):
            self.long_term_memory.load_from_file(long_file)

    def display_all_memories(self) -> str:
        """
        Display all memory layers.
        """
        result = "MEMORY SYSTEM STATUS\n"
        result += "=" * 80 + "\n\n"
        
        result += self.medium_term_memory.display_memories()
        result += "\n\n"
        result += self.long_term_memory.display_memories()
        
        return result

# Backward-compatible original function
def reflect(
    agent_name: str,
    current_date: str,
    today_schedule: List[dict],
    memory: List[MemoryEntry],
    top_k: int = 5,
) -> dict:
    """
    Original reflect function for backward compatibility.
    """
    recent = sorted(memory, key=lambda e: e.timestamp, reverse=True)[:top_k]
    mem_texts = "\n".join(f"- {e.text}" for e in recent)

    prompt = REFLECTION_PROMPT.format(
        agent_name=agent_name,
        current_date=current_date,
        today_tasks=json.dumps(today_schedule, ensure_ascii=False),
        recent_memories=mem_texts,
    )

    raw = local_llm_generate(prompt)

    try:
        data = json.loads(raw)
        insights = data.get("insights", [])
        adjustments = data.get("adjustments", [])
    except Exception:
        insights = []
        adjustments = []

    return {"insights": insights, "adjustments": adjustments}

# Global instance
_enhanced_reflection = EnhancedReflection()

def get_enhanced_reflection() -> EnhancedReflection:
    """Get enhanced reflection system instance."""
    return _enhanced_reflection

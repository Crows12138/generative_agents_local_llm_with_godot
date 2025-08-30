"""
Cognitive LLM Service Integration
Provides unified interface for dual-model system in basic_functions

- 1.7B Model: ALL dialogue/conversation (fast responses)
- 4B Model: Cognitive operations (planning, reflection, decision-making)
"""

import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from ai_service.cognitive_dual_model_service import get_cognitive_service
except ImportError:
    # No fallback - require cognitive service
    raise ImportError("[ERROR] Cognitive dual model service is required. Cannot proceed without AI models.")


class CognitiveLLMService:
    """
    Wrapper for cognitive dual model service
    Provides clean interface for basic_functions modules
    """
    
    def __init__(self):
        self.service = get_cognitive_service()
        if not self.service:
            raise RuntimeError("[ERROR] Failed to initialize cognitive service. AI models are required.")
    
    # ========== DIALOGUE FUNCTIONS (1.7B) ==========
    
    def generate_dialogue(self, 
                         npc_name: str, 
                         personality: str, 
                         message: str, 
                         conversation_history: List[Dict] = None) -> str:
        """
        Generate dialogue response using 1.7B model
        This is for ALL NPC conversations (fast)
        """
        if not self.service:
            raise RuntimeError("Cognitive service not available - AI models required for dialogue")
        
        return self.service.generate_dialogue(
            npc_name, personality, message, conversation_history
        )
    
    def generate_conversation_response(self, prompt: str) -> str:
        """
        Simple conversation generation using 1.7B
        Used by executor for talk_to actions
        """
        if not self.service:
            return "Hello there!"
        
        # Use dialogue model for fast response
        return self.service.generate_dialogue(
            "NPC", 
            "You are a conversational NPC",
            prompt,
            None
        )
    
    # ========== COGNITIVE FUNCTIONS (4B) ==========
    
    def generate_plan(self, agent_name: str, prompt: str) -> str:
        """
        Generate a plan using 4B model
        Used by plan.py for daily scheduling
        """
        if not self.service:
            raise RuntimeError("Cognitive service required for plan generation")
        
        # Extract context from prompt
        current_state = {"agent": agent_name}
        goals = ["Complete daily tasks", "Interact with others"]
        memories = []
        
        # Use cognitive model for deep planning
        plan = self.service.cognitive_planning(
            agent_name,
            current_state,
            goals,
            memories
        )
        
        # Convert plan to expected format if needed
        return plan
    
    def generate_reflection(self, agent_name: str, prompt: str) -> str:
        """
        Generate reflection/insights using 4B model
        Used by reflection.py for daily and weekly reflections
        """
        if not self.service:
            raise RuntimeError("Cognitive service required for reflection generation")
        
        # Extract experiences from prompt
        experiences = ["Completed tasks", "Had conversations"]
        emotions = {"satisfaction": 0.7, "energy": 0.6}
        
        # Use cognitive model for deep reflection
        reflection = self.service.cognitive_reflection(
            agent_name,
            experiences,
            emotions
        )
        
        # Format as expected JSON
        return f'{{"insights": ["{reflection}"], "adjustments": ["Apply learnings"]}}'
    
    def make_decision(self, agent_name: str, prompt: str) -> str:
        """
        Make a complex decision using 4B model
        Used by decider modules
        """
        if not self.service:
            raise RuntimeError("Cognitive service required for decision making")
        
        # Parse decision context from prompt
        situation = "Need to decide next action"
        options = ["move_towards", "talk_to", "wait", "explore"]
        priorities = ["Safety", "Social interaction", "Exploration"]
        
        # Use cognitive model for decision
        choice_idx = self.service.cognitive_decision(
            agent_name,
            situation,
            options,
            priorities
        )
        
        return options[choice_idx]
    
    def generate_cognitive_response(self, prompt: str, use_deep_thinking: bool = False) -> str:
        """
        General purpose cognitive response
        Automatically selects model based on use_deep_thinking flag
        """
        if not self.service:
            raise RuntimeError("Cognitive service required for cognitive response")
        
        if use_deep_thinking:
            # Use 4B for complex cognitive tasks
            return self.service.cognitive_planning(
                "Agent",
                {"prompt": prompt},
                [],
                []
            )
        else:
            # Use 1.7B for simple responses
            return self.service.generate_dialogue(
                "Agent",
                "General purpose agent",
                prompt,
                None
            )


# Global instance
_cognitive_llm_service = None

def get_cognitive_llm_service() -> CognitiveLLMService:
    """Get or create global cognitive LLM service"""
    global _cognitive_llm_service
    if _cognitive_llm_service is None:
        _cognitive_llm_service = CognitiveLLMService()
    return _cognitive_llm_service


# ========== REPLACEMENT FUNCTIONS FOR EXISTING CODE ==========

def local_llm_generate_cognitive(prompt: str, use_deep_thinking: bool = False) -> str:
    """
    Replacement for local_llm_generate that uses cognitive dual model
    
    Args:
        prompt: The prompt to generate from
        use_deep_thinking: If True, uses 4B model; if False, uses 1.7B model
    """
    service = get_cognitive_llm_service()
    
    # Analyze prompt to determine if it needs deep thinking
    if use_deep_thinking or any(keyword in prompt.lower() for keyword in [
        "plan", "schedule", "reflect", "insight", "analyze", "decide", 
        "think", "consider", "evaluate", "reason", "understand"
    ]):
        # Use 4B for cognitive tasks
        return service.generate_cognitive_response(prompt, use_deep_thinking=True)
    else:
        # Use 1.7B for simple responses and dialogue
        return service.generate_cognitive_response(prompt, use_deep_thinking=False)
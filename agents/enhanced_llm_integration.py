"""Enhanced LLM integration for agents with complete output handling"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ai_service.direct_llm_service import direct_llm_service
from agents.llm_output_adapter import get_llm_adapter

class EnhancedLLMIntegration:
    """Enhanced LLM integration that ensures complete outputs for agent tasks"""
    
    def __init__(self):
        self.direct_service = direct_llm_service
        self.adapter = get_llm_adapter()
        
        # Task type mapping for agent functions
        self.agent_task_types = {
            "perceive": "list",
            "plan": "daily_plan", 
            "reflect": "reflection",
            "converse": "conversation",
            "decide": "action",
            "think": "default"
        }
    
    def generate_for_agent(self, prompt: str, context_type: str, agent_name: str = "Agent") -> dict:
        """Generate complete response for agent tasks"""
        
        # Determine expected output type
        expected_type = self.agent_task_types.get(context_type, "default")
        
        # Generate complete output using direct service
        raw_output = self.direct_service.generate_complete(
            prompt=prompt,
            expected_type=expected_type,
            skip_preamble=True
        )
        
        # Use adapter to convert to structured format
        adapted_output = self.adapter.adapt_output(raw_output, context_type, agent_name)
        
        return {
            "raw_output": raw_output,
            "adapted_output": adapted_output,
            "context_type": context_type,
            "enhanced": True,
            "complete": len(raw_output) > 50,  # Basic completeness check
            "agent": agent_name
        }
    
    def generate_perception(self, environment: dict, agent_name: str) -> dict:
        """Generate complete perception output"""
        
        prompt = f"""Character: {agent_name}
Environment: {environment.get('description', 'unknown')}
Events: {', '.join(environment.get('events', []))}
Objects: {', '.join(environment.get('objects', []))}

List the most important things you observe and how they affect your understanding of the situation."""
        
        return self.generate_for_agent(prompt, "perceive", agent_name)
    
    def generate_daily_plan(self, agent_name: str, personality: str, goals: list = None) -> dict:
        """Generate complete daily plan"""
        
        goals_text = f"Goals: {', '.join(goals)}" if goals else ""
        prompt = f"""Character: {agent_name}
Personality: {personality}
{goals_text}

Create a detailed daily schedule from 8am to 8pm with specific activities and times."""
        
        return self.generate_for_agent(prompt, "plan", agent_name)
    
    def generate_conversation(self, agent_name: str, other_person: str, message: str, context: dict) -> dict:
        """Generate complete conversation response"""
        
        prompt = f"""Character: {agent_name}
Speaking with: {other_person}
Their message: "{message}"
Context: {context.get('description', 'casual conversation')}

Respond naturally and appropriately to continue the conversation."""
        
        return self.generate_for_agent(prompt, "converse", agent_name)
    
    def generate_decision(self, agent_name: str, situation: str, available_actions: list) -> dict:
        """Generate complete decision with reasoning"""
        
        actions_text = ', '.join(available_actions) if available_actions else "any appropriate action"
        prompt = f"""Character: {agent_name}
Situation: {situation}
Available actions: {actions_text}

Choose the best action and explain your reasoning briefly."""
        
        return self.generate_for_agent(prompt, "decide", agent_name)
    
    def generate_reflection(self, agent_name: str, recent_events: list, current_mood: str) -> dict:
        """Generate complete reflection"""
        
        events_text = '; '.join(recent_events[:3])  # Last 3 events
        prompt = f"""Character: {agent_name}
Recent events: {events_text}
Current mood: {current_mood}

Reflect on how these experiences have affected you and what you've learned."""
        
        return self.generate_for_agent(prompt, "reflect", agent_name)

# Global instance
enhanced_llm_integration = EnhancedLLMIntegration()
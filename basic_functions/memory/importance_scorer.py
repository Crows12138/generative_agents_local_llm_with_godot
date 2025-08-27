"""
Importance scoring system for memory entries.
This module provides AI-powered importance scoring for memories.
"""

import json
from typing import Dict, Any, Optional
from basic_functions.memory.memory import MemoryType

try:
    from ai_service.ai_service import local_llm_generate
except ImportError:
    def local_llm_generate(prompt: str) -> str:
        return "5"  # Default fallback score

class ImportanceScorer:
    """
    AI-powered importance scoring system for memory entries.
    """
    
    def __init__(self):
        self.cache = {}  # Simple cache for repeated scoring
    
    def score_event_importance(
        self, 
        event_description: str, 
        persona_name: str,
        memory_type: MemoryType,
        context: str = ""
    ) -> float:
        """
        Score the importance of an event for a specific persona.
        
        Args:
            event_description: Description of the event
            persona_name: Name of the persona
            memory_type: Type of memory (sight, action, interaction, reflection)
            context: Additional context about the situation
            
        Returns:
            Importance score from 1.0 to 10.0
        """
        # Check cache first
        cache_key = f"{persona_name}_{event_description}_{memory_type.value}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Create scoring prompt
        prompt = self._create_scoring_prompt(
            event_description, persona_name, memory_type, context
        )
        
        try:
            response = local_llm_generate(prompt)
            score = self._parse_score_response(response)
            
            # Cache the result
            self.cache[cache_key] = score
            
            return score
        except Exception as e:
            print(f"Warning: Failed to score importance for '{event_description}': {e}")
            # Return default score based on memory type
            return self._get_default_score(memory_type)
    
    def _create_scoring_prompt(
        self, 
        event_description: str, 
        persona_name: str,
        memory_type: MemoryType,
        context: str
    ) -> str:
        """Create a prompt for scoring event importance."""
        
        memory_type_descriptions = {
            MemoryType.SIGHT: "something you observed",
            MemoryType.ACTION: "an action you performed", 
            MemoryType.INTERACTION: "an interaction with someone or something",
            MemoryType.REFLECTION: "a thought or reflection you had"
        }
        
        return f"""
You are {persona_name}. Rate the importance of this event for you:

Event: {event_description}
Type: {memory_type_descriptions.get(memory_type, "an event")}
Context: {context if context else "No additional context"}

Consider these factors:
1. Personal relevance - How much does this affect you personally?
2. Emotional impact - How strong are the emotions involved?
3. Long-term consequences - Will this have lasting effects?
4. Uniqueness - Is this unusual or memorable?
5. Social significance - Does this involve important relationships?

Rate from 1.0 to 10.0 where:
- 1.0 = Completely unimportant, easily forgotten
- 5.0 = Moderately important, worth remembering
- 10.0 = Extremely important, life-changing

Respond with only the number (e.g., "7.5"):
"""
    
    def _parse_score_response(self, response: str) -> float:
        """Parse the LLM response to extract the score."""
        try:
            # Clean the response
            cleaned = response.strip()
            
            # Try to extract a number
            import re
            numbers = re.findall(r'\d+\.?\d*', cleaned)
            if numbers:
                score = float(numbers[0])
                # Clamp to valid range
                return max(1.0, min(10.0, score))
            
            # Fallback parsing
            if any(word in cleaned.lower() for word in ['high', 'important', 'significant']):
                return 8.0
            elif any(word in cleaned.lower() for word in ['medium', 'moderate']):
                return 5.0
            elif any(word in cleaned.lower() for word in ['low', 'minor', 'unimportant']):
                return 2.0
            else:
                return 5.0
                
        except (ValueError, TypeError):
            return 5.0
    
    def _get_default_score(self, memory_type: MemoryType) -> float:
        """Get default importance score based on memory type."""
        defaults = {
            MemoryType.SIGHT: 3.0,
            MemoryType.ACTION: 4.0,
            MemoryType.INTERACTION: 6.0,
            MemoryType.REFLECTION: 7.0
        }
        return defaults.get(memory_type, 5.0)
    
    def score_batch_importance(
        self, 
        events: list, 
        persona_name: str,
        context: str = ""
    ) -> Dict[str, float]:
        """
        Score importance for multiple events at once.
        
        Args:
            events: List of dicts with 'description' and 'memory_type' keys
            persona_name: Name of the persona
            context: Additional context
            
        Returns:
            Dict mapping event descriptions to importance scores
        """
        results = {}
        
        for event in events:
            description = event.get('description', '')
            memory_type = event.get('memory_type', MemoryType.SIGHT)
            
            if description:
                score = self.score_event_importance(
                    description, persona_name, memory_type, context
                )
                results[description] = score
        
        return results
    
    def clear_cache(self):
        """Clear the importance score cache."""
        self.cache.clear()

# Global instance
importance_scorer = ImportanceScorer() 
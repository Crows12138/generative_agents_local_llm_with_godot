"""
LLM Output Format Adapter
Specifically solves the mismatch between local LLM (Qwen) output and Reverie expected format
"""

import re
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class AdapterConfig:
    """Adapter configuration"""
    max_retries: int = 2
    fallback_enabled: bool = True
    debug_mode: bool = False
    simplified_mode: bool = True

class QwenOutputAdapter:
    """
    Qwen Model Output Adapter
    Specifically handles Qwen model output format and converts it to Reverie expected format
    """
    
    def __init__(self, config: AdapterConfig = None):
        self.config = config or AdapterConfig()
        self.logger = logging.getLogger(__name__)
        
        # Qwen common output patterns
        self.qwen_patterns = {
            'direct_answer': r'^([^{}\[\]]+?)\.?\s*$',
            'explanation_format': r'(?:based on|according to|considering).+?[,.]\s*(.+)',
            'json_like': r'\{[^{}]*\}',
            'list_format': r'^\d+\.?\s*(.+?)(?:\n|$)',
            'chinese_action': r'(analyze|think|decide|act|converse|respond):?\s*(.+)',
            'action_verb': r'^(I\s+(?:should|will|would|can|need to)\s+)(.+)',
        }
        
        # Reverie expected output format templates
        self.reverie_templates = {
            'perceive': {
                'format': 'list',
                'example': ['event1', 'event2', 'event3']
            },
            'retrieve': {
                'format': 'dict',
                'example': {'relevant_memory': 'content', 'importance': 5}
            },
            'plan': {
                'format': 'dict', 
                'example': {'action': 'specific_action', 'reasoning': 'why_this_action'}
            },
            'reflect': {
                'format': 'dict',
                'example': {'insight': 'reflection_content', 'mood': 'emotional_state'}
            },
            'execute': {
                'format': 'dict',
                'example': {'action': 'executed_action', 'result': 'outcome'}
            },
            'converse': {
                'format': 'string',
                'example': 'Hello, how are you?'
            }
        }

    def adapt_output(self, raw_output: str, context_type: str, persona_name: str = "Agent") -> Any:
        """
        Convert Qwen raw output to Reverie expected format
        
        Args:
            raw_output: Qwen's raw output
            context_type: Context type (perceive, retrieve, plan, etc.)
            persona_name: Character name
            
        Returns:
            Converted formatted output
        """
        if not raw_output or not raw_output.strip():
            return self._get_safe_fallback(context_type, persona_name)
        
        # Clean output
        cleaned_output = self._clean_raw_output(raw_output)
        
        # Select adaptation strategy based on context type
        try:
            if context_type == 'perceive':
                return self._adapt_perceive_output(cleaned_output, persona_name)
            elif context_type == 'retrieve':
                return self._adapt_retrieve_output(cleaned_output, persona_name)
            elif context_type == 'plan':
                return self._adapt_plan_output(cleaned_output, persona_name)
            elif context_type == 'reflect':
                return self._adapt_reflect_output(cleaned_output, persona_name)
            elif context_type == 'execute':
                return self._adapt_execute_output(cleaned_output, persona_name)
            elif context_type == 'converse':
                return self._adapt_converse_output(cleaned_output, persona_name)
            else:
                return self._generic_adapt(cleaned_output, context_type, persona_name)
                
        except Exception as e:
            if self.config.debug_mode:
                self.logger.error(f"Adaptation failed for {context_type}: {e}")
            return self._get_safe_fallback(context_type, persona_name)

    def _clean_raw_output(self, raw_output: str) -> str:
        """Clean Qwen's raw output"""
        cleaned = raw_output.strip()
        
        # Remove common Qwen prefixes
        qwen_prefixes = [
            "As an AI assistant,", "Based on the provided information,", "Based on the context,",
            "Considering the current situation,", "My answer is:", "Output:", "Result:",
            "Based on the context,", "As an AI,", "My response:",
            "Output:", "Result:", "Answer:"
        ]
        
        for prefix in qwen_prefixes:
            if cleaned.startswith(prefix):
                cleaned = cleaned[len(prefix):].strip()
        
        # Remove extra symbols
        cleaned = re.sub(r'^[：:：\-\s]+', '', cleaned)
        cleaned = re.sub(r'[。\.]{2,}$', '.', cleaned)
        
        return cleaned

    def _adapt_perceive_output(self, output: str, persona_name: str) -> List[str]:
        """Adapt perception output to list format"""
        if self.config.simplified_mode:
            # Simplified mode: directly split into event list
            events = []
            
            # Method 1: Split by punctuation
            sentences = re.split(r'[。！？；，\n]', output)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 3:
                    events.append(sentence[:50])  # Limit length
            
            # Method 2: If no events found, use overall content
            if not events:
                events = [output[:50] if len(output) > 50 else output]
            
            return events[:5]  # Max 5 events
        else:
            # Standard mode: try to parse more complex formats
            return self._extract_list_items(output, 'perceived_event')

    def _adapt_retrieve_output(self, output: str, persona_name: str) -> Dict[str, Any]:
        """Adapt retrieval output to dictionary format"""
        return {
            'relevant_memory': output[:100],
            'importance': 5,
            'confidence': 0.7,
            'timestamp': 'recent',
            'source': f'{persona_name}_memory'
        }

    def _adapt_plan_output(self, output: str, persona_name: str) -> Dict[str, Any]:
        """Adapt planning output to dictionary format"""
        # Try to extract action words
        action_match = re.search(r'(analyze|think|decide|act|go|do|say|look|listen|walk|stop|wait|talk|work|rest)', output.lower())
        action = action_match.group(1) if action_match else 'think'
        
        # English action word matching
        en_action_match = re.search(r'(analyze|think|decide|act|go|do|say|look|listen|walk|stop|wait|talk|work|rest)', output.lower())
        if en_action_match:
            action = en_action_match.group(1)
        
        return {
            'action': action,
            'reasoning': output[:100],
            'priority': 'medium',
            'expected_outcome': 'positive',
            'duration': 'short'
        }

    def _adapt_reflect_output(self, output: str, persona_name: str) -> Dict[str, Any]:
        """Adapt reflection output to dictionary format"""
        # Try to identify mood words
        mood_patterns = {
            'happy': r'(happy|pleased|satisfied)',
            'neutral': r'(calm|normal)',
            'concerned': r'(worried|confused)',
            'focused': r'(concentrated|serious)'
        }
        
        mood = 'neutral'
        for mood_type, pattern in mood_patterns.items():
            if re.search(pattern, output, re.IGNORECASE):
                mood = mood_type
                break
        
        return {
            'insight': output[:100],
            'mood': mood,
            'confidence': 0.6,
            'importance': 3,
            'actionable': True
        }

    def _adapt_execute_output(self, output: str, persona_name: str) -> Dict[str, Any]:
        """Adapt execution output to dictionary format"""
        return {
            'action': output[:50] if len(output) > 50 else output,
            'result': 'completed',
            'success': True,
            'feedback': 'action_executed',
            'next_step': 'continue'
        }

    def _adapt_converse_output(self, output: str, persona_name: str) -> str:
        """Adapt conversation output to string format"""
        # Conversation is the simplest, just return the cleaned text
        if len(output) > 200:
            output = output[:200] + "..."
        
        # Ensure output looks like a natural conversation
        if not re.match(r'^[A-Z\u4e00-\u9fff]', output):
            output = output.capitalize()
        
        if not output.endswith(('.', '!', '?', '。', '！', '？')):
            output += '.'
            
        return output

    def _extract_list_items(self, text: str, default_prefix: str = "item") -> List[str]:
        """Extract list items from text"""
        items = []
        
        # Method 1: Numbered lists
        numbered_items = re.findall(r'^\d+\.?\s*(.+?)(?=\n\d+\.|\n|$)', text, re.MULTILINE)
        if numbered_items:
            return numbered_items[:5]
        
        # Method 2: Separator splitting
        separators = ['\n', '；', ';', '，', ',', '。', '.']
        for sep in separators:
            if sep in text:
                items = [item.strip() for item in text.split(sep) if item.strip()]
                if len(items) > 1:
                    return items[:5]
        
        # Method 3: Default single item
        return [text[:50] if len(text) > 50 else text]

    def _generic_adapt(self, output: str, context_type: str, persona_name: str) -> Any:
        """Generic adaptation logic"""
        template = self.reverie_templates.get(context_type, {})
        format_type = template.get('format', 'string')
        
        if format_type == 'list':
            return self._extract_list_items(output)
        elif format_type == 'dict':
            return {'content': output, 'type': context_type, 'source': persona_name}
        else:
            return output

    def _get_safe_fallback(self, context_type: str, persona_name: str) -> Any:
        """Get safe fallback output"""
        fallbacks = {
            'perceive': [f'{persona_name} is observing the environment'],
            'retrieve': {'relevant_memory': f'{persona_name} recalls recent events', 'importance': 3},
            'plan': {'action': 'observe', 'reasoning': f'{persona_name} decides to observe first'},
            'reflect': {'insight': f'{persona_name} is thinking', 'mood': 'neutral'},
            'execute': {'action': 'wait', 'result': 'waiting'},
            'converse': f'Hello, I am {persona_name}.'
        }
        
        return fallbacks.get(context_type, f'{persona_name} is processing...')

# Global adapter instance
_global_adapter = None

def get_llm_adapter(config: AdapterConfig = None) -> QwenOutputAdapter:
    """Get global LLM adapter instance"""
    global _global_adapter
    if _global_adapter is None:
        _global_adapter = QwenOutputAdapter(config or AdapterConfig())
    return _global_adapter

def adapt_llm_output(raw_output: str, context_type: str, persona_name: str = "Agent") -> Any:
    """Convenient function: adapt LLM output"""
    adapter = get_llm_adapter()
    return adapter.adapt_output(raw_output, context_type, persona_name)

# Test function
def test_adapter():
    """Test adapter functionality"""
    adapter = QwenOutputAdapter(AdapterConfig(debug_mode=True))
    
    test_cases = [
        ("I see a person walking, the weather is nice, birds are singing.", "perceive", "TestAgent"),
        ("I should go to the store to buy things, because I need to replenish food.", "plan", "TestAgent"), 
        ("Hello, how are you today? I hope everything is going well.", "converse", "TestAgent"),
        ("I think today went well, learned new things, very satisfied.", "reflect", "TestAgent"),
    ]
    
    for raw_output, context_type, persona_name in test_cases:
        result = adapter.adapt_output(raw_output, context_type, persona_name)
        print(f"\n=== {context_type.upper()} ===")
        print(f"Input: {raw_output[:50]}...")
        print(f"Output: {result}")

if __name__ == "__main__":
    test_adapter()
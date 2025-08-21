"""
Unified AI Response Parser
Intelligent parsing system with machine learning capabilities and error handling
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from enum import Enum
import pickle
from collections import defaultdict, Counter

class ParseType(Enum):
    """Types of parsing operations"""
    DECISION = "decision"
    EMOTION = "emotion"
    ACTION = "action"
    CHAT_RESPONSE = "chat_response"
    REASONING = "reasoning"
    CHOICE = "choice"
    MOOD = "mood"

@dataclass
class ParseResult:
    """Result of a parsing operation"""
    parse_type: ParseType
    success: bool
    value: Any
    confidence: float
    raw_input: str
    alternatives: List[Any] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "parse_type": self.parse_type.value,
            "success": self.success,
            "value": self.value,
            "confidence": self.confidence,
            "raw_input": self.raw_input,
            "alternatives": self.alternatives,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat()
        }

@dataclass
class ParsePattern:
    """Pattern for parsing specific content"""
    name: str
    regex: str
    parse_type: ParseType
    confidence: float = 1.0
    description: str = ""
    examples: List[str] = field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate of this pattern"""
        total = self.success_count + self.failure_count
        return self.success_count / total if total > 0 else 0.0

class LearningParser:
    """Learning-based parser that improves over time"""
    
    def __init__(self, training_data_path: Optional[str] = None):
        self.training_data_path = training_data_path or "ai_service/parser_training_data.json"
        self.patterns: Dict[ParseType, List[ParsePattern]] = defaultdict(list)
        self.success_history: List[ParseResult] = []
        self.failure_history: List[ParseResult] = []
        self.logger = logging.getLogger(__name__)
        
        # Load existing patterns and training data
        self._load_patterns()
        self._load_training_data()
    
    def _load_patterns(self):
        """Load predefined parsing patterns"""
        
        # Decision patterns
        self.patterns[ParseType.DECISION].extend([
            ParsePattern(
                name="choice_format",
                regex=r"CHOICE:\s*(\d+|[^\\n]+)",
                parse_type=ParseType.DECISION,
                confidence=0.9,
                description="Extract CHOICE: formatted decisions",
                examples=["CHOICE: 1", "CHOICE: go to store"]
            ),
            ParsePattern(
                name="action_format", 
                regex=r"ACTION:\s*([^\\n]+)",
                parse_type=ParseType.ACTION,
                confidence=0.85,
                description="Extract ACTION: formatted actions",
                examples=["ACTION: walk to store", "ACTION: talk to Bob"]
            ),
            ParsePattern(
                name="decision_verb",
                regex=r"I (?:choose|decide|will|want to)\\s+([^.\\n]+)",
                parse_type=ParseType.DECISION,
                confidence=0.7,
                description="Extract decisions with common verbs",
                examples=["I choose to go home", "I will talk to Alice"]
            )
        ])
        
        # Reasoning patterns
        self.patterns[ParseType.REASONING].extend([
            ParsePattern(
                name="reason_format",
                regex=r"REASON(?:ING)?:\\s*([^\\n]+)",
                parse_type=ParseType.REASONING,
                confidence=0.9,
                description="Extract REASON: formatted reasoning",
                examples=["REASON: I need to rest", "REASONING: It seems logical"]
            ),
            ParsePattern(
                name="because_reasoning",
                regex=r"because\\s+([^.\\n]+)",
                parse_type=ParseType.REASONING,
                confidence=0.6,
                description="Extract reasoning with 'because'",
                examples=["I choose this because I am tired"]
            )
        ])
        
        # Emotion patterns
        self.patterns[ParseType.EMOTION].extend([
            ParsePattern(
                name="emotion_words",
                regex=r"\\b(happy|sad|angry|excited|worried|confused|neutral|relaxed|frustrated|pleased)\\b",
                parse_type=ParseType.EMOTION,
                confidence=0.8,
                description="Direct emotion word detection",
                examples=["I feel happy", "I'm angry about this"]
            ),
            ParsePattern(
                name="feeling_expressions",
                regex=r"I (?:feel|am|\'m)\\s+(\\w+)",
                parse_type=ParseType.EMOTION,
                confidence=0.7,
                description="Emotion expressions with 'I feel/am'",
                examples=["I feel excited", "I am worried"]
            )
        ])
        
        # Mood patterns
        self.patterns[ParseType.MOOD].extend([
            ParsePattern(
                name="mood_keywords",
                regex=r"\\b(contemplative|joyful|melancholy|anxious|enthusiastic|irritated|peaceful)\\b",
                parse_type=ParseType.MOOD,
                confidence=0.8,
                description="Mood-specific keywords",
                examples=["feeling contemplative", "in a joyful mood"]
            )
        ])
    
    def _load_training_data(self):
        """Load training data from file"""
        try:
            training_path = Path(self.training_data_path)
            if training_path.exists():
                with open(training_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    
                # Load success/failure history
                for item in data.get('success_history', []):
                    result = ParseResult(
                        parse_type=ParseType(item['parse_type']),
                        success=item['success'],
                        value=item['value'],
                        confidence=item['confidence'],
                        raw_input=item['raw_input'],
                        alternatives=item.get('alternatives', []),
                        metadata=item.get('metadata', {}),
                        timestamp=datetime.fromisoformat(item['timestamp'])
                    )
                    self.success_history.append(result)
                
                # Update pattern statistics
                pattern_stats = data.get('pattern_stats', {})
                for parse_type_str, patterns_list in pattern_stats.items():
                    parse_type = ParseType(parse_type_str)
                    for i, stats in enumerate(patterns_list):
                        if i < len(self.patterns[parse_type]):
                            self.patterns[parse_type][i].success_count = stats.get('success_count', 0)
                            self.patterns[parse_type][i].failure_count = stats.get('failure_count', 0)
                
                self.logger.info(f"Loaded {len(self.success_history)} training examples")
                
        except Exception as e:
            self.logger.warning(f"Could not load training data: {e}")
    
    def _save_training_data(self):
        """Save training data to file"""
        try:
            training_path = Path(self.training_data_path)
            training_path.parent.mkdir(exist_ok=True)
            
            # Prepare data for serialization
            data = {
                'success_history': [result.to_dict() for result in self.success_history[-1000:]],  # Keep last 1000
                'failure_history': [result.to_dict() for result in self.failure_history[-1000:]],  # Keep last 1000
                'pattern_stats': {}
            }
            
            # Save pattern statistics
            for parse_type, patterns_list in self.patterns.items():
                data['pattern_stats'][parse_type.value] = [
                    {
                        'name': p.name,
                        'success_count': p.success_count,
                        'failure_count': p.failure_count,
                        'success_rate': p.success_rate
                    }
                    for p in patterns_list
                ]
            
            with open(training_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Could not save training data: {e}")
    
    def parse(self, text: str, parse_type: ParseType, options: Optional[List[str]] = None) -> ParseResult:
        """Parse text using learned patterns"""
        
        # Try patterns in order of confidence/success rate
        patterns = sorted(
            self.patterns[parse_type],
            key=lambda p: (p.success_rate, p.confidence),
            reverse=True
        )
        
        alternatives = []
        best_result = None
        
        for pattern in patterns:
            try:
                match = re.search(pattern.regex, text, re.IGNORECASE)
                if match:
                    value = match.group(1).strip() if match.groups() else match.group(0).strip()
                    
                    # Additional processing based on parse type
                    processed_value = self._post_process_value(value, parse_type, options)
                    
                    confidence = pattern.confidence * pattern.success_rate
                    if confidence == 0:
                        confidence = pattern.confidence * 0.5  # Give new patterns a chance
                    
                    result = ParseResult(
                        parse_type=parse_type,
                        success=True,
                        value=processed_value,
                        confidence=confidence,
                        raw_input=text,
                        alternatives=alternatives,
                        metadata={
                            'pattern_name': pattern.name,
                            'pattern_regex': pattern.regex,
                            'match_text': match.group(0)
                        }
                    )
                    
                    if not best_result or confidence > best_result.confidence:
                        if best_result:
                            alternatives.append(best_result.value)
                        best_result = result
                    else:
                        alternatives.append(processed_value)
                        
            except Exception as e:
                self.logger.debug(f"Pattern {pattern.name} failed: {e}")
                continue
        
        # If no pattern matched, try fallback
        if not best_result:
            best_result = self._fallback_parse(text, parse_type, options)
        
        # Record result for learning
        self._record_result(best_result)
        
        return best_result
    
    def _post_process_value(self, value: str, parse_type: ParseType, options: Optional[List[str]] = None) -> Any:
        """Post-process parsed value based on type"""
        
        if parse_type == ParseType.DECISION and options:
            # Try to match to available options
            value_lower = value.lower()
            
            # Direct match
            for option in options:
                if option.lower() == value_lower:
                    return option
            
            # Partial match
            for option in options:
                if value_lower in option.lower() or option.lower() in value_lower:
                    return option
            
            # Number match
            try:
                index = int(value) - 1
                if 0 <= index < len(options):
                    return options[index]
            except ValueError:
                pass
            
            # Default to first option
            return options[0] if options else value
        
        elif parse_type == ParseType.EMOTION:
            # Normalize emotion
            emotion_map = {
                'pleased': 'happy',
                'frustrated': 'angry',
                'peaceful': 'relaxed',
                'anxious': 'worried'
            }
            return emotion_map.get(value.lower(), value.lower())
        
        return value
    
    def _fallback_parse(self, text: str, parse_type: ParseType, options: Optional[List[str]] = None) -> ParseResult:
        """Fallback parsing when patterns fail"""
        
        # Simple keyword-based fallback
        fallback_value = None
        confidence = 0.3
        
        if parse_type == ParseType.DECISION and options:
            # Find option mentioned in text
            text_lower = text.lower()
            for option in options:
                if option.lower() in text_lower:
                    fallback_value = option
                    break
            if not fallback_value:
                fallback_value = options[0]  # Default to first
                
        elif parse_type == ParseType.EMOTION:
            # Look for any emotion words
            emotions = ['happy', 'sad', 'angry', 'excited', 'worried', 'confused', 'neutral', 'relaxed']
            text_lower = text.lower()
            for emotion in emotions:
                if emotion in text_lower:
                    fallback_value = emotion
                    break
            if not fallback_value:
                fallback_value = 'neutral'
                
        elif parse_type == ParseType.ACTION:
            # Extract first verb phrase
            action_verbs = ['go', 'walk', 'talk', 'eat', 'rest', 'work', 'think', 'move']
            words = text.lower().split()
            for i, word in enumerate(words):
                if word in action_verbs and i + 1 < len(words):
                    fallback_value = ' '.join(words[i:i+3])  # Take verb + next 2 words
                    break
            if not fallback_value:
                fallback_value = 'idle'
                
        else:
            # Generic fallback - take first sentence
            sentences = text.split('.')
            fallback_value = sentences[0].strip() if sentences else text.strip()
        
        return ParseResult(
            parse_type=parse_type,
            success=bool(fallback_value),
            value=fallback_value,
            confidence=confidence,
            raw_input=text,
            metadata={'fallback': True}
        )
    
    def _record_result(self, result: ParseResult):
        """Record parsing result for learning"""
        if result.success:
            self.success_history.append(result)
            
            # Update pattern success count
            pattern_name = result.metadata.get('pattern_name')
            if pattern_name:
                for pattern in self.patterns[result.parse_type]:
                    if pattern.name == pattern_name:
                        pattern.success_count += 1
                        break
        else:
            self.failure_history.append(result)
            
            # Update pattern failure count
            pattern_name = result.metadata.get('pattern_name')
            if pattern_name:
                for pattern in self.patterns[result.parse_type]:
                    if pattern.name == pattern_name:
                        pattern.failure_count += 1
                        break
        
        # Periodically save training data
        if len(self.success_history) % 50 == 0:  # Save every 50 successes
            self._save_training_data()
    
    def add_training_example(self, text: str, parse_type: ParseType, expected_value: Any, options: Optional[List[str]] = None):
        """Add a training example"""
        result = self.parse(text, parse_type, options)
        
        # Check if parsing was correct
        if result.value == expected_value:
            result.success = True
            result.confidence = min(1.0, result.confidence + 0.1)  # Boost confidence
        else:
            result.success = False
            result.confidence = max(0.1, result.confidence - 0.1)  # Reduce confidence
            result.metadata['expected_value'] = expected_value
        
        self._record_result(result)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics"""
        stats = {
            'total_success': len(self.success_history),
            'total_failures': len(self.failure_history),
            'success_rate': len(self.success_history) / max(len(self.success_history) + len(self.failure_history), 1),
            'pattern_stats': {}
        }
        
        for parse_type, patterns_list in self.patterns.items():
            stats['pattern_stats'][parse_type.value] = [
                {
                    'name': p.name,
                    'success_rate': p.success_rate,
                    'total_uses': p.success_count + p.failure_count
                }
                for p in patterns_list
            ]
        
        return stats

class UnifiedParser:
    """Unified interface for all AI response parsing"""
    
    def __init__(self, enable_learning: bool = True):
        self.enable_learning = enable_learning
        self.learning_parser = LearningParser() if enable_learning else None
        self.logger = logging.getLogger(__name__)
    
    def parse_decision(self, response: str, available_options: List[str]) -> Tuple[str, str, float]:
        """Parse decision response from AI"""
        try:
            if self.learning_parser:
                # Use learning parser
                choice_result = self.learning_parser.parse(response, ParseType.DECISION, available_options)
                reason_result = self.learning_parser.parse(response, ParseType.REASONING)
                
                chosen_option = choice_result.value if choice_result.success else available_options[0]
                reasoning = reason_result.value if reason_result.success else "No specific reason provided"
                confidence = (choice_result.confidence + reason_result.confidence) / 2
                
                return chosen_option, reasoning, confidence
            else:
                # Fallback to simple parsing
                return self._simple_parse_decision(response, available_options)
                
        except Exception as e:
            self.logger.error(f"Decision parsing error: {e}")
            return available_options[0], "Parsing error occurred", 0.1
    
    def parse_emotion(self, response: str) -> Optional[str]:
        """Parse emotion from AI response"""
        try:
            if self.learning_parser:
                result = self.learning_parser.parse(response, ParseType.EMOTION)
                return result.value if result.success else None
            else:
                return self._simple_parse_emotion(response)
                
        except Exception as e:
            self.logger.error(f"Emotion parsing error: {e}")
            return None
    
    def parse_action(self, response: str) -> Optional[str]:
        """Parse action from AI response"""
        try:
            if self.learning_parser:
                result = self.learning_parser.parse(response, ParseType.ACTION)
                return result.value if result.success else None
            else:
                return self._simple_parse_action(response)
                
        except Exception as e:
            self.logger.error(f"Action parsing error: {e}")
            return None
    
    def parse_mood(self, response: str) -> str:
        """Parse mood from AI response"""
        try:
            if self.learning_parser:
                result = self.learning_parser.parse(response, ParseType.MOOD)
                return result.value if result.success else "neutral"
            else:
                return self._simple_parse_mood(response)
                
        except Exception as e:
            self.logger.error(f"Mood parsing error: {e}")
            return "neutral"
    
    def _simple_parse_decision(self, response: str, available_options: List[str]) -> Tuple[str, str, float]:
        """Simple decision parsing fallback"""
        lines = response.strip().split('\\n')
        chosen_option = available_options[0]  # default
        reasoning = "No specific reason"
        
        for line in lines:
            if "CHOICE:" in line or "ACTION:" in line:
                choice_text = line.split(":")[-1].strip()
                for option in available_options:
                    if option.lower() in choice_text.lower():
                        chosen_option = option
                        break
            elif "REASON:" in line or "REASONING:" in line:
                reasoning = line.split(":")[-1].strip()
        
        return chosen_option, reasoning, 0.7
    
    def _simple_parse_emotion(self, response: str) -> Optional[str]:
        """Simple emotion parsing fallback"""
        emotions = ["happy", "sad", "angry", "surprised", "neutral", "confused", "excited", "worried", "relaxed"]
        for emotion in emotions:
            if emotion in response.lower():
                return emotion
        return None
    
    def _simple_parse_action(self, response: str) -> Optional[str]:
        """Simple action parsing fallback"""
        if "ACTION:" in response:
            return response.split("ACTION:")[-1].strip().split('\\n')[0]
        return None
    
    def _simple_parse_mood(self, response: str) -> str:
        """Simple mood parsing fallback"""
        mood_keywords = {
            "happy": ["joy", "happy", "glad", "pleased", "delighted"],
            "sad": ["sad", "unhappy", "melancholy", "depressed"],
            "worried": ["worried", "anxious", "concerned", "nervous"],
            "excited": ["excited", "thrilled", "eager", "enthusiastic"],
            "angry": ["angry", "frustrated", "annoyed", "irritated"]
        }
        
        for mood_name, keywords in mood_keywords.items():
            if any(keyword in response.lower() for keyword in keywords):
                return mood_name
        
        return "contemplative"  # default mood
    
    def add_training_example(self, text: str, parse_type: str, expected_value: Any, options: Optional[List[str]] = None):
        """Add training example for learning"""
        if self.learning_parser:
            try:
                parse_type_enum = ParseType(parse_type)
                self.learning_parser.add_training_example(text, parse_type_enum, expected_value, options)
            except ValueError:
                self.logger.error(f"Invalid parse type: {parse_type}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get parsing statistics"""
        if self.learning_parser:
            return self.learning_parser.get_statistics()
        return {"learning_disabled": True}

# Global unified parser instance
_unified_parser = None

def get_unified_parser() -> UnifiedParser:
    """Get global unified parser instance"""
    global _unified_parser
    if _unified_parser is None:
        _unified_parser = UnifiedParser(enable_learning=True)
    return _unified_parser

# Training data collection utilities
def collect_training_data_from_logs(log_directory: str, output_file: str):
    """Collect training data from log files"""
    import glob
    
    training_examples = []
    log_files = glob.glob(f"{log_directory}/*.log")
    
    for log_file in log_files:
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                content = f.read()
                
                # Extract examples from logs (implement based on log format)
                # This is a template - customize based on actual log format
                examples = extract_examples_from_log_content(content)
                training_examples.extend(examples)
                
        except Exception as e:
            print(f"Error processing {log_file}: {e}")
    
    # Save training examples
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(training_examples, f, indent=2)
    
    print(f"Collected {len(training_examples)} training examples to {output_file}")

def extract_examples_from_log_content(content: str) -> List[Dict[str, Any]]:
    """Extract training examples from log content"""
    examples = []
    
    # Look for successful decision patterns
    decision_pattern = r"Decision: (.+?) -> (.+?) \\(confidence: ([0-9.]+)\\)"
    for match in re.finditer(decision_pattern, content):
        examples.append({
            "text": match.group(1),
            "parse_type": "decision",
            "expected_value": match.group(2),
            "confidence": float(match.group(3))
        })
    
    # Add more extraction patterns as needed
    
    return examples

# Test the unified parser
def test_unified_parser():
    """Test the unified parser"""
    print("=== Unified Parser Test ===\\n")
    
    parser = get_unified_parser()
    
    # Test decision parsing
    print("1. Testing decision parsing:")
    response = "I think I should go to the store. CHOICE: go to store REASON: I need supplies for dinner."
    options = ["go to store", "stay home", "visit friend", "take a nap"]
    choice, reason, confidence = parser.parse_decision(response, options)
    print(f"Choice: {choice}")
    print(f"Reason: {reason}")
    print(f"Confidence: {confidence:.2f}\\n")
    
    # Test emotion parsing
    print("2. Testing emotion parsing:")
    response = "I feel really excited about this new adventure!"
    emotion = parser.parse_emotion(response)
    print(f"Emotion: {emotion}\\n")
    
    # Test action parsing
    print("3. Testing action parsing:")
    response = "ACTION: walk to the market and buy some apples"
    action = parser.parse_action(response)
    print(f"Action: {action}\\n")
    
    # Test mood parsing
    print("4. Testing mood parsing:")
    response = "I'm feeling quite contemplative today, thinking about life and its mysteries."
    mood = parser.parse_mood(response)
    print(f"Mood: {mood}\\n")
    
    # Get statistics
    print("5. Parser statistics:")
    stats = parser.get_statistics()
    print(json.dumps(stats, indent=2))
    
    print("\\n=== Test completed ===")

if __name__ == "__main__":
    test_unified_parser()
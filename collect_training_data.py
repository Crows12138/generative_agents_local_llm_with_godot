#!/usr/bin/env python3
"""
Training Data Collection Script
Collect successful parsing examples for model training from log files
"""

import os
import sys
import json
import re
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime
import glob

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from ai_service.unified_parser import ParseType, extract_examples_from_log_content

class TrainingDataCollector:
    """Collects training data from various sources"""
    
    def __init__(self, output_dir: str = "training_data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.examples = []
    
    def collect_from_logs(self, log_directory: str, pattern: str = "*.log"):
        """Collect training data from log files"""
        log_files = glob.glob(str(Path(log_directory) / pattern))
        
        print(f"Found {len(log_files)} log files to process...")
        
        for log_file in log_files:
            try:
                print(f"Processing {log_file}...")
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    
                examples = self._extract_from_log_content(content)
                self.examples.extend(examples)
                print(f"  Extracted {len(examples)} examples")
                
            except Exception as e:
                print(f"Error processing {log_file}: {e}")
        
        print(f"Total examples collected: {len(self.examples)}")
    
    def collect_from_ai_service_logs(self, log_directory: str = "logs"):
        """Collect from AI service specific logs"""
        ai_log_files = glob.glob(str(Path(log_directory) / "*ai_service*.log"))
        ai_log_files.extend(glob.glob(str(Path(log_directory) / "*bridge*.log")))
        
        for log_file in ai_log_files:
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    examples = self._extract_ai_service_examples(content)
                    self.examples.extend(examples)
                    
            except Exception as e:
                print(f"Error processing AI service log {log_file}: {e}")
    
    def collect_from_debug_sessions(self, debug_dir: str = "debug_system"):
        """Collect from debug session files"""
        debug_files = glob.glob(str(Path(debug_dir) / "*_session.json"))
        debug_files.extend(glob.glob(str(Path(debug_dir) / "*_traces.json")))
        
        for debug_file in debug_files:
            try:
                with open(debug_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    examples = self._extract_from_debug_data(data)
                    self.examples.extend(examples)
                    
            except Exception as e:
                print(f"Error processing debug file {debug_file}: {e}")
    
    def _extract_from_log_content(self, content: str) -> List[Dict[str, Any]]:
        """Extract training examples from log content"""
        examples = []
        
        # Decision parsing examples
        decision_patterns = [
            r"Decision request: (.+?) -> Choice: (.+?) \\(confidence: ([0-9.]+)\\)",
            r"ACTION: (.+?) REASON: (.+?)(?:\\n|$)",
            r"CHOICE: (.+?) REASON: (.+?)(?:\\n|$)",
            r"Character .+ decides to: (.+?) because (.+?)(?:\\n|$)"
        ]
        
        for pattern in decision_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                if len(match.groups()) >= 2:
                    examples.append({
                        "text": match.group(0),
                        "parse_type": "decision",
                        "expected_value": match.group(1).strip(),
                        "reasoning": match.group(2).strip() if len(match.groups()) > 2 else "",
                        "confidence": float(match.group(3)) if len(match.groups()) > 2 and match.group(3).replace('.', '').isdigit() else 0.8,
                        "source": "log_file",
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Emotion parsing examples
        emotion_patterns = [
            r"Emotion detected: (\\w+) from response: (.+?)(?:\\n|$)",
            r"Character feeling (\\w+): (.+?)(?:\\n|$)",
            r"Emotional state: (\\w+) - (.+?)(?:\\n|$)"
        ]
        
        for pattern in emotion_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                examples.append({
                    "text": match.group(2).strip(),
                    "parse_type": "emotion",
                    "expected_value": match.group(1).lower().strip(),
                    "source": "log_file",
                    "timestamp": datetime.now().isoformat()
                })
        
        # Action parsing examples
        action_patterns = [
            r"Action chosen: (.+?) from response: (.+?)(?:\\n|$)",
            r"Agent action: (.+?) - (.+?)(?:\\n|$)"
        ]
        
        for pattern in action_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                examples.append({
                    "text": match.group(2).strip(),
                    "parse_type": "action",
                    "expected_value": match.group(1).strip(),
                    "source": "log_file",
                    "timestamp": datetime.now().isoformat()
                })
        
        return examples
    
    def _extract_ai_service_examples(self, content: str) -> List[Dict[str, Any]]:
        """Extract examples from AI service logs"""
        examples = []
        
        # Look for API call patterns
        api_patterns = [
            r"POST /ai/chat.*?response: (.+?)(?:\\n|$)",
            r"POST /ai/decide.*?chosen_option: (.+?).*?reasoning: (.+?)(?:\\n|$)",
            r"POST /ai/think.*?thought: (.+?).*?mood: (.+?)(?:\\n|$)"
        ]
        
        for pattern in api_patterns:
            for match in re.finditer(pattern, content, re.MULTILINE | re.DOTALL):
                if "chat" in pattern:
                    examples.append({
                        "text": match.group(1).strip(),
                        "parse_type": "chat_response",
                        "expected_value": match.group(1).strip(),
                        "source": "ai_service_log",
                        "timestamp": datetime.now().isoformat()
                    })
                elif "decide" in pattern:
                    examples.append({
                        "text": match.group(0),
                        "parse_type": "decision",
                        "expected_value": match.group(1).strip(),
                        "reasoning": match.group(2).strip(),
                        "source": "ai_service_log",
                        "timestamp": datetime.now().isoformat()
                    })
                elif "think" in pattern:
                    examples.append({
                        "text": match.group(1).strip(),
                        "parse_type": "mood",
                        "expected_value": match.group(2).strip(),
                        "source": "ai_service_log",
                        "timestamp": datetime.now().isoformat()
                    })
        
        return examples
    
    def _extract_from_debug_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract examples from debug session data"""
        examples = []
        
        # Extract from agent actions
        if "agent_actions" in data:
            for action_data in data["agent_actions"]:
                if "ai_response" in action_data and "chosen_action" in action_data:
                    examples.append({
                        "text": action_data["ai_response"],
                        "parse_type": "decision",
                        "expected_value": action_data["chosen_action"],
                        "options": action_data.get("available_actions", []),
                        "source": "debug_session",
                        "timestamp": action_data.get("timestamp", datetime.now().isoformat())
                    })
        
        # Extract from conversation data
        if "conversations" in data:
            for conv_data in data["conversations"]:
                if "ai_response" in conv_data:
                    examples.append({
                        "text": conv_data["ai_response"],
                        "parse_type": "chat_response",
                        "expected_value": conv_data["ai_response"],
                        "emotion": conv_data.get("detected_emotion"),
                        "source": "debug_session",
                        "timestamp": conv_data.get("timestamp", datetime.now().isoformat())
                    })
        
        return examples
    
    def add_synthetic_examples(self):
        """Add synthetic training examples for better coverage"""
        synthetic_examples = [
            # Decision examples
            {
                "text": "I choose to go to the store because I need groceries.",
                "parse_type": "decision",
                "expected_value": "go to the store",
                "options": ["go to the store", "stay home", "visit friend"],
                "source": "synthetic"
            },
            {
                "text": "ACTION: walk to park REASON: need fresh air",
                "parse_type": "decision",
                "expected_value": "walk to park",
                "source": "synthetic"
            },
            {
                "text": "CHOICE: 1 REASONING: It seems like the best option",
                "parse_type": "decision",
                "expected_value": "1",
                "source": "synthetic"
            },
            
            # Emotion examples
            {
                "text": "I feel really happy about this outcome!",
                "parse_type": "emotion",
                "expected_value": "happy",
                "source": "synthetic"
            },
            {
                "text": "This makes me quite angry and frustrated.",
                "parse_type": "emotion",
                "expected_value": "angry",
                "source": "synthetic"
            },
            {
                "text": "I'm feeling worried about the situation.",
                "parse_type": "emotion",
                "expected_value": "worried",
                "source": "synthetic"
            },
            
            # Action examples
            {
                "text": "ACTION: talk to the shopkeeper about prices",
                "parse_type": "action",
                "expected_value": "talk to the shopkeeper about prices",
                "source": "synthetic"
            },
            {
                "text": "I will move to the center of the room.",
                "parse_type": "action",
                "expected_value": "move to the center of the room",
                "source": "synthetic"
            },
            
            # Mood examples
            {
                "text": "I'm in a contemplative mood, thinking deeply about life.",
                "parse_type": "mood",
                "expected_value": "contemplative",
                "source": "synthetic"
            },
            {
                "text": "Feeling quite joyful and enthusiastic today!",
                "parse_type": "mood",
                "expected_value": "joyful",
                "source": "synthetic"
            }
        ]
        
        # Add timestamps
        for example in synthetic_examples:
            example["timestamp"] = datetime.now().isoformat()
        
        self.examples.extend(synthetic_examples)
        print(f"Added {len(synthetic_examples)} synthetic examples")
    
    def save_training_data(self, filename: str = None):
        """Save collected training data"""
        if filename is None:
            filename = f"training_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_path = self.output_dir / filename
        
        # Organize examples by type
        organized_data = {
            "metadata": {
                "total_examples": len(self.examples),
                "collection_date": datetime.now().isoformat(),
                "source_types": list(set(ex.get("source", "unknown") for ex in self.examples))
            },
            "examples_by_type": {},
            "all_examples": self.examples
        }
        
        # Group by parse type
        for example in self.examples:
            parse_type = example.get("parse_type", "unknown")
            if parse_type not in organized_data["examples_by_type"]:
                organized_data["examples_by_type"][parse_type] = []
            organized_data["examples_by_type"][parse_type].append(example)
        
        # Save data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(organized_data, f, indent=2, ensure_ascii=False)
        
        print(f"Training data saved to: {output_path}")
        
        # Print summary
        print("\\nTraining Data Summary:")
        print(f"Total examples: {len(self.examples)}")
        for parse_type, examples in organized_data["examples_by_type"].items():
            print(f"  {parse_type}: {len(examples)} examples")
        
        return str(output_path)
    
    def validate_training_data(self):
        """Validate collected training data"""
        print("\\nValidating training data...")
        
        validation_results = {
            "valid_examples": 0,
            "invalid_examples": 0,
            "missing_fields": [],
            "parse_type_distribution": {}
        }
        
        required_fields = ["text", "parse_type", "expected_value"]
        
        for example in self.examples:
            is_valid = True
            
            # Check required fields
            for field in required_fields:
                if field not in example or not example[field]:
                    validation_results["missing_fields"].append(field)
                    is_valid = False
            
            # Count parse types
            parse_type = example.get("parse_type", "unknown")
            validation_results["parse_type_distribution"][parse_type] = validation_results["parse_type_distribution"].get(parse_type, 0) + 1
            
            if is_valid:
                validation_results["valid_examples"] += 1
            else:
                validation_results["invalid_examples"] += 1
        
        # Print validation results
        print(f"Valid examples: {validation_results['valid_examples']}")
        print(f"Invalid examples: {validation_results['invalid_examples']}")
        if validation_results["missing_fields"]:
            print(f"Missing fields found: {set(validation_results['missing_fields'])}")
        
        print("Parse type distribution:")
        for parse_type, count in validation_results["parse_type_distribution"].items():
            print(f"  {parse_type}: {count}")
        
        return validation_results

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Collect training data for AI parser")
    parser.add_argument("--input", default="logs", help="Input directory containing logs")
    parser.add_argument("--output", default="training_data", help="Output directory for training data")
    parser.add_argument("--pattern", default="*.log", help="Log file pattern to match")
    parser.add_argument("--include-synthetic", action="store_true", help="Include synthetic examples")
    parser.add_argument("--include-debug", action="store_true", help="Include debug session data")
    parser.add_argument("--validate", action="store_true", help="Validate collected data")
    
    args = parser.parse_args()
    
    # Create collector
    collector = TrainingDataCollector(args.output)
    
    # Collect from different sources
    if Path(args.input).exists():
        print(f"Collecting from log directory: {args.input}")
        collector.collect_from_logs(args.input, args.pattern)
        collector.collect_from_ai_service_logs(args.input)
    else:
        print(f"Warning: Input directory {args.input} not found")
    
    if args.include_debug:
        print("Collecting from debug sessions...")
        collector.collect_from_debug_sessions()
    
    if args.include_synthetic:
        print("Adding synthetic examples...")
        collector.add_synthetic_examples()
    
    # Validate if requested
    if args.validate:
        collector.validate_training_data()
    
    # Save training data
    if collector.examples:
        output_file = collector.save_training_data()
        print(f"\\nTraining data collection completed successfully!")
        print(f"Output file: {output_file}")
    else:
        print("No training examples collected!")

if __name__ == "__main__":
    main()
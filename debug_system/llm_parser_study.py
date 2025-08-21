import re
from typing import Dict, List, Any, Match, Optional


class LLMOutputParser:
    """Deep understanding of how LLM output becomes game actions"""
    
    def __init__(self):
        # Define all possible action patterns
        self.action_patterns = {
            # Movement actions
            "move_to_location": {
                "patterns": [
                    r"(?:walk|go|move|head) (?:to|towards) (?:the )?(\w+)",
                    r"(?:I should|I will|Let me) go to (?:the )?(\w+)",
                ],
                "extract": self.extract_location,
                "game_action": "move"
            },
            
            # Conversation actions
            "talk_to_person": {
                "patterns": [
                    r"(?:talk|speak|chat) (?:to|with) (\w+)",
                    r"(?:say|tell) (?:something to )?(\w+)",
                ],
                "extract": self.extract_person,
                "game_action": "talk"
            },
            
            # Object interaction actions
            "use_object": {
                "patterns": [
                    r"(?:use|pick up|take|grab) (?:the )?(\w+)",
                    r"(?:drink|pour|mix|make) (?:a |some )?(\w+)",
                ],
                "extract": self.extract_object,
                "game_action": "interact"
            },
            
            # Work actions (bartender specific)
            "work_action": {
                "patterns": [
                    r"(?:clean|wipe|polish) (?:the )?(\w+)",
                    r"(?:check|inspect|look at) (?:the )?(\w+)",
                ],
                "extract": self.extract_work_target,
                "game_action": "work"
            }
        }
    
    def extract_location(self, match: Match) -> str:
        """Extract location from regex match"""
        return match.group(1)
    
    def extract_person(self, match: Match) -> str:
        """Extract person name from regex match"""
        return match.group(1)
    
    def extract_object(self, match: Match) -> str:
        """Extract object name from regex match"""
        return match.group(1)
    
    def extract_work_target(self, match: Match) -> str:
        """Extract work target from regex match"""
        return match.group(1)
    
    def parse_with_explanation(self, llm_output: str) -> Dict[str, Any]:
        """Parse and explain each step"""
        print("\n" + "="*50)
        print(f"Original LLM Output: '{llm_output}'")
        print("="*50)
        
        # Try each pattern
        for action_name, config in self.action_patterns.items():
            print(f"\nTrying pattern: {action_name}")
            
            for pattern in config["patterns"]:
                print(f"  Testing regex: {pattern}")
                match = re.search(pattern, llm_output.lower())
                
                if match:
                    print(f"  Match found!")
                    target = config["extract"](match)
                    action = {
                        "type": config["game_action"],
                        "target": target,
                        "original": llm_output
                    }
                    print(f"  Generated action: {action}")
                    return action
                else:
                    print(f"  No match")
        
        print("\nNo matching pattern found, returning idle")
        return {"type": "idle", "original": llm_output}
    
    def test_common_outputs(self) -> None:
        """Test common LLM outputs"""
        test_cases = [
            "I should walk to the bar counter to serve customers",
            "Let me clean the bar counter",
            "Time to talk to Alice about her order",
            "I'll grab a bottle of whiskey",
            "Maybe I should check the inventory",
            "Just standing here watching the customers",
        ]
        
        for test in test_cases:
            result = self.parse_with_explanation(test)
            print(f"\n{'='*50}")
            print(f"Final result: {result}")
            print(f"{'='*50}\n")
    
    def analyze_pattern_coverage(self, llm_outputs: List[str]) -> Dict[str, Any]:
        """Analyze pattern coverage for a set of LLM outputs"""
        results = {
            "total_outputs": len(llm_outputs),
            "matched": 0,
            "unmatched": 0,
            "pattern_usage": {},
            "unmatched_outputs": []
        }
        
        for output in llm_outputs:
            parsed = self.parse_with_explanation(output)
            
            if parsed["type"] == "idle":
                results["unmatched"] += 1
                results["unmatched_outputs"].append(output)
            else:
                results["matched"] += 1
                action_type = parsed["type"]
                results["pattern_usage"][action_type] = results["pattern_usage"].get(action_type, 0) + 1
        
        return results
    
    def suggest_new_patterns(self, unmatched_outputs: List[str]) -> List[Dict[str, str]]:
        """Suggest new patterns for unmatched outputs"""
        suggestions = []
        
        for output in unmatched_outputs:
            # Simple heuristics for pattern suggestions
            lower_output = output.lower()
            
            if any(word in lower_output for word in ["wait", "stand", "watch", "observe"]):
                suggestions.append({
                    "pattern": r"(?:wait|stand|watch|observe)",
                    "action_type": "idle",
                    "example": output
                })
            
            elif any(word in lower_output for word in ["think", "consider", "wonder"]):
                suggestions.append({
                    "pattern": r"(?:think|consider|wonder)",
                    "action_type": "think",
                    "example": output
                })
        
        return suggestions
    
    def export_pattern_analysis(self, filename: str = "pattern_analysis.md") -> None:
        """Export pattern analysis to markdown file"""
        with open(filename, "w", encoding='utf-8') as f:
            f.write("# LLM Output Pattern Analysis\n\n")
            
            f.write("## Current Patterns\n\n")
            for action_name, config in self.action_patterns.items():
                f.write(f"### {action_name}\n")
                f.write(f"**Game Action**: {config['game_action']}\n\n")
                f.write("**Patterns**:\n")
                for pattern in config["patterns"]:
                    f.write(f"- `{pattern}`\n")
                f.write("\n")
            
            f.write("## Test Results\n\n")
            f.write("Run `test_common_outputs()` to see pattern matching results.\n")


class ActionTypeAnalyzer:
    """Analyze action types and their distribution"""
    
    def __init__(self):
        self.action_counts: Dict[str, int] = {}
        self.action_examples: Dict[str, List[str]] = {}
    
    def record_action(self, action_type: str, original_text: str) -> None:
        """Record an action for analysis"""
        self.action_counts[action_type] = self.action_counts.get(action_type, 0) + 1
        
        if action_type not in self.action_examples:
            self.action_examples[action_type] = []
        
        if len(self.action_examples[action_type]) < 5:  # Keep max 5 examples
            self.action_examples[action_type].append(original_text)
    
    def get_distribution(self) -> Dict[str, float]:
        """Get action type distribution as percentages"""
        total = sum(self.action_counts.values())
        if total == 0:
            return {}
        
        return {
            action_type: (count / total) * 100
            for action_type, count in self.action_counts.items()
        }
    
    def print_analysis(self) -> None:
        """Print action analysis"""
        distribution = self.get_distribution()
        
        print("\n" + "="*50)
        print("ACTION TYPE ANALYSIS")
        print("="*50)
        
        for action_type, percentage in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
            count = self.action_counts[action_type]
            print(f"{action_type:15}: {count:3d} ({percentage:5.1f}%)")
            
            if action_type in self.action_examples:
                print("  Examples:")
                for example in self.action_examples[action_type][:3]:
                    print(f"    - {example[:60]}...")
                print()


# Global instances
global_parser = LLMOutputParser()
global_analyzer = ActionTypeAnalyzer()


def get_parser() -> LLMOutputParser:
    """Get global parser instance"""
    return global_parser


def get_analyzer() -> ActionTypeAnalyzer:
    """Get global analyzer instance"""
    return global_analyzer
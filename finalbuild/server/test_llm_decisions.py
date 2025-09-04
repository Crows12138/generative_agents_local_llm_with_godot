#!/usr/bin/env python3
"""
Test LLM decision making with clear prompt and response display
"""

import asyncio
import json
import time
from pathlib import Path
from gpt4all import GPT4All
from bar_state import SimpleStateReceiver as StateToPromptConverter


class LLMDecisionTester:
    """Test LLM decisions with detailed logging"""
    
    def __init__(self, model_name: str = "Llama-3.2-3B-Instruct-Q4_0.gguf"):
        # Initialize model
        model_path = Path(__file__).parent.parent.parent / "models" / "llms"
        self.model = GPT4All(
            model_name=model_name,
            model_path=str(model_path),
            allow_download=False,
            verbose=False
        )
        
        self.converter = StateToPromptConverter()
        self.test_results = []
    
    def test_scenario(self, scenario_name: str, state_dict: dict) -> dict:
        """Test a single scenario and capture all details"""
        
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*60}")
        
        # 1. Show input state
        print("\n[INPUT STATE FROM GODOT]")
        print(json.dumps(state_dict, indent=2))
        
        # 2. Convert to perception
        processed = self.converter.receive_godot_state(state_dict)
        print(f"\n[PERCEPTION]")
        print(f"{processed['perception']}")
        
        # 3. Create prompt
        prompt = f"""You are Bob, an experienced bartender.

{processed['perception']}

What single action do you take?
Choose ONE: serve_customer, clean_counter, clear_table, restock, take_break, observe

Respond with ONLY the action word."""
        
        print(f"\n[PROMPT TO LLM]")
        print("-" * 40)
        print(prompt)
        print("-" * 40)
        
        # 4. Get LLM response
        start_time = time.time()
        response = self.model.generate(prompt, max_tokens=150, temp=0.3)
        response_time = time.time() - start_time
        
        # Clean response
        response_clean = response.strip().lower()
        
        print(f"\n[LLM RAW RESPONSE]")
        print(f"'{response}'")
        
        # 5. Extract action
        valid_actions = ["serve_customer", "clean_counter", "clear_table", 
                        "restock", "take_break", "observe"]
        
        extracted_action = "observe"  # default
        for action in valid_actions:
            if action in response_clean:
                extracted_action = action
                break
        
        print(f"\n[EXTRACTED ACTION]")
        print(f"{extracted_action}")
        
        print(f"\n[RESPONSE TIME]")
        print(f"{response_time:.2f} seconds")
        
        # 6. Compare with priority suggestion
        suggested_action = processed['priority_action']
        matches_suggestion = extracted_action == suggested_action
        
        print(f"\n[ANALYSIS]")
        print(f"Suggested priority action: {suggested_action}")
        print(f"Matches suggestion: {matches_suggestion}")
        
        # Store result
        result = {
            "scenario": scenario_name,
            "input_state": state_dict,
            "perception": processed['perception'],
            "prompt": prompt,
            "raw_response": response,
            "extracted_action": extracted_action,
            "suggested_action": suggested_action,
            "matches_suggestion": matches_suggestion,
            "response_time": response_time
        }
        
        self.test_results.append(result)
        return result
    
    def run_all_tests(self):
        """Run all test scenarios"""
        
        print("\n" + "="*60)
        print("STARTING LLM DECISION TESTS")
        print("="*60)
        
        # Test scenarios
        scenarios = [
            {
                "name": "Normal Bar",
                "state": {
                    "counter_dirty": False,
                    "counter_has_customers": False,
                    "table_dirty": False,
                    "table_has_customers": False,
                    "shelf_low": False,
                    "shelf_empty": False
                }
            },
            {
                "name": "Dirty Counter Only",
                "state": {
                    "counter_dirty": True,
                    "counter_has_customers": False,
                    "table_dirty": False,
                    "table_has_customers": False,
                    "shelf_low": False,
                    "shelf_empty": False
                }
            },
            {
                "name": "Customers Waiting",
                "state": {
                    "counter_dirty": False,
                    "counter_has_customers": True,
                    "table_dirty": False,
                    "table_has_customers": False,
                    "shelf_low": False,
                    "shelf_empty": False
                }
            },
            {
                "name": "Multiple Problems",
                "state": {
                    "counter_dirty": True,
                    "counter_has_customers": True,
                    "table_dirty": True,
                    "table_has_customers": False,
                    "shelf_low": True,
                    "shelf_empty": False
                }
            },
            {
                "name": "Emergency - Empty Shelf",
                "state": {
                    "counter_dirty": False,
                    "counter_has_customers": False,
                    "table_dirty": False,
                    "table_has_customers": False,
                    "shelf_low": False,
                    "shelf_empty": True
                }
            }
        ]
        
        for scenario in scenarios:
            self.test_scenario(scenario["name"], scenario["state"])
            time.sleep(1)  # Small delay between tests
        
        # Save results
        self.save_results()
        
        # Print summary
        self.print_summary()
    
    def save_results(self):
        """Save test results to JSON"""
        
        output = {
            "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "Llama-3.2-3B-Instruct-Q4_0.gguf",
            "total_tests": len(self.test_results),
            "results": self.test_results,
            "summary": self.generate_summary()
        }
        
        with open("llm_decision_test_results.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        
        print(f"\n[RESULTS SAVED TO llm_decision_test_results.json]")
    
    def generate_summary(self) -> dict:
        """Generate test summary"""
        
        total = len(self.test_results)
        matches = sum(1 for r in self.test_results if r["matches_suggestion"])
        avg_time = sum(r["response_time"] for r in self.test_results) / total if total > 0 else 0
        
        action_counts = {}
        for r in self.test_results:
            action = r["extracted_action"]
            action_counts[action] = action_counts.get(action, 0) + 1
        
        return {
            "total_scenarios": total,
            "matches_priority": matches,
            "match_rate": f"{(matches/total*100):.1f}%" if total > 0 else "0%",
            "average_response_time": f"{avg_time:.2f}s",
            "action_distribution": action_counts
        }
    
    def print_summary(self):
        """Print test summary"""
        
        print("\n" + "="*60)
        print("TEST SUMMARY")
        print("="*60)
        
        summary = self.generate_summary()
        
        print(f"\nTotal Scenarios Tested: {summary['total_scenarios']}")
        print(f"Matches Priority Action: {summary['matches_priority']} ({summary['match_rate']})")
        print(f"Average Response Time: {summary['average_response_time']}")
        
        print("\nAction Distribution:")
        for action, count in summary['action_distribution'].items():
            print(f"  {action}: {count}")
        
        print("\nDetailed Results per Scenario:")
        for r in self.test_results:
            print(f"  {r['scenario']}: {r['extracted_action']} "
                  f"(Suggested: {r['suggested_action']}, "
                  f"Match: {'Yes' if r['matches_suggestion'] else 'No'})")


if __name__ == "__main__":
    print("Initializing LLM Decision Tester...")
    tester = LLMDecisionTester()
    tester.run_all_tests()
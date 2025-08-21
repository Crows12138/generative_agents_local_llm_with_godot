import sys
import os
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_system.flow_tracer import FlowTracer, get_tracer
from debug_system.llm_parser_study import LLMOutputParser, get_parser, get_analyzer


class InteractiveTester:
    """Interactive testing of LLM and action systems"""
    
    def __init__(self):
        self.tracer = get_tracer()
        self.parser = get_parser()
        self.analyzer = get_analyzer()
        self.test_scenarios = self._load_test_scenarios()
        
    def _load_test_scenarios(self) -> Dict[str, Dict[str, Any]]:
        """Load predefined test scenarios"""
        return {
            "morning": {
                "perception": "Bar is empty, counter needs cleaning, glasses scattered",
                "expected_action": "clean",
                "description": "Early morning bar opening scenario"
            },
            "customer": {
                "perception": "Alice sitting at bar, looking thirsty, waving at bartender",
                "expected_action": "serve",
                "description": "Customer service scenario"
            },
            "busy": {
                "perception": "Multiple customers waiting, drinks to make, noise level high",
                "expected_action": "prioritize",
                "description": "Busy bar management scenario"
            },
            "conversation": {
                "perception": "Regular customer Bob wants to chat about his day",
                "expected_action": "talk",
                "description": "Social interaction scenario"
            },
            "inventory": {
                "perception": "Running low on whiskey, need to check stock",
                "expected_action": "check",
                "description": "Inventory management scenario"
            }
        }
    
    def run_interactive_session(self) -> None:
        """Run interactive testing session"""
        print("""
        ╔════════════════════════════════════════╗
        ║   LLM Action System Interactive Tester ║
        ╠════════════════════════════════════════╣
        ║ Commands:                              ║
        ║  1. test <scenario> - Test scenario    ║
        ║  2. llm <text>     - Test LLM parsing  ║
        ║  3. flow           - Show flow trace   ║
        ║  4. scenarios      - List scenarios    ║
        ║  5. analyze        - Show analysis     ║
        ║  6. clear          - Clear traces      ║
        ║  7. help           - Show this help    ║
        ║  8. quit           - Exit              ║
        ╚════════════════════════════════════════╝
        """)
        
        while True:
            try:
                cmd = input("\n> ").strip().lower()
                
                if not cmd:
                    continue
                elif cmd.startswith("test"):
                    scenario = cmd.split()[1] if len(cmd.split()) > 1 else "default"
                    self.test_scenario(scenario)
                elif cmd.startswith("llm"):
                    text = cmd[4:].strip()
                    if text:
                        self.test_llm_parsing(text)
                    else:
                        print("Please provide text to parse: llm <text>")
                elif cmd == "flow":
                    self.show_complete_flow()
                elif cmd == "scenarios":
                    self.list_scenarios()
                elif cmd == "analyze":
                    self.show_analysis()
                elif cmd == "clear":
                    self.clear_traces()
                elif cmd == "help":
                    self.show_help()
                elif cmd == "quit":
                    print("Goodbye!")
                    break
                else:
                    print(f"Unknown command: {cmd}. Type 'help' for available commands.")
                    
            except KeyboardInterrupt:
                print("\nGoodbye!")
                break
            except Exception as e:
                print(f"Error: {e}")
    
    def test_scenario(self, scenario: str) -> None:
        """Test a specific scenario"""
        if scenario not in self.test_scenarios:
            print(f"Unknown scenario: {scenario}")
            print("Available scenarios:", ", ".join(self.test_scenarios.keys()))
            return
            
        data = self.test_scenarios[scenario]
        print(f"\n📍 Testing scenario: {scenario}")
        print(f"📝 Description: {data['description']}")
        print(f"📥 Perception: {data['perception']}")
        
        # Simulate complete flow
        self.tracer.clear_trace()
        
        # Step 1: Perception
        self.tracer.trace_perception("TestBot", data['perception'])
        
        # Step 2: Build LLM prompt (simulated)
        prompt = f"You are a bartender. Current situation: {data['perception']}. What should you do?"
        self.tracer.trace_llm_prompt(prompt)
        
        # Step 3: Simulate LLM response (predefined responses for testing)
        llm_responses = {
            "morning": "I should clean the bar counter and organize the glasses",
            "customer": "I need to go to Alice and ask what she'd like to drink",
            "busy": "I should prioritize the waiting customers and serve them quickly",
            "conversation": "I'll talk to Bob and listen to what he has to say",
            "inventory": "I should check the whiskey inventory and restock if needed"
        }
        
        llm_response = llm_responses.get(scenario, "I'll just stand here and observe")
        self.tracer.trace_llm_response(llm_response)
        
        # Step 4: Parse action
        parsed_action = self.parser.parse_with_explanation(llm_response)
        self.tracer.trace_action_parsing(llm_response, parsed_action)
        
        # Step 5: Simulate execution
        execution_result = {"status": "success", "message": f"Executed {parsed_action['type']} action"}
        self.tracer.trace_execution(parsed_action, execution_result)
        
        # Record for analysis
        self.analyzer.record_action(parsed_action['type'], llm_response)
        
        print(f"\n✅ Scenario test completed!")
        print(f"Expected: {data['expected_action']}, Got: {parsed_action['type']}")
    
    def test_llm_parsing(self, text: str) -> None:
        """Test LLM output parsing"""
        print(f"\n🧪 Testing LLM parsing for: '{text}'")
        
        result = self.parser.parse_with_explanation(text)
        self.analyzer.record_action(result['type'], text)
        
        print(f"\n📊 Parse result: {result}")
    
    def show_complete_flow(self) -> None:
        """Show complete flow trace"""
        print("\n🔄 Complete Flow Trace:")
        self.tracer.print_summary()
        
        # Ask if user wants detailed view
        response = input("\nShow detailed flow? (y/n): ").strip().lower()
        if response == 'y':
            self.tracer.save_flow_diagram("temp_flow.md")
            print("Detailed flow saved to temp_flow.md")
    
    def list_scenarios(self) -> None:
        """List available test scenarios"""
        print("\n📋 Available Test Scenarios:")
        print("="*40)
        for name, data in self.test_scenarios.items():
            print(f"{name:12}: {data['description']}")
    
    def show_analysis(self) -> None:
        """Show action analysis"""
        print("\n📈 Action Analysis:")
        self.analyzer.print_analysis()
    
    def clear_traces(self) -> None:
        """Clear all traces and analysis"""
        self.tracer.clear_trace()
        self.analyzer.action_counts.clear()
        self.analyzer.action_examples.clear()
        print("✅ All traces and analysis cleared!")
    
    def show_help(self) -> None:
        """Show detailed help"""
        print("""
        📖 Detailed Help:
        
        • test <scenario>  : Run predefined scenario tests
        • llm <text>       : Parse any LLM output text
        • flow             : Show current flow trace
        • scenarios        : List all available scenarios
        • analyze          : Show action type distribution
        • clear            : Clear all traces and data
        • help             : Show this help message
        • quit             : Exit the tester
        
        Examples:
        > test morning
        > llm I should clean the bar
        > analyze
        """)
    
    def batch_test(self, test_cases: list) -> Dict[str, Any]:
        """Run batch tests and return results"""
        results = {
            "total_tests": len(test_cases),
            "successful_parses": 0,
            "failed_parses": 0,
            "action_distribution": {}
        }
        
        for test_case in test_cases:
            parsed = self.parser.parse_with_explanation(test_case)
            action_type = parsed['type']
            
            if action_type != 'idle':
                results["successful_parses"] += 1
            else:
                results["failed_parses"] += 1
            
            results["action_distribution"][action_type] = results["action_distribution"].get(action_type, 0) + 1
            self.analyzer.record_action(action_type, test_case)
        
        return results
    
    def run_performance_test(self, iterations: int = 100) -> Dict[str, float]:
        """Run performance tests"""
        import time
        
        test_text = "I should walk to the bar counter to serve customers"
        
        # Warm up
        for _ in range(10):
            self.parser.parse_with_explanation(test_text)
        
        # Actual test
        start_time = time.time()
        for _ in range(iterations):
            self.parser.parse_with_explanation(test_text)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / iterations
        
        return {
            "total_time": total_time,
            "average_time": avg_time,
            "iterations": iterations,
            "operations_per_second": iterations / total_time
        }


def main():
    """Main entry point for interactive testing"""
    tester = InteractiveTester()
    tester.run_interactive_session()


if __name__ == "__main__":
    main()
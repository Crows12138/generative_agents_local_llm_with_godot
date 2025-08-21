#!/usr/bin/env python3
"""
Test script for the complete core loop analysis system
Tests integration with existing bar agents
"""

import sys
import os
from typing import Dict, Any

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_system.flow_tracer import get_tracer
from debug_system.llm_parser_study import get_parser, get_analyzer
from debug_system.performance_analyzer import get_performance_analyzer
from debug_system.interactive_tester import InteractiveTester


def test_flow_tracer():
    """Test the flow tracer functionality"""
    print("Testing Flow Tracer...")
    
    tracer = get_tracer()
    tracer.clear_trace()
    
    # Simulate a complete action cycle
    tracer.trace_perception("TestAgent", "Customer Alice is waiting at the bar")
    tracer.trace_llm_prompt("You are a bartender. Alice is waiting. What do you do?")
    tracer.trace_llm_response("I should go to Alice and ask what she wants to drink")
    
    parsed_action = {"type": "move", "target": "Alice", "action": "serve"}
    tracer.trace_action_parsing("I should go to Alice", parsed_action)
    
    execution_result = {"status": "success", "result": "Moved to Alice"}
    tracer.trace_execution(parsed_action, execution_result)
    
    tracer.print_summary()
    print("Flow Tracer test completed!\n")


def test_llm_parser():
    """Test the LLM output parser"""
    print("Testing LLM Parser...")
    
    parser = get_parser()
    analyzer = get_analyzer()
    
    # Test various LLM outputs
    test_outputs = [
        "I should walk to the bar counter",
        "Let me talk to Alice about her order", 
        "I need to clean the glasses",
        "Time to check the inventory",
        "Just standing here observing customers"
    ]
    
    for output in test_outputs:
        result = parser.parse_with_explanation(output)
        analyzer.record_action(result['type'], output)
    
    analyzer.print_analysis()
    print("LLM Parser test completed!\n")


def test_performance_analyzer():
    """Test the performance analyzer"""
    print("Testing Performance Analyzer...")
    
    perf_analyzer = get_performance_analyzer()
    
    # Start monitoring
    perf_analyzer.start_monitoring(interval=0.5)
    
    # Run several action cycles
    for i in range(3):
        print(f"Running cycle {i+1}...")
        timings = perf_analyzer.profile_action_cycle(f"TestAgent_{i}")
    
    # Stop monitoring
    perf_analyzer.stop_monitoring()
    
    # Generate analysis
    analysis = perf_analyzer.analyze_bottlenecks()
    print("\nBottleneck Analysis:")
    for phase, data in analysis.get("phase_analysis", {}).items():
        if phase != "total_cycle":
            print(f"  {phase}: {data['average']*1000:.1f}ms avg")
    
    print("Performance Analyzer test completed!\n")


def test_interactive_mode():
    """Test interactive mode briefly"""
    print("Testing Interactive Tester...")
    
    tester = InteractiveTester()
    
    # Run some batch tests
    test_cases = [
        "I should walk to the bar",
        "Let me serve the customer",
        "Time to clean up"
    ]
    
    results = tester.batch_test(test_cases)
    print(f"Batch test results: {results}")
    
    print("Interactive Tester test completed!\n")


def test_integration():
    """Test integration between all components"""
    print("Testing System Integration...")
    
    tracer = get_tracer()
    parser = get_parser()
    perf_analyzer = get_performance_analyzer()
    
    tracer.clear_trace()
    perf_analyzer.start_monitoring(interval=0.1)
    
    # Simulate integrated workflow
    agent_name = "IntegratedTestAgent"
    perception = "Bar is busy, multiple customers waiting, need to prioritize"
    
    # Step 1: Perception
    tracer.trace_perception(agent_name, perception)
    
    # Step 2: LLM Processing
    prompt = f"You are a bartender. Situation: {perception}. What should you do?"
    tracer.trace_llm_prompt(prompt)
    
    llm_response = "I should quickly serve the waiting customers in order"
    tracer.trace_llm_response(llm_response)
    
    # Step 3: Parse action
    parsed_action = parser.parse_with_explanation(llm_response)
    tracer.trace_action_parsing(llm_response, parsed_action)
    
    # Step 4: Execute
    execution_result = {"status": "success", "customers_served": 3}
    tracer.trace_execution(parsed_action, execution_result)
    
    # Stop monitoring and analyze
    perf_analyzer.stop_monitoring()
    
    print("\nIntegration Test Summary:")
    tracer.print_summary()
    
    perf_summary = perf_analyzer.get_summary()
    print(f"Performance: {perf_summary}")
    
    print("Integration test completed!\n")


def save_test_reports():
    """Save test reports to files"""
    print("Saving test reports...")
    
    tracer = get_tracer()
    perf_analyzer = get_performance_analyzer()
    
    # Save flow diagram
    tracer.save_flow_diagram("test_flow_results.md")
    
    # Save performance report
    perf_analyzer.generate_report("test_performance_report.md")
    
    # Save raw data
    perf_analyzer.export_raw_data("test_performance_data.json")
    
    print("Reports saved!")


def main():
    """Main test runner"""
    print("Starting Core Loop Analysis System Tests")
    print("=" * 60)
    
    try:
        # Run individual component tests
        test_flow_tracer()
        test_llm_parser()
        test_performance_analyzer()
        test_interactive_mode()
        
        # Run integration test
        test_integration()
        
        # Save reports
        save_test_reports()
        
        print("All tests completed successfully!")
        print("\nGenerated files:")
        print("- test_flow_results.md")
        print("- test_performance_report.md") 
        print("- test_performance_data.json")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
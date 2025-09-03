#!/usr/bin/env python3
"""
Integration demo showing how to use the debug dashboard with bar agents
This demonstrates real-world usage with the cozy bar demo
"""

import sys
import os
import time
import threading
from datetime import datetime

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_system.debug_dashboard import DebugDashboard
from debug_system.flow_tracer import get_tracer
from debug_system.performance_analyzer import get_performance_analyzer

def simulate_bar_integration():
    """Simulate integration with the cozy bar demo"""
    
    print("=== Bar Integration Demo ===")
    print("Simulating debug dashboard integration with cozy bar demo")
    print("This shows how to monitor real agent behavior\n")
    
    dashboard = DebugDashboard()
    tracer = get_tracer()
    perf_analyzer = get_performance_analyzer()
    
    # Start performance monitoring
    perf_analyzer.start_monitoring(interval=1.0)
    
    # Initialize bar agents (simulated)
    bar_agents = {
        "Alice_Bartender": {
            "role": "bartender",
            "position": (5, 2),
            "skills": ["serve_drinks", "clean_bar", "take_orders"],
            "status": "active"
        },
        "Bob_Customer": {
            "role": "customer", 
            "position": (3, 4),
            "order": "whiskey",
            "status": "waiting"
        },
        "Carol_Waiter": {
            "role": "waiter",
            "position": (7, 6), 
            "skills": ["serve_food", "clean_tables"],
            "status": "active"
        }
    }
    
    # Add agents to dashboard
    for agent_name, info in bar_agents.items():
        dashboard.add_agent(agent_name, info["position"], info["status"])
        dashboard.add_event("init", agent_name, f"{info['role'].title()} initialized", "info")
    
    print("Agents initialized. Starting simulation...")
    print("Press Ctrl+C to stop\n")
    
    try:
        # Simulate a realistic bar scenario
        simulation_steps = [
            # Morning setup (0-5 seconds)
            (1, "Alice_Bartender", "clean", "bar_counter", "Morning setup routine"),
            (2, "Carol_Waiter", "check", "tables", "Checking table cleanliness"),
            (3, "Alice_Bartender", "check", "inventory", "Checking drink inventory"),
            
            # Customer service (5-15 seconds)
            (5, "Bob_Customer", "move", "bar_counter", "Bob approaches the bar"),
            (6, "Alice_Bartender", "talk", "Bob_Customer", "Taking Bob's order"),
            (7, "Alice_Bartender", "work", "prepare_drink", "Preparing whiskey for Bob"),
            (8, "Carol_Waiter", "clean", "table_3", "Cleaning table 3"),
            
            # Service completion (15-25 seconds)
            (10, "Alice_Bartender", "interact", "Bob_Customer", "Serving whiskey to Bob"),
            (11, "Bob_Customer", "idle", None, "Bob enjoys his drink"),
            (12, "Carol_Waiter", "move", "kitchen", "Going to kitchen"),
            (13, "Alice_Bartender", "clean", "glasses", "Cleaning used glasses"),
            
            # Busy period simulation (25-35 seconds)
            (15, "Alice_Bartender", "work", "multiple_orders", "Handling multiple drink orders"),
            (16, "Carol_Waiter", "work", "food_service", "Serving food to table 2"),
            (18, "Bob_Customer", "talk", "Alice_Bartender", "Bob compliments the service"),
            
            # Error scenarios (35-45 seconds)
            (20, "Alice_Bartender", "error", "equipment", "Coffee machine malfunction"),
            (21, "Carol_Waiter", "work", "help_bartender", "Carol helps with equipment"),
            (22, "Alice_Bartender", "work", "repair", "Fixing coffee machine"),
            
            # Recovery and wrap-up (45+ seconds)
            (25, "Alice_Bartender", "idle", None, "Taking a short break"),
            (26, "Bob_Customer", "move", "exit", "Bob preparing to leave"),
            (27, "Carol_Waiter", "clean", "final_cleanup", "Final table cleanup"),
            (28, "Alice_Bartender", "work", "closing_prep", "Preparing for closing")
        ]
        
        start_time = time.time()
        step_index = 0
        cycle_count = 0
        
        while step_index < len(simulation_steps):
            current_time = time.time() - start_time
            
            # Execute simulation steps
            if step_index < len(simulation_steps):
                step_time, agent, action, target, description = simulation_steps[step_index]
                
                if current_time >= step_time:
                    # Clear previous trace
                    tracer.clear_trace()
                    
                    # Trace the complete action flow
                    tracer.trace_perception(agent, description)
                    
                    # Simulate LLM processing
                    llm_prompt = f"As a {bar_agents[agent]['role']}, you observe: {description}. What should you do?"
                    tracer.trace_llm_prompt(llm_prompt)
                    
                    llm_response = f"I should {action}"
                    if target:
                        llm_response += f" with {target}"
                    tracer.trace_llm_response(llm_response)
                    
                    # Parse action
                    parsed_action = {"type": action, "target": target, "original": llm_response}
                    tracer.trace_action_parsing(llm_response, parsed_action)
                    
                    # Execute action
                    execution_result = {"status": "success" if action != "error" else "failed", "description": description}
                    tracer.trace_execution(parsed_action, execution_result)
                    
                    # Update dashboard
                    dashboard.update_agent(agent, action=action, target=target)
                    
                    # Add appropriate event
                    severity = "error" if action == "error" else "warning" if "malfunction" in description else "info"
                    dashboard.add_event("simulation", agent, description, severity)
                    
                    # Profile performance
                    timings = perf_analyzer.profile_action_cycle(agent)
                    
                    step_index += 1
                    cycle_count += 1
            
            # Show dashboard state every 3 seconds
            if int(current_time) % 3 == 0:
                print(f"\n--- Simulation Time: {int(current_time)}s ---")
                dashboard.console_fallback()
                
            time.sleep(0.5)
        
        print(f"\nSimulation completed after {cycle_count} cycles!")
        
    except KeyboardInterrupt:
        print("\nSimulation stopped by user")
    
    finally:
        # Stop monitoring and save reports
        perf_analyzer.stop_monitoring()
        
        # Save comprehensive reports
        dashboard.save_dashboard_report("bar_integration_dashboard.md")
        tracer.save_flow_diagram("bar_integration_flow.md")
        perf_analyzer.generate_report("bar_integration_performance.md")
        
        print("\nIntegration demo complete!")
        print("Generated reports:")
        print("  - bar_integration_dashboard.md")
        print("  - bar_integration_flow.md") 
        print("  - bar_integration_performance.md")


def show_integration_guide():
    """Show how to integrate the dashboard with existing code"""
    
    print("=== Integration Guide ===\n")
    
    print("1. Basic Integration:")
    print("```python")
    print("from debug_system.debug_dashboard import DebugDashboard")
    print("")
    print("# Initialize dashboard")
    print("dashboard = DebugDashboard()")
    print("")
    print("# Add your agents")
    print("dashboard.add_agent('agent_name', position=(x, y), status='active')")
    print("")
    print("# Update agent states during execution")
    print("dashboard.update_agent('agent_name', action='move', target='destination')")
    print("")
    print("# Add events for important occurrences")
    print("dashboard.add_event('type', 'agent_name', 'description', 'severity')")
    print("```")
    
    print("\n2. Performance Monitoring:")
    print("```python")
    print("from debug_system.performance_analyzer import get_performance_analyzer")
    print("")
    print("perf_analyzer = get_performance_analyzer()")
    print("perf_analyzer.start_monitoring()")
    print("")
    print("# Profile agent cycles")
    print("timings = perf_analyzer.profile_action_cycle(agent_name)")
    print("```")
    
    print("\n3. Flow Tracing:")
    print("```python")
    print("from debug_system.flow_tracer import get_tracer")
    print("")
    print("tracer = get_tracer()")
    print("tracer.trace_perception(agent_name, perception_data)")
    print("tracer.trace_llm_response(llm_output)")
    print("tracer.trace_execution(action, result)")
    print("```")
    
    print("\n4. Live Dashboard:")
    print("```python")
    print("# Start real-time monitoring")
    print("dashboard.start_monitoring(update_interval=1.0)")
    print("")
    print("# Or use console fallback")
    print("dashboard.console_fallback()")
    print("```")
    
    print("\n5. Report Generation:")
    print("```python")
    print("# Save comprehensive reports")
    print("dashboard.save_dashboard_report('my_report.md')")
    print("tracer.save_flow_diagram('flow_diagram.md')")
    print("perf_analyzer.generate_report('performance.md')")
    print("```")


def main():
    """Main function"""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            simulate_bar_integration()
        elif sys.argv[1] == "--guide":
            show_integration_guide()
        elif sys.argv[1] == "--help":
            print("Bar Integration Demo")
            print("Options:")
            print("  --demo  : Run bar integration simulation")
            print("  --guide : Show integration guide")
            print("  --help  : Show this help")
        else:
            print("Unknown option. Use --help for available options.")
    else:
        print("Bar Integration Demo")
        print("\nChoose an option:")
        print("  python bar_integration_demo.py --demo  : Run integration demo")
        print("  python bar_integration_demo.py --guide : Show integration guide")
        print("  python bar_integration_demo.py --help  : Show help")


if __name__ == "__main__":
    main()
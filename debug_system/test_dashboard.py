#!/usr/bin/env python3
"""
Test script for the debug dashboard
Demonstrates real-time monitoring capabilities
"""

import sys
import os
import time
import threading
import random
from datetime import datetime

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_system.debug_dashboard import DebugDashboard


def simulate_bar_scenario(dashboard: DebugDashboard):
    """Simulate a realistic bar scenario with multiple agents"""
    
    # Initialize bar agents
    agents = {
        "Bartender_Alice": {"role": "bartender", "position": (5, 2)},
        "Customer_Bob": {"role": "customer", "position": (3, 4)},
        "Customer_Carol": {"role": "customer", "position": (7, 4)},
        "Waiter_Dave": {"role": "waiter", "position": (6, 6)}
    }
    
    # Add agents to dashboard
    for agent_name, info in agents.items():
        dashboard.add_agent(agent_name, info["position"], "active")
        dashboard.add_event("system", agent_name, f"{info['role'].title()} {agent_name} entered the bar", "info")
    
    # Simulate agent behaviors
    scenarios = [
        # Morning setup
        {"time": 0, "agent": "Bartender_Alice", "action": "clean", "target": "bar_counter", "description": "Morning cleaning routine"},
        {"time": 1, "agent": "Waiter_Dave", "action": "check", "target": "tables", "description": "Checking table setup"},
        
        # Customers arrive
        {"time": 3, "agent": "Customer_Bob", "action": "move", "target": "bar_counter", "description": "Bob approaches the bar"},
        {"time": 4, "agent": "Bartender_Alice", "action": "talk", "target": "Customer_Bob", "description": "Taking Bob's order"},
        
        # Busy period
        {"time": 6, "agent": "Customer_Carol", "action": "move", "target": "bar_counter", "description": "Carol joins the queue"},
        {"time": 7, "agent": "Bartender_Alice", "action": "work", "target": "drink_preparation", "description": "Preparing drinks"},
        {"time": 8, "agent": "Waiter_Dave", "action": "move", "target": "Customer_Carol", "description": "Dave assists Carol"},
        
        # Service completion
        {"time": 10, "agent": "Bartender_Alice", "action": "interact", "target": "Customer_Bob", "description": "Serving Bob's drink"},
        {"time": 12, "agent": "Customer_Bob", "action": "idle", "target": None, "description": "Bob enjoys his drink"},
        {"time": 13, "agent": "Waiter_Dave", "action": "work", "target": "table_service", "description": "Cleaning tables"},
        
        # Error scenario
        {"time": 15, "agent": "Bartender_Alice", "action": "error", "target": "equipment", "description": "Coffee machine malfunction"},
        {"time": 16, "agent": "Bartender_Alice", "action": "work", "target": "repair", "description": "Fixing equipment"},
        
        # Recovery
        {"time": 18, "agent": "Bartender_Alice", "action": "idle", "target": None, "description": "Equipment fixed, taking a break"},
        {"time": 20, "agent": "Customer_Carol", "action": "talk", "target": "Waiter_Dave", "description": "Carol compliments the service"}
    ]
    
    start_time = time.time()
    scenario_index = 0
    
    while scenario_index < len(scenarios) and dashboard.is_running:
        current_time = time.time() - start_time
        
        if scenario_index < len(scenarios) and current_time >= scenarios[scenario_index]["time"]:
            scenario = scenarios[scenario_index]
            
            # Update agent state
            dashboard.update_agent(
                scenario["agent"],
                action=scenario["action"],
                target=scenario["target"]
            )
            
            # Add event with appropriate severity
            severity = "error" if scenario["action"] == "error" else "warning" if "malfunction" in scenario["description"] else "info"
            dashboard.add_event("scenario", scenario["agent"], scenario["description"], severity)
            
            # Simulate random position changes for movement
            if scenario["action"] == "move":
                new_pos = (random.randint(1, 10), random.randint(1, 8))
                dashboard.update_agent(scenario["agent"], position=new_pos)
            
            scenario_index += 1
        
        # Add some random background activity
        if random.random() < 0.3:  # 30% chance per iteration
            agent_name = random.choice(list(agents.keys()))
            background_actions = ["observe", "wait", "think", "prepare"]
            action = random.choice(background_actions)
            dashboard.update_agent(agent_name, action=action)
        
        time.sleep(0.5)
    
    # Scenario complete
    dashboard.add_event("system", "Simulation", "Bar scenario simulation completed", "info")


def test_performance_integration(dashboard: DebugDashboard):
    """Test performance monitoring integration"""
    
    # Simulate some performance data
    from debug_system.performance_analyzer import get_performance_analyzer
    perf_analyzer = get_performance_analyzer()
    
    agents = ["Alice", "Bob", "Carol"]
    
    for i in range(10):
        agent = random.choice(agents)
        
        # Profile some cycles
        timings = perf_analyzer.profile_action_cycle(f"TestAgent_{i}")
        
        # Update dashboard with performance event
        avg_time = timings.get("total_cycle", 0) * 1000
        if avg_time > 100:
            dashboard.add_event("performance", agent, f"Slow cycle detected: {avg_time:.1f}ms", "warning")
        else:
            dashboard.add_event("performance", agent, f"Normal cycle: {avg_time:.1f}ms", "info")
        
        time.sleep(0.2)


def test_basic_dashboard():
    """Test basic dashboard functionality"""
    print("Testing basic dashboard functionality...")
    
    dashboard = DebugDashboard()
    
    # Add some test agents
    dashboard.add_agent("TestAgent1", (1, 1), "active")
    dashboard.add_agent("TestAgent2", (2, 2), "idle")
    dashboard.add_agent("TestAgent3", (3, 3), "error")
    
    # Add some test events
    dashboard.add_event("test", "TestAgent1", "Agent initialized successfully", "info")
    dashboard.add_event("test", "TestAgent2", "Agent waiting for input", "warning")
    dashboard.add_event("test", "TestAgent3", "Agent encountered an error", "error")
    
    # Update some agent states
    dashboard.update_agent("TestAgent1", action="move", target="destination_a")
    dashboard.update_agent("TestAgent2", action="work", target="task_b")
    
    # Save a test report
    dashboard.save_dashboard_report("test_dashboard_report.md")
    
    print("Basic test completed. Report saved to test_dashboard_report.md")


def test_full_dashboard_simulation():
    """Test full dashboard with live simulation"""
    print("Starting full dashboard simulation...")
    print("This will run a live dashboard with simulated bar scenario")
    print("Press Ctrl+C to stop")
    
    dashboard = DebugDashboard()
    
    # Start bar scenario simulation in background
    sim_thread = threading.Thread(target=simulate_bar_scenario, args=(dashboard,))
    sim_thread.daemon = True
    sim_thread.start()
    
    # Start performance monitoring in background
    perf_thread = threading.Thread(target=test_performance_integration, args=(dashboard,))
    perf_thread.daemon = True
    perf_thread.start()
    
    try:
        # Start the live dashboard
        dashboard.start_monitoring(update_interval=0.5)
    except KeyboardInterrupt:
        print("\nStopping simulation...")
        dashboard.stop_monitoring()
        dashboard.save_dashboard_report("full_simulation_report.md")
        print("Simulation complete. Report saved to full_simulation_report.md")


def main():
    """Main test runner"""
    if len(sys.argv) > 1:
        if sys.argv[1] == "--basic":
            test_basic_dashboard()
        elif sys.argv[1] == "--full":
            test_full_dashboard_simulation()
        elif sys.argv[1] == "--help":
            print("Dashboard Test Options:")
            print("  --basic : Test basic dashboard functionality")
            print("  --full  : Run full live dashboard simulation")
            print("  --help  : Show this help")
        else:
            print("Unknown option. Use --help for available options.")
    else:
        print("Dashboard Test Script")
        print("Options:")
        print("  --basic : Test basic functionality")
        print("  --full  : Run full simulation")
        print("  --help  : Show help")


if __name__ == "__main__":
    main()
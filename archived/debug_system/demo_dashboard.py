#!/usr/bin/env python3
"""
Demo script for debug dashboard
Shows console fallback mode when rich is not available
"""

import sys
import os
import time
import random

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_system.debug_dashboard import DebugDashboard


def run_demo():
    """Run a simple demo of the dashboard"""
    
    print("=== Debug Dashboard Demo ===")
    print("This demo shows the dashboard in console fallback mode")
    print("Press Ctrl+C to stop\n")
    
    dashboard = DebugDashboard()
    
    # Add some demo agents
    agents = [
        ("Alice_Bartender", (5, 2), "active"),
        ("Bob_Customer", (3, 4), "active"),
        ("Carol_Waiter", (7, 6), "idle")
    ]
    
    for name, pos, status in agents:
        dashboard.add_agent(name, pos, status)
    
    # Simulate some activity
    actions = ["move", "talk", "work", "clean", "serve", "wait"]
    targets = ["bar", "table", "kitchen", "customer", "equipment"]
    
    try:
        for cycle in range(20):
            print(f"\n--- Cycle {cycle + 1} ---")
            
            # Update agents randomly
            for name, _, _ in agents:
                if random.random() < 0.7:  # 70% chance to act
                    action = random.choice(actions)
                    target = random.choice(targets) if action != "wait" else None
                    new_pos = (random.randint(1, 10), random.randint(1, 8))
                    
                    dashboard.update_agent(name, 
                                         action=action, 
                                         target=target,
                                         position=new_pos)
                    
                    # Add some events
                    if random.random() < 0.3:  # 30% chance for special event
                        severity = random.choice(["info", "warning", "error"])
                        descriptions = {
                            "info": f"{name} completed task successfully",
                            "warning": f"{name} encountered minor issue",
                            "error": f"{name} failed to complete action"
                        }
                        dashboard.add_event("demo", name, descriptions[severity], severity)
            
            # Show dashboard state (console fallback)
            dashboard.console_fallback()
            
            time.sleep(2)  # Wait 2 seconds between updates
            
    except KeyboardInterrupt:
        print("\n\nDemo stopped by user")
    
    # Save final report
    dashboard.save_dashboard_report("demo_dashboard_report.md")
    print("Demo complete! Report saved to demo_dashboard_report.md")


def show_dashboard_features():
    """Show dashboard features"""
    
    print("=== Debug Dashboard Features ===\n")
    
    print("Core Features:")
    print("  - Real-time agent monitoring")
    print("  - Event logging with severity levels")
    print("  - Performance metrics integration")
    print("  - Action distribution analysis")
    print("  - Rich terminal UI (when available)")
    print("  - Console fallback mode")
    print("  - Report generation")
    
    print("\nDashboard Sections:")
    print("  - Header: Time, agent counts, system status")
    print("  - Agents: Name, position, action, target, status")
    print("  - Events: Real-time event log with timestamps")
    print("  - Performance: Cycle times, memory, CPU usage")
    print("  - Actions: Distribution of agent actions")
    print("  - Footer: Control commands and shortcuts")
    
    print("\nUsage Modes:")
    print("  - Interactive monitoring with rich UI")
    print("  - Console output for basic systems")
    print("  - Simulation mode with test data")
    print("  - Integration with existing agent systems")
    
    print("\nPerformance Integration:")
    print("  - LLM response time tracking")
    print("  - Action parsing performance")
    print("  - Memory and CPU monitoring")
    print("  - Bottleneck identification")
    
    print("\nReport Generation:")
    print("  - Markdown reports with agent states")
    print("  - Event logs with timestamps")
    print("  - Performance summaries")
    print("  - Action distribution analysis")


def main():
    """Main demo function"""
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--demo":
            run_demo()
        elif sys.argv[1] == "--features":
            show_dashboard_features()
        elif sys.argv[1] == "--help":
            print("Debug Dashboard Demo")
            print("Options:")
            print("  --demo     : Run interactive demo")
            print("  --features : Show dashboard features")
            print("  --help     : Show this help")
        else:
            print("Unknown option. Use --help for available options.")
    else:
        print("Debug Dashboard Demo")
        print("\nChoose an option:")
        print("  python demo_dashboard.py --demo     : Run interactive demo")
        print("  python demo_dashboard.py --features : Show dashboard features")
        print("  python demo_dashboard.py --help     : Show help")


if __name__ == "__main__":
    main()
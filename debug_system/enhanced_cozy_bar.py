#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced cozy_bar demo with detailed debugging output
"""
import sys
import os
import inspect
import json
from datetime import datetime
from typing import Dict, Any

# Set console encoding to UTF-8
if sys.platform == "win32":
    import codecs
    sys.stdout = codecs.getwriter("utf-8")(sys.stdout.detach())
    sys.stderr = codecs.getwriter("utf-8")(sys.stderr.detach())

# Add cozy_bar_demo path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'cozy_bar_demo'))

def debug_trace(message: str, data: Any = None):
    """Ultra-detailed debugging information"""
    frame = inspect.currentframe().f_back
    timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    
    print(f"""
{'='*60}
🔍 [{timestamp}] DEBUG: {message}
📍 Location: {os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}
📦 Function: {frame.f_code.co_name}
""")
    
    if data:
        if isinstance(data, dict):
            print(f"📊 Data:")
            for key, value in data.items():
                print(f"   {key}: {value}")
        else:
            print(f"📊 Data: {data}")
    print('='*60)

def run_enhanced_cozy_bar():
    """Run enhanced cozy_bar with detailed debugging"""
    debug_trace("Starting cozy_bar_demo")
    
    try:
        # Import core modules
        debug_trace("Importing modules")
        from core.bar_agents import BarAgent, BarSimulation
        from core.bar_renderer import BarRenderer
        debug_trace("Module import successful")
        
        # Create simulation
        config_path = "../cozy_bar_demo/config/room_config.json"
        debug_trace("Loading configuration", {"config_path": config_path})
        
        # Check if config exists
        if not os.path.exists(config_path):
            debug_trace("Config file not found, using default settings")
            # Create basic simulation without renderer
            simulation = BarSimulation()
            
            # Create agents manually
            agents_data = {
                "Bob": ("bartender", (5, 3)),
                "Charlie": ("regular customer", (2, 4)),
                "Sam": ("musician", (8, 2))
            }
            
            for name, (role, position) in agents_data.items():
                agent = BarAgent(name, role, position)
                simulation.add_agent(agent)
                debug_trace(f"Created agent {name}", {
                    "role": role,
                    "position": position,
                    "initial_status": agent.get_status()
                })
        else:
            # Use renderer if config exists
            renderer = BarRenderer(config_path)
            simulation = BarSimulation()
            
            debug_trace("Creating agents from config")
            # Add agents from config
            npc_roles = renderer.room["npc_roles"]
            spawn_points = renderer.room["spawn_points"]
            
            for name, role in npc_roles.items():
                position = spawn_points[name]
                agent = BarAgent(name, role.split(" - ")[0], position)
                simulation.add_agent(agent)
                debug_trace(f"Created agent {name}", {
                    "role": role,
                    "position": position,
                    "initial_status": agent.get_status()
                })
        
        debug_trace("Starting simulation loop")
        
        # Run simulation steps with detailed monitoring
        for step in range(3):
            debug_trace(f"=== SIMULATION STEP {step+1} ===")
            
            # Let each agent act
            for name, agent in simulation.agents.items():
                debug_trace(f"Agent {name} starting action")
                
                # Record pre-action state
                before_status = agent.get_status()
                debug_trace(f"{name} pre-action state", before_status)
                
                # Execute specific action
                try:
                    action_result = agent.bar_specific_actions()
                    debug_trace(f"{name} performed action", {"action": action_result})
                except Exception as e:
                    debug_trace(f"Error in {name} action", {"error": str(e)})
                
                # Generate dialogue
                try:
                    dialogue = agent.generate_bar_dialogue("general conversation", "environment")
                    debug_trace(f"{name} generated dialogue", {"dialogue": dialogue})
                except Exception as e:
                    debug_trace(f"Error in {name} dialogue", {"error": str(e)})
                
                # Record post-action state
                after_status = agent.get_status()
                debug_trace(f"{name} post-action state", after_status)
                
                # Show state changes
                state_changes = {}
                for key in before_status.keys():
                    if before_status[key] != after_status[key]:
                        state_changes[key] = {
                            "before": before_status[key],
                            "after": after_status[key]
                        }
                
                if state_changes:
                    debug_trace(f"{name} state changes detected", state_changes)
            
            # Simulate time passage
            debug_trace("Simulating time passage")
            simulation.simulate_time_passage(5)
            
            # Show recent events
            recent_events = simulation.get_recent_events(10)
            debug_trace("Recent events", {"events": recent_events})
            
            # Show scene description
            scene_desc = simulation.get_scene_description()
            debug_trace("Current scene", {"description": scene_desc})
            
            print(f"\n🎭 SCENE AFTER STEP {step+1}:")
            print(scene_desc)
            print("\n📰 RECENT EVENTS:")
            for event in recent_events:
                print(f"  • {event}")
            
            # Agent relationships and interactions
            print(f"\n👥 AGENT RELATIONSHIPS:")
            for name, agent in simulation.agents.items():
                if hasattr(agent, 'relationships') and agent.relationships:
                    print(f"  {name}: {agent.relationships}")
            
            if step < 2:  # Don't pause on last step
                input("\n⏸️  Press Enter to continue to next step...")
        
        debug_trace("Simulation completed successfully")
        
    except Exception as e:
        debug_trace(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🍺 Enhanced Cozy Bar Debugging System")
    print("=" * 50)
    run_enhanced_cozy_bar()
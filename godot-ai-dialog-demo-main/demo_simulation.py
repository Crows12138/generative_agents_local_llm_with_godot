import sys
import json
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import List, Tuple

# Force unbuffered output
sys.stdout = sys.__stdout__
sys.stderr = sys.__stderr__

from basic_functions.persona import Persona
from ai_service.ai_service import set_active_model, get_active_model
from basic_functions.maze import Maze, Area, Object
from basic_functions.perception.describe import describe_cell
from basic_functions.perception.perceive import perceive
from basic_functions.memory.memory import Memory
from basic_functions.time_manager import GameTimeManager, TimeSettings
from basic_functions.executor.executor import SimpleExecutor
from basic_functions.decider.decider import ActionIntent
from basic_functions.plan.enhanced_plan import generate_enhanced_daily_schedule, get_current_task_from_schedule
from basic_functions.decider.optimized_decider import OptimizedDecider


def load_settings(filename: str = "settings_demo_simulation.json") -> Tuple[Maze, List[Persona]]:
    """Build the world and agents from ``filename``."""
    with open(filename, encoding="utf-8") as f:
        cfg = json.load(f)

    maze_cfg = cfg["maze"]
    maze = Maze(maze_cfg["width"], maze_cfg["height"])

    for area in maze_cfg.get("areas", []):
        maze.add_area(
            Area(area["name"], area["x1"], area["y1"], area["x2"], area["y2"]),
            override=area.get("override", False),
        )

    for obj in maze_cfg.get("objects", []):
        maze.place_object(
            Object(
                obj["name"],
                obj["x"],
                obj["y"],
                description=obj.get("description", ""),
            ),
            int(obj["x"]),
            int(obj["y"]),
            0,
        )

    agents = []
    for a in cfg.get("agents", []):
        agent = Persona(a["name"], tuple(a["initial_location"]))
        agent.long_term_goals = a.get("long_term_goals", [])
        
        # Set personality description from personality settings if available
        personality = a.get("personality", None)
        if personality and "self_identification" in personality:
            agent.personality_description = personality["self_identification"]
        elif personality and "traits" in personality:
            # Fallback: create description from traits
            traits = personality["traits"]
            agent.personality_description = f"I am {a['name']}, a person who is {', '.join(traits)}."
        
        maze.place_agent(agent, int(agent.x), int(agent.y), 0)
        agents.append(agent)

    return maze, agents


class ActionDrivenSimulation:
    """Action-driven simulation where characters think after completing actions."""
    
    def __init__(self, maze: Maze, agents: List[Persona]):
        self.maze = maze
        self.agents = agents
        self.executor = SimpleExecutor()
        self.deciders = {}
        self.running = True
        
        # Initialize optimized deciders for each agent
        for agent in agents:
            self.deciders[agent.name] = OptimizedDecider()
        
        # Initialize daily schedules
        self.daily_schedules = {}
        self._generate_daily_schedules()
        
        # Action queue for processing
        self.action_queue = queue.Queue()
        
        # Start action processing thread
        self.action_thread = threading.Thread(target=self._action_worker, daemon=True)
        self.action_thread.start()
    
    def _generate_daily_schedules(self):
        """Generate enhanced daily schedules for all agents."""
        print("[DEBUG] Skipping schedule generation for faster startup")
        # Skip schedule generation for now to avoid LLM calls
        for agent in self.agents:
            self.daily_schedules[agent.name] = []
            print(f"[INFO] {agent.name}: No schedule (simplified mode)")
    
    def _action_worker(self):
        """Background worker for processing actions."""
        print("[DEBUG] Action worker started")
        while self.running:
            try:
                # Process actions for each agent
                for agent in self.agents:
                    if not self.running:
                        break
                    
                    print(f"[DEBUG] Processing agent: {agent.name}")
                    
                    # Get current task from schedule
                    current_time = datetime.now().strftime("%H:%M")
                    current_task = get_current_task_from_schedule(
                        self.daily_schedules.get(agent.name, []), 
                        current_time
                    )
                    
                    # Get surroundings
                    x, y, z = agent.location
                    surroundings = self._get_surroundings_description(x, y)
                    
                    # Get recent memories
                    recent_memories = agent.memory.entries[-3:] if len(agent.memory.entries) >= 3 else agent.memory.entries
                    
                    # Decide next action using optimized decider
                    decider = self.deciders[agent.name]
                    action_data = decider.decide(
                        persona_name=agent.name,
                        location=agent.location,
                        personality=agent.personality_description,
                        surroundings=surroundings,
                        memories=recent_memories,
                        goals="; ".join(agent.long_term_goals),
                        current_task=current_task,
                        previous_action=agent.previous_action,
                    )
                    
                    # Execute action
                    intent = ActionIntent(action_data["action"], action_data.get("target"))
                    target_str = action_data.get("target", "")
                    target_display = f" {target_str}" if target_str else ""
                    print(f"\n[ACTION] {agent.name} decides: {action_data['action']}{target_display}")
                    
                    # Execute the action
                    try:
                        self.executor.execute(intent, agent, self.maze)
                    except Exception as e:
                        print(f"[ERROR] Error executing action for {agent.name}: {e}")
                        # Add failure to memory
                        agent.memory.add(
                            text=f"Failed to execute action: {action_data['action']} {target_str}",
                            embedding=[],
                            location=agent.location,
                            event_type="error",
                            importance=0.3,
                            metadata={
                                "action": action_data['action'],
                                "target": target_str,
                                "error": str(e)
                            },
                            ttl=None,
                        )
                    
                    # Check if should think after action
                    if decider.should_think_after_action(action_data):
                        self._trigger_thinking(agent, action_data)
                    
                    # Small delay between actions
                    time.sleep(2)
                
                # Longer delay between rounds
                time.sleep(5)
                
            except Exception as e:
                print(f"[ERROR] Action processing error: {e}")
                time.sleep(1)
    
    def _get_surroundings_description(self, x: int, y: int) -> str:
        """Get description of surroundings at given location."""
        try:
            nearby_entities = self.maze.spatial.nearby(x, y, 2.0, 0.0)
            objects = [e.name for e in nearby_entities if hasattr(e, 'name') and not isinstance(e, Persona)]
            if objects:
                return f"Nearby objects: {', '.join(objects[:5])}"
            else:
                return "Open area"
        except:
            return "Unknown surroundings"
    
    def _trigger_thinking(self, agent: Persona, action_data: dict):
        """Trigger thinking after an action."""
        try:
            # Get result description from last memory entry
            result_description = "Action completed"
            if agent.memory.entries:
                last_entry = agent.memory.entries[-1]
                result_description = last_entry.text[:100] + "..." if len(last_entry.text) > 100 else last_entry.text
            
            # Generate reflection
            decider = self.deciders[agent.name]
            reflection = decider.generate_reflection(agent.name, action_data, result_description)
            
            # Add reflection to memory
            agent.memory.add(
                text=f"Reflection: {reflection}",
                embedding=[],
                event_type="reflection",
                importance=0.6,
                metadata={
                    "triggered_by": action_data["action"],
                    "target": action_data.get("target"),
                },
                ttl=None,
            )
            
            print(f"[REFLECT] {agent.name} reflects: {reflection}")
            
        except Exception as e:
            print(f"[ERROR] Error generating reflection for {agent.name}: {e}")
    
    def trigger_dialogue(self, speaker_name: str, target_name: str, custom_message: str = ""):
        """Trigger a dialogue between two entities."""
        speaker = None
        for agent in self.agents:
            if agent.name.lower() == speaker_name.lower():
                speaker = agent
                break
        
        if speaker is None:
            print(f"[ERROR] Speaker '{speaker_name}' not found!")
            return
        
        # Create dialogue intent
        intent = ActionIntent("talk_to", target_name)
        
        # Execute dialogue
        print(f"\n[TALK] {speaker_name} starts talking to {target_name}...")
        self.executor.execute(intent, speaker, self.maze)
        
        # Get the conversation from memory
        if speaker.memory.entries:
            last_entry = speaker.memory.entries[-1]
            if last_entry.event_type == "talk":
                print(f"[DIALOGUE] Conversation content:")
                print(f"   {last_entry.text}")
                if "metadata" in last_entry.__dict__ and last_entry.metadata:
                    print(f"   [META] Metadata: {last_entry.metadata}")
    
    def list_available_targets(self):
        """List all available dialogue targets."""
        targets = []
        
        # Add agents
        for agent in self.agents:
            targets.append(f"[CHAR] {agent.name} (Character)")
        
        # Add objects by scanning the maze
        for x in range(self.maze.width):
            for y in range(self.maze.height):
                for z in range(self.maze.depth):
                    entities = self.maze.spatial.nearby(x, y, z, 0.0)
                    for entity in entities:
                        if hasattr(entity, 'name') and entity not in self.agents:
                            targets.append(f"[OBJ] {entity.name} (Object)")
        
        # Remove duplicates while preserving order
        seen = set()
        unique_targets = []
        for target in targets:
            if target not in seen:
                seen.add(target)
                unique_targets.append(target)
        
        return unique_targets
    
    def shutdown(self):
        """Shutdown the simulation."""
        self.running = False


def run_demo(settings_file: str = "settings_demo_simulation.json"):
    """Run the action-driven simulation automatically."""
    print("[START] Starting Automatic Action-Driven Simulation", flush=True)
    print("=" * 60, flush=True)
    
    # Load world and agents
    maze, agents = load_settings(settings_file)
    
    # Initialize simulation
    simulation = ActionDrivenSimulation(maze, agents)
    
    # Initialize agents
    for agent in agents:
        agent.memory = Memory()
        agent.previous_action = None
        agent.inventory.clear()

    print("\n" + "="*60)
    print("[SYSTEM] Automatic Action-Driven Dialogue System Activated!")
    print("[INFO] Simulation will run automatically for a set duration")
    print("[INFO] Press Ctrl+C to stop the simulation")
    print("="*60)
    
    # Run simulation automatically for a specified duration
    simulation_duration = 120  # Run for 2 minutes for extended testing
    start_time = time.time()
    
    try:
        while simulation.running:
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Check if simulation duration exceeded
            if elapsed > simulation_duration:
                print(f"\n[INFO] Simulation duration ({simulation_duration}s) reached")
                simulation.running = False
                break
            
            # Display status every 30 seconds
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                print(f"\n[STATUS] Simulation running for {int(elapsed)}s")
                for agent in agents:
                    print(f"  {agent.name}: Location {agent.location}")
            
            # Let the action thread handle all interactions
            time.sleep(1)
                    
    except KeyboardInterrupt:
        print("\n[INFO] Stopping simulation...")
        simulation.running = False
    except Exception as e:
        print(f"[ERROR] Simulation error: {e}")
    
    finally:
        # Cleanup
        simulation.shutdown()
        print("[END] Simulation ended")


if __name__ == "__main__":
    import sys

    settings = sys.argv[1] if len(sys.argv) > 1 else "settings_demo_simulation.json"
    run_demo(settings)

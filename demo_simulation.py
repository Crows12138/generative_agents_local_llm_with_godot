import json
import time
import threading
import queue
from datetime import datetime, timedelta
from typing import List, Tuple

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
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        for agent in self.agents:
            try:
                schedule = generate_enhanced_daily_schedule(
                    agent_name=agent.name,
                    personality_description=agent.personality_description,
                    high_level_goals=agent.long_term_goals,
                    medium_term_memories="",  # Will be enhanced later
                    current_date=current_date,
                )
                self.daily_schedules[agent.name] = schedule
                print(f"\n=== {agent.name}'s Enhanced Daily Schedule ===")
                for task in schedule:
                    print(f"  {task['start']}-{task['end']}: {task['task']}")
                print("=" * 50)
            except Exception as e:
                print(f"Error generating schedule for {agent.name}: {e}")
                self.daily_schedules[agent.name] = []
    
    def _action_worker(self):
        """Background worker for processing actions."""
        while self.running:
            try:
                # Process actions for each agent
                for agent in self.agents:
                    if not self.running:
                        break
                    
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
                    print(f"\n🎯 {agent.name} decides: {action_data['action']}{target_display}")
                    
                    # Execute the action
                    try:
                        self.executor.execute(intent, agent, self.maze)
                    except Exception as e:
                        print(f"❌ Error executing action for {agent.name}: {e}")
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
                print(f"❌ Action processing error: {e}")
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
            
            print(f"💭 {agent.name} reflects: {reflection}")
            
        except Exception as e:
            print(f"❌ Error generating reflection for {agent.name}: {e}")
    
    def trigger_dialogue(self, speaker_name: str, target_name: str, custom_message: str = ""):
        """Trigger a dialogue between two entities."""
        speaker = None
        for agent in self.agents:
            if agent.name.lower() == speaker_name.lower():
                speaker = agent
                break
        
        if speaker is None:
            print(f"❌ Speaker '{speaker_name}' not found!")
            return
        
        # Create dialogue intent
        intent = ActionIntent("talk_to", target_name)
        
        # Execute dialogue
        print(f"\n🗣️  {speaker_name} starts talking to {target_name}...")
        self.executor.execute(intent, speaker, self.maze)
        
        # Get the conversation from memory
        if speaker.memory.entries:
            last_entry = speaker.memory.entries[-1]
            if last_entry.event_type == "talk":
                print(f"💬 Conversation content:")
                print(f"   {last_entry.text}")
                if "metadata" in last_entry.__dict__ and last_entry.metadata:
                    print(f"   📝 Metadata: {last_entry.metadata}")
    
    def list_available_targets(self):
        """List all available dialogue targets."""
        targets = []
        
        # Add agents
        for agent in self.agents:
            targets.append(f"👤 {agent.name} (Character)")
        
        # Add objects by scanning the maze
        for x in range(self.maze.width):
            for y in range(self.maze.height):
                for z in range(self.maze.depth):
                    entities = self.maze.spatial.nearby(x, y, z, 0.0)
                    for entity in entities:
                        if hasattr(entity, 'name') and entity not in self.agents:
                            targets.append(f"🏷️  {entity.name} (Object)")
        
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
    """Run the action-driven simulation."""
    print("🚀 Starting Action-Driven Simulation")
    print("=" * 60)
    
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
    print("🎭 Action-Driven Dialogue System Activated!")
    print("💡 You can manually trigger character dialogues:")
    print("   - Type 'talk Bob Jean' to make Bob talk to Jean")
    print("   - Type 'talk Jean Bob' to make Jean talk to Bob")
    print("   - Type 'talk Bob Tree' to make Bob talk to objects")
    print("   - Type 'targets' to see all available dialogue targets")
    print("   - Type 'help' to see help information")
    print("   - Type 'model <qwen|llama|gpt-oss>' to switch local LLM model")
    print("   - Type 'quit' to exit simulation")
    print("="*60)
    
    # Main input loop
    try:
        while simulation.running:
            try:
                user_input = input("\n💬 Enter command: ").strip()
                
                if user_input.lower() == 'quit':
                    print("👋 Exiting simulation...")
                    simulation.running = False
                    break
                elif user_input.lower() == 'help':
                    print("\n📖 Help information:")
                    print("  talk <speaker> <target>  - Trigger dialogue")
                    print("  targets                  - List all targets")
                    print("  status                   - Show character status")
                    print("  help                     - Show help")
                    print("  quit                     - Exit simulation")
                elif user_input.lower() == 'targets':
                    print("\n🎯 Available dialogue targets:")
                    targets = simulation.list_available_targets()
                    for target in targets:
                        print(f"  {target}")
                elif user_input.lower() == 'status':
                    print("\n📊 Character status:")
                    for agent in agents:
                        print(f"  {agent.name}: Location {agent.location}, Goals: {agent.long_term_goals}")
                elif user_input.lower().startswith('model'):
                    parts = user_input.split()
                    if len(parts) == 1:
                        print(f"当前模型: {get_active_model()}")
                    elif len(parts) >= 2:
                        key = parts[1].lower()
                        if set_active_model(key):
                            print(f"✅ 已切换模型为: {get_active_model()}")
                        else:
                            print("❌ 无效的模型键。可选: qwen | llama | gpt-oss")
                    else:
                        print("❌ 用法: model <qwen|llama|gpt-oss>")
                elif user_input.lower().startswith('talk '):
                    parts = user_input.split()
                    if len(parts) >= 3:
                        speaker = parts[1]
                        target = parts[2]
                        custom_message = ' '.join(parts[3:]) if len(parts) > 3 else ""
                        
                        # Validate speaker exists
                        speaker_exists = any(agent.name.lower() == speaker.lower() for agent in agents)
                        if not speaker_exists:
                            print(f"❌ Character '{speaker}' not found!")
                            continue
                        
                        simulation.trigger_dialogue(speaker, target, custom_message)
                    else:
                        print("❌ Usage: talk <speaker> <target> [message]")
                else:
                    print("❌ Unknown command. Type 'help' for help.")
                    
            except KeyboardInterrupt:
                print("\n👋 Exiting simulation...")
                simulation.running = False
                break
            except Exception as e:
                print(f"❌ Input processing error: {e}")
    
    finally:
        # Cleanup
        simulation.shutdown()
        print("✅ Simulation ended")


if __name__ == "__main__":
    import sys

    settings = sys.argv[1] if len(sys.argv) > 1 else "settings_demo_simulation.json"
    run_demo(settings)

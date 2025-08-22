"""
Cognitive Module Wrapper for Reverie Integration

Wraps reverie's cognitive modules for use with SimpleAgent system.
Provides a bridge between SimpleAgent and reverie's complex cognitive functions.
"""

import sys
import os
from typing import List, Dict, Any, Optional
from datetime import datetime

# Add reverie path for imports
reverie_path = os.path.join(os.path.dirname(__file__), '..', 'reverie', 'backend_server')
sys.path.append(reverie_path)

try:
    from persona.cognitive_modules import perceive, retrieve, plan, reflect, execute, converse
    from persona.memory_structures.associative_memory import ConceptNode
    # Add debug variable for reverie compatibility
    import builtins
    if not hasattr(builtins, 'debug'):
        builtins.debug = False
    REVERIE_COGNITIVE_AVAILABLE = True
except ImportError as e:
    print(f"[cognitive_wrapper] Reverie cognitive modules not available: {e}")
    REVERIE_COGNITIVE_AVAILABLE = False

class MockMaze:
    """Mock maze class for compatibility with reverie modules"""
    
    def __init__(self, environment_data: Dict[str, Any]):
        self.environment = environment_data
        self.tiles = {}
        self.personas = {}
        
    def get_nearby_tiles(self, curr_tile: tuple, vision_radius: int) -> List[tuple]:
        """Get tiles within vision radius"""
        x, y = curr_tile
        nearby = []
        for dx in range(-vision_radius, vision_radius + 1):
            for dy in range(-vision_radius, vision_radius + 1):
                if dx*dx + dy*dy <= vision_radius*vision_radius:
                    nearby.append((x + dx, y + dy))
        return nearby
    
    def get_tile_details(self, tile: tuple) -> Dict[str, Any]:
        """Get details for a specific tile"""
        return self.tiles.get(tile, {"events": [], "objects": [], "personas": []})
    
    def get_tile_path(self, tile: tuple, level: str = "arena") -> str:
        """Get path representation of tile"""
        return f"world:zone:{tile[0]}_{tile[1]}"
    
    def access_tile(self, tile: tuple) -> Dict[str, Any]:
        """Access tile information in reverie format"""
        tile_data = self.get_tile_details(tile)
        return {
            "world": "world",
            "sector": "zone", 
            "arena": f"{tile[0]}_{tile[1]}",
            "game_object": tile_data.get("objects", [])
        }

class CognitiveModuleWrapper:
    """Wrapper for reverie's cognitive modules"""
    
    def __init__(self, agent):
        self.agent = agent
        self.use_reverie = REVERIE_COGNITIVE_AVAILABLE and hasattr(agent, 'reverie_memory')
        
        if self.use_reverie:
            self.setup_persona_compatibility()
            print(f"[cognitive_wrapper] Initialized reverie cognitive modules for {agent.name}")
        else:
            print(f"[cognitive_wrapper] Using fallback cognitive functions for {agent.name}")
    
    def setup_persona_compatibility(self):
        """Make agent compatible with reverie's persona expectations"""
        if not self.use_reverie:
            return
            
        # Add required attributes for reverie modules
        if not hasattr(self.agent, 's_mem'):
            self.agent.s_mem = self.agent.reverie_memory.spatial_memory
        if not hasattr(self.agent, 'a_mem'):
            self.agent.a_mem = self.agent.reverie_memory.associative_memory
        if not hasattr(self.agent, 'scratch'):
            self.agent.scratch = self.agent.reverie_memory.scratch
            
        # Ensure scratch has required attributes
        if hasattr(self.agent.scratch, 'name'):
            self.agent.scratch.name = self.agent.name
        if hasattr(self.agent.scratch, 'curr_tile'):
            # Use agent location as current tile
            self.agent.scratch.curr_tile = (int(self.agent.location.x), int(self.agent.location.y))
        if hasattr(self.agent.scratch, 'vision_r'):
            self.agent.scratch.vision_r = getattr(self.agent.scratch, 'vision_r', 4)
        if hasattr(self.agent.scratch, 'att_bandwidth'):
            self.agent.scratch.att_bandwidth = getattr(self.agent.scratch, 'att_bandwidth', 3)
        if hasattr(self.agent.scratch, 'retention'):
            self.agent.scratch.retention = getattr(self.agent.scratch, 'retention', 5)
    
    def create_mock_maze(self, environment_data: Dict[str, Any]) -> MockMaze:
        """Create a mock maze object from environment data"""
        maze = MockMaze(environment_data)
        
        # Add agent's current location to maze
        agent_tile = (int(self.agent.location.x), int(self.agent.location.y))
        
        # Process events properly for maze
        events = []
        for event in environment_data.get("events", []):
            if isinstance(event, str):
                events.append({
                    "description": event,
                    "location": agent_tile,
                    "timestamp": datetime.now()
                })
            else:
                events.append(event)
        
        maze.tiles[agent_tile] = {
            "events": events,
            "objects": environment_data.get("objects", []),
            "personas": environment_data.get("other_agents", [])
        }
        
        return maze
    
    def perceive_environment(self, environment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Enhanced perception using reverie's perceive module"""
        if not self.use_reverie:
            return self._fallback_perceive(environment_data)
        
        try:
            # Create mock maze for reverie compatibility
            maze = self.create_mock_maze(environment_data)
            
            # Update agent's current position
            self.agent.scratch.curr_tile = (int(self.agent.location.x), int(self.agent.location.y))
            
            # Use reverie's perceive module
            perceived_events = perceive.perceive(self.agent, maze)
            
            # Convert ConceptNode objects to dictionaries
            results = []
            for event in perceived_events:
                if hasattr(event, 'description'):
                    results.append({
                        "description": event.description,
                        "subject": getattr(event, 'subject', ''),
                        "predicate": getattr(event, 'predicate', ''),
                        "object": getattr(event, 'object', ''),
                        "importance": getattr(event, 'poignancy', 5),
                        "timestamp": getattr(event, 'created', datetime.now())
                    })
            
            # If no reverie events, add fallback events
            if not results:
                results = self._fallback_perceive(environment_data)
            
            return results
            
        except Exception as e:
            print(f"[cognitive_wrapper] Error in reverie perceive: {e}")
            return self._fallback_perceive(environment_data)
    
    def retrieve_memories(self, focal_points: List[str], n: int = 5) -> Dict[str, Any]:
        """Enhanced memory retrieval using reverie's retrieve module"""
        if not self.use_reverie:
            return self._fallback_retrieve(focal_points, n)
        
        try:
            # Create mock perception events for retrieval
            perceived = []
            for point in focal_points:
                # Create a simple ConceptNode for each focal point
                event = type('MockEvent', (), {
                    'description': point,
                    'subject': self.agent.name,
                    'predicate': 'considers',
                    'object': point
                })()
                perceived.append(event)
            
            # Use reverie's retrieve module
            retrieved = retrieve.retrieve(self.agent, perceived)
            
            # Convert to simplified format
            result = {}
            for key, value in retrieved.items():
                result[key] = {
                    "events": [self._convert_concept_node(node) for node in value.get("events", [])],
                    "thoughts": [self._convert_concept_node(node) for node in value.get("thoughts", [])]
                }
            
            return result
            
        except Exception as e:
            print(f"[cognitive_wrapper] Error in reverie retrieve: {e}")
            return self._fallback_retrieve(focal_points, n)
    
    def plan_action(self, environment_data: Dict[str, Any], other_agents: List = None, new_day: bool = False) -> Dict[str, Any]:
        """Enhanced planning using reverie's plan module"""
        if not self.use_reverie:
            return self._fallback_plan(environment_data)
        
        try:
            # Create mock maze and personas
            maze = self.create_mock_maze(environment_data)
            personas = other_agents or []
            
            # Use reverie's plan module functions
            if new_day:
                # Generate wake up hour and daily plan
                wake_hour = plan.generate_wake_up_hour(self.agent)
                daily_plan = plan.generate_first_daily_plan(self.agent, wake_hour)
                
                return {
                    "type": "daily_plan",
                    "wake_hour": wake_hour,
                    "plan": daily_plan,
                    "timestamp": datetime.now()
                }
            else:
                # Generate immediate action plan
                # Note: This is a simplified version as the full plan module is complex
                action_plan = self._generate_action_plan(environment_data)
                
                return {
                    "type": "action_plan",
                    "actions": action_plan,
                    "timestamp": datetime.now()
                }
                
        except Exception as e:
            print(f"[cognitive_wrapper] Error in reverie plan: {e}")
            return self._fallback_plan(environment_data)
    
    def reflect_on_experience(self) -> Dict[str, Any]:
        """Enhanced reflection using reverie's reflect module"""
        if not self.use_reverie:
            return self._fallback_reflect()
        
        try:
            # Use reverie's reflect module
            reflection_result = reflect.run_reflect(self.agent)
            
            return {
                "reflections": reflection_result if reflection_result else [],
                "timestamp": datetime.now(),
                "agent": self.agent.name
            }
            
        except Exception as e:
            print(f"[cognitive_wrapper] Error in reverie reflect: {e}")
            return self._fallback_reflect()
    
    def execute_action(self, environment_data: Dict[str, Any], other_agents: List = None) -> Dict[str, Any]:
        """Enhanced action execution using reverie's execute module"""
        if not self.use_reverie:
            return self._fallback_execute(environment_data)
        
        try:
            # Create mock maze and personas
            maze = self.create_mock_maze(environment_data)
            personas = {agent.name: agent for agent in (other_agents or [])}
            
            # Generate a simple plan for execution
            available_actions = environment_data.get("available_actions", ["idle"])
            current_location = f"world:zone:{int(self.agent.location.x)}_{int(self.agent.location.y)}"
            plan = f"{current_location}:{available_actions[0] if available_actions else 'idle'}"
            
            # Use reverie's execute module with plan
            action_result = execute.execute(self.agent, maze, personas, plan)
            
            return {
                "action": action_result if action_result else available_actions[0],
                "plan": plan,
                "timestamp": datetime.now(),
                "agent": self.agent.name
            }
            
        except Exception as e:
            print(f"[cognitive_wrapper] Error in reverie execute: {e}")
            return self._fallback_execute(environment_data)
    
    def converse_with_agent(self, target_agent, environment_data: Dict[str, Any]) -> str:
        """Enhanced conversation using reverie's converse module"""
        if not self.use_reverie:
            return self._fallback_converse(target_agent)
        
        try:
            # Create mock maze for conversation context
            maze = self.create_mock_maze(environment_data)
            
            # Use reverie's converse module
            conversation = converse.agent_chat_v1(maze, self.agent, target_agent)
            
            return conversation if conversation else "Hello."
            
        except Exception as e:
            print(f"[cognitive_wrapper] Error in reverie converse: {e}")
            return self._fallback_converse(target_agent)
    
    def _convert_concept_node(self, node) -> Dict[str, Any]:
        """Convert ConceptNode to dictionary"""
        if hasattr(node, 'description'):
            return {
                "description": node.description,
                "subject": getattr(node, 'subject', ''),
                "predicate": getattr(node, 'predicate', ''),
                "object": getattr(node, 'object', ''),
                "importance": getattr(node, 'poignancy', 5),
                "timestamp": getattr(node, 'created', datetime.now())
            }
        else:
            return {"description": str(node), "importance": 3, "timestamp": datetime.now()}
    
    def _generate_action_plan(self, environment_data: Dict[str, Any]) -> List[str]:
        """Generate simple action plan"""
        actions = []
        
        # Based on agent's current state and environment
        if self.agent.energy < 0.3:
            actions.append("rest")
        elif self.agent.hunger > 0.7:
            actions.append("eat")
        elif environment_data.get("other_agents"):
            actions.append("socialize")
        else:
            actions.append("explore")
        
        return actions
    
    # Fallback methods when reverie is not available
    def _fallback_perceive(self, environment_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback perception method"""
        events = []
        for event in environment_data.get("events", []):
            events.append({
                "description": event,
                "importance": 3,
                "timestamp": datetime.now()
            })
        return events
    
    def _fallback_retrieve(self, focal_points: List[str], n: int) -> Dict[str, Any]:
        """Fallback memory retrieval"""
        result = {}
        for point in focal_points:
            # Use agent's simple memory system
            relevant = self.agent.memory.get_relevant_memories(point, n)
            result[point] = {
                "events": [{"description": mem.content, "importance": mem.importance} for mem in relevant],
                "thoughts": []
            }
        return result
    
    def _fallback_plan(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback planning method"""
        # Simple planning based on agent needs
        actions = self._generate_action_plan(environment_data)
        return {
            "type": "simple_plan",
            "actions": actions,
            "timestamp": datetime.now()
        }
    
    def _fallback_reflect(self) -> Dict[str, Any]:
        """Fallback reflection method"""
        return {
            "reflections": [f"I am {self.agent.emotional_state.value} and my current activity is {self.agent.activity.value}"],
            "timestamp": datetime.now(),
            "agent": self.agent.name
        }
    
    def _fallback_execute(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback execution method"""
        # Use agent's existing decision making
        available_actions = environment_data.get("available_actions", ["idle", "move", "interact"])
        action, reason = self.agent.decide_action(available_actions)
        
        return {
            "action": action,
            "reason": reason,
            "timestamp": datetime.now(),
            "agent": self.agent.name
        }
    
    def _fallback_converse(self, target_agent) -> str:
        """Fallback conversation method"""
        if hasattr(target_agent, 'name'):
            return self.agent.respond_to(target_agent.name, "Hello!")
        else:
            return "Hello there!"

class EnhancedAgent:
    """Enhanced agent with cognitive module integration"""
    
    def __init__(self, base_agent):
        self.base_agent = base_agent
        self.cognitive = CognitiveModuleWrapper(base_agent)
        
        # Delegate attribute access to base agent
        self.__dict__.update(base_agent.__dict__)
    
    def __getattr__(self, name):
        """Delegate missing attributes to base agent"""
        return getattr(self.base_agent, name)
    
    def think_and_act(self, environment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Complete cognitive cycle with reverie modules"""
        cycle_result = {
            "agent": self.name,
            "timestamp": datetime.now(),
            "cycle_steps": {}
        }
        
        try:
            # 1. Perceive environment
            perceptions = self.cognitive.perceive_environment(environment_data)
            cycle_result["cycle_steps"]["perceive"] = {
                "events_perceived": len(perceptions),
                "events": perceptions[:3]  # First 3 for summary
            }
            
            # 2. Retrieve relevant memories
            if perceptions:
                focal_points = [event["description"] for event in perceptions[:2]]
                memories = self.cognitive.retrieve_memories(focal_points, n=3)
                cycle_result["cycle_steps"]["retrieve"] = {
                    "focal_points": focal_points,
                    "memories_found": sum(len(mem["events"]) + len(mem["thoughts"]) for mem in memories.values())
                }
            
            # 3. Plan action if needed
            if self.needs_new_plan():
                plan_result = self.cognitive.plan_action(environment_data, new_day=False)
                cycle_result["cycle_steps"]["plan"] = plan_result
            
            # 4. Execute action
            action_result = self.cognitive.execute_action(environment_data)
            cycle_result["cycle_steps"]["execute"] = action_result
            
            # 5. Reflect periodically
            if self.should_reflect():
                reflection_result = self.cognitive.reflect_on_experience()
                cycle_result["cycle_steps"]["reflect"] = reflection_result
            
            cycle_result["success"] = True
            return cycle_result
            
        except Exception as e:
            cycle_result["success"] = False
            cycle_result["error"] = str(e)
            print(f"[EnhancedAgent] Error in cognitive cycle: {e}")
            return cycle_result
    
    def needs_new_plan(self) -> bool:
        """Check if agent needs a new plan"""
        # Simple heuristic: need new plan if no current goal or energy/hunger changes significantly
        return (
            not hasattr(self, 'current_goal') or 
            self.current_goal is None or
            self.energy < 0.2 or 
            self.hunger > 0.8
        )
    
    def should_reflect(self) -> bool:
        """Check if agent should reflect"""
        # Simple heuristic: reflect every few actions or when emotional state changes
        return (
            getattr(self, '_actions_since_reflection', 0) > 5 or
            self.emotional_state != getattr(self, '_last_emotional_state', self.emotional_state)
        )
    
    def converse_with(self, target_agent, environment_data: Dict[str, Any]) -> str:
        """Enhanced conversation with cognitive modules"""
        try:
            response = self.cognitive.converse_with_agent(target_agent, environment_data)
            
            # Update conversation state
            self.in_conversation = True
            self.conversation_partner = getattr(target_agent, 'name', str(target_agent))
            
            return response
        except Exception as e:
            print(f"[EnhancedAgent] Error in conversation: {e}")
            # Fallback to base agent conversation
            if hasattr(target_agent, 'name'):
                return self.respond_to(target_agent.name, "Hello!")
            else:
                return "Hello there!"
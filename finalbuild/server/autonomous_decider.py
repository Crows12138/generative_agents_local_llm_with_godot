#!/usr/bin/env python3
"""
Autonomous Decider System - Perception-driven decision making
Uses natural language processing for emergent behaviors rather than rigid state machines
"""

import asyncio
import time
import json
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
from gpt4all import GPT4All
from bar_state import BarEnvironment as BarStateManager, ObjectState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PerceptionDrivenDecider:
    """Decider that makes decisions based on perception rather than rigid states"""
    
    def __init__(self, name: str, personality: str = "friendly bartender", model_name: str = "Llama-3.2-3B-Instruct-Q4_0.gguf"):
        self.name = name
        self.personality = personality
        self.position = (400, 300)  # Default position at bar counter
        
        # Initialize GPT4All model with chat session for memory continuity
        try:
            # Use local models directory
            from pathlib import Path
            model_path = Path(__file__).parent.parent.parent / "models" / "llms"
            self.model = GPT4All(
                model_name=model_name,
                model_path=str(model_path),
                allow_download=False,  # Prevent downloading
                verbose=False
            )
            # Create chat session context and enter it
            session_context = self.model.chat_session(
                system_prompt=f"""You are {name}, a {personality}. 
You have your own thoughts, feelings, and motivations.
You make decisions based on what you perceive around you.
Your responses should be natural and reflect your personality."""
            )
            self.decision_session = session_context.__enter__()
            logger.info(f"Initialized {name} with {model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize model for {name}: {e}")
            self.model = None
            self.decision_session = None
        
        # Bar state manager for environment perception
        self.bar_state = BarStateManager()
        
        # Current perception state (updated each frame)
        self.perception = {
            "location": "bar_counter",
            "current_activity": "idle",
            "time_of_day": "evening",
            "customers_present": [],
            "sounds": ["quiet bar ambiance"],
            "nearby_objects": [],
            "recent_events": []
        }
        
        # Decision history for analysis (doesn't affect decisions)
        self.decision_history = []
        self.last_decision_time = time.time()
        self.decision_cooldown = 2.0  # Seconds between decisions
        
        # Current action being executed
        self.current_action = None
        self.action_start_time = None
        self.action_duration = 0
    
    def update_perception_from_bar_state(self) -> Dict[str, Any]:
        """Convert Bar state to natural language perception"""
        perception_updates = {}
        
        # Get bar state perception for this NPC's position
        bar_perception = self.bar_state.get_perception_for_npc(
            npc_position=self.position
        )
        
        # Convert state objects to natural descriptions
        nearby_descriptions = []
        sounds = ["quiet bar ambiance"]  # Default
        recent_events = []
        
        # Process immediate and nearby objects from perception
        for category in ['immediate', 'nearby']:
            for obj_info in bar_perception.get(category, []):
                obj_name = obj_info.get('name', '')
                state_str = obj_info.get('state', '')
                distance = obj_info.get('distance', 0)
            
                # Convert state to natural perception
                if state_str == 'customers_waiting':
                    nearby_descriptions.append("customers waiting at the counter")
                    sounds.append("customer chatter")
                    recent_events.append("new customers arrived")
                
                elif state_str == 'dirty' and 'counter' in obj_name:
                    nearby_descriptions.append("dirty glasses and spills on the counter")
                
                elif state_str == 'food_waiting':
                    table_name = obj_name.replace('_', ' ')
                    nearby_descriptions.append(f"customers waiting at the {table_name}")
                    sounds.append("distant conversation")
                
                elif state_str == 'needs_cleaning':
                    table_name = obj_name.replace('_', ' ')
                    nearby_descriptions.append(f"dirty dishes on the {table_name}")
                
                elif state_str == 'in_use' and 'pool' in obj_name:
                    nearby_descriptions.append("people playing pool")
                    sounds.append("pool balls clicking")
                
        
        # Update perception with natural language
        perception_updates["nearby_objects"] = nearby_descriptions
        perception_updates["sounds"] = sounds
        perception_updates["recent_events"] = recent_events
        
        # Get time-based perception
        current_hour = datetime.now().hour
        if 6 <= current_hour < 12:
            perception_updates["time_of_day"] = "morning"
        elif 12 <= current_hour < 17:
            perception_updates["time_of_day"] = "afternoon"
        elif 17 <= current_hour < 22:
            perception_updates["time_of_day"] = "evening"
        else:
            perception_updates["time_of_day"] = "night"
        
        return perception_updates
    
    async def make_decision(self) -> Optional[str]:
        """Make a decision based on current perception using natural language"""
        
        # Check cooldown
        if time.time() - self.last_decision_time < self.decision_cooldown:
            return None
        
        if not self.decision_session:
            logger.warning(f"{self.name}: No model available for decision making")
            return "observe"
        
        # Update perception from bar state
        perception_updates = self.update_perception_from_bar_state()
        self.perception.update(perception_updates)
        
        # Build natural perception description
        prompt = f"""
You are {self.name}, a {self.personality}.

Current situation:
- Location: {self.perception['location']}
- Time: {self.perception['time_of_day']}
- You see: {', '.join(self.perception['nearby_objects']) if self.perception['nearby_objects'] else 'nothing unusual'}
- You hear: {', '.join(self.perception['sounds'])}

What single action do you take?
Choose ONE from: clean_counter, serve_customer, clear_table, organize_shelf, take_break, observe

Respond with ONLY the action word.
"""
        
        try:
            # Generate decision with context memory
            response = self.decision_session.generate(prompt, max_tokens=50)
            
            # Extract action from natural language response
            action = self._extract_action_from_response(response)
            
            # Record decision
            self.decision_history.append({
                "perception": self.perception.copy(),
                "response": response,
                "action": action,
                "timestamp": time.time()
            })
            
            self.last_decision_time = time.time()
            self.current_action = action
            self.action_start_time = time.time()
            
            logger.info(f"{self.name} decides: {response[:100]}... -> Action: {action}")
            
            return action
            
        except Exception as e:
            logger.error(f"{self.name} decision error: {e}")
            return "observe"
    
    def _extract_action_from_response(self, response: str) -> str:
        """Extract concrete action from response"""
        
        response_clean = response.strip().lower()
        
        # Valid actions
        valid_actions = [
            "clean_counter", "serve_customer", "clear_table", 
            "organize_shelf", "play_pool", "take_break", 
            "walk_to", "greet", "observe"
        ]
        
        # First check for exact match
        for action in valid_actions:
            if action in response_clean:
                return action
        
        # Check for close matches (in case of typos or variations)
        action_keywords = {
            "clean_counter": ["clean", "wipe", "cleaning"],
            "serve_customer": ["serve", "serving", "help", "customer"],
            "clear_table": ["clear", "table"],
            "organize_shelf": ["organize", "restock", "shelf"],
            "play_pool": ["pool", "play"],
            "take_break": ["break", "rest", "relax"],
            "walk_to": ["walk", "go", "move"],
            "greet": ["greet", "hello", "welcome"],
            "observe": ["observe", "look", "watch", "check"]
        }
        
        # Find matching action
        for action, keywords in action_keywords.items():
            if any(keyword in response_clean for keyword in keywords):
                return action
        
        # Default to observing
        return "observe"
    
    async def execute_action(self, action: str) -> Dict[str, Any]:
        """Execute the decided action and return results"""
        
        result = {
            "action": action,
            "success": True,
            "duration": 0,
            "effects": []
        }
        
        # Simulate action execution with different durations
        action_durations = {
            "clean_counter": 5.0,
            "serve_customer": 8.0,
            "clear_table": 4.0,
            "organize_shelf": 6.0,
            "play_pool": 15.0,
            "take_break": 10.0,
            "walk_to": 3.0,
            "greet": 1.0,
            "observe": 1.0
        }
        
        duration = action_durations.get(action, 2.0)
        result["duration"] = duration
        
        # Update bar state based on action
        if action == "clean_counter":
            self.bar_state.core_objects["bar_counter"].state = None  # Clear dirty state
            result["effects"].append("counter_cleaned")
            self.perception["current_activity"] = "cleaning"
            
        elif action == "serve_customer":
            self.bar_state.core_objects["bar_counter"].state = None  # Clear waiting state
            result["effects"].append("customer_served")
            self.perception["current_activity"] = "serving customers"
            
        elif action == "clear_table":
            # Find dirty table and clear it
            for table_name in ["left_table", "center_table", "right_table"]:
                table = self.bar_state.core_objects.get(table_name)
                if table and table.state == ObjectState.TABLE_NEEDS_CLEANING:
                    table.state = None
                    result["effects"].append(f"{table_name}_cleared")
                    break
            self.perception["current_activity"] = "clearing tables"
            
        else:
            self.perception["current_activity"] = action.replace('_', ' ')
        
        # Simulate action duration
        await asyncio.sleep(min(duration, 0.5))  # Cap at 0.5s for testing
        
        return result
    
    def get_status(self) -> Dict[str, Any]:
        """Get current decider status"""
        return {
            "name": self.name,
            "position": self.position,
            "current_action": self.current_action,
            "current_activity": self.perception["current_activity"],
            "perception": self.perception,
            "decision_count": len(self.decision_history)
        }


class NaturalLanguageDecider(PerceptionDrivenDecider):
    """Extended decider with internal monologue for more complex reasoning"""
    
    def __init__(self, name: str, personality: str = "thoughtful bartender", model_name: str = "Llama-3.2-3B-Instruct-Q4_0.gguf"):
        super().__init__(name, personality, model_name)
        
        # Additional inner monologue session for deeper thoughts
        if self.model:
            # Create inner monologue session
            monologue_context = self.model.chat_session(
                system_prompt=f"""You are {name}'s inner thoughts.
You reflect on what you see, how you feel, and what matters to you.
Your thoughts flow naturally, considering both immediate needs and personal feelings.
You have memories and experiences that shape your decisions."""
            )
            self.inner_monologue = monologue_context.__enter__()
        else:
            self.inner_monologue = None
        
        # Emotional state tracking
        self.emotional_state = {
            "mood": "content",
            "energy": 100,
            "stress": 0,
            "satisfaction": 50
        }
    
    async def think_and_act(self) -> Optional[str]:
        """Extended decision making with internal monologue"""
        
        # Check cooldown
        if time.time() - self.last_decision_time < self.decision_cooldown:
            return None
        
        if not self.inner_monologue:
            # Fall back to basic decision making
            return await self.make_decision()
        
        # Update perception
        perception_updates = self.update_perception_from_bar_state()
        self.perception.update(perception_updates)
        
        # Generate internal thoughts first
        thought_prompt = f"""
You are {self.name}.
You see: {', '.join(self.perception['nearby_objects']) if self.perception['nearby_objects'] else 'a quiet bar'}
Mood: {self.emotional_state['mood']}
Energy: {self.emotional_state['energy']}%

What is your single thought? (max 10 words)
"""
        
        try:
            inner_thought = self.inner_monologue.generate(thought_prompt, max_tokens=40)
            
            # Now make decision based on thoughts
            decision_prompt = f"""
Thought: {inner_thought}

What action do you take?
Choose ONE: clean_counter, serve_customer, clear_table, take_break, observe

Respond with ONLY the action.
"""
            
            decision = self.decision_session.generate(decision_prompt, max_tokens=30)
            
            action = self._extract_action_from_response(decision)
            
            # Update emotional state based on action
            self._update_emotional_state(action)
            
            # Record with inner thoughts
            self.decision_history.append({
                "perception": self.perception.copy(),
                "inner_thought": inner_thought,
                "decision": decision,
                "action": action,
                "emotional_state": self.emotional_state.copy(),
                "timestamp": time.time()
            })
            
            self.last_decision_time = time.time()
            self.current_action = action
            
            logger.info(f"{self.name} thinks: {inner_thought[:50]}...")
            logger.info(f"{self.name} decides: {decision[:50]}... -> {action}")
            
            return action
            
        except Exception as e:
            logger.error(f"{self.name} thinking error: {e}")
            return "observe"
    
    def _update_emotional_state(self, action: str):
        """Update emotional state based on actions"""
        
        # Energy cost of actions
        energy_cost = {
            "clean_counter": 10,
            "serve_customer": 15,
            "clear_table": 8,
            "organize_shelf": 12,
            "play_pool": 5,
            "take_break": -20,  # Restores energy
            "observe": -2
        }
        
        # Update energy
        cost = energy_cost.get(action, 5)
        self.emotional_state["energy"] = max(0, min(100, self.emotional_state["energy"] - cost))
        
        # Update mood based on energy and activity
        if self.emotional_state["energy"] < 30:
            self.emotional_state["mood"] = "tired"
        elif action in ["serve_customer", "greet"]:
            self.emotional_state["mood"] = "friendly"
        elif action == "play_pool":
            self.emotional_state["mood"] = "playful"
        elif action == "take_break":
            self.emotional_state["mood"] = "relaxed"
        else:
            self.emotional_state["mood"] = "content"
        
        # Satisfaction from completing tasks
        if action in ["clean_counter", "serve_customer", "clear_table"]:
            self.emotional_state["satisfaction"] = min(100, self.emotional_state["satisfaction"] + 10)


async def test_decider_system():
    """Test the autonomous decider system and output results as JSON"""
    
    # Create deciders with different personalities
    bartender = NaturalLanguageDecider("Bob", "experienced bartender")
    helper = PerceptionDrivenDecider("Alice", "eager assistant")
    
    # Simulate some bar states
    bartender.bar_state.core_objects["bar_counter"].state = ObjectState.COUNTER_DIRTY
    bartender.bar_state.core_objects["left_table"].state = ObjectState.TABLE_FOOD_WAITING
    
    logger.info("Starting decider simulation...")
    
    # Collect simulation data
    simulation_data = {
        "start_time": time.time(),
        "npcs": {
            "Sam": {
                "type": "NaturalLanguageDecider",
                "personality": "experienced bartender",
                "cycles": []
            },
            "Alice": {
                "type": "PerceptionDrivenDecider", 
                "personality": "eager assistant",
                "cycles": []
            }
        },
        "events": [],
        "bar_state_changes": []
    }
    
    # Run simulation
    for i in range(5):
        logger.info(f"\n--- Cycle {i+1} ---")
        cycle_data = {"cycle": i+1, "actions": {}}
        
        # Capture initial bar state
        initial_bar_state = {
            obj_name: obj.state.value if obj.state else None
            for obj_name, obj in bartender.bar_state.core_objects.items()
        }
        
        # Bartender with inner monologue
        action = await bartender.think_and_act()
        if action:
            result = await bartender.execute_action(action)
            logger.info(f"Sam executed: {action} - Effects: {result['effects']}")
            
            # Capture Sam's decision data
            if bartender.decision_history:
                latest = bartender.decision_history[-1]
                cycle_data["actions"]["Sam"] = {
                    "inner_thought": latest.get("inner_thought", ""),
                    "decision": latest.get("decision", ""),
                    "action": action,
                    "effects": result['effects'],
                    "perception": latest.get("perception", {}),
                    "emotional_state": latest.get("emotional_state", {})
                }
                simulation_data["npcs"]["Sam"]["cycles"].append(cycle_data["actions"]["Sam"])
        
        # Helper with basic perception
        helper.bar_state = bartender.bar_state  # Share bar state
        action = await helper.make_decision()
        if action:
            result = await helper.execute_action(action)
            logger.info(f"Alice executed: {action} - Effects: {result['effects']}")
            
            # Capture Alice's decision data
            if helper.decision_history:
                latest = helper.decision_history[-1]
                cycle_data["actions"]["Alice"] = {
                    "response": latest.get("response", ""),
                    "action": action,
                    "effects": result['effects'],
                    "perception": latest.get("perception", {})
                }
                simulation_data["npcs"]["Alice"]["cycles"].append(cycle_data["actions"]["Alice"])
        
        # Capture final bar state
        final_bar_state = {
            obj_name: obj.state.value if obj.state else None
            for obj_name, obj in bartender.bar_state.core_objects.items()
        }
        
        # Record state changes
        changes = {}
        for obj_name in initial_bar_state:
            if initial_bar_state[obj_name] != final_bar_state[obj_name]:
                changes[obj_name] = {
                    "from": initial_bar_state[obj_name],
                    "to": final_bar_state[obj_name]
                }
        
        if changes:
            simulation_data["bar_state_changes"].append({
                "cycle": i+1,
                "changes": changes
            })
        
        # Random bar events
        if i == 2:
            bartender.bar_state.core_objects["bar_counter"].state = ObjectState.COUNTER_CUSTOMERS_WAITING
            logger.info("EVENT: New customers arrived at counter")
            simulation_data["events"].append({
                "cycle": i+1,
                "event": "customers_arrived",
                "location": "bar_counter"
            })
        
        await asyncio.sleep(1)  # Reduced sleep for faster testing
    
    # Add summary
    simulation_data["summary"] = {
        "duration": time.time() - simulation_data["start_time"],
        "Sam": {
            "total_decisions": len(bartender.decision_history),
            "final_mood": bartender.emotional_state['mood'],
            "final_energy": bartender.emotional_state['energy'],
            "final_satisfaction": bartender.emotional_state['satisfaction']
        },
        "Alice": {
            "total_decisions": len(helper.decision_history)
        },
        "final_bar_state": {
            obj_name: obj.state.value if obj.state else None
            for obj_name, obj in bartender.bar_state.core_objects.items()
        }
    }
    
    # Save to JSON file
    output_file = Path(__file__).parent / "test_results.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(simulation_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\n--- Test Results Saved to {output_file} ---")
    
    # Also print summary
    logger.info("\n--- Simulation Summary ---")
    logger.info(f"Bob made {len(bartender.decision_history)} decisions")
    logger.info(f"Bob's final mood: {bartender.emotional_state['mood']}")
    logger.info(f"Alice made {len(helper.decision_history)} decisions")
    
    return simulation_data


if __name__ == "__main__":
    # Run test
    result = asyncio.run(test_decider_system())
    print("\n=== FINAL JSON OUTPUT ===")
    print(json.dumps(result, indent=2, ensure_ascii=False))
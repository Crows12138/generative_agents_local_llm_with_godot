#!/usr/bin/env python
"""
Enhanced Bar NPC System - Combines Persona system with BarAgent
Enhanced with dual model support for optimized performance
"""

import sys
import random
import time
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass

# Add path
sys.path.insert(0, '.')

# Import existing systems
from basic_functions.persona import Persona
from cozy_bar_demo.core.bar_agents import BarAgent, Memory

# Import dual model AI service
try:
    from ai_service.enhanced_ai_service import get_enhanced_service
    from ai_service.dual_model_scheduler import TaskType
    AI_SERVICE_AVAILABLE = True
except ImportError:
    AI_SERVICE_AVAILABLE = False
    print("[WARNING] Enhanced AI service not available, using fallback")


class EnhancedBarNPC(BarAgent):
    """Enhanced NPC combining Persona system with BarAgent and dual model AI"""
    
    def __init__(self, name: str, role: str, position: Tuple[int, int]):
        # Initialize BarAgent
        super().__init__(name, role, position)
        
        # Initialize Persona system
        self.persona = Persona(name, initial_location=(*position, 0))
        
        # Initialize AI service if available
        self.ai_service = get_enhanced_service() if AI_SERVICE_AVAILABLE else None
        
        # Enhanced attributes
        self.personality_traits = self._init_personality_traits()
        self.long_term_goals = self._init_long_term_goals()
        self.short_term_goals = []
        self.emotional_state = {
            "happiness": 50,
            "energy": 100,
            "social": 75,
            "stress": 20
        }
        
        # Cognitive abilities
        self.thinking_depth = 0  # 0=reactive, 1=thoughtful, 2=reflective
        self.last_reflection = time.time()
        self.reflection_interval = 300  # Reflect every 5 minutes
        
        # Dialogue enhancement
        self.dialogue_history = []
        self.dialogue_context = ""
        
        # Track which model was last used
        self.last_model_used = None
        
        print(f"[OK] Enhanced NPC {name} ({role}) initialized with dual model support")
    
    def _init_personality_traits(self) -> Dict[str, float]:
        """Initialize personality traits based on role"""
        traits = {
            "bartender": {
                "friendliness": 0.9,
                "patience": 0.8,
                "wisdom": 0.7,
                "humor": 0.6,
                "curiosity": 0.5
            },
            "regular customer": {
                "friendliness": 0.6,
                "patience": 0.4,
                "wisdom": 0.8,
                "humor": 0.5,
                "curiosity": 0.7
            },
            "musician": {
                "friendliness": 0.7,
                "patience": 0.5,
                "wisdom": 0.6,
                "humor": 0.8,
                "curiosity": 0.9
            }
        }
        return traits.get(self.role, {"friendliness": 0.5, "patience": 0.5})
    
    def _init_long_term_goals(self) -> List[str]:
        """Initialize long-term goals based on role"""
        goals = {
            "bartender": [
                "create a welcoming atmosphere",
                "remember regular customers' preferences",
                "master new cocktail recipes",
                "build lasting relationships"
            ],
            "regular customer": [
                "find meaning in conversations",
                "explore philosophical questions",
                "enjoy good drinks",
                "connect with interesting people"
            ],
            "musician": [
                "perfect my performance",
                "connect with the audience",
                "write new songs",
                "share my music with others"
            ]
        }
        return goals.get(self.role, ["exist", "interact"])
    
    def perceive_environment(self) -> Dict[str, Any]:
        """Perceive the current environment
        感知当前环境"""
        
        perception = {
            "time": datetime.now(),
            "location": self.position,
            "nearby_people": [],  # Would be filled by scene manager
            "ambient_mood": self._calculate_ambient_mood(),
            "personal_state": self.emotional_state,
            "recent_events": self.memories[-5:] if len(self.memories) > 0 else []
        }
        
        return perception
    
    def _calculate_ambient_mood(self) -> str:
        """Calculate the ambient mood of the bar"""
        hour = datetime.now().hour
        
        if 17 <= hour < 20:
            return "lively_early_evening"
        elif 20 <= hour < 23:
            return "peak_social_hours"
        elif 23 <= hour or hour < 2:
            return "late_night_contemplative"
        else:
            return "quiet_afternoon"
    
    def think(self, perception: Dict[str, Any]) -> str:
        """Think about the current situation
        思考当前情况"""
        
        # Simple reactive thinking
        if self.thinking_depth == 0:
            return self._reactive_think(perception)
        
        # Thoughtful consideration
        elif self.thinking_depth == 1:
            return self._thoughtful_think(perception)
        
        # Deep reflection
        else:
            return self._reflective_think(perception)
    
    def _reactive_think(self, perception: Dict[str, Any]) -> str:
        """Quick reactive thinking"""
        mood = perception["ambient_mood"]
        
        if self.role == "bartender":
            if mood == "peak_social_hours":
                return "need to keep up with orders"
            else:
                return "time to chat with customers"
        
        elif self.role == "regular customer":
            if self.emotional_state["happiness"] < 30:
                return "feeling melancholic tonight"
            else:
                return "enjoying the atmosphere"
        
        elif self.role == "musician":
            if mood == "peak_social_hours":
                return "time to perform"
            else:
                return "planning the setlist"
        
        return "observing the scene"
    
    def _thoughtful_think(self, perception: Dict[str, Any]) -> str:
        """Thoughtful consideration"""
        recent_memories = perception.get("recent_events", [])
        
        if recent_memories:
            last_memory = recent_memories[-1]
            if hasattr(last_memory, 'content'):
                return f"reflecting on: {last_memory.content[:30]}..."
        
        return "considering what to do next"
    
    def _reflective_think(self, perception: Dict[str, Any]) -> str:
        """Deep reflection on experiences"""
        
        # Check if it's time to reflect
        current_time = time.time()
        if current_time - self.last_reflection > self.reflection_interval:
            self.last_reflection = current_time
            return self._generate_reflection()
        
        return "contemplating life experiences"
    
    def _generate_reflection(self) -> str:
        """Generate a deep reflection"""
        reflections = {
            "bartender": [
                "Every drink tells a story, every customer has a journey",
                "Years behind this bar taught me: listening is the best service",
                "The perfect cocktail is 10% recipe, 90% understanding the person"
            ],
            "regular customer": [
                "Some nights you drink to remember, others to forget",
                "This bar has seen all my phases - triumph, defeat, and everything between",
                "The whiskey doesn't judge, and Bob doesn't ask questions"
            ],
            "musician": [
                "Music is the language when words fail",
                "Every performance is a conversation with the audience",
                "The best songs come from the heart, not the head"
            ]
        }
        
        role_reflections = reflections.get(self.role, ["Life goes on"])
        reflection = random.choice(role_reflections)
        
        # Store as important memory
        self.add_memory(f"Reflected: {reflection}", "contemplative", 0.9)
        
        return reflection
    
    def decide_action(self, thought: str) -> str:
        """Decide on an action based on thought
        基于思考决定行动"""
        
        # Update emotional state
        self._update_emotional_state()
        
        # Choose action based on role and state
        if self.role == "bartender":
            if "orders" in thought:
                return "serve_drinks"
            elif "chat" in thought:
                return "engage_conversation"
            else:
                return "maintain_bar"
        
        elif self.role == "regular customer":
            if self.emotional_state["social"] > 60:
                return "start_conversation"
            elif self.drunk_level > 5:
                return "philosophical_rambling"
            else:
                return "quiet_drinking"
        
        elif self.role == "musician":
            if "perform" in thought:
                return "play_music"
            elif "planning" in thought:
                return "prepare_setlist"
            else:
                return "casual_interaction"
        
        return "observe"
    
    def _update_emotional_state(self):
        """Update emotional state based on various factors"""
        hour = datetime.now().hour
        
        # Energy decreases over time
        self.emotional_state["energy"] -= 0.5
        
        # Social battery changes based on interactions
        if len(self.dialogue_history) > 0:
            self.emotional_state["social"] -= 2
        
        # Happiness affected by role and time
        if self.role == "musician" and 19 <= hour <= 22:
            self.emotional_state["happiness"] += 5
        
        # Clamp values
        for key in self.emotional_state:
            self.emotional_state[key] = max(0, min(100, self.emotional_state[key]))
    
    def generate_dialogue(self, player_input: str = "", context: str = "") -> str:
        """Generate contextual dialogue using dual model AI
        """
        
        # Store player input
        if player_input:
            self.dialogue_history.append({
                "speaker": "player",
                "text": player_input,
                "timestamp": datetime.now()
            })
        
        # Get current perception
        perception = self.perceive_environment()
        
        # Think about the situation
        thought = self.think(perception)
        
        # Use AI service if available
        if self.ai_service and AI_SERVICE_AVAILABLE:
            # Map thinking depth to task type
            if self.thinking_depth == 0:
                task_type = TaskType.REACTIVE
            elif self.thinking_depth == 1:
                task_type = TaskType.THOUGHTFUL
            else:  # thinking_depth >= 2
                task_type = TaskType.REFLECTION
            
            # Build prompt with context
            prompt = self._build_ai_prompt(player_input, thought, perception)
            
            # Generate using appropriate model
            ai_context = {
                "thinking_depth": self.thinking_depth,
                "mood": self.mood,
                "energy": self.emotional_state["energy"],
                "role": self.role
            }
            
            response = self.ai_service.generate(
                prompt=prompt,
                task_type=task_type,
                context=ai_context,
                use_cache=True
            )
            
            # Track which model was used
            self.last_model_used = "4B" if task_type == TaskType.REACTIVE else "30B"
            
        else:
            # Fallback to personality-based response
            response = self._generate_personality_based_response(
                player_input, 
                thought,
                perception
            )
            self.last_model_used = "fallback"
        
        # Store NPC response
        self.dialogue_history.append({
            "speaker": self.name,
            "text": response,
            "timestamp": datetime.now(),
            "model_used": self.last_model_used
        })
        
        # Keep dialogue history manageable
        if len(self.dialogue_history) > 20:
            self.dialogue_history = self.dialogue_history[-20:]
        
        return response
    
    def _build_ai_prompt(self, player_input: str, thought: str, perception: Dict) -> str:
        """Build prompt for AI service"""
        prompt = f"""You are {self.name}, a {self.role} at a cozy bar.
Your personality: {self.personality}
Current mood: {self.mood}
Current thought: {thought}
Time: {perception.get('ambient_mood', 'evening')}

Customer says: "{player_input}"

Respond in character, keeping it natural and conversational (1-2 sentences):"""
        
        return prompt
    
    def _generate_personality_based_response(self, 
                                            player_input: str,
                                            thought: str,
                                            perception: Dict) -> str:
        """Generate response based on personality traits"""
        
        # Use existing bar dialogue as base
        base_response = super().generate_bar_dialogue(player_input)
        
        # Enhance based on personality traits
        if self.personality_traits.get("friendliness", 0) > 0.7:
            # Add warmth to response
            if self.role == "bartender":
                base_response = f"*smiles warmly* {base_response}"
        
        if self.personality_traits.get("wisdom", 0) > 0.7:
            # Add depth to response
            if random.random() < 0.3:
                base_response += f" ...{thought}"
        
        return base_response
    
    def interact_with_npc(self, other_npc: 'EnhancedBarNPC') -> Tuple[str, str]:
        """Interact with another NPC
        与另一个NPC互动"""
        
        # Generate context-aware interaction
        my_thought = self.think(self.perceive_environment())
        their_perception = other_npc.perceive_environment()
        
        # Generate dialogue
        my_dialogue = f"{self.name} to {other_npc.name}: "
        
        if self.role == "bartender" and other_npc.role == "regular customer":
            my_dialogue += "The usual tonight, or feeling adventurous?"
        elif self.role == "musician" and other_npc.role == "bartender":
            my_dialogue += "Bob, how's the crowd looking tonight?"
        elif self.role == "regular customer" and other_npc.role == "musician":
            my_dialogue += "Play something melancholic, Sam. It's that kind of night."
        else:
            my_dialogue += "How's your evening going?"
        
        # Get response
        other_response = other_npc.generate_dialogue(my_dialogue)
        
        # Store interaction in memory
        self.add_memory(f"Talked with {other_npc.name}", "social", 0.5)
        other_npc.add_memory(f"Talked with {self.name}", "social", 0.5)
        
        return (my_dialogue, other_response)


def test_enhanced_npc():
    """Test the enhanced NPC system
    测试增强版NPC系统"""
    
    print("\n" + "="*50)
    print("Testing Enhanced Bar NPC System")
    print("="*50)
    
    # Create NPCs
    bob = EnhancedBarNPC("Bob", "bartender", (5, 3))
    alice = EnhancedBarNPC("Alice", "regular customer", (3, 4))
    sam = EnhancedBarNPC("Sam", "musician", (7, 6))
    
    # Test perception
    print("\n[Test 1: Perception]")
    bob_perception = bob.perceive_environment()
    print(f"Bob perceives: {bob_perception['ambient_mood']}")
    
    # Test thinking
    print("\n[Test 2: Thinking]")
    bob.thinking_depth = 0  # Reactive
    print(f"Bob (reactive): {bob.think(bob_perception)}")
    
    bob.thinking_depth = 2  # Reflective
    print(f"Bob (reflective): {bob.think(bob_perception)}")
    
    # Test dialogue generation
    print("\n[Test 3: Dialogue Generation]")
    player_says = "Hello Bob! What's good tonight?"
    bob_response = bob.generate_dialogue(player_says)
    print(f"Player: {player_says}")
    print(f"Bob: {bob_response}")
    
    # Test NPC interaction
    print("\n[Test 4: NPC Interaction]")
    alice_says, sam_response = alice.interact_with_npc(sam)
    print(alice_says)
    print(f"Sam: {sam_response}")
    
    # Show emotional states
    print("\n[Test 5: Emotional States]")
    for npc in [bob, alice, sam]:
        print(f"{npc.name}: happiness={npc.emotional_state['happiness']}, "
              f"energy={npc.emotional_state['energy']}, "
              f"social={npc.emotional_state['social']}")
    
    # Test memory
    print("\n[Test 6: Memory System]")
    print(f"Bob has {len(bob.memories)} memories")
    if bob.memories:
        latest = bob.memories[-1]
        print(f"Latest: {latest.content}")
    
    print("\n[SUCCESS] Enhanced NPC system working!")
    return True


if __name__ == "__main__":
    success = test_enhanced_npc()
    sys.exit(0 if success else 1)
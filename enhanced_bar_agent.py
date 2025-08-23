"""Enhanced Bar Agent with Reverie Cognitive Capabilities

This integrates Stanford Reverie's cognitive architecture with our bar agent system,
providing sophisticated AI characters with memory, planning, and reflection capabilities.
"""

import sys
import os
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum
import json

# Add project paths
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from reverie_integration.persona_wrapper import PersonaWrapper
from agents.simple_agents import SimpleAgent, Location, EmotionalState

class CognitiveMode(Enum):
    """Cognitive processing modes"""
    REACTIVE = "reactive"          # Quick reactions, minimal planning
    DELIBERATIVE = "deliberative"  # Careful planning and reflection
    ADAPTIVE = "adaptive"          # Balance based on situation

class EnhancedBarAgent(SimpleAgent):
    """Enhanced Bar Agent with Reverie cognitive capabilities"""
    
    def __init__(self, 
                 name: str,
                 role: str,
                 position: Tuple[int, int],
                 personality: str = "friendly bartender",
                 background: str = None,
                 cognitive_mode: CognitiveMode = CognitiveMode.ADAPTIVE):
        
        # Initialize base SimpleAgent
        super().__init__(
            name=name,
            personality=personality,
            background=background or f"Experienced {role} who works at the bar",
            location=Location(f"{role} station", position[0], position[1]),
            emotional_state=EmotionalState.NEUTRAL
        )
        
        self.role = role
        self.cognitive_mode = cognitive_mode
        
        # Initialize Reverie persona wrapper
        self.persona = PersonaWrapper(
            name=name,
            age=28,
            personality=personality,
            background=self.background,
            lifestyle=f"{name} is a {role} who works at a cozy bar, interacting with customers and maintaining the establishment"
        )
        
        # Enhanced cognitive state
        self.work_shift = {"start": "18:00", "end": "02:00"}  # 6 PM to 2 AM
        self.customer_relationships = {}  # Track relationships with customers
        self.work_priorities = ["customer_service", "cleanliness", "atmosphere"]
        self.current_focus = "customer_service"
        
        # Performance tracking
        self.cognitive_stats = {
            "perceptions": 0,
            "decisions": 0,
            "conversations": 0,
            "reflections": 0,
            "total_response_time": 0.0
        }
        
        # Initialize with basic daily plan
        self._initialize_daily_routine()
        
        print(f"[Enhanced Agent] {name} initialized with {role} role and {cognitive_mode.value} cognitive mode")
    
    def _initialize_daily_routine(self):
        """Set up basic daily routine and work priorities"""
        
        current_time = datetime.now()
        
        # Create initial plan if it's work hours
        hour = current_time.hour
        if 18 <= hour or hour <= 2:  # Work hours
            self.persona.plan("today")
            self.current_focus = "customer_service"
        else:
            self.current_focus = "preparation"
    
    def enhanced_perceive(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Enhanced perception using Reverie cognitive system"""
        
        start_time = datetime.now()
        
        try:
            # Use Reverie persona for sophisticated perception
            observations = self.persona.perceive(environment)
            
            # Enhance observations with role-specific insights
            role_insights = self._add_role_specific_insights(observations, environment)
            
            # Update customer relationships based on perceptions
            self._update_customer_relationships(environment)
            
            # Determine situational priorities
            priorities = self._assess_situational_priorities(observations, environment)
            
            result = {
                "observations": observations,
                "role_insights": role_insights,
                "priorities": priorities,
                "enhanced": True,
                "perception_time": (datetime.now() - start_time).total_seconds(),
                "cognitive_mode": self.cognitive_mode.value
            }
            
            self.cognitive_stats["perceptions"] += 1
            return result
            
        except Exception as e:
            print(f"[Enhanced Agent] Perception error: {e}")
            # Fallback to basic perception
            return self.perceive_with_cognition(environment)
    
    def enhanced_decide(self, environment: Dict[str, Any], perception_result: Dict[str, Any] = None) -> Dict[str, Any]:
        """Enhanced decision making using Reverie cognitive system"""
        
        start_time = datetime.now()
        
        try:
            # Get perception if not provided
            if not perception_result:
                perception_result = self.enhanced_perceive(environment)
            
            observations = perception_result.get("observations", [])
            priorities = perception_result.get("priorities", ["customer_service"])
            
            # Retrieve relevant memories for context
            relevant_memories = []
            for priority in priorities[:2]:  # Top 2 priorities
                memories = self.persona.retrieve(priority, top_k=3)
                relevant_memories.extend(memories)
            
            # Make decision using Reverie system
            decision = self.persona.decide(observations, relevant_memories)
            
            # Enhance decision with role-specific considerations
            enhanced_decision = self._enhance_decision_with_role_knowledge(
                decision, environment, priorities
            )
            
            # Adjust based on cognitive mode
            final_decision = self._adjust_decision_for_mode(enhanced_decision, environment)
            
            result = {
                "decision": final_decision,
                "observations_considered": len(observations),
                "memories_used": len(relevant_memories),
                "priorities": priorities,
                "enhanced": True,
                "decision_time": (datetime.now() - start_time).total_seconds(),
                "cognitive_mode": self.cognitive_mode.value
            }
            
            self.cognitive_stats["decisions"] += 1
            return result
            
        except Exception as e:
            print(f"[Enhanced Agent] Decision error: {e}")
            # Fallback to basic decision
            return self.decide_with_cognition(environment)
    
    def enhanced_converse(self, other_person: str, message: str, environment: Dict[str, Any]) -> str:
        """Enhanced conversation using Reverie cognitive system"""
        
        start_time = datetime.now()
        
        try:
            # Check relationship history
            relationship = self.customer_relationships.get(other_person, {})
            
            # Build conversation context
            conversation_context = {
                "location": environment.get("location", "bar"),
                "time": environment.get("time", datetime.now().strftime("%H:%M")),
                "relationship": relationship,
                "current_focus": self.current_focus,
                "role": self.role
            }
            
            # Generate response using Reverie system
            response = self.persona.converse(other_person, message, conversation_context)
            
            # Enhance with role-specific touches
            enhanced_response = self._add_role_specific_conversation_elements(
                response, other_person, message, environment
            )
            
            # Update relationship based on conversation
            self._update_relationship_from_conversation(
                other_person, message, enhanced_response
            )
            
            self.cognitive_stats["conversations"] += 1
            self.cognitive_stats["total_response_time"] += (datetime.now() - start_time).total_seconds()
            
            return enhanced_response
            
        except Exception as e:
            print(f"[Enhanced Agent] Conversation error: {e}")
            # Fallback to basic conversation
            return self.converse_with_cognition(other_person, message, environment)
    
    def enhanced_reflect(self, force_reflection: bool = False) -> Dict[str, Any]:
        """Enhanced reflection using Reverie cognitive system"""
        
        try:
            # Determine if reflection is needed
            should_reflect = (
                force_reflection or
                self.cognitive_stats["perceptions"] >= 10 or
                self.cognitive_stats["conversations"] >= 5 or
                datetime.now().hour in [22, 1]  # Natural reflection times
            )
            
            if not should_reflect:
                return {"reflections": [], "enhanced": False, "reason": "not needed yet"}
            
            # Perform reflection using Reverie system
            insights = self.persona.reflect()
            
            # Add role-specific reflections
            work_reflections = self._generate_work_reflections()
            
            # Analyze performance and adjust priorities
            performance_insights = self._analyze_performance()
            
            # Reset some statistics after reflection
            self.cognitive_stats["perceptions"] = 0
            self.cognitive_stats["conversations"] = 0
            
            result = {
                "insights": insights,
                "work_reflections": work_reflections,
                "performance_insights": performance_insights,
                "enhanced": True,
                "reflection_time": datetime.now().isoformat(),
                "cognitive_mode": self.cognitive_mode.value
            }
            
            self.cognitive_stats["reflections"] += 1
            return result
            
        except Exception as e:
            print(f"[Enhanced Agent] Reflection error: {e}")
            return {"reflections": [], "enhanced": False, "error": str(e)}
    
    def full_cognitive_cycle(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Complete cognitive cycle: perceive -> decide -> act -> reflect"""
        
        cycle_start = datetime.now()
        
        # 1. Enhanced Perception
        perception = self.enhanced_perceive(environment)
        
        # 2. Enhanced Decision Making
        decision = self.enhanced_decide(environment, perception)
        
        # 3. Execute Action (if needed)
        action_result = None
        if decision.get("decision", {}).get("action") != "continue current activity":
            action_result = self._execute_decision(decision, environment)
        
        # 4. Conversation (if people are present)
        conversations = []
        people = environment.get("people", [])
        for person in people[:2]:  # Limit to 2 people to avoid overwhelming
            if person != self.name:
                # Simulate natural conversation opportunity
                greeting = f"Hello {self.name}!" if person not in self.customer_relationships else f"Hi again, {self.name}!"
                response = self.enhanced_converse(person, greeting, environment)
                conversations.append({
                    "person": person,
                    "exchange": f"{person}: {greeting} | {self.name}: {response}"
                })
        
        # 5. Reflection (periodic)
        reflection = self.enhanced_reflect()
        
        # Compile cycle results
        cycle_time = (datetime.now() - cycle_start).total_seconds()
        
        cycle_result = {
            "cycle_time": cycle_time,
            "perception": perception,
            "decision": decision,
            "action_result": action_result,
            "conversations": conversations,
            "reflection": reflection,
            "agent_state": {
                "current_focus": self.current_focus,
                "emotional_state": self.emotional_state.value,
                "cognitive_mode": self.cognitive_mode.value,
                "relationships_count": len(self.customer_relationships)
            },
            "performance_stats": self.cognitive_stats.copy()
        }
        
        return cycle_result
    
    def _add_role_specific_insights(self, observations: List[str], environment: Dict[str, Any]) -> List[str]:
        """Add role-specific insights to observations"""
        
        insights = []
        
        if self.role == "bartender":
            # Bartender-specific observations
            if "customer" in str(observations).lower():
                insights.append("Customer service opportunity detected")
            if "drink" in str(observations).lower() or "glass" in str(observations).lower():
                insights.append("Beverage-related action may be needed")
            if "clean" in str(observations).lower() or "dirty" in str(observations).lower():
                insights.append("Cleanliness maintenance required")
        
        return insights
    
    def _assess_situational_priorities(self, observations: List[str], environment: Dict[str, Any]) -> List[str]:
        """Assess current situational priorities"""
        
        priorities = self.work_priorities.copy()
        
        # Adjust priorities based on observations
        obs_text = " ".join(observations).lower()
        
        if "busy" in obs_text or len(environment.get("people", [])) > 3:
            priorities = ["customer_service"] + [p for p in priorities if p != "customer_service"]
        
        if "dirty" in obs_text or "clean" in obs_text:
            if "cleanliness" not in priorities[:2]:
                priorities.insert(1, "cleanliness")
        
        return priorities[:3]  # Top 3 priorities
    
    def _enhance_decision_with_role_knowledge(self, decision: Dict[str, Any], environment: Dict[str, Any], priorities: List[str]) -> Dict[str, Any]:
        """Enhance decision with role-specific knowledge"""
        
        enhanced = decision.copy()
        
        if self.role == "bartender":
            # Add bartender-specific context
            enhanced["professional_context"] = "As an experienced bartender, focusing on customer satisfaction and bar atmosphere"
            
            # Adjust action based on current priorities
            if "customer_service" in priorities[:2] and "customer" in str(environment.get("people", [])):
                enhanced["action"] = enhanced.get("action", "greet and assist customers")
        
        return enhanced
    
    def _adjust_decision_for_mode(self, decision: Dict[str, Any], environment: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust decision based on cognitive mode"""
        
        if self.cognitive_mode == CognitiveMode.REACTIVE:
            # Quick, immediate responses
            decision["urgency"] = "immediate"
            decision["planning_depth"] = "minimal"
        
        elif self.cognitive_mode == CognitiveMode.DELIBERATIVE:
            # Thoughtful, considered responses
            decision["urgency"] = "considered"
            decision["planning_depth"] = "thorough"
        
        else:  # ADAPTIVE
            # Adjust based on situation
            people_count = len(environment.get("people", []))
            if people_count > 2:
                decision["urgency"] = "immediate"  # Busy situation
            else:
                decision["urgency"] = "considered"  # Calm situation
        
        return decision
    
    def _add_role_specific_conversation_elements(self, response: str, person: str, message: str, environment: Dict[str, Any]) -> str:
        """Add role-specific elements to conversation"""
        
        if self.role == "bartender" and len(response) > 20:
            # Ensure bartender responses feel authentic
            if not any(word in response.lower() for word in ["drink", "bar", "serve", "help", "what can", "welcome"]):
                # Add a subtle bartender touch if missing
                response = response.rstrip() + " What can I get for you tonight?"
        
        return response
    
    def _update_customer_relationships(self, environment: Dict[str, Any]):
        """Update customer relationship tracking"""
        
        for person in environment.get("people", []):
            if person != self.name and person not in ["staff", "employee"]:
                if person not in self.customer_relationships:
                    self.customer_relationships[person] = {
                        "first_seen": datetime.now().isoformat(),
                        "visit_count": 1,
                        "last_conversation": None,
                        "preferences": {},
                        "relationship_level": "new"
                    }
                else:
                    self.customer_relationships[person]["visit_count"] += 1
    
    def _update_relationship_from_conversation(self, person: str, their_message: str, my_response: str):
        """Update relationship based on conversation"""
        
        if person in self.customer_relationships:
            relationship = self.customer_relationships[person]
            relationship["last_conversation"] = {
                "their_message": their_message,
                "my_response": my_response,
                "timestamp": datetime.now().isoformat()
            }
            
            # Simple relationship level progression
            if relationship["visit_count"] > 5:
                relationship["relationship_level"] = "regular"
            elif relationship["visit_count"] > 2:
                relationship["relationship_level"] = "familiar"
    
    def _execute_decision(self, decision: Dict[str, Any], environment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a decision and return results"""
        
        action = decision.get("decision", {}).get("action", "continue")
        
        # Simple execution simulation
        execution_result = {
            "action_taken": action,
            "success": True,
            "duration": 1.0,  # Simulated 1 second
            "environment_effect": f"Performed {action}"
        }
        
        return execution_result
    
    def _generate_work_reflections(self) -> List[str]:
        """Generate work-specific reflections"""
        
        reflections = []
        
        # Reflect on customer interactions
        if self.cognitive_stats["conversations"] > 0:
            avg_response_time = self.cognitive_stats["total_response_time"] / max(1, self.cognitive_stats["conversations"])
            if avg_response_time > 5.0:
                reflections.append("I should work on responding more quickly to customers")
            else:
                reflections.append("I'm maintaining good response times with customers")
        
        # Reflect on customer relationships
        regular_customers = sum(1 for rel in self.customer_relationships.values() 
                              if rel.get("relationship_level") == "regular")
        if regular_customers > 0:
            reflections.append(f"I'm building good relationships - {regular_customers} regular customers now")
        
        return reflections
    
    def _analyze_performance(self) -> Dict[str, Any]:
        """Analyze current performance and suggest improvements"""
        
        analysis = {
            "efficiency": "good" if self.cognitive_stats["decisions"] > 0 else "needs_improvement",
            "customer_engagement": "good" if self.cognitive_stats["conversations"] > 0 else "low",
            "reflection_frequency": "good" if self.cognitive_stats["reflections"] > 0 else "needs_improvement"
        }
        
        return analysis
    
    def get_enhanced_status(self) -> Dict[str, Any]:
        """Get comprehensive status of enhanced agent"""
        
        base_status = self.get_cognitive_summary()
        persona_memory = self.persona.get_memory_summary()
        
        enhanced_status = {
            "base_agent": base_status,
            "persona_memory": persona_memory,
            "cognitive_mode": self.cognitive_mode.value,
            "role": self.role,
            "current_focus": self.current_focus,
            "customer_relationships": len(self.customer_relationships),
            "performance_stats": self.cognitive_stats,
            "work_shift": self.work_shift
        }
        
        return enhanced_status
    
    def cleanup(self):
        """Clean up resources"""
        
        super().cleanup()
        self.persona.cleanup()
        
        # Save important state if needed
        print(f"[Enhanced Agent] {self.name} cleaned up. Final stats: {self.cognitive_stats}")
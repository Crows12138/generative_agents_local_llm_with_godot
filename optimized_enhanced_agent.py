"""Optimized Enhanced Bar Agent - Performance Version

This agent integrates all performance optimizations to achieve:
- 5-10 second cognitive cycles (vs 50+ seconds)
- Intelligent caching and prompt optimization
- Smart degradation based on system load
- Parallel processing support
"""

import sys
import os
import time
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from enum import Enum

# Add project paths
project_root = os.path.dirname(__file__)
sys.path.insert(0, project_root)

from enhanced_bar_agent import EnhancedBarAgent, CognitiveMode
from optimization.performance_optimizer import performance_optimizer, selective_cognition
from optimization.smart_degradation import smart_degradation, LoadLevel
from reverie_integration.persona_wrapper import PersonaWrapper

class OptimizedEnhancedAgent(EnhancedBarAgent):
    """Performance-optimized version of Enhanced Bar Agent"""
    
    def __init__(self, 
                 name: str,
                 role: str,
                 position: Tuple[int, int],
                 personality: str = "friendly bartender",
                 background: str = None,
                 cognitive_mode: CognitiveMode = CognitiveMode.ADAPTIVE,
                 performance_mode: str = "balanced"):
        
        # Initialize base agent
        super().__init__(name, role, position, personality, background, cognitive_mode)
        
        # Performance optimization settings
        self.performance_mode = performance_mode  # fast, balanced, quality
        performance_optimizer.set_performance_mode(performance_mode)
        
        # Optimization state
        self.optimization_active = True
        self.cache_enabled = True
        self.smart_degradation_enabled = True
        
        # Performance tracking
        self.optimized_stats = {
            "cache_hits": 0,
            "fast_responses": 0,
            "degraded_responses": 0,
            "avg_cycle_time": 0,
            "total_optimized_cycles": 0
        }
        
        print(f"[Optimized Agent] {name} initialized with {performance_mode} performance mode")
    
    def fast_cognitive_cycle(self, environment: Dict[str, Any]) -> Dict[str, Any]:
        """Ultra-fast cognitive cycle with all optimizations enabled"""
        
        cycle_start = time.time()
        
        # Step 1: Smart degradation assessment
        if self.smart_degradation_enabled:
            current_load = smart_degradation.assess_system_load()
            load_config = smart_degradation.get_optimal_config(current_load)
            
            if current_load == LoadLevel.CRITICAL:
                # Emergency fast response
                return self._emergency_response(environment, cycle_start)
        else:
            load_config = {"modules": "all", "max_tokens": 400}
        
        # Step 2: Selective cognition check
        context_summary = self._build_context_summary(environment)
        agent_state = self._get_agent_state()
        
        if not selective_cognition.should_use_full_cognition(context_summary, agent_state):
            # Quick response path
            return self._quick_cognitive_response(environment, context_summary, cycle_start)
        
        # Step 3: Optimized full cognitive cycle
        return self._optimized_full_cycle(environment, load_config, cycle_start)
    
    def _emergency_response(self, environment: Dict[str, Any], cycle_start: float) -> Dict[str, Any]:
        """Emergency response for critical system load"""
        
        # Use only cached/predefined responses
        context = str(environment.get("events", [])).lower()
        
        if "hello" in context or "customer" in context:
            response = "Hello! Welcome, what can I get for you?"
            action = "greet_customer"
        elif "order" in context or "drink" in context:
            response = "Coming right up!"
            action = "prepare_drink"
        else:
            response = "I'm here to help when you need me."
            action = "remain_attentive"
        
        cycle_time = time.time() - cycle_start
        self._update_optimized_stats("emergency", cycle_time)
        
        return {
            "mode": "emergency",
            "cycle_time": cycle_time,
            "response": response,
            "action": action,
            "enhanced": False,
            "degradation_level": "critical"
        }
    
    def _quick_cognitive_response(self, environment: Dict[str, Any], context: str, cycle_start: float) -> Dict[str, Any]:
        """Quick response for simple situations"""
        
        # Try cached/quick response first
        quick_response = selective_cognition.get_quick_response(context, "conversation")
        
        if quick_response:
            cycle_time = time.time() - cycle_start
            self._update_optimized_stats("cached", cycle_time)
            
            return {
                "mode": "quick",
                "cycle_time": cycle_time,
                "response": quick_response,
                "action": "continue_current_activity",
                "enhanced": False,
                "cache_hit": True
            }
        
        # Generate optimized response
        optimized_prompt = f"As {self.name} the {self.role}, respond to: {context[:200]}"
        
        response = performance_optimizer.optimize_llm_call(
            optimized_prompt, 
            "conversation",
            context={"character": self.name, "role": self.role}
        )
        
        cycle_time = time.time() - cycle_start
        self._update_optimized_stats("quick", cycle_time)
        
        return {
            "mode": "quick",
            "cycle_time": cycle_time,
            "response": response,
            "action": "respond_naturally",
            "enhanced": True,
            "optimized": True
        }
    
    def _optimized_full_cycle(self, environment: Dict[str, Any], load_config: Dict, cycle_start: float) -> Dict[str, Any]:
        """Full cognitive cycle with all optimizations"""
        
        results = {}
        
        # Step 1: Optimized Perception
        if not smart_degradation.should_skip_module("perception", load_config):
            perception_start = time.time()
            perception = self._optimized_perception(environment, load_config)
            results["perception"] = perception
            results["perception_time"] = time.time() - perception_start
        else:
            results["perception"] = {"observations": ["basic environment scan"], "skipped": True}
            results["perception_time"] = 0
        
        # Step 2: Optimized Decision Making
        if not smart_degradation.should_skip_module("decision", load_config):
            decision_start = time.time()
            decision = self._optimized_decision(environment, results.get("perception"), load_config)
            results["decision"] = decision
            results["decision_time"] = time.time() - decision_start
        else:
            results["decision"] = {"action": "continue_current_activity", "skipped": True}
            results["decision_time"] = 0
        
        # Step 3: Optimized Conversation (if people present)
        conversations = []
        people = environment.get("people", [])
        if people and not smart_degradation.should_skip_module("conversation", load_config):
            conv_start = time.time()
            
            # Limit conversations based on load
            max_conversations = 3 if load_config.get("modules") == "all" else 1
            
            for person in people[:max_conversations]:
                if person != self.name:
                    conv = self._optimized_conversation(person, environment, load_config)
                    if conv:
                        conversations.append(conv)
            
            results["conversations"] = conversations
            results["conversation_time"] = time.time() - conv_start
        else:
            results["conversations"] = []
            results["conversation_time"] = 0
        
        # Step 4: Selective Reflection
        if (not smart_degradation.should_skip_module("reflection", load_config) and
            self.cognitive_stats["conversations"] >= 3):
            reflection_start = time.time()
            reflection = self._optimized_reflection(load_config)
            results["reflection"] = reflection
            results["reflection_time"] = time.time() - reflection_start
        else:
            results["reflection"] = {"insights": [], "skipped": True}
            results["reflection_time"] = 0
        
        # Final results
        total_cycle_time = time.time() - cycle_start
        self._update_optimized_stats("full", total_cycle_time)
        
        results.update({
            "mode": "optimized_full",
            "total_cycle_time": total_cycle_time,
            "load_level": load_config.get("load_level", "unknown"),
            "optimizations_active": True,
            "performance_mode": self.performance_mode
        })
        
        return results
    
    def _optimized_perception(self, environment: Dict[str, Any], load_config: Dict) -> Dict[str, Any]:
        """Optimized perception with smart prompting"""
        
        # Build efficient perception prompt
        events = environment.get("events", [])[:3]  # Limit events for speed
        objects = environment.get("objects", [])[:5]  # Limit objects
        
        context = {
            "character": self.name,
            "location": environment.get("location", "workplace"),
            "events": ", ".join(events),
            "objects": ", ".join(objects)
        }
        
        # Use optimized prompt based on load
        if load_config.get("modules") == "minimal":
            prompt = f"{self.name} at {context['location']}: {context['events']}. Top 3 observations:"
        else:
            prompt = f"Character {self.name} observing {context['location']}. Events: {context['events']}. Key insights (max 5):"
        
        # Generate with performance optimization
        observations = performance_optimizer.optimize_llm_call(prompt, "list", context)
        
        # Parse observations
        observation_list = []
        for line in observations.split('\n'):
            line = line.strip()
            if line and len(line) > 10:
                cleaned = line.lstrip('0123456789.- ')
                if cleaned:
                    observation_list.append(cleaned)
        
        return {
            "observations": observation_list[:5],  # Limit for speed
            "priorities": self._quick_priority_assessment(observation_list, environment),
            "optimized": True
        }
    
    def _optimized_decision(self, environment: Dict[str, Any], perception: Dict, load_config: Dict) -> Dict[str, Any]:
        """Optimized decision making"""
        
        observations = perception.get("observations", []) if perception else []
        
        # Build decision context
        situation = f"{self.name} sees: {'; '.join(observations[:3])}"
        
        # Use optimized prompt
        if load_config.get("modules") == "minimal":
            prompt = f"{situation}. Best action:"
        else:
            prompt = f"Character: {self.name}\nSituation: {situation}\nChoose action with brief reason:"
        
        # Generate decision
        decision_text = performance_optimizer.optimize_llm_call(prompt, "action", {
            "character": self.name,
            "situation": situation
        })
        
        # Parse decision
        action = decision_text.split('.')[0] if decision_text else "continue_current_activity"
        
        return {
            "action": action,
            "reasoning": decision_text,
            "observations_considered": len(observations),
            "optimized": True
        }
    
    def _optimized_conversation(self, person: str, environment: Dict[str, Any], load_config: Dict) -> Optional[Dict[str, Any]]:
        """Optimized conversation generation"""
        
        # Check relationship for context
        relationship = self.customer_relationships.get(person, {})
        
        # Generate appropriate greeting/response
        if relationship.get("relationship_level") == "new":
            message = f"Hello! Welcome to {environment.get('location', 'the bar')}."
        else:
            message = f"Good to see you again, {person}!"
        
        # Use optimized conversation prompt
        context = {
            "speaker": self.name,
            "listener": person,
            "character": self.name,
            "relationship": relationship.get("relationship_level", "new")
        }
        
        prompt = f"{self.name} greets {person}. Respond naturally as friendly {self.role}:"
        
        response = performance_optimizer.optimize_llm_call(prompt, "conversation", context)
        
        # Update relationship tracking
        self._quick_relationship_update(person)
        
        return {
            "person": person,
            "message": message,
            "response": response,
            "optimized": True
        }
    
    def _optimized_reflection(self, load_config: Dict) -> Dict[str, Any]:
        """Optimized reflection process"""
        
        # Quick reflection based on recent activity
        recent_conversations = self.cognitive_stats["conversations"]
        
        if recent_conversations > 0:
            insight = f"Served {recent_conversations} customers effectively"
        else:
            insight = "Maintained professional readiness"
        
        return {
            "insights": [insight],
            "optimized": True,
            "quick_reflection": True
        }
    
    def _build_context_summary(self, environment: Dict[str, Any]) -> str:
        """Build quick context summary for decision making"""
        
        parts = []
        
        if environment.get("people"):
            parts.append(f"{len(environment['people'])} people present")
        
        if environment.get("events"):
            parts.append(f"events: {', '.join(environment['events'][:2])}")
        
        return "; ".join(parts)
    
    def _get_agent_state(self) -> Dict[str, Any]:
        """Get current agent state for optimization decisions"""
        
        return {
            "last_interaction_time": time.time() - 60,  # Simulate recent interaction
            "customer_count": len(self.customer_relationships),
            "current_focus": self.current_focus,
            "conversation_count": self.cognitive_stats.get("conversations", 0)
        }
    
    def _quick_priority_assessment(self, observations: List[str], environment: Dict[str, Any]) -> List[str]:
        """Quick priority assessment for optimization"""
        
        priorities = ["customer_service"]
        
        obs_text = " ".join(observations).lower()
        
        if "busy" in obs_text or len(environment.get("people", [])) > 2:
            priorities.insert(0, "efficiency")
        
        if "clean" in obs_text:
            priorities.append("cleanliness")
        
        return priorities[:3]
    
    def _quick_relationship_update(self, person: str):
        """Quick relationship update for performance"""
        
        if person not in self.customer_relationships:
            self.customer_relationships[person] = {
                "first_seen": datetime.now().isoformat(),
                "visit_count": 1,
                "relationship_level": "new"
            }
        else:
            rel = self.customer_relationships[person]
            rel["visit_count"] = rel.get("visit_count", 0) + 1
            
            if rel["visit_count"] > 3:
                rel["relationship_level"] = "familiar"
    
    def _update_optimized_stats(self, response_type: str, cycle_time: float):
        """Update optimization statistics"""
        
        self.optimized_stats["total_optimized_cycles"] += 1
        
        if response_type == "cached":
            self.optimized_stats["cache_hits"] += 1
        elif response_type in ["quick", "emergency"]:
            self.optimized_stats["fast_responses"] += 1
        
        # Update average cycle time
        total_cycles = self.optimized_stats["total_optimized_cycles"]
        current_avg = self.optimized_stats["avg_cycle_time"]
        new_avg = ((current_avg * (total_cycles - 1)) + cycle_time) / total_cycles
        self.optimized_stats["avg_cycle_time"] = new_avg
        
        # Record response time for degradation system
        smart_degradation.record_response_time(cycle_time)
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        
        base_status = self.get_enhanced_status()
        perf_report = performance_optimizer.get_performance_report()
        degradation_status = smart_degradation.get_degradation_status()
        
        return {
            "base_status": base_status,
            "performance_mode": self.performance_mode,
            "optimization_stats": self.optimized_stats,
            "performance_report": perf_report,
            "degradation_status": degradation_status,
            "optimizations_enabled": {
                "caching": self.cache_enabled,
                "smart_degradation": self.smart_degradation_enabled,
                "selective_cognition": True,
                "prompt_optimization": True
            }
        }
    
    def set_optimization_level(self, level: str):
        """Set optimization level: maximum, balanced, conservative"""
        
        if level == "maximum":
            self.performance_mode = "fast"
            self.cache_enabled = True
            self.smart_degradation_enabled = True
            performance_optimizer.set_performance_mode("fast")
            
        elif level == "balanced":
            self.performance_mode = "balanced" 
            self.cache_enabled = True
            self.smart_degradation_enabled = True
            performance_optimizer.set_performance_mode("balanced")
            
        elif level == "conservative":
            self.performance_mode = "quality"
            self.cache_enabled = False
            self.smart_degradation_enabled = False
            performance_optimizer.set_performance_mode("quality")
        
        print(f"[Optimized Agent] {self.name} optimization level set to: {level}")

def create_optimized_bar_staff() -> List[OptimizedEnhancedAgent]:
    """Create a set of optimized bar staff agents"""
    
    agents = [
        OptimizedEnhancedAgent(
            name="Bob",
            role="head_bartender",
            position=(5, 2),
            personality="experienced, efficient leader",
            performance_mode="balanced"
        ),
        OptimizedEnhancedAgent(
            name="Alice",
            role="bartender",
            position=(7, 2), 
            personality="friendly, quick-service focused",
            performance_mode="fast"
        ),
        OptimizedEnhancedAgent(
            name="Charlie",
            role="server",
            position=(3, 5),
            personality="attentive, customer-focused",
            performance_mode="balanced"
        )
    ]
    
    return agents
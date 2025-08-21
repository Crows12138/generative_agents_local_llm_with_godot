"""
Optimized Bar Agent System
Extends the original bar_agents.py with performance optimizations and enhanced monitoring
"""

import time
import threading
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# Import original agent components
try:
    from .bar_agents import BarAgent, SimpleAgent, Memory
except ImportError:
    from bar_agents import BarAgent, SimpleAgent, Memory

# Import optimization components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'debug_system'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'ai_service'))

try:
    from debug_system.flow_tracer import get_tracer
    from debug_system.performance_analyzer import get_performance_analyzer
    from debug_system.debug_dashboard import DebugDashboard
    from ai_service.optimized_ai_service import get_optimized_ai_service
    from ai_service.optimized_unified_parser import get_optimized_unified_parser
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("[optimized_bar_agents] Optimization components not available")


@dataclass
class AgentAction:
    """Enhanced action with optimization metadata"""
    action_type: str
    target: Optional[str]
    timestamp: datetime
    confidence: float
    processing_time: float
    method: str  # "cached", "fast_path", "full_parse"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "timestamp": self.timestamp.isoformat(),
            "confidence": self.confidence,
            "processing_time": self.processing_time,
            "method": self.method
        }


class OptimizedBarAgent(BarAgent):
    """Enhanced Bar Agent with integrated optimizations"""
    
    def __init__(self, name: str, role: str, position: Tuple[int, int]):
        super().__init__(name, role, position)
        
        # Initialize optimization components
        if OPTIMIZATION_AVAILABLE:
            self._setup_optimizations()
        
        # Enhanced tracking
        self.action_history: List[AgentAction] = []
        self.total_actions = 0
        self.cached_decisions = 0
        self.fast_path_decisions = 0
        
        # Performance metrics
        self.avg_decision_time = 0.0
        self.last_optimization_stats = {}
        
        # Thread safety
        self.lock = threading.RLock()
    
    def _setup_optimizations(self) -> None:
        """Setup optimization components"""
        # Get optimized services
        self.ai_service = get_optimized_ai_service()
        self.parser = get_optimized_unified_parser()
        self.flow_tracer = get_tracer()
        self.perf_analyzer = get_performance_analyzer()
        
        # Optimize for agent workload
        self.ai_service.optimize_for_scenario("production")
        self.parser.optimize_for_workload("action_heavy")
        
        print(f"[{self.name}] Optimizations initialized")
    
    def perceive(self, environment: Dict[str, Any]) -> str:
        """Enhanced perception with flow tracing"""
        
        if OPTIMIZATION_AVAILABLE:
            # Start flow tracing
            perception_data = self._format_perception(environment)
            self.flow_tracer.trace_perception(self.name, perception_data)
            return perception_data
        else:
            return super().perceive(environment)
    
    def _format_perception(self, environment: Dict[str, Any]) -> str:
        """Format perception data efficiently"""
        # Quick perception formatting for common scenarios
        if "customers" in environment:
            customer_count = len(environment.get("customers", []))
            if customer_count == 0:
                return "Bar is quiet, no customers present"
            elif customer_count == 1:
                customer = environment["customers"][0]
                return f"One customer {customer.get('name', 'unknown')} is at the bar"
            else:
                return f"Bar is busy with {customer_count} customers"
        
        return "Standard bar environment"
    
    def decide_action(self, perception: str, use_optimizations: bool = True) -> AgentAction:
        """Enhanced decision making with optimizations"""
        
        with self.lock:
            self.total_actions += 1
            start_time = time.time()
            
            if OPTIMIZATION_AVAILABLE and use_optimizations:
                return self._optimized_decision(perception)
            else:
                return self._basic_decision(perception)
    
    def _optimized_decision(self, perception: str) -> AgentAction:
        """Optimized decision making process"""
        
        # Check for simple pattern-based decisions first
        quick_decision = self._quick_decision_check(perception)
        if quick_decision:
            self.fast_path_decisions += 1
            processing_time = time.time() - time.time()  # Near zero
            
            return AgentAction(
                action_type=quick_decision["action"],
                target=quick_decision.get("target"),
                timestamp=datetime.now(),
                confidence=quick_decision["confidence"],
                processing_time=processing_time,
                method="fast_path"
            )
        
        # Use optimized AI service for complex decisions
        prompt = self._build_decision_prompt(perception)
        
        try:
            # Generate with optimizations
            response = self.ai_service.generate(
                prompt,
                use_optimizations=True,
                agent_name=self.name
            )
            
            # Parse with optimized parser
            parse_result = self.parser.parse_action(response, use_optimizations=True)
            
            # Check if this was a cache hit
            method = "cached" if "cached" in str(parse_result.metadata) else "full_parse"
            if method == "cached":
                self.cached_decisions += 1
            
            processing_time = time.time() - start_time
            
            return AgentAction(
                action_type=parse_result.value.get("action", "idle"),
                target=parse_result.value.get("target"),
                timestamp=datetime.now(),
                confidence=parse_result.confidence,
                processing_time=processing_time,
                method=method
            )
            
        except Exception as e:
            print(f"[{self.name}] Decision error: {e}")
            return self._fallback_decision(perception)
    
    def _quick_decision_check(self, perception: str) -> Optional[Dict[str, Any]]:
        """Quick pattern-based decision for common scenarios"""
        
        perception_lower = perception.lower()
        
        # Common bar scenarios with quick decisions
        if "customer" in perception_lower and "waiting" in perception_lower:
            return {"action": "serve", "target": "customer", "confidence": 0.85}
        
        if "empty" in perception_lower or "quiet" in perception_lower:
            return {"action": "clean", "target": "bar", "confidence": 0.8}
        
        if "busy" in perception_lower:
            return {"action": "prioritize", "target": "customers", "confidence": 0.9}
        
        if "inventory" in perception_lower or "stock" in perception_lower:
            return {"action": "check", "target": "inventory", "confidence": 0.75}
        
        return None
    
    def _build_decision_prompt(self, perception: str) -> str:
        """Build optimized decision prompt"""
        
        # Optimized prompt that's shorter but effective
        base_prompt = f"Bartender {self.name} observes: {perception}. Action?"
        
        # Add context based on recent memories
        if self.memories:
            recent_memory = self.memories[-1]
            if "customer" in recent_memory.content.lower():
                base_prompt += " Consider customer service priority."
        
        return base_prompt
    
    def _basic_decision(self, perception: str) -> AgentAction:
        """Basic decision making fallback"""
        
        # Simple rule-based decisions
        if "customer" in perception.lower():
            action_type = "serve"
            target = "customer"
        elif "clean" in perception.lower():
            action_type = "clean"
            target = "bar"
        else:
            action_type = "idle"
            target = None
        
        processing_time = time.time() - time.time()  # Near zero
        
        return AgentAction(
            action_type=action_type,
            target=target,
            timestamp=datetime.now(),
            confidence=0.7,
            processing_time=processing_time,
            method="basic_rules"
        )
    
    def _fallback_decision(self, perception: str) -> AgentAction:
        """Fallback decision when optimizations fail"""
        
        return AgentAction(
            action_type="idle",
            target=None,
            timestamp=datetime.now(),
            confidence=0.5,
            processing_time=0.001,
            method="fallback"
        )
    
    def execute_action(self, action: AgentAction) -> Dict[str, Any]:
        """Execute action with tracing"""
        
        if OPTIMIZATION_AVAILABLE:
            # Trace execution
            self.flow_tracer.trace_execution(
                action.to_dict(),
                {"status": "executed", "agent": self.name}
            )
        
        # Record action in history
        self.action_history.append(action)
        
        # Keep history limited
        if len(self.action_history) > 100:
            self.action_history = self.action_history[-50:]
        
        # Update performance metrics
        self._update_performance_metrics(action)
        
        # Execute the action (simplified for demo)
        result = {
            "action": action.action_type,
            "target": action.target,
            "success": True,
            "timestamp": action.timestamp.isoformat()
        }
        
        # Add memory of the action
        self.add_memory(
            f"Performed {action.action_type} on {action.target or 'general'}",
            emotion="neutral",
            importance=0.6
        )
        
        return result
    
    def _update_performance_metrics(self, action: AgentAction) -> None:
        """Update performance metrics"""
        
        # Update average decision time
        if self.total_actions > 0:
            self.avg_decision_time = (
                (self.avg_decision_time * (self.total_actions - 1) + action.processing_time)
                / self.total_actions
            )
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        
        with self.lock:
            cache_rate = (self.cached_decisions / self.total_actions * 100) if self.total_actions > 0 else 0
            fast_path_rate = (self.fast_path_decisions / self.total_actions * 100) if self.total_actions > 0 else 0
            
            stats = {
                "agent_name": self.name,
                "agent_role": self.role,
                "total_actions": self.total_actions,
                "cached_decisions": self.cached_decisions,
                "fast_path_decisions": self.fast_path_decisions,
                "cache_rate_percent": cache_rate,
                "fast_path_rate_percent": fast_path_rate,
                "avg_decision_time_ms": self.avg_decision_time * 1000,
                "memory_count": len(self.memories),
                "action_history_size": len(self.action_history)
            }
            
            # Add optimization stats if available
            if OPTIMIZATION_AVAILABLE:
                if hasattr(self, 'ai_service'):
                    stats["ai_service_stats"] = self.ai_service.get_optimization_stats()
                if hasattr(self, 'parser'):
                    stats["parser_stats"] = self.parser.get_optimization_stats()
            
            return stats
    
    def optimize_for_scenario(self, scenario: str) -> None:
        """Optimize agent for specific scenarios"""
        
        if not OPTIMIZATION_AVAILABLE:
            return
        
        if scenario == "busy_bar":
            # Optimize for high-throughput decisions
            self.ai_service.optimize_for_scenario("high_throughput")
            self.parser.optimize_for_workload("action_heavy")
            
        elif scenario == "quiet_bar":
            # Optimize for quality and detailed decisions
            self.ai_service.optimize_for_scenario("production")
            self.parser.optimize_for_workload("mixed")
            
        elif scenario == "debugging":
            # Full monitoring for debugging
            self.ai_service.optimize_for_scenario("debugging")
            
        print(f"[{self.name}] Optimized for {scenario} scenario")
    
    def generate_agent_report(self, filename: Optional[str] = None) -> str:
        """Generate comprehensive agent performance report"""
        
        if not filename:
            filename = f"{self.name}_performance_report.md"
        
        stats = self.get_performance_stats()
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write(f"# Agent Performance Report: {self.name}\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n")
            f.write(f"**Role**: {self.role}\n")
            f.write(f"**Position**: {self.position}\n\n")
            
            f.write("## Performance Statistics\n\n")
            f.write(f"- Total actions: {stats['total_actions']}\n")
            f.write(f"- Average decision time: {stats['avg_decision_time_ms']:.2f}ms\n")
            f.write(f"- Cache hit rate: {stats['cache_rate_percent']:.1f}%\n")
            f.write(f"- Fast path rate: {stats['fast_path_rate_percent']:.1f}%\n\n")
            
            f.write("## Recent Actions\n\n")
            for action in self.action_history[-10:]:
                f.write(f"- {action.timestamp.strftime('%H:%M:%S')}: ")
                f.write(f"{action.action_type}({action.target}) ")
                f.write(f"[{action.method}, {action.processing_time*1000:.1f}ms]\n")
            
            if OPTIMIZATION_AVAILABLE and "ai_service_stats" in stats:
                ai_stats = stats["ai_service_stats"]
                f.write("\n## AI Service Statistics\n\n")
                f.write(f"- Total LLM calls: {ai_stats.get('llm_stats', {}).get('total_calls', 0)}\n")
                f.write(f"- LLM cache hit rate: {ai_stats.get('llm_stats', {}).get('cache_hit_rate_percent', 0):.1f}%\n")
        
        print(f"[{self.name}] Report saved to {filename}")
        return filename


class OptimizedBarEnvironment:
    """Optimized bar environment with monitoring"""
    
    def __init__(self):
        self.agents: Dict[str, OptimizedBarAgent] = {}
        self.environment_state = {
            "customers": [],
            "bar_cleanliness": 100,
            "inventory_level": 100,
            "time_of_day": "afternoon"
        }
        
        # Initialize monitoring if available
        if OPTIMIZATION_AVAILABLE:
            self.dashboard = DebugDashboard()
            self.dashboard.start_monitoring = lambda: None  # Disable auto-start
        
        self.step_count = 0
    
    def add_agent(self, agent: OptimizedBarAgent) -> None:
        """Add agent to environment"""
        self.agents[agent.name] = agent
        
        if OPTIMIZATION_AVAILABLE:
            self.dashboard.add_agent(agent.name, agent.position, "active")
    
    def simulate_step(self) -> Dict[str, Any]:
        """Simulate one environment step with all agents"""
        
        self.step_count += 1
        step_results = {}
        
        for agent_name, agent in self.agents.items():
            # Agent perceives environment
            perception = agent.perceive(self.environment_state)
            
            # Agent decides action
            action = agent.decide_action(perception)
            
            # Agent executes action
            result = agent.execute_action(action)
            
            # Update dashboard if available
            if OPTIMIZATION_AVAILABLE:
                self.dashboard.update_agent(
                    agent_name,
                    action=action.action_type,
                    target=action.target,
                    position=agent.position
                )
                
                self.dashboard.add_event(
                    "action",
                    agent_name,
                    f"{action.action_type} on {action.target or 'general'}",
                    "info"
                )
            
            step_results[agent_name] = {
                "perception": perception,
                "action": action.to_dict(),
                "result": result
            }
        
        return step_results
    
    def get_environment_stats(self) -> Dict[str, Any]:
        """Get comprehensive environment statistics"""
        
        total_actions = sum(agent.total_actions for agent in self.agents.values())
        total_cached = sum(agent.cached_decisions for agent in self.agents.values())
        total_fast_path = sum(agent.fast_path_decisions for agent in self.agents.values())
        
        stats = {
            "step_count": self.step_count,
            "agent_count": len(self.agents),
            "total_actions": total_actions,
            "total_cached_decisions": total_cached,
            "total_fast_path_decisions": total_fast_path,
            "cache_rate_percent": (total_cached / total_actions * 100) if total_actions > 0 else 0,
            "fast_path_rate_percent": (total_fast_path / total_actions * 100) if total_actions > 0 else 0,
            "agent_stats": {name: agent.get_performance_stats() for name, agent in self.agents.items()}
        }
        
        return stats
    
    def generate_environment_report(self, filename: str = "bar_environment_report.md") -> None:
        """Generate comprehensive environment report"""
        
        stats = self.get_environment_stats()
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write("# Bar Environment Performance Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            
            f.write("## Environment Statistics\n\n")
            f.write(f"- Simulation steps: {stats['step_count']}\n")
            f.write(f"- Active agents: {stats['agent_count']}\n")
            f.write(f"- Total actions: {stats['total_actions']}\n")
            f.write(f"- Overall cache rate: {stats['cache_rate_percent']:.1f}%\n")
            f.write(f"- Overall fast path rate: {stats['fast_path_rate_percent']:.1f}%\n\n")
            
            f.write("## Agent Performance Summary\n\n")
            for agent_name, agent_stats in stats["agent_stats"].items():
                f.write(f"### {agent_name} ({agent_stats['agent_role']})\n")
                f.write(f"- Actions: {agent_stats['total_actions']}\n")
                f.write(f"- Avg decision time: {agent_stats['avg_decision_time_ms']:.2f}ms\n")
                f.write(f"- Cache rate: {agent_stats['cache_rate_percent']:.1f}%\n")
                f.write(f"- Fast path rate: {agent_stats['fast_path_rate_percent']:.1f}%\n\n")
        
        print(f"Environment report saved to {filename}")


def demo_optimized_bar_agents():
    """Demo the optimized bar agent system"""
    print("=== Optimized Bar Agents Demo ===\n")
    
    # Create environment
    env = OptimizedBarEnvironment()
    
    # Create optimized agents
    alice = OptimizedBarAgent("Alice", "bartender", (5, 2))
    bob = OptimizedBarAgent("Bob", "customer", (3, 4))
    carol = OptimizedBarAgent("Carol", "waiter", (7, 6))
    
    # Add agents to environment
    for agent in [alice, bob, carol]:
        env.add_agent(agent)
    
    # Optimize agents for scenario
    alice.optimize_for_scenario("busy_bar")
    bob.optimize_for_scenario("quiet_bar")
    carol.optimize_for_scenario("busy_bar")
    
    # Run simulation
    print("Running optimized simulation...")
    for step in range(10):
        print(f"Step {step + 1}:")
        results = env.simulate_step()
        
        for agent_name, result in results.items():
            action = result["action"]
            print(f"  {agent_name}: {action['action_type']}({action.get('target', 'N/A')}) "
                  f"[{action['method']}, {action['processing_time']*1000:.1f}ms]")
        
        time.sleep(0.1)  # Brief pause
    
    # Show statistics
    print("\n--- Performance Statistics ---")
    stats = env.get_environment_stats()
    
    print(f"Total actions: {stats['total_actions']}")
    print(f"Cache rate: {stats['cache_rate_percent']:.1f}%")
    print(f"Fast path rate: {stats['fast_path_rate_percent']:.1f}%")
    
    # Generate reports
    env.generate_environment_report("demo_bar_environment_report.md")
    
    for agent in [alice, bob, carol]:
        agent.generate_agent_report(f"demo_{agent.name.lower()}_report.md")
    
    print("\nDemo complete! Reports generated.")


if __name__ == "__main__":
    demo_optimized_bar_agents()
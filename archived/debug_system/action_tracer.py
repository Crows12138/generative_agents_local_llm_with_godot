#!/usr/bin/env python3
"""
Action Tracer - Tracks and analyzes agent actions and decisions
"""
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

@dataclass
class ActionTrace:
    """Record of an agent action"""
    timestamp: datetime
    agent_name: str
    action_type: str
    action: str
    context: Dict[str, Any]
    decision_factors: Dict[str, Any]
    success: bool
    duration: float
    outcome: Optional[str] = None

@dataclass
class StateChange:
    """Record of state changes"""
    attribute: str
    old_value: Any
    new_value: Any
    change_reason: str

class ActionTracer:
    """Comprehensive action tracking and analysis system"""
    
    def __init__(self):
        self.traces: List[ActionTrace] = []
        self.agent_states: Dict[str, Dict[str, Any]] = {}
        self.action_chains: Dict[str, List[str]] = {}
        self.start_time = datetime.now()
        
    def capture_state(self, agent_name: str, state: Dict[str, Any]):
        """Capture current agent state"""
        if agent_name not in self.agent_states:
            self.agent_states[agent_name] = {}
        
        # Store previous state for comparison
        previous_state = self.agent_states[agent_name].copy()
        self.agent_states[agent_name] = state.copy()
        
        return previous_state
        
    def trace_action(self, 
                    agent_name: str,
                    action_type: str,
                    action: str,
                    context: Dict[str, Any] = None,
                    decision_factors: Dict[str, Any] = None,
                    duration: float = 0.0,
                    success: bool = True,
                    outcome: Optional[str] = None):
        """Record an agent action"""
        
        trace = ActionTrace(
            timestamp=datetime.now(),
            agent_name=agent_name,
            action_type=action_type,
            action=action,
            context=context or {},
            decision_factors=decision_factors or {},
            success=success,
            duration=duration,
            outcome=outcome
        )
        
        self.traces.append(trace)
        
        # Update action chains
        if agent_name not in self.action_chains:
            self.action_chains[agent_name] = []
        self.action_chains[agent_name].append(action)
        
        # Keep only last 20 actions per agent
        if len(self.action_chains[agent_name]) > 20:
            self.action_chains[agent_name] = self.action_chains[agent_name][-20:]
    
    def detect_state_changes(self, agent_name: str, new_state: Dict[str, Any]) -> List[StateChange]:
        """Detect and record state changes"""
        if agent_name not in self.agent_states:
            return []
        
        old_state = self.agent_states[agent_name]
        changes = []
        
        for key, new_value in new_state.items():
            old_value = old_state.get(key)
            if old_value != new_value:
                changes.append(StateChange(
                    attribute=key,
                    old_value=old_value,
                    new_value=new_value,
                    change_reason="action_result"
                ))
        
        return changes
    
    def get_agent_timeline(self, agent_name: str, last_n: int = 10) -> List[ActionTrace]:
        """Get recent actions for a specific agent"""
        agent_traces = [t for t in self.traces if t.agent_name == agent_name]
        return agent_traces[-last_n:] if agent_traces else []
    
    def analyze_action_patterns(self, agent_name: str) -> Dict[str, Any]:
        """Analyze patterns in agent actions"""
        agent_traces = [t for t in self.traces if t.agent_name == agent_name]
        
        if not agent_traces:
            return {"message": f"No actions recorded for {agent_name}"}
        
        # Action type distribution
        action_types = {}
        for trace in agent_traces:
            action_types[trace.action_type] = action_types.get(trace.action_type, 0) + 1
        
        # Success rate
        successful_actions = len([t for t in agent_traces if t.success])
        success_rate = (successful_actions / len(agent_traces)) * 100
        
        # Average action duration
        durations = [t.duration for t in agent_traces if t.duration > 0]
        avg_duration = sum(durations) / len(durations) if durations else 0
        
        # Most common actions
        actions = {}
        for trace in agent_traces:
            actions[trace.action] = actions.get(trace.action, 0) + 1
        
        most_common_actions = sorted(actions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Action chains (sequences)
        action_chain = self.action_chains.get(agent_name, [])
        
        return {
            "total_actions": len(agent_traces),
            "action_types": action_types,
            "success_rate": f"{success_rate:.1f}%",
            "average_duration": f"{avg_duration:.3f}s",
            "most_common_actions": most_common_actions,
            "recent_action_chain": action_chain[-10:],
            "first_action": agent_traces[0].timestamp.strftime("%H:%M:%S"),
            "last_action": agent_traces[-1].timestamp.strftime("%H:%M:%S")
        }
    
    def print_action_timeline(self, agent_name: str = None, last_n: int = 10):
        """Print formatted action timeline"""
        if agent_name:
            traces = self.get_agent_timeline(agent_name, last_n)
            title = f"🎯 ACTION TIMELINE - {agent_name} (Last {last_n})"
        else:
            traces = self.traces[-last_n:]
            title = f"🎯 GLOBAL ACTION TIMELINE (Last {last_n})"
        
        print(f"\n{title}")
        print("=" * 80)
        
        for i, trace in enumerate(traces, 1):
            status = "✅" if trace.success else "❌"
            duration_str = f"({trace.duration:.3f}s)" if trace.duration > 0 else ""
            
            print(f"\n{status} #{i} - {trace.timestamp.strftime('%H:%M:%S.%f')[:-3]}")
            print(f"  👤 Agent: {trace.agent_name}")
            print(f"  📋 Type: {trace.action_type}")
            print(f"  🎬 Action: {trace.action} {duration_str}")
            
            if trace.context:
                print(f"  🌍 Context: {trace.context}")
            
            if trace.decision_factors:
                print(f"  🧠 Decision Factors: {trace.decision_factors}")
            
            if trace.outcome:
                print(f"  📊 Outcome: {trace.outcome}")
    
    def analyze_interactions(self) -> Dict[str, Any]:
        """Analyze agent interactions and dependencies"""
        interactions = {}
        agent_mentions = {}
        
        for trace in self.traces:
            # Look for other agent names in action context
            for other_agent in self.agent_states.keys():
                if other_agent != trace.agent_name:
                    if other_agent.lower() in trace.action.lower():
                        key = f"{trace.agent_name} -> {other_agent}"
                        interactions[key] = interactions.get(key, 0) + 1
                        
                        if trace.agent_name not in agent_mentions:
                            agent_mentions[trace.agent_name] = {}
                        agent_mentions[trace.agent_name][other_agent] = agent_mentions[trace.agent_name].get(other_agent, 0) + 1
        
        return {
            "direct_interactions": interactions,
            "agent_mention_patterns": agent_mentions,
            "most_interactive_agents": sorted(agent_mentions.items(), key=lambda x: sum(x[1].values()), reverse=True)
        }
    
    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        if not self.traces:
            return {"message": "No action data to analyze"}
        
        total_actions = len(self.traces)
        unique_agents = len(set(t.agent_name for t in self.traces))
        session_duration = datetime.now() - self.start_time
        
        # Overall success rate
        successful_actions = len([t for t in self.traces if t.success])
        overall_success_rate = (successful_actions / total_actions) * 100
        
        # Action type breakdown
        action_types = {}
        for trace in self.traces:
            action_types[trace.action_type] = action_types.get(trace.action_type, 0) + 1
        
        # Agent activity levels
        agent_activity = {}
        for trace in self.traces:
            agent_activity[trace.agent_name] = agent_activity.get(trace.agent_name, 0) + 1
        
        return {
            "session_summary": {
                "duration": str(session_duration),
                "total_actions": total_actions,
                "unique_agents": unique_agents,
                "overall_success_rate": f"{overall_success_rate:.1f}%"
            },
            "action_breakdown": action_types,
            "agent_activity": sorted(agent_activity.items(), key=lambda x: x[1], reverse=True),
            "interaction_analysis": self.analyze_interactions()
        }
    
    def save_trace_data(self, filename: str = None):
        """Save trace data to JSON file"""
        if filename is None:
            filename = f"action_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Convert traces to serializable format
        serializable_traces = []
        for trace in self.traces:
            trace_dict = asdict(trace)
            trace_dict['timestamp'] = trace.timestamp.isoformat()
            serializable_traces.append(trace_dict)
        
        data = {
            "session_info": {
                "start_time": self.start_time.isoformat(),
                "end_time": datetime.now().isoformat(),
                "total_traces": len(self.traces)
            },
            "summary": self.generate_summary_report(),
            "traces": serializable_traces,
            "agent_states": self.agent_states,
            "action_chains": self.action_chains
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"📁 Action trace data saved to: {filename}")

# Global tracer instance
tracer = ActionTracer()

def trace_agent_action(agent_name: str, action_type: str = "general"):
    """Decorator to automatically trace agent actions"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            
            # Capture pre-action state if possible
            if args and hasattr(args[0], 'get_status'):
                pre_state = args[0].get_status()
                tracer.capture_state(agent_name, pre_state)
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Determine context from function arguments
                context = {
                    "function": func.__name__,
                    "args_count": len(args),
                    "kwargs": list(kwargs.keys())
                }
                
                # Capture post-action state
                if args and hasattr(args[0], 'get_status'):
                    post_state = args[0].get_status()
                    state_changes = tracer.detect_state_changes(agent_name, post_state)
                    if state_changes:
                        context["state_changes"] = len(state_changes)
                
                # Record successful action
                tracer.trace_action(
                    agent_name=agent_name,
                    action_type=action_type,
                    action=str(result),
                    context=context,
                    duration=duration,
                    success=True,
                    outcome="completed"
                )
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                
                # Record failed action
                tracer.trace_action(
                    agent_name=agent_name,
                    action_type=action_type,
                    action=f"FAILED: {func.__name__}",
                    context={"error": str(e)},
                    duration=duration,
                    success=False,
                    outcome="error"
                )
                
                raise e
        
        return wrapper
    return decorator

if __name__ == "__main__":
    # Test the action tracer
    print("🎯 Action Tracer Test")
    
    # Simulate some actions
    tracer.trace_action("Bob", "dialogue", "Serves a drink to Charlie", {"recipient": "Charlie"}, {"mood": "friendly"}, 1.2, True)
    tracer.trace_action("Charlie", "action", "Takes a sip of whiskey", {"drink": "whiskey"}, {"thirst": 8}, 0.5, True)
    tracer.trace_action("Sam", "music", "Starts playing guitar", {"instrument": "guitar"}, {"energy": 90}, 2.1, True)
    
    tracer.print_action_timeline()
    
    print("\n📊 AGENT ANALYSIS")
    print("=" * 50)
    for agent in ["Bob", "Charlie", "Sam"]:
        analysis = tracer.analyze_action_patterns(agent)
        print(f"\n{agent}:")
        for key, value in analysis.items():
            print(f"  {key}: {value}")
    
    print("\n📋 SUMMARY REPORT")
    print("=" * 50)
    report = tracer.generate_summary_report()
    for key, value in report.items():
        print(f"{key}: {value}")
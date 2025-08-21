from datetime import datetime
import json
import os
from typing import List, Dict, Any


class FlowTracer:
    """Trace complete flow from perception to execution"""
    
    def __init__(self):
        self.flow_steps: List[Dict[str, Any]] = []
        self.current_step = 0
        
    def trace_perception(self, agent_name: str, raw_perception: str) -> None:
        """Step 1: Agent perceives environment"""
        step = {
            "step": 1,
            "phase": "PERCEPTION",
            "agent": agent_name,
            "input": raw_perception,
            "timestamp": datetime.now().isoformat()
        }
        self.flow_steps.append(step)
        print(f"""
        ┌─ Step 1: PERCEPTION ─────────────────
        │ Agent: {agent_name}
        │ Sees: {raw_perception}
        └──────────────────────────────────────
        """)
        
    def trace_llm_prompt(self, prompt: str) -> None:
        """Step 2: Build LLM prompt"""
        step = {
            "step": 2,
            "phase": "LLM_PROMPT",
            "prompt": prompt,
            "timestamp": datetime.now().isoformat()
        }
        self.flow_steps.append(step)
        print(f"""
        ┌─ Step 2: LLM PROMPT ─────────────────
        │ Sending to LLM:
        │ {prompt[:200]}...
        └──────────────────────────────────────
        """)
        
    def trace_llm_response(self, response: str) -> None:
        """Step 3: LLM response"""
        step = {
            "step": 3,
            "phase": "LLM_RESPONSE",
            "response": response,
            "timestamp": datetime.now().isoformat()
        }
        self.flow_steps.append(step)
        print(f"""
        ┌─ Step 3: LLM RESPONSE ────────────────
        │ LLM says:
        │ {response}
        └──────────────────────────────────────
        """)
        
    def trace_action_parsing(self, raw_response: str, parsed_action: Dict[str, Any]) -> None:
        """Step 4: Parse action"""
        step = {
            "step": 4,
            "phase": "ACTION_PARSING",
            "raw_response": raw_response,
            "parsed_action": parsed_action,
            "timestamp": datetime.now().isoformat()
        }
        self.flow_steps.append(step)
        print(f"""
        ┌─ Step 4: ACTION PARSING ──────────────
        │ Raw: "{raw_response}"
        │ ↓
        │ Parsed: {parsed_action}
        └──────────────────────────────────────
        """)
        
    def trace_execution(self, action: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Step 5: Execute action"""
        step = {
            "step": 5,
            "phase": "EXECUTION",
            "action": action,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        self.flow_steps.append(step)
        print(f"""
        ┌─ Step 5: EXECUTION ───────────────────
        │ Action: {action}
        │ Result: {result}
        └──────────────────────────────────────
        """)
        
    def trace_state_update(self, old_state: Dict[str, Any], new_state: Dict[str, Any]) -> None:
        """Step 6: Update state"""
        step = {
            "step": 6,
            "phase": "STATE_UPDATE",
            "old_state": old_state,
            "new_state": new_state,
            "timestamp": datetime.now().isoformat()
        }
        self.flow_steps.append(step)
        print(f"""
        ┌─ Step 6: STATE UPDATE ────────────────
        │ State changes detected
        │ Old → New
        └──────────────────────────────────────
        """)
        
    def save_flow_diagram(self, filename: str = "action_flow.md") -> None:
        """Save flow diagram"""
        with open(filename, "w", encoding='utf-8') as f:
            f.write("# Action Flow Diagram\n\n")
            f.write("```mermaid\n")
            f.write("graph TD\n")
            f.write("    A[Perception] --> B[LLM Prompt]\n")
            f.write("    B --> C[LLM Response]\n")
            f.write("    C --> D[Parse Action]\n")
            f.write("    D --> E[Execute]\n")
            f.write("    E --> F[Update State]\n")
            f.write("    F --> A\n")
            f.write("```\n\n")
            
            f.write("## Flow Details\n\n")
            for step in self.flow_steps:
                f.write(f"### Step {step['step']}: {step['phase']}\n")
                f.write(f"**Time**: {step['timestamp']}\n\n")
                if 'agent' in step:
                    f.write(f"**Agent**: {step['agent']}\n")
                if 'input' in step:
                    f.write(f"**Input**: {step['input']}\n")
                if 'response' in step:
                    f.write(f"**Response**: {step['response']}\n")
                f.write("\n---\n\n")
                
    def save_flow_json(self, filename: str = "flow_trace.json") -> None:
        """Save detailed JSON data"""
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(self.flow_steps, f, indent=2, ensure_ascii=False)
            
    def clear_trace(self) -> None:
        """Clear trace data"""
        self.flow_steps.clear()
        self.current_step = 0
        
    def get_summary(self) -> Dict[str, Any]:
        """Get flow summary"""
        if not self.flow_steps:
            return {"status": "no_data"}
            
        phases = [step['phase'] for step in self.flow_steps]
        start_time = self.flow_steps[0]['timestamp']
        end_time = self.flow_steps[-1]['timestamp']
        
        return {
            "total_steps": len(self.flow_steps),
            "phases": phases,
            "start_time": start_time,
            "end_time": end_time,
            "duration": "calculated_duration",
            "status": "complete" if len(phases) >= 5 else "incomplete"
        }
        
    def print_summary(self) -> None:
        """Print flow summary"""
        summary = self.get_summary()
        print(f"""
        ╔═══════════════════════════════════════╗
        ║           FLOW SUMMARY                ║
        ╠═══════════════════════════════════════╣
        ║ Steps: {summary.get('total_steps', 0):^29} ║
        ║ Status: {summary.get('status', 'unknown'):^28} ║
        ║ Phases: {', '.join(summary.get('phases', [])):^28} ║
        ╚═══════════════════════════════════════╝
        """)


# Global flow tracer instance
global_tracer = FlowTracer()


def get_tracer() -> FlowTracer:
    """Get global tracer instance"""
    return global_tracer
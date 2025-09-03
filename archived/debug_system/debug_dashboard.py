#!/usr/bin/env python3
"""
Real-time debugging dashboard for generative agents
Complete monitoring and visualization system
"""

import sys
import os
import time
import threading
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from rich.console import Console
    from rich.table import Table
    from rich.live import Live
    from rich.layout import Layout
    from rich.panel import Panel
    from rich.progress import Progress, BarColumn, TextColumn
    from rich.text import Text
    from rich.columns import Columns
    HAS_RICH = True
except ImportError:
    HAS_RICH = False
    print("Warning: rich library not available, using basic console output")

from debug_system.flow_tracer import get_tracer
from debug_system.performance_analyzer import get_performance_analyzer
from debug_system.llm_parser_study import get_analyzer


@dataclass
class AgentState:
    """Agent state information"""
    name: str
    position: tuple
    action: str
    target: Optional[str]
    status: str
    last_update: datetime


@dataclass
class SystemEvent:
    """System event information"""
    timestamp: datetime
    event_type: str
    agent_name: str
    description: str
    severity: str  # info, warning, error


class DebugDashboard:
    """Real-time debugging dashboard for generative agents"""
    
    def __init__(self):
        self.console = Console() if HAS_RICH else None
        self.layout = Layout() if HAS_RICH else None
        self.is_running = False
        self.update_thread: Optional[threading.Thread] = None
        
        # Data storage
        self.agents: Dict[str, AgentState] = {}
        self.events: List[SystemEvent] = []
        self.performance_data: Dict[str, Any] = {}
        
        # Components
        self.tracer = get_tracer()
        self.perf_analyzer = get_performance_analyzer()
        self.action_analyzer = get_analyzer()
        
        if HAS_RICH:
            self.setup_layout()
        
    def setup_layout(self) -> None:
        """Setup dashboard layout"""
        if not HAS_RICH:
            return
            
        self.layout.split(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=5)
        )
        
        self.layout["main"].split_row(
            Layout(name="agents", ratio=2),
            Layout(name="events", ratio=2),
            Layout(name="performance", ratio=1)
        )
        
        # Split performance section
        self.layout["performance"].split(
            Layout(name="metrics"),
            Layout(name="actions")
        )
    
    def add_agent(self, name: str, position: tuple = (0, 0), status: str = "idle") -> None:
        """Add or update agent state"""
        self.agents[name] = AgentState(
            name=name,
            position=position,
            action="idle",
            target=None,
            status=status,
            last_update=datetime.now()
        )
        
        self.add_event("agent", name, f"Agent {name} added to dashboard", "info")
    
    def update_agent(self, name: str, **kwargs) -> None:
        """Update agent state"""
        if name in self.agents:
            agent = self.agents[name]
            for key, value in kwargs.items():
                if hasattr(agent, key):
                    setattr(agent, key, value)
            agent.last_update = datetime.now()
            
            action = kwargs.get('action', 'unknown')
            self.add_event("action", name, f"Agent {name} performing: {action}", "info")
    
    def add_event(self, event_type: str, agent_name: str, description: str, severity: str = "info") -> None:
        """Add system event"""
        event = SystemEvent(
            timestamp=datetime.now(),
            event_type=event_type,
            agent_name=agent_name,
            description=description,
            severity=severity
        )
        
        self.events.append(event)
        
        # Keep only last 50 events
        if len(self.events) > 50:
            self.events = self.events[-50:]
    
    def create_agents_table(self):
        """Create agents status table"""
        if not HAS_RICH:
            return "Agents table (rich not available)"
        table = Table(title="Agent Status", title_style="bold blue")
        table.add_column("Name", style="cyan", no_wrap=True)
        table.add_column("Position", style="magenta")
        table.add_column("Action", style="green")
        table.add_column("Target", style="yellow")
        table.add_column("Status", style="white")
        table.add_column("Last Update", style="dim")
        
        for agent in self.agents.values():
            status_style = "green" if agent.status == "active" else "yellow" if agent.status == "idle" else "red"
            
            table.add_row(
                agent.name,
                f"{agent.position[0]}, {agent.position[1]}",
                agent.action,
                agent.target or "-",
                f"[{status_style}]{agent.status}[/{status_style}]",
                agent.last_update.strftime("%H:%M:%S")
            )
        
        return table
    
    def create_events_table(self):
        """Create events log table"""
        if not HAS_RICH:
            return "Events table (rich not available)"
        table = Table(title="System Events", title_style="bold green")
        table.add_column("Time", style="dim", width=8)
        table.add_column("Type", style="cyan", width=8)
        table.add_column("Agent", style="yellow", width=12)
        table.add_column("Description", style="white")
        
        # Show last 15 events
        recent_events = self.events[-15:]
        
        for event in recent_events:
            severity_style = "red" if event.severity == "error" else "yellow" if event.severity == "warning" else "green"
            
            table.add_row(
                event.timestamp.strftime("%H:%M:%S"),
                f"[{severity_style}]{event.event_type}[/{severity_style}]",
                event.agent_name,
                event.description[:40] + "..." if len(event.description) > 40 else event.description
            )
        
        return table
    
    def create_performance_panel(self):
        """Create performance metrics panel"""
        if not HAS_RICH:
            return "Performance panel (rich not available)"
        summary = self.perf_analyzer.get_summary()
        
        if summary.get("status") == "no_data":
            content = Text("No performance data available", style="dim")
        else:
            content = Text()
            content.append("Performance Metrics\n\n", style="bold")
            content.append(f"Total Cycles: {summary.get('total_cycles', 0)}\n")
            content.append(f"Avg Cycle Time: {summary.get('avg_cycle_time', 0)*1000:.1f}ms\n")
            content.append(f"Memory Usage: {summary.get('avg_memory_mb', 0):.1f}MB\n")
            content.append(f"CPU Usage: {summary.get('avg_cpu_percent', 0):.1f}%\n")
            
            bottleneck = summary.get('bottleneck_phase', 'unknown')
            if bottleneck != 'unknown':
                content.append(f"\nBottleneck: {bottleneck}", style="red bold")
        
        return Panel(content, title="Performance", border_style="blue")
    
    def create_actions_panel(self):
        """Create action analysis panel"""
        if not HAS_RICH:
            return "Actions panel (rich not available)"
        distribution = self.action_analyzer.get_distribution()
        
        if not distribution:
            content = Text("No action data available", style="dim")
        else:
            content = Text()
            content.append("Action Distribution\n\n", style="bold")
            
            for action_type, percentage in sorted(distribution.items(), key=lambda x: x[1], reverse=True):
                bar_length = int(percentage / 5)  # Scale to fit
                bar = "#" * bar_length + "." * (20 - bar_length)
                content.append(f"{action_type:8}: {bar} {percentage:.1f}%\n")
        
        return Panel(content, title="Actions", border_style="green")
    
    def create_header(self):
        """Create dashboard header"""
        if not HAS_RICH:
            return "Header (rich not available)"
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        agent_count = len(self.agents)
        active_agents = sum(1 for agent in self.agents.values() if agent.status == "active")
        
        header_text = Text()
        header_text.append("Generative Agents Debug Dashboard", style="bold cyan")
        header_text.append(f" | {current_time}", style="dim")
        header_text.append(f" | Agents: {active_agents}/{agent_count}", style="green")
        
        return Panel(header_text, style="white on blue")
    
    def create_footer(self):
        """Create dashboard footer with controls"""
        if not HAS_RICH:
            return "Footer (rich not available)"
        footer_text = Text()
        footer_text.append("Controls: ", style="bold")
        footer_text.append("[q] Quit  ", style="cyan")
        footer_text.append("[r] Reset  ", style="yellow")
        footer_text.append("[p] Performance  ", style="green")
        footer_text.append("[s] Save Report", style="magenta")
        
        return Panel(footer_text, style="white on black")
    
    def update_display(self):
        """Update dashboard display"""
        if not HAS_RICH:
            return self.console_fallback()
        
        # Update all sections
        self.layout["header"].update(self.create_header())
        self.layout["agents"].update(self.create_agents_table())
        self.layout["events"].update(self.create_events_table())
        self.layout["metrics"].update(self.create_performance_panel())
        self.layout["actions"].update(self.create_actions_panel())
        self.layout["footer"].update(self.create_footer())
        
        return self.layout
    
    def console_fallback(self) -> None:
        """Fallback console output when rich is not available"""
        print("\n" + "="*80)
        print(f"DEBUG DASHBOARD - {datetime.now().strftime('%H:%M:%S')}")
        print("="*80)
        
        print(f"\nAGENTS ({len(self.agents)}):")
        for agent in self.agents.values():
            print(f"  {agent.name}: {agent.action} at {agent.position} [{agent.status}]")
        
        print(f"\nRECENT EVENTS ({len(self.events[-5:])}):")
        for event in self.events[-5:]:
            print(f"  {event.timestamp.strftime('%H:%M:%S')} {event.agent_name}: {event.description}")
        
        summary = self.perf_analyzer.get_summary()
        if summary.get("status") != "no_data":
            print(f"\nPERFORMANCE:")
            print(f"  Cycles: {summary.get('total_cycles', 0)}")
            print(f"  Avg Time: {summary.get('avg_cycle_time', 0)*1000:.1f}ms")
        
        print("="*80)
    
    def start_monitoring(self, update_interval: float = 1.0) -> None:
        """Start real-time monitoring"""
        if not HAS_RICH:
            print("Starting basic monitoring mode...")
            self.is_running = True
            self.update_thread = threading.Thread(target=self._monitor_loop, args=(update_interval,))
            self.update_thread.daemon = True
            self.update_thread.start()
            return
        
        print("Starting dashboard...")
        self.is_running = True
        
        try:
            with Live(self.update_display(), refresh_per_second=1/update_interval, screen=True) as live:
                while self.is_running:
                    live.update(self.update_display())
                    time.sleep(update_interval)
        except KeyboardInterrupt:
            self.stop_monitoring()
    
    def _monitor_loop(self, interval: float) -> None:
        """Basic monitoring loop for fallback mode"""
        while self.is_running:
            self.console_fallback()
            time.sleep(interval)
    
    def stop_monitoring(self) -> None:
        """Stop monitoring"""
        self.is_running = False
        if self.update_thread:
            self.update_thread.join(timeout=2.0)
        print("Dashboard stopped.")
    
    def simulate_agent_activity(self) -> None:
        """Simulate some agent activity for testing"""
        import random
        
        # Add test agents
        test_agents = ["Alice", "Bob", "Charlie"]
        for agent_name in test_agents:
            self.add_agent(agent_name, (random.randint(0, 10), random.randint(0, 10)), "active")
        
        # Simulate activity
        actions = ["move", "talk", "work", "idle"]
        targets = ["bar", "customer", "kitchen", "storage"]
        
        for _ in range(10):
            agent = random.choice(test_agents)
            action = random.choice(actions)
            target = random.choice(targets) if action != "idle" else None
            
            self.update_agent(agent, 
                            action=action,
                            target=target,
                            position=(random.randint(0, 10), random.randint(0, 10)))
            
            # Simulate performance data
            self.perf_analyzer.profile_action_cycle(agent)
            
            time.sleep(0.5)
    
    def save_dashboard_report(self, filename: str = "dashboard_report.md") -> None:
        """Save dashboard state to report"""
        with open(filename, "w", encoding='utf-8') as f:
            f.write("# Debug Dashboard Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            
            f.write("## Agent Status\n\n")
            f.write("| Name | Position | Action | Target | Status | Last Update |\n")
            f.write("|------|----------|--------|--------|--------|-------------|\n")
            
            for agent in self.agents.values():
                f.write(f"| {agent.name} | {agent.position} | {agent.action} | ")
                f.write(f"{agent.target or '-'} | {agent.status} | {agent.last_update.strftime('%H:%M:%S')} |\n")
            
            f.write("\n## Recent Events\n\n")
            for event in self.events[-20:]:
                f.write(f"- **{event.timestamp.strftime('%H:%M:%S')}** [{event.severity.upper()}] ")
                f.write(f"{event.agent_name}: {event.description}\n")
            
            f.write("\n## Performance Summary\n\n")
            summary = self.perf_analyzer.get_summary()
            if summary.get("status") != "no_data":
                f.write(f"- Total Cycles: {summary.get('total_cycles', 0)}\n")
                f.write(f"- Average Cycle Time: {summary.get('avg_cycle_time', 0)*1000:.1f}ms\n")
                f.write(f"- Memory Usage: {summary.get('avg_memory_mb', 0):.1f}MB\n")
                f.write(f"- CPU Usage: {summary.get('avg_cpu_percent', 0):.1f}%\n")
        
        print(f"Dashboard report saved to {filename}")


def main():
    """Main entry point for dashboard"""
    dashboard = DebugDashboard()
    
    print("Debug Dashboard Starting...")
    print("Commands:")
    print("  --simulate : Run with simulated data")
    print("  --help     : Show this help")
    
    if len(sys.argv) > 1:
        if "--simulate" in sys.argv:
            print("Running with simulated data...")
            
            # Start simulation in background
            sim_thread = threading.Thread(target=dashboard.simulate_agent_activity)
            sim_thread.daemon = True
            sim_thread.start()
            
            # Start dashboard
            try:
                dashboard.start_monitoring(update_interval=0.5)
            except KeyboardInterrupt:
                dashboard.stop_monitoring()
                dashboard.save_dashboard_report()
        
        elif "--help" in sys.argv:
            print("Debug Dashboard for Generative Agents")
            print("Real-time monitoring of agent states, events, and performance")
            return
    
    else:
        print("Starting dashboard in monitoring mode...")
        print("Use Ctrl+C to stop and save report")
        
        try:
            dashboard.start_monitoring()
        except KeyboardInterrupt:
            dashboard.stop_monitoring()
            dashboard.save_dashboard_report()


if __name__ == "__main__":
    main()
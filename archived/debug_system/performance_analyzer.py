import time
import psutil
import threading
import json
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
try:
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available, plotting features disabled")


@dataclass
class PerformanceMetric:
    """Single performance measurement"""
    timestamp: str
    phase: str
    duration: float
    memory_mb: float
    cpu_percent: float
    agent_name: Optional[str] = None
    additional_data: Optional[Dict[str, Any]] = None


class PerformanceAnalyzer:
    """Analyze system performance bottlenecks"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetric] = []
        self.phase_times: Dict[str, List[float]] = {
            "perception": [],
            "llm_call": [],
            "action_parsing": [],
            "execution": [],
            "total_cycle": []
        }
        self.system_metrics: Dict[str, List[float]] = {
            "memory_usage": [],
            "cpu_usage": [],
            "timestamps": []
        }
        self.is_monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        self.active_measurements: Dict[str, float] = {}
        
    def start_measurement(self, measurement_id: str) -> None:
        """Start timing a specific measurement"""
        import time
        self.active_measurements[measurement_id] = time.time()
    
    def end_measurement(self, measurement_id: str) -> float:
        """End timing a specific measurement and return duration"""
        import time
        if measurement_id not in self.active_measurements:
            return 0.0
        
        start_time = self.active_measurements.pop(measurement_id)
        duration = time.time() - start_time
        
        # Store the measurement in appropriate phase
        if measurement_id in self.phase_times:
            self.phase_times[measurement_id].append(duration)
        elif "llm" in measurement_id.lower():
            self.phase_times["llm_call"].append(duration)
        
        return duration
        
    def start_monitoring(self, interval: float = 1.0) -> None:
        """Start continuous system monitoring"""
        if self.is_monitoring:
            return
            
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_system, 
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        print(f"System monitoring started (interval: {interval}s)")
    
    def stop_monitoring(self) -> None:
        """Stop system monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        print("System monitoring stopped")
    
    def _monitor_system(self, interval: float) -> None:
        """Internal system monitoring loop"""
        while self.is_monitoring:
            try:
                # Get current system metrics
                memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                cpu_percent = psutil.cpu_percent()
                timestamp = time.time()
                
                # Store metrics
                self.system_metrics["memory_usage"].append(memory_mb)
                self.system_metrics["cpu_usage"].append(cpu_percent)
                self.system_metrics["timestamps"].append(timestamp)
                
                # Keep only last 1000 measurements
                for key in self.system_metrics:
                    if len(self.system_metrics[key]) > 1000:
                        self.system_metrics[key] = self.system_metrics[key][-1000:]
                
                time.sleep(interval)
                
            except Exception as e:
                print(f"Monitor error: {e}")
                time.sleep(interval)
    
    def profile_action_cycle(self, agent_name: str) -> Dict[str, float]:
        """Profile a complete action cycle"""
        cycle_start = time.time()
        timings = {}
        
        # Simulate action cycle phases with timing
        print(f"\nProfiling action cycle for agent: {agent_name}")
        
        # Phase 1: Perception
        start = time.time()
        self._simulate_perception()
        perception_time = time.time() - start
        timings["perception"] = perception_time
        self.phase_times["perception"].append(perception_time)
        
        # Phase 2: LLM Call
        start = time.time()
        self._simulate_llm_call()
        llm_time = time.time() - start
        timings["llm_call"] = llm_time
        self.phase_times["llm_call"].append(llm_time)
        
        # Phase 3: Action Parsing
        start = time.time()
        self._simulate_action_parsing()
        parse_time = time.time() - start
        timings["action_parsing"] = parse_time
        self.phase_times["action_parsing"].append(parse_time)
        
        # Phase 4: Execution
        start = time.time()
        self._simulate_execution()
        exec_time = time.time() - start
        timings["execution"] = exec_time
        self.phase_times["execution"].append(exec_time)
        
        # Total cycle time
        total_time = time.time() - cycle_start
        timings["total_cycle"] = total_time
        self.phase_times["total_cycle"].append(total_time)
        
        # Record current system state
        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_percent = psutil.cpu_percent()
        
        # Store metric
        metric = PerformanceMetric(
            timestamp=datetime.now().isoformat(),
            phase="complete_cycle",
            duration=total_time,
            memory_mb=memory_mb,
            cpu_percent=cpu_percent,
            agent_name=agent_name,
            additional_data=timings
        )
        self.metrics.append(metric)
        
        # Print results
        self._print_cycle_results(timings, memory_mb, cpu_percent)
        
        return timings
    
    def _simulate_perception(self) -> None:
        """Simulate perception phase"""
        time.sleep(0.001)  # Minimal simulation
    
    def _simulate_llm_call(self) -> None:
        """Simulate LLM call phase"""
        time.sleep(0.1)  # Simulate network/processing delay
    
    def _simulate_action_parsing(self) -> None:
        """Simulate action parsing phase"""
        time.sleep(0.002)  # Minimal simulation
    
    def _simulate_execution(self) -> None:
        """Simulate execution phase"""
        time.sleep(0.005)  # Minimal simulation
    
    def _print_cycle_results(self, timings: Dict[str, float], memory_mb: float, cpu_percent: float) -> None:
        """Print cycle performance results"""
        print(f"""
        Performance Analysis Results:
        - Perception:     {timings['perception']:.3f}s
        - LLM Call:       {timings['llm_call']:.3f}s {"(slow)" if timings['llm_call'] > 1 else ""}
        - Action Parsing: {timings['action_parsing']:.3f}s
        - Execution:      {timings['execution']:.3f}s
        - Total Cycle:    {timings['total_cycle']:.3f}s
        - Memory:         {memory_mb:.1f} MB
        - CPU:            {cpu_percent:.1f}%
        """)
    
    def analyze_bottlenecks(self) -> Dict[str, Any]:
        """Analyze performance bottlenecks"""
        if not self.phase_times["total_cycle"]:
            return {"error": "No performance data available"}
        
        bottlenecks = {}
        
        # Calculate averages for each phase
        for phase, times in self.phase_times.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                std_dev = (sum((x - avg_time) ** 2 for x in times) / len(times)) ** 0.5
                
                bottlenecks[phase] = {
                    "average": avg_time,
                    "max": max_time,
                    "min": min_time,
                    "std_dev": std_dev,
                    "samples": len(times),
                    "is_bottleneck": avg_time > 0.5  # Mark as bottleneck if > 500ms
                }
        
        # Identify biggest bottleneck
        bottleneck_phase = max(bottlenecks.keys(), 
                              key=lambda x: bottlenecks[x]["average"] if x != "total_cycle" else 0)
        
        return {
            "phase_analysis": bottlenecks,
            "biggest_bottleneck": bottleneck_phase,
            "total_samples": len(self.metrics)
        }
    
    def generate_report(self, filename: str = "performance_report.md") -> None:
        """Generate comprehensive performance report"""
        analysis = self.analyze_bottlenecks()
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write("# Performance Analysis Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            
            if "error" in analysis:
                f.write(f"**Error**: {analysis['error']}\n")
                return
            
            f.write("## Summary\n\n")
            f.write(f"- **Total Cycles Analyzed**: {analysis['total_samples']}\n")
            f.write(f"- **Biggest Bottleneck**: {analysis['biggest_bottleneck']}\n\n")
            
            f.write("## Phase Analysis\n\n")
            f.write("| Phase | Avg (ms) | Max (ms) | Min (ms) | Std Dev | Samples | Bottleneck |\n")
            f.write("|-------|----------|----------|----------|---------|---------|------------|\n")
            
            for phase, data in analysis["phase_analysis"].items():
                if phase == "total_cycle":
                    continue
                    
                f.write(f"| {phase} | {data['average']*1000:.1f} | {data['max']*1000:.1f} | ")
                f.write(f"{data['min']*1000:.1f} | {data['std_dev']*1000:.1f} | ")
                f.write(f"{data['samples']} | {'Warning: Yes' if data['is_bottleneck'] else 'OK: No'} |\n")
            
            f.write("\n## Recommendations\n\n")
            self._write_recommendations(f, analysis)
    
    def _write_recommendations(self, f, analysis: Dict[str, Any]) -> None:
        """Write performance recommendations"""
        bottleneck = analysis["biggest_bottleneck"]
        phase_data = analysis["phase_analysis"]
        
        if bottleneck == "llm_call":
            f.write("### LLM Call Optimization\n")
            f.write("- Consider using smaller/faster models\n")
            f.write("- Implement response caching\n")
            f.write("- Use parallel processing for multiple agents\n")
            f.write("- Optimize prompt length\n\n")
        
        elif bottleneck == "perception":
            f.write("### Perception Optimization\n")
            f.write("- Reduce environment scanning frequency\n")
            f.write("- Implement selective perception\n")
            f.write("- Cache environmental data\n\n")
        
        elif bottleneck == "execution":
            f.write("### Execution Optimization\n")
            f.write("- Optimize action implementations\n")
            f.write("- Use asynchronous execution\n")
            f.write("- Reduce game state updates\n\n")
        
        # Memory recommendations
        if self.system_metrics["memory_usage"]:
            avg_memory = sum(self.system_metrics["memory_usage"]) / len(self.system_metrics["memory_usage"])
            if avg_memory > 500:  # > 500MB
                f.write("### Memory Optimization\n")
                f.write("- High memory usage detected\n")
                f.write("- Consider reducing model size\n")
                f.write("- Implement garbage collection\n\n")
    
    def plot_performance(self, save_path: str = "performance_charts.png") -> None:
        """Generate performance visualization plots"""
        if not HAS_MATPLOTLIB:
            print("Matplotlib not available, skipping plot generation")
            return
            
        if not self.phase_times["total_cycle"]:
            print("No performance data to plot")
            return
        
        try:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Plot 1: Phase times
            phases = ["perception", "llm_call", "action_parsing", "execution"]
            avg_times = [sum(self.phase_times[phase])/len(self.phase_times[phase]) * 1000 
                        for phase in phases if self.phase_times[phase]]
            phase_labels = [phase for phase in phases if self.phase_times[phase]]
            
            ax1.bar(phase_labels, avg_times)
            ax1.set_title("Average Phase Times")
            ax1.set_ylabel("Time (ms)")
            ax1.tick_params(axis='x', rotation=45)
            
            # Plot 2: LLM response times over time
            if self.phase_times["llm_call"]:
                ax2.plot(self.phase_times["llm_call"])
                ax2.set_title("LLM Response Times")
                ax2.set_ylabel("Time (s)")
                ax2.set_xlabel("Cycle Number")
            
            # Plot 3: Memory usage over time
            if self.system_metrics["memory_usage"]:
                ax3.plot(self.system_metrics["memory_usage"])
                ax3.set_title("Memory Usage")
                ax3.set_ylabel("Memory (MB)")
                ax3.set_xlabel("Time")
            
            # Plot 4: CPU usage over time
            if self.system_metrics["cpu_usage"]:
                ax4.plot(self.system_metrics["cpu_usage"])
                ax4.set_title("CPU Usage")
                ax4.set_ylabel("CPU (%)")
                ax4.set_xlabel("Time")
            
            plt.tight_layout()
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Performance charts saved to {save_path}")
            
        except Exception as e:
            print(f"Error generating plots: {e}")
    
    def export_raw_data(self, filename: str = "performance_data.json") -> None:
        """Export raw performance data"""
        data = {
            "phase_times": self.phase_times,
            "system_metrics": {
                "memory_usage": self.system_metrics["memory_usage"],
                "cpu_usage": self.system_metrics["cpu_usage"],
                "timestamps": [t for t in self.system_metrics["timestamps"]]
            },
            "detailed_metrics": [
                {
                    "timestamp": m.timestamp,
                    "phase": m.phase,
                    "duration": m.duration,
                    "memory_mb": m.memory_mb,
                    "cpu_percent": m.cpu_percent,
                    "agent_name": m.agent_name,
                    "additional_data": m.additional_data
                }
                for m in self.metrics
            ]
        }
        
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        
        print(f"Raw performance data exported to {filename}")
    
    def clear_data(self) -> None:
        """Clear all performance data"""
        self.metrics.clear()
        for key in self.phase_times:
            self.phase_times[key].clear()
        for key in self.system_metrics:
            self.system_metrics[key].clear()
        print("All performance data cleared")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        if not self.metrics:
            return {"status": "no_data"}
        
        analysis = self.analyze_bottlenecks()
        
        return {
            "total_cycles": len(self.metrics),
            "bottleneck_phase": analysis.get("biggest_bottleneck"),
            "avg_cycle_time": sum(self.phase_times["total_cycle"])/len(self.phase_times["total_cycle"]) if self.phase_times["total_cycle"] else 0,
            "avg_memory_mb": sum(self.system_metrics["memory_usage"])/len(self.system_metrics["memory_usage"]) if self.system_metrics["memory_usage"] else 0,
            "avg_cpu_percent": sum(self.system_metrics["cpu_usage"])/len(self.system_metrics["cpu_usage"]) if self.system_metrics["cpu_usage"] else 0
        }


# Global performance analyzer instance
global_analyzer = PerformanceAnalyzer()


def get_performance_analyzer() -> PerformanceAnalyzer:
    """Get global performance analyzer instance"""
    return global_analyzer
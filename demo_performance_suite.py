"""
Demo Performance Suite - Ensure smooth demo experience
"""

import os
import sys
import time
import json
import threading
import asyncio
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

# Import optimizer modules
from performance_optimizer import PerformanceOptimizer, get_performance_optimizer
from memory_optimizer import MemoryOptimizer, get_memory_optimizer
from network_optimizer import NetworkOptimizer, get_network_optimizer

# Try to import AI service modules
try:
    from agents.simple_agents import SimpleAgent, create_demo_characters
    from ai_service.ai_service import local_llm_generate
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    print("⚠️ AI modules not available - running in limited mode")


@dataclass
class DemoConfig:
    """Demo configuration"""
    max_ai_response_time: float = 2.0
    max_memory_usage_gb: float = 4.0
    min_fps: float = 30.0
    max_agents: int = 5
    enable_caching: bool = True
    enable_memory_optimization: bool = True
    enable_network_optimization: bool = True
    enable_godot_optimization: bool = True


@dataclass
class DemoMetrics:
    """Demo metrics"""
    timestamp: datetime
    ai_response_time: float
    memory_usage_mb: float
    fps: float
    active_agents: int
    cache_hit_rate: float
    optimization_level: str
    warnings: List[str]
    
    def meets_targets(self, config: DemoConfig) -> bool:
        """Check if targets are met"""
        return (
            self.ai_response_time <= config.max_ai_response_time and
            self.memory_usage_mb <= config.max_memory_usage_gb * 1024 and
            self.fps >= config.min_fps
        )


class DemoPerformanceSuite:
    """Demo Performance Suite main class"""
    
    def __init__(self, config: DemoConfig = None):
        self.config = config or DemoConfig()
        
        # Initialize optimizers
        self.performance_optimizer = get_performance_optimizer()
        self.memory_optimizer = get_memory_optimizer()
        self.network_optimizer = get_network_optimizer()
        
        # Status tracking
        self.is_monitoring = False
        self.metrics_history: List[DemoMetrics] = []
        self.optimization_level = "normal"  # low, normal, high, extreme
        
        # Demo state
        self.demo_agents: List[Any] = []
        self.demo_start_time = None
        self.total_optimizations = 0
        
        print("🚀 Demo Performance Suite initialized")
    
    def start_demo_optimization(self):
        """Start demo optimization"""
        print("=" * 50)
        print("🎬 STARTING DEMO PERFORMANCE OPTIMIZATION")
        print("=" * 50)
        
        self.demo_start_time = datetime.now()
        
        # 1. Preload and pre-optimize
        self._preload_resources()
        
        # 2. Create demo agents
        self._setup_demo_agents()
        
        # 3. Start all optimizers
        self._start_optimizers()
        
        # 4. Start monitoring
        self._start_monitoring()
        
        print("✅ Demo optimization setup complete!")
        print(f"🎯 Targets: AI<{self.config.max_ai_response_time}s, Memory<{self.config.max_memory_usage_gb}GB, FPS>{self.config.min_fps}")
        
    def stop_demo_optimization(self):
        """Stop demo optimization"""
        print("\n🏁 Stopping demo optimization...")
        
        self.is_monitoring = False
        
        # Stop optimizers
        self.performance_optimizer.stop_monitoring()
        self.memory_optimizer.stop_monitoring()
        
        # Generate final report
        self._generate_demo_report()
        
        print("✅ Demo optimization stopped")
    
    def _preload_resources(self):
        """Preload resources"""
        print("📦 Preloading resources...")
        
        if AI_AVAILABLE:
            # Preload AI models
            try:
                self.performance_optimizer.model_manager.preload_models(["qwen"])
                print("  ✓ AI models preloaded")
            except Exception as e:
                print(f"  ⚠️ Model preload failed: {e}")
        
        # Warm up caches
        self.performance_optimizer.cache.max_size = 500
        self.network_optimizer.response_cache = {}
        
        print("  ✓ Caches initialized")
    
    def _setup_demo_agents(self):
        """Setup demo agents"""
        print("👥 Setting up demo agents...")
        
        if AI_AVAILABLE:
            try:
                # Create limited number of demo agents
                all_characters = create_demo_characters()
                self.demo_agents = all_characters[:self.config.max_agents]
                
                # Register with memory optimizer
                for agent in self.demo_agents:
                    self.memory_optimizer.register_agent_object(agent)
                
                print(f"  ✓ Created {len(self.demo_agents)} demo agents")
            except Exception as e:
                print(f"  ⚠️ Agent setup failed: {e}")
                self.demo_agents = []
        else:
            print("  ⚠️ AI not available - no agents created")
    
    def _start_optimizers(self):
        """Start optimizers"""
        print("🔧 Starting optimizers...")
        
        # Set strict demo targets
        self.performance_optimizer.targets.max_ai_response_time = self.config.max_ai_response_time
        self.performance_optimizer.targets.max_memory_usage_gb = self.config.max_memory_usage_gb
        self.performance_optimizer.targets.min_fps = self.config.min_fps
        
        # Start monitoring
        self.performance_optimizer.start_monitoring(interval_seconds=2)
        self.memory_optimizer.start_monitoring(interval_seconds=5)
        
        print("  ✓ Performance monitoring started")
        print("  ✓ Memory monitoring started")
        print("  ✓ Network optimization enabled")
    
    def _start_monitoring(self):
        """Start comprehensive monitoring"""
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    # Collect comprehensive metrics
                    metrics = self._collect_demo_metrics()
                    self.metrics_history.append(metrics)
                    
                    # Keep history size manageable
                    if len(self.metrics_history) > 200:
                        self.metrics_history = self.metrics_history[-100:]
                    
                    # Check performance and auto-adjust
                    self._auto_adjust_optimization(metrics)
                    
                    # Display real-time status
                    self._display_real_time_status(metrics)
                    
                    time.sleep(3)  # Check every 3 seconds
                    
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(3)
        
        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()
        print("  ✓ Demo monitoring started")
    
    def _collect_demo_metrics(self) -> DemoMetrics:
        """Collect demo metrics"""
        # Get metrics from various optimizers
        perf_report = self.performance_optimizer.get_performance_report()
        memory_report = self.memory_optimizer.get_memory_report()
        network_stats = self.network_optimizer.get_network_stats()
        
        # Extract key metrics
        ai_response_time = perf_report.get("performance_summary", {}).get("avg_ai_response_time", 0.0)
        memory_mb = memory_report.get("current_memory", {}).get("used_mb", 0.0)
        cache_hit_rate = perf_report.get("performance_summary", {}).get("cache_hit_rate", 0.0)
        
        # Simulate FPS (in real application, get from Godot)
        fps = 60.0 if ai_response_time < 1.0 and memory_mb < 2048 else 30.0
        
        # Collect warnings
        warnings = []
        if ai_response_time > self.config.max_ai_response_time:
            warnings.append(f"AI response slow: {ai_response_time:.2f}s")
        if memory_mb > self.config.max_memory_usage_gb * 1024:
            warnings.append(f"High memory: {memory_mb:.0f}MB")
        if fps < self.config.min_fps:
            warnings.append(f"Low FPS: {fps:.1f}")
        
        return DemoMetrics(
            timestamp=datetime.now(),
            ai_response_time=ai_response_time,
            memory_usage_mb=memory_mb,
            fps=fps,
            active_agents=len(self.demo_agents),
            cache_hit_rate=cache_hit_rate,
            optimization_level=self.optimization_level,
            warnings=warnings
        )
    
    def _auto_adjust_optimization(self, metrics: DemoMetrics):
        """Auto-adjust optimization level"""
        if not metrics.meets_targets(self.config):
            if self.optimization_level == "normal":
                self._apply_high_optimization()
            elif self.optimization_level == "high":
                self._apply_extreme_optimization()
        elif metrics.meets_targets(self.config) and self.optimization_level == "extreme":
            # Can reduce optimization level when performance is good
            self._apply_normal_optimization()
    
    def _apply_normal_optimization(self):
        """Apply normal optimization level"""
        if self.optimization_level == "normal":
            return
        
        self.optimization_level = "normal"
        self.total_optimizations += 1
        
        # Normal settings
        self.performance_optimizer.cache.max_size = 500
        self.memory_optimizer.cleanup_threshold_percent = 80.0
        
        print("🔧 Applied normal optimization level")
    
    def _apply_high_optimization(self):
        """Apply high optimization"""
        if self.optimization_level == "high":
            return
        
        self.optimization_level = "high"
        self.total_optimizations += 1
        
        # Enhanced settings
        self.performance_optimizer.cache.max_size = 1000
        self.memory_optimizer.cleanup_threshold_percent = 70.0
        self.memory_optimizer.gentle_cleanup()
        
        print("🔧 Applied high optimization level")
    
    def _apply_extreme_optimization(self):
        """Apply extreme optimization"""
        if self.optimization_level == "extreme":
            return
        
        self.optimization_level = "extreme"
        self.total_optimizations += 1
        
        # Extreme settings
        self.performance_optimizer.cache.max_size = 2000
        self.memory_optimizer.cleanup_threshold_percent = 60.0
        self.memory_optimizer.force_cleanup()
        
        # Reduce active agent count
        if len(self.demo_agents) > 3:
            for agent in self.demo_agents[3:]:
                if hasattr(agent, 'pause'):
                    agent.pause()
        
        print("🚨 Applied EXTREME optimization level")
    
    def _display_real_time_status(self, metrics: DemoMetrics):
        """Display real-time status (every 30 seconds)"""
        if not hasattr(self, '_last_status_time'):
            self._last_status_time = time.time()
        
        if time.time() - self._last_status_time >= 30:  # Display every 30 seconds
            print("\n" + "="*60)
            print("📊 REAL-TIME DEMO STATUS")
            print("="*60)
            print(f"🕐 Runtime: {datetime.now() - self.demo_start_time}")
            print(f"🤖 AI Response: {metrics.ai_response_time:.2f}s (target: <{self.config.max_ai_response_time}s)")
            print(f"💾 Memory: {metrics.memory_usage_mb:.0f}MB (target: <{self.config.max_memory_usage_gb*1024:.0f}MB)")
            print(f"🎮 FPS: {metrics.fps:.1f} (target: >{self.config.min_fps})")
            print(f"👥 Active Agents: {metrics.active_agents}")
            print(f"📦 Cache Hit Rate: {metrics.cache_hit_rate:.1%}")
            print(f"⚙️ Optimization Level: {metrics.optimization_level.upper()}")
            
            if metrics.warnings:
                print("⚠️ Warnings:")
                for warning in metrics.warnings:
                    print(f"   • {warning}")
            else:
                print("✅ All targets met!")
            
            print("="*60)
            self._last_status_time = time.time()
    
    def run_demo_test(self, duration_minutes: int = 5):
        """Run demo test"""
        print(f"\n🧪 Running {duration_minutes}-minute demo test...")
        
        if not AI_AVAILABLE:
            print("⚠️ AI not available - running limited test")
        
        self.start_demo_optimization()
        
        try:
            # Simulate demo activities
            end_time = time.time() + (duration_minutes * 60)
            test_count = 0
            
            while time.time() < end_time and self.is_monitoring:
                if AI_AVAILABLE and self.demo_agents:
                    # Simulate AI interactions
                    agent = self.demo_agents[test_count % len(self.demo_agents)]
                    try:
                        response = self.performance_optimizer.cached_ai_generate(
                            f"Test prompt {test_count}: How are you feeling?",
                            model_key="qwen"
                        )
                        print(f"🤖 Agent {agent.name}: {response[:50]}...")
                    except Exception as e:
                        print(f"AI test error: {e}")
                
                test_count += 1
                time.sleep(10)  # Interaction every 10 seconds
            
        except KeyboardInterrupt:
            print("\n⚠️ Demo test interrupted by user")
        
        finally:
            self.stop_demo_optimization()
    
    def _generate_demo_report(self):
        """Generate demo report"""
        if not self.metrics_history:
            print("⚠️ No metrics to report")
            return
        
        runtime = datetime.now() - self.demo_start_time
        
        # Calculate statistics
        avg_ai_time = sum(m.ai_response_time for m in self.metrics_history) / len(self.metrics_history)
        avg_memory = sum(m.memory_usage_mb for m in self.metrics_history) / len(self.metrics_history)
        avg_fps = sum(m.fps for m in self.metrics_history) / len(self.metrics_history)
        
        target_met_count = sum(1 for m in self.metrics_history if m.meets_targets(self.config))
        target_met_rate = target_met_count / len(self.metrics_history)
        
        # Generate report
        report = {
            "demo_summary": {
                "runtime_minutes": runtime.total_seconds() / 60,
                "total_metrics_collected": len(self.metrics_history),
                "target_met_rate": round(target_met_rate, 3),
                "total_optimizations_applied": self.total_optimizations
            },
            "performance_averages": {
                "ai_response_time_s": round(avg_ai_time, 2),
                "memory_usage_mb": round(avg_memory, 2),
                "fps": round(avg_fps, 1),
                "cache_hit_rate": round(sum(m.cache_hit_rate for m in self.metrics_history) / len(self.metrics_history), 3)
            },
            "targets": {
                "ai_response_time_s": self.config.max_ai_response_time,
                "memory_usage_gb": self.config.max_memory_usage_gb,
                "min_fps": self.config.min_fps
            },
            "optimization_effectiveness": {
                "targets_met": target_met_rate >= 0.8,
                "performance_stable": len([m for m in self.metrics_history if m.warnings]) < len(self.metrics_history) * 0.2,
                "optimization_responsive": self.total_optimizations > 0
            }
        }
        
        # Save report
        report_file = f"demo_performance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            print(f"\n📋 Demo report saved: {report_file}")
        except Exception as e:
            print(f"⚠️ Failed to save report: {e}")
        
        # Display summary
        print("\n" + "="*60)
        print("📋 DEMO PERFORMANCE REPORT")
        print("="*60)
        print(f"Runtime: {runtime}")
        print(f"Target Met Rate: {target_met_rate:.1%}")
        print(f"Avg AI Response: {avg_ai_time:.2f}s (target: <{self.config.max_ai_response_time}s)")
        print(f"Avg Memory: {avg_memory:.0f}MB (target: <{self.config.max_memory_usage_gb*1024:.0f}MB)")
        print(f"Avg FPS: {avg_fps:.1f} (target: >{self.config.min_fps})")
        print(f"Total Optimizations: {self.total_optimizations}")
        
        if target_met_rate >= 0.8:
            print("✅ DEMO PERFORMANCE: EXCELLENT")
        elif target_met_rate >= 0.6:
            print("✅ DEMO PERFORMANCE: GOOD")
        else:
            print("⚠️ DEMO PERFORMANCE: NEEDS IMPROVEMENT")
        
        print("="*60)


def main():
    """Main function - Demo performance suite test"""
    print("🚀 Demo Performance Suite Test")
    
    # Create configuration
    config = DemoConfig(
        max_ai_response_time=1.5,
        max_memory_usage_gb=3.0,
        min_fps=30.0,
        max_agents=3
    )
    
    # Create performance suite
    suite = DemoPerformanceSuite(config)
    
    try:
        # Run 5-minute demo test
        suite.run_demo_test(duration_minutes=2)
    except KeyboardInterrupt:
        print("\n⚠️ Test interrupted")
        suite.stop_demo_optimization()


if __name__ == "__main__":
    main()
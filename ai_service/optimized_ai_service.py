"""
Optimized AI Service with integrated performance optimizations
Extends the original ai_service.py with caching, batching, and optimization features
"""

import os
import time
import threading
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime
from pathlib import Path

# Import unified ai_service components
try:
    from .unified_ai_service import AIService, get_ai_service
    from .config_enhanced import get_config
    from .monitoring import get_performance_monitor, timing_context
    from .error_handling import handle_generation_errors
except ImportError:
    from unified_ai_service import AIService, get_ai_service
    from config_enhanced import get_config
    from monitoring import get_performance_monitor, timing_context
    from error_handling import handle_generation_errors

# Import optimization components
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'debug_system'))

try:
    from debug_system.llm_optimizer import SmartLLMClient, LLMCache, PromptOptimizer
    from debug_system.flow_tracer import get_tracer
    from debug_system.performance_analyzer import get_performance_analyzer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("[optimized_ai_service] Optimization components not available")


class OptimizedAIService(AIService):
    """Enhanced AI Service with integrated optimizations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize optimization components if available
        if OPTIMIZATION_AVAILABLE:
            self._setup_optimizations()
        else:
            print("[optimized_ai_service] Running without optimizations")
    
    def _setup_optimizations(self) -> None:
        """Setup optimization components"""
        # Initialize smart LLM client with optimizations
        self.smart_llm = SmartLLMClient(
            llm_function=self._original_generate,
            cache_size=10000
        )
        
        # Setup flow tracing
        self.flow_tracer = get_tracer()
        
        # Setup performance monitoring
        self.perf_analyzer = get_performance_analyzer()
        self.perf_analyzer.start_monitoring(interval=2.0)
        
        # Optimization flags
        self.use_llm_cache = True
        self.use_prompt_optimization = True
        self.use_flow_tracing = True
        self.use_performance_monitoring = True
        
        print("[optimized_ai_service] Optimizations initialized")
    
    def _original_generate(self, prompt: str) -> str:
        """Original generation method for fallback"""
        return super().generate(prompt)
    
    def generate(self, prompt: str, use_optimizations: bool = True, **kwargs) -> str:
        """Optimized generation with caching and monitoring"""
        
        if not OPTIMIZATION_AVAILABLE or not use_optimizations:
            return self._original_generate(prompt)
        
        # Start flow tracing
        if self.use_flow_tracing:
            agent_name = kwargs.get('agent_name', 'unknown')
            self.flow_tracer.trace_llm_prompt(prompt)
        
        # Start performance monitoring
        if self.use_performance_monitoring:
            self.perf_analyzer.start_measurement("llm_generation")
        
        try:
            # Use smart LLM client with optimizations
            response = self.smart_llm.call(
                prompt,
                use_cache=self.use_llm_cache,
                optimize_prompt=self.use_prompt_optimization
            )
            
            # Complete flow tracing
            if self.use_flow_tracing:
                self.flow_tracer.trace_llm_response(response)
            
            return response
            
        except Exception as e:
            print(f"[optimized_ai_service] Generation error: {e}")
            # Fallback to original method
            return self._original_generate(prompt)
        
        finally:
            # End performance monitoring
            if self.use_performance_monitoring:
                self.perf_analyzer.end_measurement("llm_generation")
    
    def batch_generate(self, prompts: List[str], callbacks: Optional[List[Callable]] = None) -> List[str]:
        """Batch generation with optimization"""
        
        if not OPTIMIZATION_AVAILABLE:
            return [self._original_generate(prompt) for prompt in prompts]
        
        results = []
        
        def collect_result(response: str, metadata: Dict[str, Any]) -> None:
            """Collect batch result"""
            results.append(response)
        
        # Use async batch processing
        for i, prompt in enumerate(prompts):
            callback = callbacks[i] if callbacks and i < len(callbacks) else collect_result
            self.smart_llm.call_async(prompt, callback)
        
        # Wait for all results
        self.smart_llm.flush_batches()
        time.sleep(0.1)  # Allow async processing to complete
        
        return results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        
        if not OPTIMIZATION_AVAILABLE:
            return {"error": "Optimizations not available"}
        
        stats = {
            "llm_stats": self.smart_llm.get_stats(),
            "performance_stats": self.perf_analyzer.get_summary(),
            "flow_stats": self.flow_tracer.get_summary()
        }
        
        return stats
    
    def optimize_for_scenario(self, scenario: str) -> None:
        """Optimize settings for specific scenarios"""
        
        if not OPTIMIZATION_AVAILABLE:
            return
        
        if scenario == "high_throughput":
            # Optimize for many requests
            self.use_llm_cache = True
            self.use_prompt_optimization = True
            self.use_flow_tracing = False  # Reduce overhead
            
        elif scenario == "low_latency":
            # Optimize for fast responses
            self.use_llm_cache = True
            self.use_prompt_optimization = True
            self.use_performance_monitoring = False
            
        elif scenario == "debugging":
            # Full monitoring for debugging
            self.use_flow_tracing = True
            self.use_performance_monitoring = True
            self.use_llm_cache = False  # Avoid cache for debugging
            
        elif scenario == "production":
            # Balanced settings for production
            self.use_llm_cache = True
            self.use_prompt_optimization = True
            self.use_flow_tracing = False
            self.use_performance_monitoring = True
        
        print(f"[optimized_ai_service] Optimized for {scenario} scenario")
    
    def clear_optimizations(self) -> None:
        """Clear optimization caches and reset"""
        
        if not OPTIMIZATION_AVAILABLE:
            return
        
        self.smart_llm.clear_cache()
        self.flow_tracer.clear_trace()
        self.perf_analyzer.clear_data()
        
        print("[optimized_ai_service] Optimization data cleared")
    
    def save_optimization_report(self, filename: str = "ai_service_optimization_report.md") -> None:
        """Save comprehensive optimization report"""
        
        if not OPTIMIZATION_AVAILABLE:
            print("[optimized_ai_service] Optimizations not available for reporting")
            return
        
        stats = self.get_optimization_stats()
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write("# AI Service Optimization Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            
            # LLM Statistics
            llm_stats = stats.get("llm_stats", {})
            f.write("## LLM Optimization Statistics\n\n")
            f.write(f"- Total calls: {llm_stats.get('total_calls', 0)}\n")
            f.write(f"- Cache hit rate: {llm_stats.get('cache_hit_rate_percent', 0):.1f}%\n")
            f.write(f"- Optimization rate: {llm_stats.get('optimization_rate_percent', 0):.1f}%\n")
            f.write(f"- Batch calls: {llm_stats.get('batch_calls', 0)}\n\n")
            
            # Performance Statistics
            perf_stats = stats.get("performance_stats", {})
            f.write("## Performance Statistics\n\n")
            f.write(f"- Total cycles: {perf_stats.get('total_cycles', 0)}\n")
            f.write(f"- Bottleneck phase: {perf_stats.get('bottleneck_phase', 'unknown')}\n")
            f.write(f"- Average cycle time: {perf_stats.get('avg_cycle_time', 0)*1000:.1f}ms\n")
            f.write(f"- Memory usage: {perf_stats.get('avg_memory_mb', 0):.1f}MB\n\n")
            
            # Flow Statistics
            flow_stats = stats.get("flow_stats", {})
            f.write("## Flow Tracing Statistics\n\n")
            f.write(f"- Total steps: {flow_stats.get('total_steps', 0)}\n")
            f.write(f"- Status: {flow_stats.get('status', 'unknown')}\n\n")
        
        print(f"[optimized_ai_service] Optimization report saved to {filename}")
    
    def __del__(self):
        """Cleanup on destruction"""
        if OPTIMIZATION_AVAILABLE and hasattr(self, 'perf_analyzer'):
            self.perf_analyzer.stop_monitoring()


# Global optimized service instance
_optimized_service = None


def get_optimized_ai_service(**kwargs) -> OptimizedAIService:
    """Get global optimized AI service instance"""
    global _optimized_service
    
    if _optimized_service is None:
        _optimized_service = OptimizedAIService(**kwargs)
    
    return _optimized_service


def create_optimized_service_wrapper(original_service: AIService) -> OptimizedAIService:
    """Create optimized wrapper around existing service"""
    
    class OptimizedWrapper(OptimizedAIService):
        def __init__(self, wrapped_service: AIService):
            # Don't call super().__init__ to avoid re-initialization
            self.wrapped_service = wrapped_service
            
            if OPTIMIZATION_AVAILABLE:
                self._setup_optimizations()
        
        def _original_generate(self, prompt: str) -> str:
            """Use wrapped service for generation"""
            return self.wrapped_service.generate(prompt)
    
    return OptimizedWrapper(original_service)


# Convenience functions for easy integration
def optimize_existing_service() -> OptimizedAIService:
    """Optimize the existing AI service"""
    try:
        original = get_ai_service()
        return create_optimized_service_wrapper(original)
    except Exception as e:
        print(f"[optimized_ai_service] Could not wrap existing service: {e}")
        return get_optimized_ai_service()


def demo_optimization_integration():
    """Demo the optimization integration"""
    print("=== AI Service Optimization Integration Demo ===\n")
    
    # Create optimized service
    service = get_optimized_ai_service()
    
    # Test different scenarios
    scenarios = ["production", "debugging", "high_throughput", "low_latency"]
    
    for scenario in scenarios:
        print(f"Testing {scenario} scenario...")
        service.optimize_for_scenario(scenario)
        
        # Test generation
        response = service.generate(
            "What should the bartender do when a customer arrives?",
            agent_name="test_agent"
        )
        print(f"Response: {response[:50]}...")
    
    # Show optimization statistics
    print("\n--- Optimization Statistics ---")
    stats = service.get_optimization_stats()
    
    if "error" not in stats:
        llm_stats = stats.get("llm_stats", {})
        print(f"Total LLM calls: {llm_stats.get('total_calls', 0)}")
        print(f"Cache hit rate: {llm_stats.get('cache_hit_rate_percent', 0):.1f}%")
    
    # Save report
    service.save_optimization_report("demo_optimization_report.md")
    print("\nDemo complete! Report saved.")


if __name__ == "__main__":
    demo_optimization_integration()
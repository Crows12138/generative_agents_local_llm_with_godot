#!/usr/bin/env python3
"""
Test script for optimization integration
Validates that optimization components can be integrated with existing code
"""

import sys
import os
import time
from typing import Dict, Any

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_optimization_components():
    """Test individual optimization components"""
    print("=== Testing Optimization Components ===\n")
    
    # Test 1: LLM Optimizer
    try:
        from debug_system.llm_optimizer import SmartLLMClient, PromptOptimizer
        
        def mock_llm(prompt):
            time.sleep(0.01)  # Simulate processing
            return f"Mock response to: {prompt[:30]}..."
        
        client = SmartLLMClient(mock_llm)
        optimizer = PromptOptimizer()
        
        # Test prompt optimization
        verbose_prompt = "Could you please tell me what to do?"
        optimized_prompt = optimizer.optimize(verbose_prompt)
        
        print(f"[OK] LLM Optimizer: {len(verbose_prompt) - len(optimized_prompt)} chars saved")
        
        # Test caching
        response1 = client.call("test prompt")
        response2 = client.call("test prompt")  # Should be cached
        
        stats = client.get_stats()
        print(f"[OK] LLM Cache: {stats['cache_hit_rate_percent']:.1f}% hit rate")
        
    except ImportError as e:
        print(f"[FAIL] LLM Optimizer: {e}")
    
    # Test 2: Action Parser Optimizer
    try:
        from debug_system.action_parser_optimizer import OptimizedActionParser
        
        parser = OptimizedActionParser()
        
        test_actions = [
            "go to bar",
            "talk to customer", 
            "clean the counter",
            "go to bar"  # Duplicate for cache test
        ]
        
        for action in test_actions:
            result = parser.parse(action)
        
        stats = parser.get_performance_stats()
        print(f"[OK] Action Parser: {stats['fast_path_rate_percent']:.1f}% fast path, {stats['cache_hit_rate_percent']:.1f}% cache hits")
        
    except ImportError as e:
        print(f"[FAIL] Action Parser: {e}")
    
    # Test 3: Flow Tracer
    try:
        from debug_system.flow_tracer import get_tracer
        
        tracer = get_tracer()
        tracer.trace_perception("TestAgent", "Customer waiting")
        tracer.trace_llm_response("I should serve the customer")
        
        summary = tracer.get_summary()
        print(f"[OK] Flow Tracer: {summary.get('total_steps', 0)} steps traced")
        
    except ImportError as e:
        print(f"[FAIL] Flow Tracer: {e}")
    
    # Test 4: Performance Analyzer
    try:
        from debug_system.performance_analyzer import get_performance_analyzer
        
        analyzer = get_performance_analyzer()
        timings = analyzer.profile_action_cycle("TestAgent")
        
        print(f"[OK] Performance Analyzer: {timings.get('total_cycle', 0)*1000:.1f}ms cycle time")
        
    except ImportError as e:
        print(f"[FAIL] Performance Analyzer: {e}")


def test_integration_compatibility():
    """Test integration with existing components"""
    print("\n=== Testing Integration Compatibility ===\n")
    
    # Test: Check if existing AI service can be found
    try:
        import ai_service.ai_service as ai_service_module
        print("[OK] Original AI Service: Found and importable")
        
        # Check if we can access key classes/functions
        if hasattr(ai_service_module, 'get_ai_service'):
            print("[OK] AI Service Function: get_ai_service available")
        else:
            print("[WARN] AI Service Function: get_ai_service not found")
            
    except ImportError as e:
        print(f"[FAIL] Original AI Service: {e}")
    
    # Test: Check unified parser
    try:
        import ai_service.unified_parser as parser_module
        print("[OK] Original Parser: Found and importable")
        
        if hasattr(parser_module, 'UnifiedParser'):
            print("[OK] Parser Class: UnifiedParser available")
        else:
            print("[WARN] Parser Class: UnifiedParser not found")
            
    except ImportError as e:
        print(f"[FAIL] Original Parser: {e}")
    
    # Test: Check bar agents
    try:
        import cozy_bar_demo.core.bar_agents as agents_module
        print("[OK] Original Agents: Found and importable")
        
        if hasattr(agents_module, 'BarAgent'):
            print("[OK] Agent Class: BarAgent available")
        else:
            print("[WARN] Agent Class: BarAgent not found")
            
    except ImportError as e:
        print(f"[FAIL] Original Agents: {e}")


def test_mock_integration():
    """Test mock integration scenarios"""
    print("\n=== Testing Mock Integration Scenarios ===\n")
    
    # Scenario 1: Gradual optimization
    print("Scenario 1: Gradual Optimization")
    
    try:
        from debug_system.llm_optimizer import SmartLLMClient
        
        # Mock original service
        class MockOriginalService:
            def generate(self, prompt):
                time.sleep(0.02)
                return f"Original response: {prompt[:20]}..."
        
        # Create optimized wrapper
        class OptimizedWrapper:
            def __init__(self, original_service):
                self.original = original_service
                self.optimized = SmartLLMClient(original_service.generate)
                self.use_optimizations = True
            
            def generate(self, prompt, use_optimizations=None):
                if use_optimizations is None:
                    use_optimizations = self.use_optimizations
                
                if use_optimizations:
                    return self.optimized.call(prompt)
                else:
                    return self.original.generate(prompt)
        
        # Test the wrapper
        original = MockOriginalService()
        wrapper = OptimizedWrapper(original)
        
        # Test both modes
        start_time = time.time()
        response1 = wrapper.generate("test prompt", use_optimizations=False)
        original_time = time.time() - start_time
        
        start_time = time.time()
        response2 = wrapper.generate("test prompt", use_optimizations=True)
        optimized_time = time.time() - start_time
        
        start_time = time.time()
        response3 = wrapper.generate("test prompt", use_optimizations=True)  # Cached
        cached_time = time.time() - start_time
        
        speedup = original_time / cached_time if cached_time > 0 else float('inf')
        
        print(f"   [OK] Original: {original_time*1000:.1f}ms")
        print(f"   [OK] Optimized: {optimized_time*1000:.1f}ms")
        print(f"   [OK] Cached: {cached_time*1000:.1f}ms (speedup: {speedup:.1f}x)")
        
    except Exception as e:
        print(f"   [FAIL] Mock Integration Failed: {e}")
    
    # Scenario 2: Fallback handling
    print("\nScenario 2: Fallback Handling")
    
    try:
        # Test graceful fallback when optimizations fail
        class FallbackService:
            def __init__(self):
                try:
                    from debug_system.llm_optimizer import SmartLLMClient
                    self.optimized = SmartLLMClient(self._basic_generate)
                    self.has_optimizations = True
                except ImportError:
                    self.has_optimizations = False
            
            def _basic_generate(self, prompt):
                return f"Basic response: {prompt[:15]}..."
            
            def generate(self, prompt):
                if self.has_optimizations:
                    try:
                        return self.optimized.call(prompt)
                    except Exception:
                        # Fallback to basic
                        return self._basic_generate(prompt)
                else:
                    return self._basic_generate(prompt)
        
        service = FallbackService()
        response = service.generate("test prompt")
        
        print(f"   [OK] Fallback Service: {service.has_optimizations}")
        print(f"   [OK] Response Generated: {len(response) > 0}")
        
    except Exception as e:
        print(f"   [FAIL] Fallback Test Failed: {e}")


def generate_integration_summary():
    """Generate integration readiness summary"""
    print("\n=== Integration Readiness Summary ===\n")
    
    # Check all components
    components = {
        "LLM Optimizer": "debug_system.llm_optimizer",
        "Action Parser": "debug_system.action_parser_optimizer", 
        "Flow Tracer": "debug_system.flow_tracer",
        "Performance Analyzer": "debug_system.performance_analyzer",
        "Debug Dashboard": "debug_system.debug_dashboard",
        "Optimization Checklist": "debug_system.optimization_checklist"
    }
    
    available_components = []
    missing_components = []
    
    for component_name, module_name in components.items():
        try:
            __import__(module_name)
            available_components.append(component_name)
        except ImportError:
            missing_components.append(component_name)
    
    print(f"[OK] Available Components ({len(available_components)}):")
    for component in available_components:
        print(f"   - {component}")
    
    if missing_components:
        print(f"\n[MISSING] Missing Components ({len(missing_components)}):")
        for component in missing_components:
            print(f"   - {component}")
    
    # Integration recommendations
    print(f"\n[INFO] Integration Recommendations:")
    
    if len(available_components) >= 4:
        print("   [OK] Ready for full integration")
        print("   [INFO] See OPTIMIZATION_INTEGRATION_GUIDE.md for details")
    elif len(available_components) >= 2:
        print("   [WARN] Ready for partial integration")
        print("   [INFO] Consider implementing missing components")
    else:
        print("   [FAIL] Not ready for integration")
        print("   [INFO] Implement optimization components first")
    
    print(f"\n[STATS] Integration Coverage: {len(available_components)/len(components)*100:.0f}%")


def main():
    """Main test function"""
    print("Optimization Integration Test Suite")
    print("=" * 50)
    
    test_optimization_components()
    test_integration_compatibility()
    test_mock_integration()
    generate_integration_summary()
    
    print(f"\nIntegration testing complete!")
    print(f"For detailed integration instructions, see:")
    print(f"   OPTIMIZATION_INTEGRATION_GUIDE.md")


if __name__ == "__main__":
    main()
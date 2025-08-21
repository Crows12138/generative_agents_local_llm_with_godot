#!/usr/bin/env python3
"""
Complete optimization system demonstration
Shows before/after performance comparisons for all optimizations
"""

import time
import sys
import os
from typing import Dict, Any, List

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from debug_system.optimization_checklist import OptimizationChecklist, OptimizationMeasurement
from debug_system.llm_optimizer import SmartLLMClient, PromptOptimizer
from debug_system.action_parser_optimizer import OptimizedActionParser
from debug_system.llm_parser_study import LLMOutputParser


class OptimizationDemo:
    """Comprehensive optimization demonstration"""
    
    def __init__(self):
        self.checklist = OptimizationChecklist()
        self.measurement = OptimizationMeasurement()
        
        # Mock LLM function for testing
        def mock_llm(prompt):
            time.sleep(0.05)  # Simulate LLM delay
            return f"Mock response to: {prompt[:30]}..."
        
        self.smart_llm = SmartLLMClient(mock_llm)
        self.prompt_optimizer = PromptOptimizer()
        self.optimized_parser = OptimizedActionParser()
        self.basic_parser = LLMOutputParser()
    
    def demo_llm_optimizations(self) -> Dict[str, Any]:
        """Demonstrate LLM optimization improvements"""
        print("\n=== LLM Optimization Demo ===")
        
        test_prompts = [
            "Could you please tell me what the bartender should do when a customer arrives?",
            "I would like you to make sure to explain the cleaning process",
            "Please describe how to serve drinks efficiently",
            "Could you help me understand inventory management?",
            "What should I do when the bar gets busy?"
        ]
        
        # Test without optimizations
        print("Testing without optimizations...")
        start_time = time.time()
        basic_responses = []
        for prompt in test_prompts:
            response = self.smart_llm.llm_function(prompt)
            basic_responses.append(response)
        basic_time = time.time() - start_time
        
        # Test with optimizations
        print("Testing with optimizations...")
        start_time = time.time()
        optimized_responses = []
        for prompt in test_prompts:
            response = self.smart_llm.call(prompt)
            optimized_responses.append(response)
        optimized_time = time.time() - start_time
        
        # Test cache effectiveness
        print("Testing cache effectiveness...")
        start_time = time.time()
        for prompt in test_prompts:  # Same prompts again
            response = self.smart_llm.call(prompt)
        cached_time = time.time() - start_time
        
        llm_stats = self.smart_llm.get_stats()
        
        results = {
            "basic_time": basic_time,
            "optimized_time": optimized_time,
            "cached_time": cached_time,
            "improvement_ratio": basic_time / optimized_time if optimized_time > 0 else 0,
            "cache_speedup": optimized_time / cached_time if cached_time > 0 else 0,
            "llm_stats": llm_stats
        }
        
        print(f"Basic time: {basic_time:.3f}s")
        print(f"Optimized time: {optimized_time:.3f}s")
        print(f"Cached time: {cached_time:.3f}s")
        print(f"Improvement: {results['improvement_ratio']:.1f}x")
        print(f"Cache speedup: {results['cache_speedup']:.1f}x")
        
        return results
    
    def demo_parsing_optimizations(self) -> Dict[str, Any]:
        """Demonstrate action parsing optimization improvements"""
        print("\n=== Action Parsing Optimization Demo ===")
        
        test_actions = [
            "go to bar",
            "I should walk to the kitchen area",
            "talk to customer Alice",
            "clean the bar counter thoroughly", 
            "let me check the inventory",
            "serve drink to customer",
            "prepare a cocktail",
            "wipe down the tables",
            "go to bar",  # Duplicate for cache test
            "clean the bar counter thoroughly"  # Duplicate for cache test
        ] * 10  # Repeat for better timing
        
        # Test basic parser
        print("Testing basic parser...")
        start_time = time.time()
        basic_results = []
        for action in test_actions:
            result = self.basic_parser.parse_with_explanation(action)
            basic_results.append(result)
        basic_time = time.time() - start_time
        
        # Test optimized parser
        print("Testing optimized parser...")
        start_time = time.time()
        optimized_results = []
        for action in test_actions:
            result = self.optimized_parser.parse(action)
            optimized_results.append(result)
        optimized_time = time.time() - start_time
        
        parser_stats = self.optimized_parser.get_performance_stats()
        
        results = {
            "basic_time": basic_time,
            "optimized_time": optimized_time,
            "improvement_ratio": basic_time / optimized_time if optimized_time > 0 else 0,
            "parser_stats": parser_stats,
            "test_count": len(test_actions)
        }
        
        print(f"Basic parsing time: {basic_time:.3f}s ({len(test_actions)} actions)")
        print(f"Optimized parsing time: {optimized_time:.3f}s")
        print(f"Improvement: {results['improvement_ratio']:.1f}x")
        print(f"Fast path hits: {parser_stats['fast_path_rate_percent']:.1f}%")
        print(f"Cache hits: {parser_stats['cache_hit_rate_percent']:.1f}%")
        
        return results
    
    def demo_prompt_optimization(self) -> Dict[str, Any]:
        """Demonstrate prompt optimization improvements"""
        print("\n=== Prompt Optimization Demo ===")
        
        verbose_prompts = [
            "Could you please make sure to tell me what the bartender should do when a customer arrives at the bar?",
            "I would like you to explain in detail how to clean the bar counter properly and thoroughly",
            "Please make sure to describe the process of serving drinks efficiently and quickly to customers",
            "Could you help me understand how to manage inventory and make sure everything is well organized?",
            "I need you to tell me what I should do when the bar gets very busy and there are many customers waiting"
        ]
        
        print("Optimizing prompts...")
        total_original_length = 0
        total_optimized_length = 0
        
        for prompt in verbose_prompts:
            original_length = len(prompt)
            optimized = self.prompt_optimizer.optimize(prompt)
            optimized_length = len(optimized)
            
            total_original_length += original_length
            total_optimized_length += optimized_length
            
            print(f"Original ({original_length} chars): {prompt}")
            print(f"Optimized ({optimized_length} chars): {optimized}")
            print(f"Savings: {original_length - optimized_length} chars\n")
        
        optimizer_stats = self.prompt_optimizer.get_optimization_stats()
        
        results = {
            "total_original_length": total_original_length,
            "total_optimized_length": total_optimized_length,
            "char_savings": total_original_length - total_optimized_length,
            "savings_percent": ((total_original_length - total_optimized_length) / total_original_length * 100) if total_original_length > 0 else 0,
            "optimizer_stats": optimizer_stats
        }
        
        print(f"Total character savings: {results['char_savings']} ({results['savings_percent']:.1f}%)")
        print(f"Estimated token savings: {results['char_savings'] // 4}")
        
        return results
    
    def run_complete_demo(self) -> None:
        """Run complete optimization demonstration"""
        print("="*60)
        print("GENERATIVE AGENTS OPTIMIZATION SYSTEM DEMO")
        print("="*60)
        
        # Record baseline metrics
        self.checklist.record_baseline("llm_response_time", 0.25)
        self.checklist.record_baseline("parsing_time", 0.01)
        self.checklist.record_baseline("token_usage", 1000)
        
        # Run optimization demos
        llm_results = self.demo_llm_optimizations()
        parsing_results = self.demo_parsing_optimizations()
        prompt_results = self.demo_prompt_optimization()
        
        # Update optimization items with measured improvements
        self.checklist.measure_improvement("llm_optimization", "prompt_caching", 
                                         "llm_response_time", llm_results["optimized_time"])
        
        self.checklist.measure_improvement("action_parsing", "regex_precompilation",
                                         "parsing_time", parsing_results["optimized_time"])
        
        self.checklist.measure_improvement("llm_optimization", "token_reduction",
                                         "token_usage", prompt_results["total_optimized_length"])
        
        # Mark optimizations as completed
        optimizations_completed = [
            ("llm_optimization", "prompt_caching", "Implemented with OrderedDict cache"),
            ("llm_optimization", "token_reduction", "Implemented prompt optimization rules"),
            ("action_parsing", "regex_precompilation", "All patterns pre-compiled"),
            ("action_parsing", "lookup_tables", "Fast lookup tables implemented"),
            ("action_parsing", "pattern_caching", "LRU cache implemented"),
            ("memory_optimization", "history_limits", "Cache size limits enforced")
        ]
        
        for category, name, notes in optimizations_completed:
            self.checklist.update_item_status(category, name, "completed", notes)
        
        # Generate comprehensive results
        print("\n" + "="*60)
        print("OPTIMIZATION RESULTS SUMMARY")
        print("="*60)
        
        print(f"\n🚀 LLM Optimizations:")
        print(f"  • Speed improvement: {llm_results['improvement_ratio']:.1f}x")
        print(f"  • Cache hit rate: {llm_results['llm_stats']['cache_hit_rate_percent']:.1f}%")
        print(f"  • Cache speedup: {llm_results['cache_speedup']:.1f}x")
        
        print(f"\n⚡ Parsing Optimizations:")
        print(f"  • Speed improvement: {parsing_results['improvement_ratio']:.1f}x")
        print(f"  • Fast path usage: {parsing_results['parser_stats']['fast_path_rate_percent']:.1f}%")
        print(f"  • Cache hit rate: {parsing_results['parser_stats']['cache_hit_rate_percent']:.1f}%")
        
        print(f"\n💡 Prompt Optimizations:")
        print(f"  • Character reduction: {prompt_results['savings_percent']:.1f}%")
        print(f"  • Estimated token savings: {prompt_results['char_savings'] // 4}")
        
        # Show quick wins implemented
        quick_wins = self.checklist.get_quick_wins()
        completed_quick_wins = [item for item in quick_wins if item.status == "completed"]
        
        print(f"\n✅ Quick Wins Implemented: {len(completed_quick_wins)}/{len(quick_wins)}")
        for item in completed_quick_wins:
            improvement = f" ({item.measured_improvement:+.1f}%)" if item.measured_improvement else ""
            print(f"  • {item.name}{improvement}")
        
        # Generate reports
        print(f"\n📊 Generating optimization reports...")
        priority_report = self.checklist.generate_priority_report()
        
        with open("optimization_demo_results.md", "w", encoding='utf-8') as f:
            f.write("# Optimization Demo Results\n\n")
            f.write(f"**Demo Date**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("## Performance Improvements\n\n")
            f.write(f"### LLM Optimizations\n")
            f.write(f"- Speed improvement: {llm_results['improvement_ratio']:.1f}x\n")
            f.write(f"- Cache hit rate: {llm_results['llm_stats']['cache_hit_rate_percent']:.1f}%\n")
            f.write(f"- Cache speedup: {llm_results['cache_speedup']:.1f}x\n\n")
            
            f.write(f"### Parsing Optimizations\n")
            f.write(f"- Speed improvement: {parsing_results['improvement_ratio']:.1f}x\n")
            f.write(f"- Fast path usage: {parsing_results['parser_stats']['fast_path_rate_percent']:.1f}%\n")
            f.write(f"- Cache hit rate: {parsing_results['parser_stats']['cache_hit_rate_percent']:.1f}%\n\n")
            
            f.write(f"### Prompt Optimizations\n")
            f.write(f"- Character reduction: {prompt_results['savings_percent']:.1f}%\n")
            f.write(f"- Estimated token savings: {prompt_results['char_savings'] // 4}\n\n")
            
            f.write("## Detailed Analysis\n\n")
            f.write(priority_report)
        
        self.checklist.export_checklist("optimization_demo_checklist.json")
        
        print("📁 Reports generated:")
        print("  • optimization_demo_results.md")
        print("  • optimization_demo_checklist.json")
        
        print(f"\n🎉 Optimization demo complete!")


def main():
    """Main demo function"""
    demo = OptimizationDemo()
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "--llm":
            demo.demo_llm_optimizations()
        elif sys.argv[1] == "--parsing":
            demo.demo_parsing_optimizations()
        elif sys.argv[1] == "--prompts":
            demo.demo_prompt_optimization()
        elif sys.argv[1] == "--help":
            print("Optimization Demo Options:")
            print("  --llm     : Demo LLM optimizations only")
            print("  --parsing : Demo parsing optimizations only")
            print("  --prompts : Demo prompt optimizations only")
            print("  --help    : Show this help")
            print("  (no args) : Run complete demo")
        else:
            print("Unknown option. Use --help for available options.")
    else:
        demo.run_complete_demo()


if __name__ == "__main__":
    main()
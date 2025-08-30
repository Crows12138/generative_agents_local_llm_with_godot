"""
Performance monitoring utilities for tracking AI call times and system performance.
"""

import time
from datetime import datetime
from typing import Dict, List, Optional
from collections import defaultdict


class PerformanceMonitor:
    """
    Monitor and track performance metrics for AI calls and system operations.
    """
    
    def __init__(self):
        self.metrics = {}
        self.completed_metrics = defaultdict(list)
        self.ai_call_count = 0
        self.ai_total_time = 0.0
        
    def start_timer(self, name: str) -> None:
        """Start timing an operation."""
        self.metrics[name] = {
            'start': time.time(),
            'start_str': datetime.now().strftime('%H:%M:%S.%f')[:-3]
        }
        
    def end_timer(self, name: str) -> float:
        """End timing an operation and return elapsed time."""
        if name not in self.metrics:
            print(f"[PERF WARNING] Timer '{name}' was not started")
            return 0.0
            
        start_time = self.metrics[name]['start']
        elapsed = time.time() - start_time
        
        # Print performance info
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"[PERF {timestamp}] {name}: {elapsed:.3f}s")
        
        # Store for statistics
        self.completed_metrics[name].append(elapsed)
        
        # Clean up
        del self.metrics[name]
        
        return elapsed
    
    def log_ai_call(self, prompt_preview: str, start: bool = True) -> None:
        """Log AI call start/end with prompt preview."""
        timestamp = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        
        if start:
            # Truncate prompt for display
            preview = prompt_preview[:80].replace('\n', ' ')
            if len(prompt_preview) > 80:
                preview += "..."
            print(f"[AI_CALL_START {timestamp}] {preview}")
        else:
            print(f"[AI_CALL_END {timestamp}]")
            
    def track_ai_call(self, elapsed: float) -> None:
        """Track AI call statistics."""
        self.ai_call_count += 1
        self.ai_total_time += elapsed
        
    def print_statistics(self) -> None:
        """Print performance statistics summary."""
        print("\n" + "="*60)
        print("[PERFORMANCE STATISTICS]")
        print("="*60)
        
        # Overall metrics
        total_time = sum(sum(times) for times in self.completed_metrics.values())
        print(f"\nTotal tracked time: {total_time:.2f}s")
        
        # Per-operation statistics
        print("\nOperation Statistics:")
        for operation, times in self.completed_metrics.items():
            if times:
                avg_time = sum(times) / len(times)
                max_time = max(times)
                min_time = min(times)
                total = sum(times)
                print(f"  {operation}:")
                print(f"    Count: {len(times)}")
                print(f"    Avg: {avg_time:.3f}s")
                print(f"    Min: {min_time:.3f}s")
                print(f"    Max: {max_time:.3f}s")
                print(f"    Total: {total:.3f}s")
        
        # AI call statistics
        if self.ai_call_count > 0:
            print(f"\nAI Call Statistics:")
            print(f"  Total calls: {self.ai_call_count}")
            print(f"  Total time: {self.ai_total_time:.2f}s")
            print(f"  Average time: {self.ai_total_time/self.ai_call_count:.3f}s")
            
            # Calculate AI percentage
            if total_time > 0:
                ai_percentage = (self.ai_total_time / total_time) * 100
                print(f"  AI time percentage: {ai_percentage:.1f}%")
        
        print("="*60)


# Global instance for easy access
perf_monitor = PerformanceMonitor()
"""
Performance Monitoring and Logging
Performance monitoring and logging module providing detailed performance metrics and structured logging
"""

import time
import logging
import psutil
import threading
import gc
import weakref
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
from collections import deque, defaultdict
import json

@dataclass
class PerformanceMetrics:
    """Performance metrics"""
    request_count: int = 0
    total_tokens: int = 0
    total_time: float = 0.0
    avg_time_per_request: float = 0.0
    avg_tokens_per_second: float = 0.0
    peak_memory_mb: float = 0.0
    current_memory_mb: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    
    def update_request(self, tokens: int, duration: float, memory_mb: float):
        """Update request metrics"""
        self.request_count += 1
        self.total_tokens += tokens
        self.total_time += duration
        self.avg_time_per_request = self.total_time / self.request_count
        self.avg_tokens_per_second = self.total_tokens / self.total_time if self.total_time > 0 else 0
        self.current_memory_mb = memory_mb
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
    
    def update_error(self):
        """Update error metrics"""
        self.error_count += 1
        self.error_rate = self.error_count / max(self.request_count, 1)

@dataclass
class MemoryStats:
    """Memory statistics - Integrated from memory_optimizer"""
    total_mb: float
    used_mb: float
    available_mb: float
    percent_used: float
    python_objects: int
    gc_collections: Dict[int, int]

class MemoryPool:
    """Object pool manager - Integrated from memory_optimizer"""
    
    def __init__(self):
        self.pools: Dict[str, List] = {}
        self.max_sizes: Dict[str, int] = {}
    
    def get_object(self, pool_name: str, factory_func=None):
        """Get object from pool"""
        if pool_name not in self.pools:
            self.pools[pool_name] = []
            self.max_sizes[pool_name] = 10
        
        if self.pools[pool_name]:
            return self.pools[pool_name].pop()
        elif factory_func:
            return factory_func()
        return None
    
    def return_object(self, pool_name: str, obj):
        """Return object to pool"""
        if pool_name not in self.pools:
            self.pools[pool_name] = []
            self.max_sizes[pool_name] = 10
        
        if len(self.pools[pool_name]) < self.max_sizes[pool_name]:
            # Clean object state
            if hasattr(obj, 'reset'):
                obj.reset()
            self.pools[pool_name].append(obj)

class WeakReferenceManager:
    """Weak reference manager - Integrated from memory_optimizer"""
    
    def __init__(self):
        self.references: Dict[str, weakref.WeakSet] = {}
        self.cleanup_callbacks: Dict[str, List] = {}
    
    def register_object(self, category: str, obj):
        """Register object weak reference"""
        if category not in self.references:
            self.references[category] = weakref.WeakSet()
            self.cleanup_callbacks[category] = []
        
        self.references[category].add(obj)
    
    def add_cleanup_callback(self, category: str, callback):
        """Add cleanup callback"""
        if category not in self.cleanup_callbacks:
            self.cleanup_callbacks[category] = []
        self.cleanup_callbacks[category].append(callback)
    
    def cleanup_category(self, category: str):
        """Clean up objects in specified category"""
        if category in self.references:
            # Execute cleanup callbacks
            for callback in self.cleanup_callbacks.get(category, []):
                try:
                    callback()
                except Exception as e:
                    print(f"Cleanup callback error: {e}")
            
            # Clear weak reference set
            self.references[category].clear()
    
    def get_alive_count(self, category: str) -> int:
        """Get count of alive objects"""
        if category in self.references:
            return len(self.references[category])
        return 0

class PerformanceMonitor:
    """Performance monitor"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = PerformanceMetrics()
        self.request_times = deque(maxlen=window_size)
        self.request_tokens = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # Metrics by model
        self.model_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        
        # Memory optimization components - Integrated from memory_optimizer
        self.memory_pool = MemoryPool()
        self.weak_ref_manager = WeakReferenceManager()
        self.memory_history: List[MemoryStats] = []
        
        # Memory thresholds
        self.warning_threshold_percent = 75.0
        self.critical_threshold_percent = 85.0
        self.cleanup_threshold_percent = 80.0
        
    def get_memory_usage(self) -> float:
        """Get current memory usage (MB)"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def record_request(self, model_key: str, tokens: int, duration: float):
        """Record request"""
        with self.lock:
            memory_mb = self.get_memory_usage()
            
            # Update global metrics
            self.metrics.update_request(tokens, duration, memory_mb)
            
            # Update model-specific metrics
            self.model_metrics[model_key].update_request(tokens, duration, memory_mb)
            
            # Update sliding window
            self.request_times.append(duration)
            self.request_tokens.append(tokens)
            self.memory_usage.append(memory_mb)
    
    def record_error(self, model_key: str):
        """Record error"""
        with self.lock:
            self.metrics.update_error()
            self.model_metrics[model_key].update_error()
    
    def get_current_stats(self) -> Dict[str, Any]:
        """Get current statistics"""
        with self.lock:
            uptime = time.time() - self.start_time
            
            # Recent performance data
            recent_avg_time = sum(self.request_times) / len(self.request_times) if self.request_times else 0
            recent_avg_tokens = sum(self.request_tokens) / len(self.request_tokens) if self.request_tokens else 0
            recent_avg_memory = sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0
            
            return {
                'uptime_seconds': uptime,
                'global_metrics': {
                    'requests': self.metrics.request_count,
                    'total_tokens': self.metrics.total_tokens,
                    'avg_time_per_request': self.metrics.avg_time_per_request,
                    'avg_tokens_per_second': self.metrics.avg_tokens_per_second,
                    'error_count': self.metrics.error_count,
                    'error_rate': self.metrics.error_rate,
                    'peak_memory_mb': self.metrics.peak_memory_mb,
                    'current_memory_mb': self.metrics.current_memory_mb,
                },
                'recent_metrics': {
                    'avg_time_per_request': recent_avg_time,
                    'avg_tokens_per_request': recent_avg_tokens,
                    'avg_memory_mb': recent_avg_memory,
                    'window_size': len(self.request_times),
                },
                'model_metrics': {
                    model: {
                        'requests': metrics.request_count,
                        'total_tokens': metrics.total_tokens,
                        'avg_time': metrics.avg_time_per_request,
                        'tokens_per_sec': metrics.avg_tokens_per_second,
                        'error_count': metrics.error_count,
                        'error_rate': metrics.error_rate,
                    }
                    for model, metrics in self.model_metrics.items()
                }
            }
    
    def reset_stats(self):
        """Reset statistics"""
        with self.lock:
            self.metrics = PerformanceMetrics()
            self.model_metrics.clear()
            self.request_times.clear()
            self.request_tokens.clear()
            self.memory_usage.clear()
            self.start_time = time.time()

class TimingContext:
    """Timing context manager"""
    
    def __init__(self, monitor: PerformanceMonitor, model_key: str, 
                 logger: Optional[logging.Logger] = None):
        self.monitor = monitor
        self.model_key = model_key
        self.logger = logger
        self.start_time = None
        self.tokens = 0
    
    def __enter__(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.info(f"Starting generation with model '{self.model_key}'")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        
        if exc_type is None:
            # Successfully completed
            self.monitor.record_request(self.model_key, self.tokens, duration)
            if self.logger:
                self.logger.info(
                    f"Generation completed: model={self.model_key}, "
                    f"tokens={self.tokens}, duration={duration:.3f}s"
                )
        else:
            # Error occurred
            self.monitor.record_error(self.model_key)
            if self.logger:
                self.logger.error(
                    f"Generation failed: model={self.model_key}, "
                    f"duration={duration:.3f}s, error={exc_val}"
                )
    
    def set_tokens(self, tokens: int):
        """Set the number of generated tokens"""
        self.tokens = tokens

def monitor_performance(monitor: PerformanceMonitor, model_key: str = None):
    """Performance monitoring decorator"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Try to get model_key from parameters
            actual_model_key = model_key
            if actual_model_key is None:
                # Look for model_key in function parameters
                if 'model_key' in kwargs:
                    actual_model_key = kwargs['model_key'] or 'unknown'
                elif len(args) > 1:
                    actual_model_key = args[1] or 'unknown'
                else:
                    actual_model_key = 'unknown'
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # Try to calculate token count
                tokens = 0
                if isinstance(result, str):
                    tokens = len(result.split())  # Simple token estimation
                
                duration = time.time() - start_time
                monitor.record_request(actual_model_key, tokens, duration)
                
                return result
                
            except Exception as e:
                monitor.record_error(actual_model_key)
                raise
        
        return wrapper
    return decorator

# Logging configuration
class AIServiceLogger:
    """AI service logger manager"""
    
    def __init__(self, name: str = "ai_service", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Avoid duplicate handlers
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup log handlers"""
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # File handler
        try:
            file_handler = logging.FileHandler('ai_service.log', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
        except:
            file_handler = None
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if file_handler:
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """Get logger"""
        return self.logger

# Global instances
_performance_monitor = PerformanceMonitor()
_ai_logger = AIServiceLogger()

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor"""
    return _performance_monitor

def get_ai_logger() -> logging.Logger:
    """Get AI service logger"""
    return _ai_logger.get_logger()

@contextmanager
def timing_context(model_key: str):
    """Timing context manager"""
    monitor = get_performance_monitor()
    logger = get_ai_logger()
    
    with TimingContext(monitor, model_key, logger) as ctx:
        yield ctx

# API endpoint support
def get_health_status() -> Dict[str, Any]:
    """Get health status"""
    monitor = get_performance_monitor()
    stats = monitor.get_current_stats()
    
    # Determine health status
    is_healthy = (
        stats['global_metrics']['error_rate'] < 0.5 and  # Error rate less than 50%
        stats['global_metrics']['current_memory_mb'] < 8000  # Memory usage less than 8GB
    )
    
    return {
        'status': 'healthy' if is_healthy else 'unhealthy',
        'timestamp': time.time(),
        'metrics': stats
    }

def get_metrics_json() -> str:
    """Get metrics in JSON format"""
    monitor = get_performance_monitor()
    stats = monitor.get_current_stats()
    return json.dumps(stats, indent=2)

# Performance analysis tools
def analyze_performance() -> Dict[str, Any]:
    """Analyze performance data"""
    monitor = get_performance_monitor()
    stats = monitor.get_current_stats()
    
    analysis = {
        'performance_grade': 'A',  # A, B, C, D, F
        'bottlenecks': [],
        'recommendations': []
    }
    
    global_metrics = stats['global_metrics']
    recent_metrics = stats['recent_metrics']
    
    # Analyze response time
    if global_metrics['avg_time_per_request'] > 10:
        analysis['bottlenecks'].append('High response time')
        analysis['recommendations'].append('Consider using a smaller model or optimizing prompts')
        analysis['performance_grade'] = 'C'
    
    # Analyze memory usage
    if global_metrics['current_memory_mb'] > 4000:
        analysis['bottlenecks'].append('High memory usage')
        analysis['recommendations'].append('Monitor memory usage and consider model optimization')
        if analysis['performance_grade'] > 'B':
            analysis['performance_grade'] = 'B'
    
    # Analyze error rate
    if global_metrics['error_rate'] > 0.1:
        analysis['bottlenecks'].append('High error rate')
        analysis['recommendations'].append('Investigate and fix recurring errors')
        analysis['performance_grade'] = 'D'
    
    # Analyze throughput
    if global_metrics['avg_tokens_per_second'] < 10:
        analysis['bottlenecks'].append('Low token generation speed')
        analysis['recommendations'].append('Consider using GPU acceleration or model optimization')
        if analysis['performance_grade'] > 'C':
            analysis['performance_grade'] = 'C'
    
    return analysis

# Test functions
def test_monitoring():
    """Test monitoring functionality"""
    print("=== Monitoring Test ===")
    
    monitor = get_performance_monitor()
    logger = get_ai_logger()
    
    # Simulate some requests
    with timing_context("test_model") as ctx:
        time.sleep(0.1)  # Simulate processing time
        ctx.set_tokens(100)
    
    # Simulate error
    try:
        with timing_context("test_model") as ctx:
            raise ValueError("Test error")
    except ValueError:
        pass
    
    # Get statistics
    stats = monitor.get_current_stats()
    print("Performance Stats:")
    print(json.dumps(stats, indent=2))
    
    # Get health status
    health = get_health_status()
    print(f"\nHealth Status: {health['status']}")
    
    # Performance analysis
    analysis = analyze_performance()
    print(f"\nPerformance Analysis:")
    print(f"Grade: {analysis['performance_grade']}")
    print(f"Bottlenecks: {analysis['bottlenecks']}")
    print(f"Recommendations: {analysis['recommendations']}")
    
    print("=== Test completed ===")

if __name__ == "__main__":
    test_monitoring()
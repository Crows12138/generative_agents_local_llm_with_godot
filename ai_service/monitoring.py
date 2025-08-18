"""
Performance Monitoring and Logging
性能监控和日志记录模块，提供详细的性能指标和结构化日志
"""

import time
import logging
import psutil
import threading
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
from collections import deque, defaultdict
import json

@dataclass
class PerformanceMetrics:
    """性能指标"""
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
        """更新请求指标"""
        self.request_count += 1
        self.total_tokens += tokens
        self.total_time += duration
        self.avg_time_per_request = self.total_time / self.request_count
        self.avg_tokens_per_second = self.total_tokens / self.total_time if self.total_time > 0 else 0
        self.current_memory_mb = memory_mb
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
    
    def update_error(self):
        """更新错误指标"""
        self.error_count += 1
        self.error_rate = self.error_count / max(self.request_count, 1)

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.metrics = PerformanceMetrics()
        self.request_times = deque(maxlen=window_size)
        self.request_tokens = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.start_time = time.time()
        self.lock = threading.Lock()
        
        # 按模型分类的指标
        self.model_metrics: Dict[str, PerformanceMetrics] = defaultdict(PerformanceMetrics)
        
    def get_memory_usage(self) -> float:
        """获取当前内存使用量(MB)"""
        try:
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except:
            return 0.0
    
    def record_request(self, model_key: str, tokens: int, duration: float):
        """记录请求"""
        with self.lock:
            memory_mb = self.get_memory_usage()
            
            # 更新全局指标
            self.metrics.update_request(tokens, duration, memory_mb)
            
            # 更新模型特定指标
            self.model_metrics[model_key].update_request(tokens, duration, memory_mb)
            
            # 更新滑动窗口
            self.request_times.append(duration)
            self.request_tokens.append(tokens)
            self.memory_usage.append(memory_mb)
    
    def record_error(self, model_key: str):
        """记录错误"""
        with self.lock:
            self.metrics.update_error()
            self.model_metrics[model_key].update_error()
    
    def get_current_stats(self) -> Dict[str, Any]:
        """获取当前统计数据"""
        with self.lock:
            uptime = time.time() - self.start_time
            
            # 最近的性能数据
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
        """重置统计数据"""
        with self.lock:
            self.metrics = PerformanceMetrics()
            self.model_metrics.clear()
            self.request_times.clear()
            self.request_tokens.clear()
            self.memory_usage.clear()
            self.start_time = time.time()

class TimingContext:
    """计时上下文管理器"""
    
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
            # 成功完成
            self.monitor.record_request(self.model_key, self.tokens, duration)
            if self.logger:
                self.logger.info(
                    f"Generation completed: model={self.model_key}, "
                    f"tokens={self.tokens}, duration={duration:.3f}s"
                )
        else:
            # 发生错误
            self.monitor.record_error(self.model_key)
            if self.logger:
                self.logger.error(
                    f"Generation failed: model={self.model_key}, "
                    f"duration={duration:.3f}s, error={exc_val}"
                )
    
    def set_tokens(self, tokens: int):
        """设置生成的token数量"""
        self.tokens = tokens

def monitor_performance(monitor: PerformanceMonitor, model_key: str = None):
    """性能监控装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # 尝试从参数中获取model_key
            actual_model_key = model_key
            if actual_model_key is None:
                # 从函数参数中查找model_key
                if 'model_key' in kwargs:
                    actual_model_key = kwargs['model_key'] or 'unknown'
                elif len(args) > 1:
                    actual_model_key = args[1] or 'unknown'
                else:
                    actual_model_key = 'unknown'
            
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                
                # 尝试计算token数量
                tokens = 0
                if isinstance(result, str):
                    tokens = len(result.split())  # 简单的token估算
                
                duration = time.time() - start_time
                monitor.record_request(actual_model_key, tokens, duration)
                
                return result
                
            except Exception as e:
                monitor.record_error(actual_model_key)
                raise
        
        return wrapper
    return decorator

# 日志配置
class AIServiceLogger:
    """AI服务日志管理器"""
    
    def __init__(self, name: str = "ai_service", level: str = "INFO"):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # 避免重复添加handler
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """设置日志处理器"""
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # 文件处理器
        try:
            file_handler = logging.FileHandler('ai_service.log', encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
        except:
            file_handler = None
        
        # 格式化器
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        if file_handler:
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def get_logger(self) -> logging.Logger:
        """获取日志器"""
        return self.logger

# 全局实例
_performance_monitor = PerformanceMonitor()
_ai_logger = AIServiceLogger()

def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    return _performance_monitor

def get_ai_logger() -> logging.Logger:
    """获取AI服务日志器"""
    return _ai_logger.get_logger()

@contextmanager
def timing_context(model_key: str):
    """计时上下文管理器"""
    monitor = get_performance_monitor()
    logger = get_ai_logger()
    
    with TimingContext(monitor, model_key, logger) as ctx:
        yield ctx

# API端点支持
def get_health_status() -> Dict[str, Any]:
    """获取健康状态"""
    monitor = get_performance_monitor()
    stats = monitor.get_current_stats()
    
    # 判断健康状态
    is_healthy = (
        stats['global_metrics']['error_rate'] < 0.5 and  # 错误率小于50%
        stats['global_metrics']['current_memory_mb'] < 8000  # 内存使用小于8GB
    )
    
    return {
        'status': 'healthy' if is_healthy else 'unhealthy',
        'timestamp': time.time(),
        'metrics': stats
    }

def get_metrics_json() -> str:
    """获取JSON格式的指标"""
    monitor = get_performance_monitor()
    stats = monitor.get_current_stats()
    return json.dumps(stats, indent=2)

# 性能分析工具
def analyze_performance() -> Dict[str, Any]:
    """分析性能数据"""
    monitor = get_performance_monitor()
    stats = monitor.get_current_stats()
    
    analysis = {
        'performance_grade': 'A',  # A, B, C, D, F
        'bottlenecks': [],
        'recommendations': []
    }
    
    global_metrics = stats['global_metrics']
    recent_metrics = stats['recent_metrics']
    
    # 分析响应时间
    if global_metrics['avg_time_per_request'] > 10:
        analysis['bottlenecks'].append('High response time')
        analysis['recommendations'].append('Consider using a smaller model or optimizing prompts')
        analysis['performance_grade'] = 'C'
    
    # 分析内存使用
    if global_metrics['current_memory_mb'] > 4000:
        analysis['bottlenecks'].append('High memory usage')
        analysis['recommendations'].append('Monitor memory usage and consider model optimization')
        if analysis['performance_grade'] > 'B':
            analysis['performance_grade'] = 'B'
    
    # 分析错误率
    if global_metrics['error_rate'] > 0.1:
        analysis['bottlenecks'].append('High error rate')
        analysis['recommendations'].append('Investigate and fix recurring errors')
        analysis['performance_grade'] = 'D'
    
    # 分析吞吐量
    if global_metrics['avg_tokens_per_second'] < 10:
        analysis['bottlenecks'].append('Low token generation speed')
        analysis['recommendations'].append('Consider using GPU acceleration or model optimization')
        if analysis['performance_grade'] > 'C':
            analysis['performance_grade'] = 'C'
    
    return analysis

# 测试函数
def test_monitoring():
    """测试监控功能"""
    print("=== Monitoring Test ===")
    
    monitor = get_performance_monitor()
    logger = get_ai_logger()
    
    # 模拟一些请求
    with timing_context("test_model") as ctx:
        time.sleep(0.1)  # 模拟处理时间
        ctx.set_tokens(100)
    
    # 模拟错误
    try:
        with timing_context("test_model") as ctx:
            raise ValueError("Test error")
    except ValueError:
        pass
    
    # 获取统计数据
    stats = monitor.get_current_stats()
    print("Performance Stats:")
    print(json.dumps(stats, indent=2))
    
    # 获取健康状态
    health = get_health_status()
    print(f"\nHealth Status: {health['status']}")
    
    # 性能分析
    analysis = analyze_performance()
    print(f"\nPerformance Analysis:")
    print(f"Grade: {analysis['performance_grade']}")
    print(f"Bottlenecks: {analysis['bottlenecks']}")
    print(f"Recommendations: {analysis['recommendations']}")
    
    print("=== Test completed ===")

if __name__ == "__main__":
    test_monitoring()
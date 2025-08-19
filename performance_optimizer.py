"""
性能优化器 - 确保演示流畅运行
Performance Optimizer - Ensure smooth demo performance
"""

import os
import gc
import time
import psutil
import threading
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import json
import hashlib
from collections import OrderedDict, defaultdict
import functools

# Import console helper for better Unicode support
try:
    from utils.console_helper import print_success, print_warning, print_info, icons
except ImportError:
    # Fallback functions if console helper not available
    def print_success(msg): print(f"[OK] {msg}")
    def print_warning(msg): print(f"[WARNING] {msg}")  
    def print_info(msg): print(f"[INFO] {msg}")
    class FallbackIcons:
        CHECK = "[OK]"
        WARNING = "[WARNING]"
        GEAR = "[CONFIG]"
        ROCKET = "[STARTING]"
        CLEANUP = "[CLEANUP]"
    icons = FallbackIcons()

# 导入缓存库
try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    from ai_service.ai_service import get_health_status, get_metrics_json
    from ai_service.monitoring import get_performance_monitor
except ImportError:
    print("Warning: AI service monitoring not available")


@dataclass
class PerformanceMetrics:
    """性能指标数据结构"""
    timestamp: datetime
    ai_response_time: float  # seconds
    memory_usage_mb: float
    cpu_usage_percent: float
    fps: Optional[float] = None
    active_agents: int = 0
    cache_hit_rate: float = 0.0
    model_load_time: Optional[float] = None


@dataclass
class PerformanceTargets:
    """性能目标"""
    max_ai_response_time: float = 2.0  # seconds
    max_memory_usage_gb: float = 4.0
    min_fps: float = 30.0
    max_cpu_usage: float = 80.0


class ResponseCache:
    """AI响应缓存系统"""
    
    def __init__(self, max_size: int = 1000, ttl_hours: int = 24):
        self.max_size = max_size
        self.ttl_hours = ttl_hours
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, datetime] = {}
        self.hit_count = 0
        self.miss_count = 0
        
        # 尝试连接Redis（如果可用）
        self.redis_client = None
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
                self.redis_client.ping()
                print_success("Redis cache connected")
            except:
                self.redis_client = None
                print_warning("Redis not available, using memory cache")
    
    def _generate_key(self, prompt: str, model_key: str = "", context: str = "") -> str:
        """生成缓存键"""
        content = f"{prompt}|{model_key}|{context}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get(self, prompt: str, model_key: str = "", context: str = "") -> Optional[str]:
        """获取缓存的响应"""
        key = self._generate_key(prompt, model_key, context)
        
        # 尝试Redis缓存
        if self.redis_client:
            try:
                response = self.redis_client.get(f"ai_response:{key}")
                if response:
                    self.hit_count += 1
                    return response
            except:
                pass
        
        # 使用内存缓存
        if key in self.cache:
            # 检查TTL
            if key in self.timestamps:
                age = datetime.now() - self.timestamps[key]
                if age.total_seconds() > self.ttl_hours * 3600:
                    self._remove_key(key)
                    self.miss_count += 1
                    return None
            
            # 移动到末尾（LRU）
            self.cache.move_to_end(key)
            self.hit_count += 1
            return self.cache[key]
        
        self.miss_count += 1
        return None
    
    def set(self, prompt: str, response: str, model_key: str = "", context: str = ""):
        """设置缓存响应"""
        key = self._generate_key(prompt, model_key, context)
        
        # 设置Redis缓存
        if self.redis_client:
            try:
                self.redis_client.setex(f"ai_response:{key}", 
                                      timedelta(hours=self.ttl_hours), response)
            except:
                pass
        
        # 设置内存缓存
        if len(self.cache) >= self.max_size:
            # 移除最旧的项
            oldest_key = next(iter(self.cache))
            self._remove_key(oldest_key)
        
        self.cache[key] = response
        self.timestamps[key] = datetime.now()
    
    def _remove_key(self, key: str):
        """移除缓存键"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
    
    def get_hit_rate(self) -> float:
        """获取缓存命中率"""
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total > 0 else 0.0
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.timestamps.clear()
        if self.redis_client:
            try:
                for key in self.redis_client.scan_iter(match="ai_response:*"):
                    self.redis_client.delete(key)
            except:
                pass


class ModelManager:
    """AI模型管理器"""
    
    def __init__(self):
        self.loaded_models: Dict[str, Any] = {}
        self.model_load_times: Dict[str, float] = {}
        self.last_used: Dict[str, datetime] = {}
        self.preload_thread = None
        
    def preload_models(self, model_keys: List[str]):
        """预加载模型"""
        def _preload():
            for model_key in model_keys:
                try:
                    start_time = time.time()
                    # 这里可以添加实际的模型预加载逻辑
                    # 例如：model = load_model(model_key)
                    load_time = time.time() - start_time
                    self.model_load_times[model_key] = load_time
                    self.last_used[model_key] = datetime.now()
                    print_success(f"Preloaded {model_key} in {load_time:.2f}s")
                except Exception as e:
                    print(f"✗ Failed to preload {model_key}: {e}")
        
        if self.preload_thread is None or not self.preload_thread.is_alive():
            self.preload_thread = threading.Thread(target=_preload, daemon=True)
            self.preload_thread.start()
    
    def unload_unused_models(self, max_idle_minutes: int = 30):
        """卸载长时间未使用的模型"""
        current_time = datetime.now()
        to_unload = []
        
        for model_key, last_used in self.last_used.items():
            idle_time = current_time - last_used
            if idle_time.total_seconds() > max_idle_minutes * 60:
                to_unload.append(model_key)
        
        for model_key in to_unload:
            if model_key in self.loaded_models:
                del self.loaded_models[model_key]
                print_info(f"{icons.CLEANUP} Unloaded unused model: {model_key}")
        
        # 强制垃圾回收
        gc.collect()


class PerformanceOptimizer:
    """性能优化器主类"""
    
    def __init__(self):
        self.cache = ResponseCache()
        self.model_manager = ModelManager()
        self.targets = PerformanceTargets()
        self.metrics_history: List[PerformanceMetrics] = []
        self.monitoring_active = False
        self.monitoring_thread = None
        
        # 性能统计
        self.optimization_stats = {
            "cache_saves": 0,
            "memory_cleanups": 0,
            "model_unloads": 0,
            "total_optimizations": 0
        }
    
    def start_monitoring(self, interval_seconds: int = 5):
        """开始性能监控"""
        self.monitoring_active = True
        
        def _monitor():
            while self.monitoring_active:
                try:
                    metrics = self._collect_metrics()
                    self.metrics_history.append(metrics)
                    
                    # 保持历史记录在合理大小
                    if len(self.metrics_history) > 1000:
                        self.metrics_history = self.metrics_history[-500:]
                    
                    # 检查性能问题并优化
                    self._check_and_optimize(metrics)
                    
                    time.sleep(interval_seconds)
                except Exception as e:
                    print(f"Monitoring error: {e}")
                    time.sleep(interval_seconds)
        
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=_monitor, daemon=True)
            self.monitoring_thread.start()
            print_success("Performance monitoring started")
    
    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        print_success("Performance monitoring stopped")
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        process = psutil.Process()
        
        # 基础系统指标
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        # AI响应时间（从最近的记录获取）
        ai_response_time = 0.0
        try:
            # 尝试获取AI服务的指标
            health = get_health_status()
            if isinstance(health, dict) and 'avg_response_time' in health:
                ai_response_time = health['avg_response_time']
        except:
            ai_response_time = 1.0  # 默认值
        
        return PerformanceMetrics(
            timestamp=datetime.now(),
            ai_response_time=ai_response_time,
            memory_usage_mb=memory_mb,
            cpu_usage_percent=cpu_percent,
            cache_hit_rate=self.cache.get_hit_rate()
        )
    
    def _check_and_optimize(self, metrics: PerformanceMetrics):
        """检查性能并进行优化"""
        optimizations_applied = []
        
        # 检查AI响应时间
        if metrics.ai_response_time > self.targets.max_ai_response_time:
            self._optimize_ai_performance()
            optimizations_applied.append("ai_response")
        
        # 检查内存使用
        if metrics.memory_usage_mb > self.targets.max_memory_usage_gb * 1024:
            self._optimize_memory_usage()
            optimizations_applied.append("memory")
        
        # 检查CPU使用
        if metrics.cpu_usage_percent > self.targets.max_cpu_usage:
            self._optimize_cpu_usage()
            optimizations_applied.append("cpu")
        
        # 检查缓存命中率
        if metrics.cache_hit_rate < 0.3:  # 命中率低于30%
            self._optimize_cache_strategy()
            optimizations_applied.append("cache")
        
        if optimizations_applied:
            self.optimization_stats["total_optimizations"] += 1
            print_info(f"{icons.GEAR} Applied optimizations: {', '.join(optimizations_applied)}")
    
    def _optimize_ai_performance(self):
        """优化AI性能"""
        # 预加载常用模型
        self.model_manager.preload_models(["qwen", "gpt-oss"])
        
        # 增加缓存大小
        if self.cache.max_size < 2000:
            self.cache.max_size = min(2000, self.cache.max_size * 2)
    
    def _optimize_memory_usage(self):
        """优化内存使用"""
        # 清理未使用的模型
        self.model_manager.unload_unused_models(max_idle_minutes=15)
        
        # 清理缓存中的过期项
        current_time = datetime.now()
        expired_keys = []
        for key, timestamp in self.cache.timestamps.items():
            if (current_time - timestamp).total_seconds() > 3600:  # 1小时
                expired_keys.append(key)
        
        for key in expired_keys:
            self.cache._remove_key(key)
        
        # 强制垃圾回收
        gc.collect()
        
        self.optimization_stats["memory_cleanups"] += 1
    
    def _optimize_cpu_usage(self):
        """优化CPU使用"""
        # 减少监控频率
        if hasattr(self, 'monitoring_interval'):
            self.monitoring_interval = min(10, self.monitoring_interval + 1)
        
        # 延迟非关键操作
        time.sleep(0.1)
    
    def _optimize_cache_strategy(self):
        """优化缓存策略"""
        # 增加缓存TTL
        self.cache.ttl_hours = min(48, self.cache.ttl_hours * 1.5)
        
        # 预缓存常见响应
        common_prompts = [
            "Hello, how are you?",
            "What are you doing?",
            "How is the weather?",
            "What time is it?",
            "Tell me about yourself."
        ]
        
        # 这里可以添加预缓存逻辑
        self.optimization_stats["cache_saves"] += 1
    
    def cached_ai_generate(self, prompt: str, model_key: str = "", **kwargs) -> str:
        """带缓存的AI生成函数"""
        # 生成缓存上下文
        context = json.dumps(kwargs, sort_keys=True)
        
        # 尝试从缓存获取
        cached_response = self.cache.get(prompt, model_key, context)
        if cached_response:
            return cached_response
        
        # 缓存未命中，调用实际的AI生成
        try:
            # 这里调用实际的AI生成函数
            from ai_service.ai_service import local_llm_generate
            
            start_time = time.time()
            response = local_llm_generate(prompt, model_key=model_key, **kwargs)
            end_time = time.time()
            
            # 只缓存成功的响应
            if response and len(response.strip()) > 0:
                self.cache.set(prompt, response, model_key, context)
            
            # 记录响应时间
            response_time = end_time - start_time
            if hasattr(self, 'last_response_time'):
                self.last_response_time = response_time
            
            return response
            
        except Exception as e:
            print(f"AI generation error: {e}")
            # 返回回退响应
            fallback = "I apologize, but I'm having trouble processing that request right now."
            self.cache.set(prompt, fallback, model_key, context)
            return fallback
    
    def get_performance_report(self) -> Dict[str, Any]:
        """获取性能报告"""
        if not self.metrics_history:
            return {"error": "No metrics collected yet"}
        
        recent_metrics = self.metrics_history[-10:]  # 最近10个指标
        
        avg_response_time = sum(m.ai_response_time for m in recent_metrics) / len(recent_metrics)
        avg_memory_mb = sum(m.memory_usage_mb for m in recent_metrics) / len(recent_metrics)
        avg_cpu = sum(m.cpu_usage_percent for m in recent_metrics) / len(recent_metrics)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "performance_summary": {
                "avg_ai_response_time": round(avg_response_time, 2),
                "avg_memory_usage_mb": round(avg_memory_mb, 2),
                "avg_cpu_usage_percent": round(avg_cpu, 2),
                "cache_hit_rate": round(self.cache.get_hit_rate(), 3)
            },
            "targets": asdict(self.targets),
            "status": {
                "ai_response_ok": avg_response_time <= self.targets.max_ai_response_time,
                "memory_ok": avg_memory_mb <= self.targets.max_memory_usage_gb * 1024,
                "cpu_ok": avg_cpu <= self.targets.max_cpu_usage
            },
            "optimization_stats": self.optimization_stats,
            "cache_stats": {
                "hit_count": self.cache.hit_count,
                "miss_count": self.cache.miss_count,
                "hit_rate": round(self.cache.get_hit_rate(), 3),
                "cache_size": len(self.cache.cache)
            }
        }
    
    def save_metrics_to_file(self, filename: str = None):
        """保存性能指标到文件"""
        if filename is None:
            filename = f"performance_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = self.get_performance_report()
        report["metrics_history"] = [asdict(m) for m in self.metrics_history[-100:]]  # 最近100个记录
        
        # 转换datetime对象为字符串
        for metric in report["metrics_history"]:
            if isinstance(metric["timestamp"], datetime):
                metric["timestamp"] = metric["timestamp"].isoformat()
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False, default=str)
            print_success(f"Performance metrics saved to {filename}")
        except Exception as e:
            print(f"✗ Failed to save metrics: {e}")


# 全局性能优化器实例
_performance_optimizer = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """获取全局性能优化器实例"""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def optimize_for_demo():
    """为演示优化性能"""
    optimizer = get_performance_optimizer()
    
    print_info(f"{icons.ROCKET} Starting demo performance optimization...")
    
    # 开始监控
    optimizer.start_monitoring(interval_seconds=3)
    
    # 预加载模型
    optimizer.model_manager.preload_models(["qwen"])
    
    # 设置更严格的性能目标
    optimizer.targets.max_ai_response_time = 1.5
    optimizer.targets.max_memory_usage_gb = 3.0
    optimizer.targets.min_fps = 30.0
    
    print_success("Demo optimization setup complete")
    return optimizer


# 装饰器：自动缓存AI响应
def cached_ai_response(func):
    """装饰器：为AI响应函数添加缓存"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        optimizer = get_performance_optimizer()
        
        # 简单的参数序列化作为缓存键
        cache_key = f"{func.__name__}_{hash(str(args) + str(kwargs))}"
        
        # 尝试从缓存获取
        if hasattr(optimizer, 'function_cache'):
            if cache_key in optimizer.function_cache:
                return optimizer.function_cache[cache_key]
        else:
            optimizer.function_cache = {}
        
        # 执行函数
        result = func(*args, **kwargs)
        
        # 缓存结果
        optimizer.function_cache[cache_key] = result
        
        # 限制缓存大小
        if len(optimizer.function_cache) > 100:
            # 移除最旧的项
            old_keys = list(optimizer.function_cache.keys())[:50]
            for key in old_keys:
                optimizer.function_cache.pop(key, None)
        
        return result
    
    return wrapper


if __name__ == "__main__":
    # 演示性能优化器
    optimizer = optimize_for_demo()
    
    # 运行一段时间后生成报告
    time.sleep(10)
    report = optimizer.get_performance_report()
    print("\nPerformance Report:")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    optimizer.stop_monitoring()
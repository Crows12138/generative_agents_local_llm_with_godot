"""
Memory Optimizer - Smart memory management
"""

import gc
import sys
import psutil
import threading
import time
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import weakref


@dataclass
class MemoryStats:
    """Memory statistics"""
    total_mb: float
    used_mb: float
    available_mb: float
    percent_used: float
    python_objects: int
    gc_collections: Dict[int, int]


class MemoryPool:
    """Object pool manager"""
    
    def __init__(self):
        self.pools: Dict[str, List[Any]] = {}
        self.max_pool_sizes: Dict[str, int] = {}
        self.created_count: Dict[str, int] = {}
        self.reused_count: Dict[str, int] = {}
    
    def register_pool(self, object_type: str, max_size: int = 100):
        """Register object pool"""
        self.pools[object_type] = []
        self.max_pool_sizes[object_type] = max_size
        self.created_count[object_type] = 0
        self.reused_count[object_type] = 0
    
    def get_object(self, object_type: str, factory_func=None):
        """Get object from pool"""
        if object_type not in self.pools:
            self.register_pool(object_type)
        
        pool = self.pools[object_type]
        
        if pool:
            # Reuse object from pool
            obj = pool.pop()
            self.reused_count[object_type] += 1
            return obj
        else:
            # Create new object
            if factory_func:
                obj = factory_func()
                self.created_count[object_type] += 1
                return obj
            return None
    
    def return_object(self, object_type: str, obj):
        """Return object to pool"""
        if object_type not in self.pools:
            return
        
        pool = self.pools[object_type]
        max_size = self.max_pool_sizes[object_type]
        
        if len(pool) < max_size:
            # Clean up object state (if cleanup method exists)
            if hasattr(obj, 'cleanup'):
                obj.cleanup()
            pool.append(obj)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        stats = {}
        for obj_type in self.pools:
            total_created = self.created_count.get(obj_type, 0)
            total_reused = self.reused_count.get(obj_type, 0)
            reuse_rate = total_reused / (total_created + total_reused) if (total_created + total_reused) > 0 else 0
            
            stats[obj_type] = {
                "pool_size": len(self.pools[obj_type]),
                "max_size": self.max_pool_sizes[obj_type],
                "created": total_created,
                "reused": total_reused,
                "reuse_rate": round(reuse_rate, 3)
            }
        return stats


class WeakReferenceManager:
    """弱引用管理器"""
    
    def __init__(self):
        self.references: Dict[str, weakref.WeakSet] = {}
        self.cleanup_callbacks: Dict[str, List] = {}
    
    def register_object(self, category: str, obj):
        """注册对象弱引用"""
        if category not in self.references:
            self.references[category] = weakref.WeakSet()
            self.cleanup_callbacks[category] = []
        
        self.references[category].add(obj)
    
    def add_cleanup_callback(self, category: str, callback):
        """添加清理回调"""
        if category not in self.cleanup_callbacks:
            self.cleanup_callbacks[category] = []
        self.cleanup_callbacks[category].append(callback)
    
    def cleanup_category(self, category: str):
        """清理指定类别的对象"""
        if category in self.references:
            # 执行清理回调
            for callback in self.cleanup_callbacks.get(category, []):
                try:
                    callback()
                except Exception as e:
                    print(f"Cleanup callback error: {e}")
            
            # 清空弱引用集合
            self.references[category].clear()
    
    def get_alive_count(self, category: str) -> int:
        """获取存活对象数量"""
        if category in self.references:
            return len(self.references[category])
        return 0


class MemoryOptimizer:
    """内存优化器主类"""
    
    def __init__(self):
        self.memory_pool = MemoryPool()
        self.weak_ref_manager = WeakReferenceManager()
        self.monitoring_active = False
        self.monitoring_thread = None
        self.memory_history: List[MemoryStats] = []
        
        # 内存阈值
        self.warning_threshold_percent = 75.0
        self.critical_threshold_percent = 85.0
        self.cleanup_threshold_percent = 80.0
        
        # 优化统计
        self.cleanup_count = 0
        self.gc_forced_count = 0
        self.pool_hits = 0
        
        # 注册常用对象池
        self._register_common_pools()
    
    def _register_common_pools(self):
        """注册常用对象池"""
        self.memory_pool.register_pool("agent_response", max_size=50)
        self.memory_pool.register_pool("memory_object", max_size=100)
        self.memory_pool.register_pool("dialogue_context", max_size=30)
        self.memory_pool.register_pool("location_object", max_size=20)
    
    def start_monitoring(self, interval_seconds: int = 10):
        """开始内存监控"""
        self.monitoring_active = True
        
        def _monitor():
            while self.monitoring_active:
                try:
                    stats = self._collect_memory_stats()
                    self.memory_history.append(stats)
                    
                    # 保持历史记录在合理大小
                    if len(self.memory_history) > 500:
                        self.memory_history = self.memory_history[-250:]
                    
                    # 检查内存压力并优化
                    self._check_memory_pressure(stats)
                    
                    time.sleep(interval_seconds)
                except Exception as e:
                    print(f"Memory monitoring error: {e}")
                    time.sleep(interval_seconds)
        
        if self.monitoring_thread is None or not self.monitoring_thread.is_alive():
            self.monitoring_thread = threading.Thread(target=_monitor, daemon=True)
            self.monitoring_thread.start()
            print("✓ Memory monitoring started")
    
    def stop_monitoring(self):
        """停止内存监控"""
        self.monitoring_active = False
        print("✓ Memory monitoring stopped")
    
    def _collect_memory_stats(self) -> MemoryStats:
        """收集内存统计信息"""
        # 系统内存信息
        memory = psutil.virtual_memory()
        
        # Python对象计数
        object_count = len(gc.get_objects())
        
        # GC统计
        gc_stats = {}
        for i in range(3):
            gc_stats[i] = gc.get_count()[i]
        
        return MemoryStats(
            total_mb=memory.total / 1024 / 1024,
            used_mb=memory.used / 1024 / 1024,
            available_mb=memory.available / 1024 / 1024,
            percent_used=memory.percent,
            python_objects=object_count,
            gc_collections=gc_stats
        )
    
    def _check_memory_pressure(self, stats: MemoryStats):
        """检查内存压力并采取优化措施"""
        if stats.percent_used >= self.critical_threshold_percent:
            print(f"🚨 Critical memory usage: {stats.percent_used:.1f}%")
            self.force_cleanup()
        elif stats.percent_used >= self.warning_threshold_percent:
            print(f"⚠️ High memory usage: {stats.percent_used:.1f}%")
            self.gentle_cleanup()
        elif stats.percent_used >= self.cleanup_threshold_percent:
            self.optimize_memory()
    
    def gentle_cleanup(self):
        """温和的内存清理"""
        # 清理代理对象缓存
        self.weak_ref_manager.cleanup_category("agent_cache")
        
        # 强制垃圾回收
        collected = gc.collect()
        
        print(f"🧹 Gentle cleanup: {collected} objects collected")
        self.cleanup_count += 1
    
    def force_cleanup(self):
        """强制内存清理"""
        # 清理所有缓存
        for category in list(self.weak_ref_manager.references.keys()):
            self.weak_ref_manager.cleanup_category(category)
        
        # 清空对象池
        for pool in self.memory_pool.pools.values():
            pool.clear()
        
        # 强制垃圾回收多次
        for _ in range(3):
            gc.collect()
        
        self.gc_forced_count += 1
        print("🚨 Force cleanup completed")
    
    def optimize_memory(self):
        """优化内存使用"""
        # 清理Python对象引用循环
        gc.collect()
        
        # 清理弱引用中的死对象
        for category in self.weak_ref_manager.references:
            # WeakSet会自动清理死对象，这里只是触发清理
            len(self.weak_ref_manager.references[category])
    
    def get_object_from_pool(self, object_type: str, factory_func=None):
        """从对象池获取对象"""
        obj = self.memory_pool.get_object(object_type, factory_func)
        if obj is not None:
            self.pool_hits += 1
        return obj
    
    def return_object_to_pool(self, object_type: str, obj):
        """归还对象到池"""
        self.memory_pool.return_object(object_type, obj)
    
    def register_agent_object(self, obj):
        """注册智能体对象"""
        self.weak_ref_manager.register_object("agents", obj)
    
    def register_memory_object(self, obj):
        """注册记忆对象"""
        self.weak_ref_manager.register_object("memories", obj)
    
    def get_memory_report(self) -> Dict[str, Any]:
        """获取内存报告"""
        current_stats = self._collect_memory_stats()
        
        # 计算内存趋势
        if len(self.memory_history) > 1:
            previous_stats = self.memory_history[-2]
            memory_trend = current_stats.used_mb - previous_stats.used_mb
        else:
            memory_trend = 0.0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "current_memory": {
                "total_mb": round(current_stats.total_mb, 2),
                "used_mb": round(current_stats.used_mb, 2),
                "available_mb": round(current_stats.available_mb, 2),
                "percent_used": round(current_stats.percent_used, 2),
                "python_objects": current_stats.python_objects
            },
            "memory_trend_mb": round(memory_trend, 2),
            "optimization_stats": {
                "cleanup_count": self.cleanup_count,
                "gc_forced_count": self.gc_forced_count,
                "pool_hits": self.pool_hits
            },
            "object_pools": self.memory_pool.get_stats(),
            "weak_references": {
                category: self.weak_ref_manager.get_alive_count(category)
                for category in self.weak_ref_manager.references
            },
            "thresholds": {
                "warning_percent": self.warning_threshold_percent,
                "critical_percent": self.critical_threshold_percent,
                "cleanup_percent": self.cleanup_threshold_percent
            },
            "status": {
                "memory_ok": current_stats.percent_used < self.warning_threshold_percent,
                "trend_ok": memory_trend < 50.0  # 增长小于50MB
            }
        }


# 全局内存优化器实例
_memory_optimizer = None

def get_memory_optimizer() -> MemoryOptimizer:
    """获取全局内存优化器实例"""
    global _memory_optimizer
    if _memory_optimizer is None:
        _memory_optimizer = MemoryOptimizer()
    return _memory_optimizer


def optimize_agent_memory(agent_class):
    """装饰器：为智能体类添加内存优化"""
    class OptimizedAgent(agent_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # 注册到内存优化器
            get_memory_optimizer().register_agent_object(self)
        
        def __del__(self):
            # 清理资源
            if hasattr(self, 'memory'):
                if hasattr(self.memory, 'clear'):
                    self.memory.clear()
    
    return OptimizedAgent


def memory_efficient(func):
    """装饰器：使函数更高效地使用内存"""
    import functools
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        optimizer = get_memory_optimizer()
        
        # 执行前检查内存
        stats_before = optimizer._collect_memory_stats()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # 执行后检查内存，如果增长过多则清理
            stats_after = optimizer._collect_memory_stats()
            memory_growth = stats_after.used_mb - stats_before.used_mb
            
            if memory_growth > 100:  # 增长超过100MB
                optimizer.gentle_cleanup()
    
    return wrapper


if __name__ == "__main__":
    # 测试内存优化器
    optimizer = MemoryOptimizer()
    optimizer.start_monitoring(interval_seconds=2)
    
    # 模拟内存使用
    data = []
    for i in range(1000):
        data.append([0] * 1000)
        time.sleep(0.01)
    
    # 获取报告
    report = optimizer.get_memory_report()
    print("Memory Report:")
    import json
    print(json.dumps(report, indent=2, ensure_ascii=False))
    
    # 清理
    del data
    optimizer.force_cleanup()
    
    time.sleep(2)
    optimizer.stop_monitoring()
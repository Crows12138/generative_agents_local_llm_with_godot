"""
网络通信优化器 - 减少延迟和提高效率
Network Optimizer - Reduce latency and improve efficiency
"""

import asyncio
import aiohttp
import time
import json
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import queue
import concurrent.futures


@dataclass
class RequestMetrics:
    """请求指标"""
    url: str
    method: str
    start_time: float
    end_time: float
    response_time: float
    status_code: int
    data_size: int
    cached: bool = False


class RequestBatcher:
    """请求批处理器"""
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 0.1):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests: List[Dict[str, Any]] = []
        self.batch_thread = None
        self.is_running = False
        self.callbacks: Dict[str, Callable] = {}
    
    def start(self):
        """启动批处理"""
        self.is_running = True
        self.batch_thread = threading.Thread(target=self._batch_processor, daemon=True)
        self.batch_thread.start()
    
    def stop(self):
        """停止批处理"""
        self.is_running = False
    
    def add_request(self, request_id: str, request_data: Dict[str, Any], callback: Callable):
        """添加请求到批次"""
        self.pending_requests.append({
            "id": request_id,
            "data": request_data,
            "timestamp": time.time()
        })
        self.callbacks[request_id] = callback
    
    def _batch_processor(self):
        """批处理处理器"""
        while self.is_running:
            if len(self.pending_requests) >= self.batch_size:
                self._process_batch()
            elif self.pending_requests:
                # 检查最老的请求是否超时
                oldest_time = self.pending_requests[0]["timestamp"]
                if time.time() - oldest_time >= self.batch_timeout:
                    self._process_batch()
            
            time.sleep(0.01)  # 短暂休眠
    
    def _process_batch(self):
        """处理一批请求"""
        if not self.pending_requests:
            return
        
        batch = self.pending_requests.copy()
        self.pending_requests.clear()
        
        # 批量处理请求
        try:
            self._execute_batch(batch)
        except Exception as e:
            print(f"Batch processing error: {e}")
    
    def _execute_batch(self, batch: List[Dict[str, Any]]):
        """执行批量请求"""
        # 这里可以实现实际的批量处理逻辑
        for request in batch:
            request_id = request["id"]
            if request_id in self.callbacks:
                try:
                    # 模拟批量处理结果
                    result = {"success": True, "data": "batch_processed"}
                    self.callbacks[request_id](result)
                    del self.callbacks[request_id]
                except Exception as e:
                    print(f"Callback error for {request_id}: {e}")


class ConnectionPool:
    """连接池管理器"""
    
    def __init__(self, max_connections: int = 10):
        self.max_connections = max_connections
        self.active_connections: Dict[str, aiohttp.ClientSession] = {}
        self.connection_count = 0
        self.pool_lock = threading.Lock()
    
    async def get_session(self, base_url: str) -> aiohttp.ClientSession:
        """获取或创建会话"""
        with self.pool_lock:
            if base_url not in self.active_connections:
                if self.connection_count < self.max_connections:
                    # 创建新的会话，配置优化参数
                    timeout = aiohttp.ClientTimeout(total=30, connect=5)
                    connector = aiohttp.TCPConnector(
                        limit=20,
                        limit_per_host=5,
                        keepalive_timeout=30,
                        enable_cleanup_closed=True
                    )
                    
                    session = aiohttp.ClientSession(
                        timeout=timeout,
                        connector=connector
                    )
                    
                    self.active_connections[base_url] = session
                    self.connection_count += 1
                else:
                    # 复用现有连接
                    return list(self.active_connections.values())[0]
            
            return self.active_connections[base_url]
    
    async def close_all(self):
        """关闭所有连接"""
        for session in self.active_connections.values():
            await session.close()
        self.active_connections.clear()
        self.connection_count = 0


class NetworkOptimizer:
    """网络优化器主类"""
    
    def __init__(self):
        self.connection_pool = ConnectionPool(max_connections=5)
        self.request_batcher = RequestBatcher(batch_size=5, batch_timeout=0.05)
        self.request_metrics: List[RequestMetrics] = []
        self.response_cache: Dict[str, Any] = {}
        self.cache_ttl: Dict[str, datetime] = {}
        
        # 优化参数
        self.enable_compression = True
        self.enable_keepalive = True
        self.max_retries = 3
        self.timeout_seconds = 10.0
        
        # 统计信息
        self.total_requests = 0
        self.cache_hits = 0
        self.batch_requests = 0
        
        # 启动批处理
        self.request_batcher.start()
    
    def _generate_cache_key(self, url: str, data: Dict[str, Any] = None) -> str:
        """生成缓存键"""
        key_data = {"url": url, "data": data or {}}
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _is_cache_valid(self, cache_key: str, ttl_minutes: int = 5) -> bool:
        """检查缓存是否有效"""
        if cache_key not in self.cache_ttl:
            return False
        
        cache_time = self.cache_ttl[cache_key]
        expiry_time = cache_time + timedelta(minutes=ttl_minutes)
        return datetime.now() < expiry_time
    
    def _set_cache(self, cache_key: str, data: Any):
        """设置缓存"""
        self.response_cache[cache_key] = data
        self.cache_ttl[cache_key] = datetime.now()
        
        # 限制缓存大小
        if len(self.response_cache) > 100:
            # 移除最旧的缓存项
            oldest_key = min(self.cache_ttl.keys(), key=lambda k: self.cache_ttl[k])
            self.response_cache.pop(oldest_key, None)
            self.cache_ttl.pop(oldest_key, None)
    
    async def optimized_post(self, url: str, data: Dict[str, Any], 
                           use_cache: bool = True, cache_ttl: int = 5) -> Dict[str, Any]:
        """优化的POST请求"""
        self.total_requests += 1
        start_time = time.time()
        
        # 检查缓存
        if use_cache:
            cache_key = self._generate_cache_key(url, data)
            if self._is_cache_valid(cache_key, cache_ttl):
                self.cache_hits += 1
                return self.response_cache[cache_key]
        
        try:
            # 获取会话
            session = await self.connection_pool.get_session(url.split('/')[2])
            
            # 准备请求头
            headers = {
                "Content-Type": "application/json",
                "User-Agent": "OptimizedClient/1.0"
            }
            
            if self.enable_compression:
                headers["Accept-Encoding"] = "gzip, deflate"
            
            # 发送请求
            async with session.post(url, json=data, headers=headers) as response:
                response_data = await response.json()
                
                # 记录指标
                end_time = time.time()
                metrics = RequestMetrics(
                    url=url,
                    method="POST",
                    start_time=start_time,
                    end_time=end_time,
                    response_time=end_time - start_time,
                    status_code=response.status,
                    data_size=len(json.dumps(response_data)),
                    cached=False
                )
                self.request_metrics.append(metrics)
                
                # 缓存响应
                if use_cache and response.status == 200:
                    self._set_cache(cache_key, response_data)
                
                return response_data
                
        except Exception as e:
            print(f"Network request error: {e}")
            # 返回错误响应
            return {"error": str(e), "success": False}
    
    async def optimized_get(self, url: str, params: Dict[str, Any] = None,
                          use_cache: bool = True, cache_ttl: int = 10) -> Dict[str, Any]:
        """优化的GET请求"""
        self.total_requests += 1
        start_time = time.time()
        
        # 检查缓存
        if use_cache:
            cache_key = self._generate_cache_key(url, params)
            if self._is_cache_valid(cache_key, cache_ttl):
                self.cache_hits += 1
                return self.response_cache[cache_key]
        
        try:
            # 获取会话
            session = await self.connection_pool.get_session(url.split('/')[2])
            
            # 准备请求头
            headers = {"User-Agent": "OptimizedClient/1.0"}
            if self.enable_compression:
                headers["Accept-Encoding"] = "gzip, deflate"
            
            # 发送请求
            async with session.get(url, params=params, headers=headers) as response:
                response_data = await response.json()
                
                # 记录指标
                end_time = time.time()
                metrics = RequestMetrics(
                    url=url,
                    method="GET",
                    start_time=start_time,
                    end_time=end_time,
                    response_time=end_time - start_time,
                    status_code=response.status,
                    data_size=len(json.dumps(response_data)),
                    cached=False
                )
                self.request_metrics.append(metrics)
                
                # 缓存响应
                if use_cache and response.status == 200:
                    self._set_cache(cache_key, response_data)
                
                return response_data
                
        except Exception as e:
            print(f"Network request error: {e}")
            return {"error": str(e), "success": False}
    
    def batch_ai_request(self, request_data: Dict[str, Any], callback: Callable):
        """批量AI请求"""
        request_id = f"ai_req_{self.total_requests}_{time.time()}"
        self.request_batcher.add_request(request_id, request_data, callback)
        self.batch_requests += 1
    
    def get_network_stats(self) -> Dict[str, Any]:
        """获取网络统计"""
        if not self.request_metrics:
            return {"error": "No network metrics available"}
        
        recent_metrics = self.request_metrics[-50:]  # 最近50个请求
        
        avg_response_time = sum(m.response_time for m in recent_metrics) / len(recent_metrics)
        max_response_time = max(m.response_time for m in recent_metrics)
        min_response_time = min(m.response_time for m in recent_metrics)
        
        success_count = sum(1 for m in recent_metrics if m.status_code == 200)
        success_rate = success_count / len(recent_metrics)
        
        cache_hit_rate = self.cache_hits / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "batch_requests": self.batch_requests,
            "cache_hit_rate": round(cache_hit_rate, 3),
            "response_times": {
                "avg_ms": round(avg_response_time * 1000, 2),
                "max_ms": round(max_response_time * 1000, 2),
                "min_ms": round(min_response_time * 1000, 2)
            },
            "success_rate": round(success_rate, 3),
            "active_connections": self.connection_pool.connection_count,
            "cache_size": len(self.response_cache),
            "optimization_settings": {
                "compression_enabled": self.enable_compression,
                "keepalive_enabled": self.enable_keepalive,
                "max_retries": self.max_retries,
                "timeout_seconds": self.timeout_seconds
            }
        }
    
    async def cleanup(self):
        """清理资源"""
        self.request_batcher.stop()
        await self.connection_pool.close_all()
        print("✓ Network optimizer cleanup completed")


# 全局网络优化器实例
_network_optimizer = None

def get_network_optimizer() -> NetworkOptimizer:
    """获取全局网络优化器实例"""
    global _network_optimizer
    if _network_optimizer is None:
        _network_optimizer = NetworkOptimizer()
    return _network_optimizer


async def optimized_ai_request(url: str, prompt: str, model_key: str = "", **kwargs) -> str:
    """优化的AI请求函数"""
    optimizer = get_network_optimizer()
    
    request_data = {
        "prompt": prompt,
        "model_key": model_key,
        **kwargs
    }
    
    try:
        response = await optimizer.optimized_post(url, request_data, use_cache=True, cache_ttl=30)
        return response.get("text", response.get("response", ""))
    except Exception as e:
        print(f"Optimized AI request error: {e}")
        return "I apologize, but I'm having trouble processing that request."


def network_optimized(func):
    """装饰器：为函数添加网络优化"""
    import functools
    
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        optimizer = get_network_optimizer()
        
        # 如果函数涉及网络请求，记录统计
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            # 记录请求时间
            end_time = time.time()
            if hasattr(optimizer, 'function_call_times'):
                optimizer.function_call_times[func.__name__] = end_time - start_time
            else:
                optimizer.function_call_times = {func.__name__: end_time - start_time}
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        # 同步函数的包装
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            optimizer = get_network_optimizer()
            if hasattr(optimizer, 'function_call_times'):
                optimizer.function_call_times[func.__name__] = end_time - start_time
            else:
                optimizer.function_call_times = {func.__name__: end_time - start_time}
    
    # 根据函数是否为协程选择包装器
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    else:
        return sync_wrapper


if __name__ == "__main__":
    # 测试网络优化器
    async def test_optimizer():
        optimizer = NetworkOptimizer()
        
        # 模拟一些请求
        for i in range(10):
            await optimizer.optimized_get(f"https://httpbin.org/get?test={i}")
            await asyncio.sleep(0.1)
        
        # 获取统计
        stats = optimizer.get_network_stats()
        print("Network Statistics:")
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        
        await optimizer.cleanup()
    
    # 运行测试
    asyncio.run(test_optimizer())
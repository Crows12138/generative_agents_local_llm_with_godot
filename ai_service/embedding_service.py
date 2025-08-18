"""
Enhanced Embedding Service using Sentence Transformers
替换原有的简单hash嵌入，使用本地sentence-transformers模型
"""

import os
import time
import asyncio
from pathlib import Path
from typing import List, Optional, Union
import numpy as np
from datetime import datetime, timedelta

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("[WARNING] sentence-transformers not available, falling back to hash embedding")

# 项目根目录和模型路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / "all-MiniLM-L6-v2"

# 导入配置管理
try:
    from .config_enhanced import get_config
except ImportError:
    from config_enhanced import get_config

class EmbeddingService:
    """统一的嵌入服务，支持sentence-transformers和fallback，基于配置的高级功能"""
    
    def __init__(self, config_override: Optional[dict] = None):
        # 获取配置
        self.config_manager = get_config()
        self.config = self.config_manager.config.embedding
        
        # 应用配置覆盖
        if config_override:
            for key, value in config_override.items():
                if hasattr(self.config, key):
                    setattr(self.config, key, value)
        
        # 初始化属性
        self.model_path = self.config.model_path
        self.model = None
        self.model_loaded = False
        self.embedding_dim = self.config.embedding_dim
        self.initialization_attempted = False
        self.last_error = None
        
        # 缓存系统（支持TTL）
        self._embedding_cache = {} if self.config.enable_cache else None
        self._cache_timestamps = {} if self.config.enable_cache else None
        self.cache_max_size = self.config.cache_max_size
        self.cache_ttl = timedelta(seconds=self.config.cache_ttl)
        
        # 并发控制
        self._request_semaphore = asyncio.Semaphore(self.config.max_concurrent_requests)
        self._stats = {
            'requests_total': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_processing_time': 0.0,
            'last_cleanup': datetime.now()
        }
        
    def initialize(self) -> bool:
        """初始化嵌入模型，支持重试和缓存检查"""
        if self.initialization_attempted and self.model_loaded:
            return True
            
        self.initialization_attempted = True
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("[EmbeddingService] sentence-transformers not available, using hash fallback")
            self.last_error = "sentence-transformers not available"
            return True
            
        try:
            if not Path(self.model_path).exists():
                error_msg = f"Model path not found: {self.model_path}"
                print(f"[EmbeddingService] {error_msg}")
                self.last_error = error_msg
                return False
                
            print(f"[EmbeddingService] Loading embedding model from: {self.model_path}")
            start_time = time.time()
            
            # 设置更优的加载参数
            self.model = SentenceTransformer(
                self.model_path,
                device=self.config.device,  # 使用配置的设备
                cache_folder=None  # 使用默认缓存
            )
            
            load_time = time.time() - start_time
            print(f"[EmbeddingService] Model loaded successfully in {load_time:.2f}s")
            self.model_loaded = True
            
            # 获取实际嵌入维度并预热模型
            if self.config.preload_warmup:
                test_embedding = self.model.encode("test", show_progress_bar=False)
                self.embedding_dim = len(test_embedding)
                print(f"[EmbeddingService] Model warmed up, embedding dimension: {self.embedding_dim}")
                
                # 预加载常用文本
                if self.config.preload_texts:
                    self.preload_common_texts(self.config.preload_texts)
            else:
                print(f"[EmbeddingService] Model loaded, skipping warmup")
            
            return True
            
        except Exception as e:
            error_msg = f"Failed to load model: {e}"
            print(f"[EmbeddingService] {error_msg}")
            self.last_error = error_msg
            self.model_loaded = False
            return False
    
    def get_embedding(self, text: Union[str, List[str]], 
                     normalize: bool = None,
                     use_cache: bool = None) -> Union[List[float], List[List[float]]]:
        """
        获取文本嵌入，支持缓存和批处理优化
        
        Args:
            text: 单个文本或文本列表
            normalize: 是否归一化嵌入向量 (None=使用配置默认值)
            use_cache: 是否使用缓存 (None=使用配置设置)
            
        Returns:
            嵌入向量或嵌入向量列表
        """
        # 使用配置的默认值
        if normalize is None:
            normalize = self.config.normalize
        if use_cache is None:
            use_cache = self.config.enable_cache
            
        # 更新统计
        self._stats['requests_total'] += 1
        start_time = time.time()
        if isinstance(text, str):
            text = [text]
            return_single = True
        else:
            return_single = False
            
        # 预处理文本
        processed_texts = []
        cache_keys = []
        for t in text:
            if not t or not t.strip():
                t = "empty text"
            processed_t = t.replace("\n", " ").strip()
            processed_texts.append(processed_t)
            # 创建缓存键
            cache_key = f"{hash(processed_t)}_{normalize}" if use_cache else None
            cache_keys.append(cache_key)
        
        # 检查缓存（支持TTL）
        cached_embeddings = []
        texts_to_compute = []
        cache_indices = []
        
        if use_cache and self._embedding_cache is not None:
            current_time = datetime.now()
            
            # 清理过期缓存
            self._cleanup_expired_cache(current_time)
            
            for i, (cache_key, processed_t) in enumerate(zip(cache_keys, processed_texts)):
                if (cache_key in self._embedding_cache and 
                    cache_key in self._cache_timestamps and
                    current_time - self._cache_timestamps[cache_key] <= self.cache_ttl):
                    
                    cached_embeddings.append(self._embedding_cache[cache_key])
                    self._stats['cache_hits'] += 1
                else:
                    cached_embeddings.append(None)
                    texts_to_compute.append(processed_t)
                    cache_indices.append(i)
                    self._stats['cache_misses'] += 1
        else:
            texts_to_compute = processed_texts
            cache_indices = list(range(len(processed_texts)))
            cached_embeddings = [None] * len(processed_texts)
        
        # 计算未缓存的嵌入
        if texts_to_compute:
            if self.model_loaded and self.model is not None:
                try:
                    new_embeddings = self.model.encode(
                        texts_to_compute, 
                        normalize_embeddings=normalize,
                        show_progress_bar=False
                    )
                    new_embeddings = new_embeddings.tolist()
                except Exception as e:
                    print(f"[EmbeddingService] Encoding failed: {e}, falling back to hash")
                    new_embeddings = [self._hash_embedding(t) for t in texts_to_compute]
            else:
                new_embeddings = [self._hash_embedding(t) for t in texts_to_compute]
            
            # 更新缓存和结果
            current_time = datetime.now()
            for i, (cache_idx, embedding) in enumerate(zip(cache_indices, new_embeddings)):
                cached_embeddings[cache_idx] = embedding
                
                if use_cache and cache_keys[cache_idx] and self._embedding_cache is not None:
                    # 管理缓存大小
                    if len(self._embedding_cache) >= self.cache_max_size:
                        # FIFO缓存清理：删除最旧的条目
                        oldest_key = min(self._cache_timestamps.keys(), 
                                       key=lambda k: self._cache_timestamps[k])
                        del self._embedding_cache[oldest_key]
                        del self._cache_timestamps[oldest_key]
                    
                    self._embedding_cache[cache_keys[cache_idx]] = embedding
                    self._cache_timestamps[cache_keys[cache_idx]] = current_time
        
        # 更新性能统计
        processing_time = time.time() - start_time
        self._update_avg_processing_time(processing_time)
        
        return cached_embeddings[0] if return_single else cached_embeddings
    
    def _hash_embedding(self, text: str) -> List[float]:
        """Fallback hash-based embedding (原有逻辑)"""
        import hashlib
        
        # Create a deterministic hash-based embedding
        if not text:
            text = "this is blank"
        
        # Use multiple hash functions for better distribution
        hash_funcs = [hashlib.md5, hashlib.sha1, hashlib.sha256]
        embedding = []
        
        for hash_func in hash_funcs:
            hash_obj = hash_func(text.encode())
            hash_hex = hash_obj.hexdigest()
            
            # Convert hex to floats
            for i in range(0, min(len(hash_hex), 32), 2):
                val = int(hash_hex[i:i+2], 16) / 255.0 * 2.0 - 1.0
                embedding.append(val)
        
        # Pad or truncate to target dimension
        while len(embedding) < self.embedding_dim:
            embedding.extend(embedding[:min(len(embedding), 
                                          self.embedding_dim - len(embedding))])
        
        return embedding[:self.embedding_dim]
    
    def compute_similarity(self, embedding1: List[float], 
                          embedding2: List[float]) -> float:
        """计算两个嵌入向量的余弦相似度"""
        try:
            # 转换为numpy数组
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)
            
            # 计算余弦相似度
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
                
            similarity = dot_product / (norm1 * norm2)
            return float(similarity)
            
        except Exception as e:
            print(f"[EmbeddingService] Similarity computation failed: {e}")
            return 0.0
    
    def batch_similarity(self, query_embedding: List[float], 
                        target_embeddings: List[List[float]]) -> List[float]:
        """批量计算相似度"""
        similarities = []
        for target_emb in target_embeddings:
            sim = self.compute_similarity(query_embedding, target_emb)
            similarities.append(sim)
        return similarities
    
    def _cleanup_expired_cache(self, current_time: datetime):
        """清理过期的缓存条目"""
        if not self._embedding_cache or not self._cache_timestamps:
            return
            
        expired_keys = []
        for key, timestamp in self._cache_timestamps.items():
            if current_time - timestamp > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            if key in self._embedding_cache:
                del self._embedding_cache[key]
            if key in self._cache_timestamps:
                del self._cache_timestamps[key]
        
        if expired_keys:
            print(f"[EmbeddingService] Cleaned up {len(expired_keys)} expired cache entries")
    
    def _update_avg_processing_time(self, new_time: float):
        """更新平均处理时间"""
        # 使用指数移动平均
        alpha = 0.1  # 平滑因子
        self._stats['avg_processing_time'] = (
            alpha * new_time + (1 - alpha) * self._stats['avg_processing_time']
        )
    
    def get_cache_stats(self) -> dict:
        """获取缓存和性能统计信息"""
        cache_size = len(self._embedding_cache) if self._embedding_cache else 0
        
        # 计算缓存命中率
        total_requests = self._stats['cache_hits'] + self._stats['cache_misses']
        hit_rate = (self._stats['cache_hits'] / total_requests * 100) if total_requests > 0 else 0
        
        return {
            # 缓存统计
            "cache_enabled": self.config.enable_cache,
            "cache_size": cache_size,
            "cache_max_size": self.cache_max_size,
            "cache_hit_rate": f"{hit_rate:.1f}%",
            "cache_hits": self._stats['cache_hits'],
            "cache_misses": self._stats['cache_misses'],
            "cache_ttl_seconds": self.config.cache_ttl,
            
            # 模型统计
            "model_loaded": self.model_loaded,
            "model_path": self.model_path,
            "embedding_dim": self.embedding_dim,
            "device": self.config.device,
            "last_error": self.last_error,
            
            # 性能统计
            "requests_total": self._stats['requests_total'],
            "avg_processing_time_ms": f"{self._stats['avg_processing_time'] * 1000:.2f}",
            "max_concurrent_requests": self.config.max_concurrent_requests,
            
            # 配置信息
            "normalize_embeddings": self.config.normalize,
            "batch_size": self.config.batch_size,
            "preload_warmup": self.config.preload_warmup
        }
    
    def clear_cache(self):
        """清空缓存"""
        if self._embedding_cache:
            self._embedding_cache.clear()
        if self._cache_timestamps:
            self._cache_timestamps.clear()
        print("[EmbeddingService] Cache cleared")
    
    def preload_common_texts(self, texts: List[str]):
        """预加载常用文本的嵌入"""
        if not texts:
            return
        
        print(f"[EmbeddingService] Preloading {len(texts)} common texts...")
        start_time = time.time()
        
        self.get_embedding(texts)  # 这会自动缓存
        
        load_time = time.time() - start_time
        print(f"[EmbeddingService] Preloaded {len(texts)} texts in {load_time:.2f}s")
    
    async def get_embedding_async(self, text: Union[str, List[str]], 
                                 normalize: bool = None,
                                 use_cache: bool = None) -> Union[List[float], List[List[float]]]:
        """
        异步获取文本嵌入，支持并发控制
        """
        async with self._request_semaphore:
            # 在线程池中运行同步方法以避免阻塞事件循环
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                self.get_embedding, 
                text, normalize, use_cache
            )
    
    async def batch_embedding_async(self, texts: List[str],
                                   normalize: bool = None,
                                   use_cache: bool = None,
                                   max_concurrent: int = None) -> List[List[float]]:
        """
        异步批量处理嵌入，支持并发限制
        """
        if max_concurrent is None:
            max_concurrent = self.config.max_concurrent_requests
        
        # 创建并发任务
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_single_text(text):
            async with semaphore:
                return await self.get_embedding_async(text, normalize, use_cache)
        
        # 并发执行所有任务
        tasks = [process_single_text(text) for text in texts]
        results = await asyncio.gather(*tasks)
        
        return results

# 全局嵌入服务实例
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """获取全局嵌入服务实例"""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
        _embedding_service.initialize()
    return _embedding_service

def get_embedding(text: Union[str, List[str]], 
                 model: str = "text-embedding-ada-002") -> Union[List[float], List[List[float]]]:
    """
    兼容原有接口的嵌入函数
    model参数保留但会被忽略，始终使用本地模型
    """
    service = get_embedding_service()
    return service.get_embedding(text)

# 测试函数
def test_embedding_service():
    """测试嵌入服务"""
    print("=== Embedding Service Test ===")
    
    service = EmbeddingService()
    if not service.initialize():
        print("Failed to initialize embedding service")
        return False
    
    # 测试单个文本
    text1 = "Hello world"
    emb1 = service.get_embedding(text1)
    print(f"Text: '{text1}' -> Embedding dim: {len(emb1)}")
    
    # 测试批量文本
    texts = ["Hello world", "Hi there", "Good morning"]
    embs = service.get_embedding(texts)
    print(f"Batch texts: {len(texts)} -> Embeddings: {len(embs)}")
    
    # 测试相似度
    text2 = "Hi there"
    emb2 = service.get_embedding(text2)
    similarity = service.compute_similarity(emb1, emb2)
    print(f"Similarity between '{text1}' and '{text2}': {similarity:.4f}")
    
    print("=== Test completed ===")
    return True

if __name__ == "__main__":
    test_embedding_service()
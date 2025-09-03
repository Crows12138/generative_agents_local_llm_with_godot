#!/usr/bin/env python3
"""
LLM Call Optimization System
Implements caching, batching, and token optimization for LLM calls
"""

import time
import hashlib
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
import json
import re


@dataclass
class CacheEntry:
    """LLM cache entry with metadata"""
    response: str
    timestamp: datetime
    hit_count: int
    token_count: int
    response_time: float


@dataclass
class BatchRequest:
    """Batched LLM request"""
    prompt: str
    callback: Optional[Callable] = None
    metadata: Dict[str, Any] = None


class LLMCache:
    """Intelligent LLM response caching system"""
    
    def __init__(self, max_size: int = 10000, max_age_hours: int = 24):
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.max_size = max_size
        self.max_age = timedelta(hours=max_age_hours)
        self.hit_count = 0
        self.miss_count = 0
        self.lock = threading.RLock()
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create hash key for prompt"""
        # Normalize prompt for consistent caching
        normalized = re.sub(r'\s+', ' ', prompt.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()
    
    def get(self, prompt: str) -> Optional[str]:
        """Get cached response for prompt"""
        with self.lock:
            key = self._hash_prompt(prompt)
            
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if entry is still valid
                if datetime.now() - entry.timestamp <= self.max_age:
                    # Move to end (LRU)
                    self.cache.move_to_end(key)
                    entry.hit_count += 1
                    self.hit_count += 1
                    return entry.response
                else:
                    # Remove expired entry
                    del self.cache[key]
            
            self.miss_count += 1
            return None
    
    def put(self, prompt: str, response: str, token_count: int = 0, 
            response_time: float = 0.0) -> None:
        """Cache LLM response"""
        with self.lock:
            key = self._hash_prompt(prompt)
            
            # Create cache entry
            entry = CacheEntry(
                response=response,
                timestamp=datetime.now(),
                hit_count=0,
                token_count=token_count,
                response_time=response_time
            )
            
            self.cache[key] = entry
            
            # Enforce size limit
            while len(self.cache) > self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
    
    def cleanup_expired(self) -> int:
        """Remove expired entries"""
        with self.lock:
            current_time = datetime.now()
            expired_keys = []
            
            for key, entry in self.cache.items():
                if current_time - entry.timestamp > self.max_age:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.cache[key]
            
            return len(expired_keys)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self.lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = (self.hit_count / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "cache_size": len(self.cache),
                "max_size": self.max_size,
                "hit_count": self.hit_count,
                "miss_count": self.miss_count,
                "hit_rate_percent": hit_rate,
                "total_requests": total_requests
            }
    
    def clear(self) -> None:
        """Clear cache"""
        with self.lock:
            self.cache.clear()
            self.hit_count = 0
            self.miss_count = 0


class PromptOptimizer:
    """Optimize prompts to reduce token usage"""
    
    def __init__(self):
        self.optimization_rules = self._setup_optimization_rules()
        self.token_savings = defaultdict(int)
    
    def _setup_optimization_rules(self) -> List[Dict[str, Any]]:
        """Setup prompt optimization rules"""
        return [
            # Remove verbose phrases
            {"pattern": r"\bplease\s+", "replacement": "", "description": "Remove please"},
            {"pattern": r"\bcould\s+you\s+", "replacement": "", "description": "Remove could you"},
            {"pattern": r"\bi\s+would\s+like\s+you\s+to\s+", "replacement": "", "description": "Remove verbose request"},
            {"pattern": r"\bmake\s+sure\s+to\s+", "replacement": "", "description": "Remove make sure to"},
            {"pattern": r"\bit\s+is\s+important\s+to\s+", "replacement": "", "description": "Remove it is important"},
            
            # Compress common phrases
            {"pattern": r"\byou\s+should\s+", "replacement": "should ", "description": "Compress you should"},
            {"pattern": r"\byou\s+need\s+to\s+", "replacement": "need to ", "description": "Compress you need to"},
            {"pattern": r"\byou\s+must\s+", "replacement": "must ", "description": "Compress you must"},
            
            # Remove redundant words
            {"pattern": r"\s+and\s+also\s+", "replacement": " and ", "description": "Remove redundant also"},
            {"pattern": r"\s+in\s+order\s+to\s+", "replacement": " to ", "description": "Simplify in order to"},
            {"pattern": r"\s+as\s+well\s+as\s+", "replacement": " and ", "description": "Simplify as well as"},
            
            # Whitespace optimization
            {"pattern": r"\s+", "replacement": " ", "description": "Normalize whitespace"},
            {"pattern": r"^\s+|\s+$", "replacement": "", "description": "Trim whitespace"}
        ]
    
    def optimize(self, prompt: str) -> str:
        """Optimize prompt to reduce tokens"""
        original_length = len(prompt)
        optimized = prompt
        
        # Apply optimization rules
        for rule in self.optimization_rules:
            before_length = len(optimized)
            optimized = re.sub(rule["pattern"], rule["replacement"], optimized, flags=re.IGNORECASE)
            after_length = len(optimized)
            
            if before_length != after_length:
                self.token_savings[rule["description"]] += before_length - after_length
        
        # Calculate approximate token savings
        char_savings = original_length - len(optimized)
        estimated_token_savings = char_savings // 4  # Rough estimate: 4 chars per token
        
        return optimized
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        total_savings = sum(self.token_savings.values())
        
        return {
            "total_char_savings": total_savings,
            "estimated_token_savings": total_savings // 4,
            "optimization_breakdown": dict(self.token_savings)
        }


class LLMBatcher:
    """Batch multiple LLM requests for efficiency"""
    
    def __init__(self, batch_size: int = 10, batch_timeout: float = 2.0):
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests: List[BatchRequest] = []
        self.batch_timer: Optional[threading.Timer] = None
        self.lock = threading.RLock()
        self.llm_function: Optional[Callable] = None
    
    def set_llm_function(self, llm_function: Callable) -> None:
        """Set the LLM function to use for batched calls"""
        self.llm_function = llm_function
    
    def add_request(self, prompt: str, callback: Optional[Callable] = None,
                   metadata: Dict[str, Any] = None) -> None:
        """Add request to batch"""
        with self.lock:
            request = BatchRequest(prompt=prompt, callback=callback, metadata=metadata)
            self.pending_requests.append(request)
            
            # Start timer if first request
            if len(self.pending_requests) == 1:
                self._start_batch_timer()
            
            # Process immediately if batch is full
            if len(self.pending_requests) >= self.batch_size:
                self._process_batch()
    
    def _start_batch_timer(self) -> None:
        """Start timer for batch processing"""
        if self.batch_timer:
            self.batch_timer.cancel()
        
        self.batch_timer = threading.Timer(self.batch_timeout, self._process_batch)
        self.batch_timer.start()
    
    def _process_batch(self) -> None:
        """Process current batch of requests"""
        with self.lock:
            if not self.pending_requests or not self.llm_function:
                return
            
            # Cancel timer
            if self.batch_timer:
                self.batch_timer.cancel()
                self.batch_timer = None
            
            # Extract requests
            current_batch = self.pending_requests.copy()
            self.pending_requests.clear()
            
            # Process batch
            prompts = [req.prompt for req in current_batch]
            
            try:
                # Call LLM function with batch
                if hasattr(self.llm_function, '__call__'):
                    responses = self.llm_function(prompts)
                    
                    # Execute callbacks
                    for request, response in zip(current_batch, responses):
                        if request.callback:
                            try:
                                request.callback(response, request.metadata)
                            except Exception as e:
                                print(f"Callback error: {e}")
                
            except Exception as e:
                print(f"Batch processing error: {e}")
                # Execute error callbacks
                for request in current_batch:
                    if request.callback:
                        try:
                            request.callback(None, {"error": str(e)})
                        except Exception:
                            pass
    
    def flush(self) -> None:
        """Process all pending requests immediately"""
        with self.lock:
            if self.pending_requests:
                self._process_batch()


class SmartLLMClient:
    """Smart LLM client with optimization features"""
    
    def __init__(self, llm_function: Callable, cache_size: int = 10000):
        self.llm_function = llm_function
        self.cache = LLMCache(max_size=cache_size)
        self.optimizer = PromptOptimizer()
        self.batcher = LLMBatcher()
        self.batcher.set_llm_function(self._batch_llm_call)
        
        # Statistics
        self.total_calls = 0
        self.cached_calls = 0
        self.optimized_calls = 0
        self.batch_calls = 0
        
        # Start cleanup timer
        self._start_cleanup_timer()
    
    def call(self, prompt: str, use_cache: bool = True, 
             optimize_prompt: bool = True) -> str:
        """Make optimized LLM call"""
        self.total_calls += 1
        original_prompt = prompt
        
        # Optimize prompt
        if optimize_prompt:
            prompt = self.optimizer.optimize(prompt)
            if prompt != original_prompt:
                self.optimized_calls += 1
        
        # Check cache
        if use_cache:
            cached_response = self.cache.get(prompt)
            if cached_response:
                self.cached_calls += 1
                return cached_response
        
        # Make LLM call
        start_time = time.time()
        response = self.llm_function(prompt)
        response_time = time.time() - start_time
        
        # Cache response
        if use_cache:
            estimated_tokens = len(prompt) // 4  # Rough estimate
            self.cache.put(prompt, response, estimated_tokens, response_time)
        
        return response
    
    def call_async(self, prompt: str, callback: Callable, 
                   optimize_prompt: bool = True, metadata: Dict[str, Any] = None) -> None:
        """Make asynchronous batched LLM call"""
        self.batch_calls += 1
        
        # Optimize prompt
        if optimize_prompt:
            prompt = self.optimizer.optimize(prompt)
            self.optimized_calls += 1
        
        # Check cache first
        cached_response = self.cache.get(prompt)
        if cached_response:
            self.cached_calls += 1
            if callback:
                callback(cached_response, metadata)
            return
        
        # Add to batch
        def batch_callback(response: str, batch_metadata: Dict[str, Any]) -> None:
            if response and "error" not in batch_metadata:
                # Cache successful response
                estimated_tokens = len(prompt) // 4
                self.cache.put(prompt, response, estimated_tokens, 0.0)
            
            # Execute original callback
            if callback:
                callback(response, metadata)
        
        self.batcher.add_request(prompt, batch_callback, metadata)
    
    def _batch_llm_call(self, prompts: List[str]) -> List[str]:
        """Internal batch LLM call"""
        # For demo purposes, we'll simulate batch processing
        # In real implementation, this would call the actual LLM batch API
        responses = []
        for prompt in prompts:
            response = self.llm_function(prompt)
            responses.append(response)
        return responses
    
    def _start_cleanup_timer(self) -> None:
        """Start periodic cache cleanup"""
        def cleanup():
            self.cache.cleanup_expired()
            # Schedule next cleanup
            cleanup_timer = threading.Timer(3600, cleanup)  # Every hour
            cleanup_timer.daemon = True
            cleanup_timer.start()
        
        cleanup_timer = threading.Timer(3600, cleanup)
        cleanup_timer.daemon = True
        cleanup_timer.start()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics"""
        cache_stats = self.cache.get_stats()
        optimizer_stats = self.optimizer.get_optimization_stats()
        
        cache_hit_rate = (self.cached_calls / self.total_calls * 100) if self.total_calls > 0 else 0
        optimization_rate = (self.optimized_calls / self.total_calls * 100) if self.total_calls > 0 else 0
        
        return {
            "total_calls": self.total_calls,
            "cached_calls": self.cached_calls,
            "optimized_calls": self.optimized_calls,
            "batch_calls": self.batch_calls,
            "cache_hit_rate_percent": cache_hit_rate,
            "optimization_rate_percent": optimization_rate,
            "cache_stats": cache_stats,
            "optimizer_stats": optimizer_stats
        }
    
    def flush_batches(self) -> None:
        """Flush all pending batches"""
        self.batcher.flush()
    
    def clear_cache(self) -> None:
        """Clear LLM cache"""
        self.cache.clear()


def demo_llm_optimizer():
    """Demo the LLM optimization system"""
    print("=== LLM Optimization System Demo ===\n")
    
    # Mock LLM function for demo
    def mock_llm(prompt: Union[str, List[str]]) -> Union[str, List[str]]:
        """Mock LLM function that simulates processing time"""
        time.sleep(0.1)  # Simulate LLM delay
        
        if isinstance(prompt, list):
            return [f"Response to: {p[:30]}..." for p in prompt]
        else:
            return f"Response to: {prompt[:30]}..."
    
    # Create smart LLM client
    client = SmartLLMClient(mock_llm)
    
    # Test prompt optimization
    verbose_prompt = "Could you please make sure to tell me what the bartender should do when a customer arrives?"
    print(f"Original prompt: '{verbose_prompt}'")
    
    optimized = client.optimizer.optimize(verbose_prompt)
    print(f"Optimized prompt: '{optimized}'")
    print(f"Character savings: {len(verbose_prompt) - len(optimized)}")
    
    # Test caching
    print("\n--- Testing Cache ---")
    start_time = time.time()
    response1 = client.call("What should I do?")
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    response2 = client.call("What should I do?")  # Should be cached
    second_call_time = time.time() - start_time
    
    print(f"First call time: {first_call_time:.3f}s")
    print(f"Second call time: {second_call_time:.3f}s (cached)")
    print(f"Cache speedup: {first_call_time / second_call_time:.1f}x")
    
    # Test async batching
    print("\n--- Testing Async Batching ---")
    results = []
    
    def callback(response: str, metadata: Dict[str, Any]) -> None:
        results.append(f"Async response: {response}")
    
    # Add multiple async requests
    for i in range(5):
        client.call_async(f"Request {i}", callback, metadata={"request_id": i})
    
    # Wait for batch processing
    time.sleep(3)
    client.flush_batches()
    
    print(f"Received {len(results)} async responses")
    
    # Show statistics
    print("\n--- Performance Statistics ---")
    stats = client.get_stats()
    print(f"Total calls: {stats['total_calls']}")
    print(f"Cache hit rate: {stats['cache_hit_rate_percent']:.1f}%")
    print(f"Optimization rate: {stats['optimization_rate_percent']:.1f}%")
    print(f"Batch calls: {stats['batch_calls']}")
    
    cache_stats = stats['cache_stats']
    print(f"Cache size: {cache_stats['cache_size']}/{cache_stats['max_size']}")
    print(f"Cache hits/misses: {cache_stats['hit_count']}/{cache_stats['miss_count']}")
    
    optimizer_stats = stats['optimizer_stats']
    print(f"Estimated token savings: {optimizer_stats['estimated_token_savings']}")


if __name__ == "__main__":
    demo_llm_optimizer()
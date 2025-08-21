"""
Optimized Unified AI Response Parser
Extends the original unified_parser.py with performance optimizations
"""

import re
import time
import threading
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from collections import defaultdict, OrderedDict

# Import original parser components
try:
    from .unified_parser import UnifiedParser, ParseResult, ParseType
except ImportError:
    from unified_parser import UnifiedParser, ParseResult, ParseType

# Import optimization components
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'debug_system'))

try:
    from debug_system.action_parser_optimizer import OptimizedActionParser, FastActionLookup
    from debug_system.flow_tracer import get_tracer
    OPTIMIZATION_AVAILABLE = True
except ImportError:
    OPTIMIZATION_AVAILABLE = False
    print("[optimized_unified_parser] Optimization components not available")


class OptimizedUnifiedParser(UnifiedParser):
    """Enhanced Unified Parser with integrated optimizations"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize optimization components
        if OPTIMIZATION_AVAILABLE:
            self._setup_optimizations()
        
        # Performance tracking
        self.total_parses = 0
        self.cache_hits = 0
        self.fast_path_hits = 0
        self.optimization_enabled = OPTIMIZATION_AVAILABLE
        
        # Thread safety
        self.lock = threading.RLock()
    
    def _setup_optimizations(self) -> None:
        """Setup optimization components"""
        # Initialize optimized action parser
        self.action_parser = OptimizedActionParser()
        
        # Initialize fast lookup
        self.fast_lookup = FastActionLookup()
        
        # Setup flow tracing
        self.flow_tracer = get_tracer()
        
        # Parse cache for all parse types
        self.parse_cache: Dict[str, OrderedDict] = {
            parse_type.value: OrderedDict()
            for parse_type in ParseType
        }
        self.max_cache_size = 1000
        
        print("[optimized_unified_parser] Optimizations initialized")
    
    def parse_action(self, text: str, use_optimizations: bool = True, **kwargs) -> ParseResult:
        """Optimized action parsing"""
        
        with self.lock:
            self.total_parses += 1
            start_time = time.time()
            
            # Use optimizations if available
            if self.optimization_enabled and use_optimizations:
                return self._optimized_action_parse(text, **kwargs)
            else:
                # Fallback to original method
                return super().parse_action(text, **kwargs)
    
    def _optimized_action_parse(self, text: str, **kwargs) -> ParseResult:
        """Optimized action parsing implementation"""
        
        # Check cache first
        cache_key = self._get_cache_key(text)
        cache = self.parse_cache[ParseType.ACTION.value]
        
        if cache_key in cache:
            self.cache_hits += 1
            cached_result = cache[cache_key]
            # Move to end (LRU)
            cache.move_to_end(cache_key)
            return cached_result
        
        # Try fast path lookup
        fast_result = self.fast_lookup.fast_lookup(text)
        if fast_result and fast_result.confidence > 0.8:
            self.fast_path_hits += 1
            
            # Convert to ParseResult format
            result = ParseResult(
                parse_type=ParseType.ACTION,
                success=True,
                value={
                    "action": fast_result.action_type,
                    "target": fast_result.target,
                    "confidence": fast_result.confidence
                },
                confidence=fast_result.confidence,
                raw_input=text,
                metadata={"method": "fast_path", "pattern": fast_result.pattern_used}
            )
            
            # Cache result
            self._cache_result(ParseType.ACTION.value, cache_key, result)
            return result
        
        # Use optimized action parser
        optimized_result = self.action_parser.parse(text)
        
        # Convert to ParseResult format
        result = ParseResult(
            parse_type=ParseType.ACTION,
            success=optimized_result.action_type != "idle",
            value={
                "action": optimized_result.action_type,
                "target": optimized_result.target,
                "confidence": optimized_result.confidence
            },
            confidence=optimized_result.confidence,
            raw_input=text,
            metadata={
                "method": "optimized_regex",
                "pattern": optimized_result.pattern_used,
                "processing_time": optimized_result.processing_time
            }
        )
        
        # Cache result
        self._cache_result(ParseType.ACTION.value, cache_key, result)
        
        return result
    
    def parse_decision(self, text: str, use_optimizations: bool = True, **kwargs) -> ParseResult:
        """Optimized decision parsing"""
        
        with self.lock:
            self.total_parses += 1
            
            if self.optimization_enabled and use_optimizations:
                return self._optimized_decision_parse(text, **kwargs)
            else:
                return super().parse_decision(text, **kwargs)
    
    def _optimized_decision_parse(self, text: str, **kwargs) -> ParseResult:
        """Optimized decision parsing implementation"""
        
        # Check cache
        cache_key = self._get_cache_key(text)
        cache = self.parse_cache[ParseType.DECISION.value]
        
        if cache_key in cache:
            self.cache_hits += 1
            cached_result = cache[cache_key]
            cache.move_to_end(cache_key)
            return cached_result
        
        # Fast decision patterns
        text_lower = text.lower().strip()
        
        # Quick decision detection
        if any(word in text_lower for word in ["yes", "agree", "accept", "ok", "sure"]):
            result = ParseResult(
                parse_type=ParseType.DECISION,
                success=True,
                value={"decision": "yes", "confidence": 0.9},
                confidence=0.9,
                raw_input=text,
                metadata={"method": "fast_decision"}
            )
        elif any(word in text_lower for word in ["no", "disagree", "reject", "refuse"]):
            result = ParseResult(
                parse_type=ParseType.DECISION,
                success=True,
                value={"decision": "no", "confidence": 0.9},
                confidence=0.9,
                raw_input=text,
                metadata={"method": "fast_decision"}
            )
        else:
            # Fallback to original method
            result = super().parse_decision(text, **kwargs)
        
        # Cache result
        self._cache_result(ParseType.DECISION.value, cache_key, result)
        
        return result
    
    def parse_emotion(self, text: str, use_optimizations: bool = True, **kwargs) -> ParseResult:
        """Optimized emotion parsing"""
        
        with self.lock:
            self.total_parses += 1
            
            if self.optimization_enabled and use_optimizations:
                return self._optimized_emotion_parse(text, **kwargs)
            else:
                return super().parse_emotion(text, **kwargs)
    
    def _optimized_emotion_parse(self, text: str, **kwargs) -> ParseResult:
        """Optimized emotion parsing implementation"""
        
        # Check cache
        cache_key = self._get_cache_key(text)
        cache = self.parse_cache[ParseType.EMOTION.value]
        
        if cache_key in cache:
            self.cache_hits += 1
            cached_result = cache[cache_key]
            cache.move_to_end(cache_key)
            return cached_result
        
        # Fast emotion keywords
        emotion_keywords = {
            "happy": ["happy", "joy", "glad", "cheerful", "excited", "pleased"],
            "sad": ["sad", "unhappy", "depressed", "down", "melancholy"],
            "angry": ["angry", "mad", "furious", "annoyed", "frustrated"],
            "fear": ["afraid", "scared", "nervous", "anxious", "worried"],
            "surprise": ["surprised", "amazed", "shocked", "astonished"],
            "neutral": ["calm", "neutral", "normal", "steady"]
        }
        
        text_lower = text.lower()
        best_emotion = "neutral"
        best_confidence = 0.3
        
        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    confidence = 0.8 if len(keyword) > 4 else 0.6
                    if confidence > best_confidence:
                        best_emotion = emotion
                        best_confidence = confidence
        
        result = ParseResult(
            parse_type=ParseType.EMOTION,
            success=best_confidence > 0.5,
            value={"emotion": best_emotion, "confidence": best_confidence},
            confidence=best_confidence,
            raw_input=text,
            metadata={"method": "keyword_emotion"}
        )
        
        # Cache result
        self._cache_result(ParseType.EMOTION.value, cache_key, result)
        
        return result
    
    def batch_parse(self, texts: List[str], parse_type: ParseType, 
                   use_optimizations: bool = True) -> List[ParseResult]:
        """Batch parsing with optimizations"""
        
        results = []
        
        # Choose parse method based on type
        if parse_type == ParseType.ACTION:
            parse_method = self.parse_action
        elif parse_type == ParseType.DECISION:
            parse_method = self.parse_decision
        elif parse_type == ParseType.EMOTION:
            parse_method = self.parse_emotion
        else:
            # Fallback to generic parse
            parse_method = lambda text, **kwargs: super().parse(text, parse_type, **kwargs)
        
        # Process all texts
        for text in texts:
            result = parse_method(text, use_optimizations=use_optimizations)
            results.append(result)
        
        return results
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Normalize text for consistent caching
        normalized = re.sub(r'\s+', ' ', text.strip().lower())
        return normalized[:100]  # Limit key length
    
    def _cache_result(self, parse_type: str, cache_key: str, result: ParseResult) -> None:
        """Cache parsing result"""
        cache = self.parse_cache[parse_type]
        cache[cache_key] = result
        
        # Enforce cache size limit
        while len(cache) > self.max_cache_size:
            cache.popitem(last=False)
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        
        with self.lock:
            cache_hit_rate = (self.cache_hits / self.total_parses * 100) if self.total_parses > 0 else 0
            fast_path_rate = (self.fast_path_hits / self.total_parses * 100) if self.total_parses > 0 else 0
            
            stats = {
                "total_parses": self.total_parses,
                "cache_hits": self.cache_hits,
                "fast_path_hits": self.fast_path_hits,
                "cache_hit_rate_percent": cache_hit_rate,
                "fast_path_rate_percent": fast_path_rate,
                "optimization_enabled": self.optimization_enabled
            }
            
            # Add cache sizes
            for parse_type, cache in self.parse_cache.items():
                stats[f"{parse_type}_cache_size"] = len(cache)
            
            # Add action parser stats if available
            if hasattr(self, 'action_parser'):
                stats["action_parser_stats"] = self.action_parser.get_performance_stats()
            
            return stats
    
    def clear_caches(self) -> None:
        """Clear all optimization caches"""
        
        with self.lock:
            for cache in self.parse_cache.values():
                cache.clear()
            
            if hasattr(self, 'action_parser'):
                self.action_parser.clear_cache()
            
            # Reset counters
            self.cache_hits = 0
            self.fast_path_hits = 0
            
            print("[optimized_unified_parser] Caches cleared")
    
    def optimize_for_workload(self, workload_type: str) -> None:
        """Optimize parser for specific workload types"""
        
        if workload_type == "action_heavy":
            # Optimize for many action parses
            self.max_cache_size = 2000
            if hasattr(self, 'action_parser'):
                self.action_parser.max_cache_size = 2000
                
        elif workload_type == "mixed":
            # Balanced optimization
            self.max_cache_size = 1000
            
        elif workload_type == "memory_constrained":
            # Minimize memory usage
            self.max_cache_size = 500
            if hasattr(self, 'action_parser'):
                self.action_parser.max_cache_size = 500
        
        print(f"[optimized_unified_parser] Optimized for {workload_type} workload")
    
    def generate_optimization_report(self, filename: str = "parser_optimization_report.md") -> None:
        """Generate optimization report"""
        
        stats = self.get_optimization_stats()
        
        with open(filename, "w", encoding='utf-8') as f:
            f.write("# Parser Optimization Report\n\n")
            f.write(f"**Generated**: {datetime.now().isoformat()}\n\n")
            
            f.write("## Overall Statistics\n\n")
            f.write(f"- Total parses: {stats['total_parses']}\n")
            f.write(f"- Cache hit rate: {stats['cache_hit_rate_percent']:.1f}%\n")
            f.write(f"- Fast path rate: {stats['fast_path_rate_percent']:.1f}%\n")
            f.write(f"- Optimization enabled: {stats['optimization_enabled']}\n\n")
            
            f.write("## Cache Sizes\n\n")
            for parse_type in ParseType:
                cache_key = f"{parse_type.value}_cache_size"
                if cache_key in stats:
                    f.write(f"- {parse_type.value}: {stats[cache_key]} entries\n")
            
            # Action parser specific stats
            if "action_parser_stats" in stats:
                action_stats = stats["action_parser_stats"]
                f.write("\n## Action Parser Statistics\n\n")
                f.write(f"- Total parses: {action_stats.get('total_parses', 0)}\n")
                f.write(f"- Fast path hits: {action_stats.get('fast_path_hits', 0)}\n")
                f.write(f"- Cache hits: {action_stats.get('cache_hits', 0)}\n")
                f.write(f"- Regex parses: {action_stats.get('regex_parses', 0)}\n")
        
        print(f"[optimized_unified_parser] Report saved to {filename}")


# Global optimized parser instance
_optimized_parser = None


def get_optimized_unified_parser(**kwargs) -> OptimizedUnifiedParser:
    """Get global optimized parser instance"""
    global _optimized_parser
    
    if _optimized_parser is None:
        _optimized_parser = OptimizedUnifiedParser(**kwargs)
    
    return _optimized_parser


def demo_parser_optimization():
    """Demo the parser optimization"""
    print("=== Optimized Unified Parser Demo ===\n")
    
    # Create optimized parser
    parser = get_optimized_unified_parser()
    
    # Test action parsing
    print("Testing action parsing...")
    action_tests = [
        "go to bar",
        "I should walk to the kitchen",
        "talk to customer Alice",
        "clean the counter",
        "go to bar",  # Duplicate for cache test
    ]
    
    for text in action_tests:
        result = parser.parse_action(text)
        print(f"'{text}' -> {result.value} (confidence: {result.confidence:.2f})")
    
    # Test emotion parsing
    print("\nTesting emotion parsing...")
    emotion_tests = [
        "I am very happy today",
        "This makes me sad",
        "I'm feeling angry about this",
        "I am very happy today",  # Duplicate for cache test
    ]
    
    for text in emotion_tests:
        result = parser.parse_emotion(text)
        print(f"'{text}' -> {result.value}")
    
    # Show optimization statistics
    print("\n--- Optimization Statistics ---")
    stats = parser.get_optimization_stats()
    
    print(f"Total parses: {stats['total_parses']}")
    print(f"Cache hit rate: {stats['cache_hit_rate_percent']:.1f}%")
    print(f"Fast path rate: {stats['fast_path_rate_percent']:.1f}%")
    
    # Generate report
    parser.generate_optimization_report("demo_parser_report.md")
    print("\nDemo complete! Report saved.")


if __name__ == "__main__":
    demo_parser_optimization()
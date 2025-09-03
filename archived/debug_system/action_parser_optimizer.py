#!/usr/bin/env python3
"""
Action Parsing Optimization System
Optimizes LLM output parsing with caching, pre-compilation, and fast paths
"""

import re
import time
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Pattern, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict, OrderedDict
import hashlib


@dataclass
class ParseResult:
    """Action parsing result with metadata"""
    action_type: str
    target: Optional[str]
    confidence: float
    processing_time: float
    pattern_used: str
    original_text: str


@dataclass
class PatternStats:
    """Statistics for a parsing pattern"""
    pattern: str
    usage_count: int
    success_count: int
    average_time: float
    last_used: datetime


class FastActionLookup:
    """Fast lookup table for common action patterns"""
    
    def __init__(self):
        self.exact_matches: Dict[str, ParseResult] = {}
        self.keyword_maps: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
        self.prefix_tree: Dict[str, Any] = {}
        self.build_lookup_tables()
    
    def build_lookup_tables(self) -> None:
        """Build optimized lookup tables"""
        
        # Common exact phrases and their actions
        exact_actions = {
            "go to bar": ("move", "bar", 0.95),
            "clean the counter": ("work", "counter", 0.90),
            "talk to customer": ("talk", "customer", 0.90),
            "serve drink": ("work", "drink", 0.85),
            "check inventory": ("work", "inventory", 0.90),
            "take order": ("talk", "customer", 0.85),
            "prepare cocktail": ("work", "cocktail", 0.90),
            "wipe table": ("work", "table", 0.85),
        }
        
        for phrase, (action_type, target, confidence) in exact_actions.items():
            result = ParseResult(
                action_type=action_type,
                target=target,
                confidence=confidence,
                processing_time=0.001,  # Very fast for exact matches
                pattern_used="exact_match",
                original_text=phrase
            )
            self.exact_matches[phrase.lower()] = result
        
        # Keyword-based fast matching
        keyword_actions = {
            "move": [("walk", 0.9), ("go", 0.9), ("move", 0.95), ("head", 0.8)],
            "talk": [("talk", 0.95), ("speak", 0.9), ("chat", 0.85), ("ask", 0.8)],
            "work": [("clean", 0.9), ("wipe", 0.85), ("prepare", 0.8), ("make", 0.75)],
            "interact": [("use", 0.8), ("grab", 0.85), ("take", 0.8), ("pick", 0.75)]
        }
        
        for action_type, keywords in keyword_actions.items():
            for keyword, confidence in keywords:
                self.keyword_maps[keyword.lower()].append((action_type, confidence))
        
        # Build prefix tree for fast prefix matching
        self.build_prefix_tree()
    
    def build_prefix_tree(self) -> None:
        """Build prefix tree for fast pattern matching"""
        for phrase in self.exact_matches.keys():
            current = self.prefix_tree
            for char in phrase:
                if char not in current:
                    current[char] = {}
                current = current[char]
            current['_end'] = phrase
    
    def fast_lookup(self, text: str) -> Optional[ParseResult]:
        """Attempt fast lookup for common patterns"""
        text_lower = text.lower().strip()
        
        # Exact match check
        if text_lower in self.exact_matches:
            return self.exact_matches[text_lower]
        
        # Prefix tree search for partial matches
        words = text_lower.split()
        for i in range(len(words)):
            for j in range(i + 1, min(i + 4, len(words) + 1)):  # Check up to 3-word phrases
                phrase = " ".join(words[i:j])
                if phrase in self.exact_matches:
                    return self.exact_matches[phrase]
        
        # Keyword-based fast matching
        best_match = None
        best_confidence = 0.0
        
        for word in words:
            if word in self.keyword_maps:
                for action_type, confidence in self.keyword_maps[word]:
                    if confidence > best_confidence:
                        best_confidence = confidence
                        # Extract target from text
                        target = self._extract_target_fast(text_lower, word)
                        best_match = ParseResult(
                            action_type=action_type,
                            target=target,
                            confidence=confidence,
                            processing_time=0.002,
                            pattern_used="keyword_match",
                            original_text=text
                        )
        
        return best_match
    
    def _extract_target_fast(self, text: str, action_word: str) -> Optional[str]:
        """Fast target extraction for keyword matches"""
        words = text.split()
        try:
            action_index = words.index(action_word)
            # Look for target after action word
            if action_index + 1 < len(words):
                # Skip common prepositions
                skip_words = {"to", "the", "a", "an", "with", "at", "on"}
                for i in range(action_index + 1, len(words)):
                    if words[i] not in skip_words:
                        return words[i]
        except ValueError:
            pass
        return None


class OptimizedActionParser:
    """Optimized action parser with multiple optimization strategies"""
    
    def __init__(self):
        self.fast_lookup = FastActionLookup()
        self.compiled_patterns: Dict[str, Pattern] = {}
        self.pattern_stats: Dict[str, PatternStats] = {}
        self.parse_cache: OrderedDict[str, ParseResult] = OrderedDict()
        self.max_cache_size = 10000
        
        # Performance tracking
        self.total_parses = 0
        self.fast_path_hits = 0
        self.cache_hits = 0
        self.regex_parses = 0
        
        self.lock = threading.RLock()
        self.precompile_patterns()
    
    def precompile_patterns(self) -> None:
        """Pre-compile all regex patterns for faster matching"""
        patterns = {
            # Movement patterns
            "move_to": r"(?:walk|go|move|head)\s+(?:to|towards)\s+(?:the\s+)?(\w+)",
            "move_simple": r"(?:walk|go|move)\s+(\w+)",
            
            # Communication patterns  
            "talk_to": r"(?:talk|speak|chat)\s+(?:to|with)\s+(\w+)",
            "say_to": r"(?:say|tell)\s+(?:\w+\s+)?(?:to\s+)?(\w+)",
            
            # Work patterns
            "clean_object": r"(?:clean|wipe|polish)\s+(?:the\s+)?(\w+)",
            "check_object": r"(?:check|inspect|examine)\s+(?:the\s+)?(\w+)",
            "prepare_object": r"(?:prepare|make|mix)\s+(?:a\s+|the\s+)?(\w+)",
            
            # Interaction patterns
            "use_object": r"(?:use|grab|take|pick\s+up)\s+(?:the\s+)?(\w+)",
            "serve_object": r"(?:serve|give|bring)\s+(?:a\s+|the\s+)?(\w+)",
            
            # Complex patterns
            "action_target_location": r"(\w+)\s+(?:the\s+)?(\w+)\s+(?:at|in|on)\s+(?:the\s+)?(\w+)",
            "multiple_actions": r"(\w+)\s+(?:and|then)\s+(\w+)",
        }
        
        action_mappings = {
            "move_to": "move",
            "move_simple": "move", 
            "talk_to": "talk",
            "say_to": "talk",
            "clean_object": "work",
            "check_object": "work",
            "prepare_object": "work",
            "use_object": "interact",
            "serve_object": "work",
            "action_target_location": "complex",
            "multiple_actions": "complex"
        }
        
        for name, pattern in patterns.items():
            compiled = re.compile(pattern, re.IGNORECASE)
            self.compiled_patterns[name] = compiled
            
            # Initialize stats
            self.pattern_stats[name] = PatternStats(
                pattern=pattern,
                usage_count=0,
                success_count=0,
                average_time=0.0,
                last_used=datetime.now()
            )
    
    def parse(self, text: str, use_cache: bool = True) -> ParseResult:
        """Parse action from text with optimizations"""
        with self.lock:
            self.total_parses += 1
            start_time = time.time()
            
            # Cache check
            if use_cache:
                cache_key = self._get_cache_key(text)
                if cache_key in self.parse_cache:
                    self.cache_hits += 1
                    cached_result = self.parse_cache[cache_key]
                    # Move to end (LRU)
                    self.parse_cache.move_to_end(cache_key)
                    return cached_result
            
            # Fast path lookup
            fast_result = self.fast_lookup.fast_lookup(text)
            if fast_result and fast_result.confidence > 0.8:
                self.fast_path_hits += 1
                if use_cache:
                    self._cache_result(text, fast_result)
                return fast_result
            
            # Regex pattern matching
            result = self._regex_parse(text)
            self.regex_parses += 1
            
            # Update timing
            processing_time = time.time() - start_time
            result.processing_time = processing_time
            
            # Cache result
            if use_cache:
                self._cache_result(text, result)
            
            return result
    
    def _regex_parse(self, text: str) -> ParseResult:
        """Parse using pre-compiled regex patterns"""
        text_lower = text.lower().strip()
        best_result = None
        best_confidence = 0.0
        
        # Try patterns in order of specificity (most specific first)
        pattern_order = [
            "action_target_location", "move_to", "talk_to", "clean_object",
            "check_object", "prepare_object", "use_object", "serve_object",
            "say_to", "move_simple", "multiple_actions"
        ]
        
        for pattern_name in pattern_order:
            if pattern_name not in self.compiled_patterns:
                continue
                
            pattern = self.compiled_patterns[pattern_name]
            stats = self.pattern_stats[pattern_name]
            
            start_time = time.time()
            match = pattern.search(text_lower)
            pattern_time = time.time() - start_time
            
            # Update pattern statistics
            stats.usage_count += 1
            stats.average_time = (stats.average_time * (stats.usage_count - 1) + pattern_time) / stats.usage_count
            stats.last_used = datetime.now()
            
            if match:
                stats.success_count += 1
                action_type, target, confidence = self._extract_action_info(pattern_name, match)
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_result = ParseResult(
                        action_type=action_type,
                        target=target,
                        confidence=confidence,
                        processing_time=pattern_time,
                        pattern_used=pattern_name,
                        original_text=text
                    )
                    
                    # If we found a high-confidence match, stop searching
                    if confidence > 0.9:
                        break
        
        # Return best result or default
        if best_result:
            return best_result
        else:
            return ParseResult(
                action_type="idle",
                target=None,
                confidence=0.1,
                processing_time=0.001,
                pattern_used="default",
                original_text=text
            )
    
    def _extract_action_info(self, pattern_name: str, match: re.Match) -> Tuple[str, Optional[str], float]:
        """Extract action type, target, and confidence from regex match"""
        action_mappings = {
            "move_to": ("move", match.group(1), 0.95),
            "move_simple": ("move", match.group(1), 0.85),
            "talk_to": ("talk", match.group(1), 0.90),
            "say_to": ("talk", match.group(1), 0.85),
            "clean_object": ("work", match.group(1), 0.90),
            "check_object": ("work", match.group(1), 0.85),
            "prepare_object": ("work", match.group(1), 0.85),
            "use_object": ("interact", match.group(1), 0.80),
            "serve_object": ("work", match.group(1), 0.85),
        }
        
        if pattern_name in action_mappings:
            return action_mappings[pattern_name]
        
        # Handle complex patterns
        if pattern_name == "action_target_location":
            action = match.group(1)
            target = match.group(2)
            location = match.group(3)
            return (self._classify_action(action), f"{target}@{location}", 0.75)
        
        elif pattern_name == "multiple_actions":
            action1 = match.group(1)
            action2 = match.group(2)
            return (self._classify_action(action1), action2, 0.70)
        
        return ("unknown", None, 0.1)
    
    def _classify_action(self, action_word: str) -> str:
        """Classify action word into action type"""
        move_words = {"walk", "go", "move", "head", "run"}
        talk_words = {"talk", "speak", "chat", "say", "tell", "ask"}
        work_words = {"clean", "wipe", "prepare", "make", "serve", "check"}
        interact_words = {"use", "grab", "take", "pick", "get"}
        
        action_lower = action_word.lower()
        
        if action_lower in move_words:
            return "move"
        elif action_lower in talk_words:
            return "talk"
        elif action_lower in work_words:
            return "work"
        elif action_lower in interact_words:
            return "interact"
        else:
            return "unknown"
    
    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text"""
        # Normalize text for consistent caching
        normalized = re.sub(r'\s+', ' ', text.strip().lower())
        return hashlib.md5(normalized.encode()).hexdigest()[:16]
    
    def _cache_result(self, text: str, result: ParseResult) -> None:
        """Cache parsing result"""
        cache_key = self._get_cache_key(text)
        self.parse_cache[cache_key] = result
        
        # Enforce cache size limit
        while len(self.parse_cache) > self.max_cache_size:
            self.parse_cache.popitem(last=False)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        with self.lock:
            fast_path_rate = (self.fast_path_hits / self.total_parses * 100) if self.total_parses > 0 else 0
            cache_hit_rate = (self.cache_hits / self.total_parses * 100) if self.total_parses > 0 else 0
            regex_rate = (self.regex_parses / self.total_parses * 100) if self.total_parses > 0 else 0
            
            # Pattern statistics
            pattern_stats = {}
            for name, stats in self.pattern_stats.items():
                success_rate = (stats.success_count / stats.usage_count * 100) if stats.usage_count > 0 else 0
                pattern_stats[name] = {
                    "usage_count": stats.usage_count,
                    "success_count": stats.success_count,
                    "success_rate_percent": success_rate,
                    "average_time_ms": stats.average_time * 1000,
                    "last_used": stats.last_used.isoformat()
                }
            
            return {
                "total_parses": self.total_parses,
                "fast_path_hits": self.fast_path_hits,
                "cache_hits": self.cache_hits,
                "regex_parses": self.regex_parses,
                "fast_path_rate_percent": fast_path_rate,
                "cache_hit_rate_percent": cache_hit_rate,
                "regex_rate_percent": regex_rate,
                "cache_size": len(self.parse_cache),
                "pattern_statistics": pattern_stats
            }
    
    def optimize_patterns(self) -> List[str]:
        """Optimize pattern order based on usage statistics"""
        recommendations = []
        
        # Find most used patterns
        sorted_patterns = sorted(
            self.pattern_stats.items(),
            key=lambda x: x[1].usage_count,
            reverse=True
        )
        
        # Find patterns with low success rates
        low_success_patterns = [
            name for name, stats in self.pattern_stats.items()
            if stats.usage_count > 10 and (stats.success_count / stats.usage_count) < 0.3
        ]
        
        if low_success_patterns:
            recommendations.append(f"Consider improving patterns: {', '.join(low_success_patterns)}")
        
        # Find slow patterns
        slow_patterns = [
            name for name, stats in self.pattern_stats.items()
            if stats.average_time > 0.001
        ]
        
        if slow_patterns:
            recommendations.append(f"Consider optimizing slow patterns: {', '.join(slow_patterns)}")
        
        # Suggest fast path additions
        if self.fast_path_hits / self.total_parses < 0.5:
            recommendations.append("Consider adding more fast path patterns for common actions")
        
        return recommendations
    
    def clear_cache(self) -> None:
        """Clear parsing cache"""
        with self.lock:
            self.parse_cache.clear()
    
    def reset_stats(self) -> None:
        """Reset performance statistics"""
        with self.lock:
            self.total_parses = 0
            self.fast_path_hits = 0
            self.cache_hits = 0
            self.regex_parses = 0
            
            for stats in self.pattern_stats.values():
                stats.usage_count = 0
                stats.success_count = 0
                stats.average_time = 0.0


def demo_action_parser_optimizer():
    """Demo the action parser optimization system"""
    print("=== Action Parser Optimization Demo ===\n")
    
    parser = OptimizedActionParser()
    
    # Test cases
    test_cases = [
        "go to bar",  # Fast path exact match
        "I should walk to the kitchen",  # Regex pattern
        "talk to customer Alice",  # Regex pattern  
        "clean the bar counter",  # Fast path
        "let me check the inventory quickly",  # Regex
        "serve drink to customer",  # Complex
        "go to bar",  # Cache hit
        "clean the bar counter",  # Cache hit
        "prepare a cocktail for the customer at table 3",  # Complex
        "I need to wipe down the tables and then check inventory"  # Multiple actions
    ]
    
    print("Testing action parsing performance:")
    print("-" * 50)
    
    total_time = 0
    for i, test_case in enumerate(test_cases, 1):
        start_time = time.time()
        result = parser.parse(test_case)
        parse_time = time.time() - start_time
        total_time += parse_time
        
        print(f"{i:2d}. '{test_case}'")
        print(f"    -> {result.action_type}({result.target}) [{result.confidence:.2f}]")
        print(f"    -> {result.pattern_used} ({parse_time*1000:.2f}ms)")
        print()
    
    print(f"Total parsing time: {total_time*1000:.2f}ms")
    print(f"Average per parse: {(total_time/len(test_cases))*1000:.2f}ms")
    
    # Show performance statistics
    print("\n--- Performance Statistics ---")
    stats = parser.get_performance_stats()
    
    print(f"Total parses: {stats['total_parses']}")
    print(f"Fast path hits: {stats['fast_path_hits']} ({stats['fast_path_rate_percent']:.1f}%)")
    print(f"Cache hits: {stats['cache_hits']} ({stats['cache_hit_rate_percent']:.1f}%)")
    print(f"Regex parses: {stats['regex_parses']} ({stats['regex_rate_percent']:.1f}%)")
    print(f"Cache size: {stats['cache_size']}")
    
    # Show pattern performance
    print("\n--- Top Pattern Performance ---")
    pattern_stats = stats['pattern_statistics']
    sorted_patterns = sorted(
        pattern_stats.items(),
        key=lambda x: x[1]['usage_count'],
        reverse=True
    )
    
    for name, pstats in sorted_patterns[:5]:
        print(f"{name}: {pstats['usage_count']} uses, {pstats['success_rate_percent']:.1f}% success, {pstats['average_time_ms']:.2f}ms avg")
    
    # Show optimization recommendations
    recommendations = parser.optimize_patterns()
    if recommendations:
        print("\n--- Optimization Recommendations ---")
        for rec in recommendations:
            print(f"- {rec}")
    else:
        print("\n--- No optimization recommendations at this time ---")


if __name__ == "__main__":
    demo_action_parser_optimizer()
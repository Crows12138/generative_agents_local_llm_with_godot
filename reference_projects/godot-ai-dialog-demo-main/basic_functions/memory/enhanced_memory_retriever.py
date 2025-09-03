"""
Enhanced memory retrieval system with advanced search capabilities.
This module provides sophisticated memory retrieval with multiple search strategies.
"""

import time
import math
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from basic_functions.memory.memory import Memory, MemoryEntry, MemoryType
from basic_functions.perception.embedding import get_single_embedding
from .importance_scorer import importance_scorer

class EnhancedMemoryRetriever:
    """
    Enhanced memory retrieval system with multiple search strategies.
    """
    
    def __init__(self, memory: Memory):
        self.memory = memory
        self.retrieval_cache = {}  # Cache for retrieval results
        self.cache_ttl = 300  # 5 minutes cache TTL
    
    def retrieve_contextual_memories(
        self,
        query: str,
        persona_name: str,
        current_location: Any = None,
        current_time: str = "",
        top_k: int = 5,
        memory_types: Optional[List[MemoryType]] = None,
        time_window_hours: Optional[int] = None,
        min_importance: float = 0.0
    ) -> List[MemoryEntry]:
        """
        Retrieve memories using multiple contextual factors.
        
        Args:
            query: Text query for semantic search
            persona_name: Name of the persona
            current_location: Current location for spatial relevance
            current_time: Current time for temporal relevance
            top_k: Number of memories to retrieve
            memory_types: Filter by memory types
            time_window_hours: Only consider memories from last N hours
            min_importance: Minimum importance score
            
        Returns:
            List of relevant memory entries
        """
        # Generate query embedding
        query_embedding = get_single_embedding(query)
        
        # Get base semantic results
        semantic_results = self.memory.retrieve_similar(
            query_emb=query_embedding,
            top_k=top_k * 2,  # Get more for filtering
            filter_memory_type=memory_types[0] if memory_types and len(memory_types) == 1 else None
        )
        
        # Apply filters and scoring
        filtered_results = self._apply_filters(
            semantic_results, 
            memory_types, 
            time_window_hours, 
            min_importance
        )
        
        # Re-score with contextual factors
        scored_results = self._score_with_context(
            filtered_results,
            query,
            persona_name,
            current_location,
            current_time
        )
        
        # Return top results
        return scored_results[:top_k]
    
    def retrieve_by_emotional_context(
        self,
        emotion_keywords: List[str],
        persona_name: str,
        top_k: int = 5,
        time_window_hours: int = 24
    ) -> List[MemoryEntry]:
        """
        Retrieve memories based on emotional context.
        
        Args:
            emotion_keywords: List of emotion-related keywords
            persona_name: Name of the persona
            top_k: Number of memories to retrieve
            time_window_hours: Time window for search
            
        Returns:
            List of emotionally relevant memories
        """
        # Create emotion-based query
        emotion_query = f"memories about {', '.join(emotion_keywords)}"
        
        return self.retrieve_contextual_memories(
            query=emotion_query,
            persona_name=persona_name,
            top_k=top_k,
            time_window_hours=time_window_hours
        )
    
    def retrieve_by_relationship(
        self,
        target_name: str,
        persona_name: str,
        top_k: int = 5,
        time_window_hours: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Retrieve memories related to a specific person or object.
        
        Args:
            target_name: Name of the person or object
            persona_name: Name of the persona
            top_k: Number of memories to retrieve
            time_window_hours: Time window for search
            
        Returns:
            List of relationship-related memories
        """
        relationship_query = f"memories about {target_name} and interactions with {target_name}"
        
        return self.retrieve_contextual_memories(
            query=relationship_query,
            persona_name=persona_name,
            top_k=top_k,
            time_window_hours=time_window_hours,
            memory_types=[MemoryType.INTERACTION, MemoryType.SIGHT]
        )
    
    def retrieve_by_location(
        self,
        location: Any,
        persona_name: str,
        top_k: int = 5,
        time_window_hours: Optional[int] = None
    ) -> List[MemoryEntry]:
        """
        Retrieve memories related to a specific location.
        
        Args:
            location: Location to search for
            persona_name: Name of the persona
            top_k: Number of memories to retrieve
            time_window_hours: Time window for search
            
        Returns:
            List of location-related memories
        """
        location_query = f"memories about location {location} and experiences at {location}"
        
        return self.retrieve_contextual_memories(
            query=location_query,
            persona_name=persona_name,
            top_k=top_k,
            time_window_hours=time_window_hours
        )
    
    def retrieve_recent_important_memories(
        self,
        persona_name: str,
        top_k: int = 5,
        hours: int = 24
    ) -> List[MemoryEntry]:
        """
        Retrieve recent memories with high importance scores.
        
        Args:
            persona_name: Name of the persona
            top_k: Number of memories to retrieve
            hours: Time window in hours
            
        Returns:
            List of recent important memories
        """
        # Get recent memories
        recent_memories = self.memory.get_recent_memories(hours=hours)
        
        # Score by importance
        scored_memories = []
        for memory in recent_memories:
            # Use cached importance or calculate new one
            importance = memory.importance
            if importance <= 0:
                importance = importance_scorer.score_event_importance(
                    memory.text, persona_name, memory.memory_type
                )
            
            scored_memories.append((memory, importance))
        
        # Sort by importance and return top results
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in scored_memories[:top_k]]
    
    def retrieve_pattern_memories(
        self,
        pattern_description: str,
        persona_name: str,
        top_k: int = 5
    ) -> List[MemoryEntry]:
        """
        Retrieve memories that match a behavioral pattern.
        
        Args:
            pattern_description: Description of the pattern to search for
            persona_name: Name of the persona
            top_k: Number of memories to retrieve
            
        Returns:
            List of pattern-related memories
        """
        pattern_query = f"memories showing pattern: {pattern_description}"
        
        return self.retrieve_contextual_memories(
            query=pattern_query,
            persona_name=persona_name,
            top_k=top_k
        )
    
    def _apply_filters(
        self,
        memories: List[MemoryEntry],
        memory_types: Optional[List[MemoryType]] = None,
        time_window_hours: Optional[int] = None,
        min_importance: float = 0.0
    ) -> List[MemoryEntry]:
        """Apply filters to memory results."""
        filtered = memories
        
        # Filter by memory types
        if memory_types:
            filtered = [m for m in filtered if m.memory_type in memory_types]
        
        # Filter by time window
        if time_window_hours:
            cutoff_time = time.time() - (time_window_hours * 3600)
            filtered = [m for m in filtered if m.timestamp >= cutoff_time]
        
        # Filter by importance
        if min_importance > 0:
            filtered = [m for m in filtered if m.importance >= min_importance]
        
        return filtered
    
    def _score_with_context(
        self,
        memories: List[MemoryEntry],
        query: str,
        persona_name: str,
        current_location: Any = None,
        current_time: str = ""
    ) -> List[MemoryEntry]:
        """Score memories with contextual factors."""
        scored_memories = []
        
        for memory in memories:
            # Base score from semantic similarity (already in memory.importance)
            base_score = memory.importance
            
            # Temporal recency bonus
            time_diff_hours = (time.time() - memory.timestamp) / 3600
            recency_bonus = max(0, 1.0 - (time_diff_hours / 24))  # Decay over 24 hours
            
            # Spatial relevance bonus
            spatial_bonus = 0.0
            if current_location and memory.location:
                # Simple distance-based bonus (can be enhanced)
                if memory.location == current_location:
                    spatial_bonus = 0.5
            
            # Frequency bonus (memories that occur more often)
            frequency_bonus = math.log(memory.metadata.get("freq", 1) + 1) * 0.1
            
            # Calculate final score
            final_score = base_score + recency_bonus + spatial_bonus + frequency_bonus
            
            scored_memories.append((memory, final_score))
        
        # Sort by final score
        scored_memories.sort(key=lambda x: x[1], reverse=True)
        return [memory for memory, _ in scored_memories]
    
    def get_memory_summary(
        self,
        persona_name: str,
        time_window_hours: int = 24,
        max_memories: int = 20
    ) -> str:
        """
        Generate a summary of recent memories.
        
        Args:
            persona_name: Name of the persona
            time_window_hours: Time window for summary
            max_memories: Maximum number of memories to include
            
        Returns:
            Formatted memory summary
        """
        recent_memories = self.retrieve_recent_important_memories(
            persona_name, top_k=max_memories, hours=time_window_hours
        )
        
        if not recent_memories:
            return f"No significant memories in the last {time_window_hours} hours."
        
        summary = f"Memory Summary (Last {time_window_hours} hours):\n"
        summary += "=" * 50 + "\n"
        
        for i, memory in enumerate(recent_memories, 1):
            importance_stars = "★" * int(memory.importance)
            summary += f"{i}. [{memory.memory_type.value.upper()}] {memory.text}\n"
            summary += f"   Importance: {memory.importance:.1f} {importance_stars}\n"
            summary += f"   Time: {memory.get_formatted_time()}\n\n"
        
        return summary
    
    def clear_cache(self):
        """Clear the retrieval cache."""
        self.retrieval_cache.clear() 
"""
Integrated Enhanced Memory System
Combines existing memory functionality with enhanced features from generative_agents_main.
"""

import json
import time
import math
import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict

# Import existing memory types
from .memory import MemoryType as BaseMemoryType, MemoryEntry as BaseMemoryEntry

class EnhancedMemoryType(Enum):
    """Extended memory types combining existing and new types."""
    # Existing types
    SIGHT = "sight"
    ACTION = "action"
    INTERACTION = "interaction"
    REFLECTION = "reflection"
    
    # New enhanced types
    EVENT = "event"
    THOUGHT = "thought"
    CHAT = "chat"
    SOCIAL = "social"
    PLANNING = "planning"

@dataclass
class EnhancedMemoryEntry:
    """Enhanced memory entry with SPO structure and advanced features."""
    # Core fields (compatible with existing MemoryEntry)
    timestamp: float
    text: str
    embedding: List[float]
    location: Any = None
    event_type: str = ""
    importance: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
    expiry_timestamp: Optional[float] = None
    memory_type: EnhancedMemoryType = EnhancedMemoryType.SIGHT
    
    # Enhanced fields (from generative_agents_main)
    node_id: str = ""
    subject: str = ""
    predicate: str = ""
    object: str = ""
    keywords: List[str] = None
    access_count: int = 0
    last_accessed: float = 0.0
    
    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []
        if self.metadata is None:
            self.metadata = {}
        if not self.node_id:
            self.node_id = f"node_{int(self.timestamp * 1000)}"
        if not self.last_accessed:
            self.last_accessed = self.timestamp
    
    def spo_summary(self) -> Tuple[str, str, str]:
        """Get subject-predicate-object summary."""
        return (self.subject, self.predicate, self.object)
    
    def get_age_hours(self) -> float:
        """Get age of memory in hours."""
        return (time.time() - self.timestamp) / 3600
    
    def get_recency_score(self) -> float:
        """Calculate recency score (higher = more recent)."""
        age_hours = self.get_age_hours()
        return math.exp(-age_hours / 24)  # Decay over 24 hours
    
    def get_importance_score(self) -> float:
        """Calculate importance score."""
        return self.importance
    
    def get_relevance_score(self, query_embedding: List[float]) -> float:
        """Calculate relevance score based on embedding similarity."""
        if not self.embedding or not query_embedding:
            return 0.0
        
        # Cosine similarity
        dot_product = sum(a * b for a, b in zip(self.embedding, query_embedding))
        magnitude_a = math.sqrt(sum(a * a for a in self.embedding))
        magnitude_b = math.sqrt(sum(b * b for b in query_embedding))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)
    
    def is_expired(self) -> bool:
        """Check if memory is expired."""
        if self.expiry_timestamp is None:
            return False
        return time.time() > self.expiry_timestamp
    
    def get_formatted_time(self) -> str:
        """Return formatted timestamp for display."""
        return datetime.datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")
    
    def __str__(self) -> str:
        return f"[{self.memory_type.value}] {self.get_formatted_time()}: {self.text}"
    
    def to_base_memory_entry(self) -> BaseMemoryEntry:
        """Convert to base memory entry for compatibility."""
        return BaseMemoryEntry(
            timestamp=self.timestamp,
            text=self.text,
            embedding=self.embedding,
            location=self.location,
            event_type=self.event_type,
            importance=self.importance,
            metadata=self.metadata,
            expiry_timestamp=self.expiry_timestamp,
            memory_type=BaseMemoryType(self.memory_type.value)
        )

class IntegratedEnhancedMemory:
    """
    Integrated memory system that combines existing functionality with enhanced features.
    Maintains backward compatibility while adding advanced capabilities.
    """
    
    def __init__(self):
        # Core memory storage
        self.entries: List[EnhancedMemoryEntry] = []
        self.node_counter = 0
        
        # Enhanced indexes for faster retrieval
        self.type_index: Dict[EnhancedMemoryType, List[str]] = defaultdict(list)
        self.keyword_index: Dict[str, List[str]] = defaultdict(list)
        self.location_index: Dict[Tuple[int, int], List[str]] = defaultdict(list)
        self.subject_index: Dict[str, List[str]] = defaultdict(list)
        
        # Backward compatibility
        self._base_memory = None  # Will be created on demand
    
    def add(
        self,
        text: str,
        embedding: List[float],
        location: Any = None,
        event_type: str = "",
        importance: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None,
        memory_type: EnhancedMemoryType = EnhancedMemoryType.SIGHT,
        persona_name: str = "",
        auto_score_importance: bool = True,
        # Enhanced parameters
        subject: str = "",
        predicate: str = "",
        object: str = "",
        keywords: List[str] = None,
    ) -> str:
        """Add a new memory entry with enhanced features."""
        
        now = time.time()
        expiry = now + ttl if ttl is not None else None
        
        # Auto-score importance if enabled
        if auto_score_importance and importance <= 0 and persona_name:
            try:
                from .importance_scorer import importance_scorer
                importance = importance_scorer.score_event_importance(
                    text, persona_name, memory_type.value
                )
            except Exception as e:
                print(f"Warning: Failed to auto-score importance: {e}")
                importance = self._get_default_importance(memory_type)
        
        # Generate node ID
        self.node_counter += 1
        node_id = f"node_{self.node_counter}"
        
        # Create enhanced memory entry
        entry = EnhancedMemoryEntry(
            timestamp=now,
            text=text,
            embedding=embedding,
            location=location,
            event_type=event_type,
            importance=importance,
            metadata=metadata or {},
            expiry_timestamp=expiry,
            memory_type=memory_type,
            node_id=node_id,
            subject=subject or persona_name,
            predicate=predicate or event_type,
            object=object or text[:50],
            keywords=keywords or [],
            last_accessed=now
        )
        
        self.entries.append(entry)
        
        # Update indexes
        self._update_indexes(entry)
        
        return node_id
    
    def _update_indexes(self, entry: EnhancedMemoryEntry):
        """Update all indexes for a memory entry."""
        # Type index
        self.type_index[entry.memory_type].append(entry.node_id)
        
        # Keyword index
        for keyword in entry.keywords:
            self.keyword_index[keyword].append(entry.node_id)
        
        # Location index
        if entry.location:
            if isinstance(entry.location, (list, tuple)) and len(entry.location) >= 2:
                loc_key = (int(entry.location[0]), int(entry.location[1]))
                self.location_index[loc_key].append(entry.node_id)
        
        # Subject index
        if entry.subject:
            self.subject_index[entry.subject].append(entry.node_id)
    
    def retrieve_similar(
        self,
        query_emb: List[float],
        top_k: int = 5,
        filter_event: Optional[str] = None,
        filter_memory_type: Optional[EnhancedMemoryType] = None,
        # Enhanced filters
        keywords: List[str] = None,
        location: Tuple[float, float, float] = None,
        subject: str = None,
        min_importance: float = 0.0,
    ) -> List[EnhancedMemoryEntry]:
        """Retrieve similar memories with enhanced filtering."""
        
        # Remove expired entries
        self.cleanup_expired()
        
        # Apply filters
        candidates = self.entries
        
        if filter_memory_type:
            candidates = [e for e in candidates if e.memory_type == filter_memory_type]
        
        if filter_event:
            candidates = [e for e in candidates if e.event_type == filter_event]
        
        if keywords:
            keyword_matches = set()
            for keyword in keywords:
                if keyword in self.keyword_index:
                    keyword_matches.update(self.keyword_index[keyword])
            candidates = [e for e in candidates if e.node_id in keyword_matches]
        
        if location:
            loc_key = (int(location[0]), int(location[1]))
            if loc_key in self.location_index:
                location_matches = set(self.location_index[loc_key])
                candidates = [e for e in candidates if e.node_id in location_matches]
        
        if subject:
            if subject in self.subject_index:
                subject_matches = set(self.subject_index[subject])
                candidates = [e for e in candidates if e.node_id in subject_matches]
        
        if min_importance > 0:
            candidates = [e for e in candidates if e.importance >= min_importance]
        
        # Calculate similarity scores
        scored_candidates = []
        for entry in candidates:
            score = entry.get_relevance_score(query_emb)
            scored_candidates.append((entry, score))
        
        # Sort by score and return top results
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, score in scored_candidates[:top_k]]
    
    def retrieve_relevant_memories(
        self,
        query: str = None,
        memory_types: List[EnhancedMemoryType] = None,
        keywords: List[str] = None,
        location: Tuple[float, float, float] = None,
        max_results: int = 10,
        min_importance: float = 0.0,
    ) -> List[EnhancedMemoryEntry]:
        """Retrieve relevant memories based on multiple criteria."""
        
        candidates = []
        
        # Filter by memory type
        if memory_types:
            for mem_type in memory_types:
                candidates.extend([e for e in self.entries if e.memory_type == mem_type])
        else:
            candidates = self.entries.copy()
        
        # Apply other filters
        if min_importance > 0:
            candidates = [e for e in candidates if e.importance >= min_importance]
        
        if keywords:
            keyword_matches = set()
            for keyword in keywords:
                if keyword in self.keyword_index:
                    keyword_matches.update(self.keyword_index[keyword])
            candidates = [e for e in candidates if e.node_id in keyword_matches]
        
        if location:
            loc_key = (int(location[0]), int(location[1]))
            if loc_key in self.location_index:
                location_matches = set(self.location_index[loc_key])
                candidates = [e for e in candidates if e.node_id in location_matches]
        
        # Calculate relevance scores
        scored_candidates = []
        for entry in candidates:
            score = self._calculate_relevance_score(entry, query, location)
            scored_candidates.append((entry, score))
        
        # Sort by relevance score and return top results
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, score in scored_candidates[:max_results]]
    
    def _calculate_relevance_score(
        self,
        entry: EnhancedMemoryEntry,
        query: str = None,
        location: Tuple[float, float, float] = None,
    ) -> float:
        """Calculate overall relevance score for a memory entry."""
        
        # Base score from importance and recency
        base_score = entry.get_importance_score() * 0.4 + entry.get_recency_score() * 0.3
        
        # Location relevance
        location_score = 0.0
        if location and entry.location:
            if isinstance(entry.location, (list, tuple)) and len(entry.location) >= 2:
                distance = math.sqrt(
                    (location[0] - entry.location[0])**2 + 
                    (location[1] - entry.location[1])**2
                )
                location_score = math.exp(-distance / 10.0)  # Decay over 10 units
                base_score += location_score * 0.2
        
        # Access frequency bonus
        access_bonus = min(entry.access_count * 0.05, 0.1)  # Max 10% bonus
        base_score += access_bonus
        
        return base_score
    
    # Backward compatibility methods
    def add_sight(self, text: str, embedding: List[float], location: Any = None, ttl: Optional[float] = None, persona_name: str = "") -> str:
        return self.add(text, embedding, location, "sight", ttl=ttl, memory_type=EnhancedMemoryType.SIGHT, persona_name=persona_name)
    
    def add_action(self, text: str, embedding: List[float], location: Any = None, ttl: Optional[float] = None, persona_name: str = "") -> str:
        return self.add(text, embedding, location, "action", ttl=ttl, memory_type=EnhancedMemoryType.ACTION, persona_name=persona_name)
    
    def get_memories_by_type(self, memory_type: EnhancedMemoryType) -> List[EnhancedMemoryEntry]:
        return [e for e in self.entries if e.memory_type == memory_type]
    
    def get_recent_memories(self, hours: int = 24) -> List[EnhancedMemoryEntry]:
        cutoff_time = time.time() - hours * 3600
        return [e for e in self.entries if e.timestamp >= cutoff_time]
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed entries."""
        original_count = len(self.entries)
        self.entries = [e for e in self.entries if not e.is_expired()]
        removed_count = original_count - len(self.entries)
        
        # Rebuild indexes after cleanup
        if removed_count > 0:
            self._rebuild_indexes()
        
        return removed_count
    
    def _rebuild_indexes(self):
        """Rebuild all indexes after cleanup."""
        # Clear all indexes
        for index in self.type_index.values():
            index.clear()
        self.keyword_index.clear()
        self.location_index.clear()
        self.subject_index.clear()
        
        # Rebuild indexes
        for entry in self.entries:
            self._update_indexes(entry)
    
    def _get_default_importance(self, memory_type: EnhancedMemoryType) -> float:
        """Get default importance for memory type."""
        defaults = {
            EnhancedMemoryType.SIGHT: 0.3,
            EnhancedMemoryType.ACTION: 0.5,
            EnhancedMemoryType.INTERACTION: 0.7,
            EnhancedMemoryType.REFLECTION: 0.8,
            EnhancedMemoryType.EVENT: 0.6,
            EnhancedMemoryType.THOUGHT: 0.6,
            EnhancedMemoryType.CHAT: 0.7,
            EnhancedMemoryType.SOCIAL: 0.8,
            EnhancedMemoryType.PLANNING: 0.5,
        }
        return defaults.get(memory_type, 0.5)
    
    def get_memory_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get a summary of recent memories."""
        recent_memories = self.get_recent_memories(hours)
        
        summary = {
            "total_memories": len(recent_memories),
            "by_type": {},
            "most_important": [],
            "recent_events": [],
            "recent_thoughts": []
        }
        
        for mem_type in EnhancedMemoryType:
            type_memories = [mem for mem in recent_memories if mem.memory_type == mem_type]
            summary["by_type"][mem_type.value] = len(type_memories)
        
        # Get most important memories
        important_memories = sorted(recent_memories, key=lambda x: x.importance, reverse=True)[:5]
        summary["most_important"] = [
            {
                "description": mem.text,
                "importance": mem.importance,
                "type": mem.memory_type.value,
                "age_hours": mem.get_age_hours()
            }
            for mem in important_memories
        ]
        
        # Get recent events and thoughts
        recent_events = [mem for mem in recent_memories if mem.memory_type in [EnhancedMemoryType.EVENT, EnhancedMemoryType.ACTION]][:3]
        recent_thoughts = [mem for mem in recent_memories if mem.memory_type in [EnhancedMemoryType.THOUGHT, EnhancedMemoryType.REFLECTION]][:3]
        
        summary["recent_events"] = [mem.text for mem in recent_events]
        summary["recent_thoughts"] = [mem.text for mem in recent_thoughts]
        
        return summary
    
    def save(self, filepath: str):
        """Save memory to file."""
        data = {
            "node_counter": self.node_counter,
            "entries": [asdict(entry) for entry in self.entries]
        }
        
        # Convert datetime objects to strings
        for entry_data in data["entries"]:
            entry_data["memory_type"] = entry_data["memory_type"].value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str):
        """Load memory from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.node_counter = data["node_counter"]
        self.entries.clear()
        
        # Clear indexes
        for index in self.type_index.values():
            index.clear()
        self.keyword_index.clear()
        self.location_index.clear()
        self.subject_index.clear()
        
        for entry_data in data["entries"]:
            entry_data["memory_type"] = EnhancedMemoryType(entry_data["memory_type"])
            
            entry = EnhancedMemoryEntry(**entry_data)
            self.entries.append(entry)
            
            # Rebuild indexes
            self._update_indexes(entry)
    
    # Property for backward compatibility
    @property
    def base_memory(self):
        """Get base memory object for backward compatibility."""
        if self._base_memory is None:
            from .memory import Memory
            self._base_memory = Memory()
            # Convert entries to base format
            for entry in self.entries:
                self._base_memory.entries.append(entry.to_base_memory_entry())
        return self._base_memory 
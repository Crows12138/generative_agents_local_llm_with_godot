"""
Enhanced Memory System inspired by Generative Agents
Provides better memory structures and retrieval mechanisms.
"""

import json
import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import math

class MemoryType(Enum):
    EVENT = "event"
    THOUGHT = "thought"
    CHAT = "chat"
    REFLECTION = "reflection"

@dataclass
class MemoryNode:
    """Enhanced memory node with more detailed information."""
    node_id: str
    memory_type: MemoryType
    created: datetime.datetime
    expiration: Optional[datetime.datetime]
    last_accessed: datetime.datetime
    
    # Subject-Predicate-Object structure
    subject: str
    predicate: str
    object: str
    
    # Content and metadata
    description: str
    importance: float  # 0.0 to 1.0
    keywords: List[str]
    location: Tuple[float, float, float]
    
    # Embedding and retrieval
    embedding: List[float]
    access_count: int = 0
    
    def spo_summary(self) -> Tuple[str, str, str]:
        """Get subject-predicate-object summary."""
        return (self.subject, self.predicate, self.object)
    
    def get_age_hours(self) -> float:
        """Get age of memory in hours."""
        return (datetime.datetime.now() - self.created).total_seconds() / 3600
    
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
        
        # Simple cosine similarity
        dot_product = sum(a * b for a, b in zip(self.embedding, query_embedding))
        magnitude_a = math.sqrt(sum(a * a for a in self.embedding))
        magnitude_b = math.sqrt(sum(b * b for b in query_embedding))
        
        if magnitude_a == 0 or magnitude_b == 0:
            return 0.0
        
        return dot_product / (magnitude_a * magnitude_b)

class EnhancedMemory:
    """Enhanced memory system with better retrieval and organization."""
    
    def __init__(self):
        self.memories: Dict[str, MemoryNode] = {}
        self.node_counter = 0
        
        # Indexes for faster retrieval
        self.type_index: Dict[MemoryType, List[str]] = {
            MemoryType.EVENT: [],
            MemoryType.THOUGHT: [],
            MemoryType.CHAT: [],
            MemoryType.REFLECTION: []
        }
        self.keyword_index: Dict[str, List[str]] = {}
        self.location_index: Dict[Tuple[int, int], List[str]] = {}
    
    def add_memory(self, 
                   memory_type: MemoryType,
                   subject: str,
                   predicate: str,
                   object: str,
                   description: str,
                   importance: float = 0.5,
                   keywords: List[str] = None,
                   location: Tuple[float, float, float] = None,
                   embedding: List[float] = None,
                   expiration: Optional[datetime.datetime] = None) -> str:
        """Add a new memory node."""
        
        self.node_counter += 1
        node_id = f"node_{self.node_counter}"
        
        now = datetime.datetime.now()
        
        memory_node = MemoryNode(
            node_id=node_id,
            memory_type=memory_type,
            created=now,
            expiration=expiration,
            last_accessed=now,
            subject=subject,
            predicate=predicate,
            object=object,
            description=description,
            importance=importance,
            keywords=keywords or [],
            location=location or (0.0, 0.0, 0.0),
            embedding=embedding or []
        )
        
        self.memories[node_id] = memory_node
        
        # Update indexes
        self.type_index[memory_type].append(node_id)
        
        for keyword in memory_node.keywords:
            if keyword not in self.keyword_index:
                self.keyword_index[keyword] = []
            self.keyword_index[keyword].append(node_id)
        
        if location:
            loc_key = (int(location[0]), int(location[1]))
            if loc_key not in self.location_index:
                self.location_index[loc_key] = []
            self.location_index[loc_key].append(node_id)
        
        return node_id
    
    def retrieve_relevant_memories(self, 
                                 query: str = None,
                                 memory_types: List[MemoryType] = None,
                                 keywords: List[str] = None,
                                 location: Tuple[float, float, float] = None,
                                 max_results: int = 10,
                                 min_importance: float = 0.0) -> List[MemoryNode]:
        """Retrieve relevant memories based on multiple criteria."""
        
        candidates = []
        
        # Filter by memory type
        if memory_types:
            for mem_type in memory_types:
                candidates.extend([self.memories[node_id] for node_id in self.type_index[mem_type]])
        else:
            candidates = list(self.memories.values())
        
        # Filter by importance
        candidates = [mem for mem in candidates if mem.importance >= min_importance]
        
        # Filter by keywords
        if keywords:
            keyword_matches = []
            for keyword in keywords:
                if keyword in self.keyword_index:
                    keyword_matches.extend(self.keyword_index[keyword])
            keyword_matches = set(keyword_matches)
            candidates = [mem for mem in candidates if mem.node_id in keyword_matches]
        
        # Filter by location
        if location:
            loc_key = (int(location[0]), int(location[1]))
            if loc_key in self.location_index:
                location_matches = set(self.location_index[loc_key])
                candidates = [mem for mem in candidates if mem.node_id in location_matches]
        
        # Calculate relevance scores
        scored_candidates = []
        for memory in candidates:
            score = self._calculate_relevance_score(memory, query, location)
            scored_candidates.append((memory, score))
        
        # Sort by relevance score and return top results
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return [mem for mem, score in scored_candidates[:max_results]]
    
    def _calculate_relevance_score(self, 
                                 memory: MemoryNode, 
                                 query: str = None,
                                 location: Tuple[float, float, float] = None) -> float:
        """Calculate overall relevance score for a memory."""
        
        # Base score from importance and recency
        base_score = memory.get_importance_score() * 0.4 + memory.get_recency_score() * 0.3
        
        # Location relevance
        location_score = 0.0
        if location and memory.location:
            distance = math.sqrt(
                (location[0] - memory.location[0])**2 + 
                (location[1] - memory.location[1])**2
            )
            location_score = math.exp(-distance / 10.0)  # Decay over 10 units
            base_score += location_score * 0.2
        
        # Access frequency bonus
        access_bonus = min(memory.access_count * 0.05, 0.1)  # Max 10% bonus
        base_score += access_bonus
        
        return base_score
    
    def get_recent_memories(self, hours: int = 24, memory_types: List[MemoryType] = None) -> List[MemoryNode]:
        """Get memories from the last N hours."""
        cutoff_time = datetime.datetime.now() - datetime.timedelta(hours=hours)
        
        candidates = []
        if memory_types:
            for mem_type in memory_types:
                candidates.extend([self.memories[node_id] for node_id in self.type_index[mem_type]])
        else:
            candidates = list(self.memories.values())
        
        return [mem for mem in candidates if mem.created >= cutoff_time]
    
    def get_memories_by_type(self, memory_type: MemoryType) -> List[MemoryNode]:
        """Get all memories of a specific type."""
        return [self.memories[node_id] for node_id in self.type_index[memory_type]]
    
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
        
        for mem_type in MemoryType:
            type_memories = [mem for mem in recent_memories if mem.memory_type == mem_type]
            summary["by_type"][mem_type.value] = len(type_memories)
        
        # Get most important memories
        important_memories = sorted(recent_memories, key=lambda x: x.importance, reverse=True)[:5]
        summary["most_important"] = [
            {
                "description": mem.description,
                "importance": mem.importance,
                "type": mem.memory_type.value,
                "age_hours": mem.get_age_hours()
            }
            for mem in important_memories
        ]
        
        # Get recent events and thoughts
        recent_events = [mem for mem in recent_memories if mem.memory_type == MemoryType.EVENT][:3]
        recent_thoughts = [mem for mem in recent_memories if mem.memory_type == MemoryType.THOUGHT][:3]
        
        summary["recent_events"] = [mem.description for mem in recent_events]
        summary["recent_thoughts"] = [mem.description for mem in recent_thoughts]
        
        return summary
    
    def save(self, filepath: str):
        """Save memory to file."""
        data = {
            "node_counter": self.node_counter,
            "memories": {node_id: asdict(memory) for node_id, memory in self.memories.items()}
        }
        
        # Convert datetime objects to strings
        for memory_data in data["memories"].values():
            memory_data["created"] = memory_data["created"].isoformat()
            memory_data["last_accessed"] = memory_data["last_accessed"].isoformat()
            if memory_data["expiration"]:
                memory_data["expiration"] = memory_data["expiration"].isoformat()
            memory_data["memory_type"] = memory_data["memory_type"].value
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    
    def load(self, filepath: str):
        """Load memory from file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.node_counter = data["node_counter"]
        self.memories.clear()
        
        # Clear indexes
        for index in self.type_index.values():
            index.clear()
        self.keyword_index.clear()
        self.location_index.clear()
        
        for node_id, memory_data in data["memories"].items():
            # Convert strings back to datetime objects
            memory_data["created"] = datetime.datetime.fromisoformat(memory_data["created"])
            memory_data["last_accessed"] = datetime.datetime.fromisoformat(memory_data["last_accessed"])
            if memory_data["expiration"]:
                memory_data["expiration"] = datetime.datetime.fromisoformat(memory_data["expiration"])
            memory_data["memory_type"] = MemoryType(memory_data["memory_type"])
            
            memory_node = MemoryNode(**memory_data)
            self.memories[node_id] = memory_node
            
            # Rebuild indexes
            self.type_index[memory_node.memory_type].append(node_id)
            
            for keyword in memory_node.keywords:
                if keyword not in self.keyword_index:
                    self.keyword_index[keyword] = []
                self.keyword_index[keyword].append(node_id)
            
            if memory_node.location:
                loc_key = (int(memory_node.location[0]), int(memory_node.location[1]))
                if loc_key not in self.location_index:
                    self.location_index[loc_key] = []
                self.location_index[loc_key].append(node_id) 
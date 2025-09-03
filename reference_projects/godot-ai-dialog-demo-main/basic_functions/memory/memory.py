import time
import math
from typing import List, Dict, Any, Optional
from datetime import datetime
from enum import Enum

class MemoryType(Enum):
    SIGHT = "sight"
    ACTION = "action"
    INTERACTION = "interaction"
    REFLECTION = "reflection"

class MemoryEntry:
    def __init__(
        self,
        timestamp: float,
        text: str,
        embedding: List[float],
        location: Any = None,
        event_type: str = "",
        importance: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        expiry_timestamp: Optional[float] = None,
        memory_type: MemoryType = MemoryType.SIGHT,
    ):
        self.timestamp = timestamp  # Unix timestamp in seconds
        self.text = text            # The text content of the memory entry
        self.embedding = embedding  # Embedding vector as a plain list
        self.location = location    # Optional location information
        self.event_type = event_type  # Category or type of event
        self.importance = importance  # Importance weight for retrieval
        self.metadata = metadata if metadata is not None else {}
        self.expiry_timestamp = expiry_timestamp  # None means no expiration
        self.memory_type = memory_type  # Type of memory (sight, action, etc.)

    def is_expired(self) -> bool:
        if self.expiry_timestamp is None:
            return False
        return time.time() > self.expiry_timestamp

    def get_formatted_time(self) -> str:
        """Return formatted timestamp for display"""
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S")

    def __str__(self) -> str:
        return f"[{self.memory_type.value}] {self.get_formatted_time()}: {self.text}"

class Memory:
    """
    A simple memory system that stores text entries with their embeddings and timestamps.
    It allows adding new entries, retrieving similar entries based on cosine similarity,
    and clearing the memory.
    """
    def __init__(self):
        self.entries: List[MemoryEntry] = []

    def add(
        self,
        text: str,
        embedding: List[float],
        location: Any = None,
        event_type: str = "",
        importance: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
        ttl: Optional[float] = None,
        memory_type: MemoryType = MemoryType.SIGHT,
        persona_name: str = "",
        auto_score_importance: bool = True,
    ) -> None:
        now = time.time()
        expiry = now + ttl if ttl is not None else None

        existing = next((e for e in self.entries if e.text == text), None)
        if existing is not None:
            existing.timestamp = now
            existing.expiry_timestamp = expiry
            freq = existing.metadata.get("freq", 1) + 1
            existing.metadata["freq"] = freq
            existing.importance = math.log(freq + 1)
            # move to end to keep most recent ordering
            self.entries.remove(existing)
            self.entries.append(existing)
        else:
            # Auto-score importance if enabled and no importance provided
            if auto_score_importance and importance <= 0 and persona_name:
                try:
                    from .importance_scorer import importance_scorer
                    importance = importance_scorer.score_event_importance(
                        text, persona_name, memory_type
                    )
                except Exception as e:
                    print(f"Warning: Failed to auto-score importance: {e}")
                    # Require AI to generate importance
                    raise RuntimeError(f"Failed to get importance score for {memory_type}: {e}")
            
            entry = MemoryEntry(
                timestamp=now,
                text=text,
                embedding=embedding,
                location=location,
                event_type=event_type,
                importance=importance,
                metadata=metadata if metadata is not None else {"freq": 1},
                expiry_timestamp=expiry,
                memory_type=memory_type,
            )
            # ensure frequency counter exists
            entry.metadata.setdefault("freq", 1)
            self.entries.append(entry)

    def add_sight(self, text: str, embedding: List[float], location: Any = None, ttl: Optional[float] = None, persona_name: str = "") -> None:
        """Add a sight memory entry"""
        self.add(text, embedding, location, event_type="sight", memory_type=MemoryType.SIGHT, ttl=ttl, persona_name=persona_name)

    def add_action(self, text: str, embedding: List[float], location: Any = None, ttl: Optional[float] = None, persona_name: str = "") -> None:
        """Add an action memory entry"""
        self.add(text, embedding, location, event_type="action", memory_type=MemoryType.ACTION, ttl=ttl, persona_name=persona_name)

    def retrieve_similar(
        self,
        query_emb: List[float],
        top_k: int = 5,
        filter_event: Optional[str] = None,
        filter_memory_type: Optional[MemoryType] = None,
    ) -> List[MemoryEntry]:
        # Remove expired entries
        self.entries = [e for e in self.entries if not e.is_expired()]

        def dot(a: List[float], b: List[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        def norm(a: List[float]) -> float:
            return math.sqrt(sum(x * x for x in a))

        q_norm = norm(query_emb)
        sims: List[float] = []
        for e in self.entries:
            if filter_event and e.event_type != filter_event:
                sims.append(-1.0)
                continue
            if filter_memory_type and e.memory_type != filter_memory_type:
                sims.append(-1.0)
                continue

            e_norm = norm(e.embedding)
            if q_norm == 0 or e_norm == 0:
                score = 0.0
            else:
                score = dot(e.embedding, query_emb) / (e_norm * q_norm)
            # weight by importance
            score *= (1.0 + e.importance)
            sims.append(score)

        # Rank indices by similarity, only non-negative scores
        idxs = [i for i, s in enumerate(sims) if s >= 0.0]
        idxs.sort(key=lambda i: sims[i], reverse=True)
        top_idxs = idxs[:min(top_k, len(idxs))]
        return [self.entries[i] for i in top_idxs]

    def get_memories_by_type(self, memory_type: MemoryType) -> List[MemoryEntry]:
        """Get all memories of a specific type"""
        return [e for e in self.entries if e.memory_type == memory_type and not e.is_expired()]

    def get_memories_by_timerange(self, start_time: float, end_time: float) -> List[MemoryEntry]:
        """Get all memories within a time range"""
        return [e for e in self.entries if start_time <= e.timestamp <= end_time and not e.is_expired()]

    def get_recent_memories(self, hours: int = 24) -> List[MemoryEntry]:
        """Get memories from the last N hours"""
        cutoff = time.time() - (hours * 3600)
        return [e for e in self.entries if e.timestamp >= cutoff and not e.is_expired()]

    def display_memories(self, limit: int = 20) -> str:
        """Display recent memories in a formatted way"""
        recent = sorted(self.entries, key=lambda e: e.timestamp, reverse=True)[:limit]
        recent = [e for e in recent if not e.is_expired()]
        
        if not recent:
            return "No memories found."
        
        result = "Recent Memories:\n"
        result += "=" * 50 + "\n"
        for entry in recent:
            result += f"{entry}\n"
        return result

    def clear(self) -> None:
        """Clear all memory entries."""
        self.entries.clear()

    def get_all_texts(self) -> List[str]:
        """Return the list of all stored texts."""
        return [e.text for e in self.entries]

    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed entries"""
        initial_count = len(self.entries)
        self.entries = [e for e in self.entries if not e.is_expired()]
        return initial_count - len(self.entries)
    
    def _get_default_importance(self, memory_type: MemoryType) -> float:
        """No default importance - AI must generate."""
        raise RuntimeError(f"AI must generate importance score for {memory_type}")

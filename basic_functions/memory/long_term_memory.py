import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from basic_functions.memory.medium_term_memory import MediumTermMemoryEntry


@dataclass
class LongTermMemoryEntry:
    """Long-term memory entry that stores self-perception and core beliefs"""
    timestamp: float  # Creation timestamp
    category: str  # Category (personality, values, preferences, beliefs, etc.)
    content: str  # Content description
    confidence: float  # Confidence level (0.0-1.0)
    source_summary: str  # Source summary
    embedding: List[float]  # Content embedding
    update_count: int = 1  # Update count
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def get_formatted_time(self) -> str:
        """Return formatted timestamp for display"""
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d")

    def __str__(self) -> str:
        return f"[Long-term] {self.category}: {self.content} (confidence: {self.confidence:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "timestamp": self.timestamp,
            "category": self.category,
            "content": self.content,
            "confidence": self.confidence,
            "source_summary": self.source_summary,
            "embedding": self.embedding,
            "update_count": self.update_count,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LongTermMemoryEntry':
        """Create from dictionary"""
        return cls(**data)


class LongTermMemory:
    """
    Long-term memory system that stores self-perception formed from medium-term memory analysis
    These perceptions appear in the decider's self-introduction
    Long-term memories don't expire but can be replaced or updated by new ones
    """
    
    def __init__(self):
        self.entries: List[LongTermMemoryEntry] = []

    def add_or_update_belief(
        self,
        category: str,
        content: str,
        confidence: float,
        source_summary: str,
        embedding: List[float],
        replace_threshold: float = 0.8,  # Similarity threshold for replacement vs addition
    ) -> None:
        """Add or update belief"""
        now = time.time()
        
        # Check if similar entry exists
        similar_entry = self._find_similar_entry(category, embedding, replace_threshold)
        
        if similar_entry:
            # Update existing entry
            similar_entry.content = content
            similar_entry.confidence = max(similar_entry.confidence, confidence)
            similar_entry.source_summary = source_summary
            similar_entry.embedding = embedding
            similar_entry.timestamp = now
            similar_entry.update_count += 1
        else:
            # Create new entry
            entry = LongTermMemoryEntry(
                timestamp=now,
                category=category,
                content=content,
                confidence=confidence,
                source_summary=source_summary,
                embedding=embedding
            )
            self.entries.append(entry)

    def _find_similar_entry(
        self,
        category: str,
        embedding: List[float],
        threshold: float
    ) -> Optional[LongTermMemoryEntry]:
        """Find similar entry"""
        def dot(a: List[float], b: List[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        def norm(a: List[float]) -> float:
            return (sum(x * x for x in a)) ** 0.5

        q_norm = norm(embedding)
        if q_norm == 0:
            return None

        for entry in self.entries:
            if entry.category == category:
                e_norm = norm(entry.embedding)
                if e_norm > 0:
                    similarity = dot(entry.embedding, embedding) / (e_norm * q_norm)
                    if similarity >= threshold:
                        return entry
        return None

    def get_personality_summary(self) -> str:
        """Get personality summary for decider's self-introduction"""
        if not self.entries:
            return "I am still learning about myself."
        
        # Group by category
        categories = {}
        for entry in self.entries:
            if entry.category not in categories:
                categories[entry.category] = []
            categories[entry.category].append(entry)
        
        # Generate personality description
        personality_parts = []
        
        # Sort by confidence and take top entries from each category
        for category, entries in categories.items():
            entries.sort(key=lambda e: e.confidence, reverse=True)
            top_entries = entries[:3]  # Max 3 entries per category
            
            if category == "personality":
                personality_parts.append(f"My personality traits: {', '.join(e.content for e in top_entries)}")
            elif category == "values":
                personality_parts.append(f"My values: {', '.join(e.content for e in top_entries)}")
            elif category == "preferences":
                personality_parts.append(f"My preferences: {', '.join(e.content for e in top_entries)}")
            elif category == "beliefs":
                personality_parts.append(f"My beliefs: {', '.join(e.content for e in top_entries)}")
            elif category == "goals":
                personality_parts.append(f"My goals: {', '.join(e.content for e in top_entries)}")
            else:
                personality_parts.append(f"About {category}: {', '.join(e.content for e in top_entries)}")
        
        return "\n".join(personality_parts)

    def get_entries_by_category(self, category: str) -> List[LongTermMemoryEntry]:
        """Get entries by specific category"""
        return [e for e in self.entries if e.category == category]

    def get_high_confidence_entries(self, min_confidence: float = 0.7) -> List[LongTermMemoryEntry]:
        """Get high-confidence entries"""
        return [e for e in self.entries if e.confidence >= min_confidence]

    def display_memories(self, limit: int = 20) -> str:
        """Display long-term memories"""
        # Sort by confidence
        sorted_entries = sorted(self.entries, key=lambda e: e.confidence, reverse=True)[:limit]
        
        if not sorted_entries:
            return "No long-term memories found."
        
        result = "Long-term Memories (Self-Knowledge):\n"
        result += "=" * 70 + "\n"
        
        # Group by category for display
        categories = {}
        for entry in sorted_entries:
            if entry.category not in categories:
                categories[entry.category] = []
            categories[entry.category].append(entry)
        
        for category, entries in categories.items():
            result += f"\n{category.upper()}:\n"
            result += "-" * 30 + "\n"
            for entry in entries:
                result += f"  {entry.content} (confidence: {entry.confidence:.2f})\n"
                result += f"    Source: {entry.source_summary}\n"
                result += f"    Updated: {entry.update_count} times\n"
        
        return result

    def remove_low_confidence_entries(self, threshold: float = 0.3) -> int:
        """Remove low-confidence entries"""
        initial_count = len(self.entries)
        self.entries = [e for e in self.entries if e.confidence >= threshold]
        return initial_count - len(self.entries)

    def merge_similar_entries(self, similarity_threshold: float = 0.9) -> int:
        """Merge similar entries"""
        merged_count = 0
        
        # Process by category
        categories = {}
        for entry in self.entries:
            if entry.category not in categories:
                categories[entry.category] = []
            categories[entry.category].append(entry)
        
        new_entries = []
        
        for category, entries in categories.items():
            merged_in_category = set()
            
            for i, entry1 in enumerate(entries):
                if i in merged_in_category:
                    continue
                    
                similar_entries = [entry1]
                
                for j, entry2 in enumerate(entries[i+1:], i+1):
                    if j in merged_in_category:
                        continue
                        
                    # Calculate similarity
                    similarity = self._calculate_similarity(entry1.embedding, entry2.embedding)
                    if similarity >= similarity_threshold:
                        similar_entries.append(entry2)
                        merged_in_category.add(j)
                
                if len(similar_entries) > 1:
                    # Merge entries
                    merged_entry = self._merge_entries(similar_entries)
                    new_entries.append(merged_entry)
                    merged_count += len(similar_entries) - 1
                else:
                    new_entries.append(entry1)
        
        self.entries = new_entries
        return merged_count

    def _calculate_similarity(self, emb1: List[float], emb2: List[float]) -> float:
        """Calculate similarity between two embeddings"""
        def dot(a: List[float], b: List[float]) -> float:
            return sum(x * y for x, y in zip(a, b))

        def norm(a: List[float]) -> float:
            return (sum(x * x for x in a)) ** 0.5

        norm1, norm2 = norm(emb1), norm(emb2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot(emb1, emb2) / (norm1 * norm2)

    def _merge_entries(self, entries: List[LongTermMemoryEntry]) -> LongTermMemoryEntry:
        """Merge multiple entries"""
        # Select the highest confidence entry as base
        base_entry = max(entries, key=lambda e: e.confidence)
        
        # Merge content
        contents = [e.content for e in entries]
        merged_content = f"{base_entry.content} (merged from: {', '.join(contents[1:])})"
        
        # Average confidence
        avg_confidence = sum(e.confidence for e in entries) / len(entries)
        
        # Merge sources
        sources = [e.source_summary for e in entries]
        merged_source = f"Merged from {len(entries)} sources: {'; '.join(sources)}"
        
        return LongTermMemoryEntry(
            timestamp=time.time(),
            category=base_entry.category,
            content=merged_content,
            confidence=avg_confidence,
            source_summary=merged_source,
            embedding=base_entry.embedding,
            update_count=sum(e.update_count for e in entries)
        )

    def save_to_file(self, filename: str) -> None:
        """Save to file"""
        data = [entry.to_dict() for entry in self.entries]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filename: str) -> None:
        """Load from file"""
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.entries = [LongTermMemoryEntry.from_dict(item) for item in data]
        except FileNotFoundError:
            self.entries = []

    def clear(self) -> None:
        """Clear all records"""
        self.entries.clear()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics"""
        if not self.entries:
            return {"total_entries": 0}
        
        categories = {}
        total_confidence = 0
        total_updates = 0
        
        for entry in self.entries:
            if entry.category not in categories:
                categories[entry.category] = 0
            categories[entry.category] += 1
            total_confidence += entry.confidence
            total_updates += entry.update_count
        
        return {
            "total_entries": len(self.entries),
            "categories": categories,
            "average_confidence": total_confidence / len(self.entries),
            "total_updates": total_updates,
            "most_common_category": max(categories.items(), key=lambda x: x[1])[0] if categories else None
        }

    def get_latest_goals(self) -> List[Dict[str, Any]]:
        """Get latest goal-related entries for high-level task formation"""
        goal_entries = [entry for entry in self.entries if entry.category == "goals"]
        
        # Sort by timestamp (newest first) and confidence
        goal_entries.sort(key=lambda x: (x.timestamp, x.confidence), reverse=True)
        
        # Return top goals with their metadata
        return [{"content": entry.content, "confidence": entry.confidence} 
                for entry in goal_entries[:3]]  # Top 3 latest/highest confidence goals
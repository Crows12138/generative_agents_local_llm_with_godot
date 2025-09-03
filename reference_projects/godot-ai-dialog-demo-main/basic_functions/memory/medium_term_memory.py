import time
import json
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
from basic_functions.memory.memory import MemoryEntry, MemoryType

@dataclass
class MediumTermMemoryEntry:
    """Medium-term memory entry that stores daily behavior summary."""
    timestamp: float  # Timestamp of the day
    date: str  # Date string (YYYY-MM-DD)
    behavior_summary: str  # Behavior summary
    expiry_timestamp: Optional[float] = None

    def is_expired(self) -> bool:
        if self.expiry_timestamp is None:
            return False
        return time.time() > self.expiry_timestamp

    def get_formatted_time(self) -> str:
        return datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d")

    def __str__(self) -> str:
        return f"[Medium-term] {self.date}: {self.behavior_summary[:100]}..."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "date": self.date,
            "behavior_summary": self.behavior_summary,
            "expiry_timestamp": self.expiry_timestamp
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MediumTermMemoryEntry':
        return cls(**data)

class MediumTermMemory:
    """
    Medium-term memory system: only one behavior summary is saved per day.
    """
    def __init__(self):
        self.entries: List[MediumTermMemoryEntry] = []

    def add_daily_summary(
        self,
        date: str,
        behavior_summary: str,
        ttl_days: int = 7,  # Default retention is 7 days
    ) -> None:
        now = time.time()
        expiry = now + (ttl_days * 24 * 3600) if ttl_days > 0 else None
        # Check if a record for this date already exists
        existing = next((e for e in self.entries if e.date == date), None)
        if existing:
            existing.behavior_summary = behavior_summary
            existing.expiry_timestamp = expiry
            existing.timestamp = now
        else:
            entry = MediumTermMemoryEntry(
                timestamp=now,
                date=date,
                behavior_summary=behavior_summary,
                expiry_timestamp=expiry
            )
            self.entries.append(entry)

    def get_recent_entries(self, days: int = 7) -> List[MediumTermMemoryEntry]:
        cutoff = time.time() - (days * 24 * 3600)
        return [e for e in self.entries if e.timestamp >= cutoff and not e.is_expired()]

    def get_entry_by_date(self, date: str) -> Optional[MediumTermMemoryEntry]:
        return next((e for e in self.entries if e.date == date and not e.is_expired()), None)

    def display_memories(self, limit: int = 10) -> str:
        recent = sorted(self.entries, key=lambda e: e.timestamp, reverse=True)[:limit]
        recent = [e for e in recent if not e.is_expired()]
        if not recent:
            return "No medium-term memories found."
        result = "Medium-term Memories:\n"
        result += "=" * 60 + "\n"
        for entry in recent:
            result += f"{entry}\n"
            result += "-" * 60 + "\n"
        return result

    def cleanup_expired(self) -> int:
        initial_count = len(self.entries)
        self.entries = [e for e in self.entries if not e.is_expired()]
        return initial_count - len(self.entries)

    def save_to_file(self, filename: str) -> None:
        data = [entry.to_dict() for entry in self.entries]
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_from_file(self, filename: str) -> None:
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                data = json.load(f)
            self.entries = [MediumTermMemoryEntry.from_dict(item) for item in data]
        except FileNotFoundError:
            self.entries = []

    def clear(self) -> None:
        self.entries.clear()

    def get_behavior_patterns_for_prompt(self, days_back: int = 7) -> str:
        recent_entries = self.get_recent_entries(days_back)
        if not recent_entries:
            return "No recent behavior patterns available."
        # Directly concatenate all behavior_summary fields
        summaries = [e.behavior_summary for e in recent_entries]
        return "Recent behavior summaries:\n" + "\n".join(f"- {s}" for s in summaries) 
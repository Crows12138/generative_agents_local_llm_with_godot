#!/usr/bin/env python3
"""
NPC Memory Manager for persistent conversation storage
Saves NPC dialogue history to JSON files for continuity
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

class NPCMemoryManager:
    """Manages memory storage and retrieval for NPCs"""
    
    def __init__(self, memory_dir: str = "../npc_memories"):
        """Initialize memory manager with storage directory"""
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(exist_ok=True)
        self.memories: Dict[str, List[Dict]] = {}
        self.load_all_memories()
    
    def load_all_memories(self):
        """Load all existing memory files on startup"""
        for json_file in self.memory_dir.glob("*.json"):
            npc_name = json_file.stem
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    self.memories[npc_name] = json.load(f)
                print(f"Loaded {len(self.memories[npc_name])} memories for {npc_name}")
            except Exception as e:
                print(f"Error loading memories for {npc_name}: {e}")
                self.memories[npc_name] = []
    
    def calculate_importance(self, text: str, is_deep_thinking: bool = False) -> float:
        """
        Calculate importance score for a memory (0-10 scale)
        
        Args:
            text: The dialogue text
            is_deep_thinking: Whether this was a deep thinking response
            
        Returns:
            Importance score between 0 and 10
        """
        score = 2.0  # Base score
        
        # Deep thinking adds significant importance
        if is_deep_thinking:
            score += 3.0
        
        # Keywords and their weights
        keywords = {
            'whiskey': 1.5,
            'advice': 2.0,
            'years': 1.0,
            'experience': 1.5,
            'remember': 2.0,
            'story': 1.5,
            'wisdom': 1.5,
            'life': 1.0,
            'important': 1.5,
            'special': 1.5,
            'favorite': 1.0,
            'always': 1.0
        }
        
        # Check for keywords in text
        text_lower = text.lower()
        for keyword, weight in keywords.items():
            if keyword in text_lower:
                score += weight
                # Only count each keyword once
                break
        
        # Cap at 10
        return min(score, 10.0)
    
    def save_memory(self, 
                    npc_name: str, 
                    user_input: str,
                    npc_response: str,
                    is_deep_thinking: bool = False,
                    metadata: Optional[Dict[str, Any]] = None):
        """
        Save a conversation memory for an NPC
        
        Args:
            npc_name: Name of the NPC (e.g., "Bob")
            user_input: What the user said
            npc_response: What the NPC responded
            is_deep_thinking: Whether this was a deep thinking response
            metadata: Optional additional data
        """
        # Initialize memory list if new NPC
        if npc_name not in self.memories:
            self.memories[npc_name] = []
        
        # Calculate importance
        importance = self.calculate_importance(
            f"{user_input} {npc_response}", 
            is_deep_thinking
        )
        
        # Create memory entry
        memory_entry = {
            "timestamp": time.time(),
            "datetime": datetime.now().isoformat(),
            "user_input": user_input,
            "npc_response": npc_response,
            "is_deep_thinking": is_deep_thinking,
            "importance": importance,
            "metadata": metadata or {}
        }
        
        # Add to memory
        self.memories[npc_name].append(memory_entry)
        
        # Save to file
        self._save_to_file(npc_name)
        
        # Print confirmation
        thinking_tag = " [DEEP]" if is_deep_thinking else ""
        print(f"Memory saved for {npc_name}{thinking_tag} (importance: {importance:.1f})")
    
    def _save_to_file(self, npc_name: str):
        """Save memories to JSON file"""
        file_path = self.memory_dir / f"{npc_name}.json"
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.memories.get(npc_name, []), f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error saving memories for {npc_name}: {e}")
    
    def get_memories(self, npc_name: str, limit: int = -1) -> List[Dict]:
        """
        Get memories for an NPC
        
        Args:
            npc_name: Name of the NPC
            limit: Maximum number of memories to return (-1 for all)
            
        Returns:
            List of memory entries
        """
        memories = self.memories.get(npc_name, [])
        
        if limit > 0:
            # Return most recent memories
            return memories[-limit:]
        
        return memories
    
    def get_memory_stats(self, npc_name: str) -> Dict[str, Any]:
        """Get statistics about an NPC's memories"""
        memories = self.memories.get(npc_name, [])
        
        if not memories:
            return {
                "total": 0,
                "deep_thinking_count": 0,
                "deep_thinking_percentage": 0.0,
                "average_importance": 0,
                "first_memory": None,
                "last_memory": None
            }
        
        deep_count = sum(1 for m in memories if m.get("is_deep_thinking", False))
        avg_importance = sum(m.get("importance", 0) for m in memories) / len(memories)
        
        return {
            "total": len(memories),
            "deep_thinking_count": deep_count,
            "deep_thinking_percentage": (deep_count / len(memories)) * 100,
            "average_importance": avg_importance,
            "first_memory": memories[0]["datetime"] if memories else None,
            "last_memory": memories[-1]["datetime"] if memories else None
        }
    
    def clear_memories(self, npc_name: str):
        """Clear all memories for an NPC"""
        if npc_name in self.memories:
            self.memories[npc_name] = []
            self._save_to_file(npc_name)
            print(f"Cleared memories for {npc_name}")
    
    def export_readable(self, npc_name: str, output_file: Optional[str] = None) -> str:
        """
        Export memories in a human-readable format
        
        Args:
            npc_name: Name of the NPC
            output_file: Optional file to save to
            
        Returns:
            Formatted string of memories
        """
        memories = self.memories.get(npc_name, [])
        
        if not memories:
            return f"No memories found for {npc_name}"
        
        output = []
        output.append(f"=== Conversation History for {npc_name} ===\n")
        output.append(f"Total memories: {len(memories)}\n")
        
        stats = self.get_memory_stats(npc_name)
        output.append(f"Deep thinking responses: {stats['deep_thinking_count']} ({stats['deep_thinking_percentage']:.1f}%)")
        output.append(f"Average importance: {stats['average_importance']:.1f}\n")
        output.append("=" * 50 + "\n")
        
        for i, memory in enumerate(memories, 1):
            dt = datetime.fromisoformat(memory['datetime'])
            thinking = " [DEEP THINKING]" if memory.get('is_deep_thinking') else ""
            importance = memory.get('importance', 0)
            
            output.append(f"\n#{i} - {dt.strftime('%Y-%m-%d %H:%M:%S')}{thinking}")
            output.append(f"Importance: {importance:.1f}/10")
            output.append(f"User: {memory['user_input']}")
            output.append(f"{npc_name}: {memory['npc_response']}")
            output.append("-" * 50)
        
        result = "\n".join(output)
        
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            print(f"Exported to {output_file}")
        
        return result


# Test the memory manager
if __name__ == "__main__":
    # Create manager
    manager = NPCMemoryManager()
    
    # Test saving some memories
    manager.save_memory(
        "Bob",
        "What's your best whiskey?",
        "Well, that's a question I get asked a lot. *thoughtfully* After 20 years behind this bar, I'd say the best whiskey isn't always the most expensive one.",
        is_deep_thinking=True
    )
    
    manager.save_memory(
        "Bob",
        "How's business?",
        "Can't complain. Tuesday nights are always quiet, gives me time to polish the glasses properly.",
        is_deep_thinking=False
    )
    
    # Show stats
    stats = manager.get_memory_stats("Bob")
    print(f"\nBob's memory stats: {json.dumps(stats, indent=2)}")
    
    # Export readable format
    print("\n" + manager.export_readable("Bob"))
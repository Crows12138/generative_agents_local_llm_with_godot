#!/usr/bin/env python3
"""
Memory Viewer for GPT4All NPC System
View a specific NPC's conversation memories
"""

import json
from pathlib import Path
import sys

def main():
    # Get NPC name from arguments
    npc_name = sys.argv[1] if len(sys.argv) > 1 else "Bob"
    
    # Load memories from the standard format
    memory_file = Path(__file__).parent.parent / "npc_memories" / f"{npc_name}.json"
    
    if not memory_file.exists():
        print(f"No memory file found for {npc_name}")
        return
    
    # Load memories
    with open(memory_file, 'r', encoding='utf-8') as f:
        memories = json.load(f)
    
    if not memories:
        print(f"No memories found for {npc_name}")
        return
    
    print(f"=== {npc_name}'s Memories ===")
    print(f"Total: {len(memories)} conversation(s)\n")
    
    # Show each memory
    for i, memory in enumerate(memories):
        print(f"--- Conversation #{i + 1} ---")
        
        # Show datetime if available
        if 'datetime' in memory:
            print(f"Time: {memory['datetime']}")
        
        # Show conversation
        print(f"User: {memory.get('user_input', 'N/A')}")
        print(f"{npc_name}: {memory.get('npc_response', 'N/A')}")
        
        # Show metadata
        metadata = memory.get('metadata', {})
        if metadata:
            response_time = metadata.get('response_time', 0)
            model = metadata.get('model', 'Unknown')
            print(f"Response Time: {response_time:.2f}s")
            print(f"Model: {model}")
        
        print()  # Empty line between entries

if __name__ == "__main__":
    main()

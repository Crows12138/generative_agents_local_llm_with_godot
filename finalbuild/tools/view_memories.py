#!/usr/bin/env python3
"""
Memory Viewer Utility
View and analyze stored NPC conversation memories
Usage: python view_memories.py [npc_name] [--last N] [--stats] [--export]
"""

import argparse
import json
from pathlib import Path
from datetime import datetime
import sys
import os
# Add server directory to path
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'server'))
from memory_integration import NPCMemoryManager


def print_memory_entry(entry, index):
    """Print a single memory entry in a formatted way"""
    dt = datetime.fromisoformat(entry['datetime'])
    thinking = " [DEEP THINKING]" if entry.get('is_deep_thinking') else ""
    importance = entry.get('importance', 0)
    
    print(f"\n#{index} - {dt.strftime('%Y-%m-%d %H:%M:%S')}{thinking}")
    print(f"Importance: {importance:.1f}/10")
    print(f"User: {entry['user_input']}")
    print(f"NPC: {entry['npc_response']}")
    
    # Show metadata if present
    metadata = entry.get('metadata', {})
    if metadata:
        if 'response_time' in metadata:
            print(f"Response time: {metadata['response_time']:.2f}s")
    
    print("-" * 50)


def view_memories(npc_name, last_n=None, show_stats=False, export_file=None):
    """View memories for a specific NPC"""
    manager = NPCMemoryManager()
    
    # Get memories
    memories = manager.get_memories(npc_name, limit=last_n if last_n else -1)
    
    if not memories:
        print(f"No memories found for {npc_name}")
        return
    
    # Show statistics if requested
    if show_stats:
        stats = manager.get_memory_stats(npc_name)
        print("\n" + "=" * 60)
        print(f"MEMORY STATISTICS FOR {npc_name.upper()}")
        print("=" * 60)
        print(f"Total memories: {stats['total']}")
        print(f"Deep thinking responses: {stats['deep_thinking_count']} ({stats['deep_thinking_percentage']:.1f}%)")
        print(f"Average importance: {stats['average_importance']:.1f}/10")
        if stats['first_memory']:
            print(f"First memory: {stats['first_memory']}")
        if stats['last_memory']:
            print(f"Last memory: {stats['last_memory']}")
        print("=" * 60)
    
    # Export if requested
    if export_file:
        readable = manager.export_readable(npc_name, export_file)
        print(f"\nExported to {export_file}")
        return
    
    # Display memories
    print(f"\n=== Conversation History for {npc_name} ===")
    print(f"Showing {'last ' + str(last_n) if last_n else 'all'} memories")
    print("=" * 50)
    
    for i, memory in enumerate(memories, 1):
        print_memory_entry(memory, i)
    
    # Show summary
    deep_count = sum(1 for m in memories if m.get('is_deep_thinking'))
    avg_importance = sum(m.get('importance', 0) for m in memories) / len(memories)
    
    print(f"\nSummary: {len(memories)} memories shown")
    print(f"Deep thinking: {deep_count}/{len(memories)}")
    print(f"Average importance: {avg_importance:.1f}/10")


def list_npcs():
    """List all NPCs with saved memories"""
    memory_dir = Path("npc_memories")
    if not memory_dir.exists():
        print("No memories directory found. No NPCs have memories yet.")
        return
    
    json_files = list(memory_dir.glob("*.json"))
    if not json_files:
        print("No NPC memories found.")
        return
    
    print("\n=== NPCs with Saved Memories ===")
    print("=" * 40)
    
    manager = NPCMemoryManager()
    for json_file in json_files:
        npc_name = json_file.stem
        stats = manager.get_memory_stats(npc_name)
        if stats['total'] > 0:
            print(f"\n{npc_name}:")
            print(f"  Total memories: {stats['total']}")
            print(f"  Deep thinking: {stats['deep_thinking_count']} ({stats['deep_thinking_percentage']:.1f}%)")
            print(f"  Avg importance: {stats['average_importance']:.1f}/10")
    
    print("\n" + "=" * 40)
    print("\nUse: python view_memories.py [npc_name] to view details")


def main():
    parser = argparse.ArgumentParser(
        description="View and analyze NPC conversation memories"
    )
    parser.add_argument(
        "npc_name",
        nargs="?",
        help="Name of the NPC (e.g., Bob, Alice). Leave empty to list all NPCs."
    )
    parser.add_argument(
        "--last",
        "-l",
        type=int,
        help="Show only the last N memories"
    )
    parser.add_argument(
        "--stats",
        "-s",
        action="store_true",
        help="Show statistics for the NPC"
    )
    parser.add_argument(
        "--export",
        "-e",
        type=str,
        help="Export memories to a readable text file"
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear all memories for the NPC (use with caution!)"
    )
    
    args = parser.parse_args()
    
    # If no NPC name, list all NPCs
    if not args.npc_name:
        list_npcs()
        return
    
    # Clear memories if requested
    if args.clear:
        response = input(f"Are you sure you want to clear all memories for {args.npc_name}? (yes/no): ")
        if response.lower() == "yes":
            manager = NPCMemoryManager()
            manager.clear_memories(args.npc_name)
            print(f"Memories cleared for {args.npc_name}")
        else:
            print("Clear operation cancelled")
        return
    
    # View memories
    view_memories(
        args.npc_name,
        last_n=args.last,
        show_stats=args.stats,
        export_file=args.export
    )


if __name__ == "__main__":
    main()
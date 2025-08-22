"""
Memory Adapter for Reverie Integration

Adapter to use reverie's memory system in simple agents.
Provides a bridge between SimpleAgent and reverie's memory structures.
"""

import sys
import os
import json
import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

# Add reverie path for imports
reverie_path = os.path.join(os.path.dirname(__file__), '..', 'reverie', 'backend_server')
sys.path.append(reverie_path)

try:
    from persona.memory_structures.spatial_memory import MemoryTree
    from persona.memory_structures.associative_memory import AssociativeMemory, ConceptNode
    from persona.memory_structures.scratch import Scratch
    REVERIE_AVAILABLE = True
except ImportError as e:
    print(f"[memory_adapter] Reverie memory structures not available: {e}")
    REVERIE_AVAILABLE = False

class ReverieMemoryAdapter:
    """Adapter to use reverie's memory system in simple agents"""
    
    def __init__(self, agent_name: str, memory_dir: Optional[str] = None):
        self.agent_name = agent_name
        self.memory_dir = memory_dir or self._create_memory_dir()
        
        if not REVERIE_AVAILABLE:
            print(f"[memory_adapter] Warning: Reverie not available for {agent_name}")
            self._init_fallback_memory()
            return
            
        try:
            # Initialize reverie memory structures
            self._init_spatial_memory()
            self._init_associative_memory()
            self._init_scratch()
            
            print(f"[memory_adapter] Initialized reverie memory for {agent_name}")
        except Exception as e:
            print(f"[memory_adapter] Error initializing reverie memory: {e}")
            self._init_fallback_memory()
    
    def _create_memory_dir(self) -> str:
        """Create memory directory for this agent"""
        memory_dir = Path(f"memory_cache/{self.agent_name}")
        memory_dir.mkdir(parents=True, exist_ok=True)
        return str(memory_dir)
    
    def _init_spatial_memory(self):
        """Initialize spatial memory"""
        spatial_file = os.path.join(self.memory_dir, "spatial_memory.json")
        if not os.path.exists(spatial_file):
            # Create empty spatial memory file
            with open(spatial_file, 'w') as f:
                json.dump({}, f)
        
        try:
            self.spatial_memory = MemoryTree(spatial_file)
        except Exception as e:
            print(f"[memory_adapter] Could not load MemoryTree, using simplified version: {e}")
            # Create simplified spatial memory
            self.spatial_memory = type('SimpleSpatialMemory', (), {
                'tree': {},
                'save': lambda self, path: None,
                'get_str_accessible_sector_arenas': lambda self, sector: f"Location: {sector}"
            })()
    
    def _init_associative_memory(self):
        """Initialize associative memory"""
        # Create necessary files for associative memory
        nodes_file = os.path.join(self.memory_dir, "nodes.json")
        embeddings_file = os.path.join(self.memory_dir, "embeddings.json")
        kw_strength_file = os.path.join(self.memory_dir, "kw_strength.json")
        
        if not os.path.exists(nodes_file):
            with open(nodes_file, 'w') as f:
                json.dump({}, f)
        
        if not os.path.exists(embeddings_file):
            with open(embeddings_file, 'w') as f:
                json.dump({}, f)
                
        if not os.path.exists(kw_strength_file):
            with open(kw_strength_file, 'w') as f:
                json.dump({}, f)
        
        try:
            self.associative_memory = AssociativeMemory(self.memory_dir)
        except Exception as e:
            print(f"[memory_adapter] Could not load AssociativeMemory, using simplified version: {e}")
            # Create a simplified associative memory
            self.associative_memory = type('SimpleAssociativeMemory', (), {
                'seq_event': [],
                'seq_thought': [],
                'seq_chat': [],
                'add_event': self._simple_add_event,
                'add_thought': self._simple_add_thought,
                'retrieve_relevant_memories': self._simple_retrieve,
                'retrieve_relevant_events': self._simple_retrieve_events,
                'retrieve_relevant_thoughts': self._simple_retrieve_thoughts
            })()
    
    def _init_scratch(self):
        """Initialize scratch memory"""
        scratch_file = os.path.join(self.memory_dir, "scratch.json")
        if not os.path.exists(scratch_file):
            # Create basic scratch data
            scratch_data = {
                "name": self.agent_name,
                "curr_time": datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "vision_r": 4,
                "att_bandwidth": 3,
                "retention": 5
            }
            with open(scratch_file, 'w') as f:
                json.dump(scratch_data, f)
        
        try:
            self.scratch = Scratch(scratch_file)
            self.scratch.name = self.agent_name
            self.scratch.curr_time = datetime.datetime.now()
        except Exception as e:
            print(f"[memory_adapter] Could not load Scratch, using simplified version: {e}")
            # Create simplified scratch
            self.scratch = type('SimpleScratch', (), {
                'name': self.agent_name,
                'curr_time': datetime.datetime.now(),
                'vision_r': 4,
                'att_bandwidth': 3,
                'retention': 5,
                'curr_tile': (0, 0),
                'act_description': 'idle',
                'act_address': 'world:zone:0_0',
                'act_start_time': datetime.datetime.now(),
                'act_duration': 60,
                'act_pronunciatio': 'idle',
                'act_event': [],
                'act_obj_description': {},
                'act_obj_pronunciatio': {},
                'act_obj_event': {}
            })()
    
    def _init_fallback_memory(self):
        """Initialize fallback memory system when reverie is not available"""
        self.spatial_memory = {"tree": {}}
        self.associative_memory = {"events": [], "thoughts": [], "chats": []}
        self.scratch = {"name": self.agent_name, "curr_time": datetime.datetime.now()}
    
    def add_spatial_memory(self, location_data: Dict[str, Any]):
        """Add spatial memories"""
        if not REVERIE_AVAILABLE:
            self.spatial_memory["tree"].update(location_data)
            return
            
        try:
            # location_data format: {"world": {"sector": {"arena": ["objects"]}}}
            self.spatial_memory.tree.update(location_data)
            self.spatial_memory.save(os.path.join(self.memory_dir, "spatial_memory.json"))
        except Exception as e:
            print(f"[memory_adapter] Error adding spatial memory: {e}")
    
    def add_event_memory(self, description: str, importance: int = 5):
        """Add event to associative memory"""
        if not REVERIE_AVAILABLE:
            event = {
                "description": description,
                "importance": importance,
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.associative_memory["events"].append(event)
            return
            
        try:
            # Create event node
            created = self.scratch.curr_time or datetime.datetime.now()
            expiration = None  # No expiration
            s, p, o = self._extract_triple(description)
            keywords = set([s, p, o])
            
            # Generate embedding (simplified)
            embedding = self._generate_embedding(description)
            
            self.associative_memory.add_event(
                created, expiration, s, p, o,
                description, keywords, importance,
                embedding, None
            )
        except Exception as e:
            print(f"[memory_adapter] Error adding event memory: {e}")
    
    def add_thought_memory(self, thought: str, importance: int = 4):
        """Add thought to associative memory"""
        if not REVERIE_AVAILABLE:
            thought_entry = {
                "description": thought,
                "importance": importance,
                "timestamp": datetime.datetime.now().isoformat()
            }
            self.associative_memory["thoughts"].append(thought_entry)
            return
            
        try:
            created = self.scratch.curr_time or datetime.datetime.now()
            expiration = None
            s, p, o = self._extract_triple(thought)
            keywords = set([s, p, o])
            embedding = self._generate_embedding(thought)
            
            self.associative_memory.add_thought(
                created, expiration, s, p, o,
                thought, keywords, importance,
                embedding, None
            )
        except Exception as e:
            print(f"[memory_adapter] Error adding thought memory: {e}")
    
    def retrieve_relevant_memories(self, query: str, n: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant memories for a query"""
        if not REVERIE_AVAILABLE:
            # Simple fallback retrieval
            all_memories = []
            for event in self.associative_memory.get("events", [])[-n:]:
                all_memories.append({
                    "description": event["description"],
                    "type": "event",
                    "importance": event["importance"]
                })
            return all_memories
            
        try:
            # Get focal points for retrieval
            focal_points = [query]
            
            # Retrieve from associative memory
            retrieved = []
            for focal in focal_points:
                try:
                    nodes = self.associative_memory.retrieve_relevant_memories(focal, n)
                    for node in nodes:
                        retrieved.append({
                            "description": node.description,
                            "type": node.type,
                            "importance": node.poignancy,
                            "created": node.created,
                            "subject": node.subject,
                            "predicate": node.predicate,
                            "object": node.object
                        })
                except Exception as e:
                    print(f"[memory_adapter] Error retrieving memories for '{focal}': {e}")
            
            return retrieved[:n]
        except Exception as e:
            print(f"[memory_adapter] Error in retrieve_relevant_memories: {e}")
            return []
    
    def get_spatial_context(self, location: str) -> str:
        """Get spatial context for a location"""
        if not REVERIE_AVAILABLE:
            return f"Location: {location}"
            
        try:
            # Try to get accessible areas
            if hasattr(self.spatial_memory, 'get_str_accessible_sector_arenas'):
                # Need to format location properly for reverie (world:sector format)
                if ":" not in location:
                    location = f"world:{location}"
                return self.spatial_memory.get_str_accessible_sector_arenas(location)
            else:
                return f"Location: {location}"
        except Exception as e:
            print(f"[memory_adapter] Error getting spatial context: {e}")
            return f"Location: {location}"
    
    def update_current_time(self, current_time: datetime.datetime):
        """Update current time in scratch memory"""
        if REVERIE_AVAILABLE and hasattr(self.scratch, 'curr_time'):
            self.scratch.curr_time = current_time
        else:
            self.scratch["curr_time"] = current_time
    
    def _extract_triple(self, description: str) -> tuple:
        """Extract subject-predicate-object triple from description"""
        # Simplified extraction logic
        words = description.split()
        if len(words) >= 3:
            return words[0], words[1], " ".join(words[2:])
        elif len(words) == 2:
            return self.agent_name, words[0], words[1]
        else:
            return self.agent_name, "experiences", description
    
    def _generate_embedding(self, text: str) -> tuple:
        """Generate text embedding (simplified for compatibility)"""
        # In a real implementation, this would use sentence transformers
        # For now, return a simple hash-based embedding
        embedding_key = f"emb_{hash(text) % 10000}"
        embedding_vector = [float(ord(c) % 256) / 255.0 for c in text[:768]]
        
        # Pad or truncate to standard size
        if len(embedding_vector) < 768:
            embedding_vector.extend([0.0] * (768 - len(embedding_vector)))
        else:
            embedding_vector = embedding_vector[:768]
            
        return (embedding_key, embedding_vector)
    
    def save_memories(self):
        """Save all memories to disk"""
        if not REVERIE_AVAILABLE:
            # Save fallback memories
            memory_file = os.path.join(self.memory_dir, "fallback_memory.json")
            with open(memory_file, 'w') as f:
                json.dump({
                    "spatial": self.spatial_memory,
                    "associative": self.associative_memory,
                    "scratch": self.scratch
                }, f, default=str)
            return
            
        try:
            # Save spatial memory
            if hasattr(self.spatial_memory, 'save'):
                self.spatial_memory.save(os.path.join(self.memory_dir, "spatial_memory.json"))
            
            # Save associative memory
            if hasattr(self.associative_memory, 'save'):
                self.associative_memory.save(self.memory_dir)
            
            # Save scratch
            if hasattr(self.scratch, 'save'):
                self.scratch.save(os.path.join(self.memory_dir, "scratch.json"))
                
            print(f"[memory_adapter] Saved memories for {self.agent_name}")
        except Exception as e:
            print(f"[memory_adapter] Error saving memories: {e}")
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """Get a summary of current memory state"""
        summary = {
            "agent_name": self.agent_name,
            "memory_dir": self.memory_dir,
            "reverie_available": REVERIE_AVAILABLE
        }
        
        if REVERIE_AVAILABLE:
            try:
                summary.update({
                    "spatial_locations": len(self.spatial_memory.tree) if hasattr(self.spatial_memory, 'tree') else 0,
                    "associative_events": len(self.associative_memory.seq_event) if hasattr(self.associative_memory, 'seq_event') else 0,
                    "associative_thoughts": len(self.associative_memory.seq_thought) if hasattr(self.associative_memory, 'seq_thought') else 0,
                    "current_time": str(self.scratch.curr_time) if hasattr(self.scratch, 'curr_time') else "Unknown"
                })
            except Exception as e:
                summary["error"] = str(e)
        else:
            summary.update({
                "fallback_events": len(self.associative_memory.get("events", [])),
                "fallback_thoughts": len(self.associative_memory.get("thoughts", [])),
                "fallback_spatial": len(self.spatial_memory.get("tree", {}))
            })
        
        return summary
    
    def _simple_add_event(self, created, expiration, s, p, o, description, keywords, importance, embedding, filling):
        """Simplified event addition for when reverie memory fails"""
        event = {
            'type': 'event',
            'created': created,
            'subject': s,
            'predicate': p,
            'object': o,
            'description': description,
            'importance': importance,
            'keywords': keywords
        }
        self.associative_memory.seq_event.append(event)
    
    def _simple_add_thought(self, created, expiration, s, p, o, description, keywords, importance, embedding, filling):
        """Simplified thought addition for when reverie memory fails"""
        thought = {
            'type': 'thought',
            'created': created,
            'subject': s,
            'predicate': p,
            'object': o,
            'description': description,
            'importance': importance,
            'keywords': keywords
        }
        self.associative_memory.seq_thought.append(thought)
    
    def _simple_retrieve(self, query, n=5):
        """Simplified memory retrieval for when reverie memory fails"""
        all_memories = []
        
        # Get events and thoughts
        if hasattr(self.associative_memory, 'seq_event'):
            all_memories.extend(self.associative_memory.seq_event)
        if hasattr(self.associative_memory, 'seq_thought'):
            all_memories.extend(self.associative_memory.seq_thought)
        
        # Simple relevance scoring based on keyword match
        relevant = []
        query_words = query.lower().split()
        
        for memory in all_memories:
            score = 0
            description = memory.get('description', '').lower()
            keywords = memory.get('keywords', set())
            
            # Score based on word matches
            for word in query_words:
                if word in description:
                    score += 2
                if isinstance(keywords, (set, list)) and word in str(keywords).lower():
                    score += 1
            
            if score > 0:
                memory_obj = type('SimpleMemory', (), memory)()
                memory_obj.poignancy = memory.get('importance', 5)
                relevant.append(memory_obj)
        
        # Sort by relevance and return top n
        relevant.sort(key=lambda x: x.poignancy, reverse=True)
        return relevant[:n]
    
    def _simple_retrieve_events(self, subject, predicate, obj):
        """Simplified event retrieval for reverie compatibility"""
        query = f"{subject} {predicate} {obj}"
        all_events = getattr(self.associative_memory, 'seq_event', [])
        
        relevant = []
        query_words = query.lower().split()
        
        for event in all_events:
            score = 0
            description = event.get('description', '').lower()
            
            for word in query_words:
                if word in description:
                    score += 1
            
            if score > 0:
                memory_obj = type('SimpleMemory', (), event)()
                memory_obj.poignancy = event.get('importance', 5)
                relevant.append(memory_obj)
        
        relevant.sort(key=lambda x: x.poignancy, reverse=True)
        return relevant[:5]
    
    def _simple_retrieve_thoughts(self, subject, predicate, obj):
        """Simplified thought retrieval for reverie compatibility"""
        query = f"{subject} {predicate} {obj}"
        all_thoughts = getattr(self.associative_memory, 'seq_thought', [])
        
        relevant = []
        query_words = query.lower().split()
        
        for thought in all_thoughts:
            score = 0
            description = thought.get('description', '').lower()
            
            for word in query_words:
                if word in description:
                    score += 1
            
            if score > 0:
                memory_obj = type('SimpleMemory', (), thought)()
                memory_obj.poignancy = thought.get('importance', 4)
                relevant.append(memory_obj)
        
        relevant.sort(key=lambda x: x.poignancy, reverse=True)
        return relevant[:5]
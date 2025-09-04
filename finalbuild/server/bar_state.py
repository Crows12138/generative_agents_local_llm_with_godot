"""
Object State System for Bar Environment
Implements binary state system with simple Godot communication
"""

import time
import random
import json
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
from pathlib import Path
import math


class ObjectState(Enum):
    """Common states for interactive objects"""
    # Bar overall atmosphere states
    BAR_NORMAL = "normal"          # Everything is fine, regular business
    BAR_BUSY = "busy"             # Many customers, high activity  
    BAR_QUIET = "quiet"           # Few customers, peaceful
    BAR_EMPTY = "empty"           # No customers
    BAR_DIRTY = "dirty"           # Floor and general area needs cleaning
    
    # Bar counter specific states
    COUNTER_CUSTOMERS_WAITING = "customers_waiting"  # Needs service
    COUNTER_DIRTY = "dirty"                         # Needs cleaning
    # No COUNTER_NORMAL - that's just the absence of problems
    
    # Shelf states
    SHELF_FULL = "full"
    SHELF_HALF = "half_empty"
    SHELF_LOW = "low_stock"
    SHELF_EMPTY = "empty"
    
    # Table states
    TABLE_EMPTY = "empty"
    TABLE_OCCUPIED = "occupied"
    TABLE_FOOD_WAITING = "food_waiting"  # Customers waiting for food
    TABLE_HAS_FOOD = "has_food"          # Food is on the table
    TABLE_NEEDS_CLEANING = "needs_cleaning"
    
    # Pool table states
    POOL_EMPTY = "empty"  # No one playing
    POOL_IN_USE = "in_use"
    POOL_WAITING = "waiting_for_player"


class InteractableObject:
    """Base class for all interactive objects"""
    
    def __init__(self, name: str, position: Tuple[int, int], object_type: str):
        self.name = name
        self.position = position
        self.object_type = object_type
        self.state = None
        self.signals = []
        self.last_interaction = None
        self.cooldown = 0
        self.properties = {}
        
        # Performance optimization
        self._dirty = False  # Track if state changed
        self._last_update_time = 0
        self._update_interval = 1.0  # Update every 1 second instead of every frame
        
    def should_update(self, current_time: float) -> bool:
        """Check if enough time has passed for an update"""
        if current_time - self._last_update_time >= self._update_interval:
            self._last_update_time = current_time
            return True
        return False
    
    def update(self, current_time: float, delta_time: float):
        """Update object state over time"""
        # Always decrease cooldown
        if self.cooldown > 0:
            self.cooldown = max(0, self.cooldown - delta_time)
            
        # Only do full update at intervals
        if not self.should_update(current_time):
            return
            
        old_state = self.state
        self._perform_update(current_time, delta_time)
        
        # Mark dirty if state changed
        if old_state != self.state:
            self._dirty = True
    
    def _perform_update(self, current_time: float, delta_time: float):
        """Override in subclasses for specific update logic"""
        pass
            
    def interact(self, action: str, actor: str, current_time: float) -> str:
        """Handle interaction with the object"""
        if self.cooldown > 0:
            return f"{self.name} is not ready for interaction yet"
        
        self.last_interaction = {
            "action": action,
            "actor": actor,
            "time": current_time
        }
        return f"{actor} interacted with {self.name}"
    
    def get_active_signals(self) -> List[str]:
        """Return currently active signals"""
        return self.signals.copy()
    
    def add_signal(self, signal: str):
        """Add a signal if not already present"""
        if signal not in self.signals:
            self.signals.append(signal)
            
    def remove_signal(self, signal: str):
        """Remove a signal if present"""
        if signal in self.signals:
            self.signals.remove(signal)
            
    def get_description(self) -> str:
        """Get human-readable description of object state"""
        return f"{self.name} is {self.state}"


class BarCounter(InteractableObject):
    """The main bar counter where drinks are served"""
    
    def __init__(self, position: Tuple[int, int] = (400, 280)):
        super().__init__("bar_counter", position, "service")
        self.state = None  # Start with no issues
        self.properties = {
            "cleanliness": 100,  # 0-100
            "items_on_counter": ["books", "candelabra", "bust_sculpture"],
            "customers_waiting": 0
        }
        
    def _perform_update(self, current_time: float, delta_time: float):
        """Bar counter specific update logic - Only track actionable states"""
        # Priority order:
        # 1. COUNTER_CUSTOMERS_WAITING - Most urgent, customers need service
        # 2. COUNTER_DIRTY - Needs cleaning
        # None - Everything is fine, no action needed
        
        # Bar gets dirty over time
        if random.random() < 0.02 * delta_time:  # 2% chance per second
            self.properties["cleanliness"] -= random.randint(5, 15)
            
        # Clear all signals first
        self.remove_signal("customers_waiting")
        self.remove_signal("needs_cleaning")
        
        # Only set state if there's a problem
        if self.properties["customers_waiting"] > 2:
            # Priority 1: Customers waiting
            self.state = ObjectState.COUNTER_CUSTOMERS_WAITING
            self.add_signal("customers_waiting")
            
        elif self.properties["cleanliness"] < 30:
            # Priority 2: Dirty counter
            self.state = ObjectState.COUNTER_DIRTY
            self.add_signal("needs_cleaning")
            
        else:
            # No problems - no state needed
            self.state = None
                
    def interact(self, action: str, actor: str, current_time: float) -> str:
        result = super().interact(action, actor, current_time)
        
        if action == "clean":
            self.is_dirty = False
            self.properties["is_dirty"] = False
            # Update state based on remaining problems
            if self.customers_waiting == 0:
                self.state = None  # No problems left
            self.remove_signal("needs_cleaning")
            self.cooldown = 5.0  # Can't clean again for 5 seconds
            return f"{actor} cleaned the bar counter until it sparkled"
            
        elif action == "serve_drink":
            if self.customers_waiting > 0:
                self.customers_waiting -= 1
                self.properties["customers_waiting"] = self.customers_waiting
            # Service makes things dirty sometimes
            if random.random() < 0.3:
                self.is_dirty = True
                self.properties["is_dirty"] = True
            # Update state
            if self.customers_waiting == 0 and not self.is_dirty:
                self.state = None
            return f"{actor} served a drink at the bar"
            
        elif action == "lean_on":
            return f"{actor} is leaning on the bar counter"
            
        return result


class LiquorShelf(InteractableObject):
    """Shelf containing bottles of various spirits"""
    
    def __init__(self, position: Tuple[int, int] = (400, 150)):
        super().__init__("liquor_shelf", position, "storage")
        self.state = None  # Normal state - shelf is well stocked
        self.properties = {
            "max_bottles": 20,
            "current_bottles": 18,
            "bottle_types": ["red_wine", "whiskey", "gin", "vodka"],
            "last_restock": 0
        }
        
    def _perform_update(self, current_time: float, delta_time: float):
        """Liquor shelf specific update logic"""
        # Update state based on stock level
        ratio = self.properties["current_bottles"] / self.properties["max_bottles"]
        
        if ratio >= 0.75:
            self.state = None  # Normal - well stocked
            self.remove_signal("low_stock")
            self.remove_signal("needs_restock")
        elif ratio >= 0.5:
            self.state = ObjectState.SHELF_HALF
            self.remove_signal("needs_restock")
            self.add_signal("low_stock")
        elif ratio > 0.25:
            self.state = ObjectState.SHELF_LOW
            self.add_signal("low_stock")
            self.add_signal("needs_restock")
        else:
            self.state = ObjectState.SHELF_EMPTY
            self.add_signal("needs_urgent_restock")
            
    def interact(self, action: str, actor: str, current_time: float) -> str:
        result = super().interact(action, actor, current_time)
        
        if action == "take_bottle":
            if self.properties["current_bottles"] > 0:
                self.properties["current_bottles"] -= 1
                bottle = random.choice(self.properties["bottle_types"])
                return f"{actor} took a bottle of {bottle} from the shelf"
            return f"The shelf is empty!"
            
        elif action == "restock":
            bottles_added = self.properties["max_bottles"] - self.properties["current_bottles"]
            self.properties["current_bottles"] = self.properties["max_bottles"]
            self.properties["last_restock"] = current_time
            self.cooldown = 10.0
            return f"{actor} restocked {bottles_added} bottles on the shelf"
            
        elif action == "organize":
            self.cooldown = 3.0
            return f"{actor} organized the bottles on the shelf"
            
        return result


class DiningTable(InteractableObject):
    """A table where customers can sit and eat"""
    
    def __init__(self, name: str, position: Tuple[int, int], seats: int = 4):
        super().__init__(name, position, "seating")
        self.state = ObjectState.TABLE_EMPTY
        self.properties = {
            "max_seats": seats,
            "occupied_seats": [],
            "has_food": False,
            "has_dishes": False,
            "cleanliness": 100,
            "food_type": None
        }
        
    def _perform_update(self, current_time: float, delta_time: float):
        """Dining table specific update logic - Priority based single state"""
        # Priority order (highest to lowest):
        # 1. TABLE_NEEDS_CLEANING - Most urgent, needs immediate attention
        # 2. TABLE_FOOD_WAITING - Customers waiting, needs service
        # 3. TABLE_HAS_FOOD - Customers eating, normal state
        # 4. TABLE_OCCUPIED - Just sitting, no food yet
        # 5. TABLE_EMPTY - Default state
        
        # Clear all signals first
        self.remove_signal("needs_cleaning")
        self.remove_signal("customers_waiting_for_food")
        
        # Determine single state based on priority
        if self.properties["has_dishes"] and self.properties["cleanliness"] < 50:
            # Priority 1: Dirty table needs cleaning
            self.state = ObjectState.TABLE_NEEDS_CLEANING
            self.add_signal("needs_cleaning")
            
        elif len(self.properties["occupied_seats"]) > 0 and not self.properties["has_food"]:
            # Priority 2: Occupied but no food = waiting for food
            self.state = ObjectState.TABLE_FOOD_WAITING
            self.add_signal("customers_waiting_for_food")
            
        elif len(self.properties["occupied_seats"]) > 0 and self.properties["has_food"]:
            # Priority 3: Occupied with food = eating
            self.state = ObjectState.TABLE_HAS_FOOD
            
        elif len(self.properties["occupied_seats"]) > 0:
            # Priority 4: Just occupied
            self.state = ObjectState.TABLE_OCCUPIED
            
        else:
            # Priority 5: Empty (default)
            self.state = ObjectState.TABLE_EMPTY
            
    def interact(self, action: str, actor: str, current_time: float) -> str:
        result = super().interact(action, actor, current_time)
        
        if action == "sit":
            if len(self.properties["occupied_seats"]) < self.properties["max_seats"]:
                self.properties["occupied_seats"].append(actor)
                return f"{actor} sat down at {self.name}"
            return f"{self.name} is full"
            
        elif action == "leave":
            if actor in self.properties["occupied_seats"]:
                self.properties["occupied_seats"].remove(actor)
                if self.properties["has_food"]:
                    self.properties["has_food"] = False
                    self.properties["has_dishes"] = True
                return f"{actor} left {self.name}"
                
        elif action == "clear_table":
            self.properties["has_dishes"] = False
            self.properties["cleanliness"] = 100
            self.properties["food_type"] = None
            self.cooldown = 2.0
            return f"{actor} cleared and cleaned {self.name}"
            
        elif action == "serve_food":
            if not self.properties["has_dishes"]:
                self.properties["has_food"] = True
                self.properties["food_type"] = random.choice(["steak", "salad", "pasta", "soup"])
                return f"{actor} served {self.properties['food_type']} at {self.name}"
                
        return result


class PoolTable(InteractableObject):
    """Pool table for entertainment"""
    
    def __init__(self, position: Tuple[int, int] = (650, 500)):
        super().__init__("pool_table", position, "entertainment")
        self.state = ObjectState.POOL_EMPTY
        self.properties = {
            "current_players": [],
            "max_players": 2,
            "game_duration": 0,
            "balls_remaining": 16,
            "cues_available": 4
        }
        
    def _perform_update(self, current_time: float, delta_time: float):
        """Pool table specific update logic"""
        # Update game duration if in use
        if self.state == ObjectState.POOL_IN_USE:
            self.properties["game_duration"] += delta_time
            
            # Random chance to end game
            if self.properties["game_duration"] > 60 and random.random() < 0.01 * delta_time:
                self.end_game()
                
        # Update state based on players
        if len(self.properties["current_players"]) >= 2:
            self.state = ObjectState.POOL_IN_USE
            self.remove_signal("waiting_for_player")
        elif len(self.properties["current_players"]) == 1:
            self.state = ObjectState.POOL_WAITING
            self.add_signal("waiting_for_player")
        else:
            self.state = ObjectState.POOL_EMPTY
            self.remove_signal("waiting_for_player")
            
    def interact(self, action: str, actor: str, current_time: float) -> str:
        result = super().interact(action, actor, current_time)
        
        if action == "join_game":
            if len(self.properties["current_players"]) < self.properties["max_players"]:
                self.properties["current_players"].append(actor)
                if len(self.properties["current_players"]) == 2:
                    self.properties["game_duration"] = 0
                    self.properties["balls_remaining"] = 16
                    return f"{actor} joined the pool game. Game started!"
                return f"{actor} is waiting for another player at the pool table"
            return f"Pool table is full"
            
        elif action == "leave_game":
            if actor in self.properties["current_players"]:
                self.properties["current_players"].remove(actor)
                return f"{actor} left the pool game"
                
        elif action == "play_shot":
            if actor in self.properties["current_players"] and self.state == ObjectState.POOL_IN_USE:
                self.properties["balls_remaining"] -= random.randint(0, 1)
                if self.properties["balls_remaining"] <= 1:
                    self.end_game()
                    return f"{actor} won the game!"
                return f"{actor} took a shot. {self.properties['balls_remaining']} balls remaining"
                
        return result
    
    def end_game(self):
        """End the current game"""
        self.properties["current_players"].clear()
        self.properties["game_duration"] = 0
        self.properties["balls_remaining"] = 16
        self.add_signal("game_finished")
        self.cooldown = 5.0




class BarEnvironment:
    """Manages all objects in the bar environment"""
    
    def __init__(self):
        # Load shared configuration
        self.config = self._load_config()
        
        # Layer 1: Core interactive objects (full state management)
        self.core_objects = {
            "bar_counter": BarCounter(),
            "liquor_shelf": LiquorShelf(),
            "left_table": DiningTable("left_table", (150, 500)),
            "center_table": DiningTable("center_table", (400, 680)),
            "right_table": DiningTable("right_table", (650, 680)),  # Added third table
            "pool_table": PoolTable()
        }
        
        # Layer 2: Simple objects (basic state only)
        self.simple_objects = {
            "cat": {"position": (200, 420), "state": "sleeping", "mood": "content"},
            "jukebox": {"position": (100, 200), "playing": False, "song": None}
        }
        
        # Layer 3: Static props (no state needed)
        self.static_props = {
            "deer_head": {"position": (400, 50), "description": "mounted trophy"},
            "carpet": {"position": (400, 400), "description": "ornate Persian rug"},
            "torches": {"positions": [(100, 100), (700, 100)], "always_lit": True},
            "paintings": {"count": 3, "description": "landscape paintings"}
        }
        
        self.last_update = time.time()
        
        # For Godot integration
        self.pending_visual_updates = []  # Updates to send to Godot
        
    def _load_config(self) -> Dict:
        """Load shared configuration file"""
        config_file = Path(__file__).parent.parent / "shared_objects_config.json"
        if config_file.exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def update(self):
        """Update all dynamic objects"""
        current_time = time.time()
        delta_time = current_time - self.last_update
        
        # Update all core objects and track state changes
        for name, obj in self.core_objects.items():
            old_state = obj.state
            old_signals = obj.get_active_signals().copy()
            
            obj.update(current_time, delta_time)
            
            # Check if visual update needed for Godot
            if old_state != obj.state or old_signals != obj.get_active_signals():
                self._queue_visual_update(name, obj)
            
        self.last_update = current_time
        
    def get_perception_for_npc(self, npc_position: Tuple[int, int], perception_radius: int = 150) -> Dict:
        """Build perception data for an NPC at given position
        Only reports objects that need attention or are not in normal state"""
        perception_data = {
            "immediate": [],  # Very close objects that need attention
            "nearby": [],     # Nearby objects with non-normal states
            "ambient": [],    # Environmental description (only if unusual)
            "signals": []     # Active signals that require action
        }
        
        # Check core objects - only report if NOT in normal state or has signals
        for name, obj in self.core_objects.items():
            distance = self._calculate_distance(npc_position, obj.position)
            
            # Skip objects in normal/default states unless they have active signals
            is_normal = self._is_normal_state(obj)
            has_signals = len(obj.get_active_signals()) > 0
            
            if not is_normal or has_signals:
                if distance < 50:
                    perception_data["immediate"].append({
                        "name": name,
                        "type": obj.object_type,
                        "state": obj.state.value if obj.state else "unknown",
                        "distance": distance,
                        "signals": obj.get_active_signals()
                    })
                elif distance < perception_radius:
                    perception_data["nearby"].append({
                        "name": name,
                        "type": obj.object_type,
                        "state": obj.state.value if obj.state else "unknown",
                        "distance": distance
                    })
                
            # Only collect actionable signals
            for signal in obj.get_active_signals():
                perception_data["signals"].append(f"{name}: {signal}")
                
        # Add simple objects if nearby
        for name, props in self.simple_objects.items():
            distance = self._calculate_distance(npc_position, props["position"])
            if distance < perception_radius:
                perception_data["nearby"].append({
                    "name": name,
                    "distance": distance,
                    "properties": {k: v for k, v in props.items() if k != "position"}
                })
                
        # Add ambient description
        perception_data["ambient"] = self._get_ambient_description(npc_position)
        
        return perception_data
        
    def interact_with_object(self, object_name: str, action: str, actor: str) -> str:
        """Process interaction with an object"""
        if object_name in self.core_objects:
            return self.core_objects[object_name].interact(
                action, actor, time.time()
            )
        elif object_name in self.simple_objects:
            # Simple interaction for layer 2 objects
            return f"{actor} interacted with {object_name}"
        else:
            return f"{object_name} is not interactive"
            
    def _is_normal_state(self, obj: InteractableObject) -> bool:
        """Check if an object is in its normal/default state that doesn't need attention"""
        
        # No state (None) means normal for ALL objects
        # This is the preferred way - only set state when something needs attention
        if obj.state is None:
            return True
        
        # These states are also considered normal (legacy support)
        normal_states = [
            ObjectState.BAR_NORMAL,      # Bar in normal operation
            ObjectState.BAR_QUIET,        # Quiet bar is fine
            ObjectState.BAR_EMPTY,        # Empty bar is fine  
            ObjectState.SHELF_FULL,       # Full shelf is good
            ObjectState.TABLE_EMPTY,      # Empty tables are normal
            ObjectState.POOL_EMPTY,       # Empty pool table is normal
        ]
        
        # Check if state is considered normal
        if obj.state in normal_states:
            return True
            
        # Any other state means something needs attention
        return False
    
    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two positions"""
        return math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
        
    def _get_ambient_description(self, position: Tuple[int, int]) -> str:
        """Generate ambient description based on position"""
        descriptions = []
        
        # Check proximity to fireplace
        if "fireplace" in self.core_objects:
            fireplace = self.core_objects["fireplace"]
            distance = self._calculate_distance(position, fireplace.position)
            if distance < fireplace.properties["warmth_radius"]:
                if fireplace.state == ObjectState.FIREPLACE_LIT:
                    descriptions.append("The warmth of the fireplace is comforting")
                elif fireplace.state == ObjectState.FIREPLACE_DYING:
                    descriptions.append("The dying fire casts dancing shadows")
                    
        # Check bar area
        if self._calculate_distance(position, (400, 280)) < 100:
            descriptions.append("The bar area bustles with activity")
            
        # Pool table area
        if self._calculate_distance(position, (650, 500)) < 100:
            pool = self.core_objects.get("pool_table")
            if pool and pool.state == ObjectState.POOL_IN_USE:
                descriptions.append("The sound of pool balls clacking fills the air")
                
        return ". ".join(descriptions) if descriptions else "The bar has a cozy atmosphere"
        
    def get_status_summary(self) -> str:
        """Get a summary of all object states and signals"""
        summary = []
        summary.append("=== Bar Environment Status ===\n")
        
        # Core objects status
        summary.append("Core Objects:")
        for name, obj in self.core_objects.items():
            signals = obj.get_active_signals()
            signal_str = f" [!{', '.join(signals)}]" if signals else ""
            summary.append(f"  - {name}: {obj.state.value if obj.state else 'unknown'}{signal_str}")
            
        # Active signals
        all_signals = []
        for name, obj in self.core_objects.items():
            for signal in obj.get_active_signals():
                all_signals.append(f"{name}: {signal}")
                
        if all_signals:
            summary.append("\nActive Signals:")
            for signal in all_signals:
                summary.append(f"  ⚠ {signal}")
                
        return "\n".join(summary)
    
    def _queue_visual_update(self, object_name: str, obj: InteractableObject):
        """Queue a visual update to send to Godot"""
        visual_state = self._determine_visual_state(obj)
        
        update = {
            "type": "object_state_update",
            "object": object_name,
            "visual_state": visual_state,
            "state": obj.state.value if obj.state else "unknown",
            "signals": obj.get_active_signals(),
            "properties": {
                k: v for k, v in obj.properties.items() 
                if k in ["cleanliness", "current_bottles", "occupied_seats", "current_players"]
            }
        }
        
        self.pending_visual_updates.append(update)
    
    def _determine_visual_state(self, obj: InteractableObject) -> str:
        """Convert object state to visual representation for Godot"""
        if isinstance(obj, BarCounter):
            cleanliness = obj.properties["cleanliness"]
            if obj.properties["wet_level"] > 50:
                return "wet"
            elif cleanliness > 80:
                return "clean"
            elif cleanliness > 50:
                return "slightly_dirty"
            else:
                return "dirty"
                
        elif isinstance(obj, LiquorShelf):
            ratio = obj.properties["current_bottles"] / obj.properties["max_bottles"]
            if ratio > 0.75:
                return "full"
            elif ratio > 0.5:
                return "half_empty"
            elif ratio > 0.25:
                return "low"
            else:
                return "empty"
                
        elif isinstance(obj, DiningTable):
            if len(obj.properties["occupied_seats"]) > 0:
                return f"occupied_{len(obj.properties['occupied_seats'])}"
            elif obj.properties["has_dishes"]:
                return "needs_cleaning"
            elif obj.properties["has_food"]:
                return "food_ready"
            else:
                return "empty"
                
        elif isinstance(obj, PoolTable):
            if len(obj.properties["current_players"]) >= 2:
                return "game_active"
            elif len(obj.properties["current_players"]) == 1:
                return "waiting_player"
            else:
                return "idle"
                
        return "default"
    
    def get_pending_visual_updates(self) -> List[Dict]:
        """Get and clear pending visual updates for Godot"""
        updates = self.pending_visual_updates.copy()
        self.pending_visual_updates.clear()
        return updates
    
    def process_godot_interaction(self, data: Dict) -> Dict:
        """Process interaction request from Godot"""
        object_name = data.get("object")
        action = data.get("action")
        actor = data.get("actor")
        
        if object_name in self.core_objects:
            result = self.interact_with_object(object_name, action, actor)
            
            # Queue visual update if state changed
            if "cleaned" in result or "restocked" in result or "joined" in result:
                self._queue_visual_update(object_name, self.core_objects[object_name])
                
            return {
                "success": True,
                "message": result,
                "updates": self.get_pending_visual_updates()
            }
        
        return {
            "success": False,
            "message": f"Unknown object: {object_name}"
        }
    
    def get_npc_decision(self, npc_name: str, npc_position: Tuple[int, int], 
                        npc_role: str = "customer") -> Dict:
        """Make decision for NPC based on perception and role"""
        perception = self.get_perception_for_npc(npc_position)
        
        decision = {
            "action": None,
            "target": None,
            "priority": 0
        }
        
        # Check signals for urgent tasks
        for signal in perception["signals"]:
            if "needs_cleaning" in signal and npc_role in ["bartender", "cleaner"]:
                object_name = signal.split(":")[0]
                decision["action"] = "clean"
                decision["target"] = object_name
                decision["priority"] = 80
                break
            elif "needs_restock" in signal and npc_role == "bartender":
                object_name = signal.split(":")[0]
                decision["action"] = "restock"
                decision["target"] = object_name
                decision["priority"] = 70
                break
                
        # If no urgent tasks, look for opportunities
        if not decision["action"]:
            for obj in perception["immediate"]:
                if obj["type"] == "seating" and "empty" in obj["state"]:
                    decision["action"] = "sit"
                    decision["target"] = obj["name"]
                    decision["priority"] = 30
                    break
                elif obj["type"] == "entertainment" and "idle" in obj["state"]:
                    decision["action"] = "join_game"
                    decision["target"] = obj["name"]
                    decision["priority"] = 40
                    break
                    
        return decision
    
    def save_state(self, filename: str = None) -> str:
        """Save current environment state to JSON for persistence"""
        state_data = {
            "timestamp": time.time(),
            "core_objects": {},
            "simple_objects": self.simple_objects
        }
        
        # Save core objects state
        for obj_name, obj in self.core_objects.items():
            state_data["core_objects"][obj_name] = {
                "state": obj.state.value if obj.state else None,
                "properties": obj.properties,
                "signals": obj.get_active_signals(),
                "cooldown": obj.cooldown
            }
        
        json_data = json.dumps(state_data, indent=2)
        
        # Save to file if filename provided
        if filename:
            filepath = Path(__file__).parent / "saved_states" / filename
            filepath.parent.mkdir(exist_ok=True)
            with open(filepath, 'w') as f:
                f.write(json_data)
                
        return json_data
    
    def load_state(self, filename: str):
        """Load environment state from JSON file"""
        filepath = Path(__file__).parent / "saved_states" / filename
        if not filepath.exists():
            return False
            
        with open(filepath, 'r') as f:
            state_data = json.load(f)
            
        # Restore core objects state
        for obj_name, saved_state in state_data["core_objects"].items():
            if obj_name in self.core_objects:
                obj = self.core_objects[obj_name]
                obj.properties = saved_state["properties"]
                obj.cooldown = saved_state.get("cooldown", 0)
                # Restore signals
                obj.signals = saved_state.get("signals", [])
                
        # Restore simple objects
        self.simple_objects = state_data.get("simple_objects", {})
        
        return True
    
    def get_perception_description(self, perception: Dict, npc_role: str = "customer") -> str:
        """Convert perception data to natural language for LLM understanding"""
        description = []
        
        # Priority 1: Urgent signals
        urgent_signals = []
        for signal in perception["signals"]:
            if "customers_waiting" in signal:
                obj_name = signal.split(":")[0]
                if "bar" in obj_name:
                    urgent_signals.append("Customers are waiting at the bar!")
                elif "table" in obj_name:
                    urgent_signals.append(f"Customers at {obj_name} are waiting for their food!")
            elif "needs_cleaning" in signal:
                obj_name = signal.split(":")[0]
                urgent_signals.append(f"The {obj_name.replace('_', ' ')} needs cleaning.")
            elif "needs_restock" in signal:
                urgent_signals.append("The liquor shelf is running low on stock.")
        
        if urgent_signals and npc_role in ["bartender", "cleaner"]:
            description.extend(urgent_signals)
        
        # Priority 2: Immediate objects that aren't normal
        for obj in perception.get("immediate", []):
            if obj["state"] not in ["normal", "empty", "clean"]:
                if obj["type"] == "seating" and obj["state"] == "food_waiting":
                    description.append(f"The customers at {obj['name']} look hungry.")
                elif obj["type"] == "entertainment" and obj["state"] == "waiting_for_player":
                    description.append("Someone is waiting for a pool game partner.")
        
        # Priority 3: General ambient
        if perception.get("ambient") and len(description) < 3:
            description.append(perception["ambient"])
        
        # If nothing special, report normalcy
        if not description:
            if npc_role == "bartender":
                description.append("Everything at the bar seems to be in order.")
            else:
                description.append("The bar has a pleasant atmosphere.")
        
        return " ".join(description)


class SimpleStateReceiver:
    """Receive binary states from Godot and convert to prompts"""
    
    def __init__(self):
        # Priority mapping for different states
        self.priority_map = {
            "counter_has_customers": 10,
            "table_has_customers": 8,
            "shelf_empty": 7,
            "counter_dirty": 5,
            "table_dirty": 4,
            "shelf_low": 3,
            "pool_waiting": 2
        }
        
        # State to natural language descriptions
        self.state_descriptions = {
            "counter_dirty": "the bar counter is dirty",
            "counter_has_customers": "customers are waiting at the bar",
            "table_dirty": "a table needs cleaning",
            "table_has_customers": "customers are at a table",
            "shelf_low": "the liquor shelf is running low",
            "shelf_empty": "the liquor shelf is empty",
            "pool_waiting": "someone is waiting for a pool partner"
        }
        
        # State to action mapping
        self.action_map = {
            "counter_has_customers": "serve_customer",
            "table_has_customers": "take_order",
            "shelf_empty": "restock_urgent",
            "counter_dirty": "clean_counter",
            "table_dirty": "clear_table",
            "shelf_low": "restock",
            "pool_waiting": "join_pool"
        }
    
    def receive_godot_state(self, state_dict: Dict[str, bool]) -> Dict[str, Any]:
        """
        Receive simple binary state from Godot
        Input: {"counter_dirty": true, "counter_has_customers": false, ...}
        """
        # Find active states
        active_states = [state for state, is_active in state_dict.items() if is_active]
        
        if not active_states:
            return {
                "perception": "Everything looks normal in the bar",
                "priority_action": "observe",
                "active_states": []
            }
        
        # Sort by priority
        active_states.sort(key=lambda x: self.priority_map.get(x, 0), reverse=True)
        
        # Create perception text
        descriptions = []
        for state in active_states[:3]:  # Top 3 most important
            if state in self.state_descriptions:
                descriptions.append(self.state_descriptions[state])
        
        if len(descriptions) == 1:
            perception = f"You notice that {descriptions[0]}"
        elif len(descriptions) == 2:
            perception = f"You notice that {descriptions[0]} and {descriptions[1]}"
        else:
            perception = f"You notice that {', '.join(descriptions[:-1])}, and {descriptions[-1]}"
        
        # Get priority action
        priority_state = active_states[0]
        priority_action = self.action_map.get(priority_state, "observe")
        
        return {
            "perception": perception,
            "priority_action": priority_action,
            "active_states": active_states
        }
    
    def create_llm_prompt(self, npc_name: str, state_dict: Dict[str, bool]) -> str:
        """Create a prompt for LLM based on state"""
        result = self.receive_godot_state(state_dict)
        
        prompt = f"""You are {npc_name}, a bartender.

{result['perception']}

What single action do you take?
Options: serve_customer, clean_counter, clear_table, restock, take_break, observe

Respond with ONLY the action word."""
        
        return prompt
    
    def to_godot_format(self, bar_env: 'BarEnvironment') -> Dict[str, bool]:
        """Convert BarEnvironment state to simple format for Godot"""
        godot_state = {
            "counter_dirty": False,
            "counter_has_customers": False,
            "table_dirty": False,
            "table_has_customers": False,
            "shelf_low": False,
            "shelf_empty": False,
            "pool_waiting": False
        }
        
        # Check bar counter
        counter = bar_env.core_objects.get("bar_counter")
        if counter:
            if counter.state == ObjectState.COUNTER_DIRTY:
                godot_state["counter_dirty"] = True
            elif counter.state == ObjectState.COUNTER_CUSTOMERS_WAITING:
                godot_state["counter_has_customers"] = True
        
        # Check tables
        for table_name in ["left_table", "center_table", "right_table"]:
            table = bar_env.core_objects.get(table_name)
            if table and table.state:
                if table.state == ObjectState.TABLE_NEEDS_CLEANING:
                    godot_state["table_dirty"] = True
                elif table.state == ObjectState.TABLE_FOOD_WAITING:
                    godot_state["table_has_customers"] = True
        
        # Check shelf
        shelf = bar_env.core_objects.get("liquor_shelf")
        if shelf:
            if shelf.state == ObjectState.SHELF_EMPTY:
                godot_state["shelf_empty"] = True
            elif shelf.state == ObjectState.SHELF_LOW:
                godot_state["shelf_low"] = True
        
        # Check pool table
        pool = bar_env.core_objects.get("pool_table")
        if pool and pool.state == ObjectState.POOL_WAITING:
            godot_state["pool_waiting"] = True
        
        return godot_state


# Test the system
if __name__ == "__main__":
    # Create environment
    env = BarEnvironment()
    
    # Simulate some time passing
    for i in range(5):
        time.sleep(0.1)
        env.update()
        
    # Test NPC perception
    npc_position = (400, 300)  # Near the bar
    perception = env.get_perception_for_npc(npc_position)
    
    print("NPC Perception at bar:")
    print(f"Immediate objects: {perception['immediate']}")
    print(f"Nearby objects: {perception['nearby']}")
    print(f"Ambient: {perception['ambient']}")
    print(f"Signals: {perception['signals']}")
    
    # Test interactions
    print("\n=== Testing Interactions ===")
    print(env.interact_with_object("bar_counter", "clean", "Bob"))
    print(env.interact_with_object("liquor_shelf", "take_bottle", "Alice"))
    print(env.interact_with_object("pool_table", "join_game", "Sam"))
    print(env.interact_with_object("fireplace", "add_wood", "Bob"))
    
    # Get status
    print("\n" + env.get_status_summary())
    
    # Test SimpleStateReceiver
    print("\n=== Testing SimpleStateReceiver ===")
    receiver = SimpleStateReceiver()
    
    # Convert current bar state to Godot format
    godot_state = receiver.to_godot_format(env)
    print("\nGodot State Format:")
    print(json.dumps(godot_state, indent=2))
    
    # Test different scenarios
    test_states = [
        {
            "name": "Normal",
            "state": {
                "counter_dirty": False,
                "counter_has_customers": False,
                "table_dirty": False,
                "table_has_customers": False,
                "shelf_low": False,
                "shelf_empty": False,
                "pool_waiting": False
            }
        },
        {
            "name": "Dirty Counter",
            "state": {
                "counter_dirty": True,
                "counter_has_customers": False,
                "table_dirty": False,
                "table_has_customers": False,
                "shelf_low": False,
                "shelf_empty": False,
                "pool_waiting": False
            }
        },
        {
            "name": "Multiple Problems",
            "state": {
                "counter_dirty": True,
                "counter_has_customers": True,
                "table_dirty": True,
                "table_has_customers": False,
                "shelf_low": True,
                "shelf_empty": False,
                "pool_waiting": False
            }
        }
    ]
    
    for test in test_states:
        print(f"\n--- Scenario: {test['name']} ---")
        result = receiver.receive_godot_state(test['state'])
        print(f"Perception: {result['perception']}")
        print(f"Priority Action: {result['priority_action']}")
        print(f"Active States: {result['active_states']}")
        
        # Show LLM prompt
        prompt = receiver.create_llm_prompt("Sam", test['state'])
        print(f"\nLLM Prompt:\n{prompt}")
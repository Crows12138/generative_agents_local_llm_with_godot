extends Node
# State Detector - Efficient state compression using bit flags
# Detects bar environment state and sends compressed data to Python

signal state_changed(state_flags: int)

# Bit flag definitions - matches Python server
enum StateFlags {
	COUNTER_DIRTY = 1,        # 1 << 0
	COUNTER_CUSTOMERS = 2,    # 1 << 1  
	TABLE_DIRTY = 4,          # 1 << 2
	TABLE_CUSTOMERS = 8,      # 1 << 3
	SHELF_LOW = 16,           # 1 << 4
	SHELF_EMPTY = 32,         # 1 << 5
	POOL_WAITING = 64,        # 1 << 6
	MUSIC_PLAYING = 128,      # 1 << 7
}

# References to scene objects to monitor
@export var bar_counter: Node2D
@export var tables: Array[Node2D] = []
@export var shelf: Node2D
@export var pool_table: Node2D
@export var music_player: Node2D

# Detection parameters
@export var detection_interval: float = 0.5  # Check every 0.5 seconds
@export var customer_detection_range: float = 100.0
@export var dirt_threshold: float = 30.0  # Below this cleanliness = dirty
@export var vision_range: float = 300.0  # How far NPC can see
@export var peripheral_vision_range: float = 500.0  # Reduced awareness at edge
@export var peripheral_detection_chance: float = 0.5  # 50% chance to notice at edge
@export var state_stability_threshold: int = 3  # State must be consistent for N checks before changing

var current_state: int = 0
var previous_state: int = 0
var detection_timer: Timer
var state_consistency_count: int = 0
var pending_state: int = 0

func _ready():
	# Create detection timer
	detection_timer = Timer.new()
	detection_timer.wait_time = detection_interval
	detection_timer.timeout.connect(_detect_state)
	add_child(detection_timer)
	detection_timer.start()
	
	print("[StateDetector] Initialized with interval: ", detection_interval)

func _detect_state():
	"""Main detection loop - runs periodically with stability check"""
	var new_state = get_compressed_state()
	
	# Check state stability to prevent flickering
	if new_state == pending_state:
		state_consistency_count += 1
		
		# Only emit if state is stable and different from current
		if state_consistency_count >= state_stability_threshold and new_state != current_state:
			previous_state = current_state
			current_state = new_state
			state_changed.emit(current_state)
			_log_state_change()
			state_consistency_count = 0
	else:
		# State changed, reset consistency counter
		pending_state = new_state
		state_consistency_count = 1

func get_compressed_state() -> int:
	"""Get current state as compressed bit flags"""
	var state = 0
	var npc_pos = _get_npc_position()
	
	# Check bar counter (if in range)
	if bar_counter and _is_in_vision_range(bar_counter, npc_pos):
		if _is_counter_dirty():
			state |= StateFlags.COUNTER_DIRTY
		if _has_counter_customers():
			state |= StateFlags.COUNTER_CUSTOMERS
	
	# Check tables (only visible ones)
	if _is_any_table_dirty_in_range(npc_pos):
		state |= StateFlags.TABLE_DIRTY
	if _has_table_customers_in_range(npc_pos):
		state |= StateFlags.TABLE_CUSTOMERS
	
	# Check shelf (if in range)
	if shelf and _is_in_vision_range(shelf, npc_pos):
		if _is_shelf_low():
			state |= StateFlags.SHELF_LOW
		if _is_shelf_empty():
			state |= StateFlags.SHELF_EMPTY
	
	# Check pool table (if in range)
	if pool_table and _is_in_vision_range(pool_table, npc_pos) and _has_pool_waiting():
		state |= StateFlags.POOL_WAITING
	
	# Check music (can be heard from farther)
	if music_player and _is_music_playing():
		state |= StateFlags.MUSIC_PLAYING  # Music can be heard everywhere
	
	return state

# Individual state detection functions
func _is_counter_dirty() -> bool:
	if not bar_counter:
		return false
	
	# Check if counter has 'cleanliness' property
	if bar_counter.has_method("get_cleanliness"):
		return bar_counter.get_cleanliness() < dirt_threshold
	
	# Alternative: check for dirt nodes
	var dirt_nodes = bar_counter.get_children().filter(
		func(child): return child.name.contains("dirt") or child.name.contains("spill")
	)
	return dirt_nodes.size() > 0

func _has_counter_customers() -> bool:
	if not bar_counter:
		return false
	
	# Check for customers near counter
	var customers = get_tree().get_nodes_in_group("customers")
	for customer in customers:
		if customer.global_position.distance_to(bar_counter.global_position) < customer_detection_range:
			# Additional check: is customer waiting (not already being served)
			if customer.has_method("is_waiting") and customer.is_waiting():
				return true
	return false

func _is_any_table_dirty() -> bool:
	for table in tables:
		if not table:
			continue
		
		# Check table cleanliness
		if table.has_method("get_cleanliness"):
			if table.get_cleanliness() < dirt_threshold:
				return true
		
		# Check for dirty dishes
		if table.has_method("has_dirty_dishes"):
			if table.has_dirty_dishes():
				return true
	return false

func _is_any_table_dirty_in_range(npc_pos: Vector2) -> bool:
	for table in tables:
		if not table or not _is_in_vision_range(table, npc_pos):
			continue
		
		# Check table cleanliness
		if table.has_method("get_cleanliness"):
			if table.get_cleanliness() < dirt_threshold:
				return true
		
		# Check for dirty dishes
		if table.has_method("has_dirty_dishes"):
			if table.has_dirty_dishes():
				return true
	return false

func _has_table_customers() -> bool:
	for table in tables:
		if not table:
			continue
		
		# Check if table is occupied
		if table.has_method("is_occupied"):
			if table.is_occupied():
				return true
		
		# Alternative: check proximity
		var customers = get_tree().get_nodes_in_group("customers")
		for customer in customers:
			if customer.global_position.distance_to(table.global_position) < 50.0:
				return true
	return false

func _has_table_customers_in_range(npc_pos: Vector2) -> bool:
	for table in tables:
		if not table or not _is_in_vision_range(table, npc_pos):
			continue
		
		# Check if table is occupied
		if table.has_method("is_occupied"):
			if table.is_occupied():
				return true
		
		# Alternative: check proximity
		var customers = get_tree().get_nodes_in_group("customers")
		for customer in customers:
			if customer.global_position.distance_to(table.global_position) < 50.0:
				return true
	return false

func _is_shelf_low() -> bool:
	if not shelf or not shelf.has_method("get_stock_level"):
		return false
	return shelf.get_stock_level() < 30  # Less than 30% stock

func _is_shelf_empty() -> bool:
	if not shelf or not shelf.has_method("get_stock_level"):
		return false
	return shelf.get_stock_level() <= 0

func _has_pool_waiting() -> bool:
	if not pool_table:
		return false
	
	# Check if someone is waiting at pool table
	if pool_table.has_method("get_waiting_count"):
		return pool_table.get_waiting_count() > 0
	
	# Alternative: check proximity
	var players = get_tree().get_nodes_in_group("customers")
	var nearby_count = 0
	for player in players:
		if player.global_position.distance_to(pool_table.global_position) < 80.0:
			nearby_count += 1
	return nearby_count == 1  # Exactly one person waiting

func _is_music_playing() -> bool:
	if not music_player:
		return false
	
	if music_player.has_method("is_playing"):
		return music_player.is_playing()
	
	# Check AudioStreamPlayer
	if music_player is AudioStreamPlayer2D:
		return music_player.playing
	
	return false

func _log_state_change():
	"""Debug logging for state changes"""
	var changed_flags = current_state ^ previous_state
	var changes = []
	
	if changed_flags & StateFlags.COUNTER_DIRTY:
		changes.append("counter_dirty: " + str(bool(current_state & StateFlags.COUNTER_DIRTY)))
	if changed_flags & StateFlags.COUNTER_CUSTOMERS:
		changes.append("counter_customers: " + str(bool(current_state & StateFlags.COUNTER_CUSTOMERS)))
	if changed_flags & StateFlags.TABLE_DIRTY:
		changes.append("table_dirty: " + str(bool(current_state & StateFlags.TABLE_DIRTY)))
	if changed_flags & StateFlags.TABLE_CUSTOMERS:
		changes.append("table_customers: " + str(bool(current_state & StateFlags.TABLE_CUSTOMERS)))
	if changed_flags & StateFlags.SHELF_LOW:
		changes.append("shelf_low: " + str(bool(current_state & StateFlags.SHELF_LOW)))
	if changed_flags & StateFlags.SHELF_EMPTY:
		changes.append("shelf_empty: " + str(bool(current_state & StateFlags.SHELF_EMPTY)))
	
	if changes.size() > 0:
		var npc_name = get_parent().get_parent().name if get_parent() and get_parent().get_parent() else "Unknown"
		print("[StateDetector-", npc_name, "] State changed: ", ", ".join(changes))
		print("  ", _get_vision_description())

# Manual state setters for testing
func set_counter_dirty(dirty: bool):
	if dirty:
		current_state |= StateFlags.COUNTER_DIRTY
	else:
		current_state &= ~StateFlags.COUNTER_DIRTY
	state_changed.emit(current_state)

func set_counter_customers(has_customers: bool):
	if has_customers:
		current_state |= StateFlags.COUNTER_CUSTOMERS
	else:
		current_state &= ~StateFlags.COUNTER_CUSTOMERS
	state_changed.emit(current_state)

# Utility functions
func decode_state(state_int: int) -> Dictionary:
	"""Decode bit flags to dictionary (for debugging)"""
	return {
		"counter_dirty": bool(state_int & StateFlags.COUNTER_DIRTY),
		"counter_customers": bool(state_int & StateFlags.COUNTER_CUSTOMERS),
		"table_dirty": bool(state_int & StateFlags.TABLE_DIRTY),
		"table_customers": bool(state_int & StateFlags.TABLE_CUSTOMERS),
		"shelf_low": bool(state_int & StateFlags.SHELF_LOW),
		"shelf_empty": bool(state_int & StateFlags.SHELF_EMPTY),
		"pool_waiting": bool(state_int & StateFlags.POOL_WAITING),
		"music_playing": bool(state_int & StateFlags.MUSIC_PLAYING),
	}

# Vision range helpers
func _get_npc_position() -> Vector2:
	"""Get the NPC's current position"""
	# Try to get the NPC node (parent of parent)
	var npc = get_parent().get_parent() if get_parent() else null
	if npc and npc is Node2D:
		return npc.global_position
	return Vector2.ZERO

func _is_in_vision_range(target: Node2D, npc_pos: Vector2) -> bool:
	"""Check if target is within NPC's vision range"""
	if not target:
		return false
	
	var distance = npc_pos.distance_to(target.global_position)
	
	# Full vision within normal range
	if distance <= vision_range:
		return true
	
	# Stable peripheral vision - avoid flickering
	if distance <= peripheral_vision_range:
		# Use deterministic detection to prevent state flickering
		var hash_input = str(int(npc_pos.x / 50)) + str(int(npc_pos.y / 50)) + target.name
		var hash_val = hash(hash_input) % 100
		return hash_val < (peripheral_detection_chance * 100)
	
	return false

func _get_vision_description() -> String:
	"""Get a description of what's in vision (for debug)"""
	var npc_pos = _get_npc_position()
	var visible_objects = []
	
	if bar_counter and _is_in_vision_range(bar_counter, npc_pos):
		visible_objects.append("BarCounter")
	
	if shelf and _is_in_vision_range(shelf, npc_pos):
		visible_objects.append("Shelf")
	
	for i in range(tables.size()):
		if tables[i] and _is_in_vision_range(tables[i], npc_pos):
			visible_objects.append("Table" + str(i+1))
	
	if pool_table and _is_in_vision_range(pool_table, npc_pos):
		visible_objects.append("PoolTable")
	
	return "Visible: " + str(visible_objects)

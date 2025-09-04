extends Node
# Action Executor - Manages action queue and smooth execution
# Receives decisions from Python and executes them in Godot

signal action_started(action_name: String)
signal action_completed(action_name: String, result: String)
signal queue_updated(queue_size: int)

# Action buffer with priority queue
var action_buffer: Array = []
var is_executing: bool = false
var current_action: Dictionary = {}

# NPC reference
@export var npc: CharacterBody2D
@export var animation_player: AnimatedSprite2D

# Action timing
@export var default_action_duration: float = 2.0
var action_durations = {
	"serve_customer": 3.0,
	"clean_counter": 2.5,
	"clear_table": 2.0,
	"restock": 4.0,
	"observe": 1.0,
	"take_break": 5.0
}

# Movement parameters
@export var movement_speed: float = 100.0
@export var arrival_threshold: float = 10.0

# Target locations for actions
@export var location_markers: Dictionary = {}  # Set in editor

func _ready():
	print("[ActionExecutor] Ready for NPC: ", npc.name if npc else "Unknown")
	
	# Auto-find animation player if not set
	if not animation_player and npc:
		animation_player = npc.get_node("AnimatedSprite2D")

func queue_action(action: Dictionary):
	"""Add action to priority queue"""
	var priority = action.get("priority", 0)
	var inserted = false
	
	# Insert by priority (higher priority first)
	for i in range(action_buffer.size()):
		if action_buffer[i].get("priority", 0) < priority:
			action_buffer.insert(i, action)
			inserted = true
			break
	
	if not inserted:
		action_buffer.append(action)
	
	queue_updated.emit(action_buffer.size())
	print("[ActionExecutor] Queued action: ", action.action, " (priority: ", priority, ")")
	
	# Start execution if not busy
	if not is_executing:
		_execute_next_action()

func clear_queue():
	"""Clear all pending actions"""
	action_buffer.clear()
	queue_updated.emit(0)

func interrupt_current_action():
	"""Stop current action and move to next"""
	is_executing = false
	_execute_next_action()

func _execute_next_action():
	"""Execute the next action in queue"""
	if action_buffer.is_empty():
		is_executing = false
		return
	
	is_executing = true
	current_action = action_buffer.pop_front()
	queue_updated.emit(action_buffer.size())
	
	action_started.emit(current_action.action)
	print("[ActionExecutor] Executing: ", current_action.action)
	
	# Route to appropriate handler
	match current_action.action:
		"serve_customer":
			await _do_serve_customer()
		"clean_counter":
			await _do_clean_counter()
		"clear_table":
			await _do_clear_table()
		"restock":
			await _do_restock()
		"observe":
			await _do_observe()
		"take_break":
			await _do_take_break()
		_:
			print("[ActionExecutor] Unknown action: ", current_action.action)
			_on_action_complete("unknown")

# Action implementations
func _do_serve_customer():
	"""Serve customers at the bar"""
	# Move to counter
	if await _move_to_location("bar_counter"):
		# Play serving animation
		if animation_player:
			animation_player.play("serve")
		
		# Wait for animation
		await get_tree().create_timer(action_durations.get("serve_customer", 3.0)).timeout
		
		# Update bar state (reduce customer count)
		var bar_counter = get_node_or_null("/root/CozyBar/BarCounter")
		if bar_counter and bar_counter.has_method("customer_served"):
			bar_counter.customer_served()
		
		_on_action_complete("success")
	else:
		_on_action_complete("failed_movement")

func _do_clean_counter():
	"""Clean the bar counter"""
	# Move to counter
	if await _move_to_location("bar_counter"):
		# Play cleaning animation
		if animation_player:
			animation_player.play("clean")
		
		await get_tree().create_timer(action_durations.get("clean_counter", 2.5)).timeout
		
		# Update counter cleanliness
		var counter = get_node_or_null("/root/CozyBar/BarCounter")
		if counter and counter.has_method("set_cleanliness"):
			counter.set_cleanliness(100)
		
		_on_action_complete("success")
	else:
		_on_action_complete("failed_movement")

func _do_clear_table():
	"""Clear and clean a dirty table"""
	# Find nearest dirty table
	var target_table = _find_nearest_dirty_table()
	
	if target_table:
		# Move to table
		if await _move_to_position(target_table.global_position):
			# Play cleaning animation
			if animation_player:
				animation_player.play("clean")
			
			await get_tree().create_timer(action_durations.get("clear_table", 2.0)).timeout
			
			# Clear table
			if target_table.has_method("clear_table"):
				target_table.clear_table()
			
			_on_action_complete("success")
		else:
			_on_action_complete("failed_movement")
	else:
		_on_action_complete("no_target")

func _do_restock():
	"""Restock the shelf"""
	# Move to storage first
	if await _move_to_location("storage"):
		# Play pickup animation
		if animation_player:
			animation_player.play("carry")
		
		await get_tree().create_timer(1.0).timeout
		
		# Move to shelf
		if await _move_to_location("shelf"):
			# Play restock animation
			if animation_player:
				animation_player.play("restock")
			
			await get_tree().create_timer(action_durations.get("restock", 4.0)).timeout
			
			# Update shelf stock
			var shelf = get_node_or_null("/root/CozyBar/LiquorShelf")
			if shelf and shelf.has_method("set_stock_level"):
				shelf.set_stock_level(100)
			
			_on_action_complete("success")
		else:
			_on_action_complete("failed_movement")
	else:
		_on_action_complete("failed_movement")

func _do_observe():
	"""Look around and observe the environment"""
	# Simple observation - just look around
	if animation_player:
		animation_player.play("idle")
	
	# Maybe rotate to look around
	if npc:
		var original_rotation = npc.rotation
		
		# Look left
		var tween = create_tween()
		tween.tween_property(npc, "rotation", original_rotation - PI/4, 0.3)
		await tween.finished
		
		await get_tree().create_timer(0.3).timeout
		
		# Look right
		tween = create_tween()
		tween.tween_property(npc, "rotation", original_rotation + PI/4, 0.3)
		await tween.finished
		
		await get_tree().create_timer(0.3).timeout
		
		# Return to center
		tween = create_tween()
		tween.tween_property(npc, "rotation", original_rotation, 0.3)
		await tween.finished
	
	_on_action_complete("success")

func _do_take_break():
	"""Take a short break"""
	# Move to break area if exists
	if location_markers.has("break_area"):
		await _move_to_location("break_area")
	
	# Play rest animation
	if animation_player:
		animation_player.play("sit")
	
	await get_tree().create_timer(action_durations.get("take_break", 5.0)).timeout
	
	# Stand up
	if animation_player:
		animation_player.play("idle")
	
	_on_action_complete("success")

# Movement helpers
func _move_to_location(location_name: String) -> bool:
	"""Move NPC to a named location"""
	if not location_markers.has(location_name):
		print("[ActionExecutor] Unknown location: ", location_name)
		return false
	
	var target_pos = location_markers[location_name].global_position
	return await _move_to_position(target_pos)

func _move_to_position(target_pos: Vector2) -> bool:
	"""Move NPC to specific position"""
	if not npc:
		return false
	
	# Play walk animation
	if animation_player:
		animation_player.play("walk")
	
	# Simple movement - could be replaced with pathfinding
	while npc.global_position.distance_to(target_pos) > arrival_threshold:
		var direction = (target_pos - npc.global_position).normalized()
		npc.velocity = direction * movement_speed
		
		# Update animation direction
		_update_movement_animation(direction)
		
		npc.move_and_slide()
		await get_tree().process_frame
	
	# Stop and play idle
	npc.velocity = Vector2.ZERO
	if animation_player:
		animation_player.play("idle")
	
	return true

func _update_movement_animation(direction: Vector2):
	"""Update animation based on movement direction"""
	if not animation_player:
		return
	
	# Determine primary direction
	if abs(direction.x) > abs(direction.y):
		# Horizontal movement
		if direction.x > 0:
			animation_player.play("walk_right")
		else:
			animation_player.play("walk_left")
	else:
		# Vertical movement
		if direction.y > 0:
			animation_player.play("walk_down")
		else:
			animation_player.play("walk_up")

func _find_nearest_dirty_table() -> Node2D:
	"""Find the nearest table that needs cleaning"""
	var tables = get_tree().get_nodes_in_group("tables")
	var nearest_table = null
	var nearest_distance = INF
	
	for table in tables:
		if table.has_method("needs_cleaning") and table.needs_cleaning():
			var distance = npc.global_position.distance_to(table.global_position)
			if distance < nearest_distance:
				nearest_distance = distance
				nearest_table = table
	
	return nearest_table

func _on_action_complete(result: String):
	"""Called when action finishes"""
	action_completed.emit(current_action.action, result)
	print("[ActionExecutor] Completed: ", current_action.action, " - ", result)
	
	# Clear current action
	current_action = {}
	is_executing = false
	
	# Execute next action if available
	if not action_buffer.is_empty():
		# Small delay between actions
		await get_tree().create_timer(0.2).timeout
		_execute_next_action()

# Debug functions
func get_queue_info() -> Array:
	"""Get information about queued actions"""
	var info = []
	for action in action_buffer:
		info.append({
			"action": action.action,
			"priority": action.get("priority", 0)
		})
	return info

func is_busy() -> bool:
	"""Check if currently executing an action"""
	return is_executing
